import copy
import logging
import random

import numpy as np
import torch
from matminer.featurizers.composition import ElementProperty
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import Dataset
from tqdm import tqdm

from cacgan.data.schema import AtmoStructureGroup, FormulaEncoder, pd
from cacgan.utils import Composition, chunks, convert_mno
from cacgan.utils import composition_elements, diff2formula, SEED


def feat_compositions(feater: ElementProperty, comps: [Composition]) -> np.ndarray:
    """use matminer ElementProperty to generate an array of features for a list of compositions"""
    feat_array = np.zeros((len(comps), len(feater.feature_labels())))
    for i, c in tqdm(enumerate(comps)):
        feat_array[i] = feater.featurize(c.fractional_composition)
    return feat_array


def variance_threshold_selector(data: pd.DataFrame, threshold=1e-5) -> pd.DataFrame:
    """remove columns with a variance less than threshold form a pd.DataFrame"""
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def rm_nan_columns(a: np.ndarray, m="any") -> np.ndarray:
    """remove nan values from an np.ndarray"""
    if m == "any":
        return a[:, ~np.isnan(a).any(axis=0)]
    elif m == "all":
        return a[:, ~np.isnan(a).all(axis=0)]
    else:
        raise ValueError("m must be any or all!")


class GroupAB:
    def __init__(self, group_a: AtmoStructureGroup, group_b: AtmoStructureGroup, ):
        """
        select two AtmoStructureGroup objs to build up dataset
        """

        self.group_a = group_a
        self.group_b = group_b

        # elements appear in at least one group
        self.possible_elements = sorted(set(self.group_a.possible_elements + self.group_b.possible_elements))

        self.total_chem_pairs = []
        self.total_alchemy_pairs = []

        # i2js[i] would return a *set* of all possible j paired to i
        self.i2js = dict()
        self.j2is = dict()

        for i, a in enumerate(self.group_a):
            ea = composition_elements(a.composition)
            for j, b in enumerate(self.group_b):
                eb = composition_elements(b.composition)
                if ea == eb:
                    self.total_chem_pairs.append((i, j))
                    if i not in self.i2js:
                        self.i2js[i] = {j, }
                    else:
                        self.i2js[i].add(j)
                    if j not in self.j2is:
                        self.j2is[j] = {i, }
                    else:
                        self.j2is[j].add(i)
                else:
                    self.total_alchemy_pairs.append((i, j))

        self.a_haspair = sorted(set([p[0] for p in self.total_chem_pairs]))
        self.b_haspair = sorted(set([p[1] for p in self.total_chem_pairs]))

        self.a_single = [i for i in range(len(self.group_a)) if i not in self.a_haspair]
        self.b_single = [i for i in range(len(self.group_b)) if i not in self.b_haspair]

        # # you don't need this
        # random.shuffle(self.total_chem_pairs)
        # random.shuffle(self.total_alchemy_pairs)

        self.encoder = FormulaEncoder(self.possible_elements)

        self.group_a.encoded = self.encoder.encode_2d(
            [a.composition.fractional_composition.formula for a in self.group_a])
        self.group_b.encoded = self.encoder.encode_2d(
            [a.composition.fractional_composition.formula for a in self.group_b])

        self.Avalues = self.group_a.encoded.copy()
        self.Bvalues = self.group_b.encoded.copy()


class FormulaDataset(Dataset):

    def __init__(self, gab: GroupAB, mo: bool = False, mno: bool = False):
        """
        dataset passed to pytorch dataloader

        :param gab: the GroupAB object after selecting two groups of ATMO
        :param mo: should we use MO composition only?
        :param mno: only keep composition of 1M 1NM
        """
        self.gab = gab
        if mo:
            ga = self.gab.group_a.to_mocomp()
            gb = self.gab.group_b.to_mocomp()
            self.gab = GroupAB(ga, gb)
        if mno:
            ga = self.gab.group_a.keep_only_mno()
            gb = self.gab.group_b.keep_only_mno()
            self.gab = GroupAB(ga, gb)

        self.convertmno = None  # every time an AB pair is drawn, B is converted to the chem sys of A, i.e. "fake" pairing

        self.setup_params = None
        self.mode = None

    def convert_mno(self):
        """set dataset to include only mno structures"""
        self.convertmno = True

    def train(self):
        """set dataset to training mode"""
        assert self.setup_params is not None
        self.mode = "train"

    def test(self):
        """set dataset to training mode"""
        assert self.setup_params is not None
        self.mode = "test"

    def as_train(self):
        """return a copy of the data set and set it to training mode"""
        d = copy.deepcopy(self)
        d.train()
        return d

    def as_test(self):
        """return a copy of the data set and set it to test mode"""
        d = copy.deepcopy(self)
        d.test()
        return d

    def setup(
            self,
            length_multiplier=1.0,
            a_haspair_train=1.0,
            b_haspair_train=0.66,
            a_single_train=1.0,
            b_single_train=1.0,
            setup_seed=SEED,

            b_cv_fold=3,
            b_cv_index=0,
            b_cv=True,
            b_cv_rev=False
    ):
        """
        split train/valid, b_haspair_train should < 1, otherwise there would be no data in valid

        when a_haspair_train < 1, we split both A and B, that is,
            For A
            in train -- a_haspair_train + a_single_train
            in valid -- a_haspair_valid = (1 - a_haspair_train)
            For B
            in train -- b_haspair_train + b_single_train
            in valid -- b_haspair_valid = (1 - b_haspair_train)

            the question becomes if we can generate unseen B from unseen A, which sounds a bit weird...
            for validation we can only use pairs from a_haspair_valid and b_haspair_valid, which may be very few,

        when a_haspair_train = 1, the same set of A samples are used for train and valid, that is,
            For A
            in train -- a_haspair + a_single_train
            in valid -- a_haspair
            For B (this stays the same)
            in train -- b_haspair_train + b_single_train
            in valid -- b_haspair_valid = (1 - b_haspair_train)

            the question becomes if we can generate unseen B from already seen A
            this can guarantee that we are not predicting an unseen chemical system (even we saw it thru A)
        """
        assert 0 <= a_haspair_train <= 1
        assert 0 <= b_haspair_train < 1
        assert 0 <= a_single_train <= 1
        assert 0 <= b_single_train <= 1

        self.length_multiplier = length_multiplier

        random.seed(setup_seed)  # remember shuffle is in-place
        self.train_a_single = random.sample(self.gab.a_single, int(len(self.gab.a_single) * a_single_train))
        self.train_a_haspair = random.sample(self.gab.a_haspair, int(len(self.gab.a_haspair) * a_haspair_train))
        self.train_a = self.train_a_single + self.train_a_haspair

        if a_haspair_train < 1:
            self.test_a = [i for i in self.gab.a_haspair if i not in self.train_a_haspair]
            self.test_a = random.sample(self.test_a, k=len(self.test_a))
        else:
            assert a_haspair_train == 1
            self.test_a = random.sample(self.train_a_haspair, k=len(self.train_a_haspair))

        self.train_b_single = random.sample(self.gab.b_single, int(len(self.gab.b_single) * b_single_train))

        if b_cv:
            assert b_cv_index in range(b_cv_fold)
            b_has_pair = [j for j in self.gab.b_haspair]
            # print("b_has_pair", self.gab.b_haspair)
            random.Random(setup_seed).shuffle(b_has_pair)
            # print("b_has_pair shuffled", b_has_pair)
            b_haspair_chunks = list(chunks(b_has_pair, len(b_has_pair) // b_cv_fold))
            if len(b_has_pair) % b_cv_fold > 0:
                b_haspair_chunks[-2] = b_haspair_chunks[-2] + b_haspair_chunks[-1]
                b_haspair_chunks.pop(-1)
            if b_cv_rev:
                self.train_b_haspair = b_haspair_chunks[b_cv_index]
            else:
                self.train_b_haspair = []
                for i in range(b_cv_fold):
                    if i != b_cv_index:
                        self.train_b_haspair += b_haspair_chunks[i]
        else:
            random.seed(b_cv_index)
            self.train_b_haspair = random.sample(self.gab.b_haspair, int(len(self.gab.b_haspair) * b_haspair_train))
            random.seed(setup_seed)

        self.train_b = self.train_b_single + self.train_b_haspair

        self.test_b = [i for i in self.gab.b_haspair if i not in self.train_b_haspair]

        self.pairs_in_test = []
        for j in self.test_b:
            i_in_test_pool = set(self.test_a).intersection(self.gab.j2is[j])
            for i in i_in_test_pool:
                self.pairs_in_test.append((i, j))

        self.test_apool = [p[0] for p in self.pairs_in_test]
        self.test_bpool = [p[1] for p in self.pairs_in_test]
        self.train_apool = self.train_a
        self.train_bpool = self.train_b
        random.shuffle(self.train_apool)  # these shouldn't matter...
        random.shuffle(self.train_bpool)

        self.setup_params = dict(
            length_multiplier=length_multiplier,
            a_haspair_train=a_haspair_train,
            b_haspair_train=b_haspair_train,
            a_single_train=a_single_train,
            b_single_train=b_single_train,
            seed=SEED,
            b_cv_fold=b_cv_fold,
            b_cv_index=b_cv_index,
            b_cv=b_cv,
        )
        self.setup_params["details"] = self.details

        test_bpool_refcodes = [self.gab.group_b[j].identifier for j in self.test_bpool]
        self.setup_params["test_bpool_refcodes"] = test_bpool_refcodes

    @property
    def details(self):
        s = "=== meta ===\n"
        s += "A group size: {}\n".format(len(self.gab.group_a))
        s += "B group size: {}\n".format(len(self.gab.group_b))
        s += "A group amine: {}\n".format(self.gab.group_a.first_amine)
        s += "B group amine: {}\n".format(self.gab.group_b.first_amine)
        s += "total chem pairs: {}\n".format(len(self.gab.total_chem_pairs))
        s += "total alchemy pairs: {}\n".format(len(self.gab.total_alchemy_pairs))
        s += "=== dataset ===\n"
        s += "train a single: {}\n".format(len(self.train_a_single))
        s += "train b single: {}\n".format(len(self.train_b_single))
        s += "train a haspair: {}\n".format(len(self.train_a_haspair))
        s += "train b haspair: {}\n".format(len(self.train_b_haspair))
        s += "train a pool: {}\n".format(len(self.train_apool))
        s += "train b pool: {}\n".format(len(self.train_bpool))
        s += "test a: {}\n".format(len(self.test_a))
        s += "test b: {}\n".format(len(self.test_b))
        s += "test pairs: {}\n".format(len(self.pairs_in_test))
        return s

    def __getitem__(self, index):
        assert self.mode in ("train", "test",)
        if self.mode == "train":
            i = self.train_apool[index % len(self.train_apool)]
            j = random.choice(self.train_bpool)
        else:
            i = self.test_apool[index % len(self.test_apool)]
            j = self.test_bpool[index % len(self.test_bpool)]

        Avalue = self.gab.Avalues[i]
        if self.convertmno and self.mode == "train":
            Bvalue = convert_mno(Avalue, self.gab.Bvalues[j], self.gab.possible_elements)
        else:
            Bvalue = self.gab.Bvalues[j]
        Acomp = self.gab.group_a[i].composition.formula
        Bcomp = self.gab.group_b[j].composition.formula

        item_A = torch.from_numpy(Avalue)  # .to(DEVICE)
        item_B = torch.from_numpy(Bvalue)  # .to(DEVICE)

        d = {
            'A': item_A,
            'B': item_B,
            'Acomp': Acomp,
            'Bcomp': Bcomp,
        }
        return d

    def __len__(self):
        assert self.mode in ("train", "test",)
        if self.mode == "train":
            return int(len(self.train_apool) * self.length_multiplier)
        else:
            return len(self.test_apool)

    def get_baseline(self):
        """only implemented this for a_haspair_train = 1"""
        assert len(self.train_a_haspair) == len(self.gab.a_haspair)
        baselines = []
        # for each unique j in test, what is the closest a we have seen?
        for j in self.test_b:
            bformula = self.gab.group_b[j].composition.formula
            aformulae = [self.gab.group_a[i].composition.formula for i in self.gab.j2is[j]]
            deltas = [diff2formula(aformula, bformula) for aformula in aformulae]
            baselines.append(min(deltas))
        return baselines


class DimDataset:

    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, hx=None, hy=None):
        """
        dataset used for dimensionality prediction
        """
        self.x = x
        self.y = y
        self.hx = hx
        self.hy = hy

    def holdout(self, exclude_refcode: [str]):
        """
        :param exclude_refcode: to exclude a list of structures and put them in holdout data/target
        """
        assert len(exclude_refcode) > 0
        refcodes = self.x.index.tolist()
        hx = self.x.loc[exclude_refcode]
        hy = self.y.loc[exclude_refcode]
        remaining = list(set(refcodes).difference(set(exclude_refcode)))
        x = self.x.loc[remaining]
        y = self.y.loc[remaining]
        self.x = x
        self.y = y
        self.hx = hx
        self.hy = hy

    @classmethod
    def from_atmogroup(cls, atmo_group: AtmoStructureGroup, feat=False, mocomp=False):
        """create dimensionality dataset from an AtmoStructureGroup"""
        if mocomp:
            structures = atmo_group.to_mocomp().structures
        else:
            structures = atmo_group.structures

        fc = FormulaEncoder(AtmoStructureGroup(structures).possible_elements)

        comp = fc.encode_2d([a.composition.fractional_composition.formula for a in structures])
        data_df = pd.DataFrame(comp, columns=fc.possible_elements)
        dim_df = pd.DataFrame([a.dim for a in structures], columns=["dimension"])
        refcodes = [a.identifier for a in structures]
        data_df.index = refcodes
        dim_df.index = refcodes

        if feat:
            feat_arrays = []
            columns = []
            for name in ["magpie", "matscholar_el", "matminer"]:
                epfeat = ElementProperty.from_preset(name)
                farray = feat_compositions(epfeat, [s.composition for s in structures])
                feat_arrays.append(farray)
                columns += [n for n in epfeat.feature_labels()]
            feat_array = np.hstack(feat_arrays)
            logging.warning("generated feature array shape: {}".format(feat_array.shape))
            feat_df = pd.DataFrame(feat_array, columns=columns)
            feat_df = feat_df.dropna(axis=1, how="any")
            logging.warning("after removing nan columns: {}".format(feat_array.shape))
            feat_df = variance_threshold_selector(feat_df, 1e-5)
            logging.warning("after removing low variance columns: {}".format(feat_array.shape))
            data_df = pd.concat([data_df, feat_df], axis=1)

        return cls(data_df, dim_df)
