import glob
import inspect
import logging
import os
from copy import deepcopy

from cacgan.data.dataset import FormulaDataset
from cacgan.gans.augcyc import AugCycleGan
from cacgan.gans.cyc import CycleGan
from cacgan.gans.evaluation import eval_for_dist, eval_for_mindist, baseline_random, baseline_identity, eval_for_ratio
from cacgan.utils import *

_cyc_params = dict(
    # dataset params
    length_multiplier=1.0,
    a_haspair_train=1.0,
    b_haspair_train=0.66,
    a_single_train=1.0,
    b_single_train=1.0,
    setup_seed=SEED,
    b_cv_fold=3,
    b_cv_index=0,
    b_cv=True,

    # should not need to change these
    input_nc=65,
    output_nc=65,
    g_blocks=9,
    possible_elements=None,
    work_dir="/",

    # lr related, should tune
    lr=0.0002,
    lr_divider=10,
    lr_slowdown_freq=50,
    lr_slowdown_start=50,
    lr_slowdown_param=1.0,

    # cycle loss composition, should tune
    lambda_A=1.0,
    lambda_B=1.0,

    # cycle loss weight in G loss, important, should tune
    cyc_weight=0.1,

    # if use emd, only support "mp" for now
    emd=None,

    # training
    nepochs=5000
)

_aug_params = dict(
    # dataset params
    length_multiplier=1.0,
    a_haspair_train=1.0,
    b_haspair_train=0.66,
    a_single_train=1.0,
    b_single_train=1.0,
    setup_seed=SEED,
    b_cv_fold=3,
    b_cv_index=0,
    b_cv=True,

    # should not need to change these
    input_nc=65,
    output_nc=65,
    possible_elements=None,
    g_blocks=9,
    nlatent=1,
    use_bias=True,
    work_dir="/",

    # lr related, should tune
    lr=0.0002,
    lr_divider=10,
    lr_slowdown_freq=50,
    lr_slowdown_start=50,
    lr_slowdown_param=1.0,

    # cycle loss composition, should tune
    lambda_A=1.0,
    lambda_B=1.0,
    lambda_z_B=1.0,
    lambda_z_A=1.0,

    # cycle loss weight in G loss, important, should tune
    cyc_weight=0.1,

    # if use emd, only support "mp" for now
    emd=None,

    # training
    nepochs=5000
)


class Trainer:
    def __init__(self, params: dict, dataset: FormulaDataset, model_name="aug", wdir=None, ):
        """
        wrapper for training

        :param params: a dict of params used to define dataset and model
        :param model_name: aug or cyc
        :param wdir: working directory
        """
        if model_name == "aug":
            self.default_params = deepcopy(_aug_params)
        else:
            self.default_params = deepcopy(_cyc_params)
        self.user_defined_params = params

        if wdir is None:
            self.wdir = timestamp()
        else:
            self.wdir = wdir

        self.wdir = os.path.abspath(self.wdir)
        self.chkdir = os.path.join(self.wdir, "chk")
        self.evadir = os.path.join(self.wdir, "eva")
        self.model_name = model_name

        self.dataset = dataset
        self.setup_dataset()
        self.dataset.train()

        self.model = None
        self.build_model()
        self.model_loaded_epoch = None

    @property
    def params(self):
        params = dict()
        params.update(self.default_params)
        params.update(self.user_defined_params)
        return params

    def save(self):
        create_dir(self.wdir)
        d = {
            "default_params": self.default_params,
            "user_defined_params": self.user_defined_params,
            "wdir": self.wdir,
            "params": self.params,
            "model_name": self.model_name,

        }
        save_yml(d, os.path.join(self.wdir, self.__class__.__name__ + ".yml"))

    @classmethod
    def load(cls, dataset: FormulaDataset, ymlpath: typing.Union[str, pathlib.Path], change_wdir=False):
        d = load_yml(ymlpath)
        mparams = d["user_defined_params"]
        dparams = d["default_params"]
        if change_wdir:
            t = cls(mparams, dataset, d["model_name"], os.path.abspath(change_wdir))
        else:
            t = cls(mparams, dataset, d["model_name"], d["wdir"])
        t.default_params = dparams
        return t

    @staticmethod
    def extract_kwargs(params: dict, function, exclude):
        """from a param set, extract available keyword arguments based on a function"""
        kws = inspect.getfullargspec(function).args
        kws = [kw for kw in kws if kw not in exclude]
        use_kws = {}
        for k in kws:
            try:
                use_kws[k] = params[k]
            except KeyError:
                continue
        return use_kws

    def build_model(self):
        """construct a model"""
        possible_elements = self.dataset.gab.possible_elements
        if self.model_name == "cyc":
            model_constructor = CycleGan
            model_kwargs = dict(
                input_nc=len(possible_elements),
                output_nc=len(possible_elements),
                work_dir=self.wdir,
                possible_elements=possible_elements,
            )
        else:
            model_constructor = AugCycleGan
            model_kwargs = dict(
                input_nc=len(possible_elements),
                output_nc=len(possible_elements),
                work_dir=self.wdir,
                possible_elements=possible_elements,
            )
        exclude = list(model_kwargs.keys())
        exclude.append("self")
        kwargs = self.extract_kwargs(self.params, model_constructor.__init__, exclude)
        model_kwargs.update(kwargs)
        self.model = model_constructor(**model_kwargs)

    def setup_dataset(self):
        dskwarg = self.extract_kwargs(self.params, FormulaDataset.setup, ["self", ])
        self.dataset.setup(**dskwarg)

    def load_model(self, epoch: int = None):
        if epoch is None:
            self.model.load(self.last_saved_epoch)
            self.model_loaded_epoch = self.last_saved_epoch
        else:
            self.model.load(epoch)
            self.model_loaded_epoch = epoch

    def eval_model(self, eval_mode="prior", eval_quantity="mindist", ntrials=50, std=5, unique=True, zlim=10,
                   steps=2000,
                   plot=True):
        """
        evaluate model performance

        :param eval_mode: "prior" samples z from a predefined prior distribution, "opt" samples z from a linear space
        :param eval_quantity: "mindist", "ratio" or "dist", for more details see `cacgan.gans.evaluation`
        """
        create_dir(self.evadir)
        self.dataset.test()

        assert self.model is not None
        assert self.model_loaded_epoch is not None

        self.model.eval()

        if self.model_name == "aug":
            assert eval_mode in ["prior", "opt"]
            if eval_mode == "prior":
                real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded = self.model.predict_from_prior(
                    self.dataset, n_trials=ntrials, std=std, unique=unique, back_to_train=False)
            else:
                real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded = self.model.predict_from_opt(
                    self.dataset, zlim=zlim, steps=steps, unique=unique, back_to_train=False)
        else:
            real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded = self.model.predict_cyc(self.dataset,
                                                                                                    back_to_train=False,
                                                                                                    return_numpy=True,
                                                                                                    unique=unique)

        if eval_quantity == "mindist":
            b2b = eval_for_mindist(real_B_encoded, fake_B_encoded)
            bl_rand = baseline_random(real_B_encoded)
            bl_iden = baseline_identity(real_A_encoded, real_B_encoded)
            distributions = [b2b, bl_iden, bl_rand]
            names = [eval_mode, "BL identity", "BL random", ]
        elif eval_quantity == "ratio":
            real_ratio, rand_ratio, fake_ratio = eval_for_ratio(real_B_encoded, fake_B_encoded,
                                                                self.dataset.gab.possible_elements, "C", "N")
            distributions = [real_ratio, rand_ratio, fake_ratio]
            names = ["real", "rand", "fake"]
        else:
            b2b = eval_for_dist(real_B_encoded, fake_B_encoded)
            bl_rand = baseline_random(real_B_encoded)
            bl_iden = baseline_identity(real_A_encoded, real_B_encoded)
            distributions = [b2b, bl_iden, bl_rand]
            names = [eval_mode, "BL identity", "BL random", ]

        save_pkl(distributions,
                 os.path.join(self.evadir, "eval-{}-{:06}.pkl".format(eval_mode, self.model_loaded_epoch)))

        if plot:
            plot_violin(distributions, names, colors=["b", "gray", "k"],
                        saveas=os.path.join(self.evadir, "eval-{}-{}.png".format(eval_mode, self.model_loaded_epoch)))

        return distributions, names

    def train(self, load_pre_trained=False, save_freq=500, eval_after_train=True):

        assert self.model is not None
        self.dataset.train()
        save_yml(self.dataset.setup_params, os.path.join(self.wdir, "train_dataset.yml"))

        create_dir(self.wdir)
        create_dir(self.chkdir)
        self.save()

        train_kwarg = self.extract_kwargs(self.params, self.model.train, ["self", "train_set", "batchsize", ])
        train_kwarg["load_pre_trained"] = load_pre_trained
        train_kwarg["save_freq"] = save_freq
        if load_pre_trained:
            train_kwarg["starting_epoch"] = self.last_saved_epoch
            train_kwarg["load_epoch"] = self.last_saved_epoch
        self.model.train(
            self.dataset,
            batchsize=len(self.dataset.train_apool),
            **train_kwarg
        )
        self.model.save()
        self.save()
        self.model_loaded_epoch = self.model.current_epoch

        if eval_after_train:
            if self.model_name == "aug":
                distributions, names = self.eval_model("prior")
                # self.eval_model("opt")
            else:
                distributions, names = self.eval_model("cyc")
            logging.warning("baseline: {}".format(np.mean(distributions[1])))
            logging.warning("prior: {}".format(np.mean(distributions[0])))
            return np.mean(distributions[0])

    def plot_loss(self, title=None, ylim=None):
        plot_loss(os.path.join(self.chkdir, "loss.csv"), os.path.join(self.wdir, "loss.png"), samewdir=False,
                  title=title, ylim=ylim)
        plt.clf()

    @property
    def saved_epoches(self):
        subdirs = glob.glob("{}/chk/*".format(self.wdir))
        x = []
        for d in subdirs:
            bd = os.path.basename(d)
            if not os.path.isfile(os.path.join(d, "done.dummy")):
                continue
            try:
                epoch = int(bd)
            except ValueError:
                continue
            x.append(epoch)
        x = sorted(x)
        return x

    @property
    def last_saved_epoch(self):
        return self.saved_epoches[-1]
