import itertools
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from utils import csdformula_to_compositions, is_composition_type, Composition, is_mno


class AtmoStructure:

    def __init__(self, identifier: str, formula: str, amine: str, dim: int = None):
        self.identifier = identifier
        self.formula = formula
        self.amine = amine
        self.dim = dim

        unit_compositions, unit_charges, multipliers = csdformula_to_compositions(self.formula)
        self.mo_composition = Composition()
        self.composition = Composition()
        for i, c in enumerate(unit_compositions):
            if is_composition_type(c, "mo"):
                self.mo_composition += c * multipliers[i]
            self.composition += c * multipliers[i]

        self.encoded = None

    def __repr__(self):
        return "\t".join([self.identifier, self.amine, self.formula, self.composition.formula])


class AtmoStructureGroup:

    def __init__(self, structures: [AtmoStructure]):
        self.structures = structures
        self.encoded = None

    @property
    def possible_elements(self):
        es = []
        for s in self.structures:
            elements = [e.name for e in s.composition.elements]
            es += elements
        return sorted(set(es))

    @property
    def possible_mo_elements(self):
        es = []
        for s in self.structures:
            elements = [e.name for e in s.mo_composition.elements]
            es += elements
        return sorted(set(es))

    def __iter__(self):
        return self.structures.__iter__()

    def __getitem__(self, item):
        return self.structures[item]

    def __len__(self):
        return len(self.structures)

    def to_mocomp(self):
        structures = deepcopy(self.structures)
        for i in range(len(structures)):
            structures[i].composition = structures[i].mo_composition
        return AtmoStructureGroup(structures)

    def keep_only_mno(self):
        structures = deepcopy(self.structures)
        keep = []
        for i in range(len(structures)):
            c = structures[i].composition.fractional_composition
            if is_mno(c):
                keep.append(structures[i])
        return AtmoStructureGroup(keep)

    @property
    def first_amine(self):
        return self.structures[0].amine

    def group_by(self, field="amine"):
        structures = deepcopy(self.structures)
        structures = sorted(structures, key=lambda x: getattr(x, field))
        groups = []
        for k, g in itertools.groupby(structures, lambda x: getattr(x, field)):
            groups.append(AtmoStructureGroup(list(g)))  # Store group iterator as a list
        return groups

    @classmethod
    def from_csv(cls):
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "atmo.csv"))
        print(len(df))
        atmos = []
        for index, row in df.iterrows():
            try:
                atmo = AtmoStructure(row["identifier"], row["csdformula"], row["smiles"], row["dimension"])
            except ValueError:
                print(row["csdformula"])
                continue
            atmos.append(atmo)
        return cls(atmos)


class FormulaEncoder:

    def __init__(self, possible_elements: [str]):
        self.possible_elements = possible_elements

    def encode_1d(self, formula: str, element_composition=False):
        composition = Composition(formula)
        encoded = np.zeros(len(self.possible_elements))
        for i, e in enumerate(self.possible_elements):
            if element_composition:
                if e in composition:
                    encoded[i] = 1
            else:
                encoded[i] = composition[e]
        return encoded

    def encode_2d(self, fs: [str]):
        encoded = np.zeros((len(fs), len(self.possible_elements)))
        for i, f in enumerate(fs):
            encoded[i] = self.encode_1d(f)
        return encoded

    def decode_1d(self, encoded: np.ndarray, eps=1e-10):
        c = Composition()
        assert encoded.ndim == 1
        for i, s in enumerate(encoded):
            if s < eps:
                continue
            e = self.possible_elements[i]
            c += Composition(e) * s
        return c.formula

    def decode_2d(self, encoded: np.ndarray, eps=1e-10):
        assert encoded.ndim == 2
        fs = []
        for i in range(encoded.shape[0]):
            encoded_1d = encoded[i]
            fs.append(self.decode_1d(encoded_1d, eps))
        return fs
