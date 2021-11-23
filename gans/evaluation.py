import numpy as np

from settings import SEED
from utils import mindist_ra_fas


def baseline_identity(
        real_A_encoded,
        real_B_encoded,
):
    bl_B2B = []
    for i in range(real_B_encoded.shape[0]):
        rb_e = real_B_encoded[i]
        mdist = mindist_ra_fas(rb_e, real_A_encoded)
        bl_B2B.append(mdist)
    return bl_B2B


def gen_rand_composition_vector(real_B_encoded):
    rand_es = []
    for i in range(real_B_encoded.shape[0]):
        rb_e = real_B_encoded[i]
        rand_e = np.random.rand(*rb_e.shape)
        for i in range(len(rand_e)):
            if i not in np.nonzero(rb_e)[0]:
                rand_e[i] = 0
        rand_e = rand_e / np.sum(rand_e)
        rand_es.append(rand_e)
    return np.array(rand_es)


def baseline_random(
        real_B_encoded,
):
    bl_B2B = []
    np.random.seed(SEED)
    for i in range(real_B_encoded.shape[0]):
        rb_e = real_B_encoded[i]
        rand_e = np.random.rand(*rb_e.shape)
        for i in range(len(rand_e)):
            if i not in np.nonzero(rb_e)[0]:
                rand_e[i] = 0
        rand_e = rand_e / np.sum(rand_e)
        dist = np.sum(np.abs(rb_e - rand_e)) / np.count_nonzero(rb_e)
        bl_B2B.append(dist)
    return bl_B2B


def eval_for_mindist(
        real_B_encoded,
        fake_B_encoded,
):
    min_B2B = []
    fake_B_encoded_2d = fake_B_encoded.reshape(fake_B_encoded.shape[0] * fake_B_encoded.shape[1],
                                               fake_B_encoded.shape[2])

    for i in range(real_B_encoded.shape[0]):
        rb_e = real_B_encoded[i]
        mdist = mindist_ra_fas(rb_e, fake_B_encoded_2d)
        min_B2B.append(mdist)
    return min_B2B


def eval_for_dist(
        real_B_encoded,
        fake_B_encoded,
):
    min_B2B = []

    assert fake_B_encoded.shape[0] == real_B_encoded.shape[0]

    for i in range(real_B_encoded.shape[0]):
        fb_es = fake_B_encoded[:, i, :]
        rb_e = real_B_encoded[i]

        dists = np.abs(fb_es - rb_e)
        dists = np.sum(dists, axis=1)
        dists = dists / np.count_nonzero(rb_e)
        min_B2B.append(np.min(dists))
    return min_B2B


def eval_for_ratio(real_B_encoded, fake_B_encoded, possible_elements, e1, e2):
    ie1 = possible_elements.index(e1)
    ie2 = possible_elements.index(e2)
    real_ratios = real_B_encoded[:, ie1] / real_B_encoded[:, ie2]
    fake_B_encoded_2d = fake_B_encoded.reshape(fake_B_encoded.shape[0] * fake_B_encoded.shape[1],
                                               fake_B_encoded.shape[2])
    fake_ratios = fake_B_encoded_2d[:, ie1] / fake_B_encoded_2d[:, ie2]
    rand_es = gen_rand_composition_vector(real_B_encoded)
    rand_ratios = rand_es[:, ie1] / rand_es[:, ie2]
    return real_ratios, rand_ratios, fake_ratios
