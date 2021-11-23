import copy

import numpy as np
from joblib import Parallel, delayed
from skopt import dump
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Real, Integer

from gans.trainer import Trainer
from settings import SEED
from utils import timestamp

checkpoint_saver = CheckpointSaver("./checkpoint.pkl",
                                   store_objective=False, )  # keyword arguments will be passed to `skopt.dump`


def single_run(dataset, params, cvi, cvfold, main_name, save_freq=10000):
    dataset = copy.deepcopy(dataset)
    params["b_cv_fold"] = cvfold
    params["b_cv_index"] = cvi
    trainer = Trainer(params, dataset, "aug", wdir="{}-{}".format(main_name, cvi))
    trainer.save()
    prior_mean = trainer.train(load_pre_trained=False, save_freq=save_freq, eval_after_train=True)
    return prior_mean


def cv_run(dataset, params, cvfold=3, save_freq=10000):
    timestamp_cvrun = timestamp()
    cv_mean = Parallel(n_jobs=cvfold)(
        delayed(single_run)(dataset=dataset, params=params, cvfold=cvfold, cvi=i, main_name=timestamp_cvrun,
                            save_freq=save_freq) for i in
        range(cvfold))
    print("cv_run:", cv_mean)
    return np.mean(cv_mean)


def tune(dataset, chk=None, n_calls=50, nepochs=10002):
    def objective(hparams):
        g_blocks, lr_divider, lr_slowdown_param, cyc_weight, lambda_z = hparams
        lambda_z_B = lambda_z
        lambda_z_A = lambda_z
        params = dict(
            g_blocks=g_blocks,
            lr_divider=lr_divider,
            lr_slowdown_param=lr_slowdown_param,
            cyc_weight=cyc_weight,
            lambda_z_A=lambda_z_A,
            lambda_z_B=lambda_z_B,
            emd=None,
            nepochs=nepochs,
        )
        return cv_run(dataset, params, cvfold=3, save_freq=nepochs - 2)

    dimensions = [
        Integer(low=5, high=25, name="g_block"),
        Integer(low=1, high=10, name="lr_divider"),  #
        Real(low=0.9, high=0.9999, prior="uniform", name="lr_slowdown_param"),  #
        Real(low=0.01, high=10, prior="log-uniform", name="cyc_weight"),  #
        Real(low=0.01, high=10, prior="log-uniform", name="lambda_z"),  #
    ]

    if chk is not None:
        x0 = chk.x_iters
        y0 = chk.func_vals
        res = gp_minimize(objective, dimensions, x0=x0, y0=y0, random_state=SEED, n_calls=n_calls,
                          callback=[checkpoint_saver])
    else:
        res = gp_minimize(objective, dimensions, random_state=SEED, n_calls=n_calls, callback=[checkpoint_saver])
    dump(res, "tune.pkl", store_objective=False)
    return res
