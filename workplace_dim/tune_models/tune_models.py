import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV, dump
from skopt.callbacks import CheckpointSaver
from skopt.utils import Integer, Real, Categorical

from settings import SEED
from utils import load_pkl

"""
select the best model
"""
checkpoint_saver = CheckpointSaver("./checkpoint.pkl",
                                   store_objective=False, )  # keyword arguments will be passed to `skopt.dump`


def get_optimizer(clf, search_space, kws={}):
    opt = BayesSearchCV(
        clf(**kws),
        search_spaces=search_space,
        random_state=SEED,
        n_iter=100,
        verbose=1,
        cv=5,
        scoring="accuracy",
        n_jobs=5
    )
    return opt


def tune(clf, ss, kws, dimset, name):
    dimset = copy.deepcopy(dimset)
    data = dimset.x.values
    target = dimset.y.values.ravel()
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=SEED)
    opt = get_optimizer(clf, ss, kws)
    opt.fit(x_train, y_train, callback=[checkpoint_saver, ])
    print(opt.best_score_)
    print(opt.best_params_)
    dump(opt, "opt-{}.pkl".format(name))


clf_knn = KNeighborsClassifier
ss_knn = {
    'p': Integer(1, 2, prior="uniform"),
    'n_neighbors': Integer(1, 10, prior="uniform"),
    'leaf_size': Integer(20, 40, prior="uniform"),
    'metric': Categorical(["minkowski", "euclidean", "manhattan", "chebyshev", ]),
    'weights': Categorical(["distance", "uniform"]),
}
kws_knn = {}

clf_rf = RandomForestClassifier
ss_rf = {
    'criterion': Categorical(["entropy", "gini"]),
    'n_estimators': Integer(10, 500, prior="uniform"),
}
kws_rf = {}

clf_lr = LogisticRegression
ss_lr = {
    'penalty': Categorical(['l1', 'l2', ]),
    'C': Real(1e-5, 1e3, prior="log-uniform"),
}
kws_lr = {
    'max_iter': 1000,
    'solver': 'liblinear',
    'multi_class': 'ovr',
}

if __name__ == '__main__':
    dimset = load_pkl("../../datasets/dimset.pkl")
    tune(clf_knn, ss_knn, kws_knn, dimset, "knn")
    tune(clf_lr, ss_lr, kws_lr, dimset, "lr")
    tune(clf_rf, ss_rf, kws_rf, dimset, "rf")
