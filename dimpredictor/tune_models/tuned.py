import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from skopt import load, BayesSearchCV, dump

from cacgan.data.dataset import DimDataset
from cacgan.utils import SEED, load_pkl

sns.set_theme()
sns.set_style("whitegrid")

opt_knn = load("models/opt-knn.pkl")
opt_lr = load("models/opt-lr.pkl")
opt_rf = load("models/opt-rf.pkl")
opts = [opt_rf, opt_lr, opt_knn]
names = ["Random forest", "Logistic regression", "K-nearest neighbor"]

if __name__ == '__main__':
    dimset = load_pkl("../../dataset/dimset.pkl")
    dimset: DimDataset
    data = dimset.x.values
    target = dimset.y.values.ravel()

    fig = plt.figure(figsize=(3.2, 3))

    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=SEED)
    cv_scores = []
    cv_errs = []
    test_scores = []
    for opt, name in zip(opts, names):
        # for opt, name in zip(opts[-1:], names[-1:]):
        opt: BayesSearchCV
        best_clf = opt.best_estimator_
        scores = cross_val_score(best_clf, x_train, y_train, cv=5)
        print("cv scores:", scores, np.mean(scores), np.std(scores))
        print("mean cv score", np.mean(scores))
        cv_scores.append(np.mean(scores))
        test_score = opt.best_estimator_.score(x_test, y_test)
        print("test score:", test_score)
        test_scores.append(test_score)
        cv_errs.append(np.std(scores))
        dump(opt.best_estimator_, "models/tuned-{}.pkl".format(name))
    plt.hlines(0.374, xmin=-1, xmax=3, ls=":", color="k", label="Baseline")

    print(cv_errs, test_scores, cv_scores)

    width = 0.2
    xs = np.array([0.0, 1.0, 2.0])
    labels = ["Held out test", "CV"]
    i = 0
    for vs in [test_scores, cv_scores]:
        if labels[i] == "Held out test":
            plt.bar(xs, vs, width=width, label=labels[i])
        else:
            plt.bar(xs, vs, width=width, label=labels[i], yerr=cv_errs)
        i += 1
        xs += width
    plt.xticks(xs - width, [n.replace(" ", "\n") for n in names])
    plt.xlim([-0.3, 2.5])
    plt.ylim([0.3, 0.8])
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=[1.3, 1.2], loc='upper right', ncol=3)
    plt.savefig("tuned.tiff", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
