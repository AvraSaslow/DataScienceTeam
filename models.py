import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import seaborn as sns

from Tools.plotting import plot_train_test_cm
from Tools.preprocessing import load_EPL_odds, load_EPL_point_diffs, load_EPL_stats_diffs

pd.options.display.expand_frame_repr = False


def train_classifier(X_train, X_test, Y_train, Y_test, estimator, parameters=None):
    """
    Trains model. If parameters supplied, run grid search and pick best model.
    Args:
        X: Input data
        Y: Labels
        estimator: sklearn model
        parameters: dict of lists of model parameters to iterate through.
                    If None, default model is trained.

    Returns:
        accuracy score, best trained model, values for plotting
    """
    if parameters is not None:
        gs_clf = GridSearchCV(estimator, parameters, cv=3, scoring='accuracy', verbose=3)
        gs_clf.fit(X_train, Y_train)
        clf = gs_clf.best_estimator_
    else:
        clf = estimator
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)

    pred_test = clf.predict(X_test)
    acc = accuracy_score(Y_test, pred_test)
    print(f'Test accuracy: {acc}')

    plt_vals = (pred_train, Y_train, pred_test, Y_test)
    return acc, clf, plt_vals


def binary_vs_three_categories(estimator, title, parameters=None, load_func=load_EPL_odds):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(title)
    data = load_func(ties=False)
    acc_bin, clf, plt_vals = train_classifier(*data, estimator, parameters)
    plot_train_test_cm(*plt_vals, axes=axes[0])
    data = load_func(ties=True)
    acc_3, clf, plt_vals = train_classifier(*data, estimator, parameters)
    plot_train_test_cm(*plt_vals, axes=axes[1])
    plt.tight_layout()
    plt.show()
    return acc_bin, acc_3


def train_classifiers(load_func):
    scores = {}
    estimator = LinearSVC()
    params = {'C': [0.1, 0.5, 1, 1.5, 2, 5, 10]}
    acc_bin, acc_3 = binary_vs_three_categories(estimator, 'LinearSVC', params, load_func=load_func)
    scores['LinearSVC'] = {'Binary outcome': acc_bin, '3 outcomes': acc_3}

    estimator = KNeighborsClassifier()
    params = {'n_neighbors': [2, 4, 5, 6, 8],
              'p': [1, 2, 3, 4]}
    acc_bin, acc_3 = binary_vs_three_categories(estimator, 'KNN', params, load_func=load_func)
    scores['KNN'] = {'Binary outcome': acc_bin, '3 outcomes': acc_3}

    estimator = AdaBoostClassifier()
    params = {'learning_rate': [0.5, 1, 1.5],
              'n_estimators': [20, 50, 80]}
    acc_bin, acc_3 = binary_vs_three_categories(estimator, 'AdaBoost', params, load_func=load_func)
    scores['AdaBoost'] = {'Binary outcome': acc_bin, '3 outcomes': acc_3}

    print(tabulate(pd.DataFrame(scores), headers='keys', tablefmt='psql'))


def correlation_study():
    df = load_EPL_stats_diffs()
    plt.figure(figsize=(6, 6))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


if __name__ == '__main__':
    # train_classifiers(load_func=load_EPL_odds)
    # train_classifiers(load_func=load_EPL_point_diffs)
    correlation_study()

