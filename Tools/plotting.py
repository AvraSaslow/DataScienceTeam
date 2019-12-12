import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, ax, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.invert_yaxis()


def plot_train_test_cm(pred_train, Y_train, pred_test, Y_test, savepath=None, axes=None):
    cm = confusion_matrix(y_true=Y_train, y_pred=pred_train)
    if len(cm) == 3:
        labels = ['Lose', 'Tie', 'Win']
    else:
        labels = ['Lose', 'Win']

    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        ax1, ax2 = axes
    ax1.set_title('Train results')
    plot_confusion_matrix(cm, ax1, labels)
    cm = confusion_matrix(y_true=Y_test, y_pred=pred_test)
    plot_confusion_matrix(cm, ax2, labels)
    ax2.set_title('Test results')
    if savepath is not None:
        plt.savefig(savepath)
    if axes is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax1, ax2