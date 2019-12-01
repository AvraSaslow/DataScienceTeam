import pandas as pd
import numpy as np
import sqlite3 as sql
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

pd.options.display.expand_frame_repr = False


def plot_confusion_matrix(cm, fig, ax, classes,
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

conn = sql.connect('Data/database.sqlite')

# query = 'SELECT * FROM Match INNER JOIN League on League.id = Match.league_id'

query = 'SELECT * FROM Match WHERE league_id == 1729'

# fields = ['home_team_api_id', 'away_team_api_id', 'date', 'home_team_goal', 'away_team_goal']
# more_fields = ['goal', 'shoton', 'shotoff', 'possession', 'cross', 'corner', 'foulcommit']
# odds = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD']

df = pd.read_sql_query(query, conn, index_col='id')

df['GD'] = df['home_team_goal'] - df['away_team_goal']
df = df.loc[df['GD'] != 0]
df['result'] = np.where(df['GD'] > 0, 1, 0)

# CHOOSE FEATURES
# --------------------------------------------------------------------------------------------------
FEATURES = ['B365H', 'BWH', 'LBH']
df = df.loc[:, ['result'] + FEATURES]
# --------------------------------------------------------------------------------------------------

df = df.sample(frac=1)  # shuffle
df = df.dropna()  # drop NaNs

split = 0.80
split_idx = int(split * len(df))
train_df = df.iloc[:split_idx, :]
test_df = df.iloc[split_idx:, :]


X_train = train_df.loc[:, FEATURES]
T_train = train_df.loc[:, 'result']


RF = RandomForestClassifier()
RF.fit(X_train, T_train)
training_results = RF.predict(X_train)

Y = training_results
T = T_train
cm = confusion_matrix(y_true=T, y_pred=Y)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Train results')
plot_confusion_matrix(cm, fig, ax1, ['loss', 'win'])

# TEST
X_test = test_df.loc[:, FEATURES]
T_test = test_df.loc[:, 'result']

testing_results = RF.predict(X_test)

Y = testing_results
T = T_test
cm = confusion_matrix(y_true=T, y_pred=Y)
plot_confusion_matrix(cm, fig, ax2, ['loss', 'win'])
ax2.set_title('Test results')
plt.tight_layout()
plt.show()
