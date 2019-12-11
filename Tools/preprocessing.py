import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split

pd.options.display.expand_frame_repr = False

DATAPATH = pathlib.Path(__file__).parents[1].joinpath('Data')


def _load_EPL(ties, features=None, doublecount=False):
    EPL = pd.read_csv(DATAPATH.joinpath('EPL2.csv'), index_col=0).drop(
            columns=['id', 'index', 'home_team_api_id', 'away_team_api_id', 'match_api_id'])

    # ars = EPL.loc[EPL['team_name_stats'] == 'Arsenal']

    if doublecount:
        games = EPL
    else:
        games = EPL.iloc[::2, :]

    if not ties:
        games = games.loc[games['outcome'] != 'D']
    print(games.head(1))

    if features is None:
        X = games
    else:
        X = games.loc[:, features]
    Y = []
    num_wins, num_losses, num_draws = 0, 0, 0
    for i, row in games.iterrows():
        diff = row['home_team_goal'] - row['away_team_goal']
        if diff == 0:
            num_draws += 1
            Y.append(1)
        elif diff > 0:
            num_wins += 1
            Y.append(2)
        elif diff < 0:
            num_losses += 1
            Y.append(0)
    print(f'Wins: {num_wins}, Losses: {num_losses}, Draws: {num_draws}')

    return X, Y


def load_EPL_odds(ties=True):
    features = ['B365H', 'B365A', 'B365D']
    X, Y = _load_EPL(ties=ties, features=features)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, Y_train, Y_test


def load_EPL_stats_diffs(ties=True):
    features = ['fouls', 'crosses', 'corners', 'possession', 'on_target_shots', 'off_target_shots']
    X, Y = _load_EPL(ties=ties, features=features, doublecount=True)
    df = pd.DataFrame()
    for i in X.loc[::2, :].index:
        for col in X.columns:
            df.at[i, f'{col}_diff'] = X.at[i, col] - X.at[i + 1, col]
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)
    # return X_train, X_test, Y_train, Y_test
    df['outcome'] = Y[::2]
    return df


def get_diff(X, i, col_val, cols_to_sub):
    """ Get difference between two features (home - away) """
    return X.at[i, col_val] - X.at[i, cols_to_sub[0]] - (X.at[i + 1, col_val] - X.at[i, cols_to_sub[1]])


def load_EPL_point_diffs(ties=True):
    X, Y = _load_EPL(ties=ties, doublecount=True)
    new_X = X.copy()
    for i in X.loc[::2, :].index:
        new_X.at[i, 'GF_diff'] = get_diff(X, i, 'GF', ('home_team_goal', 'away_team_goal'))
        new_X.at[i, 'GA_diff'] = get_diff(X, i, 'GA', ('away_team_goal', 'home_team_goal'))
        new_X.at[i, 'points_diff'] = get_diff(X, i, 'cum_points', ['points_earned'] * 2)

    new_X = new_X.loc[::2, ['GF_diff', 'GA_diff', 'points_diff']]
    new_Y = Y[::2]
    # print(X.loc[:, ['GA', 'GF', 'home_team_goal', 'away_team_goal', 'points_earned', 'cum_points']].head(20))
    X_train, X_test, Y_train, Y_test = train_test_split(new_X, new_Y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    load_EPL_stats_diffs()