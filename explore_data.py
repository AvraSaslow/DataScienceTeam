import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

pd.options.display.expand_frame_repr = False

EPL = pd.read_csv('Data/EPL.csv')


def plot_attributes_over_seasons(attributes_to_plot, team=None, relative=False):
    """
    Plots mean and std of given attributes over all recorded seasons of the EPL
    Args:
        attributes_to_plot: list of attributes to plot (e.g. ['home_team_goal', 'away_team_goal'])
        relative: if True, scales relative to first season (good for comparing attributes)
        team: specify a team, else average over all teams
    """
    seasons = EPL['season'].unique()

    mean_vals, sd_vals = defaultdict(list), defaultdict(list)
    denom = {}
    if team is None:
        df = EPL
    else:
        df = EPL[(EPL['home_team'] == team) | (EPL['away_team'] == team)]
        for attr in attributes_to_plot:
            assert not attr.endswith('_team'), f'cannot use {attr} with specified team'

    for season in seasons:
        season_df = df.loc[df['season'] == season]
        for attr in attributes_to_plot:
            if team is None:
                stats = season_df.describe().loc[['mean', 'std'], :]
                avg, sd = stats.at['mean', attr], stats.at['std', attr]
            else:
                stats = np.concatenate(
                    [season_df.loc[season_df['home_team'] == team, attr + '_home_team'].values,
                     season_df.loc[season_df['away_team'] == team, attr + '_away_team'].values])
                avg, sd = stats.mean(), stats.std()
            if season == seasons[0]:
                denom[attr] = {'avg': avg, 'sd': sd}
            if relative:
                avg, sd = avg / denom[attr]['avg'], sd / denom[attr]['sd']
            mean_vals[attr].append(avg)
            sd_vals[attr].append(sd)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 6))
    for attr in attributes_to_plot:
        axes[0].plot(seasons, mean_vals[attr], label=attr)
        axes[1].plot(seasons, sd_vals[attr], label=attr)
    axes[0].legend()
    axes[0].set_title('Mean')
    axes[1].legend()
    axes[1].set_title('Std Deviation')
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    print(EPL.columns)
    print(EPL['home_team'].unique())
    plot_attributes_over_seasons(
        attributes_to_plot=['goal', 'on_target_shot', 'off_target_shot'],
        team='Manchester United',
        relative=False
    )