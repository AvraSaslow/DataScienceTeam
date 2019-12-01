import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

pd.options.display.expand_frame_repr = False

EPL = pd.read_csv('Data/EPL.csv')


def plot_attributes_over_seasons(attributes_to_plot=None):
    """
    Plots mean and std of given attributes over all recorded seasons of the EPL
    Args:
        attributes_to_plot: list of attributes to plot (e.g. ['home_team_goal', 'away_team_goal'])
    """
    if attributes_to_plot is None:
        attributes_to_plot = []
    seasons = EPL['season'].unique()

    mean_vals, sd_vals = defaultdict(list), defaultdict(list)
    for season in seasons:
        stats = EPL.loc[EPL['season'] == season].describe().loc[['mean', 'std'], :]
        for attr in attributes_to_plot:
            mean_vals[attr].append(stats.at['mean', attr])
            sd_vals[attr].append(stats.at['std', attr])

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
    plot_attributes_over_seasons(attributes_to_plot=['foul_home_team', 'foul_away_team'])