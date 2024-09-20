import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .radage import *

def plot_ages_concordia(ages=[],
                   t1=None,
                   t2=None,
                   tw=False,
                   labels=None,
                   max_t_labels=5,
                   concordia_env=False,
                   concordia_conf=0.95,
                   ax=None,
                   patch_dict=None,
                   concordia_style=None,
                   concordia_conf_style=None,
                   labels_style=None,
                   labels_text_style=None):
    """
    Draw age ellipses on a labeled concordia plot.

    Parameters:
    -----------
    ages: 1d array-like
        list of UPb objects to plot

    t1 : float, optional
        Youngest age to consider (Myr), defaults to youngest age in ages
    
    t2 : float, optional
        Oldest age to consider (Myr), defaults to oldest age in ages

    tw : boolean, optional
        Tera Wassergburg or conventional concordia, defaults to False

    labels : 1d array_like, optional
        Time points to label on concordia. Takes precedence over n_t_labels, defaults to None.

    max_labels : int, optional
        Maxmium number of points on concordia to be labeled with ages in Ma. Defaults to 5.

    concordia_env : boolean, optional
        Whether or not to plot uncertainty on concordia, defaults to False

    concordia_conf : float, optional
        Confidence interval for concordia uncertainty. Defaults to 0.95

    ax : axis to plot into if desired

    patch_dict : dict or list, optional
        list of dictionary of style parameters for the ellipse patch object. If a single dictionary is provided, it will be used for all ages. If None, a default style will be used.

    concordia_style : dict, optional
        Dictionary of style parameters for the concordia line

    concordia_conf_style : dict, optional
        Dictionary of style parameters for the concordia confidence interval, plotted as a patch

    labels_style : dict, optional
        Dictionary of style parameters for marker plotting of labeled times. 

    labels_text_style : dict, optional
        Dictionary of style parameters for text labels of ages.

    Returns:
    --------
    ax : matplotlib.axes
        Axis object with plot.
    """

    if t1 is None: 
        min68_age = np.min(np.array([age.age68()[0] - 3*age.age68()[1] for age in ages]))
        if tw:
            min76_age = np.min(np.array([age.age76()[0] - 3*age.age76()[0] for age in ages]))
            t1 = np.min([min68_age, min76_age])
        else:
            min75_age = np.min(np.array([age.age75()[0] - 3*age.age75()[1] for age in ages]))
            t1 = np.min([min68_age, min75_age])

    if t2 is None:
        max68_age = np.max(np.array([age.age68()[0] + 3*age.age68()[1] for age in ages]))
        if tw:
            max76_age = np.max(np.array([age.age76()[0] + 3*age.age76()[0] for age in ages]))
            t2 = np.max([max68_age, max76_age])
        else:
            max5_age = np.max(np.array([age.age75()[0] + 3*age.age75()[1] for age in ages]))
            t2 = np.max([max68_age, max5_age])

    # take some percentage below min age and above max age for plotting bounds of concordia
    pct = 0.1
    t1 = (1 - pct) * t1
    t2 = (1 + pct) * t2

    # if not provided, make labels nice round numbers located within the desired range
    if labels is None:
        locator = MaxNLocator(nbins=max_t_labels, 
                              steps=[1, 2, 5, 10])
        t_lab = locator.tick_values(t1, t2)
        n_t_labels = len(t_lab)
    else:
        t_lab = np.array(labels)
        n_t_labels = len(t_lab)

    if tw:
        x_lab, y_lab = concordia_tw(t_lab)
    else:
        x_lab, y_lab = concordia(t_lab)

    # make concordia line
    t_conc = np.linspace(t1, t2, 500)
    if tw:
        x_conc, y_conc = concordia_tw(t_conc)
    else:
        x_conc, y_conc = concordia(t_conc)

    # plot into provided axis or create new one
    if ax == None:
        ax = plt.axes()

    # plot concordia
    conc_style_def = {'color': 'k', 'linestyle': '-', 'linewidth': 1}
    if concordia_style is None:
        conc_style = conc_style_def
    else:
        conc_style = conc_style_def | concordia_style
    ax.plot(x_conc, y_conc, **conc_style)

    # concordia confident interval
    if concordia_env:
        conc_conf_style_def = {'color': 'gray', 'alpha': 0.5}
        if concordia_conf_style is None:
            conc_conf_style = conc_conf_style_def
        else:   
            conc_conf_style = conc_conf_style_def | concordia_conf_style
        lower_conc, upper_conc = concordia_confint(t_conc, conf=concordia_conf)
        conc_conf = np.concatenate([lower_conc, np.flipud(upper_conc)], axis=0)
        ax.fill(conc_conf[:, 0], conc_conf[:, 1], **conc_conf_style)

    # time labels
    labels_style_def = {'color': 'k', 'marker': 'o', 'linestyle': '', 'markersize': 5}
    if labels_style is None:
        labels_style = labels_style_def
    else:
        labels_style = labels_style_def | labels_style
    ax.plot(x_lab, y_lab, **labels_style)

    # annotation of labeled times
    labels_text_style_def = {'color': 'k', 'fontsize': 10}
    if labels_text_style is None:
        labels_text_style = labels_text_style_def
    else:
        labels_text_style = labels_text_style_def | labels_text_style
    for ii in range(n_t_labels):
        if tw:
            offset = (0, -15)
            ha = 'right'
        else:
            offset = (0, 5)
            ha = 'right'

        ax.annotate(int(t_lab[ii]),
                    xy=(x_lab[ii], y_lab[ii]),
                    xytext=offset,
                    textcoords='offset points',
                    ha=ha,
                    **labels_text_style)

    # plot age ellipses
    patch_dict = patch_dict_validator(patch_dict, len(ages))
    if tw:
        plot_ellipses_76_86(ages, ax=ax, patch_dict=patch_dict)
    else:
        plot_ellipses_68_75(ages, ax=ax, patch_dict=patch_dict)

    # enforce limits
    buf = 0.05
    ax.set_xlim([np.min(x_conc) * (1 - buf), np.max(x_conc) * (1 + buf)])
    ax.set_ylim([np.min(y_conc) * (1 - buf), np.max(y_conc) * (1 + buf)])

    if tw:
        ax.set_xlabel('$^{238}\mathrm{U}/^{206}\mathrm{Pb}$')
        ax.set_ylabel('$^{207}\mathrm{Pb}/^{206}\mathrm{Pb}$')
    else:
        ax.set_xlabel('$^{207}\mathrm{Pb}/^{235}\mathrm{U}$')
        ax.set_ylabel('$^{206}\mathrm{Pb}/^{238}\mathrm{U}$')

    return ax


def age_rank_plot_samples(samples_dict, sample_spacing=1, ax=None, 
                          sample_fontsize=10, sample_label_loc='top', **kwargs):
    """plot age rank for multiple samples

    Args:
        samples_dict (dict): dictionary with samples. each sample key has another
            dictionary with required keys 'ages', 'ages 2s' which have arrays of the
            same length to plot ages. optional keys are 
            'style': patch_dict for age_rank_plot
            'mean': weighted mean; requires 'sig' and plots a box showing a weighted
                mean across the other ages
            'sig': uncertainty on weighted mean 
        sample_spacing (int, optional): spacing between samples. Defaults to 1.
        ax (matplotlib.Axes, optional): axis to plot into. Defaults to None.
        sample_fontsize (float, optional): fontsize for labeling samples. Defaults to
        10.
        sample_label_loc (str, optional): location to label sample, 'top' or 'bottom'.
        Defaults to 'top'.
        kwargs: sent to age_rank_plot
    """
    if ax is None:
        ax = plt.axes()

    n_samples = len(samples_dict)

    style_default = {}

    # loop over samples
    age_max, age_min = np.nan, np.nan  # keep track of min and max ages
    rank_start = 0
    for sample in samples_dict:
        cur_samp = samples_dict[sample]
        n_ages = len(cur_samp['ages'])
        cur_ranks = np.arange(rank_start, n_ages+1+rank_start)
        # set style
        if 'style' in cur_samp:
            style = cur_samp['style']
        else:
            style = style_default
        # plot ranks
        age_rank_plot(cur_samp['ages'], cur_samp['ages 2s'], ranks=cur_ranks,
                      ax=ax, patch_dict=style, **kwargs)
        # plot mean
        if ('mean' in cur_samp) and ('sig' in cur_samp):
            cur_rect = Rectangle([rank_start-0.5, cur_samp['mean']-cur_samp['sig']],
                                 n_ages, 2*cur_samp['sig'],
                                 color='gray', alpha=0.5, zorder=0)
            ax.add_patch(cur_rect)

        # update min and max
        cur_max = np.max(cur_samp['ages']+cur_samp['ages 2s'])
        cur_min = np.min(cur_samp['ages']-cur_samp['ages 2s'])
        age_max = np.nanmax([age_max, cur_max])
        age_min = np.nanmin([age_min, cur_min])
        # annotate
        if sample_label_loc == 'top':
            ax.annotate(sample, (rank_start + n_ages/2 - 0.5, cur_min),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=sample_fontsize)
        elif sample_label_loc == 'bottom':
            ax.annotate(sample, (rank_start + n_ages/2 - 0.5, cur_max),
                        xytext=(0, -1), textcoords='offset points',
                        ha='center', va='top', fontsize=sample_fontsize)
        else:
            ValueError('sample_label_loc must be either "top" or "bottom"')
        # update rank start
        rank_start = rank_start + n_ages + sample_spacing

    # set limits
    ax.set_xlim([-sample_spacing, rank_start+1])
    ax.set_ylim([age_min, age_max])
    ax.invert_yaxis()


def age_rank_plot(ages, ages_2s, ranks=None, ax=None, wid=0.6, patch_dict=None):
    """rank-age plotting

    Parameters:
    -----------
        ages (array-like): age means
        ages_2s (array-like): (symmetric) age uncertainty to plot
        ranks (array-like): manually specified ranks (if plotting several different
            samples together). defaults to None
        ax (matplotlib.axes, optional): axis to plot into. Defaults to None.
        wid (float, optional): width of age bar. Defaults to 0.6.
        patch_dict (list, optional): list of style dicts for Rectangle patches.
            Defaults to None. If one is provided, same styling is used for all patches.
            Otherwise, must be same length as ages.
    """
    # set up a default stle
    patch_dict = patch_dict_validator(patch_dict, len(ages))

    if ax is None:
        ax = plt.axes()

    # sort ages
    idx_sort = np.argsort(-ages)
    ages = ages[idx_sort]
    ages_2s = ages_2s[idx_sort]
    # also sort styling
    patch_dict = [patch_dict[idx] for idx in idx_sort]

    n_ages = len(ages)

    if ranks is None:
        ranks = np.arange(n_ages)

    for ii in range(n_ages):
        bot = ages[ii] - ages_2s[ii]
        height = 2*ages_2s[ii]
        cur_rect = Rectangle([ranks[ii]-wid/2, bot], wid, height, **patch_dict[ii])
        ax.add_patch(cur_rect)

    xlim = [0-wid, n_ages+wid-1]
    vert_range = np.max(ages+ages_2s) - np.min(ages-ages_2s)
    vert_fact = 0.05
    ylim = [np.min(ages-ages_2s)-vert_fact*vert_range,
            np.max(ages+ages_2s)+vert_fact*vert_range]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def patch_dict_validator(patch_dict, n):
    """
    Validate patch_dict and returns list for styling of plotted patches.

    Parameters:
    -----------
        patch_dict : dict or list
            If a dictionary, same style will be used for all patches. If a list of dictionaries, must have length equal to n, and each dictionary will be used for each patch.
        n : int
            Number of patches to style

    Returns:
    --------
        patch_dict : list 
            validated list
    """
    # set up a default style
    patch_dict_def = {'facecolor': 'lightgray',
                      'linewidth': 0.5,
                      'edgecolor': 'k',
                      'alpha': 0.3}
    if patch_dict is None:
        patch_dict = n * [patch_dict_def]
    elif type(patch_dict) is dict:
        patch_dict = patch_dict_def | patch_dict
        patch_dict = n * [patch_dict]
    else:
        assert len(patch_dict) == n, 'Need one style dictionary per age.'

    return patch_dict


def plot_ellipses_68_75(ages, conf=0.95, patch_dict=None, ax=None):
    patch_dict = patch_dict_validator(patch_dict, len(ages))

    if ax == None:
        ax = plt.axes()
    for ii, age in enumerate(ages):
        cur_ell = age.ellipse_68_75(conf=conf, patch_dict=patch_dict[ii])
        ax.add_patch(cur_ell)


def plot_ellipses_76_86(ages, conf=0.95, patch_dict=None, ax=None):
    patch_dict = patch_dict_validator(patch_dict, len(ages))

    if ax == None:
        ax = plt.axes()
    for ii, age in enumerate(ages):
        cur_ell = age.ellipse_76_86(conf=conf, patch_dict=patch_dict[ii])
        ax.add_patch(cur_ell)