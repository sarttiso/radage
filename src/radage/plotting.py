import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from .radage import *
from .helper import *

def plot_ages_concordia(ages=[],
                   t1=None,
                   t2=None,
                   tw=False,
                   labels=None,
                   max_t_labels=10,
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

    Parameters
    ----------
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
        Maxmium number of points on concordia to be labeled with ages in Ma. Defaults to 10.

    concordia_env : boolean, optional
        Whether or not to plot uncertainty on concordia, defaults to False

    concordia_conf : float, optional
        Confidence interval for concordia uncertainty. Defaults to 0.95

    ax : matplotlib.axes, optional
        Axis to plot. If None, a new axis is created. Defaults to None.

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

    Returns
    -------
    ax : matplotlib.axes
        Axis object with plot.
    """

    if t1 is None: 
        min68_age = np.min(np.array([age.date68()[0] - 3*age.date68()[1] for age in ages]))
        if tw:
            min76_age = np.min(np.array([age.date76(conf=None) - 3*age.date76()[1] for age in ages]))
            t1 = np.min([min68_age, min76_age])
        else:
            min75_age = np.min(np.array([age.date75()[0] - 3*age.date75()[1] for age in ages]))
            t1 = np.min([min68_age, min75_age])

    if t2 is None:
        max68_age = np.max(np.array([age.date68()[0] + 3*age.date68()[1] for age in ages]))
        if tw:
            max76_age = np.max(np.array([age.date76(conf=None) + 3*age.date76()[1] for age in ages]))
            t2 = np.max([max68_age, max76_age])
        else:
            max5_age = np.max(np.array([age.date75()[0] + 3*age.date75()[1] for age in ages]))
            t2 = np.max([max68_age, max5_age])

    # if not provided, make labels nice round numbers located within the desired range
    if labels is None:
        locator = MaxNLocator(nbins=max_t_labels, 
                              steps=[1, 2, 5, 10],
                              prune='both')
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
    dt = t2 - t1
    t_conc = np.linspace(t1 - dt/2, t2 + dt/2, 500) # add buffer to make sure concordia is plotted fully
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
    
    # enforce limits
    xlim, ylim = axlim_conc([t1, t2], ax=ax, tw=tw)

    # plot age ellipses
    if tw:
        # only plot visible ellipses
        ages_plot = []
        for ii, age in enumerate(ages):
            if age.r238_206 - 3*age.r238_206_std < xlim[1] \
               and age.r238_206 + 3*age.r238_206_std > xlim[0] \
               and age.r207_206 - 3*age.r207_206_std < ylim[1] \
               and age.r207_206 + 3*age.r207_206_std > ylim[0]:
                ages_plot.append(age)
        patch_dict = patch_dict_validator(patch_dict, len(ages_plot))
        plot_ellipses_76_86(ages_plot, ax=ax, patch_dict=patch_dict)
    else:
        # only plot visible ellipses
        ages_plot = []
        for ii, age in enumerate(ages):
            if age.r207_235 - 3*age.r207_235_std < xlim[1] \
               and age.r207_235 + 3*age.r207_235_std > xlim[0] \
               and age.r206_238 - 3*age.r206_238_std < ylim[1] \
               and age.r206_238 + 3*age.r206_238_std > ylim[0]:
                ages_plot.append(age)
        patch_dict = patch_dict_validator(patch_dict, len(ages_plot))
        plot_ellipses_68_75(ages_plot, ax=ax, patch_dict=patch_dict)

    if tw:
        ax.set_xlabel('$^{238}\mathrm{U}/^{206}\mathrm{Pb}$')
        ax.set_ylabel('$^{207}\mathrm{Pb}/^{206}\mathrm{Pb}$')
    else:
        ax.set_xlabel('$^{207}\mathrm{Pb}/^{235}\mathrm{U}$')
        ax.set_ylabel('$^{206}\mathrm{Pb}/^{238}\mathrm{U}$')

    return ax

def annotate_concordia(ages, tw=False, ax=None, ann_style=None):
    """
    use this function to annotate concordia plots with times

    ages: list of numbers in Ma to annotate and plot on concordia
    tw: tera wasserberg space?
    ann_style: dict
    """
    n_ages = len(ages)
    if tw:
        x_lab, y_lab = concordia_tw(ages)
    else:
        x_lab, y_lab = concordia(ages)

    if ax == None:
        ax = plt.gca()

    if ann_style is None:
        ann_style = {'color': 'red',
                     'marker': 'o',
                     'linestyle': ''}

    for ii in range(n_ages):
        if tw:
            offset = (0, -15)
            ha = 'center'
        else:
            offset = (0, 5)
            ha = 'right'

    # time labels
    ax.plot(x_lab, y_lab, **ann_style)

    for ii in range(n_ages):
        ax.annotate(int(ages[ii]),
                    xy=(x_lab[ii], y_lab[ii]),
                    xytext=offset,
                    textcoords='offset points',
                    ha=ha)
        

def axlim_conc(tlims, ax=None, tw=False):
    """Set x and y lims for conccordia plot based on age range

    Parameters
    ----------
    tlims : array-like
        minimum and maximum age bounds to plot
    ax : matplotlib.pyplot.axes, optional
        Axis to set the limits for. If None, plt.gca(). Defaults to None.
    tw : boolean, optional
        Tera-Wasserburg concordia (True) or Wetherill (False) concordia. Defaults to False.

    Returns
    -------
    xlim : array-like
        x limits set
    ylim : array-like
        y limits set
    """
    if ax is None:
        ax = plt.gca()

    tlims = np.array(tlims)

    if tw:
        r86, r76 = concordia_tw(tlims)
        ax.set_xlim(np.flip(r86))
        ax.set_ylim(r76)
        return r86, r76
    else:
        r75, r68 = concordia(tlims)
        ax.set_xlim(r75)
        ax.set_ylim(r68)
        return r75, r68


def age_rank_plot_samples(samples_dict, sample_spacing=1, ax=None, 
                          sample_fontsize=10, sample_label_loc='top', **kwargs):
    """Plot age rank diagrams for multiple samples

    Parameters
    ----------
    samples_dict : dict
        Dictionary with samples as keys. Each key has another dictionary with required keys 'ages', 'ages 2s' which have arrays of the same length to plot ages. Optional keys are 
            'style': patch_dict for age_rank_plot()
            'mean': weighted mean; requires 'sig' and plots a box showing a weighted
            mean across the other ages
            'sig': uncertainty on weighted mean
            'xmin': Start coordinate for weighted mean box, in number of ages out of total for the sample.
            'xmax': End coordinate for weighted mean box, in number of ages out of total for the sample.
    sample_spacing : int, optional 
        Spacing between samples. Defaults to 1.
    ax : matplotlib.Axes, optional
        Axis to plot into. If None, one is created. Defaults to None.
    sample_fontsize : float, optional 
        Fontsize for labeling samples. Defaults to 10.
    sample_label_loc : str, optional
        Location to label sample, 'top' or 'bottom'. Defaults to 'top'.
    kwargs : dict
        Additional keyword arguments sent to age_rank_plot().

    Returns
    -------
    ax : matplotlib.Axes
        Axis with plot
    """
    # create axes if not provided
    if ax is None:
        ax = plt.axes()

    # loop over samples
    age_max, age_min = np.nan, np.nan  # keep track of min and max ages
    rank_start = 0
    for sample in samples_dict:
        cur_samp = samples_dict[sample]
        # number of ages in current sample
        n_ages = len(cur_samp['ages'])
        cur_ranks = np.arange(rank_start, n_ages+1+rank_start)
        # set up style
        style = patch_dict_validator(cur_samp.get('style', None), n_ages)
        # plot ranks
        age_rank_plot(cur_samp['ages'], cur_samp['ages 2s'], ranks=cur_ranks,
                      ax=ax, patch_dict=style, **kwargs)
        # plot weighted mean
        if ('mean' in cur_samp) and ('sig' in cur_samp):
            xmin = cur_samp.get('xmin', 0)
            xmax = cur_samp.get('xmax', n_ages)
            assert xmin < xmax, 'xmin must be less than xmax'
            cur_rect = Rectangle([rank_start-0.5 + xmin, 
                                  cur_samp['mean']-cur_samp['sig']],
                                 xmax-xmin, 
                                 2*cur_samp['sig'],
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

    return ax


def age_rank_plot(ages, ages_2s, ranks=None, ax=None, wid=0.6, patch_dict=None):
    """Rank-age plotting

    Parameters
    ----------
    ages : array-like 
        age means
    ages_2s : array-like
        (symmetric) age uncertainty to plot, 2-sigma. Same length as ages.
    ranks : array-like, optional
        Manually specified ranks (if plotting several different
        samples together). defaults to None
    ax : matplotlib.plot.axes, optional 
        Axis to plot into. Defaults to None. If none, one is created.
    wid : float, optional
        Width of age bar. Defaults to 0.6.
    patch_dict : dict or list or None, optional
        Style for Rectangle patches. Defaults to None. If one is provided, same styling is used for all patches. Otherwise, must be same length as ages.

    Returns
    -------
    ax : matplotlib.plot.axes
        Axis with plot
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

    return ax


def kde_plot(radages, t=None, bw='adaptive', kernel='gauss', weights='uncertainty',
             ax=None, fill=True, rug=True, 
             kde_style=None, kde_base_args=None, patch_dict=None, 
             rug_style=None):
    
    # useful to precompute dates
    if t is None or rug:
        dates_conc = np.array([age.date_207_238_concordia()[0:2] for age in radages])

    # set up default time range if not provided
    if t is None:
        t_min = np.min(dates_conc[:, 0] - 3*dates_conc[:, 1])
        t_max = np.max(dates_conc[:, 0] + 3*dates_conc[:, 1])
        t_range = t_max - t_min
        t_min = t_min - 0.05*t_range
        t_max = t_max + 0.05*t_range
        t = np.linspace(t_min, t_max, 1000)

    # set up axis
    if ax is None:
        ax = plt.axes()

    # set up base arguments for kde
    if kde_base_args is None:
        kde_base_args = {}

    # plot fill_between if requested
    cur_kde = kde(radages, t, bw=bw, kernel=kernel, weights=weights,
                  **kde_base_args) # call once
    if fill:
        # set up different default style
        patch_dict_def = {'facecolor': 'darkgray', 'alpha': 1}
        if patch_dict is None:
            patch_dict = patch_dict_def
        else:
            patch_dict = patch_dict_def | patch_dict
        patch_dict = patch_dict_validator(patch_dict, 1)
        ax.fill_between(t, cur_kde, **patch_dict[0], zorder=2)
        # if fill, make default style for kde invisible
        kde_style_def = {'linestyle': ''}
    else:
        kde_style_def = {'color': 'k', 'linestyle': '-'}

    # set up kde style
    if kde_style is None:
        kde_style = kde_style_def
    else:
        kde_style = kde_style_def | kde_style

    # plot kde
    ax.plot(t, cur_kde, **kde_style)
    if rug:
        rug_style_def = {'color': 'k', 'linewidth': 0.2}
        if rug_style is None:
            rug_style = rug_style_def
        else:
            rug_style = rug_style_def | rug_style
        # use weights to scale opacity for rug plot lines
        w = 1/dates_conc[:, 1]**2
        w = w/np.max(w)
        # get current y limits, plot rug under kde
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        yfact = 0.05
        # plot rug
        for ii, date in enumerate(dates_conc[:, 0]):
            ax.plot([date, date], [-yrange*yfact, 0], alpha=1, 
                    zorder=3, **rug_style)
        ax.set_ylim([-yfact*yrange, ylim[1]])
        ax.axhspan(-yfact*yrange, 0, facecolor='white', edgecolor='k',
                   zorder=2, alpha=1)

    
    # format axes
    ax.minorticks_on()
    ax.grid(which='minor', axis='x', linewidth=0.5)
    ax.grid(which='major', axis='x', linewidth=1.25)
    ax.set_xlim([np.min(t), np.max(t)])
    ax.set_yticks([])
    ax.set_xlabel('Age (Ma)')

    return ax


def plot_ellipses_68_75(ages, conf=0.95, patch_dict=None, ax=None):
    """Plot uncertainty ellipses for 206/238-207/235 ages

    Parameters
    ----------
    ages : list
        List of radage.UPb objects
    conf : float, optional
        Confidence level of ellipses, by default 0.95
    patch_dict : dict or list, optional
        Styling dictionary or list of dictionaries. If None, default styling. If list, must be same length as ages. By default None
    ax : matplotlib.pyplot.axes, optional
        Axes object to plot into. If None, one is generated. By default None.
    
    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes object with plot
    """
    patch_dict = patch_dict_validator(patch_dict, len(ages))

    if ax == None:
        ax = plt.axes()
    for ii, age in enumerate(ages):
        cur_ell = age.ellipse_68_75(conf=conf, patch_dict=patch_dict[ii])
        ax.add_patch(cur_ell)
    return ax


def plot_ellipses_76_86(ages, conf=0.95, patch_dict=None, ax=None):
    """Plot uncertainty ellipses for 207/206-238/206 ages

    Parameters
    ----------
    ages : list
        List of radage.UPb objects
    conf : float, optional
        Confidence level of ellipses, by default 0.95
    patch_dict : dict or list, optional
        Styling dictionary or list of dictionaries. If None, default styling. If list, must be same length as ages. By default None
    ax : matplotlib.pyplot.axes, optional
        Axes object to plot into. If None, one is generated. By default None.
    
    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes object with plot
    """
    patch_dict = patch_dict_validator(patch_dict, len(ages))

    if ax == None:
        ax = plt.axes()
    for ii, age in enumerate(ages):
        cur_ell = age.ellipse_76_86(conf=conf, patch_dict=patch_dict[ii])
        ax.add_patch(cur_ell)
    return ax