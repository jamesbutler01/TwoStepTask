import numpy as np
import matplotlib.pyplot as plt
import Details as D
import Utils as U

# Static plot parameters
buffer = 10  # Gap between each task event
numpoints = D.num_timepoints * D.numtrialepochs + buffer * D.numtrialepochs
binlength = D.num_timepoints + buffer
xtickpos = [int((i_epoch + 0.1) * binlength) for i_epoch in range(D.numtrialepochs)]
width_regplot = 15
height_regplot = 2.75


def _makeandplotavgs(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg, scale_sig):
    avgs = np.empty((avg_in.shape[0], numpoints))
    avgs.fill(np.nan)
    sems = np.copy(avgs)
    sigmarker = np.empty(numpoints)
    sigmarker.fill(np.nan)

    for i_epoch in range(D.numtrialepochs):
        start = i_epoch * binlength
        fin = i_epoch * binlength + D.num_timepoints

        if showsig:
            sigmarker[start:fin] = sig_in[i_epoch]

        for i_trans in range(avg_in.shape[0]):
            avgs[i_trans, start:fin] = avg_in[i_trans, i_epoch]
            sems[i_trans, start:fin] = sems_in[i_trans, i_epoch]

    if showsig:
        if scale_sig:
            sigmarker *= np.nanmax(avgs)
            sigmarker[sigmarker==0] = np.nan
            sigmarker += 0.05
        if show_leg:
            ax.plot(sigmarker, color='black', lw=2, label='Significant')
        else:
            ax.plot(sigmarker, color='black', lw=2)

    for i_trace, (avg_in, sem, label) in enumerate(zip(avgs, sems, leg_labels)):
        _plotpanel(ax, avg_in, sem, i_trace, ylab, label, show_leg)


def _plotpanel(ax, avg, sem, num, ylab, label, show_leg):
    if show_leg:
        ax.plot(avg, label=f'{label}', color=f'C{num}', lw=3)
    else:
        ax.plot(avg, color=f'C{num}', lw=3)
    ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.3, color=f'C{num}')
    ax.set_xlim(0, numpoints)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel(ylab, fontsize=11)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    plt.yticks(fontsize=13)


def _finalplotadjustments(f, title):
    plt.xticks(xtickpos, D.names_epochs, fontsize=11)

    plt.suptitle(title, x=0.25, fontsize=18)
    f.legend(loc='upper right', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tick_params(axis='both', which='major', labelsize=13)


def GeneralPlot(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle, scale_sig):
    numrows = avgs.shape[1]
    showlegends = [True if i==0 else False for i in range(numrows)]
    
    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areas, avgs, sems, sigclusters)):

        f, axes = plt.subplots(numrows, 1, figsize=(width_regplot, height_regplot*numrows), sharex=True)

        if numrows > 1:
            for i, (label, leg_bool) in enumerate(zip(ytitles, showlegends)):
                _makeandplotavgs(avgs_area[i], sems_area[i], sig_area[i], axes[i], ylab=label, showsig=True, leg_labels=trace_names, show_leg=leg_bool, scale_sig=scale_sig)

        else:
                _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=ytitles, showsig=True, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)

        title = maintitle+ f' ({area})'
        _finalplotadjustments(f, title)
        U.savefig(savefolder, area)


def GeneralAllAreas(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle, scale_sig):
    numrows = avgs.shape[1]
    if numrows > 1:
        raise Exception('Can only plot one row per area')

    showlegends = [True if i==0 else False for i in range(D.numareas)]

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True, sharey=True)

    for i_area, (area, avgs_area, sems_area, sig_area, axes) in enumerate(zip(D.areas, avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)
        else:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=False, scale_sig=scale_sig)

    _finalplotadjustments(f, maintitle)
    U.savefig(savefolder, 'all')

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True)

    for i_area, (area, avgs_area, sems_area, sig_area, axes) in enumerate(zip(D.areas, avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)
        else:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=False, scale_sig=scale_sig)

    _finalplotadjustments(f, maintitle)

    U.savefig(savefolder, 'all2')


def PlotDist(dists, savefolder):
    for dist, areaname in zip(dists, D.areanames):
        # Remove 0's from end
        dist = np.trim_zeros(dist, trim='b')
        histY, histX = np.histogram(dist, bins=10)
        plt.plot(histX[1:], histY, label=areaname)
    plt.xlabel('Minimum number of samples for cell')
    plt.ylabel('Count')
    plt.legend()
    plt.xlim(xmin=0)
    D.savefig(f'{savefolder}/dist')

