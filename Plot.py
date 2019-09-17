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

    if not scale_sig: sigmarker = np.copy(avgs)

    for i_epoch in range(D.numtrialepochs):
        start = i_epoch * binlength
        fin = i_epoch * binlength + D.num_timepoints

        if showsig and scale_sig: sigmarker[start:fin] = sig_in[i_epoch]

        for i_cond in range(avg_in.shape[0]):
            avgs[i_cond, start:fin] = avg_in[i_cond, i_epoch]
            sems[i_cond, start:fin] = sems_in[i_cond, i_epoch]
            if showsig and not scale_sig: sigmarker[i_cond, start:fin] = sig_in[i_cond, i_epoch]

    if showsig:
        if scale_sig:
            sigmarker *= np.nanmax(avgs)
            sigmarker[sigmarker==0] = np.nan
            sigmarker += 0.05
            if show_leg:
                ax.plot(sigmarker, color='black', lw=2, label='Significant')
            else:
                ax.plot(sigmarker, color='black', lw=2)
        else:
            for i_s, s in enumerate(sigmarker):
                ax.plot(s, color=f'C{i_s}', lw=2, ls='--')

    for i_trace, (avg_in, sem, label) in enumerate(zip(avgs, sems, leg_labels)):
        _plotpanel(ax, avg_in, sem, i_trace, ylab, label, show_leg)


# Plot single time point T-test results
def _makeandplotavgs_sigdec(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg, scale_sig):
    # SPlit into pvals and avgs
    pvals_in = avg_in[1]
    avg_in = avg_in[0]
    sems_in = sems_in[0]

    #
    avgs = np.empty((avg_in.shape[0], numpoints))
    pvals = np.empty((avg_in.shape[0], numpoints))
    avgs.fill(np.nan)
    sems = np.copy(avgs)


    for i_epoch in range(D.numtrialepochs):
        start = i_epoch * binlength
        fin = i_epoch * binlength + D.num_timepoints

        for i_cond in range(avg_in.shape[0]):
            avgs[i_cond, start:fin] = avg_in[i_cond, i_epoch]
            pvals[i_cond, start:fin] = pvals_in[i_cond, i_epoch]
            sems[i_cond, start:fin] = sems_in[i_cond, i_epoch]

    # Now plot the data
    for num, (avg, sem, label, pval) in enumerate(zip(avgs, sems, leg_labels, pvals)):

        # Split into significant and non-significant data points
        avg_sig = np.copy(avg)
        avg_nosig = np.copy(avg)

        avg_sig[pval>0.05] = np.nan
        sem[pval>0.05] = np.nan
        avg_nosig[pval<0.05] = np.nan

        if show_leg:
            ax.plot(avg_sig, label=f'{label}', color=f'C{num}', lw=3)
        else:
            ax.plot(avg_sig, color=f'C{num}', lw=3)
        ax.plot(avg_nosig, color=f'C{num}', lw=0.75)

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

# Plot permutation test results
def _makeandplotavgs_permtest(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg, scale_sig):
    # SPlit into pvals and avgs
    pvals_in = avg_in[1]
    pvals_in.fill(0.00001)  # Just put it all sig as this analysis doesn't use pvals
    avg_in = avg_in[0]
    sems_in = sems_in[0]

    avgs = np.empty((avg_in.shape[0], numpoints))
    pvals = np.empty((avg_in.shape[0], numpoints))
    avgs.fill(np.nan)
    sems = np.copy(avgs)
    sigmarker = np.copy(avgs)

    for i_epoch in range(D.numtrialepochs):
        start = i_epoch * binlength
        fin = i_epoch * binlength + D.num_timepoints

        for i_cond in range(avg_in.shape[0]):
            avgs[i_cond, start:fin] = avg_in[i_cond, i_epoch]
            pvals[i_cond, start:fin] = pvals_in[i_cond, i_epoch]
            sems[i_cond, start:fin] = sems_in[i_cond, i_epoch]
            sigmarker[i_cond, start:fin] = sig_in[i_cond]

    for i_s, s in enumerate(sigmarker):
        ax.plot(s, color=f'C{i_s}', lw=2, ls='--')

    # Now plot the data
    for i_cond, (avg, sem, label, pval, sig) in enumerate(zip(avgs, sems, leg_labels, pvals, sig_in)):

        # Split into significant and non-significant data points
        avg_sig = np.copy(avg)
        avg_nosig = np.copy(avg)

        # Erase non significant bits (below perm threshold)
        avg_sig[avg_sig<sig] = np.nan
        sem[avg_sig<sig] = np.nan

        if show_leg:
            ax.plot(avg_sig, label=f'{label}', color=f'C{i_cond}', lw=2, zorder=10)
        else:
            ax.plot(avg_sig, color=f'C{i_cond}', lw=2, zorder=10)
        ax.plot(avg_nosig, color=f'C{i_cond}', lw=0.75, zorder=1)

        ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.3, color=f'C{i_cond}')
        ax.set_xlim(0, numpoints)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(ylab, fontsize=11)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        plt.yticks(fontsize=13)


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
    plt.suptitle(title, x=0.05, fontsize=18, ha='left')
    f.legend(loc='upper right', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tick_params(axis='both', which='major', labelsize=13)


def GeneralPlot(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle, scale_sig, show_sig=True):
    numrows = avgs.shape[1]
    showlegends = [True if i==0 else False for i in range(numrows)]
    
    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areanames, avgs, sems, sigclusters)):

        f, axes = plt.subplots(numrows, 1, figsize=(width_regplot, height_regplot*numrows), sharex=True)

        if numrows > 1:
            for i, (label, leg_bool) in enumerate(zip(ytitles, showlegends)):
                _makeandplotavgs(avgs_area[i], sems_area[i], sig_area[i], axes[i], ylab=label, showsig=show_sig, leg_labels=trace_names, show_leg=leg_bool, scale_sig=scale_sig)

        else:
                _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=ytitles, showsig=show_sig, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)

        title = maintitle+ f' ({area})'
        _finalplotadjustments(f, title)
        U.savefig(savefolder, area)


def GeneralAllAreas(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle, scale_sig, show_sig=True):
    numrows = avgs.shape[1]
    if numrows > 1:
        raise Exception('Can only plot one row per area')

    showlegends = [True if i==0 else False for i in range(D.numareas)]

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)
        else:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=False, scale_sig=scale_sig)

    _finalplotadjustments(f, maintitle)

    U.savefig(savefolder, 'all')

# Plots result from t-test at every time point comparing accuracy to 50%
def DecoderSignificant_AllAreas(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle, scale_sig, show_sig=True):

    showlegends = [True if i==0 else False for i in range(D.numareas)]

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs_sigdec(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)
        else:
            _makeandplotavgs_sigdec(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=False, scale_sig=scale_sig)

    _finalplotadjustments(f, maintitle)

    U.savefig(savefolder, 'all_sig')


def DecoderPermSig_AllAreas(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle, scale_sig, show_sig=True):

    showlegends = [True if i==0 else False for i in range(D.numareas)]

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True, sharey=True)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs_permtest(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)
        else:
            _makeandplotavgs_permtest(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=False, scale_sig=scale_sig)

    _finalplotadjustments(f, maintitle)
    U.savefig(savefolder, 'all_sig_perm')

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs_permtest(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=showlegends, scale_sig=scale_sig)
        else:
            _makeandplotavgs_permtest(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=False, scale_sig=scale_sig)

    _finalplotadjustments(f, maintitle)

    U.savefig(savefolder, 'all_sig_perm2')


def PlotDist(dists, savefolder):
    linestyles = ['-', '--', '-.', ':', 'steps']

    for i_area, (dist, areaname) in enumerate(zip(dists, D.areanames)):

        for i_cond, cond in enumerate(dist):
            # Remove 0's from end
            cond = cond[~np.isnan(cond)]
            histY, histX = np.histogram(cond, bins=10)
            plt.plot(histX[1:], histY, label=f'{areaname} {i_cond}', linestyle=linestyles[i_cond%len(linestyles)], color=f'C{i_area}')

    plt.xlabel('Minimum number of samples for cell')
    plt.ylabel('Count')
    plt.legend()
    plt.xlim(xmin=0)
    D.savefig(f'{savefolder}/dist')

