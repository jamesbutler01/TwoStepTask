import numpy as np
import matplotlib.pyplot as plt
import Details as D
import Utils as U

# Static plot parameters
buffer = 5  # Gap between each task event
numpoints = D.n_timepoints * D.numtrialepochs + buffer * D.numtrialepochs
binlength = D.n_timepoints + buffer
xtickpos = [int((i_epoch * binlength) + D.converttimetosmoothedtrace(0)) for i_epoch in range(D.numtrialepochs)]
width_regplot = 15
height_regplot = 2.75


def _makeandplotavgs(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg, scale_sig, n_epochs=D.numtrialepochs):
    numpoints = D.n_timepoints * n_epochs + buffer * n_epochs
    avgs = np.empty((avg_in.shape[0], numpoints))
    print(n_epochs, numpoints)
    avgs.fill(np.nan)
    sems = np.copy(avgs)
    sigmarker = np.empty(numpoints)
    sigmarker.fill(np.nan)

    if not scale_sig: sigmarker = np.copy(avgs)

    for i_epoch in range(n_epochs):
        start = i_epoch * binlength
        fin = i_epoch * binlength + D.n_timepoints

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
        _plotpanel(ax, avg_in, sem, i_trace, ylab, label, show_leg, numpoints)


# Plot single time point T-test results
def _makeandplotavgs_sigdec(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg, scale_sig, peak_epoch=None, peak_tp=None, n_epochs=D.numtrialepochs, plotscalar=1):
    numpoints = D.n_timepoints * n_epochs + buffer * n_epochs

    # Split into pvals and avgs
    pvals_in = avg_in[1]
    avg_in = avg_in[0]
    sems_in = sems_in[0]

    avgs = np.empty((avg_in.shape[0], numpoints))
    pvals = np.empty((avg_in.shape[0], numpoints))
    avgs.fill(np.nan)
    sems = np.copy(avgs)

    for i_epoch in range(n_epochs):
        start = i_epoch * binlength
        fin = i_epoch * binlength + D.n_timepoints

        for i_cond in range(avg_in.shape[0]):
            avgs[i_cond, start:fin] = avg_in[i_cond, i_epoch]
            pvals[i_cond, start:fin] = pvals_in[i_cond, i_epoch]
            sems[i_cond, start:fin] = sems_in[i_cond, i_epoch]

    # Now plot the data
    for i_cond, (avg, sem, label, pval) in enumerate(zip(avgs, sems, leg_labels, pvals)):

        # Split into significant and non-significant data points
        avg_sig = np.copy(avg)
        avg_nosig = np.copy(avg)

        avg_sig[pval>0.05] = np.nan
        sem[pval>0.05] = np.nan
        avg_nosig[pval<0.05] = np.nan

        if peak_tp is None:
            c = f'C{i_cond}'
            lw = 2 * plotscalar
        else:
            if i_cond==0:
                c = 'gray'
                lw=1 * plotscalar
            else:
                c=f'C{i_cond-1}'
                lw=2 * plotscalar

        if show_leg:
            ax.plot(avg_sig, label=f'{label}', color=c, lw=lw)
        else:
            ax.plot(avg_sig, color=c, lw=lw)
        ax.plot(avg_nosig, color=c, lw=1)

        ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.3, color=c)

    finalpaneltouches(ax, ylab, numpoints)

    if peak_tp is not None:
        vline = peak_epoch * binlength + peak_tp
        ax.axvline(vline, color='red', lw=2)



# Plot permutation test results
def _makeandplotavgs_permtest(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg, scale_sig, n_epochs=D.numtrialepochs):
    numpoints = D.n_timepoints * n_epochs + buffer * n_epochs

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

    for i_epoch in range(n_epochs):
        start = i_epoch * binlength
        fin = i_epoch * binlength + D.n_timepoints

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
        finalpaneltouches(ax, ylab, numpoints)


def finalpaneltouches(ax, ylab, npoints=numpoints):
    ax.set_xlim(0, npoints)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel(ylab, fontsize=11)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    plt.yticks(fontsize=13)


def _plotpanel(ax, avg, sem, num, ylab, label, show_leg, numpoints):
    if show_leg:
        ax.plot(avg, label=f'{label}', color=f'C{num}', lw=2)
    else:
        ax.plot(avg, color=f'C{num}', lw=2)
    ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.3, color=f'C{num}')
    finalpaneltouches(ax, ylab, numpoints)


def _finalplotadjustments(f, title, names_epochs=D.names_epochs):
    labelpos = np.array([i * (D.n_timepoints + 1) + D.converttimetosmoothedtrace(0) for i in
                         range(len(names_epochs))])
    plt.xticks(labelpos, names_epochs, fontsize=11)
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


def GeneralAllAreas(avgs, sems, sigclusters, trace_names, savefolder, maintitle, scale_sig, show_sig=True):
    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True, sharey=True)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
        _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=i_area==0, scale_sig=scale_sig)

    _finalplotadjustments(f, maintitle)

    U.savefig(savefolder, 'all')

sess_rows = 8
sess_cols = 8

def GeneralAllSess(avgs, sems, sigclusters, trace_names, savefolder, maintitle, scale_sig, names_epochs, show_sig=True, sigplot=False):

    f, axes_all = plt.subplots(sess_rows, sess_cols, figsize=(width_regplot*sess_cols/3, height_regplot*sess_rows), sharex=True, sharey=True)
    axes_all = np.ndarray.flatten(axes_all)

    axes_all[30].axis('off')
    axes_all = np.delete(axes_all, 30)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
        # First row only
        _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=f'Sess: {i_area}', showsig=show_sig, leg_labels=trace_names, show_leg=i_area==0, scale_sig=scale_sig, n_epochs=len(names_epochs))

        if i_area < 30:
            axes.patch.set_facecolor('lavender')
        else:
            axes.patch.set_facecolor('lightgray')

    # Switch off remaining axes
    for i_ax in range(i_area+1, len(axes_all)):
        axes_all[i_ax].axis('off')

    _finalplotadjustments(f, maintitle, names_epochs)
    plt.subplots_adjust(wspace=0.01)
    U.savefig(savefolder, 'all')

def DecoderSignificant_AllSess(avgs, sems, sigclusters, trace_names, savefolder, maintitle, peak_epoch, peak_tp, scale_sig, names_epochs, show_sig=True):

    f, axes_all = plt.subplots(sess_rows, sess_cols, figsize=(width_regplot*sess_cols/3, height_regplot*sess_rows), sharex=True, sharey=True)
    axes_all = np.ndarray.flatten(axes_all)

    axes_all[30].axis('off')
    axes_all = np.delete(axes_all, 30)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
        _makeandplotavgs_sigdec(avgs_area, sems_area, sig_area, axes, ylab=f'Sess: {i_area}', showsig=show_sig, leg_labels=trace_names, show_leg=i_area==0, scale_sig=scale_sig, peak_epoch=peak_epoch, peak_tp=peak_tp, n_epochs=len(names_epochs), plotscalar=1.5)

        if i_area < 30:
            axes.patch.set_facecolor('lavender')
        else:
            axes.patch.set_facecolor('lightgray')

    # Switch off remaining axes
    for i_ax in range(i_area+1, len(axes_all)):
        axes_all[i_ax].axis('off')

    _finalplotadjustments(f, maintitle, names_epochs)
    plt.subplots_adjust(wspace=0.01)
    U.savefig(savefolder, 'sig')

# Plots result from t-test at every time point comparing accuracy to 50%
def DecoderSignificant_AllAreas(avgs, sems, sigclusters, trace_names, savefolder, maintitle, peak_epoch, peak_tp, scale_sig, show_sig=True):

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True, sharey=True)

    for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
            _makeandplotavgs_sigdec(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=i_area==0, scale_sig=scale_sig, peak_epoch=peak_epoch, peak_tp=peak_tp)

    _finalplotadjustments(f, maintitle)

    U.savefig(savefolder, 'all_sig')


def DecoderPermSig_AllAreas(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle, scale_sig, show_sig=True):

    for suffix, sharey in zip(('', '2'), (False, True)):

        f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True, sharey=sharey)

        for i_area, (avgs_area, sems_area, sig_area, axes) in enumerate(zip(avgs, sems, sigclusters, axes_all)):
            _makeandplotavgs_permtest(avgs_area, sems_area, sig_area, axes, ylab=D.areanames[i_area], showsig=show_sig, leg_labels=trace_names, show_leg=i_area==0, scale_sig=scale_sig)

        _finalplotadjustments(f, maintitle)
        U.savefig(savefolder, 'all_sig_perm'+suffix)


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


def PlotDecStab(accs_all, savefolder, epochnames):

    fs_ax = 14
    fs_tit = 18
    width = 25
    height = 8
    n_areas = accs_all.shape[0]
    f = plt.figure(figsize=(width, height))

    gs = plt.GridSpec(3, n_areas, height_ratios=[1, 0.01, 0.5], hspace=0)

    axes = [f.add_subplot(gs[0, i]) for i in range(n_areas)]
    axes_1d = [f.add_subplot(gs[2, i]) for i in range(n_areas)]
    vmax = np.around(np.nanmax(accs_all), 2)
    vmin = np.around(np.nanmin(accs_all), 2)

    labelpos = np.array([i * (D.n_timepoints + 1) + D.converttimetosmoothedtrace(0) for i in
                         range(len(epochnames))])
    ylabelpos = len(accs_all[0]) - labelpos

    def finishing_touches(ax, arr, ax_1d, ylab, vmin, vmax):
        ax_1d.plot(np.diagonal(arr), zorder=2)

        if ylab:
            ax.set_yticks(ylabelpos)
            ax.set_yticklabels(epochnames, fontsize=fs_ax)
            ax_1d.set_yticks((vmin, vmax))
            ax_1d.set_yticklabels((vmin, vmax), fontsize=fs_ax)
            ax_1d.set_ylabel('Accuracy (%)', fontsize=fs_ax)
        else:
            ax.set_yticks([] * 2)
            ax_1d.set_yticks([]*2)

        ax.set_xticks([] * 2)
        ax.set_title(area, fontsize=fs_tit)

        for xpos, ypos in zip(labelpos, ylabelpos):
            ax_1d.axvline(xpos, color='gray', zorder=0, linestyle='--')
            ax.axvline(xpos, color='gray', zorder=3, linestyle='--')
            ax.axhline(ypos, color='gray', zorder=3, linestyle='--')

        ax_1d.set_ylim(vmin, vmax)
        ax_1d.set_xlim(0, len(np.diagonal(arr)))
        ax_1d.set_xticks(labelpos)
        ax_1d.set_xticklabels(epochnames, fontsize=fs_ax)
        ax_1d.spines['right'].set_visible(False)
        ax_1d.spines['top'].set_visible(False)

    for i_area, (area, arr, ax_2d, ax_1d) in enumerate(zip(D.areanames, accs_all, axes, axes_1d)):
        # Have to jig array around to get it bottom left
        im_arr = arr.T[::-1]
        ax_2d.imshow(im_arr, vmax=vmax, vmin=vmin, cmap='rainbow', aspect='auto', zorder=0)

        finishing_touches(ax_2d, arr, ax_1d, i_area == 0, vmin, vmax)

    plt.tight_layout()
    f.subplots_adjust(left=0.08, right=0.95)

    U.savefig(savefolder, 'allstab')

    # Now plot areas individually
    for i_area, (area, arr, ax1) in enumerate(zip(D.areanames, accs_all, axes)):
        # Have to jig array around to get it bottom left
        im_arr = arr.T[::-1]

        f = plt.figure(figsize=(width/D.numareas, height))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 0.01, 0.5], hspace=0)

        ax_2d, ax_1d = f.add_subplot(gs[0, 0]), f.add_subplot(gs[2, 0])

        vmax2 = np.around(np.nanmax(arr), 2)
        vmin2 = np.around(np.nanmin(arr), 2)
        ax_2d.imshow(im_arr, vmax=vmax2, vmin=vmin2, cmap='rainbow', aspect='auto', zorder=0)

        finishing_touches(ax_2d, arr, ax_1d, True, vmin2, vmax2)

        f.subplots_adjust(left=0.25, right=0.95)

        U.savefig(savefolder, area)
