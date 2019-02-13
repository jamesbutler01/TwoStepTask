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


def _makeandplotavgs(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg):
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
        sigmarker *= np.nanmax(avgs)
        sigmarker[sigmarker==0] = np.nan
        sigmarker += 0.05
        if show_leg:
            ax.plot(sigmarker, color='black', lw=2, label='Significant')
        else:
            ax.plot(sigmarker, color='black', lw=2)

    for i_trace, (avg_in, sem, label) in enumerate(zip(avgs, sems, leg_labels)):
        _plotpanel(ax, avg_in, sem, i_trace, ylab, label, show_leg)


def _makeandplotcounts(avg_in, sems_in, sig_in, ax, ylab, showsig, leg_labels, show_leg):
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



def GeneralPlot(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle):
    numrows = avgs.shape[1]
    showlegends = [True if i==0 else False for i in range(numrows)]
    
    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areas, avgs, sems, sigclusters)):

        f, axes = plt.subplots(numrows, 1, figsize=(width_regplot, height_regplot*numrows), sharex=True)

        if numrows > 1:
            for i, (label, leg_bool) in enumerate(zip(ytitles, showlegends)):
                _makeandplotavgs(avgs_area[i], sems_area[i], sig_area[i], axes[i], ylab=label, showsig=True, leg_labels=trace_names, show_leg=leg_bool)

        else:
                _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=ytitles, showsig=True, leg_labels=trace_names, show_leg=showlegends)

        title = maintitle+ f' ({area})'
        _finalplotadjustments(f, title)
        U.savefig(f'{savefolder}/{area}')


def GeneralAllAreas(avgs, sems, sigclusters, trace_names, savefolder, ytitles, maintitle):
    numrows = avgs.shape[1]
    if numrows > 1:
        raise Exception('Can only plot one row per area')

    showlegends = [True if i==0 else False for i in range(D.numareas)]

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True, sharey=True)

    for i_area, (area, avgs_area, sems_area, sig_area, axes) in enumerate(zip(D.areas, avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=showlegends)
        else:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=False)

    _finalplotadjustments(f, maintitle)
    U.savefig(f'{savefolder}/all')

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True)

    for i_area, (area, avgs_area, sems_area, sig_area, axes) in enumerate(zip(D.areas, avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=showlegends)
        else:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=False)

    _finalplotadjustments(f, maintitle)
    U.savefig(f'{savefolder}/all2')


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

def Rsa(rsa, area, suffix=''):
    # First remove diagonal ones
    np.fill_diagonal(rsa, np.nan)
    if suffix == '':
        labels = ['C1_0', 'C1_1', 'S2_0', 'S2_1', 'C2_2', 'C2_3', 'C2_4', 'C2_5']
    elif suffix == 'ext':
        labels = ['C1_0', 'C1_1', 'S2_0_com', 'S2_1_rare', 'S2_1_com', 'S2_0_rare',
                  'C2_2_com', 'C2_3_com', 'C2_4_com', 'C2_5_com', 'C2_2_rare',
                  'C2_3_rare', 'C2_4_rare', 'C2_5_rare']
    elif suffix == 'ext1':
        labels = ['fix0', 'C1_0', 'C1_1',
                  'trans_0_com', 'trans_1_rare', 'trans2_1_com', 'trans_0_rare',
                  'fix1',
                  'S2_0_com', 'S2_1_rare', 'S2_1_com', 'S2_0_rare',
                  'C2_2_com', 'C2_3_com', 'C2_4_com', 'C2_5_com', 'C2_2_rare',
                  'C2_3_rare', 'C2_4_rare', 'C2_5_rare', '2ndaryReinforcer']
    elif suffix == 'noC1':
        labels = ['trans_0_com', 'trans_1_rare', 'trans2_1_com', 'trans_0_rare',
                  'fix1',
                  'S2_0_com', 'S2_1_rare', 'S2_1_com', 'S2_0_rare',
                  'C2_2_com', 'C2_3_com', 'C2_4_com', 'C2_5_com', 'C2_2_rare',
                  'C2_3_rare', 'C2_4_rare', 'C2_5_rare', '2ndaryReinforcer']
    elif suffix == 'justC1':
        labels = ['C1_0', 'C1_1', 'S2_0_com', 'S2_1_rare', 'S2_1_com', 'S2_0_rare']
    elif suffix == 'justC2':
        labels = ['C2_2_com', 'C2_3_com', 'C2_4_com', 'C2_5_com', 'C2_2_rare',
                  'C2_3_rare', 'C2_4_rare', 'C2_5_rare']

    plt.figure(figsize=(9, 6))
    # im = plt.imshow(rsa, cmap=plt.get_cmap('bwr'), vmin=zmin, vmax=zmax)
    im = plt.imshow(rsa, cmap=plt.get_cmap('bwr'))
    plt.suptitle('RSA ' + area)
    plt.colorbar(im, label='Correlation')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{D.dir_savefig}rsa/{suffix}_{area}')
    plt.close('all')


def RsaCaC2(rsa, area, suffix, period):
    # First remove diagonal ones
    np.fill_diagonal(rsa, np.nan)

    if suffix == 'rewhist0':
        labels = ['Rew0', 'Rew1', 'Rew2', 'C2_0', 'C2_1', 'C2_2']
        title = f'Prev reward values for choice2 (depth 1) (Period: {period}) '
    elif suffix == 'rewhist1':
        labels = ['Rew0', 'Rew1', 'Rew2', 'C2_0_0', 'C2_0_1', 'C2_0_2', 'C2_1_0', 'C2_1_1', 'C2_1_2', 'C2_2_0', 'C2_2_1', 'C2_2_2']
        title = f'Prev reward values for choice2 (depth 2) (Period: {period}) '
    elif suffix == 'rewhist2':
        labels = [f'{a}_{b}_{c}' for a in range(3) for b in range(3) for c in range(3)]
        labels = [f'{a}_{b}' for a in range(3) for b in range(3)]
        title = f'Prev reward values for choice2 (depth 3) (Period: {period}) '
    elif suffix == 'rewhisttrans0':
        labels = ['C2_0_com', 'C2_1_com', 'C2_2_com', 'C2_0_rare', 'C2_1_rare', 'C2_2_rare']
        title = f'Prev reward values for choice2 (depth 1) (Period: {period}) '
    elif suffix == 'rewhisttrans1':
        labels = ['C2_0_0_com', 'C2_0_1_com', 'C2_0_2_com', 'C2_1_0_com', 'C2_1_1_com', 'C2_1_2_com', 'C2_2_0_com', 'C2_2_1_com', 'C2_2_2_com', 
                  'C2_0_0_rare', 'C2_0_1_rare', 'C2_0_2_rare', 'C2_1_0_rare', 'C2_1_1_rare', 'C2_1_2_rare', 'C2_2_0_rare', 'C2_2_1_rare', 'C2_2_2_rare']
        title = f'Prev reward values for choice2 by transition (depth 2) (Period: {period}) '
    elif suffix == 'rew_trans0':
        title = f'Reward by transition by choice at choice1 (depth 1) (Period: {period}) '
        labels = ['rew0_rep_com', 'rew0_rep_rare', 'rew0_switch_com', 'rew0_switch_rare',
                  'rew1_rep_com', 'rew1_rep_rare', 'rew1_switch_com', 'rew1_switch_rare',
                  'rew2_rep_com', 'rew2_rep_rare', 'rew2_switch_com', 'rew2_switch_rare']
    elif suffix == 'betas0':
        title = f'Betas by transition by choice at choice1 (depth 1) (Period: {period}) '
        labels = ('Repeat common', 'Repeat rare', 'Switch common', 'Switch rare')
    elif suffix == 'rewhist3':
        labels = ['0 0', '0 1', '0 2', '1 0', '1 1', '1 2', '2 0', '2 1', '2 2', ]
        title = f'Change in prev reward values for choice2 (depth 1) (Period: {period}) '
    elif suffix == 'rewhistchange0' or suffix == 'rewhistchange1':
        prevchoicevalues = np.empty((27, 3))
        rew_hist_diffs0 = np.empty((27, 1))
        rew_hist_diffs1 = np.empty((27, 2))
        a = -1
        b = -1
        c = -1
        for i in range(27):
            if i % 3 == 0:
                c += 1
            if i % 9 == 0:
                b += 1
                c = 0
            if i % 27 == 0:
                b = 0
                c = 0
                a += 1
            prevchoicevalues[i, 0] = a
            prevchoicevalues[i, 1] = b
            prevchoicevalues[i, 2] = c
            #prevchoicevalues[i, 3] = i % 3
            rew_hist_diffs0[i] = np.diff(prevchoicevalues[i][0:2])
            rew_hist_diffs1[i] = np.diff(prevchoicevalues[i])

        unique_rewhistchanges1 = np.array(np.unique(rew_hist_diffs1, axis=0), dtype=int)
        title = f'Change in prev reward values for choice2 (depth 2) (Period: {period}) '
        labels = unique_rewhistchanges1
    elif suffix[:6] == 'areas_':
        labels = D.areanames
        title = f'Similarity between areas for {period}{suffix}'


    plt.figure(figsize=(9, 6))
    im = plt.imshow(rsa, cmap=plt.get_cmap('bwr'))
    plt.suptitle(title + area)
    plt.colorbar(im, label='Correlation')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{D.dir_savefig}rsa/c2/{period}/{suffix}/{area}')
    plt.close('all')









