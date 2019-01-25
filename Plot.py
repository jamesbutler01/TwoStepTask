import numpy as np
import matplotlib.pyplot as plt

# Static plot parameters
import Details as D
buffer = 10  # Gap between each task event
numpoints = D.num_timepoints * D.numtrialepochs + buffer * D.numtrialepochs
binlength = D.num_timepoints + buffer
xtickpos = [int((i_epoch + 0.1) * binlength) for i_epoch in range(D.numtrialepochs)]
width_regplot = 15
height_regplot = 2.75


def Deep15(betasallareas_avg, betasallareas_sem, sigclusters, saveprefix):
    if saveprefix == 'r':
        datatypelabel = 'reward'
    else:
        datatypelabel = 'q-value'

    num_x = 4
    num_timepoints = D.smooth_outputlength
    epochs = (D.sc_madefixation, D.sc_choice1on, D.sc_choice1made, D.sc_transition, D.sc_choice2on, D.sc_choice2made,
              D.sc_secondaryreinforceron)
    epochnames = (
    'Fixation', '\nChoice 1 on', 'Choice 1 made', '\nTransition Revealed', 'Choice 2 on', '\nChoice 2 made',
    'Secondary Reinforcer')
    numepochspertrial = len(epochs)

    # Now stich together into one big trace for each area
    betasallareas_avg = np.swapaxes(betasallareas_avg, 0, 2)
    betasallareas_sem = np.swapaxes(betasallareas_sem, 0, 2)
    sigclusters = np.swapaxes(sigclusters, 0, 2)

    buffer = 10  # Gap between each task event
    numpoints = num_timepoints * numepochspertrial + buffer * (numepochspertrial - 1)
    binlength = num_timepoints+buffer

    # Static plot parameters
    xlabels = [name for name in epochnames]
    xtickpos = [int((i_epoch + 0.1) * binlength) for i_epoch in range(numepochspertrial)]
    labels = ('Repeat common', 'Repeat rare', 'Switch common', 'Switch rare')
    labels_twolines = ('Repeat\ncommon', 'Repeat\nrare', 'Switch\ncommon', 'Switch\nrare')

    def makeandplotavgs(ax, i_trialoffset, tr_offset, ylab, sigmarkers, betas_avgs, betas_sems, leg):
        avgs = np.empty((num_x, numpoints))
        avgs.fill(np.nan)
        sems = np.copy(avgs)
        sigmarker = np.zeros(numpoints)

        for i_epoch in range(numepochspertrial):
            start = i_epoch * binlength
            fin = i_epoch * binlength + num_timepoints

            sigmarker[start:fin] = sigmarkers[i_abs, i_epoch + tr_offset]

            for i_beta in range(num_x):
                avgs[i_beta, start:fin] = betas_avgs[i_abs, i_epoch + tr_offset, i_beta]
                sems[i_beta, start:fin] = betas_sems[i_abs, i_epoch + tr_offset, i_beta]

        sigmarker *= np.nanmax(avgs)
        sigmarker[sigmarker==0] = np.nan
        sigmarker += 0.05
        if leg:
            ax.plot(sigmarker, color='black', lw=2, label='Significant')
        else:
            ax.plot(sigmarker, color='black', lw=2)

        for i_beta, (avg, sem, label) in enumerate(zip(avgs, sems, labels)):
            plotpanel(ax, avg, sem, i_trialoffset, ylab, leg, i_beta)

        # # Also plot grand mean
        # grandmean = np.mean(avgs, axis=0)
        # ax.plot(grandmean, color='black', ls='--', lw=2)


    def plotpanel(ax, avg, sem, pos, ylab, leg, colornumber):
        if leg:
            ax.plot(avg, label=labels[colornumber], color=f'C{colornumber}', lw=3)
        else:
            ax.plot(avg, color=f'C{colornumber}', lw=3)
        ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.3, color=f'C{colornumber}')
        ax.set_xlim(0, numpoints)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        if pos != 3:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        else:
            ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(ylab, fontsize=11)
        plt.xticks(xtickpos, xlabels, fontsize=11)
        plt.yticks(fontsize=13)

    def finalplotadjustments(f):
        plt.suptitle(f'Betas for t+0 {datatypelabel} ({area}){abstitle}', x=0.25, fontsize=18)
        f.legend(loc='upper right', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tick_params(axis='both', which='major', labelsize=13)


    # 2x1 plot
    for i_abs, (abslabel , abstitle) in enumerate(zip(('mean', 'abs'), ('', ' (absolute)'))):
        for i_area, (area, betasonearea_avg, betasonearea_sem, clustersonearea) in enumerate(zip(D.areas, betasallareas_avg, betasallareas_sem, sigclusters)):

            f, axes = plt.subplots(2, 1, figsize=(15,6), sharex=True, sharey=True)

            for i_trialoffset, (tr_offset, ax, ylab, leg) in enumerate(zip((0, numepochspertrial), axes, ('t+1', 't+2'), (True, False))):
                makeandplotavgs(ax, i_trialoffset, tr_offset, ylab, clustersonearea, betasonearea_avg, betasonearea_sem, leg)

            finalplotadjustments(f)
            if abslabel == 'mean':
                plt.savefig(f'{D.dir_savefig}{saveprefix}_betas_{area}_{abslabel}')
            plt.close('all')

    # 1x5 plot
    for i_abs, (abslabel , abstitle) in enumerate(zip(('mean', 'abs'), ('', ' (absolute)'))):
        for i_area, (area, betasonearea_avg, betasonearea_sem, clustersonearea) in enumerate(zip(D.areas, betasallareas_avg, betasallareas_sem, sigclusters)):

            f, axes = plt.subplots(6, 1, figsize=(15,11), sharex=True)

            makeandplotavgs(axes[0], 0, 0, 't+1 activity \n(split by t+1)', clustersonearea, betasonearea_avg, betasonearea_sem, True)
            makeandplotavgs(axes[1], 1, numepochspertrial, 't+2 activity \n(split by t+1)', clustersonearea, betasonearea_avg, betasonearea_sem, False)

            for i_trialoffset, (tr_offset, ax, ylab) in enumerate(
                    zip([numepochspertrial * (i + 2) for i in range(4)], axes[2:], ['t+2 activity \nfor t+1 =\n '+label+'\n (split by t+2)' for label in labels])):
                makeandplotavgs(ax, i_trialoffset, tr_offset, ylab, clustersonearea, betasonearea_avg, betasonearea_sem, False)


            finalplotadjustments(f)
            if abslabel == 'mean':
                plt.savefig(f'{D.dir_savefig}{saveprefix}_betas_2_{area}_{abslabel}')
            plt.close('all')


def TwoAheadReg(avgs_orig, sems_orig, sigclusters_orig, saveprefix, p):
    if saveprefix == 'r':
        datatypelabel = 'reward'
    else:
        datatypelabel = 'q-value'

    avgs = np.copy(avgs_orig)
    sems = np.copy(sems_orig)
    sigclusters = np.copy(sigclusters_orig)

    # Static plot parameters
    buffer = 10  # Gap between each task event
    numpoints = p.num_timepoints * p.numtrialepochs + buffer * p.numtrialepochs
    binlength = p.num_timepoints+buffer
    xtickpos = [int((i_epoch + 0.1) * binlength) for i_epoch in range(p.numtrialepochs)]

    def makeandplotavgs(avgs_choice, sems_choice, clusters_choice, ax, ylab):
        avgs = np.empty((p.num_trans_hists, numpoints))
        avgs.fill(np.nan)
        sems = np.copy(avgs)
        sigmarker = np.zeros(numpoints)

        for i_epoch in range(p.numtrialepochs):
            start = i_epoch * binlength
            fin = i_epoch * binlength + p.num_timepoints

            sigmarker[start:fin] = clusters_choice[i_epoch]

            for i_trans in range(p.num_trans_hists):
                avgs[i_trans, start:fin] = avgs_choice[i_trans, i_epoch]
                sems[i_trans, start:fin] = sems_choice[i_trans, i_epoch]

        sigmarker *= np.nanmax(avgs)
        sigmarker[sigmarker==0] = np.nan
        sigmarker += 0.05
        if ylab == p.names_choice_hists[0]:
            ax.plot(sigmarker, color='black', lw=2, label='Significant')
        else:
            ax.plot(sigmarker, color='black', lw=2)

        for i_trans, (avg, sem, label) in enumerate(zip(avgs, sems, p.names_trans_hists)):
            plotpanel(ax, avg, sem, ylab, i_trans)

    def plotpanel(ax, avg, sem, ylab, colornumber):
        if ylab == p.names_choice_hists[0]:
            ax.plot(avg, label=p.names_trans_hists[colornumber], color=f'C{colornumber}', lw=3)
        else:
            ax.plot(avg, color=f'C{colornumber}', lw=3)
        ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.3, color=f'C{colornumber}')
        ax.set_xlim(0, numpoints)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        if colornumber != 3:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        else:
            ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(ylab, fontsize=11)
        plt.xticks(xtickpos, p.names_epochs, fontsize=11)
        plt.yticks(fontsize=13)

    def finalplotadjustments(f, area):
        plt.suptitle(f't+2 Betas for t+0 {datatypelabel} ({area})', x=0.25, fontsize=18)
        f.legend(loc='upper right', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tick_params(axis='both', which='major', labelsize=13)

    # 1x4 plot
    for i_area, (area, avgs_area, sems_area, clusters_area) in enumerate(zip(D.areas, avgs, sems, sigclusters)):

        f, axes = plt.subplots(p.num_choice_hists, 1, figsize=(15,2.75*p.num_choice_hists), sharex=True)

        for avgs_choice, sems_choice, clusters_choice, ax, ylab in zip(avgs_area, sems_area, clusters_area, axes, p.names_choice_hists):
            makeandplotavgs(avgs_choice, sems_choice, clusters_choice, ax, ylab)


        finalplotadjustments(f, area)
        plt.savefig(f'{D.dir_savefig}{p.savefolder}/{saveprefix}_{area}')
        plt.close('all')


def RegOneAhead(avgs_orig, sems_orig, sigclusters_orig, saveprefix, p):
    if saveprefix == 'r':
        datatypelabel = 'reward'
    else:
        datatypelabel = 'q-value'

    avgs = np.copy(avgs_orig)
    sems = np.copy(sems_orig)
    sigclusters = np.copy(sigclusters_orig)

    # Static plot parameters
    buffer = 10  # Gap between each task event
    numpoints = p.num_timepoints * p.numtrialepochs + buffer * p.numtrialepochs
    binlength = p.num_timepoints + buffer
    xtickpos = [int((i_epoch + 0.1) * binlength) for i_epoch in range(p.numtrialepochs)]

    def makeandplotavgs(avgs_choice, sems_choice, clusters_choice, ax, i_choice):
        avgs = np.empty((p.num_trans_hists, numpoints))
        avgs.fill(np.nan)
        sems = np.copy(avgs)
        sigmarker = np.zeros(numpoints)

        for i_epoch in range(p.numtrialepochs):
            start = i_epoch * binlength
            fin = i_epoch * binlength + p.num_timepoints

            sigmarker[start:fin] = clusters_choice[i_epoch]

            for i_trans in range(p.num_trans_hists):
                avgs[i_trans, start:fin] = avgs_choice[i_trans, i_epoch]
                sems[i_trans, start:fin] = sems_choice[i_trans, i_epoch]

        sigmarker *= np.nanmax(avgs)
        sigmarker[sigmarker == 0] = np.nan
        sigmarker += 0.05
#        if ylab == p.names_choice_hists[0]:
  #          ax.plot(sigmarker, color='black', lw=2, label='Significant')
 #       else:
   #         ax.plot(sigmarker, color='black', lw=2)

        for i_trans, (avg, sem) in enumerate(zip(avgs, sems)):
            plotpanel(ax, avg, sem, i_trans, i_choice)

    def plotpanel(ax, avg, sem, i_trans, i_choice):
        num = i_choice * 2 + i_trans
        ax.plot(avg, label=f'{p.names[num]}', color=f'C{num}', lw=3)
        ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.3, color=f'C{num}')
        ax.set_xlim(0, numpoints)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        if num != 3:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        else:
            ax.xaxis.set_ticks_position('bottom')
        #ax.set_ylabel(ylab, fontsize=11)
        plt.xticks(xtickpos, p.names_epochs, fontsize=11)
        plt.yticks(fontsize=13)

    def finalplotadjustments(f, area):
        plt.suptitle(f't+1 Betas for t+0 {datatypelabel} ({area})', x=0.25, fontsize=18)
        f.legend(loc='upper right', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tick_params(axis='both', which='major', labelsize=13)

    # 1x4 plot
    for i_area, (area, avgs_area, sems_area, clusters_area) in enumerate(zip(D.areas, avgs, sems, sigclusters)):

        f, axes = plt.subplots(1, 1, figsize=(15,4), sharex=True)

        for i_choice, (avgs_choice, sems_choice, clusters_choice) in enumerate(zip(avgs_area, sems_area, clusters_area)):
            makeandplotavgs(avgs_choice, sems_choice, clusters_choice, axes, i_choice)

        finalplotadjustments(f, area)
        plt.savefig(f'{D.dir_savefig}reg/{p.savefolder}/{saveprefix}_{area}')
        plt.close('all')


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


def RegTwoAheadAxB(avgs_orig, sems_orig, avgs_orig2, sems_orig2, trace_names, savefolder):
    avgs = np.copy(avgs_orig)
    sems = np.copy(sems_orig)

    numplots = 2

    for i_area, (area, avgs_area, sems_area, avgs_area2, sems_area2) in enumerate(zip(D.areas, avgs, sems, avgs_orig2, sems_orig2)):

        f, axes = plt.subplots(numplots, 1, figsize=(width_regplot,height_regplot*numplots), sharex=True)

        _makeandplotavgs(avgs_area, sems_area, None, axes[0], ylab='t+0 reward betas \n(A(r)_x_B)', showsig=False, leg_labels=trace_names, show_leg=True)
        _makeandplotavgs(avgs_area2, sems_area2, None, axes[1], ylab='t+1 reward betas \n(A_x(r)_B)', showsig=False, leg_labels=trace_names, show_leg=False)
        title = f't+0 and t+1 betas during t+1, split by t+2 behaviour ({area})'
        _finalplotadjustments(f, title)
        plt.savefig(f'{D.dir_savefig}reg/{savefolder}/{area}')
        plt.close('all')


def RegTwoAheadAAR(avgs_orig, sems_orig, sigclusters, trace_names, savefolder):

    numplots = 1

    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areas, avgs_orig, sems_orig, sigclusters)):

        f, axes = plt.subplots(numplots, 1, figsize=(width_regplot, height_regplot*numplots), sharex=True)

        _makeandplotavgs(avgs_area, sems_area, sig_area, axes, ylab='Average reward betas', showsig=True, leg_labels=trace_names, show_leg=True)
        title = f'Betas for t+0 reward in t+2 ({area})'
        _finalplotadjustments(f, title)
        plt.savefig(f'{D.dir_savefig}reg/{savefolder}/{area}')
        plt.close('all')


def RegPrevC2(avgs_orig, sems_orig, sigclusters, trace_names, savefolder):

    numplots = 1

    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areas, avgs_orig, sems_orig, sigclusters)):

        f, axes = plt.subplots(numplots, 1, figsize=(width_regplot, height_regplot*numplots), sharex=True)

        _makeandplotavgs(avgs_area, sems_area, sig_area, axes, ylab='Average reward betas', showsig=True, leg_labels=trace_names, show_leg=True)
        title = f'Betas for t+0 and t+1 reward in t+2 ({area})'
        _finalplotadjustments(f, title)
        plt.savefig(f'{D.dir_savefig}reg/{savefolder}/{area}')
        plt.close('all')

def RegAAR(avgs_orig, sems_orig, sigclusters, trace_names, savefolder, titlesuffix, fileprefix, titlesuffix2):

    numplots = 1

    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areas, avgs_orig, sems_orig, sigclusters)):

        f, axes = plt.subplots(numplots, 1, figsize=(width_regplot, height_regplot*numplots), sharex=True)

        _makeandplotavgs(avgs_area, sems_area, sig_area, axes, ylab='Average FR', showsig=True, leg_labels=trace_names, show_leg=True)
        title = f'{titlesuffix2} FR depending on t+2 choice ({titlesuffix}) ({area})'
        _finalplotadjustments(f, title)
        plt.savefig(f'{D.dir_savefig}reg/{savefolder}/{titlesuffix}/{fileprefix}_{area}')
        plt.close('all')


def RegZeroAhead(avgs_orig, sems_orig, sigclusters, trace_names, savefolder):

    numplots = 2

    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areas, avgs_orig, sems_orig, sigclusters)):

        f, axes = plt.subplots(numplots, 1, figsize=(width_regplot, height_regplot*numplots), sharex=True)

        _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes[0], ylab='Average reward betas', showsig=True, leg_labels=trace_names, show_leg=True)
        _makeandplotavgs(avgs_area[1], sems_area[1], sig_area[1], axes[1], ylab='Average reward betas', showsig=True, leg_labels=trace_names, show_leg=False)
        title = f't+1 betas during t+2, split by t+1 behaviour ({area})'
        _finalplotadjustments(f, title)
        plt.savefig(f'{D.dir_savefig}reg/{savefolder}/{area}')
        plt.close('all')

def RegPrevC2(avgs_orig, sems_orig, sigclusters, trace_names, savefolder):

    numplots = avgs_orig.shape[1]

    for i_area, (area, avgs_area, sems_area, sig_area) in enumerate(zip(D.areas, avgs_orig, sems_orig, sigclusters)):

        f, axes = plt.subplots(numplots, 1, figsize=(width_regplot, height_regplot*numplots), sharex=True)

        for i, (label, leg_bool) in enumerate(zip(('t0', 't1', 'best t', 'rel best t', 'rel t1'), (True, False, False, False, False))):
            _makeandplotavgs(avgs_area[i], sems_area[i], sig_area[i], axes[i], ylab=f'{label} Reward betas', showsig=True, leg_labels=trace_names, show_leg=leg_bool)
        title = f't+0 and t+1 betas during t+2, split by transition types ({area})'
        _finalplotadjustments(f, title)
        plt.savefig(f'{D.dir_savefig}reg/{savefolder}/{area}')
        plt.close('all')

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
        D.savefig(f'{savefolder}/{area}')


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
    D.savefig(f'{savefolder}/all')

    f, axes_all = plt.subplots(D.numareas, 1, figsize=(width_regplot, height_regplot*D.numareas), sharex=True)

    for i_area, (area, avgs_area, sems_area, sig_area, axes) in enumerate(zip(D.areas, avgs, sems, sigclusters, axes_all)):
        if i_area == 0:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=showlegends)
        else:
            _makeandplotavgs(avgs_area[0], sems_area[0], sig_area[0], axes, ylab=D.areanames[i_area], showsig=True, leg_labels=trace_names, show_leg=False)

    _finalplotadjustments(f, maintitle)
    D.savefig(f'{savefolder}/all2')


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

def OverviewVis(all_avgs, all_sems, sigclusters, saveprefix, trialnumber, datatypelabel2, savefolder):
    if saveprefix == 'r':
        datatypelabel = 'reward'
    else:
        datatypelabel = 'q-value'

    num_x = 4
    num_timepoints = D.smooth_outputlength

    # Now stich together into one big trace for each area
    all_avgs = np.swapaxes(all_avgs, 0, 2)
    all_sems = np.swapaxes(all_sems, 0, 2)
    sigclusters = np.swapaxes(sigclusters, 0, 2)

    buffer = 10  # Gap between each task event
    numpoints = num_timepoints * D.numtrialepochs + buffer * (D.numtrialepochs - 1)
    binlength = num_timepoints+buffer

    # Static plot parameters
    xlabels = [name for name in D.names_epochs]
    xtickpos = [int((i_epoch + 0.1) * binlength) for i_epoch in range(D.numtrialepochs)]
    labels = ('Repeat common', 'Repeat rare', 'Switch common', 'Switch rare')
    labels_twolines = ('Repeat\ncommon', 'Repeat\nrare', 'Switch\ncommon', 'Switch\nrare')

    def makeandplotavgs(ax, i_trialoffset, tr_offset, ylab, sigmarkers, betas_avgs, betas_sems, leg):
        avgs = np.empty((num_x, numpoints))
        avgs.fill(np.nan)
        sems = np.copy(avgs)
        sigmarker = np.zeros(numpoints)

        for i_epoch in range(D.numtrialepochs):
            start = i_epoch * binlength
            fin = i_epoch * binlength + num_timepoints

            sigmarker[start:fin] = sigmarkers[i_abs, i_epoch + tr_offset]

            for i_beta in range(num_x):
                avgs[i_beta, start:fin] = betas_avgs[i_abs, i_epoch + tr_offset, i_beta]
                sems[i_beta, start:fin] = betas_sems[i_abs, i_epoch + tr_offset, i_beta]

        sigmarker *= np.nanmax(avgs)
        sigmarker[sigmarker==0] = np.nan
        sigmarker += 0.05
        if leg:
            ax.plot(sigmarker, color='black', lw=2, label='Significant')
        else:
            ax.plot(sigmarker, color='black', lw=2)

        for i_beta, (avg, sem, label) in enumerate(zip(avgs, sems, labels)):
            plotpanel(ax, avg, sem, i_trialoffset, ylab, leg, i_beta)


    def plotpanel(ax, avg, sem, pos, ylab, leg, colornumber):
        if leg:
            ax.plot(avg, label=labels[colornumber], color=f'C{colornumber}', lw=3)
        else:
            ax.plot(avg, color=f'C{colornumber}', lw=3)
        ax.fill_between(range(len(avg)), avg - sem, avg + sem, alpha=0.15, color=f'C{colornumber}')
        ax.set_xlim(0, numpoints)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        if pos != 3:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        else:
            ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(ylab, fontsize=11)
        plt.xticks(xtickpos, xlabels, fontsize=11)
        plt.yticks(fontsize=13)

    def finalplotadjustments(f):
        plt.suptitle(f' {datatypelabel2} for t+0 {datatypelabel} split by t+{trialnumber} behaviour ({area}){abstitle}', x=0.25, fontsize=18)
        f.legend(loc='upper right', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tick_params(axis='both', which='major', labelsize=13)


    # 2x1 plot
    for i_abs, (abslabel , abstitle) in enumerate(zip(('mean', 'abs'), ('', ' (absolute)'))):
        for i_area, (area, betasonearea_avg, betasonearea_sem, clustersonearea) in enumerate(zip(D.areas, all_avgs, all_sems, sigclusters)):

            f, axes = plt.subplots(3, 1, figsize=(15,9), sharex=True, sharey=True)

            for i_trialoffset, (tr_offset, ax, ylab, leg) in enumerate(zip((0, D.numtrialepochs, D.numtrialepochs*2), axes, ('t+0', 't+1', 't+2'), (True, False, False))):
                makeandplotavgs(ax, i_trialoffset, tr_offset, ylab, clustersonearea, betasonearea_avg, betasonearea_sem, leg)

            finalplotadjustments(f)
            if abslabel == 'mean':
                plt.savefig(f'{D.dir_savefig}/overview/{savefolder}/{saveprefix}/{trialnumber}_{area}_{abslabel}')
            plt.close('all')


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









