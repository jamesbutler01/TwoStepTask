import ImportData
import Details as D
import numpy as np
from functools import partial
from multiprocessing import Process, Pool
import statsmodels.api as sm
import multiprocessing.managers
import Maths
import TimeFunction
import Plot
np.seterr(all='raise')

num_traces = 4
totalnumepochs = D.numtrialepochs * 3  # t+0, t+1, t+2

if False:
    area = D.areas[0]
    data = ImportData.EntireArea(area)
    counter = np.zeros(data.n)
    out_betas = np.zeros((totalnumepochs, data.n, num_traces, D.num_timepoints))
    x_type = 'r'
    trialnumber = 0
    func_loc = Maths.cod
    firing_norm = 'norm'
    cell = 15
    offset = 0
    epoch = D.epochs[0]
    i_epoch=0

class MyManager(multiprocessing.managers.BaseManager):
    pass

MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def analysecell(counter, out_betas, data, x_type, trialnumber, func_loc, firing_norm, cell):
    counter[cell] = 1
    if int(sum(counter)) % 20 == 0:
        print(f'{firing_norm} {x_type} {func_loc} {int(sum(counter))}/{data.n}')
    #for cell in range(data.n):
    trialdata = data.behavdata[cell]

    for i_epoch, epoch in enumerate(D.epochs):

        if firing_norm == 'raw':
            zero_fr = data.generateaverageepoch(cell, epoch)
        elif firing_norm == 'norm':
            zero_fr = data.generatenormalisedepoch(cell, epoch)

        zero_repeat = trialdata.repeatc2atc1
        zero_transition = trialdata.transition

        if x_type == 'r_coll':
            zero_rew = trialdata.rew_coll
        elif x_type == 'r':
            zero_rew = trialdata.rewgiven
        elif x_type == 'q':
            zero_rew = trialdata.q_chosen2

        def shiftarralongone(arr1, arr2, arr3):
            return arr1[1:], arr2[1:], arr3[1:]

        # Get data from different trials in the future
        one_fr, one_transition, one_repeat = shiftarralongone(zero_fr, zero_transition, zero_repeat)
        two_fr, two_transition, two_repeat = shiftarralongone(one_fr, one_transition, one_repeat)

        def splitbycond(repeat, transition, arr):
            rep_common = arr[(repeat == 1) & (transition == 1)]
            rep_rare = arr[(repeat == 1) & (transition == 2)]
            switch_common = arr[(repeat == 0) & (transition == 1)]
            switch_rare = arr[(repeat == 0) & (transition == 2)]

            return rep_common, rep_rare, switch_common, switch_rare

        def regression(x, y, repeat, transition, offset):
            x_vals_constant = sm.add_constant(x.T)

            x_rep_common, x_rep_rare, x_switch_common, x_switch_rare = splitbycond(repeat, transition, x_vals_constant)
            y_rep_common, y_rep_rare, y_switch_common, y_switch_rare = splitbycond(repeat, transition, y)

            out_betas[i_epoch + offset, cell, 0] = func_loc(x_rep_common, y_rep_common)
            out_betas[i_epoch + offset, cell, 1] = func_loc(x_rep_rare, y_rep_rare)
            out_betas[i_epoch + offset, cell, 2] = func_loc(x_switch_common, y_switch_common)
            out_betas[i_epoch + offset, cell, 3] = func_loc(x_switch_rare, y_switch_rare)

        if trialnumber == 0:
            regression(zero_rew, zero_fr, zero_repeat, zero_transition, 0)
            regression(zero_rew[:-1], one_fr, zero_repeat[:-1], zero_transition[:-1], D.numtrialepochs)
            regression(zero_rew[:-2], two_fr, zero_repeat[:-2], zero_transition[:-2], D.numtrialepochs * 2)
        elif trialnumber == 1:
            regression(zero_rew[:-1], zero_fr[:-1], one_repeat, one_transition, 0)
            regression(zero_rew[:-1], one_fr, one_repeat, one_transition, D.numtrialepochs)
            regression(zero_rew[:-2], two_fr, one_repeat[:-1], one_transition[:-1], D.numtrialepochs * 2)
        elif trialnumber == 2:
            regression(zero_rew[:-2], zero_fr[:-2], two_repeat, two_transition, 0)
            regression(zero_rew[:-2], one_fr[:-1], two_repeat, two_transition, D.numtrialepochs)
            regression(zero_rew[:-2], two_fr, two_repeat, two_transition, D.numtrialepochs * 2)


if __name__ == "__main__":
    timer = TimeFunction.Timer()
    prefix = 'r'
    funcs = (Maths.cod, Maths.regression)
    funclabels = ('CoD', 'Regression')
    firing_norms = ('norm', 'raw')
    for funclabel, func_loc in zip(funclabels, funcs):
        for prefix in ('r', 'r_coll'):
            for firing_norm in firing_norms:
                for trialnumber in range(3):
                    allavgs = np.zeros((totalnumepochs, 2, D.numareas, num_traces, D.num_timepoints))
                    allsems = np.zeros((totalnumepochs, 2, D.numareas, num_traces, D.num_timepoints))
                    sigclusters = np.zeros((totalnumepochs, 2, D.numareas, D.num_timepoints))
                    permutations = np.zeros((totalnumepochs, 2, D.numareas, D.numperms))

                    m = MyManager()
                    m.start()

                    for i_area, area in enumerate(D.areas):
                        data = ImportData.EntireArea(area)

                        counter = m.np_zeros(data.n)
                        betas = m.np_zeros((totalnumepochs, data.n, num_traces, D.num_timepoints))

                        pool = Pool(5)
                        func = partial(analysecell, counter, betas, data, prefix, trialnumber, func_loc, firing_norm)
                        run_list = range(50)
                        run_list = range(data.n)
                        pool.map(func, run_list)  # Now put run_list in the second argument of local_func
                        pool.close()

                        betas = np.array(betas)
                        betas = np.swapaxes(betas, 1, 2)  # Swap dimensions to index by beta

                        for i_epoch, beta_epoch in enumerate(betas):
                            for i_beta, beta in enumerate(beta_epoch):
                                avg = np.mean(beta, axis=0)
                                sem = Maths.sem(beta)
                                avg_abs = np.mean(np.abs(beta), axis=0)
                                sem_abs = Maths.sem(np.abs(beta))

                                allavgs[i_epoch, 0, i_area, i_beta] = avg
                                allsems[i_epoch, 0, i_area, i_beta] = sem

                                allavgs[i_epoch, 1, i_area, i_beta] = avg_abs
                                allsems[i_epoch, 1, i_area, i_beta] = sem_abs

                            # Permutation test for significance at every timepoint
                            if np.std(beta_epoch) == 0.001:
                                lengths = m.np_zeros(D.numperms)
                                pool = Pool(5)
                                func = partial(Maths.doperms, beta_epoch, lengths)
                                run_list = range(D.numperms)
                                pool.map(func, run_list)  # Now put run_list in the second argument of local_func
                                pool.close()

                                lengths = np.array(lengths)
                                permutations[i_epoch, 0, i_area] = np.sort(lengths)

                                # Actual data
                                sigcluster, clusterlength = Maths.findsignificancecluster(beta_epoch)
                                if len(np.where(lengths > clusterlength)[0]) > D.sigthreshold or funclabel=='CoD':
                                    sigcluster.fill(0)  # Erase cluster if it wasn't significant

                                sigclusters[i_epoch, 0, i_area] = sigcluster

                    print(timer.elapsedtime())

                    Plot.OverviewVis(allavgs, allsems, sigclusters, prefix, trialnumber, funclabel, f'{firing_norm}/{funclabel}')
