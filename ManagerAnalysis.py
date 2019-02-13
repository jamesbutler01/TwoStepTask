import ImportData
import Details as D
import numpy as np
from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers
import Maths
import TimeFunction
import Plot
import Utils

class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

if False:
    import ImportData
    cell = 1
    area = D.areas[0]
    data = ImportData.EntireArea(area)
    trialdata = data.behavdata[cell]
    x_type = 'r'
    num_conds=4
    out_betas = np.zeros((1, num_conds, D.numtrialepochs, data.n, D.num_timepoints))
    data_inc = 't0 high'
    fr='norm'
    firingperiod = 't1'


class Run:

    def __init__(self, function_to_run, run_sig_test, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, plotfunc, doavgs=True):
        timer = TimeFunction.Timer()

        m = MyManager()
        m.start()
        all_avgs, all_sem, sigclusters = Utils.getarrs(num_conds, num_rows)

        for i_area, area in enumerate(D.areas):
            data = ImportData.EntireArea(area)

            counter = m.np_zeros(data.n)
            betas = m.np_zeros((num_rows, num_conds, D.numtrialepochs, data.n, D.num_timepoints))

            pool = Pool(5)
            func = partial(function_to_run, counter, betas, data)
            run_list = range(data.n)
            pool.map(func, run_list)
            pool.close()

            betas = np.array(betas)

            for i_row, row_epoch in enumerate(betas):
                for i_choice, cond_epoch in enumerate(row_epoch):  # Now trialepochs x cell x timepoints
                    for i_trial, trial_epoch in enumerate(cond_epoch):  # Now cell by timepoints
                        if doavgs:
                            avg = np.nanmean(trial_epoch, axis=0)
                            sem = Maths.sem(trial_epoch)
                            all_avgs[i_area, i_row, i_choice, i_trial] = avg
                            all_sem[i_area, i_row, i_choice, i_trial] = sem
                        else:
                            for i_t, t in enumerate(trial_epoch.T):
                                all_avgs[i_area, i_row, i_choice, i_trial, i_t] = int(len(t[t<0.05])/len(t) * 100)  # Num cells significant
                            all_sem[i_area, i_row, i_choice, i_trial] = 0


                if run_sig_test:
                    sigclusters[i_area, i_row] = Maths.permtest(row_epoch)

        print(timer.elapsedtime())

        plotfunc(all_avgs, all_sem, sigclusters, trace_names, savefolder, ytitles, maintitle)

