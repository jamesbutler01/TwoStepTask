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
import ShiftTrials
Shifter = ShiftTrials.Shift(2)

np.seterr(all='raise')

import importlib

importlib.reload(Plot)

savefolder = 'AAR'
trace_name = (('A(2)_AR(x)_A', 'A(2)_AR(x)_B'))
num_cond = len(trace_name)
maintitle = f't2 betas depending on t+2 choice'
num_rows = 1

class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

if False:
    cell = 1
    area = D.areas[0]
    data = ImportData.EntireArea(area)
    trialdata = data.behavdata[cell]
    x_type = 'r'
    num_conds=4
    out_betas = np.zeros((num_conds, D.numtrialepochs, data.n, D.num_timepoints))
    data_inc = 't0 high'
    fr='norm'
    firingperiod = 't1'


def analysecell(counter, out_betas, data, firingperiod, data_inc, cell):
    counter[cell] = 1
    if int(sum(counter)) % 20 == 0:
        print(f'{firingperiod} {data_inc} {int(sum(counter))}/{data.n}')

    td = data.behavdata[cell]

    for i_epoch, epoch in enumerate(D.epochs):
        mask_aar = D.get_A_AR_trials(td)
        mask_aar = Shifter.shift(mask_aar, 1)
        mask_t0high = np.array(Shifter.shift(td.rew_coll, 0), dtype=bool)
        mask_t1low = np.invert(np.array(Shifter.shift(td.rew_coll, 1), dtype=bool))
        mask_t1high = np.array(Shifter.shift(td.rew_coll, 1), dtype=bool)
        mask_t0comm = np.array((Shifter.shift(td.transition, 0)==1), dtype=bool)

        static_mask = mask_aar & mask_t0high

        # t+2 outcome
        t0_c1g = D.shifttrial(td.c1given, t=0, max_t=2)
        t2_c1c = D.shifttrial(td.c1chosen, t=2, max_t=2)
        t2choseB = (t0_c1g != t2_c1c)

        y = data.generatenormalisedepoch(cell, epoch)
        x = Shifter.shift(td.rew_coll, 1)

        if firingperiod == 't1':
            y = D.shifttrial(y, 1, 2)
        elif firingperiod == 't2':
            y = D.shifttrial(y, 2, 2)
        elif firingperiod == 't0':
            y = D.shifttrial(y, 0, 2)

        for t2choice in (0, 1):
            mask = static_mask & (t2choseB == t2choice)
            x_const = sm.add_constant(x[mask])
            out_betas[t2choice, i_epoch, cell] = Maths.regression(x_const, y[mask])

if __name__ == "__main__":
    timer = TimeFunction.Timer()

    m = MyManager()
    m.start()
    suffix='t2'
    all_avgs = np.zeros((D.numareas, num_rows, num_cond, D.numtrialepochs, D.num_timepoints))
    all_sem = np.zeros((D.numareas, num_rows, num_cond, D.numtrialepochs, D.num_timepoints))
    sigclusters = np.zeros((D.numareas, num_rows, D.numtrialepochs, D.num_timepoints))

    for i_area, area in enumerate(D.areas):
        data = ImportData.EntireArea(area)

        counter = m.np_zeros(data.n)
        betas = m.np_zeros((num_cond, D.numtrialepochs, data.n, D.num_timepoints))

        pool = Pool(5)
        func = partial(analysecell, counter, betas, data, suffix, '')
        run_list = range(data.n)
        pool.map(func, run_list)  # Now put run_list in the second argument of local_func
        pool.close()

        betas = np.array(betas)

        for i_choice, choice_epoch in enumerate(betas):  # trialepochs x trans x cell x timepoints
            for i_trial, trial_epoch in enumerate(choice_epoch):  # trans x cell x timepoints
                avg = np.nanmean(trial_epoch, axis=0)
                sem = Maths.sem(trial_epoch)

                all_avgs[i_area, 0, i_choice, i_trial] = avg
                all_sem[i_area, 0, i_choice, i_trial] = sem

        sigclusters[i_area] = Maths.permtest(betas)

    print(timer.elapsedtime())

    Plot.GeneralPlot(all_avgs, all_sem, sigclusters, trace_name, f'reg/{savefolder}/{suffix}/reg/', 'Avg Betas', maintitle)

