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
    import Dec_c2_dir_AA_AB as dec
    unique_func = dec.createmasks
    i_area = 0
    import ImportData
    i_area = 1
    data = ImportData.EntireArea(D.areas[i_area])
    dists_all = np.empty((D.numareas, data.n))


def removeinvalidentries(arr_in, check_arr):
    return arr_in[check_arr != -1]


def analysearea(unique_func, num_conds, accs_all, sems_all, run_sig_test, sigclusters, decoder, minsamples, dists_all, i_area):

    data = ImportData.EntireArea(D.areas[i_area])
    counts = np.empty((num_conds, data.n), dtype=int)
    validcells = np.ones(data.n, bool)
    min_trials = 999
    for cell in range(data.n):
        td = data.behavdata[cell]

        x_data, masks = unique_func(td)

        masks = [removeinvalidentries(mask, x_data) for mask in masks]
        x_data = removeinvalidentries(x_data, x_data)

        labels = [np.unique(x_data[mask]) for mask in masks]
        numitems = [len(label) for label in labels]

        cell_min = 999
        for i, (mask, label) in enumerate(zip(masks, labels)):
            this_min = np.min([sum(x_data[mask] == lab) for lab in label])

            if this_min < cell_min:
                cell_min = this_min

            counts[i, cell] = this_min

        dists_all[i_area, cell] = cell_min

        if cell_min < min_trials:
            min_trials = cell_min

        if cell_min < minsamples:
            validcells[cell] = False

    print(f'Area: {D.areas[i_area]} has {min_trials} minimum samples, {int(np.mean(validcells)*100)}% cells included ({sum(validcells)})')

    y_all = np.empty((D.numtrialepochs, num_conds, sum(validcells), min_trials * np.max(numitems), D.num_timepoints))
    x_all = np.empty((D.numtrialepochs, num_conds, sum(validcells), min_trials * np.max(numitems)))

    y_all.fill(np.nan)
    x_all.fill(np.nan)

    # If we have included all cells then we don't need to iterate over different sub populations
    numiters = D.dec_numiters_cellselection
    if int(np.mean(validcells) * 100) == 100:
        numiters = 1

    accs_perms = np.empty((num_conds, D.numtrialepochs, D.num_timepoints, numiters))

    # Do multiple loops with different subsets of cells
    for i_cellselection in range(numiters):

        for i_cell, cell in enumerate(np.where(validcells == 1)[0]):

            td = data.behavdata[cell]

            x_data, masks = unique_func(td)

            for i_epoch, epoch in enumerate(D.epochs):

                y = data.generatenormalisedepoch(cell, epoch)

                for i, mask in enumerate(masks):
                    y_masked = y[mask]
                    x_masked = x_data[mask]

                    y_masked = removeinvalidentries(y_masked, x_masked)
                    x_masked = removeinvalidentries(x_masked, x_masked)

                    for i_x, x_val in enumerate(np.unique(x_masked)):
                        y_for_xval = y_masked[x_masked == x_val]

                        ind_inc_trials = np.random.choice(len(y_for_xval), min_trials, replace=False)

                        y_all[i_epoch, i, i_cell, min_trials * (i_x):min_trials * (i_x + 1), :] = y_for_xval[
                            ind_inc_trials]
                        x_all[i_epoch, i, i_cell, min_trials * (i_x):min_trials * (i_x + 1)] = i_x

        avgs, sem_traintestsplit = Maths.decode_across_epochs(x_all, y_all, decoder)

        accs_perms[:, :, :, i_cellselection] = np.mean(avgs, axis=3)

        if i_area == 0:
            print(f'Progress: {i_cellselection+1}/{numiters}')

    accs = np.mean(accs_perms, axis=3)
    accs_all[i_area, 0] = accs

    std = np.nanstd(accs_perms, axis=3)
    sem_cellselection = std / np.sqrt(numiters)

    # Add the SEM for sampling different subsets of cells to the SEM from different train/test splits of the data
    sems_all[i_area, 0] = sem_traintestsplit + sem_cellselection

    if run_sig_test:

        if D.dec_numiters_traintestsplit < 4:
            raise Exception('Not enough permutations for states (D.dec_numiters_traintestsplit)')

        # Do sig test between different traintestsplit runs of the decoder (?)
        sigdata = np.swapaxes(avgs, -1, -2)
        sigclusters[i_area, 0] = Maths.permtest(sigdata, multiproc=False)


class Run:

    def __new__(cls, function_to_run, run_sig_test, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, decoder, minsamples=4):
        timer = TimeFunction.Timer()

        if num_rows > 1:
            raise Exception('Multiple rows not implemented yet')

        m = MyManager()
        m.start()

        accs_all, sems_all, sigclusters = Utils.getarrs(num_conds, num_rows)
        accs_all = m.np_zeros(accs_all.shape)
        sems_all = m.np_zeros(sems_all.shape)
        sigclusters = m.np_zeros(sigclusters.shape)
        dists_all = m.np_zeros((D.numareas, 300))

        pool = Pool(5)
        func = partial(analysearea, function_to_run, num_conds, accs_all, sems_all, run_sig_test, sigclusters, decoder, minsamples, dists_all)
        run_list = range(len(D.areas))
        pool.map(func, run_list)
        pool.close()

        accs_all = np.array(accs_all)

        print(timer.elapsedtime())

        Plot.GeneralPlot(accs_all, sems_all, sigclusters, trace_names, savefolder, ytitles, maintitle)

        Plot.GeneralAllAreas(accs_all, sems_all, sigclusters, trace_names, savefolder, ytitles, maintitle)

        Plot.PlotDist(dists_all, savefolder)
        return accs_all, sems_all, sigclusters, trace_names, savefolder, ytitles, maintitle




