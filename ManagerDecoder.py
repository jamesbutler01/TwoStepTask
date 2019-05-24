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


def removeinvalidentries(arr_in, check_arr):
    return arr_in[check_arr != -1]


def analysearea(unique_func, num_conds, accs_all, sems_all, run_sig_test, sigclusters, decoder, minsamples, dists_all, i_area):

    data = ImportData.EntireArea(D.areas[i_area])
    validcells = np.ones(data.n, bool)
    min_across_cells = 999

    # First we need to work out what the minimum number of samples across all cells
    for cell in range(data.n):
        td = data.behavdata[cell]

        x_data, masks = unique_func(td)

        # Remove any -1 entries in the x values
        masks = [removeinvalidentries(mask, x_data) for mask in masks]
        x_data = removeinvalidentries(x_data, x_data)

        labels = [np.unique(x_data[mask]) for mask in masks]
        numitems = [len(label) for label in labels]

        cell_min = 999

        # Iterate through each condition
        for i_mask, (mask, label) in enumerate(zip(masks, labels)):

            # Get fewest number of trials for the different labels
            this_min = np.min([sum(x_data[mask] == lab) for lab in label])

            # If lower than current lowest store it
            if this_min < cell_min:
                cell_min = this_min
        
        # Store data to plot
        dists_all[i_area, cell] = cell_min
        
        # If this cell lower n than all previous cells then update min_across_cells
        if cell_min < min_across_cells:
            min_across_cells = cell_min
        
        # If not above specified minimum then disable this cell for the future analysis
        if cell_min < minsamples:
            validcells[cell] = False
    
    # Now we know the maximum of trials to use with the decoder
    print(f'Area: {D.areas[i_area]} has {min_across_cells} minimum samples, {int(np.mean(validcells)*100)}% cells included ({sum(validcells)})')
  
    # If we have included all cells then we don't need to iterate over different sub populations
    numiters = D.dec_numiters_cellselection
    if False not in validcells:
        numiters = 1
    
    # Make holding arrays for analysis
    accs_perms = np.empty((num_conds, D.numtrialepochs, D.num_timepoints, numiters))
    y_all = np.empty((D.numtrialepochs, num_conds, sum(validcells), min_across_cells * np.max(numitems), D.num_timepoints))
    x_all = np.empty((D.numtrialepochs, num_conds, sum(validcells), min_across_cells * np.max(numitems)))
    y_all.fill(np.nan)
    x_all.fill(np.nan)

    # Do multiple loops with different subsets of cells
    for i_cellselection in range(numiters):
        
        # Loop to collect all the data from the different cells that had enough trials
        for i_cell, cell in enumerate(np.where(validcells == 1)[0]):

            td = data.behavdata[cell]

            x_data, masks = unique_func(td)
            
            # For each epoch
            for i_epoch, epoch in enumerate(D.epochs):

                y = data.generatenormalisedepoch(cell, epoch)

                # For each condition get the data
                for i_mask, mask in enumerate(masks):
                    y_masked = y[mask]
                    x_masked = x_data[mask]

                    # Remove any potential -1's in the x data
                    y_masked = removeinvalidentries(y_masked, x_masked)
                    x_masked = removeinvalidentries(x_masked, x_masked)

                    # For each label
                    for i_x, x_val in enumerate(np.unique(x_masked)):

                        # Get the y data just for this condition and this x-label
                        y_for_xval = y_masked[x_masked == x_val]

                        # Choose random subset of y data
                        ind_inc_trials = np.random.choice(len(y_for_xval), min_across_cells, replace=False)

                        # Store x and y data in holding arrays
                        y_all[i_epoch, i_mask, i_cell, min_across_cells * (i_x):min_across_cells * (i_x + 1), :] = y_for_xval[
                            ind_inc_trials]
                        x_all[i_epoch, i_mask, i_cell, min_across_cells * (i_x):min_across_cells * (i_x + 1)] = i_x

        # Now we have all our data with equal number of samples for each label and each cell, we can run the decoder
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




