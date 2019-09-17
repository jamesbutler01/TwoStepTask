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


def analysearea(unique_func, num_conds, accs_out, sems_out, sigclusters, dists_all, peak_epoch, peak_cond, train_across_conds, upsample, i_area):

    data = ImportData.EntireArea(D.areas[i_area])
    dists = Maths.nans(dists_all[i_area].shape)

    # First we need to work out what the minimum number of samples across all cells
    for cell in range(data.n):
        td = data.behavdata[cell]

        x_data, masks = unique_func(td)

        # Remove any -1 entries in the x values
        masks = [removeinvalidentries(mask, x_data) for mask in masks]
        x_data = removeinvalidentries(x_data, x_data)

        labels = [np.unique(x_data[mask]) for mask in masks]
        numitems = [len(label) for label in labels]

        # Iterate through each condition
        for i_cond, (m_cond, label) in enumerate(zip(masks, labels)):

            # Get fewest number of trials for the different labels
            dists[i_cond, cell] = np.min([sum(x_data[m_cond] == lab) for lab in label])

    dists_all[i_area] = dists
    trials_per_label = np.array(np.nanmin(dists, axis=1), dtype=int)
    original_trials_per_label = np.copy(trials_per_label)

    # Any without enough trials will be excluded
    trials_per_label[trials_per_label < D.dec_minsamples] = D.dec_minsamples

    # Only allow cells with enough trials per label
    validcells = dists.T[:data.n] >= D.dec_minsamples

    # Need same amount of trials for each cond in this case
    if train_across_conds or True:

        # Correct min trials per label to all be the same
        trials_per_label[trials_per_label != np.min(trials_per_label)] = np.min(trials_per_label)

        # Redo mask to make sure same cells are excluded from every analysis
        new_valid_cells = np.nanmin(dists.T[:data.n], axis=1) > trials_per_label[0]
        for i in range(len(validcells.T)):
            validcells[:, i] = new_valid_cells[:len(validcells)]

    max_across_conds = np.max(trials_per_label)

    num_labels = len(label)

    # Now we know the maximum of trials to use with the decoder
    print(f'{D.areanames[i_area]} has {original_trials_per_label} min trials per condition, {int(np.mean(validcells)*100)}% cells included ({sum(validcells)})')

    def run_decoder(permtest=False):

        # Make holding arrays for analysis
        accs_perms = np.empty((num_conds, D.numtrialepochs, D.num_timepoints, D.dec_numiters))

        for i_cellselection in range(D.dec_numiters):

            y_all = np.empty((D.numtrialepochs, num_conds, data.n, max_across_conds * np.max(numitems), D.num_timepoints))
            x_all = np.empty((D.numtrialepochs, num_conds, data.n, max_across_conds * np.max(numitems)))
            y_all.fill(np.nan)
            x_all.fill(np.nan)

            # Loop to collect all the data from the different cells that had enough trials
            for cell, cell_validity in enumerate(validcells):

                td = data.behavdata[cell]

                x_data, m_conds = unique_func(td)

                # For each epoch
                for i_epoch, epoch in enumerate(D.epochs):

                    y = data.generatenormalisedepoch(cell, epoch)

                    # For each condition get the data
                    for i_cond, (m_cond, min_trials, valid_cell) in enumerate(zip(m_conds, trials_per_label, cell_validity)):

                        # Skip cells without enough trials for a certain condition
                        if not valid_cell:
                            continue

                        y_masked = y[m_cond]
                        x_masked = x_data[m_cond]

                        # Remove any potential -1's in the x data
                        y_masked = removeinvalidentries(y_masked, x_masked)
                        x_masked = removeinvalidentries(x_masked, x_masked)

                        # Shuffle
                        if permtest:
                            np.random.shuffle(x_masked)

                        # For each label
                        for i_x, x_val in enumerate(np.unique(x_masked)):

                            # Get the y data just for this condition and this x-label
                            y_for_xval = y_masked[x_masked == x_val]

                            # Choose random subset of y data
                            ind_inc_trials = np.random.choice(len(y_for_xval), min_trials, replace=False)

                            # Store x and y data in holding arrays
                            y_all[i_epoch, i_cond, cell, min_trials * (i_x):min_trials * (i_x + 1), :] = y_for_xval[ind_inc_trials]
                            x_all[i_epoch, i_cond, cell, min_trials * (i_x):min_trials * (i_x + 1)] = i_x

            # Now we have all our data with equal number of samples for each label and each cell, we can run the decoder
            scores = Maths.decode_across_epochs(x_all, y_all, peak_epoch, peak_cond, train_across_conds, permtest)

            accs_perms[:, :, :, i_cellselection] = scores

            if not permtest: print(f'{D.areanames[i_area]} Progress: {i_cellselection+1}/{D.dec_numiters}')

        return accs_perms

    accs_allperms = run_decoder(False)

    accs_out[i_area, 0] = np.mean(accs_allperms, axis=3)

    null_dist = Maths.calc_sig_length_null_dist()

    # Also do t-test at every time point
    for i_cond in range(num_conds):

        for i_epoch in range(D.numtrialepochs):

            pvals = np.empty(D.num_timepoints)

            for i_ti in range(D.num_timepoints):
                pvals[i_ti] = Maths.ttest_1samp(accs_allperms[i_cond, i_epoch, i_ti], 1/num_labels)

            # Swap NaNs to 1's
            pvals[np.isnan(pvals)] = 1

            # See which runs are significantly long
            sigcluster = np.zeros(D.num_timepoints)

            pvals_bool = pvals < 0.05

            for i in range(D.num_timepoints):

                onecluster, clusterlength = Maths.findlongestrun(pvals_bool)

                if sum(null_dist > clusterlength) > D.sigthreshold_onetailed or clusterlength < 1:

                    # If no significance then stop looking
                    break

                else:

                    # Significant, so add it to marker
                    sigcluster += onecluster

                    # Erase points that gave significance (set different conds to same value so they are not sig. anymore)
                    pvals_bool[sigcluster == 1] = 1

            pvals[~np.array(sigcluster, dtype=bool)] = 1

            accs_out[i_area, 1, i_cond, i_epoch] = pvals

    std = np.nanstd(accs_allperms, axis=3)
    sem = std / np.sqrt(D.dec_numiters)

    sems_out[i_area, 0] = sem

    print(f'{D.areanames[i_area]} Finished decoding')

    if D.dec_do_perms:

        accs_permtest = np.empty((num_conds, D.numperms))

        for i in range(D.numperms):

            one_perm = run_decoder(True)

            accs_permtest[:, i] = np.mean(one_perm[:, 0, 0, :], axis=1)

            print(f'{D.areanames[i_area]} Permutation progress: {i+1}/{D.numperms}')

        # Get CI
        accs_sorted = np.sort(accs_permtest, axis=1)  # Sort permutations
        sigthreshold = D.sigthreshold_onetailed
        if sigthreshold == 0: sigthreshold = 1  # -0 indexing doesn't work
        accs_ci = accs_sorted[:, -sigthreshold]  # Take the 95th highest permutation
        if num_conds > D.numtrialepochs:
            raise Exception('You cannot have more conds than epochs as no where to store the perm data')

        sigclusters[i_area] = accs_ci


class Run:

    def __new__(cls, function_to_run, trace_names, peak_epoch=None, peak_cond=None, train_across_conds=False, areas=range(D.numareas), maintitle='Decoder', savefolder='dec/temp', upsample=None):

        if upsample is None:
            upsample = D.dec_upsample

        timer = TimeFunction.Timer()
        num_conds = len(trace_names)
        accs_all, sems_all, _ = Utils.getarrs(num_conds, 2)
        sigclusters = np.empty((D.numareas, num_conds))
        dists_all = np.zeros((D.numareas, num_conds, 300))

        if D.domultiproc:

            m = MyManager()
            m.start()

            sigclusters = m.np_zeros(sigclusters.shape)
            accs_all = m.np_zeros(accs_all.shape)
            sems_all = m.np_zeros(sems_all.shape)
            dists_all = m.np_zeros(dists_all.shape)

            pool = Pool(5)
            func = partial(analysearea, function_to_run, num_conds, accs_all, sems_all, sigclusters, dists_all, peak_epoch, peak_cond, train_across_conds, upsample)
            pool.map(func, areas)
            pool.close()

            accs_all = np.array(accs_all)

        else:

            dists_all = np.zeros((D.numareas, num_conds, 300))

            for i_area in areas:
                analysearea(function_to_run, num_conds, accs_all, sems_all, sigclusters, dists_all, peak_epoch, peak_cond, train_across_conds, upsample, i_area)

        # Plot details
        ylabel = 'Accuracy (%)'
        trials_per_label = np.array(np.nanmin(dists_all[0], axis=1), dtype=int)
        trials_per_label[trials_per_label < D.dec_minsamples] = D.dec_minsamples
        trace_names = [t+f' ({n})' for t, n in zip(trace_names, trials_per_label)]

        Plot.GeneralAllAreas(accs_all[:, 0:1], sems_all[:, 0:1], sigclusters[:, 0:1], trace_names, savefolder, ylabel, maintitle, scale_sig=False, show_sig=False)

        Plot.DecoderSignificant_AllAreas(accs_all, sems_all, sigclusters, trace_names, savefolder, ylabel, maintitle, scale_sig=False, show_sig=False)

        if D.dec_do_perms:
            Plot.DecoderPermSig_AllAreas(accs_all, sems_all, sigclusters, trace_names, savefolder, ylabel, maintitle, scale_sig=False, show_sig=True)

        Plot.PlotDist(dists_all, savefolder)

        print(timer.elapsedtime())

        return accs_all, sems_all, sigclusters, trace_names, savefolder, ylabel, maintitle




