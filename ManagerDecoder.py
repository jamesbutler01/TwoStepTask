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


def run_decoder(n_epochs, num_conds, data, num_labels, num_samples_per_label, validcells, epochs, train_epochs, unique_func, stability, peak_epoch, peak_tp, train_across_conds, accs_perms, do_permtest, i_cellselection):

        y_all = Maths.nans((n_epochs, num_conds, data.n, num_labels, num_samples_per_label, D.n_timepoints))
        x_all = Maths.nans((n_epochs, num_conds, data.n, num_labels, num_samples_per_label))

        # Loop to collect all the data from the different cells that had enough trials
        for cell, cell_validity in enumerate(validcells):

            td = data.behavdata[cell]

            x_datas, m_conds = unique_func(td)

            # Allow users to only specify one x data
            if np.array(x_datas).ndim == 1:
                x_datas = [x_datas] * len(m_conds)

            # For each epoch
            for i_epoch, epoch in enumerate(epochs):

                # For each condition get the data
                for i_cond, (m_cond, x_data, valid_cell) in enumerate(zip(m_conds, x_datas, cell_validity)):

                    # If we're training across epochs, then use training epoch
                    if stability and num_conds == 2 and i_cond == 0:
                        y = data.generate_epoch_norm(cell, train_epochs[i_epoch])
                    else:
                        y = data.generate_epoch_norm(cell, epoch)

                    # Skip cells without enough trials for a certain condition
                    if not valid_cell:
                        continue

                    y_masked = y[m_cond]
                    x_masked = x_data[m_cond]

                    # Remove any potential -1's in the x data
                    y_masked = Maths.dec_removeinvalidentries(y_masked, x_masked)
                    x_masked = Maths.dec_removeinvalidentries(x_masked, x_masked)

                    # Shuffle
                    if do_permtest:
                        np.random.shuffle(x_masked)

                    # For each label
                    for i_x, x_val in enumerate(np.unique(x_masked)):

                        # Get the y data just for this condition and this x-label
                        y_for_xval = y_masked[x_masked == x_val]

                        # Choose random subset of y data
                        ind_inc_trials = np.random.choice(len(y_for_xval), num_samples_per_label, replace=False)

                        # Store x and y data in holding arrays
                        y_all[i_epoch, i_cond, cell, i_x, :num_samples_per_label, :] = y_for_xval[ind_inc_trials]
                        x_all[i_epoch, i_cond, cell, i_x, :num_samples_per_label] = i_x

        # Now we have all our data with equal number of samples for each label and each cell, we can run the decoder
        if stability:
            scores = Maths.decode_stability(x_all, y_all)
        else:
            scores = Maths.decode_across_epochs(x_all, y_all, peak_epoch, peak_tp, train_across_conds, do_permtest)

        accs_perms[..., i_cellselection] = scores

        if not do_permtest: print(f'Progress: {i_cellselection + 1}/{D.dec_numiters}')

        return np.array(accs_perms)



def analysearea(unique_func, num_conds, accs_out, sems_out, sigclusters, dists_all, epochs, stability, peak_epoch, peak_tp, train_across_conds, train_epochs, i_area):

    data = ImportData.EntireArea(D.areas[i_area])
    n_epochs = len(epochs)

    num_labels, num_samples_per_label, validcells, dists_all[i_area] = Maths.calculate_min_trials_per_area(data, unique_func, dists_all, i_area)

    # # Make holding arrays for analysis
    if not stability:
        accs_allperms = np.empty((num_conds, n_epochs, D.n_timepoints, D.dec_numiters))
    else:
        n_pnts = n_epochs * (D.n_timepoints + 1)
        accs_allperms = Maths.nans((n_pnts, n_pnts, D.dec_numiters))


    if D.domultiproc:

        m = MyManager()
        m.start()

        accs_allperms = m.np_zeros(accs_allperms.shape)

        pool = Pool(D.n_cores)
        func = partial(run_decoder, n_epochs, num_conds, data, num_labels, num_samples_per_label, validcells, epochs, train_epochs, unique_func, stability, peak_epoch, peak_tp, train_across_conds, accs_allperms, False)
        pool.map(func, range(D.dec_numiters))
        pool.close()

        accs_allperms = np.array(accs_allperms)

    else:

        for i_cellselection in range(D.dec_numiters):

                accs_allperms = run_decoder(n_epochs, num_conds, data, num_labels, num_samples_per_label, validcells, epochs, train_epochs, unique_func, stability, peak_epoch, peak_tp, train_across_conds, accs_allperms, False, i_cellselection)

    if not stability:

        # Get average and SEM
        accs_out[i_area, 0] = np.mean(accs_allperms, axis=3)
        std = np.nanstd(accs_allperms, axis=3)
        sem = std / np.sqrt(D.dec_numiters)
        sems_out[i_area, 0] = sem

        # T-test at every time point
        accs_out[i_area, 1] = Maths.ttest_every_timepoint(accs_allperms, num_labels)

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

    else:

        accs_out[i_area] = np.mean(accs_allperms, axis=2)


class Run:

    def __new__(cls, function_to_run, trace_names=None, epochs=D.epochs, epochnames=D.names_epochs, stability=False, train_epochs=None, peak_epoch=None, peak_tp=None, train_across_conds=False, maintitle='Decoder', savefolder='dec/temp', upsample=None):
        # stability = Test stability of decoding over time and return 2D matrix of accuracies
        # train_epochs = For stability, train and test on two different epochs and conds
        # peak_epoch = train on this epoch and test on all others
        # peak_tp = train on this tp and test on all others
        # train_across_conds = collapse across conds to train decoder and then test within each cond's held out data

        timer = TimeFunction.Timer()
        n_epochs = len(epochs)

        n = D.numareas

        if stability:
            n_pnts = n_epochs * (D.n_timepoints + 1)
            accs_all = Maths.nans((n, n_pnts, n_pnts))
            sems_all, sigclusters = np.copy(accs_all), np.copy(accs_all)
            num_conds = 1
            if train_epochs is not None:
                num_conds+=1
            dists_all = np.zeros((n, num_conds, 300))
        else:
            num_conds = len(trace_names)
            if peak_epoch is not None:
                num_conds += 1
                trace_names.insert(0, 'Train data')
            accs_all, sems_all, _ = Utils.getarrs(num_conds, 2)
            sigclusters = np.empty((n, num_conds))
            dists_all = np.zeros((n, num_conds, 300))

        for i_area in range(n):

            print(f'Analysing {D.areanames[i_area]}...')

            analysearea(function_to_run, num_conds, accs_all, sems_all, sigclusters, dists_all, epochs, stability, peak_epoch, peak_tp, train_across_conds, train_epochs, i_area)

        if not stability:
            ylabel = 'Accuracy (%)'
            trials_per_label = np.array(np.nanmin(dists_all[0], axis=1), dtype=int)
            trials_per_label[trials_per_label < D.dec_minsamples] = D.dec_minsamples
            trace_names = [t+f' ({n})' for t, n in zip(trace_names, trials_per_label)]

            Plot.GeneralAllAreas(accs_all[:, 0:1], sems_all[:, 0:1], sigclusters[:, 0:1], trace_names, savefolder, maintitle, scale_sig=False, show_sig=False)

            Plot.DecoderSignificant_AllAreas(accs_all, sems_all, sigclusters, trace_names, savefolder, maintitle, peak_epoch=peak_epoch, peak_tp=peak_tp, scale_sig=False, show_sig=False)

            if D.dec_do_perms:
                Plot.DecoderPermSig_AllAreas(accs_all, sems_all, sigclusters, trace_names, savefolder, ylabel, maintitle, scale_sig=False, show_sig=True)

            Plot.PlotDist(dists_all, savefolder)

        else:

            Plot.PlotDecStab(accs_all, savefolder, epochnames)

        print(timer.elapsedtime())

        return accs_all, sems_all, sigclusters, trace_names, savefolder, maintitle




