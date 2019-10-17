import types
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


def run_decoder(data, unique_func, p, accs_perms, do_permtest, i_cellselection):

    if p.peak_epoch is not None:

        # Need to train decoder first
        x_train, y_train = Maths.collect_trials_for_decoding(data, unique_func, p, True, do_permtest)
        cond_to_train = 0
        _, dec_to_test = Maths.decode_epoch_traintestsplit(x_train, y_train, p.peak_epoch, cond_to_train, y_train.shape[-1], peak_tp=p.peak_tp)

    else:

        dec_to_test = None

    x_all, y_all = Maths.collect_trials_for_decoding(data, unique_func, p, False, do_permtest)

    # Now we have all our data with equal number of samples for each label and each cell, we can run the decoder
    if p.stability:
        scores = Maths.decode_stability(x_all, y_all)
    else:
        scores = Maths.decode_across_epochs(x_all, y_all, dec_to_test, p.train_across_conds, do_permtest)

    accs_perms[..., i_cellselection] = scores

    if not do_permtest: print(f'Progress: {i_cellselection + 1}/{D.dec_numiters}')

    return np.array(accs_perms)



def analysearea(unique_func, accs_out, sems_out, sigclusters, dists_all, p, i_area):

    data = ImportData.EntireArea(D.areas[i_area])

    p.num_labels, p.num_samples_per_label, p.validcells, dists_all[i_area] = Maths.calculate_min_trials_per_area(data, unique_func, dists_all, i_area)

    # If training on one epoch, then calculate train trial numbers seperately
    if p.peak_epoch is not None:

        p.num_labels_train, p.num_samples_per_label_train, _, _ = Maths.calculate_min_trials_per_area(data, unique_func, dists_all, i_area, train_epoch_only=True)

        # And include all cells so have same features in all decoders
        p.validcells = np.ones(dists_all[i_area].shape, dtype=bool)

    # Make holding arrays for analysis
    if not p.stability:
        accs_allperms = np.empty((p.num_conds, p.n_epochs, D.n_timepoints, D.dec_numiters))
    else:
        n_pnts = p.n_epochs * (D.n_timepoints + 1)
        accs_allperms = Maths.nans((n_pnts, n_pnts, D.dec_numiters))

    if D.domultiproc:

        m = MyManager()
        m.start()

        accs_allperms = m.np_zeros(accs_allperms.shape)

        pool = Pool(D.n_cores)
        func = partial(run_decoder, data, unique_func, p, accs_allperms, False)
        pool.map(func, range(D.dec_numiters))
        pool.close()

        accs_allperms = np.array(accs_allperms)

    else:

        for i_cellselection in range(D.dec_numiters):

                accs_allperms = run_decoder(data, unique_func, p, accs_allperms, False, i_cellselection)

    if not p.stability:

        # Get average and SEM
        accs_out[i_area, 0] = np.mean(accs_allperms, axis=3)
        std = np.nanstd(accs_allperms, axis=3)
        sem = std / np.sqrt(D.dec_numiters)
        sems_out[i_area, 0] = sem

        # T-test at every time point
        accs_out[i_area, 1] = Maths.ttest_every_timepoint(accs_allperms, p.num_labels)

        print(f'{D.areanames[i_area]} Finished decoding')

        if D.dec_do_perms:

            accs_permtest = np.empty((p.num_conds, D.numperms))

            for i in range(D.numperms):

                one_perm = run_decoder(True)

                accs_permtest[:, i] = np.mean(one_perm[:, 0, 0, :], axis=1)

                print(f'{D.areanames[i_area]} Permutation progress: {i+1}/{D.numperms}')

            # Get CI
            accs_sorted = np.sort(accs_permtest, axis=1)  # Sort permutations
            sigthreshold = D.sigthreshold_onetailed
            if sigthreshold == 0: sigthreshold = 1  # -0 indexing doesn't work
            accs_ci = accs_sorted[:, -sigthreshold]  # Take the 95th highest permutation
            if p.num_conds > D.numtrialepochs:
                raise Exception('You cannot have more conds than epochs as no where to store the perm data')

            sigclusters[i_area] = accs_ci

    else:

        accs_out[i_area] = np.mean(accs_allperms, axis=2)


class Run:

    def __new__(cls, function_to_run, trace_names=None, epochs=D.epochs, epochnames=D.names_epochs, stability=False, train_epochs=None, peak_epoch=None, peak_tp=None, peak_halfwidth=None, train_across_conds=False, maintitle='Decoder', savefolder='dec/temp'):
        # stability = Test stability of decoding over time and return 2D matrix of accuracies
        # train_epochs = For stability, train and test on two different epochs and conds
        # peak_epoch, peak_tp, peak_halfwidth = train on this epoch, at this timepoint, at this halfwidth, and test on all others
            # For this condition, you must supply an extra condition and mask, to be used as the peak training data, in index 0
        # train_across_conds = collapse across conds to train decoder and then test within each cond's held out data

        timer = TimeFunction.Timer()

        # Log all parameters in single object to pass around functions
        p = types.SimpleNamespace()
        p.epochs = epochs
        p.n_epochs = len(epochs)
        p.n = D.numareas
        p.stability = stability
        p.train_epochs = train_epochs
        p.peak_epoch = peak_epoch
        p.peak_tp = peak_tp
        p.peak_halfwidth = peak_halfwidth if peak_halfwidth is not None else D.smooth_window_halfwidth
        p.train_across_conds = train_across_conds

        if p.stability and p.num_conds > 2:
            raise Exception('Stability only works with one or two conds')

        if stability:
            p.n_pnts = p.n_epochs * (D.n_timepoints + 1)
            accs_all = Maths.nans((p.n, p.n_pnts, p.n_pnts))
            sems_all, sigclusters = np.copy(accs_all), np.copy(accs_all)
            p.num_conds = 1
            if train_epochs is not None:
                p.num_conds+=1
        else:
            p.num_conds = len(trace_names)
            if peak_epoch is not None:
                p.num_conds += 1
                trace_names.insert(0, 'Train data')
            accs_all, sems_all, _ = Utils.getarrs(p.num_conds, 2, p.n_epochs)
            sigclusters = np.empty((p.n, p.num_conds))

        dists_all = np.zeros((p.n, p.num_conds, 300))

        for i_area in range(p.n):

            print(f'Analysing {D.areanames[i_area]}...')

            analysearea(function_to_run, accs_all, sems_all, sigclusters, dists_all, p, i_area)

        if not stability:
            ylabel = 'Accuracy (%)'
            trials_per_label = np.array(np.nanmin(dists_all[0], axis=1), dtype=int)
            trials_per_label[trials_per_label < D.dec_minsamples] = D.dec_minsamples
            trace_names = [t+f' ({n})' for t, n in zip(trace_names, trials_per_label)]

            Plot.GeneralAllAreas(accs_all[:, 0:1], sems_all[:, 0:1], sigclusters[:, 0:1], trace_names, savefolder, maintitle, scale_sig=False, names_epochs=epochnames, show_sig=False)

            Plot.DecoderSignificant_AllAreas(accs_all, sems_all, sigclusters, trace_names, savefolder, maintitle, peak_epoch=peak_epoch, peak_tp=peak_tp, names_epochs=epochnames, scale_sig=False, show_sig=False)

            if D.dec_do_perms:
                Plot.DecoderPermSig_AllAreas(accs_all, sems_all, sigclusters, trace_names, savefolder, ylabel, maintitle, scale_sig=False, show_sig=True)

            Plot.PlotDist(dists_all, savefolder)

        else:

            Plot.PlotDecStab(accs_all, savefolder, epochnames)

        print(timer.elapsedtime())

        return accs_all, sems_all, sigclusters, trace_names, savefolder, maintitle




