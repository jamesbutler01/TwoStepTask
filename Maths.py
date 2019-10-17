import scipy
import numpy as np
import scipy.special as special  # F-distribution lookup table
from numba import jit
import Details as D
from multiprocessing import Process, Pool
import multiprocessing.managers
from functools import partial
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

class MyManager(multiprocessing.managers.BaseManager):
    pass

MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def normalise(arr):
    arr -= np.nanmin(arr)
    arr /= np.nanmax(arr)
    return arr

def sem(arr):
    if arr.ndim == 1:
        std = np.nanstd(arr)
        std /= np.sqrt(np.shape(arr))
        std=std[0]
    else:
        std = np.nanstd(arr, axis=0)
        std /= np.sqrt(np.shape(arr[1]))
    return std


#@jit(nopython=True)
def jit_sq(arr):
    return np.square(arr)

def cod(x_val, y_val):
    out = np.empty(y_val.shape[1])
    x = x_val[:,1]
    for i, y in enumerate(y_val.T):
        if np.std(y) < 0.000001:
            out[i] = 0
        else:
            #r = _copd_helper(x_val, y)
            r = jit_sq(corrcoef(get_pred_vals(x_val, y), y))
            #rsquared = rsqared_adj(r, y.shape[0])
            out[i] = r
    return out

def rsqared_adj(r, n):
    return 1-(1-r) * ( (n - 1) / (n - 2 - 1) )

def _copd_helper(x, y):
    pred = get_pred_vals(x, y)
    residSSEfull = get_sse(y, pred)
    residSSEred =  np.var(y) * y.shape[0]
    return (residSSEfull - residSSEred) / residSSEred


def corrcoef(x, y):
    c = _jit_corrcoef_helper(x, y)
    np.clip(c.real, -1, 1, out=c.real)
    return c[0, 1]

#@jit(nopython=True)
def _jit_corrcoef_helper(x, y):
    c = np.cov(x, y, True)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev.reshape((-1, 1))
    c /= stddev.reshape((1, -1))
    return c

#@jit(nopython=True)
def fr(x_val, y_val):
    return np.mean(y_val)

#@jit(nopython=True)
def jit_regression(x_val, y_val):
    model = np.linalg.lstsq(x_val, y_val, rcond=None)[0]
    return model[1]

#@jit(nopython=True)
def get_pred_vals(x_val, y_val):
    model = np.linalg.lstsq(x_val, y_val, rcond=None)[0]
    return x_val[:,1]*model[1] + model[0]

#@jit(nopython=True)
def get_sse(x1, x2):
    return np.sum(np.square(x1-x2))

#@jit(nopython=True)
def jit_regression_2reg(x_val, y_val):
    model = np.linalg.lstsq(x_val, y_val)[0]
    return model[1], model[2]

def regression(x_val, y_val):
    if len(np.unique(x_val)) < 2:
        return np.nan
    if np.std(y_val) == 0:
        return 0
    if len(np.shape(x_val)) == 1:
        x_val = sm.add_constant(x_val)
    return jit_regression(x_val, y_val)


def permtest(arr, multiproc=True):
    out = np.empty((arr.shape[1], arr.shape[3]))
    arr = np.swapaxes(arr, 0, 1)  # To index by trial epoch

    if multiproc:
        m = MyManager()
        m.start()

    for i_epoch, epoch in enumerate(arr):

        # Get distribution
        epoch_buff = np.copy(epoch)

        if multiproc:

            dist = m.np_zeros(D.numperms)
            pool = Pool(D.n_cores)
            func = partial(doperms, epoch_buff, dist)
            run_list = range(D.numperms)
            pool.map(func, run_list)
            pool.close()
            dist = np.sort(np.array(dist))

        else:

            dist = np.empty(D.numperms)
            for i in range(D.numperms):
                doperms(epoch_buff, dist, i)

        # Now check observed data
        sigcluster = np.zeros(epoch.shape[-1])

        for i in range(epoch.shape[-1]):

            onecluster, clusterlength = findsignificancecluster(epoch)

            if sum(dist > clusterlength) > D.sigthreshold_onetailed or clusterlength < 1:
                # If no significance then stop looking
                break
            else:
                # Significant, so add it to marker
                sigcluster += onecluster
                # Erase points that gave significance (set different conds to same value so they are not sig. anymore)
                epoch = np.swapaxes(epoch, 0, 2)
                epoch[sigcluster == 1] = 1
                epoch = np.swapaxes(epoch, 0, 2)
        else:
            print('Error, surely shouldnt get this far (i.e. 100 separate clusters)')

        out[i_epoch] = sigcluster

    return np.array(out)


def doperms(arr, out, pos):
    shuffledarr = shufflearr(arr)
    out[pos] = findsignificancecluster(shuffledarr)[1]


def findsignificancecluster(arr):
    arr = np.swapaxes(arr, 0, 2)  # To index by timepoint
    pvals = np.zeros(arr.shape[0])
    for t_i, t in enumerate(arr):
        # Remove any cells that had nan's
        t = t[~np.isnan(t).any(axis=1)]
        pvals[t_i] = anova(t.T)
    if len(pvals[~np.isnan(pvals)]) != len(pvals):
        raise Exception('Invalid data - nans present')
    pvalsbool = np.array(pvals < 0.05, dtype=int)
    return findlongestrun(pvalsbool)


def shufflearr(arr):
    arr = np.swapaxes(arr, 0, 1)  # To index by cell (now cell x trans_hist x time)
    # Now randomise the labels for every cells beta values
    out = np.zeros(arr.shape)
    for cell_i, cell in enumerate(arr):
        out[cell_i] = cell[np.random.permutation(arr.shape[1])]
    out = np.swapaxes(out, 0, 1)  # Undo axis swap
    return out


def findlongestrun(arr):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], arr, [0]))
    difs = np.diff(bounded)  # get 1 at run starts and -1 at run ends
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    lengths = run_ends - run_starts
    out = np.zeros(arr.shape)
    if len(lengths > 0):
        maxlength_i = np.argmax(lengths)
        out[run_starts[maxlength_i]:run_ends[maxlength_i]] = 1
        maxlength = np.max(lengths)
    else:
        maxlength = 0
    return out, maxlength


def anova(arr):
    def _square_of_sums(a):
        s = np.sum(a, 0)
        if not np.isscalar(s):
            return s.astype(float) * s
        else:
            return float(s) * s

    def _sum_of_squares(a):
        return np.sum(a * a, 0)

    # # If all inputs equivalent return 0, not nan as default behaviour
    # if sum([(np.asarray(arr[0]) == x).all() for x in arr[1:]]) == len(arr)-1:
    #     return 0

    args = [np.asarray(arg, dtype=float) for arg in arr]

    # ANOVA on N groups, each in its own array
    num_groups = len(args)
    alldata = np.concatenate(args)
    bign = len(alldata)

    # Determine the mean of the data, and subtract that from all inputs to a
    # variance (via sum_of_sq / sq_of_sum) calculation.  Variance is invariance
    # to a shift in location, and centering all data around zero vastly
    # improves numerical stability.
    offset = alldata.mean()
    alldata -= offset

    sstot = _sum_of_squares(alldata) - (_square_of_sums(alldata) / float(bign))
    ssbn = 0
    for a in args:
        ssbn += _square_of_sums(a - offset) / float(len(a))

    # Naming: variables ending in bn/b are for "between treatments", wn/w are
    # for "within treatments"
    ssbn -= (_square_of_sums(alldata) / float(bign))
    sswn = sstot - ssbn
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    if f < 0:  # correct rounding errors :/
        f = 0
    prob = special.fdtrc(dfbn, dfwn, f)  # equivalent to stats.f.sf

    return prob


def normalisedata(arr):
    if D.rsa_norm_method == 'normalise':
        arr -= np.min(arr)
        arr /= np.max(arr)
    elif D.rsa_norm_method == 'standardise':
        arr -= np.mean(arr)
        arr /= np.std(arr)
    else:
        raise Exception('Error!')
    return arr


def decode_across_epochs(x, y, dec_to_test=None, train_across_conds=False, permtest=False):

    if permtest:
        n_epochs = 1
        n_timepoints = 1
    else:
        n_epochs = y.shape[0]
        n_timepoints = y.shape[-1]

    numconds = y.shape[1]
    accuracies = np.empty((numconds, n_epochs, n_timepoints))

    # Collapse across conditions to then test within conditions
    if train_across_conds:

        for epoch in range(n_epochs):

            accuracies[:, epoch] = decode_epoch_acrossconds(x, y, epoch, n_timepoints)

        return accuracies


    for epoch in range(n_epochs):

        for cond in range(numconds):

            if dec_to_test is not None:

                accuracies[cond, epoch], _ = decode_epoch_traintestsplit(x, y, epoch, cond, n_timepoints, dec_in=dec_to_test)

            else:

                if D.dec_leaveoneout:

                    accuracies[cond, epoch], _ = decode_epoch_leaveoneout(x, y, epoch, cond, n_timepoints)

                else:

                    accuracies[cond, epoch], _ = decode_epoch_traintestsplit(x, y, epoch, cond, n_timepoints)


    return accuracies


# For each timepoint, train a decoder and then test decoder on all other time points
# Returns 2D matrix of decoder at all time points
def decode_stability(x, y):

    n_epochs = y.shape[0]
    n_timepoints = D.n_timepoints
    cond = 0
    n_pnts = n_epochs * (n_timepoints+1)
    accuracies = nans((n_pnts, n_pnts))

    if x.shape[1] == 1:
        # If testing within condition, then choose indices for train test split
        samples_per_label = x.shape[-1]
        num_test_samples = int(samples_per_label * D.dec_test_size)

        if num_test_samples == 0 or num_test_samples == samples_per_label:
            raise Exception('Not enough data to split into train and test')

        test_inds = np.random.choice(samples_per_label, num_test_samples, replace=False)
        train_mask = np.ones(samples_per_label, dtype=bool)
        train_mask[test_inds] = False
        train_inds = np.arange(samples_per_label)[train_mask]

        x_test, x_train = x[:, :, :, :, test_inds], x[:, :, :, :, train_inds]
        y_test, y_train = y[:, :, :, :, test_inds], y[:, :, :, :, train_inds]

    else:
        # Otherwise the 2 conditions are our train and test data
        x_test, x_train = x[:, :1,], x[:, 1:]
        y_test, y_train = y[:, :1], y[:, 1:]

    for epoch in range(n_epochs):

        # Get a decoder for each time point
        _, decs_across_timepoints = decode_epoch_stability(x_train, y_train, epoch, cond)

        for i_dec, dec in enumerate(decs_across_timepoints):

            # Now test these decoders on all time points
            for test_epoch in range(n_epochs):

                i_start = test_epoch * (n_timepoints+1)
                i_stop = i_start + n_timepoints

                accuracies[i_dec + (epoch * (n_timepoints+1)), i_start:i_stop], _ = decode_epoch_stability(x_test, y_test, test_epoch, cond, dec_in=dec)

    return accuracies


def decode_epoch_stability(x, y, epoch, cond, dec_in=None):

    n_timepoints = y.shape[-1]
    accs = np.zeros(n_timepoints)
    decs = []

    for ti in range(n_timepoints):
        y_arr = y[epoch, cond, :, :, :, ti]  # y by trials
        x_arr = x[epoch, cond, :, :, :]  # x by trials

        # Collapse across labels
        y_arr = y_arr.reshape((y_arr.shape[0], y_arr.shape[1] * y_arr.shape[2])).T
        x_arr = x_arr.reshape((x_arr.shape[0], x_arr.shape[1] * x_arr.shape[2])).T

        if dec_in == None:

            # If no decoder provided we need to train one
            dec = create_and_train_decoder(x_arr, y_arr)
            decs.append(dec)

        else:

            # Decoder provided so test it on all samples
            score = test_decoder(dec_in, x_arr, y_arr)
            accs[ti] = score

    return accs, decs



# Decode all timepoints of a single epoch
# If no decoder is provided, then it splits data into train test split and creates one
# Returns accuracies for each time point, and the best decoder
def decode_epoch_traintestsplit(x, y, epoch, cond, n_timepoints, dec_in=None, peak_tp=None, return_all_decs=False):

    accs = np.zeros(n_timepoints)
    decs = []

    for ti in range(n_timepoints):
        y_arr = y[epoch, cond, :, :, :, ti]  # y by trials
        x_arr = x[epoch, cond, :, :, :]  # x by trials

        y_arr = np.reshape(y_arr, (y_arr.shape[0], y_arr.shape[1] * y_arr.shape[2]))
        x_arr = np.reshape(x_arr, (x_arr.shape[0], x_arr.shape[1] * x_arr.shape[2]))

        # Filter out NaNs across both dimensions
        if np.isnan(x_arr).any():
            # First remove trials without entries (trailing nans)
            x_arr = x_arr[:, ~np.isnan(x_arr).all(axis=0)]
            y_arr = y_arr[:, ~np.isnan(y_arr).all(axis=0)]

            # Now remove neurons without any entries
            x_arr = x_arr[~np.isnan(x_arr).all(axis=1)]
            y_arr = y_arr[~np.isnan(y_arr).all(axis=1)]

        if dec_in == None:

            # If no decoder provided we need to train one
            x_train, x_test, y_train, y_test = splitdata(x_arr, y_arr)

            dec = create_and_train_decoder(x_train, y_train)

        else:

            # Decoder provided so test it on all samples
            x_test, y_test, dec = x_arr.T, y_arr.T, dec_in

        score = test_decoder(dec, x_test, y_test)

        if score > np.max(accs) or ti == 0:
            best_dec = dec

        decs.append(dec)
        accs[ti] = score

    if peak_tp is not None:
        best_dec = decs[peak_tp]

    if return_all_decs:
        best_dec = decs

    return accs, best_dec


# Train across all conds then test on the individual conds
def decode_epoch_acrossconds(x, y, epoch, n_timepoints):

    n_conds = x.shape[1]
    accs = np.zeros((n_conds, n_timepoints))

    for ti in range(n_timepoints):

        # First we split data into train and test across conditions
        for i_cond in range(n_conds):

            y_arr = y[epoch, i_cond, :, :, ti]  # y by trials
            x_arr = x[epoch, i_cond, :, :]  # x by trials

            # Filter out NaNs across both dimensions
            if np.isnan(x_arr).any():
                x_arr = filternans2d(x_arr)
                y_arr = filternans2d(y_arr)

            x_train, x_test, y_train, y_test = splitdata(x_arr, y_arr)

            if i_cond == 0:
                x_train_all, y_train_all = np.copy(x_train), np.copy(y_train)
                x_test_all = np.empty((n_conds, x_test.shape[0], x_test.shape[1]))
                y_test_all = np.empty((n_conds, y_test.shape[0], y_test.shape[1]))
            else:
                try:
                    x_train_all = np.vstack((x_train_all, x_train))
                    y_train_all = np.vstack((y_train_all, y_train))
                except ValueError:
                    print('You have different number of cells in the different conditions')

            x_test_all[i_cond] = x_test
            y_test_all[i_cond] = y_test

        # Finally train decoder
        dec = create_and_train_decoder(x_train_all, y_train_all)

        # Now loop through conds and test on different left out data
        for i_cond in range(n_conds):

            score = test_decoder(dec, x_test_all[i_cond], y_test_all[i_cond])

            accs[i_cond, ti] = score

    return accs


# Use leave one out method instead
def decode_epoch_leaveoneout(x, y, epoch, cond, n_timepoints):

    accs = np.zeros(n_timepoints)

    for ti in range(n_timepoints):
        y_arr = y[epoch, cond, :, :, ti]  # y by trials
        x_arr = x[epoch, cond, :, :]  # x by trials

        y_arr = np.reshape(y_arr, (y_arr.shape[0], y_arr.shape[1] * y_arr.shape[2]))
        x_arr = np.reshape(x_arr, (x_arr.shape[0], x_arr.shape[1] * x_arr.shape[2]))

        # Filter out NaNs across both dimensions
        if np.isnan(x_arr).any():
            # First remove trials without entries (trailing nans)
            x_arr = x_arr[:, ~np.isnan(x_arr).all(axis=0)]
            y_arr = y_arr[:, ~np.isnan(y_arr).all(axis=0)]

            # Now remove neurons without any entries
            x_arr = x_arr[~np.isnan(x_arr).all(axis=1)]
            y_arr = y_arr[~np.isnan(y_arr).all(axis=1)]


        n = x_arr.shape[1]
        scores = []

        for i in range(n):
            m = np.ones(n, dtype=bool)
            m[i] = False

            # Now find other corresponding labels to exclude so there's an even number
            n_per_labs = [sum(x_arr[0][m]==l) for l in np.unique(x_arr[0])]

            # For all labels
            for i_n, n_per_lab in enumerate(n_per_labs):

                # If this label has too many trials
                if n_per_lab != np.min(n_per_labs):
                    # Pick a random one and add it to the mask
                    l = np.unique(x_arr[0])[i_n]  # Find which label it is
                    ls = np.where(x_arr[0]==l)[0]  # Find occurences of this label
                    rand_l = ls[np.random.randint(len(ls))]  # Pick random occurrence
                    m[rand_l] = False

            dec = create_and_train_decoder(x_arr[:, m].T, y_arr[:, m].T)

            score = test_decoder(dec, x_arr[:, ~m].T, y_arr[:, ~m].T)

            scores.append(score)

        if np.mean(scores) > np.max(accs):
            best_dec = dec

        accs[ti] = np.mean(scores)

    return accs, best_dec


# Works out the the minimum number of samples across all cells
def calculate_min_trials_per_area(data, unique_func, dists_all, i_area, train_epoch_only=False):

    if train_epoch_only:
        dists = nans([1, dists_all[i_area].shape[-1]])
    else:
        dists = nans(dists_all[i_area].shape)

    for cell in range(data.n):
        td = data.behavdata[cell]

        x_datas, masks = unique_func(td)

        # Allow users to only specify one x data
        if np.array(x_datas).ndim == 1:
            x_datas = [x_datas] * len(masks)

        # Only look at first condition in this case
        if train_epoch_only:
            x_datas, masks = [x_datas[0]], [masks[0]]

        # Remove any -1 entries in the x values
        masks = [dec_removeinvalidentries(mask, x_data) for mask, x_data in zip(masks, x_datas)]
        x_datas = [dec_removeinvalidentries(x_data, x_data) for x_data in x_datas]

        labels = [np.unique(x_data[mask]) for mask, x_data in zip(masks, x_datas)]
        numitems = [len(label) for label in labels]

        # Iterate through each condition
        for i_cond, (m_cond, label, x_data) in enumerate(zip(masks, labels, x_datas)):
            # Get fewest number of trials for the different labels
            dists[i_cond, cell] = np.min([sum(x_data[m_cond] == lab) for lab in label])

    min_trials_per_label = np.array(np.nanmin(dists, axis=1), dtype=int)
    original_trials_per_label = np.copy(min_trials_per_label)

    # Any without enough trials will be excluded
    min_trials_per_label[min_trials_per_label < D.dec_minsamples] = D.dec_minsamples

    # Only allow cells with enough trials per label
    validcells = dists.T[:data.n] >= D.dec_minsamples

    # Correct min trials per label to all be the same
    min_trials_per_label[min_trials_per_label != np.min(min_trials_per_label)] = np.min(min_trials_per_label)

    # Redo mask to make sure same cells are excluded from every analysis
    new_valid_cells = np.nanmin(dists.T[:data.n], axis=1) >= min_trials_per_label[0]
    for i in range(len(validcells.T)):
        validcells[:, i] = new_valid_cells[:len(validcells)]

    max_across_conds = np.max(min_trials_per_label)
    num_labels = len(label)
    num_samples_per_label = min_trials_per_label[0]

    # Now we know the maximum of trials to use with the decoder
    print(f'{i_area} has {original_trials_per_label} min trials per condition, {int(np.mean(validcells) * 100)}% cells included ({sum(validcells)})')

    return num_labels, num_samples_per_label, validcells, dists


def collect_trials_for_decoding(data, unique_func, p, train_epoch_only, do_permtest):
    if train_epoch_only:
        num_conds = 1
        num_labels = p.num_labels_train
        num_samples_per_label = p.num_samples_per_label_train
    else:
        num_conds = p.num_conds
        num_labels = p.num_labels
        num_samples_per_label = p.num_samples_per_label
    validcells = p.validcells

    y_all = nans((p.n_epochs, num_conds, data.n, num_labels, num_samples_per_label, D.n_timepoints))
    x_all = nans((p.n_epochs, num_conds, data.n, num_labels, num_samples_per_label))

    # Loop to collect all the data from the different cells that had enough trials
    for cell, cell_validity in enumerate(validcells):

        td = data.behavdata[cell]

        x_datas, m_conds = unique_func(td)

        if train_epoch_only:
            x_datas, m_conds = [x_datas[0]], [m_conds[0]]

        if len(x_datas) != num_conds or len(m_conds) != num_conds:
            raise Exception('You are not providing appropriate number of x datas and condition files')

        # Allow users to only specify one x data
        if np.array(x_datas).ndim == 1:
            x_datas = [x_datas] * len(m_conds)

        # For each epoch
        for i_epoch, epoch in enumerate(p.epochs):

            # For each condition get the data
            for i_cond, (m_cond, x_data, valid_cell) in enumerate(zip(m_conds, x_datas, cell_validity)):

                # Skip cells without enough trials for a certain condition
                if not valid_cell:
                    continue

                # If we're training across epochs, then use training epoch
                if p.stability and num_conds == 2 and i_cond == 0:
                    y = data.generate_epoch_norm(cell, p.train_epochs[i_epoch], p.peak_halfwidth)
                else:
                    y = data.generate_epoch_norm(cell, epoch, p.peak_halfwidth)

                y_masked = y[m_cond]
                x_masked = x_data[m_cond]

                # Remove any potential -1's in the x data
                y_masked = dec_removeinvalidentries(y_masked, x_masked)
                x_masked = dec_removeinvalidentries(x_masked, x_masked)

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

    return x_all, y_all



def create_and_train_decoder(x, y):

    if D.decoder == 'Logistic Regression':
        decode_inst = LogisticRegression()
    elif D.decoder == 'LDA':
        decode_inst = LinearDiscriminantAnalysis()
    elif D.decoder == 'SVM':
        decode_inst = svm.SVC(probability=D.dec_allow_probs, gamma='scale')

    decode_inst.fit(y, x[:, 0])

    return decode_inst


def test_decoder(dec, x, y):
    score = dec.score(y, x[:, 0])
    return score


def splitdata(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x.T, y.T, test_size=D.dec_test_size, stratify=x.T)

    # Ensure correct test sample size such that even number of each class
    nudge = 0
    while len(x_test[:, 0]) % len(np.unique(x_test[:, 0])) != 0:
        nudge += 0.01
        x_train, x_test, y_train, y_test = train_test_split(x.T, y.T, test_size=D.dec_test_size+nudge, stratify=x.T)
        if nudge > 1:
            raise Exception('while timeout')

    # Check test set has even number of the different categories
    if np.sum(np.diff([np.sum(x_test[:, 0]==x_v) for x_v in np.unique(x_test[:, 0])])) != 0 or len(np.unique(x_test[:, 0])) == 1:
        raise Exception('Uneven number of samples in test group')

    return x_train, x_test, y_train, y_test


def nans(shape_of_arr):
    out = np.zeros(shape_of_arr)
    out.fill(np.nan)
    return out


def ttest_1samp(x, exp_mean):
    t, p = scipy.stats.ttest_1samp(x, exp_mean)
    return p


def filternans2d(arr):
    arr = arr[:, ~np.isnan(arr).all(axis=0)]
    arr = arr[~np.isnan(arr).all(axis=1)]
    return arr


def calc_sig_length_null_dist():

    # See how often you get runs of significance in each epoch
    dist = []
    n = D.n_timepoints
    for i in range(D.numperms):
        bin = np.zeros(n)
        ind_inc_trials = np.random.choice(n, int(n * 0.05), replace=False)
        bin[ind_inc_trials] = 1
        _, l = findlongestrun(bin)
        dist.append(l)
    dist_sorted = np.sort(np.array(dist))

    return dist_sorted


def ttest_every_timepoint(accs_allperms, num_labels):
    null_dist = calc_sig_length_null_dist()
    num_conds = accs_allperms.shape[0]
    n_epochs = accs_allperms.shape[1]

    out = np.empty((num_conds, n_epochs, D.n_timepoints))

    # Also do t-test at every time point
    for i_cond in range(num_conds):

        for i_epoch in range(n_epochs):

            pvals = np.empty(D.n_timepoints)

            for i_ti in range(D.n_timepoints):
                pvals[i_ti] = ttest_1samp(accs_allperms[i_cond, i_epoch, i_ti], 1 / num_labels)

            # Swap NaNs to 1's
            pvals[np.isnan(pvals)] = 1

            # See which runs are significantly long
            sigcluster = np.zeros(D.n_timepoints)

            pvals_bool = pvals < 0.05

            for i in range(D.n_timepoints):

                onecluster, clusterlength = findlongestrun(pvals_bool)

                if sum(null_dist > clusterlength) > D.sigthreshold_onetailed or clusterlength < 1:

                    # If no significance then stop looking
                    break

                else:

                    # Significant, so add it to marker
                    sigcluster += onecluster

                    # Erase points that gave significance (set different conds to same value so they are not sig. anymore)
                    pvals_bool[sigcluster == 1] = 1

            pvals[~np.array(sigcluster, dtype=bool)] = 1

            out[i_cond, i_epoch] = pvals

    return out


def dec_removeinvalidentries(arr_in, check_arr):
    return arr_in[check_arr != -1]
