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
            pool = Pool(5)
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


def decode_across_epochs(x, y, peak_epoch=None, peak_cond=None, train_across_conds=False, permtest=False):

    if permtest:
        n_epochs = 1
        n_timepoints = 1
    else:
        n_epochs = D.numtrialepochs
        n_timepoints = D.num_timepoints

    numconds = y.shape[1]
    accuracies = np.empty((numconds, n_epochs, n_timepoints))

    # Collapse across conditions to then test within conditions
    if train_across_conds:

        for epoch in range(n_epochs):

            accuracies[:, epoch] = decode_epoch_acrossconds(x, y, epoch, n_timepoints)

        return accuracies


    # If they specified a peak_epoch, then train on this epoch and test on all others
    if peak_epoch is not None:

        accuracies[peak_cond, peak_epoch], dec_to_test = decode_epoch_traintestsplit(x, y, peak_epoch, peak_cond, n_timepoints)


    for epoch in range(n_epochs):

        for cond in range(numconds):

            if epoch==peak_epoch and cond == peak_cond and not train_across_conds:

                continue

            if peak_epoch is None:

                # Point to point decoding

                if D.dec_leaveoneout:

                    accuracies[cond, epoch], _ = decode_epoch_leaveoneout(x, y, epoch, cond, n_timepoints)

                else:

                    accuracies[cond, epoch], _ = decode_epoch_traintestsplit(x, y, epoch, cond, n_timepoints)

            else:

                accuracies[cond, epoch], _ = decode_epoch_traintestsplit(x, y, epoch, cond, n_timepoints, dec_in=dec_to_test)

    return accuracies


# Decode all timepoints of a single epoch
# If no decoder is provided, then it splits data into train test split and creates one
# Returns accuracies for each time point, and the best decoder
def decode_epoch_traintestsplit(x, y, epoch, cond, n_timepoints, dec_in=None):

    accs = np.zeros(n_timepoints)

    for ti in range(n_timepoints):
        y_arr = y[epoch, cond, :, :, ti]  # y by trials
        x_arr = x[epoch, cond, :, :]  # x by trials

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

        accs[ti] = score

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


def create_and_train_decoder(x, y):

    if D.decoder == 'Logistic Regression':
        decode_inst = LogisticRegression()
    elif D.decoder == 'LDA':
        decode_inst = LinearDiscriminantAnalysis()
    elif D.decoder == 'SVM':
        decode_inst = svm.SVC()

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
    n = D.num_timepoints
    for i in range(D.numperms):
        bin = np.zeros(n)
        ind_inc_trials = np.random.choice(n, int(n * 0.05), replace=False)
        bin[ind_inc_trials] = 1
        _, l = findlongestrun(bin)
        dist.append(l)
    dist_sorted = np.sort(np.array(dist))

    return dist_sorted

