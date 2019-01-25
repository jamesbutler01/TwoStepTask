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
    model = np.linalg.lstsq(x_val, y_val)[0]
    return model[1]

#@jit(nopython=True)
def get_pred_vals(x_val, y_val):
    model = np.linalg.lstsq(x_val, y_val)[0]
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


def permtest(arr):
    out = np.empty((arr.shape[1], arr.shape[3]))
    arr = np.swapaxes(arr, 0, 1)  # To index by trial epoch

    m = MyManager()
    m.start()

    for i_epoch, epoch in enumerate(arr):
        # Get distribution
        epoch_buff = np.copy(epoch)
        dist = m.np_zeros(D.numperms)

        pool = Pool(5)
        func = partial(doperms, epoch_buff, dist)
        run_list = range(D.numperms)
        pool.map(func, run_list)
        pool.close()
        #
        # dist = np.empty(D.numperms)
        # for i in range(D.numperms):
        #     doperms(epoch_buff, dist, i)

        dist = np.sort(np.array(dist))

        # Observed
        sigcluster, clusterlength = findsignificancecluster(epoch)

        if len(np.where(dist > clusterlength)[0]) > D.sigthreshold:
            sigcluster.fill(0)  # Erase cluster if it wasn't significant

        out[i_epoch] = sigcluster

    return out


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


def decode_across_epochs(x, y, decoder):
    numchoices = y.shape[1]

    accuracies = np.empty((numchoices, D.numtrialepochs, D.num_timepoints))
    sems = np.empty((numchoices, D.numtrialepochs, D.num_timepoints))

    for epoch in range(D.numtrialepochs):
        for choice in range(numchoices):
            for ti in range(D.num_timepoints):
                y_aar = y[epoch, choice, :, :, ti]  # y by trials
                x_aar = x[epoch, choice, :, :]  # y by trials

                accuracies[choice, epoch, ti], sems[choice, epoch, ti] = rundecoder(x_aar, y_aar, decoder)

    return accuracies, sems


def rundecoder(x, y, decoder):
    if np.isnan(x).any():
        x = x[:, :np.min(np.where(np.isnan(x))[1])]
        y = y[:, :np.min(np.where(np.isnan(y))[1])]

    score = np.empty(D.dec_numiters_traintestsplit)

    for i in range(D.dec_numiters_traintestsplit):

        # Randomly split into train and test
        x_train, x_test, y_train, y_test = train_test_split(x.T, y.T, test_size=D.dec_test_size)
        x_clip = np.copy(x_test[:, 0])
        if len(x_clip) % 2 == 1:
            x_clip = x_clip[:-1]

        # Keep resampling until test set has even number of the different labels
        counter = 0
        while np.sum(np.diff([np.sum(x_clip==x_v) for x_v in np.unique(x_clip)])) != 0 or len(np.unique(x_clip)) == 1:
            x_train, x_test, y_train, y_test = train_test_split(x.T, y.T, test_size=D.dec_test_size)
            x_clip = np.copy(x_test[:, 0])
            if len(x_clip) % 2 == 1:
                x_clip = x_clip[:-1]
            counter += 1
            if counter > 999999:
                raise Exception(f'1234{x_test}')

        # Do decoding
        if decoder == 'Logistic Regression':
            decode_inst = LogisticRegression()
        elif decoder == 'LDA':
            decode_inst = LinearDiscriminantAnalysis()
        elif decoder == 'SVM':
            decode_inst = svm.SVC()

        decode_inst.fit(y_train, x_train[:, 0])

        # Log score of model
        score[i] = decode_inst.score(y_test, x_test[:, 0])

    return np.mean(score), sem(score)