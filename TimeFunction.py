import time
import statsmodels.api as sm
from scipy import stats
import numpy as np
from numba import jit
import statsmodels.api as sm
import Maths
from statsmodels.formula.api import ols


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsedtime(self):
        elapsed_time = time.time() - self.start
        out = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        return out


x = np.array(np.random.randint(0,4,4))
A = np.vstack([np.ones(len(x)), x]).T
y = np.array([-1, 0.2, 0.9, 2.1])

##### Corrcoef
def corrcoef():
    return Maths.corrcoef(x, y)[0, 1]

def np_corrcoef():
    c = np.cov(x, y, True)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    e = stddev[:, None]
    c /= e
    c /= stddev[None, :]
    np.clip(c.real, -1, 1, out=c.real)
    return c[0, 1]

def jit_corrcoef():
    c = jit_corrcoef_helper(x, y)
    np.clip(c.real, -1, 1, out=c.real)
    return c[0, 1]

@jit(nopython=True)
def jit_corrcoef_helper(x, y):
    c = np.cov(x, y, True)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev.reshape((-1, 1))
    c /= stddev.reshape((1, -1))
    return c



##### REgression

def sm_ols():
    x = np.array(np.random.randint(0, 4, 4))
    A = np.vstack([np.ones(len(x)), x, x, x]).T
    y = np.array([-1, 0.56, 0.9, 1.1])
    def runregression(A, y):

      a = sm.OLS(y, A).fit()
      b = a.params[0]

    runregression(A, y)

def np_ols():
    x = np.array(np.random.randint(0, 4, 4))
    A = np.vstack([np.ones(len(x)), x, x, x]).T
    y = np.array([-1, 0.2, 0.9, 2.1])
    def runregression(A, y):

      a = np.linalg.lstsq(A, y, rcond=None)[0]

    runregression(A, y)

@jit(nopython=True)
def runregression(A, y):
    a = np.linalg.lstsq(A, y)[0]

def np_ols_jit():
    x = np.array(np.random.randint(0, 4, 4))
    A = np.vstack([np.ones(len(x)), x, x, x]).T
    y = np.array([-1, 0.2, 0.9, 2.1])

    runregression(A, y)


## CoD
def cod_sm():
    x = np.array(np.random.randint(0, 4, 4))
    A = np.vstack([np.ones(len(x)), x, x, x]).T
    y = np.array([-1, 0.2, 0.9, 2.1])
    def runregression(A, y):

      a = sm.OLS(y, A).fit().rsquared

    runregression(A, y)


def cod_np():
    x = np.array(np.random.randint(0, 4, 4))
    A = np.vstack([np.ones(len(x)), x]).T
    y = np.array([-1, 0.2, 0.9, 2.1])
    def runregression(A, y):

      a = np.square(Maths.corrcoef(A.T[1], y)[0, 1])

    runregression(A, y)

def cod_jit():
    x = np.array(np.random.randint(0, 4, 4))
    A = np.vstack([np.ones(len(x)), x, x, x]).T
    y = np.array([-1, 0.2, 0.9, 2.1])
    def runregression(A, y):

      a = jit_sq(Maths.corrcoef(A.T[1], y)[0, 1])

    runregression(A, y)

@jit(nopython=True)
def jit_sq(arr):
    return np.square(arr)

# Anovas
def scipystats():
    arr = [np.asarray(arg, dtype=float) for arg in arr_anova]
    stats.f_oneway(arr[0], arr[1], arr[2], arr[3])[1]

# Permutations
arr = np.random.randint(10, size=(278, 4, 110))


@jit(nopython=True)
def jit_shufflearr():
    # Now randomise the labels for every cells beta values
    out = np.empty(arr.shape)
    for cell_i in range(len(arr)):
        out[cell_i] = arr[cell_i][np.random.permutation(4)]
    return out

def shufflearr():
    # Now randomise the labels for every cells beta values
    out = np.empty(arr.shape)
    for cell_i in range(len(arr)):
        out[cell_i] = arr[cell_i][np.random.permutation(4)]
    return out

arr_anova =np.random.randint(10, size=(4,278))

import scipy.special as special  # F-distribution lookup table

def scipyanova():
    # Ripped straight from scipy.stats.f_oneway but adapted to accept lists as input
    def _chk_asarray(a, axis):
        if axis is None:
            a = np.ravel(a)
            outaxis = 0
        else:
            a = np.asarray(a)
            outaxis = axis

        if a.ndim == 0:
            a = np.atleast_1d(a)

        return a, outaxis


    def _square_of_sums(a, axis=0):
        a, axis = _chk_asarray(a, axis)
        s = np.sum(a, axis)
        if not np.isscalar(s):
            return s.astype(float) * s
        else:
            return float(s) * s


    def _sum_of_squares(a, axis=0):
        a, axis = _chk_asarray(a, axis)
        return np.sum(a * a, axis)


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


def anova_reduced():
    def _square_of_sums(a):
        s = np.sum(a, 0)
        if not np.isscalar(s):
            return s.astype(float) * s
        else:
            return float(s) * s

    def _sum_of_squares(a):
        return np.sum(a * a, 0)

    args = [np.asarray(arg, dtype=float) for arg in arr_anova]
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

@jit(nopython=True)
def anova_helper(sstot, ssbn, num_groups, bign):
    sswn = sstot - ssbn
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    return dfbn, dfwn, f

@jit(nopython=True)
def _sum_of_squares(a):
    return np.sum(a * a, 0)

#@jit(nopython=True)
def _square_of_sums(a):
    s = np.sum(a, 0)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s

def jit_anova():
    args = [np.asarray(arg, dtype=float) for arg in arr_anova]
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
    dfbn, dfwn, f= anova_helper(sstot, ssbn, num_groups, bign)

    prob = special.fdtrc(dfbn, dfwn, f)  # equivalent to stats.f.sf

    return prob

def anova_used():
    def _square_of_sums(a):
        s = np.sum(a, 0)
        if not np.isscalar(s):
            return s.astype(float) * s
        else:
            return float(s) * s

    def _sum_of_squares(a):
        return np.sum(a * a, 0)

    args = [np.asarray(arg, dtype=float) for arg in arr_anova]
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


if __name__ == '__main__':
    import timeit
    # Corrcoef
    print(timeit.timeit("corrcoef()", setup="from __main__ import corrcoef", number=10000))
    print(timeit.timeit("corrcoef()", setup="from __main__ import corrcoef", number=10000))
    print(timeit.timeit("corrcoef()", setup="from __main__ import corrcoef", number=10000))
    # Corrcoef
    print(timeit.timeit("np_corrcoef()", setup="from __main__ import np_corrcoef", number=10000))
    print(timeit.timeit("np_corrcoef()", setup="from __main__ import np_corrcoef", number=10000))
    print(timeit.timeit("np_corrcoef()", setup="from __main__ import np_corrcoef", number=10000))
    # Corrcoef
    print(timeit.timeit("jit_corrcoef()", setup="from __main__ import jit_corrcoef", number=10000))
    print(timeit.timeit("jit_corrcoef()", setup="from __main__ import jit_corrcoef", number=10000))
    print(timeit.timeit("jit_corrcoef()", setup="from __main__ import jit_corrcoef", number=10000))
    
    # Anova
    print(timeit.timeit("anova_used()", setup="from __main__ import anova_used", number=1000))
    print(timeit.timeit("anova_used()", setup="from __main__ import anova_used", number=1000))
    print(timeit.timeit("anova_used()", setup="from __main__ import anova_used", number=1000))
    # Calling actual function
    print(timeit.timeit("scipystats()", setup="from __main__ import scipystats", number=1000))
    print(timeit.timeit("scipystats()", setup="from __main__ import scipystats", number=1000))
    print(timeit.timeit("scipystats()", setup="from __main__ import scipystats", number=1000))
    # Function copied and pasted
    print(timeit.timeit("scipyanova()", setup="from __main__ import scipyanova", number=1000))
    print(timeit.timeit("scipyanova()", setup="from __main__ import scipyanova", number=1000))
    print(timeit.timeit("scipyanova()", setup="from __main__ import scipyanova", number=1000))
    # Function edited
    print(timeit.timeit("anova_reduced()", setup="from __main__ import anova_reduced", number=1000))
    print(timeit.timeit("anova_reduced()", setup="from __main__ import anova_reduced", number=1000))
    print(timeit.timeit("anova_reduced()", setup="from __main__ import anova_reduced", number=1000))
    # Jit attempt
    print(timeit.timeit("jit_anova()", setup="from __main__ import jit_anova", number=1000))
    print(timeit.timeit("jit_anova()", setup="from __main__ import jit_anova", number=1000))
    print(timeit.timeit("jit_anova()", setup="from __main__ import jit_anova", number=1000))

    if False:
        print(timeit.timeit("cod_np()", setup="from __main__ import cod_np", number=10000))
        print(timeit.timeit("cod_np()", setup="from __main__ import cod_np", number=10000))
        print(timeit.timeit("cod_np()", setup="from __main__ import cod_np", number=10000))

        print(timeit.timeit("cod_sm()", setup="from __main__ import cod_sm", number=10000))
        print(timeit.timeit("cod_sm()", setup="from __main__ import cod_sm", number=10000))
        print(timeit.timeit("cod_sm()", setup="from __main__ import cod_sm", number=10000))

        print(timeit.timeit("cod_jit()", setup="from __main__ import cod_jit", number=10000))
        print(timeit.timeit("cod_jit()", setup="from __main__ import cod_jit", number=10000))
        print(timeit.timeit("cod_jit()", setup="from __main__ import cod_jit", number=10000))

        print(timeit.timeit("jit_shufflearr()", setup="from __main__ import jit_shufflearr", number=10000))
        print(timeit.timeit("jit_shufflearr()", setup="from __main__ import jit_shufflearr", number=10000))
        print(timeit.timeit("jit_shufflearr()", setup="from __main__ import jit_shufflearr", number=10000))

      # Linear regression
        print(timeit.timeit("sm_ols()", setup="from __main__ import sm_ols", number=10000))
        print(timeit.timeit("sm_ols()", setup="from __main__ import sm_ols", number=10000))
        print(timeit.timeit("sm_ols()", setup="from __main__ import sm_ols", number=10000))

        print(timeit.timeit("np_ols()", setup="from __main__ import np_ols", number=10000))
        print(timeit.timeit("np_ols()", setup="from __main__ import np_ols", number=10000))
        print(timeit.timeit("np_ols()", setup="from __main__ import np_ols", number=10000))

        print(timeit.timeit("np_ols_jit()", setup="from __main__ import np_ols_jit", number=10000))
        print(timeit.timeit("np_ols_jit()", setup="from __main__ import np_ols_jit", number=10000))
        print(timeit.timeit("np_ols_jit()", setup="from __main__ import np_ols_jit", number=10000))


