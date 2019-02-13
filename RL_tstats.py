import Details as D
import numpy as np
import multiprocessing.managers
import ShiftTrials
import Utils
import UtilsRsa
import scipy
import Maths
Shifter = ShiftTrials.Shift(2)


class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def analysecell(counter, out_betas, data, cell):
    Utils.updatecounts(counter, cell, data.n)

    td = data.behavdata[cell]


    x = td.rewgiven

    for i_epoch, epoch in enumerate(D.epochs):
        y = UtilsRsa.getavgfr(epoch, data, cell)
        highy = y[x == 2]
        medy = y[x == 1]
        lowy = y[x == 0]

        high_mean = np.mean(highy)
        low_mean = np.mean(lowy)

        [t_score, p_val] = scipy.stats.ttest_1samp(medy, np.mean([[high_mean, low_mean]]))




if __name__ == "__main__":
    import ManagerAnalysis

    maintitle = 't0 betas by t1 outcome'
    ytitles = 'Mean Betas'
    savefolder = 'reg/AAR/t1'
    trace_names = ('A(x)_AR', 'A(x)_AC', 'A(x)_BR', 'A(x)_BC')
    num_conds = len(trace_names)
    num_rows = 1
    ManagerAnalysis.Run(analysecell, False, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names)
