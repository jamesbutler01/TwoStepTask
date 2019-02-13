import Details as D
import numpy as np
import multiprocessing.managers
import ShiftTrials
import Utils
import Maths
import Plot
Shifter = ShiftTrials.Shift(2)


class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def analysecell(counter, out_betas, data, cell):
    Utils.updatecounts(counter, cell, data.n)

    td = data.behavdata[cell]

    masks = (D.get_A_AR_trials(td), D.get_A_AC_trials(td), D.get_A_BR_trials(td), D.get_A_BC_trials(td))

    x = td.previousreward

    for i_epoch, epoch in enumerate(D.epochs):
        y = data.generatenormalisedepoch(cell, epoch)

        for i_mask, mask in enumerate(masks):
            out_betas[0, i_mask, i_epoch, cell] = Maths.regression(x[mask], y[mask])



if __name__ == "__main__":
    import ManagerAnalysis

    maintitle = 't0 betas by t1 outcome'
    ytitles = 'Mean Betas'
    savefolder = 'reg/AAR/t1'
    trace_names = ('A(x)_AR', 'A(x)_AC', 'A(x)_BR', 'A(x)_BC')
    num_conds = len(trace_names)
    num_rows = 1
    plotfunc = Plot.GeneralPlot
    ManagerAnalysis.Run(analysecell, False, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, plotfunc)
