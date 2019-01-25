import Details as D
import numpy as np
import multiprocessing.managers
import ShiftTrials
import Utils
import Maths
Shifter = ShiftTrials.Shift(2)


class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def analysecell(counter, out_betas, data, cell):
    Utils.updatecounts(counter, cell, data.n)

    td = data.behavdata[cell]

    t0c1dir = Shifter.shift(td.c1dir, 0)
    t0c2dir = Shifter.shift(td.c2dir, 0)
    t0rew = Shifter.shift(td.rewgiven, 0)
    t0trans = Shifter.shift(td.transition, 0)
    t0switchc2 = Shifter.shift(td.switchchoice2, 0)

    for i_epoch, epoch in enumerate(D.epochs):
        y = data.generatenormalisedepoch(cell, epoch)
        y = Shifter.shift(y, 0)

        out_betas[0, 0, i_epoch, cell] = Maths.anova(y[(t0rew == i)] for i in np.unique(t0rew))
        out_betas[1, 0, i_epoch, cell] = Maths.anova(y[(t0rew == i) & (t0switchc2 == 0)] for i in np.unique(t0rew))
        out_betas[2, 0, i_epoch, cell] = Maths.anova(y[(t0rew == i) & (t0switchc2 == 1)] for i in np.unique(t0rew))
        out_betas[3, 0, i_epoch, cell] = Maths.anova(y[t0c1dir == i] for i in np.unique(t0c1dir))
        out_betas[4, 0, i_epoch, cell] = Maths.anova(y[t0c2dir == i] for i in np.unique(t0c2dir))
        out_betas[5, 0, i_epoch, cell] = Maths.anova(y[t0trans == i] for i in np.unique(t0trans))

if __name__ == "__main__":
    import ManagerAnalysis

    maintitle = 'FR depending on next trial choice'
    ytitles = ['Reward', 'Exp Reward', 'Novel Reward', 'C1 Direction', 'C2 Direction', 'Transition']
    savefolder = 'Anovas'
    trace_names = ['% significant cells']
    num_conds = len(trace_names)
    num_rows = len(ytitles)
    ManagerAnalysis.Run(analysecell, False, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, False)
