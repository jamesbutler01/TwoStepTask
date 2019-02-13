import Details as D
import numpy as np
import multiprocessing.managers
import ShiftTrials
import Utils
Shifter = ShiftTrials.Shift(2)


class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def analysecell(counter, out_betas, data, cell):
    Utils.updatecounts(counter, cell, data.n)

    td = data.behavdata[cell]

    mask_aar = D.get_A_AR_trials(td)
    mask_aac = D.get_A_AC_trials(td)
    mask_t0rew_high = td.previousreward == 2
    mask_t0rew_low = td.previousreward != 2

    mask1 = mask_t0rew_high & mask_aar
    mask2 = mask_t0rew_high & mask_aac
    mask3 = mask_t0rew_low & mask_aar
    mask4 = mask_t0rew_low & mask_aac

    masks = (mask1, mask2, mask3, mask4)

    for i_epoch, epoch in enumerate(D.epochs):
        y = data.generatenormalisedepoch(cell, epoch)

        for i_mask, mask in enumerate(masks):
            out_betas[0, i_mask, i_epoch, cell] = np.mean(y[mask], axis=0)


if __name__ == "__main__":
    import ManagerAnalysis

    maintitle = 't1 FR depending on t1 choice and transition'
    ytitles = 'Avg FR (norm)'
    savefolder = 'FR/AAR/byT1'
    trace_names = ('A(2)_AR', 'A(2)_AC', 'A(low)_AR', 'A(low)_AC')
    num_conds = len(trace_names)
    num_rows = 1
    ManagerAnalysis.Run(analysecell, True, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names)
