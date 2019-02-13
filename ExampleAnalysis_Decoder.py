import Details as D
import numpy as np

def createmasks(td):

    x_data = td.c1chosen

    mask_aar = D.get_A_AR_trials(td)
    mask_aac = D.get_A_AC_trials(td)
    mask_abc = D.get_A_BC_trials(td)

    switchc1 = mask_abc | mask_aar

    masks = [(mask_aac & (td.rewgiven == 2) & (td.previousreward == 2)),
             (mask_abc & (td.rewgiven == 2)& (td.previousreward != 2)),
             (mask_abc & (td.rewgiven != 2)& (td.previousreward != 2))]

    mask_aac = D.get_A_AC_trials(td)
    mask_abc = D.get_A_BC_trials(td)

    masks = [(mask_aac & (td.previousreward == 2)),
             (mask_abc & (td.previousreward == 2))]

    return x_data, masks


if __name__ == "__main__":
    import ManagerDecoder

    maintitle = 'Decoding t0 choice1 on t0'
    ytitles = 'Accuracy'
    savefolder = 'dec/c1/id/B(high)_B(low)'
    trace_names = ['A(high)_A(high)', 'A(low)_B(high)', 'A(low)_B(low)']
    trace_names = ['A(high)_A', 'A(high)_B']
    num_conds = len(trace_names)
    num_rows = 1
    decoder = D.decoders[1]
    minsamples = 8
    ManagerDecoder.Run(createmasks, False, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, decoder, minsamples)



