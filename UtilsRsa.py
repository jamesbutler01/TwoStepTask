import numpy as np
import Details as D

# Now do RSA
def rsa(arr, second_arr, suffix, plotfunc, area):
    rsa = np.empty((arr.shape[1], arr.shape[1]))
    for i_state, state in enumerate(arr.T):
        state_mask = ~np.isnan(state)
        for j_state, state2 in enumerate(second_arr.T):
            rsa[i_state, j_state] = Maths.corrcoef(state, state2)[0, 1]

    plotfunc(rsa, area, suffix)

    return rsa.flatten()

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


def getavgfr(sc, data, cell):
    fr = data.generatenormalisedepoch(cell, sc)
    fr_window = np.mean(fr[:, D.rsa_start:D.rsa_stop], axis=1)  # Just take average FR just after the event
    return fr_window
