import numpy as np

domultiproc = True

areas = ['DMStr', 'DLPFC', 'DLStr', 'ACC', 'FP']  # DMStr = caudate, DLStr = putamen
areanames = ['Caudate', 'DLPFC', 'Putamen', 'ACC', 'FP']  # DMStr = caudate, DLStr = putamen
numareas = len(areas)
names = ['Charlie', 'Jacob']

dir_main = 'C:/James/Data/TwoStep/'

dir_subjs = (dir_main+'CharlieData/', dir_main+'JacobData/')
dir_qvals = [dir_subjs[i]+'qvals/qvals_sess_' for i in range(2)]
dir_spikes = (dir_main + 'CharlieData/neuronaldata/', dir_main+'JacobData/neuronaldata/')
dir_task_details = dir_main + 'PreparedData.mat'
dir_npstorage = dir_main+'npy/'
dir_savefig = 'figures/'

# Indices for matlab 'PreparedData' behavioural details
ind_totalnumtrials = 0
ind_goodtrialsind = 1
ind_trialtype = 2
ind_picchosen1 = 3
ind_sidechosen1 = 4
ind_transition = 5
ind_picchosen2 = 6
ind_sidechosen2 = 7
ind_rewardgiven = 8

# Strobe codes
sc_startoftrial = 9
sc_madefixation = 22
sc_choice1on = 23
sc_choice1made = 24
sc_transition = 25  # Colour change to indicate which direction they went
sc_madesecondfixation = 32
sc_choice2on = 33
sc_madefixation0 = 22
sc_choice2made = 34
sc_choice2state = 35
sc_secondaryreinforceron = 37
sc_pumpon = 39
sc_pumpoff = 40
sc_endoftrial = 18

# Smoothing traces
smooth_window_halfwidth = 50
smooth_step = 10
smooth_prewindow = 100
smooth_postwindow = 800
smooth_outputlength = int((smooth_prewindow+smooth_postwindow) / smooth_step)
smooth_savedir = f'{smooth_window_halfwidth}_{smooth_step}_{smooth_prewindow}_{smooth_postwindow}/'

def converttimetosmoothedtrace(time):
    time += smooth_prewindow
    time /= smooth_step
    return int(time)

# Stats
numperms = 1000

# Rsa
rsa_start = converttimetosmoothedtrace(200)
rsa_stop = converttimetosmoothedtrace(600)
rsa_norm_method = 'normalise'

# Trial epochs
num_timepoints = smooth_outputlength
epochs = (sc_madefixation, sc_choice1on, sc_choice1made, sc_transition, sc_choice2on, sc_choice2made, sc_secondaryreinforceron)
names_epochs = ('Fixation', 'Choice 1 on', '\nChoice 1 made', '\nTransition Revealed', 'Choice 2 on', '\nChoice 2 made', 'Secondary Reinforcer')
numtrialepochs = len(epochs)

# Permutation tests
sigthreshold = numperms * 0.005  # .05% p-value
sigthreshold = numperms * 0.025  # 2.5% p-value

# Decoder
dec_numiters_traintestsplit = 6
dec_numiters_cellselection = 5
dec_test_size = 0.2
decoders = ('Logistic Regression', 'SVM', 'LDA')


def savefig(savedir):
    import matplotlib.pyplot as plt
    plt.savefig(f'{dir_savefig}{savedir}')
    plt.close('all')


def get_A_AR_trials(td):
    out = np.zeros(td.c1chosen.shape, dtype=bool)
    for i, (given, chosen, trans) in enumerate(zip(td.c1given[:-1], td.c1chosen[1:], td.transition[1:])):
        if given == chosen and trans == 2:
            out[i+1] = 1
        else:
            out[i+1] = 0
    return out


def get_A_AC_trials(td):
    out = np.zeros(td.c1chosen.shape, dtype=bool)
    for i, (given, chosen, trans) in enumerate(zip(td.c1given[:-1], td.c1chosen[1:], td.transition[1:])):
        if given == chosen and trans == 1:
            out[i+1] = 1
        else:
            out[i+1] = 0
    return out


def get_A_BR_trials(td):
    out = np.zeros(td.c1chosen.shape, dtype=bool)
    for i, (given, chosen, trans) in enumerate(zip(td.c1given[:-1], td.c1chosen[1:], td.transition[1:])):
        if given != chosen and trans == 2:
            out[i+1] = 1
        else:
            out[i+1] = 0
    return out


def get_A_BC_trials(td):
    out = np.zeros(td.c1chosen.shape, dtype=bool)
    for i, (given, chosen, trans) in enumerate(zip(td.c1given[:-1], td.c1chosen[1:], td.transition[1:])):
        if given != chosen and trans == 1:
            out[i+1] = 1
        else:
            out[i+1] = 0
    return out


def shifttrial(arr, t, max_t):
    if t == max_t:
        return arr[t:]  # Can't index by -0 :(
    return arr[t:-(max_t - t)]
