import numpy as np
import Details as D
import os
import matplotlib.pyplot as plt

domultiproc = True
dec_numiters = 10

numperms = 30
n_cores = 14

areas = ['DMStr', 'DLPFC', 'DLStr', 'ACC', 'FP']  # DMStr = caudate, DLStr = putamen
areanames = ['Caudate', 'DLPFC', 'Putamen', 'ACC', 'FP']  # DMStr = caudate, DLStr = putamen
numareas = len(areas)
names = ['Charlie', 'Jacob']
n_sess = 57

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
ind_dirsgiven_c1 = 3
ind_picchosen1 = 4
ind_sidechosen1 = 5
ind_rt_c1 = 6
ind_transition = 7
ind_dirsgiven_c2 = 8
ind_picchosen2 = 9
ind_sidechosen2 = 10
ind_rt_c2 = 11
ind_c2values = 12
ind_rewardgiven = 13

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
static_prewindow = 1000
static_postwindow = 1000
smooth_window_halfwidth = 50
smooth_step = 10

smooth_prewindow = static_prewindow
smooth_postwindow = static_postwindow

smooth_savedir = f'{smooth_window_halfwidth}_{smooth_step}_{static_prewindow}_{static_postwindow}/'

def calc_smooth_outputlength(prewindow, postwindow, smooth_step):
    return (prewindow+postwindow) // smooth_step

smooth_outputlength = calc_smooth_outputlength(smooth_prewindow, smooth_postwindow, smooth_step)

def converttimetosmoothedtrace(time):
    time += smooth_prewindow
    time /= smooth_step
    return int(time)

# Rsa
rsa_start = converttimetosmoothedtrace(200)
rsa_stop = converttimetosmoothedtrace(600)
rsa_norm_method = 'normalise'

# Trial epochs
n_timepoints = smooth_outputlength
epochs = (sc_madefixation, sc_choice1on, sc_choice1made, sc_transition, sc_choice2on, sc_choice2made, sc_secondaryreinforceron)
names_epochs = ('Fixation', 'Choice 1 on', 'Choice 1 made', '\nTransition Revealed', 'Choice 2 on', '\nChoice 2 made', 'Secondary Reinforcer')
numtrialepochs = len(epochs)

# Permutation tests
sigthreshold_onetailed = int(numperms * 0.05)  # .05% p-value
sigthreshold_twotailed = int(numperms * 0.025)  # .05% p-value

# Decoder
dec_test_size = 0.2
dec_minsamples = 6
dec_do_perms = False
dec_leaveoneout = False
dec_allow_probs = True
decoders = ('Logistic Regression', 'SVM', 'LDA')
decoder = decoders[1]

def savefig_makefolder(savedir, name):
    path = f'{D.dir_savefig}{savedir}'
    if not os.path.exists(path):
        os.makedirs(path)
    import matplotlib.pyplot as plt
    plt.savefig(f'{path}/{name}')
    plt.close('all')


def savefig(savedir):
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
