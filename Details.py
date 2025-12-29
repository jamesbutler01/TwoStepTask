import Details as D
import numpy as np
import os
import matplotlib.pyplot as plt

# Stats
run_perms = False
numperms = 25
do_multiproc = False
n_cores = 1
# areas = ['DLPFC', 'ACC', 'DLStr', 'DMStr', 'FP']  # DMStr = caudate, DLStr = putamen
# areanames = ['DLPFC', 'ACC', 'Putamen', 'Caudate', 'FP']
areas = ['FP', 'ACC', 'DLPFC', 'DMStr', 'DLStr']  # DMStr = caudate, DLStr = putamen
areas = ['FP', 'ACC', 'DLPFC', 'Caudate', 'Putamen']  # DMStr = caudate, DLStr = putamen
areanames = ['FP', 'ACC', 'DLPFC', 'Caudate', 'Putamen']  # DMStr = caudate, DLStr = putamen
areaindex= {'FP' : 'FP', 'ACC' : 'ACC', 'DLPFC' : 'DLPFC', 'Caudate' : 'DMStr', 'Putamen' : 'DLStr'}
numareas = len(areas)
numsessions = 57
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
ind_dirsgiven_c1 = np.nan  # removed
ind_picchosen1 = 3
ind_sidechosen1 = 4
ind_rt_c1 = 5
ind_transition = 6
ind_dirsgiven_c2 = np.nan  # removed
ind_picchosen2 = 7
ind_sidechosen2 = 8
ind_rt_c2 = 9
ind_rewardraw = 10
ind_c2valuesRaw = 11
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
num_timepoints = smooth_outputlength

def converttimetosmoothedtrace(time):
    time += smooth_prewindow
    time /= smooth_step
    return int(time)


# Trial epochs

# Actual all epochs
epochs_all = (sc_madefixation, sc_choice1on, sc_choice1made, sc_transition, sc_choice2on, sc_choice2made, sc_secondaryreinforceron)
epochs_all = (sc_madefixation, sc_choice1on, sc_transition, sc_choice2on, sc_choice2made, sc_secondaryreinforceron)
names_epochs_all = ('Fixation', 'Choice 1 on', 'Choice 1 made', '\nTransition Revealed', 'Choice 2 on', '\nChoice 2 made', 'Secondary Reinforcer')
names_epochs_all = ('Fixation', 'Choice 1 on', '\nTransition Revealed', 'Choice 2 on', '\nChoice 2 made', 'Secondary Reinforcer')

# Included epochs
epochs_inc = (sc_madefixation, sc_choice1on, sc_choice1made, sc_choice2on, sc_choice2made, sc_secondaryreinforceron)
names_epochs_inc = ('Fixation', 'Choice 1 on', 'Choice 1 made', 'Choice 2 on', 'Choice 2 made', 'Secondary Reinforcer')

names_epochs = ['Fixation']
epochs = [sc_madefixation]


epochs=[D.sc_choice2made, D.sc_choice2on]
names_epochs = ['C2 on', 'C2 made']
epochs = [sc_transition]
names_epochs = ['Transition revealed']
epochs = [sc_transition, sc_secondaryreinforceron]
names_epochs = ['Transition', 'Secondary reinforcer']
epochs = epochs_inc
names_epochs = names_epochs_inc
epochs = [sc_secondaryreinforceron]
names_epochs = ['Secondary reinforcer']
epochs = [sc_madefixation, sc_choice1on, sc_choice1made]
names_epochs = ['Fixation', 'Choice 1 made', 'Choice 1 on']

numtrialepochs = len(epochs)

# Permutation tests
pval = 0.05
sigthreshold = int(numperms * pval)

# Decoder
dec_numiters_cellselection = 1
dec_numiters_traintestsplit = 10
dec_test_size = 0.2
decoders = ('Logistic Regression', 'SVM', 'LDA')

# Pop analysis
pop_window_start = 200
pop_window_stop = 500
pop_norm_method = 'normalise'
pop_numstates_d1 = 8
pop_numstates_d1all = 12
pop_numstates_d1_rarecomm = 16
pop_numstates_d2 = 14
pop_numstates_d3= 4
pop_numstates_d4= 3
pop_numstates_d5= 3
pop_numstates_d1coll= 3
pop_numstates_coll = pop_numstates_d1 * 2 + pop_numstates_d2 + 2
pop_epochs = (D.sc_choice1made, D.sc_choice2made, D.sc_secondaryreinforceron)
pop_names_epochs = ('choice1', 'choice2', 'feedback')
pop_names_datatypes = ('fr(norm)', 'tstat', 'tstat(norm)', 'P-values')
pop_names_ss = ('d1', 'd1all', 'd1rarecomm', 'd2', 'd3', 'dRew', 'd1coll', 'd5')
pop_numstates_per_ss = (pop_numstates_d1, pop_numstates_d1all, pop_numstates_d1_rarecomm, pop_numstates_d2, pop_numstates_d3, pop_numstates_d4, pop_numstates_d1coll, pop_numstates_d5)
pop_num_epochs = len(pop_names_epochs)
pop_num_datatypes = len(pop_names_datatypes)
pop_num_statespaces = len(pop_names_ss)
pop_labels = [
    ['A(0)_AC', 'A(0)_BC', 'A(0)_BR', 'A(1)_AC', 'A(1)_BC', 'A(1)_BR', 'A(2)_AC', 'A(2)_AR'],
    ['A(0)_AC', 'A(1)_AC', 'A(2)_AC', 'A(0)_AR', 'A(1)_AR', 'A(2)_AR', 'A(0)_BC', 'A(1)_BC', 'A(2)_BC', 'A(0)_BR',
    'A(1)_BR', 'A(2)_BR'],
    ['AC(0)_AC', 'rare', 'AC(0)_BR', 'rare', 'AC(0)_BC', 'rare', 'AC(1)_AC', 'rare',
     'AC(1)_BR', 'rare', 'AC(1)_BC', 'rare', 'AC(2)_AC', 'rare', 'AC(2)_AR', 'rare'],
    ['A(0)_AC(0)', 'A(2)_AR(0)', 'A(0)_BR(0)', 'A(0)_BC(0)', 'A(1)_BC(0)', 'A(1)_AC(1)', 'A(2)_AR(1)',
     'A(1)_BR(1)', 'A(0)_BC(1)', 'A(1)_BC(1)', 'A(2)_AC(2)', 'A(2)_AR(2)', 'A(0)_BC(2)', 'A(1)_BC(2)', '-PE'],
    ['A(2)_AC(2)', 'A(2)_AR(2)', 'A(0)_BC(2)', 'A(1)_BC(2)'],
    ['0', '1', '2'],
    ['A(2)_AC', 'A(2)_AR', 'A(low)_*'],
    ['A(2)_AC', 'A(2)_AR', 'A(low)_*'],
    ]
pop_labels_alt = [
    ['Repeat low', 'Switch low', 'Switch low (rare)', 'Repeat med.', 'Switch med.', 'Switch med. (rare)', 'Repeat high', 'Repeat high (rare)'],
    ['Repeat low', 'Repeat med.', 'Repeat high', 'Repeat low (rare)', 'Repeat med. (rare)', 'Repeat high (rare)', 'Switch low', 'Switch med.', 'Switch high', 'Switch low (rare)',
    'Switch med. (rare)', 'Switch high (rare)'],
    ['Success-Repeat low', 'rare', 'Success-Switch low (rare)', 'rare', 'Success-Switch low', 'rare', 'Success-Repeat med.', 'rare',
     'Success-Switch med. (rare)', 'rare', 'Success-Switch med.', 'rare', 'Success-Repeat high', 'rare', 'Success-Repeat high (rare)', 'rare'],
    ['Repeat low-Low', 'Repeat high (rare)-Low', 'Switch lowRLow', 'Switch lowCLow', 'Switch med.CLow', 'Repeat med.-Med', 'Repeat high (rare)-Med',
     'Switch med.RMed', 'Switch lowCMed', 'Switch med.CMed', 'Repeat high-High', 'Repeat high (rare)-High', 'Switch lowCHigh', 'Switch med.CHigh', '-PE'],
    ['Repeat high-High', 'Repeat high (rare)-High', 'Switch lowCHigh', 'Switch med.CHigh'],
    ['0', '1', '2'],
    ['Repeat high', 'Repeat high (rare)', 'A(low)_*'],
    ['Repeat high', 'Repeat high (rare)', 'A(low)_*'],
    ]


pop_labels_coll = [
    'C1 A(0)_AC', 'C1 A(0)_BC', 'C1 A(0)_BR', 'C1 A(1)_AC', 'C1 A(1)_BC', 'C1 A(1)_BR', 'C1 A(2)_AC', 'C1 A(2)_AR',
    'C2 A(0)_AC', 'C2 A(0)_BC', 'C2 A(0)_BR', 'C2 A(1)_AC', 'C2 A(1)_BC', 'C2 A(1)_BR', 'C2 A(2)_AC', 'C2 A(2)_AR',
    'FB A(0)_AC(0)', 'FB A(2)_AR(0)', 'FB A(0)_BR(0)', 'FB A(0)_BC(0)', 'FB A(1)_BC(0)', 'FB A(1)_AC(1)', 'FB A(2)_AR(1)',
    'FB A(1)_BR(1)', 'FB A(0)_BC(1)', 'FB A(1)_BC(1)', 'FB A(2)_AC(2)', 'FB A(2)_AR(2)', 'FB A(0)_BC(2)', 'FB A(1)_BC(2)',
    'FB A(2)_AC(0)', 'FB A(2)_AC(1)'
    ]


def calc_smooth_outputlength(prewindow, postwindow):
    return (prewindow+postwindow)//smooth_step

def savefig(savedir, name):
    path = f'{D.dir_savefig}{savedir}'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{name}')


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

