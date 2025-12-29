import Details as D
import numpy as np
import os
import matplotlib.pyplot as plt

# Stats
numperms = 25
do_multiproc = False
n_cores = 1
areas = ['FP', 'ACC', 'DLPFC', 'Caudate', 'Putamen']
areaindex= {'FP' : 'FP', 'ACC' : 'ACC', 'DLPFC' : 'DLPFC', 'Caudate' : 'DMStr', 'Putamen' : 'DLStr'}
numareas = len(areas)
numsessions = 57
names = ['Charlie', 'Jacob']
dir_main = 'C:/James/Data/TwoStep/'
dir_subjs = (dir_main+'CharlieData/', dir_main+'JacobData/')
dir_qvals = [dir_subjs[i]+'qvals/qvals_sess_' for i in range(2)]
dir_spikes = (dir_main + 'CharlieData/neuronaldata/', dir_main+'JacobData/neuronaldata/')
dir_task_details = dir_main + 'PreparedData.mat'
dir_npstorage = 'tmp/'
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


# Actual all epochs
epochs_all = (sc_madefixation, sc_choice1on, sc_choice1made, sc_transition, sc_choice2on, sc_choice2made, sc_secondaryreinforceron)
names_epochs_all = ('Fixation', 'Choice 1 on', 'Choice 1 made', '\nTransition Revealed', 'Choice 2 on', '\nChoice 2 made', 'Secondary Reinforcer')


def calc_smooth_outputlength(prewindow, postwindow):
    return (prewindow+postwindow)//smooth_step



