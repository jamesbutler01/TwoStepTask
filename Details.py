import Details as D
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

n_cores = 1
areas = ['misc', 'ACC', 'DLPFC', 'Caudate', 'Putamen']
areaindex= {'misc' : '', 'ACC' : 'ACC', 'DLPFC' : 'DLPFC', 'Caudate' : 'DMStr', 'Putamen' : 'DLStr'}
names = ['Charlie', 'Jacob']
dir_main = 'data/'
dir_subjs = (dir_main+'CharlieData/', dir_main+'JacobData/')
dir_qvals = [dir_subjs[i]+'qvals/qvals_sess_' for i in range(2)]
dir_spikes = (dir_main + 'CharlieData/neuronaldata/', dir_main+'JacobData/neuronaldata/')
dir_task_details = dir_main + 'PreparedData.mat'
current_drive = Path.cwd().drive
dir_local_storage = Path(f"{current_drive}/tmp/")
dir_local_storage.mkdir(parents=True, exist_ok=True)


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

# Smoothing and storing rasters
smooth_window_halfwidth = 50  # Gaussian sigma used for smoothing
smooth_step = 10  # Output resolution in ms (10 = 1 timepoint every 10 ms)
static_prewindow = 2000
static_postwindow = 2000
statictimepoints = static_prewindow + static_postwindow

static_save_dir = f'{smooth_window_halfwidth}_{smooth_step}_{static_prewindow}_{static_postwindow}/'
static_save_path = f'{dir_local_storage}/{static_save_dir}'

directory = os.path.dirname(static_save_path)
if not os.path.exists(directory):
    os.makedirs(directory)

epochs_all = (sc_madefixation, sc_choice1on, sc_choice1made, sc_transition, sc_choice2on, sc_choice2made, sc_secondaryreinforceron)
names_epochs_all = ('Fixation', 'Choice 1 on', 'Choice 1 made', '\nTransition Revealed', 'Choice 2 on', '\nChoice 2 made', 'Secondary Reinforcer')