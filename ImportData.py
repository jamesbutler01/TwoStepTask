"""
Data import and neural activity extraction for multi-area recording analysis.
"""

from skimage.measure import block_reduce
import numpy as np
import scipy.io
import scipy.ndimage
import pandas as pd
import Details as D
from pathlib import Path
import os


class EntireArea:
    """
    Load and process neural data for all neurons in a brain area.

    Attributes
    ----------
    area : str
        Brain area name (e.g., 'ACC', 'DLPFC', 'Caudate', 'Putamen')
    n : int
        Number of neurons in the area
    behavdata : list
        List of GetBehavInfoForCell objects, one per neuron
    numTimepoints : int
        Number of timepoints in smoothed neural activity
    res : int
        Temporal resolution (bin size) in milliseconds
    smooth_prewindow : int
        Duration of pre-event window in milliseconds
    smooth_postwindow : int
        Duration of post-event window in milliseconds
    """

    def __init__(self, area, smooth_prewindow,
                 smooth_postwindow):
        """
        Initialize EntireArea object for a brain area.

        Parameters
        ----------
        area : str
            Brain area name
        smooth_prewindow : int
            Pre-event window duration (ms)
        smooth_postwindow : int
            Post-event window duration (ms)
        """
        self.area = area
        self.i_area = D.areas.index(area)
        self.cells_index = GetCellsIndexForArea(area)
        self.n = self.cells_index.n

        self.behavdata = []
        struct = scipy.io.loadmat(D.dir_task_details)['PreparedData']

        cell_inds = np.arange(0, self.n)
        self.cell_inds = cell_inds
        self.n = len(self.cell_inds)
        self.uniqueSessions = np.array(self.cells_index.unique_sess_index)
        self.subj = np.array(self.cells_index.subjindex)

        self.smooth_prewindow = smooth_prewindow
        self.smooth_postwindow = smooth_postwindow
        self.numTimepoints = D.calc_smooth_outputlength(self.smooth_prewindow, self.smooth_postwindow)

        for i in range(self.n):
            self.behavdata.append(GetBehavInfoForCell(struct, self.cells_index, self.cell_inds[i]))

    def generate_epoch_norm(self, cell, epoch):
        """
        Generate z-scored (normalized) neural activity for a specific epoch.

        Parameters
        ----------
        cell : int
            Cell index (0 to n-1)
        epoch : int
            Task epoch code (from Details module)

        Returns
        -------
        out : ndarray
            Z-scored firing rates (trials x timepoints)
        """
        out = self.generate_epoch_raw(cell, epoch)

        # Z-score normalization
        out -= np.nanmean(out)
        out /= np.nanstd(out)

        return out

    @staticmethod
    def generate_glm1(td):
        """
        Generate GLM design matrix with task regressors.

        Creates a design matrix with:
        - Current trial: reward, choice, transition, and interactions
        - Previous trial: choice, transition, reward, and interactions
        - Previous-previous trial: reward
        - Nuisance regressors: linear/polynomial trends, sinusoids

        Parameters
        ----------
        td : GetBehavInfoForCell
            Trial data object

        Returns
        -------
        x : ndarray
            Design matrix (trials x features), first 2 trials removed
        """
        n = td.n

        # Previous trial regressors (centered at 0)
        prevtrans = np.copy(td.previoustransition[:n]) - 1.5
        prevc1c = np.copy(td.previouschoice1c[:n]) - 1.5
        prevrew = np.copy(td.previousreward[:n]) - 1
        prevprevreward = np.copy(td.previouspreviousreward[:n]) - 1

        # Current trial regressors
        reward = np.copy(td.rewgiven[:n]) - 1
        trans = np.copy(td.transition[:n]) - 1.5
        c1chosen = np.copy(td.c1chosen[:n]) - 1.5

        # Flip signs for consistency
        c1chosen *= -1
        trans *= -1

        # Nuisance regressors
        linEffects = np.arange(n) - n / 2
        quadEffects = (linEffects ** 2) - np.mean(linEffects ** 2)
        pentEffects = (linEffects ** 3) - np.mean(linEffects ** 3)
        hexEffects = (linEffects ** 4) - np.mean(linEffects ** 4)
        sin1 = np.sin(np.arange(n) / 60)
        sin2 = np.sin(np.arange(n) / 40)
        sin3 = np.sin(np.arange(n) / 20)
        sin4 = np.sin(np.arange(n) / 10)

        # Build design matrix
        x = np.vstack((
            np.ones(n),                    # 0: intercept
            prevrew,                       # 1: previous reward
            c1chosen,                      # 2: current choice 1
            trans,                         # 3: current transition
            prevrew * trans,               # 4: prev reward × transition
            prevrew * c1chosen,            # 5: prev reward × choice
            trans * c1chosen,              # 6: transition × choice
            trans * prevrew * c1chosen,    # 7: 3-way interaction
            prevc1c,                       # 8: previous choice
            prevtrans,                     # 9: previous transition
            prevrew * prevc1c,             # 10: prev reward × prev choice
            prevrew * prevtrans,           # 11: prev reward × prev trans
            prevtrans * prevc1c,           # 12: prev trans × prev choice
            prevtrans * prevc1c * prevrew, # 13: 3-way previous interaction
            prevprevreward,                # 14: reward from t-2
            reward,                        # 15: current reward
            linEffects, quadEffects, pentEffects, hexEffects,
            sin1, sin2, sin3, sin4
        )).T

        # Skip first 2 trials (previous coefficients invalid)
        x = x[2:]

        return x

    def generate_epoch_raw(self, cell, epoch):
        """
        Generate raw (non-normalized) neural activity for a specific epoch.

        Loads smoothed spike trains aligned to task events. If not cached,
        generates from raw spike times and saves for future use.

        Parameters
        ----------
        cell : int
            Cell index (0 to n-1)
        epoch : int
            Task epoch code (from Details module)

        Returns
        -------
        allTrials : ndarray
            Raw firing rates (trials x timepoints)
        """
        origcell, cell = cell, self.cell_inds[cell]
        file = Path(f'{D.static_save_path}{D.areaindex[self.area]}_{cell}_{epoch}.npy')

        if file.is_file():
            allTrials = np.load(file)
        else:
            def makesmoothedtrace(spikes, timepoint):
                output = np.zeros(D.statictimepoints + 400)
                theseTs = spikes - timepoint + (D.static_prewindow + 200);
                theseTs = theseTs[
                    (theseTs > -1) & (theseTs < D.statictimepoints + 400)];  # Exclude spikes out of range
                output[theseTs] = 1;  # Add spikes to raster

                return output

            trial_timings_df = pd.read_parquet(f'data/{self.area}/trial_timings/cell_{cell}.parquet')
            spikes = np.load(f'data/{self.area}/spikes/cell_{cell}.npy')

            # For each trial make a smoothed trace
            alltraces = []

            for trial_num, trial_data in trial_timings_df.groupby('trial_number'):
                tr_strobe = trial_data['codes'].values
                tr_strobe_time = trial_data['times'].values

                timepoint = tr_strobe_time[tr_strobe == epoch]
                if len(timepoint) > 1:
                    raise Exception('Multiple strobe codes found!')
                elif len(timepoint) > 0:
                    trace = makesmoothedtrace(spikes, timepoint[0])
                    alltraces.append(trace)
            allTrials = np.array(alltraces)

            # Smooth traces
            allTrials = scipy.ndimage.gaussian_filter1d(allTrials, D.smooth_window_halfwidth, axis=1)[:, 200:-200]

            np.save(file, allTrials)
            print('raster generated and saved to', file)

        # Pick timepoint of interest
        start = D.static_prewindow
        allTrials = allTrials[:, start - self.smooth_prewindow:start + self.smooth_postwindow]

        # Downsample to the specified resolution
        allTrials = block_reduce(allTrials, block_size=(1, D.smooth_step), func=np.mean, cval=np.mean(allTrials))

        return allTrials



class GetBehavInfoForCell:
    """
    Behavioral data for a single neuron's recording session.

    Loads trial-by-trial behavioral data including choices, transitions,
    rewards, and Q-values.

    Attributes (only those used in Fig2-6)
    ----------
    n : int
        Number of trials
    c1chosen : ndarray
        Choice 1 selected (1 or 2)
    c1dir : ndarray
        Direction of choice 1 (left/right)
    c2dir : ndarray
        Direction of choice 2 (left/right)
    choice2 : ndarray
        Choice 2 selected (stimulus ID)
    transition : ndarray
        Transition type (1=common, 2=rare)
    rewgiven : ndarray
        Reward received (0, 1, or 2)
    previouschoice1c : ndarray
        Previous trial's choice 1
    previoustransition : ndarray
        Previous trial's transition
    previousreward : ndarray
        Previous trial's reward
    previouspreviousreward : ndarray
        Reward from t-2
    qvals : dict
        Q-values from model (contains multiple Q-value types)
    q_chosen1hyb : ndarray
        Hybrid model Q-value for chosen option
    q_unchosen1hyb : ndarray
        Hybrid model Q-value for unchosen option
    """

    def __init__(self, struct, cells_index, cell):
        """
        Initialize behavioral data for a cell's session.

        Parameters
        ----------
        struct : ndarray
            MATLAB structure with behavioral data
        cells_index : GetCellsIndexForArea
            Cell index information
        cell : int
            Cell index
        """
        subj = cells_index.subjindex[cell]
        sess = cells_index.sess_index[cell]

        # Load valid trials
        self.totalnumtrials = int(struct[subj][0][sess][0][D.ind_totalnumtrials].flatten())
        validtrialsmask = np.zeros(self.totalnumtrials, dtype=bool)
        self.validtrials = np.array(struct[subj][0][sess][0][D.ind_goodtrialsind].flatten(), dtype=int)
        self.validtrials -= 1  # Correct MATLAB 1-indexing
        validtrialsmask[self.validtrials] = True
        self.validtrialsmask = np.array(validtrialsmask, dtype=int)

        # Load trial data
        self.trialtype = np.array(struct[subj][0][sess][0][D.ind_trialtype].flatten(), dtype=int)
        self.transition = np.array(struct[subj][0][sess][0][D.ind_transition].flatten(), dtype=int)
        self.c1dir = np.array(struct[subj][0][sess][0][D.ind_sidechosen1].flatten(), dtype=int)
        self.c2dir = np.array(struct[subj][0][sess][0][D.ind_sidechosen2].flatten(), dtype=int)
        self.c1chosen = np.array(struct[subj][0][sess][0][D.ind_picchosen1].flatten(), dtype=int)
        self.choice2 = np.array(struct[subj][0][sess][0][D.ind_picchosen2].flatten(), dtype=int)
        self.rewgiven = np.array(struct[subj][0][sess][0][D.ind_rewardgiven].flatten(), dtype=int)
        self.rewgiven = 3 - self.rewgiven  # Flip encoding
        self.n = len(self.trialtype)

        # Create previous trial attributes
        def makeprevarr(past, arr, datatype=int):
            """Shift array to get previous trial values."""
            output = np.empty(arr.shape, dtype=datatype)
            output[:past] = 0
            output[past:] = arr[:-past]
            return output

        self.previousreward = makeprevarr(1, self.rewgiven)
        self.previouspreviousreward = makeprevarr(1, self.previousreward)
        self.previouschoice1c = makeprevarr(1, self.c1chosen)
        self.previoustransition = makeprevarr(1, self.transition)

        # Load Q-values
        self.qvals = scipy.io.loadmat(D.dir_qvals[subj] + f'{sess}')
        self.qvalnames = list(self.qvals.keys())[3:]
        for name in self.qvalnames:
            self.qvals[name] = self.qvals[name].flatten()

        self.q_chosen1hyb = self.qvals['ChosenQhyb']
        self.q_unchosen1hyb = self.qvals['UnchosenQhyb']


class GetCellsIndexForArea:
    """
    File paths and metadata for all neurons in a brain area.

    Attributes
    ----------
    area : str
        Brain area name
    n : int
        Number of neurons
    spikefilelocs : list
        Paths to spike time files
    strobefilelocs : list
        Paths to task event files
    subjindex : list
        Subject index for each neuron (0 or 1)
    sess_index : list
        Session index for each neuron
    unique_sess_index : list
        Unique session identifier across subjects
    """

    def __init__(self, area):
        """
        Initialize cell index for an area.

        Parameters
        ----------
        area : str
            Brain area name
        """
        self.area = area

        spikefilelocs = []
        strobefilelocs = []
        session_index = []
        subjindex = []
        unique_session_index = []

        for subj in (0, 1):
            # Load neuron list from Excel
            suffix = '_Units.xlsx'
            prefix = 'CHARLIE_' if subj == 0 else 'JACOB_'
            cellindex = pd.read_excel(D.dir_subjs[subj] + prefix + D.areaindex[area] + suffix)
            folders = cellindex['Session']
            channels = cellindex['Channel']
            unit_nums = cellindex['UnitNum']
            folders = [str(folder).zfill(3) for folder in folders]

            # Compile file paths
            strobesuffix = '_StrobesData'
            file_ext = '.mat'
            for channel, folder, unit_num in zip(channels, folders, unit_nums):
                sess_name = prefix[0] + str(folder)
                sess_folder = D.dir_spikes[subj] + sess_name + '/'
                spikefilelocs.append(sess_folder + sess_name + '_Ch' + str(channel) + '_N' + str(unit_num) + file_ext)
                strobefilelocs.append(sess_folder + sess_name + strobesuffix + file_ext)
                subjindex.append(subj)
                session_index.append(int(folder) - 1)  # 0-indexed
                unique_session_index.append(int(folder) - 1 + subj * 30)

        self.spikefilelocs = spikefilelocs
        self.strobefilelocs = strobefilelocs
        self.subjindex = subjindex
        self.sess_index = session_index
        self.unique_sess_index = unique_session_index
        self.n = len(session_index)
        self.n_sessions = len(np.unique(unique_session_index))