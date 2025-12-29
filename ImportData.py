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

    def __init__(self, area, smooth_prewindow=D.smooth_prewindow,
                 smooth_postwindow=D.smooth_postwindow,
                 binSize=D.smooth_window_halfwidth,
                 res=D.smooth_step, exclude_neurons=False):
        """
        Initialize EntireArea object for a brain area.

        Parameters
        ----------
        area : str
            Brain area name
        smooth_prewindow : int, optional
            Pre-event window duration (ms)
        smooth_postwindow : int, optional
            Post-event window duration (ms)
        binSize : int, optional
            Smoothing window half-width (ms)
        res : int, optional
            Temporal resolution/step size (ms)
        exclude_neurons : bool, optional
            Whether to exclude low-firing neurons (<1 Hz)
        """
        self.area = area
        self.i_area = D.areas.index(area)
        self.cells_index = GetCellsIndexForArea(area)
        self.n = self.cells_index.n
        self.savefolder = f'{D.dir_npstorage}{D.smooth_savedir}'
        self.behavdata = []
        struct = scipy.io.loadmat(D.dir_task_details)['PreparedData']

        # Neurons with average firing rate < 1 Hz across all epochs
        exc_cells = {
            'FP': [2, 19, 23, 24, 26, 27, 30, 38, 42, 79, 85, 110, 111, 116, 119, 126, 127, 128,
                   134, 155, 157, 166, 169, 185, 186, 188, 196, 198, 205, 207, 208, 213, 217, 227,
                   228, 229, 234, 242, 250, 271],
            'ACC': [80, 88, 92, 106, 125, 187, 188, 189, 191, 192, 195, 204, 206, 207, 209, 210, 215],
            'DLPFC': [16, 48, 53, 66, 87, 135, 153, 159, 164],
            'Caudate': [24, 37, 42, 46, 54, 56, 59, 84, 85, 88, 89, 90, 91, 93, 98, 99, 101, 102,
                        103, 104, 106, 107, 109, 111],
            'Putamen': [1, 4, 10, 15, 20, 30, 34, 45, 46, 48, 53, 64, 72, 80, 85, 87, 88, 89, 94,
                        98, 103, 113, 114, 115, 118]
        }

        cell_inds = np.arange(0, self.n)
        cell_mask = np.ones(self.n, dtype=bool)
        if exclude_neurons:
            cell_mask[exc_cells[area]] = False
        self.cell_inds = cell_inds[cell_mask]
        self.n = len(self.cell_inds)
        self.uniqueSessions = np.array(self.cells_index.unique_sess_index)[cell_mask]
        self.subj = np.array(self.cells_index.subjindex)[cell_mask]

        self.res = res
        self.smooth_prewindow = smooth_prewindow
        self.smooth_postwindow = smooth_postwindow
        self.static_pre, self.static_post = 2000, 2000
        self.static_binSize = binSize
        self.numTimepoints = D.calc_smooth_outputlength(self.smooth_prewindow, self.smooth_postwindow)
        self.statictimepoints = self.static_pre + self.static_post

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
        savepath = f'{self.savefolder}{D.areaindex[self.area]}_{cell}_{epoch}.npy'
        savedfile = Path(savepath)

        if savedfile.is_file():
            # Load cached data
            allTrials = np.load(savedfile)
        else:
            # Generate from raw data
            def loadstrobesbytrials(cell):
                """Load task event strobes for each trial."""
                file = self.cells_index.strobefilelocs[cell]
                strobe_codes = scipy.io.loadmat(file)['Strobes'].flatten()
                strobe_times = scipy.io.loadmat(file)['tStrobes'].flatten()

                # Remove codes we don't care about
                strobe_times = strobe_times[strobe_codes < 41]
                strobe_codes = strobe_codes[strobe_codes < 41]

                # Remove repeating 9's and 18's
                mask = np.ones(strobe_codes.shape, dtype=bool)
                nine_count = 0
                eighteen_count = 0
                for i, code in enumerate(strobe_codes):
                    if code == 9:
                        if nine_count % 3 != 1:
                            mask[i] = False
                        nine_count += 1
                    elif code == 18:
                        if eighteen_count % 3 != 1:
                            mask[i] = False
                        eighteen_count += 1
                strobe_codes = strobe_codes[mask]
                strobe_times = strobe_times[mask]

                # Split strobes into trials
                strobe_codes_list = []
                strobe_times_list = []
                strobe_buffer = []
                times_buffer = []
                trial_counter = 0
                for i, (code, time) in enumerate(zip(strobe_codes, strobe_times)):
                    if code == 9:
                        times_buffer = []
                        strobe_buffer = []

                    strobe_buffer.append(code)
                    times_buffer.append(time)

                    if code == 18:
                        if trial_counter in self.behavdata[origcell].validtrials:
                            strobe_codes_list.append(np.array(strobe_buffer))
                            strobe_times_list.append(np.array(times_buffer))
                        trial_counter += 1

                return {'codes': np.array(strobe_codes_list), 'times': np.array(strobe_times_list)}

            spikes = scipy.io.loadmat(self.cells_index.spikefilelocs[cell])['tSpikes'].flatten()
            strobes = loadstrobesbytrials(cell)

            def makesmoothedtrace(spikes, timepoint):
                """Create smoothed peri-event time histogram."""
                output = np.zeros(self.statictimepoints + 400)

                # Create binary spike array
                for spike in spikes:
                    if spike < timepoint - self.static_pre:
                        continue
                    if spike > timepoint + self.static_post + 400:
                        break

                    time_from_event = int(spike - timepoint + self.static_pre)
                    if 0 <= time_from_event < len(output):
                        output[time_from_event] += 1

                # Smooth with Gaussian kernel
                sigma = self.static_binSize / 2.355  # Convert FWHM to sigma
                output = scipy.ndimage.gaussian_filter1d(output, sigma)

                # Downsample to desired resolution
                output = block_reduce(output[:-400], block_size=self.res, func=np.mean)

                # Select window around event
                center_idx = self.static_pre // self.res
                start_idx = center_idx - self.smooth_prewindow // self.res
                end_idx = center_idx + self.smooth_postwindow // self.res
                output = output[start_idx:end_idx]

                # Convert to Hz
                output *= 1000 / self.res

                return output

            # Process all trials
            allTrials = []
            for trial_codes, trial_times in zip(strobes['codes'], strobes['times']):
                # Find alignment event time
                if epoch in trial_codes:
                    epoch_idx = np.where(trial_codes == epoch)[0][0]
                    timepoint = trial_times[epoch_idx]
                    allTrials.append(makesmoothedtrace(spikes, timepoint))
                else:
                    # Missing event - fill with NaNs
                    allTrials.append(np.full(self.numTimepoints, np.nan))

            allTrials = np.array(allTrials)

            # Cache for future use
            np.save(savepath, allTrials)

        # Pick timepoint of interest
        start = self.static_pre
        allTrials = allTrials[:, start - self.smooth_prewindow:start + self.smooth_postwindow]

        # Downsample using self.res
        allTrials = block_reduce(allTrials, block_size=(1, self.res), func=np.mean, cval=np.mean(allTrials))

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