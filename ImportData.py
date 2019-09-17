import numpy as np
import scipy.io
import scipy.ndimage
import pandas as pd
import Details as D
from pathlib import Path


class EntireArea:
    def __init__(self, area):
        self.area = area
        self.cells_index = GetCellsIndexForArea(area)
        self.n = self.cells_index.n
        self.savefolder = f'{D.dir_npstorage}{D.smooth_savedir}'
        struct = scipy.io.loadmat(D.dir_task_details)['PreparedData']

        self.behavdata = []
        for i in range(self.n):
            self.behavdata.append(GetBehavInfoForCell(struct, self.cells_index, i))

    def generatenormalisedepoch(self, cell, epoch):
        savepath = f'{self.savefolder}norm/{self.area}_{cell}_{epoch}.npy'
        savedfile = Path(savepath)
        if savedfile.is_file():
            return np.load(savedfile)
        else:
            fix = self.generateaverageepoch(cell, D.sc_madefixation)
            fixstart = D.converttimetosmoothedtrace(50)  # ms after fix made
            fixstop = D.converttimetosmoothedtrace(400)
            fix = fix[:, fixstart:fixstop]

            raw = self.generateaverageepoch(cell, epoch)
            raw -= np.mean(fix)
            raw /= np.std(fix)

            # Make the save folder if it doesn't exist
            import os
            directory = os.path.dirname(f'{self.savefolder}norm/')
            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(savepath, raw)

            return raw

    def generateaverageepoch(self, cell, epoch):
        savepath = f'{self.savefolder}{self.area}_{cell}_{epoch}.npy'
        savedfile = Path(savepath)
        if savedfile.is_file():
            return np.load(savedfile)
        else:
            def loadspikes():
                return scipy.io.loadmat(self.cells_index.spikefilelocs[cell])['tSpikes'].flatten()

            def loadstrobesbytrials():
                file = self.cells_index.strobefilelocs[cell]
                strobe_codes = scipy.io.loadmat(file)['Strobes'].flatten()
                strobe_times = scipy.io.loadmat(file)['tStrobes'].flatten()

                # Remove codes we don't care about
                strobe_times = strobe_times[strobe_codes < 41]
                strobe_codes = strobe_codes[strobe_codes < 41]

                # Remove the repeating 9's and 18's that will make indexing annoying
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

                # Now split up strobes into each trial
                strobe_codes_list = []
                strobe_times_list = []
                strobe_buffer = []
                times_buffer = []
                trial_counter = 0
                for i, (code, time) in enumerate(zip(strobe_codes, strobe_times)):
                    if code == 9:
                        times_buffer = []
                        strobe_buffer = []
                    elif code == 18:
                        if trial_counter in self.behavdata[cell].validtrials:  # Skip invalid trials
                            strobe_codes_list.append(np.array(strobe_buffer))
                            strobe_times_list.append(np.array(times_buffer))
                        trial_counter += 1
                    else:
                        strobe_buffer.append(code)
                        times_buffer.append(time)

                out = {'codes':np.array(strobe_codes_list), 'times':np.array(strobe_times_list)}

                return out

            spikes = loadspikes()
            strobes = loadstrobesbytrials()

            # So want to make one smoothed trace per trial stacked in a matrix
            def makesmoothedtrace(spikes, timepoint):
                output = np.empty(D.smooth_outputlength)
                start = timepoint - D.smooth_prewindow
                for i in range(len(output)):
                    validspikes = np.where(np.logical_and(spikes>=start-D.smooth_window_halfwidth, spikes<=start+D.smooth_window_halfwidth))[0]
                    output[i] = len(validspikes)
                    start += D.smooth_step

                return output

            # For each trial make a smoothed trace
            alltraces = []
            for tr_strobe, tr_strobe_time in zip(strobes['codes'], strobes['times']):
                timepoint = tr_strobe_time[tr_strobe==epoch]
                if len(timepoint) > 1:
                    raise Exception('Multiple strobe codes found!')
                elif len(timepoint) > 0:
                    trace = makesmoothedtrace(spikes, timepoint)
                    alltraces.append(trace)
            alltraces = np.array(alltraces)

            # Make the save folder if it doesn't exist
            import os
            directory = os.path.dirname(self.savefolder)
            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(savepath, alltraces)

            return alltraces


# Helper to extract task details for a specific cell
class GetBehavInfoForCell:
    def __init__(self, struct, cells_index, cell):
        subj = cells_index.subjindex[cell]
        sess = cells_index.sess_index[cell]

        # Make mask of valid trials in session
        self.totalnumtrials = np.array(struct[subj][0][sess][0][D.ind_totalnumtrials].flatten(), dtype=int)
        validtrialsmask = np.zeros(self.totalnumtrials)
        self.validtrials = np.array(struct[subj][0][sess][0][D.ind_goodtrialsind].flatten(), dtype=int)
        self.validtrials -= 1  # Correct matlab indexing
        validtrialsmask[self.validtrials] = 1
        self.validtrialsmask = np.array(validtrialsmask, dtype=int)

        # Trialtype codes:
        # 1 == choice level 1 and choice level 2 trial
        # 2 == FORCED choice level 1 and choice level 2 trial
        # 3 == choice level 1 and FORCED choice level 2 trial

        self.trialtype = np.array(struct[subj][0][sess][0][D.ind_trialtype].flatten(), dtype=int)
        self.transition = np.array(struct[subj][0][sess][0][D.ind_transition].flatten(), dtype=int)
        self.c1dir = np.array(struct[subj][0][sess][0][D.ind_sidechosen1].flatten(), dtype=int)
        self.c2dir = np.array(struct[subj][0][sess][0][D.ind_sidechosen2].flatten(), dtype=int)
        self.choice1 = np.array(struct[subj][0][sess][0][D.ind_picchosen1].flatten(), dtype=int)
        self.c1chosen = self.choice1
        self.c1given = np.empty(self.c1chosen.shape, dtype=int)
        for i_c1, (c1, trans) in enumerate(zip(self.c1chosen, self.transition)):
            if (c1==1 and trans==1) or (c1==2 and trans==2):
                self.c1given[i_c1] = 1
            else:
                self.c1given[i_c1] = 2

        self.choice2 = np.array(struct[subj][0][sess][0][D.ind_picchosen2].flatten(), dtype=int)
        self.rewgiven = np.array(struct[subj][0][sess][0][D.ind_rewardgiven].flatten(), dtype=int)
        self.rewgiven = 3 - self.rewgiven  # High is low in behav file :s
        self.rew_coll = np.array(self.rewgiven == 2, dtype=int)
        self.n = len(self.trialtype)

        # Make additional attributes for previous trial parameters
        def makeprevarr(past, arr):
            output = np.empty(arr.shape, dtype=int)
            output[:past] = -1
            output[past:] = arr[:-past]
            return output

        self.previousreward = makeprevarr(1, self.rewgiven)
        self.previouspreviousreward = makeprevarr(1, self.previousreward)
        self.previousrewardcoll = makeprevarr(1, self.rew_coll)
        self.previouschoice2 = makeprevarr(1, self.choice2)
        self.previouschoice1g = makeprevarr(1, self.c1given)
        self.previouschoice1c = makeprevarr(1, self.c1chosen)
        self.previoustransition = makeprevarr(1, self.transition)

        # Attribute to say whether they switched or stuck with previous choice
        def stickorswitch(arr):
            output = np.empty(arr.shape, dtype=int)
            output[0] = -1
            for i in range(1, self.n):
                if arr[i] == arr[i-1]:
                    output[i] = 0
                else:
                    output[i] = 1
            return output

        self.switchchoice1 = stickorswitch(self.choice1)
        self.switchchoice2 = stickorswitch(self.choice2)
        self.switchrew = stickorswitch(self.rewgiven)

        self.numc2repeats = np.empty(self.switchchoice1.shape, dtype=int)
        for tr in range(len(self.numc2repeats)):
            numrepeats = 0
            while self.switchchoice2[tr-numrepeats] == 0:
                numrepeats += 1
            self.numc2repeats[tr]=numrepeats

        # See if they tried to get back to c2 from previous trial at c1
        self.repeatc2atc1 = np.zeros(self.n, dtype=int)
        self.repeatc2atc1 -= 1
        for tr, (prevc2, c1) in enumerate(zip(self.previouschoice2, self.choice1)):
            if tr > 0:
                if prevc2 == 3 or prevc2 == 4:
                    if c1 == 1:
                        self.repeatc2atc1[tr] = 1
                    else:
                        self.repeatc2atc1[tr] = 0
                if prevc2 == 5 or prevc2 == 6:
                    if c1 == 2:
                        self.repeatc2atc1[tr] = 1
                    else:
                        self.repeatc2atc1[tr] = 0

        # Number of repeated 2s in a row
        # 0 = medium, low, or unexpected 2
        # 1 = first time expected 2
        # So have to filter by rew given if you just want data for 2s response
        self.num_rew2c2repeats = np.empty(self.switchchoice1.shape, dtype=int)
        numrepeats = 0
        for i_tr, (rew, switchc2) in enumerate(zip(self.rewgiven, self.switchchoice2)):

            if rew == 2 and switchc2 == 0:
                numrepeats += 1
            else:
                numrepeats = 0

            self.num_rew2c2repeats[i_tr] = numrepeats

        # Trials since last had a 2,
        # 1 = first trial without two (Unexpected 0/1)
        # 2 = second trial without two (Expected 0/1)
        self.time_since_2 = np.empty(self.switchchoice1.shape, dtype=int)
        numrepeats = 0
        for i_tr, rew in enumerate(self.rewgiven):

            if rew != 2:
                numrepeats += 1
            else:
                numrepeats = 0

            self.time_since_2[i_tr] = numrepeats

        # Find prev rewards for choice2
        self.samec2_prevrew1 = np.zeros(self.n, dtype=int)
        self.samec2_prevrew1 -= 1  #  Set all to -1
        self.samec2_prevrew2 = np.copy(self.samec2_prevrew1)
        self.samec2_prevrew3 = np.copy(self.samec2_prevrew1)
        self.samec2_prevrew4 = np.copy(self.samec2_prevrew1)
        self.samec2_prevrew5 = np.copy(self.samec2_prevrew1)
        for tr in range(self.n):
            allsamec2 = np.where(self.choice2[:tr] == self.choice2[tr])[0]
            if len(allsamec2) > 0:
                self.samec2_prevrew1[tr] = self.rewgiven[allsamec2[-1]]
            if len(allsamec2) > 1:
                self.samec2_prevrew2[tr] = self.rewgiven[allsamec2[-2]]
            if len(allsamec2) > 2:
                self.samec2_prevrew3[tr] = self.rewgiven[allsamec2[-3]]
            if len(allsamec2) > 3:
                self.samec2_prevrew4[tr] = self.rewgiven[allsamec2[-4]]
            if len(allsamec2) > 4:
                self.samec2_prevrew5[tr] = self.rewgiven[allsamec2[-5]]

        # Load q values
        self.qvals = scipy.io.loadmat(D.dir_qvals[subj]+f'{sess}')
        self.qvalnames = list(self.qvals.keys())[3:]
        for name in self.qvalnames:
            self.qvals[name] = self.qvals[name].flatten()

        self.q_chosen1 = self.qvals['ChosenQhyb']
        self.q_chosen2 = self.qvals['Chosen2Q']


# Helper class to load all cells and their file locations for an area
class GetCellsIndexForArea:
    def __init__(self, area):
        self.area = area

        spikefilelocs = []
        strobefilelocs = []
        session_index = []
        subjindex = []
        unique_session_index = []

        for subj in (0, 1):
            # First load excel mastersheet
            suffix = '_Units.xlsx'
            if subj == 0:
                prefix = 'CHARLIE_'
            else:
                prefix = 'JACOB_'
            cellindex = pd.read_excel(D.dir_subjs[subj]+prefix+area+suffix)
            folders = cellindex['Session']
            channels = cellindex['Channel']
            unit_nums = cellindex['UnitNum']
            folders = [str(folder).zfill(3) for folder in folders]

            # Use mastersheet to compile list of file locations
            strobesuffix = '_StrobesData'
            file_ext = '.mat'
            for channel, folder, unit_num, in zip(channels, folders, unit_nums):
                sess_name = prefix[0]+str(folder)
                sess_folder = D.dir_spikes[subj]+sess_name+'/'
                spikefilelocs.append(sess_folder+sess_name+'_Ch'+str(channel)+'_N'+str(unit_num)+file_ext)
                strobefilelocs.append(sess_folder+sess_name+strobesuffix+file_ext)
                subjindex.append(subj)
                session_index.append(int(folder)-1)  # Subtract 1 for 0 indexing
                unique_session_index.append(int(folder)-1+subj*30)  # Subtract 1 for 0 indexing

        self.spikefilelocs = spikefilelocs
        self.strobefilelocs = strobefilelocs
        self.subjindex = subjindex
        self.sess_index = session_index
        self.unique_sess_index = unique_session_index
        self.n = len(session_index)
        self.n_sessions = len(np.unique(unique_session_index))
