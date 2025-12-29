from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.ndimage
import pandas as pd
import Details as D
from pathlib import Path


#peak reward tps = 420, 430, 260, 240  or acc, dlpfc,caudate putamen

class EntireArea:
    def __init__(self, area, smooth_prewindow=D.smooth_prewindow, smooth_postwindow=D.smooth_postwindow, binSize=D.smooth_window_halfwidth, res=D.smooth_step, exclude_neurons=False):
        self.area = area
        self.i_area = D.areas.index(area)
        self.cells_index = GetCellsIndexForArea(area)
        self.n = self.cells_index.n
        self.savefolder = f'{D.dir_npstorage}{D.static_savedir}'
        self.behavdata = []
        struct = scipy.io.loadmat(D.dir_task_details)['PreparedData']
        
        # These cells all add average firing rate across all epochs of < 1 Hz
        exc_cells = {'FP': [2, 19, 23, 24, 26, 27, 30, 38, 42, 79, 85, 110, 111, 116, 119, 126, 127, 128,
         134,155, 157, 166, 169, 185, 186, 188, 196, 198, 205, 207, 208, 213, 217, 227, 228, 229,
         234, 242, 250, 271], 
         'ACC' :[ 80, 88, 92, 106, 125, 187, 188, 189, 191, 192, 195, 204, 206, 207, 209, 210, 215,],
        'DLPFC' : [ 16, 48, 53, 66, 87, 135, 153, 159, 164,],
        'Caudate' : [ 24, 37, 42, 46, 54, 56, 59, 84, 85, 88, 89, 90, 91, 93, 98, 99, 101, 102,
         103, 104, 106, 107, 109, 111],
        'Putamen' : [ 1,  4, 10, 15, 20, 30, 34, 45, 46, 48, 53, 64, 72, 80, 85, 87, 88, 89, 94, 98, 103, 113, 114, 115, 118]}
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
        out = self.generate_epoch_raw(cell, epoch)[0]
        
        out -= np.nanmean(out)
        out /= np.nanstd(out)
        
        fr_pe = np.mean(out[:, self.timetopoint(D.pop_window_start):self.timetopoint(D.pop_window_stop)], axis=1)  # Average FR just after the event

        return out, fr_pe
    
    @staticmethod
    def generate_glm1(td, add_pe2=False):
        n = td.n
        prevtrans=np.copy(td.previoustransition[:n])-1.5
        prevc1c=np.copy(td.previouschoice1c[:n])-1.5
        prevrew=np.copy(td.previousreward[:n])-1
        prevprevreward=np.copy(td.previouspreviousreward[:n])-1
        reward=np.copy(td.rewgiven[:n])-1
        
        trans=np.copy(td.transition[:n])-1.5
        c1chosen=np.copy(td.c1chosen[:n])-1.5
        
        c1chosen *= -1  
        trans *= -1  
      
        linEffects = np.arange(n) - n/2
        quadEffects = (linEffects ** 2)
        quadEffects -= np.mean(quadEffects)
        pentEffects = (linEffects ** 3)
        pentEffects -= np.mean(pentEffects)
        hexEffects = (linEffects ** 4)
        hexEffects -= np.mean(hexEffects)
        sin1 = np.sin(np.arange(n)/60)
        sin2 = np.sin(np.arange(n)/40)
        sin3 = np.sin(np.arange(n)/20)
        sin4 = np.sin(np.arange(n)/10)
        
        #%
        if not add_pe2:
            x = np.vstack((
                    np.ones(n), # 0
                    prevrew,  # 1
                    c1chosen, # 2
                    trans, # 3
                    prevrew*trans, # 4
                    prevrew*c1chosen, # 5
                    trans*c1chosen, # 6
                    trans*prevrew*c1chosen,  # 7
                    prevc1c, # 8
                    prevtrans, # 9
                    prevrew*prevc1c,  # 10
                    prevrew*prevtrans, # 11
                    prevtrans*prevc1c, # 12
                    prevtrans*prevc1c*prevrew,  # 13
                    prevprevreward, # 14
                    reward,  # 15
                    linEffects, quadEffects, pentEffects, hexEffects, sin1, sin2, sin3, sin4
                )).T
        else:
            x = np.vstack((td.qvals['PE2'], np.ones(n), prevrew, c1chosen, trans,
                       prevrew*trans, prevrew*c1chosen, trans*c1chosen,
                       trans*prevrew*c1chosen, prevc1c, prevtrans,
                       prevrew*prevc1c, prevrew*prevtrans, prevtrans*prevc1c,
                       prevtrans*prevc1c*prevrew, prevprevreward, reward, 
                        linEffects, quadEffects, pentEffects, hexEffects, sin1, sin2, sin3, sin4)).T
        inds = [15, 1, 14]
        #  Skip first trial as previous coefficients are invalid
        x = x[2:]
        
        return x

    def generate_epoch_raw(self, cell, epoch):
        origcell, cell = cell, self.cell_inds[cell]
        savepath = f'{self.savefolder}{D.areaindex[self.area]}_{cell}_{epoch}.npy'
        savedfile = Path(savepath)
        if savedfile.is_file():
            allTrials = np.load(savedfile)

        else:

            def loadstrobesbytrials(cell):
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
                    
                    strobe_buffer.append(code)
                    times_buffer.append(time)

                    if code == 18:
                        if trial_counter in self.behavdata[origcell].validtrials:  # Skip invalid trials
                            strobe_codes_list.append(np.array(strobe_buffer))
                            strobe_times_list.append(np.array(times_buffer))
                        trial_counter += 1

                out = {'codes':np.array(strobe_codes_list), 'times':np.array(strobe_times_list)}

                return out

            spikes = scipy.io.loadmat(self.cells_index.spikefilelocs[cell])['tSpikes'].flatten()
            strobes = loadstrobesbytrials(cell)

            # So want to make one smoothed trace per trial stacked in a matrix
            def makesmoothedtrace(spikes, timepoint):
                output = np.zeros(self.statictimepoints+400)
                start = timepoint - D.smooth_prewindow
                theseTs = spikes - timepoint + (self.static_pre + 200); 
                theseTs = theseTs[(theseTs > -1) & (theseTs < self.statictimepoints+400)];  # Exclude spikes out of range
                output[theseTs] = 1;  # Add spikes to raster

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
            allTrials = np.array(alltraces)
            
            # Smooth traces
            allTrials = scipy.ndimage.gaussian_filter1d(allTrials, self.static_binSize, axis=1)[:, 200:-200]

            # Make the save folder if it doesn't exist
            import os
            directory = os.path.dirname(self.savefolder)
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save(savepath, allTrials)
            print('raster generated and saved to', savepath)
            
        # Pick timepoint of interest
        start = self.static_pre
        allTrials = allTrials[:, start-self.smooth_prewindow:start+self.smooth_postwindow]
        
        # Downsample using self.res
        allTrials = block_reduce(allTrials, block_size=(1,self.res), func=np.mean, cval=np.mean(allTrials))
        fr_pe = np.mean(allTrials[:, D.pop_window_start:D.pop_window_stop], axis=1)  # Average FR just after the event
        
        return allTrials, fr_pe
    
    # Convert time to point
    def timetopoint(self, tp):
        pnt = (tp+self.smooth_prewindow)//self.res
        if pnt <0:
            raise ValueError('out of bounds')
        return pnt
    
    # Convert time to point
    def pointtotime(self, pnt):
        tp = (pnt*self.res)-self.smooth_prewindow
        return tp


# Helper to extract task details for a specific cell
class GetBehavInfoForCell:
    def __init__(self, struct, cells_index, cell):
        subj = cells_index.subjindex[cell]
        sess = cells_index.sess_index[cell]

        # Make mask of valid trials in session
        self.totalnumtrials = int(struct[subj][0][sess][0][D.ind_totalnumtrials].flatten())
        validtrialsmask = np.zeros(self.totalnumtrials, dtype=bool)
        self.validtrials = np.array(struct[subj][0][sess][0][D.ind_goodtrialsind].flatten(), dtype=int)
        self.validtrials -= 1  # Correct matlab indexing
        validtrialsmask[self.validtrials] = True
        self.validtrialsmask = np.array(validtrialsmask, dtype=int)

        # Trialtype codes:
        # 1 == choice level 1 and choice level 2 trial
        # 2 == forced choice level 1 and choice level 2 trial
        # 3 == choice level 1 and forced choice level 2 trial

        self.trialtype = np.array(struct[subj][0][sess][0][D.ind_trialtype].flatten(), dtype=int)
        self.c2rewvalues = np.array(struct[subj][0][sess][0][D.ind_c2values], dtype=int)
        self.c2rewvalues = 3 - self.c2rewvalues  # High is low in behav file :s
        self.transition = np.array(struct[subj][0][sess][0][D.ind_transition].flatten(), dtype=int)
        self.c1dir = np.array(struct[subj][0][sess][0][D.ind_sidechosen1].flatten(), dtype=int)
        self.c2dir = np.array(struct[subj][0][sess][0][D.ind_sidechosen2].flatten(), dtype=int)
        self.transition = np.array(struct[subj][0][sess][0][D.ind_transition].flatten(), dtype=int)
        self.c1chosen = np.array(struct[subj][0][sess][0][D.ind_picchosen1].flatten(), dtype=int)
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
        def makeprevarr(past, arr, datatype=int):
            output = np.empty(arr.shape, dtype=datatype)
            output[:past] = 0
            output[past:] = arr[:-past]
            return output

        # Make additional attributes for previous trial parameters
        def makenextarr(next, arr, datatype=int):
            output = np.empty(arr.shape, dtype=datatype)
            output[next:] = 0
            output[:next] = arr[:-next]
            return output

        self.previousreward = makeprevarr(1, self.rewgiven)
        self.previouspreviousreward = makeprevarr(1, self.previousreward)
        self.previouspreviouspreviousreward = makeprevarr(1, self.previouspreviousreward)
        self.previousrewardcoll = makeprevarr(1, self.rew_coll)
        self.previouschoice2 = makeprevarr(1, self.choice2)
        self.previouschoice1g = makeprevarr(1, self.c1given)
        self.previouschoice1c = makeprevarr(1, self.c1chosen)
        self.previoustransition = makeprevarr(1, self.transition)
        self.previousc1dir = makeprevarr(1, self.c1dir)
        self.previousc2dir = makeprevarr(1, self.c2dir)
        
        # self.nextchoice1c = makenextarr(1, self.c1chosen)

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

        self.switchchoice1c = stickorswitch(self.c1chosen)
        self.switchchoice1g = stickorswitch(self.c1given)
        self.previousswitchchoice1g = makeprevarr(1, self.switchchoice1g)

        self.switchchoice2 = stickorswitch(self.choice2)
        self.switchrew = stickorswitch(self.rewgiven)


        # Number of repeated 2s in a row
        # 0 = medium, low, or unexpected 2
        # 1 = expected 2
        # So have to filter by rew given if you just want data for 2s response
        self.num_rew2c2repeats = np.empty(self.switchchoice1c.shape, dtype=int)
        numrepeats = 0
        for i_tr, (rew, switchc2) in enumerate(zip(self.rewgiven, self.switchchoice2)):

            if rew == 2 and switchc2 == 0:
                numrepeats += 1
            else:
                numrepeats = 0

            self.num_rew2c2repeats[i_tr] = numrepeats
        self.previous_num_rew2c2repeats = makeprevarr(1, self.num_rew2c2repeats)

        # Trials since last had a 2
        # 1 = first trial without two (Unexpected 0/1)
        # 2 = second trial without two (Expected 0/1)
        self.time_since_2 = np.empty(self.switchchoice1c.shape, dtype=int)
        numrepeats = 0
        for i_tr, rew in enumerate(self.rewgiven):

            if rew != 2:
                numrepeats += 1
            else:
                numrepeats = 0

            self.time_since_2[i_tr] = numrepeats

        # See if they tried to get back to c2 from previous trial at c1
        self.repeatc2atc1 = np.zeros(self.n, dtype=int)
        self.repeatc2atc1 -= 1
        for tr, (prevc2, c1) in enumerate(zip(self.previouschoice2, self.c1chosen)):
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

        self.q_chosen1hyb = self.qvals['ChosenQhyb']
        self.q_unchosen1hyb = self.qvals['UnchosenQhyb']
        self.q_chosen1mb = self.qvals['ChosenQmb']
        self.q_unchosen1mb = self.qvals['UnchosenQmb']
        self.q_chosen1mf = self.qvals['ChosenQmf']
        self.q_unchosen1mf = self.qvals['UnchosenQmf']
        self.q_chosen2 = self.qvals['Chosen2Q']
        self.q_unchosen2 = self.qvals['Unchosen2Q']

        self.previouschoice2q = makeprevarr(1, self.q_chosen2, datatype=float)
        self.previouspreviouschoice2q = makeprevarr(2, self.q_chosen2, datatype=float)



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
            cellindex = pd.read_excel(D.dir_subjs[subj]+prefix+D.areaindex[area]+suffix)
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


# Load 5x5 list of all pairs of neurons recorded simultaneously
def LoadPairWiseIndex():
    session_areas = [[] for _ in range(57)]
    session_cells = [[] for _ in range(57)]

    for i_area, area in enumerate(D.areas):

        data = EntireArea(area)

        for cell in range(data.n):
            session_uid = data.uniqueSessions[cell]
            session_areas[session_uid].append(i_area)
            session_cells[session_uid].append(cell)

    # 5x5 to store all pairs of neurons
    pairs = [[[] for _ in range(D.numareas)] for _ in range(D.numareas)]
    pairs_n = np.zeros((5, 5))
    for i_sess, (areas, cells) in enumerate(zip(session_areas, session_cells)):
        areas = np.array(areas)
        cells = np.array(cells)

        for i_area in range(D.numareas):
            c_1 = cells[areas == i_area]
            for j_area in range(D.numareas):
                c_2 = cells[areas == j_area]

                # Across areas, all pairs count
                if i_area != j_area:
                    for cell_1 in c_1:
                        for cell_2 in c_2:
                            pairs[i_area][j_area].append([[i_area, cell_1], [j_area, cell_2]])
                            pairs_n[i_area][j_area] += 1
                else:
                    c_2 = list(c_2)
                    for i_cell_1, cell_1 in enumerate(c_1):
                        # Remove cell we are currently pairing
                        del (c_2[0])
                        # Pair it with all the remaining
                        for cell_2 in c_2:
                            pairs[i_area][j_area].append([[i_area, cell_1], [j_area, cell_2]])
                            pairs_n[i_area][j_area] += 1

    return pairs

