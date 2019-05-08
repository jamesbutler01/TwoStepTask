# Analyses each session (not each cell)
# This example finds the switch/stay probability for rare and common trials as a function of reward on previous trial
import ImportData
import Details as D
import numpy as np
import matplotlib.pyplot as plt
import Maths

n = 100
n_subj = 2

def wrapper(outarr, func):
    sessions_analysed = []
    counter = 0
    for area in D.areas:

        data = ImportData.EntireArea(area)

        for cell in range(data.n):
            session_uid = data.cells_index.unique_sess_index[cell]
            subj = data.cells_index.subjindex[cell]
            if session_uid in sessions_analysed:
                continue

            trialdata = data.behavdata[cell]

            func(counter, subj, trialdata, outarr)

            counter += 1
            sessions_analysed.append(session_uid)

    return outarr

# Rew coll
avgs = np.zeros((2, n, 4))

# This is the function each session will be analysed with
def overall(i_sess, subj, td, outarr):
    i = 0

    for trans in range(2):
        for rews in range(2):
            outarr[subj, i_sess, i] += 1 - np.mean(td.switchchoice1[(td.previousrewardcoll == rews) & (td.previoustransition == trans + 1)& (td.trialtype == 1)])
            i += 1

    return outarr

avgs = wrapper(avgs, overall)

# Now plot result
f, ax = plt.subplots(1, figsize=(5, 3))#, 2)
for i, monk in enumerate(('Subject J', 'Subject C')):
    # Make holding arrays
    numpnts =12
    avg = np.empty(numpnts)
    avg.fill(np.nan)
    sem = np.copy(avg)

    # Plot Common transitions
    subj_n = len(avgs[i][(avgs[i] != 0)[:, 0]])
    avg[i:6:3] = np.mean(avgs[i][(avgs[i] != 0)[:, 0]], axis=0)[0:2]
    sem[i:6:3] = Maths.sem(avgs[i][(avgs[i] != 0)[:, 0]])[0:2]
    ax.bar(range(numpnts), avg, 0.9, label=f'{monk} ({subj_n})', color=f'C{i}', edgecolor='black')
    ax.errorbar(range(numpnts), avg, sem, color=f'black', elinewidth=2)

    # Plot Rare transitions
    avg = np.empty(numpnts)
    avg.fill(np.nan)
    sem = np.copy(avg)
    avg[i+6::3] = np.mean(avgs[i][(avgs[i] != 0)[:, 0]], axis=0)[2:]
    sem[i+6::3] = Maths.sem(avgs[i][(avgs[i] != 0)[:, 0]])[2:]
    ax.bar(range(numpnts), avg, 0.9, color=f'C{i}', edgecolor='black')
    ax.errorbar(range(numpnts), avg, sem, color=f'black', elinewidth=2)

    labelpos = (0.5, 3.5, 6.5, 9.5)
    labels = ['Low/Med.', 'High'] * 2
    plt.xticks(labelpos, labels)
    plt.xlabel('Common                                     Rare')

    plt.ylim(ymax=1)
    plt.ylabel('Stay probability')

plt.legend(loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
D.savefig_makefolder('behav', 'modelbased')