import Details as D
import numpy as np


def getarrs(num_cond, num_rows):
    all_avgs = np.zeros((D.numareas, num_rows, num_cond, D.numtrialepochs, D.num_timepoints))
    all_sem = np.zeros((D.numareas, num_rows, num_cond, D.numtrialepochs, D.num_timepoints))
    sigclusters = np.zeros((D.numareas, num_rows, D.numtrialepochs, D.num_timepoints))

    return all_avgs, all_sem, sigclusters


def updatecounts(counter, cell, n):
    counter[cell] = 1
    if int(sum(counter)) % 20 == 0:
        print(f'{int(sum(counter))}/{n}')