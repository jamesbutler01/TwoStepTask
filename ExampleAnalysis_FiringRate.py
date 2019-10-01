import Details as D
import numpy as np
import multiprocessing.managers
import ShiftTrials
import Utils
import Plot
Shifter = ShiftTrials.Shift(2)


class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def analysecell(counter, output_frs, data, cell):

    # Print current proggress in console
    Utils.updatecounts(counter, cell, data.n)

    # Load trial data
    trialdata = data.behavdata[cell]

    # Lets look at the firing rate to different reward values
    low_reward = trialdata.rewgiven == 0
    med_reward = trialdata.rewgiven == 1
    high_reward = trialdata.rewgiven == 2

    # Put the different conditions into a single list
    conds = (low_reward, med_reward, high_reward)

    # Loop through each trial epoch that we are interested in
    for i_epoch, epoch in enumerate(D.epochs):

        # Load the y (cell firing rate) for that epoch
        y = data.generate_epoch_norm(cell, epoch)

        # Loop through each condition
        for i_cond, cond in enumerate(conds):

            # Get the average firing rate for this condition
            avg_fr = np.mean(y[cond], axis=0)

            # Store the result in the output array
            output_frs[0, i_cond, i_epoch, cell] = avg_fr


if __name__ == "__main__":
    import ManagerAnalysis

    maintitle = 'Firing rate depending on reward level'
    ytitles = 'Avg FR (norm)'
    savefolder = 'FR'
    trace_names = ('Low reward', 'Medium reward', 'High reward')
    num_conds = len(trace_names)
    num_rows = 1
    plotfunc = Plot.GeneralPlot
    ManagerAnalysis.Run(analysecell, False, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, plotfunc)
