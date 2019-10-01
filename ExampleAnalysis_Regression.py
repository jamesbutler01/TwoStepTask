import Details as D
import numpy as np
import ShiftTrials
import Utils
import Maths
import Plot
Shifter = ShiftTrials.Shift(2)


def analysecell(counter, output_betas, data, cell):
    # Print current progress in console
    Utils.updatecounts(counter, cell, data.n)

    # Load trial data
    trialdata = data.behavdata[cell]

    # Lets do a regression for the reward values they are given
    x = trialdata.rewgiven

    # Let's compare rare versus common trials
    common_trials = trialdata.transition == 1
    rare_trials = trialdata.transition == 2

    # Put the different conditions into a single list
    conds = (common_trials, rare_trials)

    # Loop through each trial epoch that we are interested in
    for i_epoch, epoch in enumerate(D.epochs):

        # Load the y (cell firing rate) for that epoch
        y = data.generate_epoch_norm(cell, epoch)

        # Loop through each condition
        for i_cond, cond in enumerate(conds):

            # Get data from just trials for this condition
            x_cond = x[cond]
            y_cond = y[cond]

            # Do regression on x and y data for this condition and get the beta value
            beta_val = Maths.regression(x_cond, y_cond)

            # Store the result in the output array
            output_betas[0, i_cond, i_epoch, cell] = beta_val


if __name__ == "__main__":
    import ManagerAnalysis

    maintitle = 'Average betas for reward on common and rare trials'
    ytitles = 'Mean Betas'
    savefolder = 'reg'
    trace_names = ('Common trials', 'Rare trials')
    num_conds = len(trace_names)
    num_rows = 1
    plotfunc = Plot.GeneralPlot
    ManagerAnalysis.Run(analysecell, True, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, plotfunc)
