import Details as D

def createmasks(trialdata):

    # Specify x data used as class labels by the decoder
    x_data = trialdata.rew_coll  # Let's look at value coding

    # Then specify masks for your different conditions to compare

    # Let's compare rare versus common trials
    common_trials = trialdata.transition == 1
    rare_trials = trialdata.transition == 2

    # Put the different conditions into a single list
    conds = (common_trials, rare_trials)

    # Return x data and conditions
    return x_data, conds


if __name__ == "__main__":
    import ManagerDecoder

    maintitle = 'Decoding reward on common vs rare trials'
    ytitles = 'Accuracy (%)'
    savefolder = 'dec/rew'
    trace_names = ['Common trials', 'Rare trials']
    num_conds = len(trace_names)
    num_rows = 1
    decoder = D.decoders[1]
    minsamples = 8
    ManagerDecoder.Run(createmasks, False, num_conds, num_rows, maintitle, ytitles, savefolder, trace_names, decoder, minsamples)



