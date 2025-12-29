import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import Details as D
import Maths
import ImportData
import Plot

# Configuration
FIGURE_DPI = 200
FONT_SIZE_LABEL = 14
FONT_SIZE_AXIS = 13
FONT_SIZE_LETTER = 16
FONT_SIZE_TICK = 12
LINE_WIDTH = 2.5

# Set matplotlib font sizes
matplotlib.rc('xtick', labelsize=FONT_SIZE_TICK)
matplotlib.rc('ytick', labelsize=FONT_SIZE_TICK)

# Analysis parameters
EPOCHS = (D.sc_transition, D.sc_choice2on, D.sc_secondaryreinforceron)
EPOCH_INDICES = [0, 100, 200, 300]
NUM_TIMEPOINTS = EPOCH_INDICES[-1]

# %% Main Analysis Loop
all_cpds = []
all_betas = []

for area in D.areas[1:]:
    data = ImportData.EntireArea(area, 0, 1000, exclude_neurons=False)
    print(f"Processing area: {area}")

    # Initialize arrays
    num_neurons = data.n
    cpds = Maths.nans((2, num_neurons, NUM_TIMEPOINTS))  # 2 coefficients: transition and interaction
    betas = Maths.nans((15, num_neurons, NUM_TIMEPOINTS))

    for cell_idx in range(num_neurons):
        print(f"{area} - Cell {cell_idx + 1}/{num_neurons}")

        trial_data = data.behavdata[cell_idx]
        concatenated_response = np.empty((trial_data.n - 2, NUM_TIMEPOINTS))

        # Process each epoch
        for epoch_idx, epoch in enumerate(EPOCHS):
            epoch_normalized = data.generate_epoch_norm(cell_idx, epoch)
            epoch_normalized = epoch_normalized[2:]  # Skip first trial

            # Fill concatenated response array
            start_idx = EPOCH_INDICES[epoch_idx]
            end_idx = EPOCH_INDICES[epoch_idx + 1]
            duration = end_idx - start_idx
            concatenated_response[:, start_idx:end_idx] = epoch_normalized[:, 0:duration]

        # Generate GLM design matrix
        design_matrix = data.generate_glm1(trial_data)

        # Compute regression coefficients
        betas[:, cell_idx] = Maths.reg(design_matrix, concatenated_response)[1:16]

        # Compute CPD for transition and interaction
        covariate_indices = (3, 4)
        cpds[0:2, cell_idx] = Maths.cpd(design_matrix, concatenated_response, covariate_indices)

    # Store results for this area
    all_cpds.append(cpds)
    all_betas.append(betas)

# %% Create Figure
fig = plt.figure(figsize=(12, 7), dpi=FIGURE_DPI)

# Define grid layouts
height_ratios_main = [1, 0.55, 1, 0.5, 1]
width_ratios_main = [1, 0.4, 1, 0.4, 0.65]
width_ratios_cpd = [1, 0.2, 1]
width_ratios_bottom = [1, 0.5, 1, 0.5, 0.6]

gs_cpd = plt.GridSpec(len(height_ratios_main), len(width_ratios_cpd),
                      height_ratios=height_ratios_main, width_ratios=width_ratios_cpd,
                      hspace=0, wspace=0)
gs_mid = plt.GridSpec(len(height_ratios_main), len(width_ratios_main),
                      height_ratios=height_ratios_main, width_ratios=width_ratios_main,
                      hspace=0, wspace=0)
gs_bottom = plt.GridSpec(len(height_ratios_main), len(width_ratios_bottom),
                         height_ratios=height_ratios_main, width_ratios=width_ratios_bottom,
                         hspace=0, wspace=0)

# Create subplots
ax_cpd = fig.add_subplot(gs_cpd[0, 0])
ax_bar = fig.add_subplot(gs_cpd[0, 2])  # Hidden panel
ax_interaction = fig.add_subplot(gs_mid[2, 0])
ax_bar_interaction = fig.add_subplot(gs_mid[2, 2])  # Hidden panel
ax_correlation = fig.add_subplot(gs_mid[2, 4])

# Hide panels B and D
ax_bar.axis('off')
ax_bar_interaction.axis('off')

all_axes = [ax_cpd, ax_interaction, ax_correlation]

# Format axes
for idx, ax in enumerate(all_axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.get_label().set_fontsize(FONT_SIZE_AXIS)
    ax.yaxis.get_label().set_fontsize(FONT_SIZE_AXIS)

    # Panel letters: A, C, E (skipping B and D)
    panel_indices = [0, 2, 4]
    x_positions = [-0.1, -0.15, -0.35]
    y_positions = [1.15, 1.15, 1.15]
    Plot.let(ax, panel_indices[idx], x=x_positions[idx], y=y_positions[idx],
             fontsize=FONT_SIZE_LETTER)

# Add horizontal zero lines
ax_cpd.axhline(0, c='k', lw=0.5, zorder=-10)

# %% Plot CPD Over Time

for iarea, area in enumerate(D.areas[1:]):
    cpds = all_cpds[iarea]
    betas = all_betas[iarea]

    for icoeff in range(2):
        cpd_concat = cpds[icoeff]

        labels = ['', 'Previous reward * Transition']
        axes = [ax_cpd, ax_interaction]
        ax = axes[icoeff]

        # Plot CPD for each epoch
        for epoch_idx, epoch_end in enumerate(EPOCH_INDICES):
            if epoch_idx == 0:
                continue

            epoch_start = EPOCH_INDICES[epoch_idx - 1]
            cpd_data = cpd_concat[:, epoch_start:epoch_end]

            if epoch_idx == 1:
                Plot.AvgSem(cpd_data, ax, xrange=range(epoch_start, epoch_end),
                            c=f'C{iarea}', label=area, showleg=False,
                            zorder=20 - iarea * 3, alpha=0.3, lw=LINE_WIDTH)
            else:
                Plot.AvgSem(cpd_data, ax, xrange=range(epoch_start, epoch_end),
                            c=f'C{iarea}', showleg=False,
                            zorder=20 - iarea * 3, alpha=0.3, lw=LINE_WIDTH)

        # Format axes
        [ax.axvline(t, c='grey', lw=0.5, zorder=-100) for t in EPOCH_INDICES[1:-1]]

        if icoeff == 0:
            ax.legend(loc='upper right', bbox_to_anchor=[1, 1.22],
                      framealpha=0.5, fontsize=FONT_SIZE_AXIS - 3, ncol=2)
            ax.set_xticks(EPOCH_INDICES[:-1])
            ax.set_xticklabels(('Transition', 'Choice 2', 'Feedback'), ha='left')
            ax.set_xlim(0, 300)
            ax.set_ylim(0.2, 1.25)
        else:
            Plot.set_xlim(data, ax, 0, 500, res=250)
            ax.set_xlabel('Time (ms) post transition')
            ax.set_xlim(0, 100)
            ax.set_ylim(0.12, 0.65)

        ax.set_title(labels[icoeff])
        ax.set_ylabel('CPD (%)')

    # Correlation scatter plot (Panel E) - only for ACC
    if area == 'ACC':
        # Get betas for transition and reward
        beta_transition = betas[2, :, :100]  # Transition: 0-100ms (neurons x time)
        beta_reward = betas[14, :, 200:]  # Reward: 200-300ms (neurons x time)

        # Compute correlation matrix across all timepoint pairs
        # Stack transition and reward betas: (neurons x time_trans+time_rew)
        combined = np.hstack([beta_transition, beta_reward])

        # Compute correlation matrix: (time_trans+time_rew x time_trans+time_rew)
        corr_matrix = np.corrcoef(combined.T)

        # Extract cross-correlations between transition and reward timepoints
        n_trans_tp = beta_transition.shape[1]
        n_rew_tp = beta_reward.shape[1]
        cross_corr = corr_matrix[:n_trans_tp, n_trans_tp:]

        # Find maximum correlation
        max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        best_trans_tp = max_idx[0]
        best_rew_tp = max_idx[1]
        max_corr = cross_corr[best_trans_tp, best_rew_tp]

        # Plot correlation at peak timepoint
        btrans = beta_transition[:, best_trans_tp]
        brew = beta_reward[:, best_rew_tp]

        print(f"ACC: Maximum correlation r={max_corr:.4f} at "
              f"transition_tp={best_trans_tp}, reward_tp={best_rew_tp + 200}")

        ax_correlation.scatter(brew, btrans, c='k')

        # Fit regression line
        import statsmodels.api as sm

        results = sm.OLS(brew, sm.add_constant(btrans)).fit()

        x = np.array([-1.2, 0.9])
        ax_correlation.plot(x, x * results.params[1] + results.params[0],
                            c='red', label=f'r={np.round(max_corr, 4)}', lw=3)

        ax_correlation.set_ylabel('Transition\ncoefficients', fontsize=FONT_SIZE_AXIS)
        ax_correlation.set_xlabel('Feedback coefficients', fontsize=FONT_SIZE_AXIS)
        ax_correlation.legend(fontsize=FONT_SIZE_AXIS - 2, loc='upper left',
                              bbox_to_anchor=[0, 1.2])
        ax_correlation.tick_params(labelsize=FONT_SIZE_AXIS)

plt.tight_layout()
plt.show()