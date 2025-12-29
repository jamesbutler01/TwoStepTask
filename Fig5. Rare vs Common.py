import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import Details as D
import Maths
import ImportData
import Plot

# Configuration
FIGURE_DPI = 400
FONT_SIZE_AXIS = 13
FONT_SIZE_LETTER = 14
LINE_WIDTH = 2.5

# Set matplotlib font sizes
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

# Analysis parameters
EPOCHS = [D.sc_choice1made]

# %% Main Analysis Loop
all_betas = []

for area in D.areas[1:]:
    data = ImportData.EntireArea(area, 0, 500, exclude_neurons=False)
    print(f"Processing area: {area}, neurons: {data.n}")

    # Initialize arrays
    num_neurons = data.n
    betas = np.empty((3, 2, 2, num_neurons, data.numTimepoints))

    for cell_idx in range(num_neurons):
        print(f"{area} - Cell {cell_idx + 1}/{num_neurons}")

        trial_data = data.behavdata[cell_idx]
        n = trial_data.n

        # Prepare nuisance regressors
        lin_effects = np.arange(n) - n / 2
        quad_effects = (lin_effects ** 2) - np.mean(lin_effects ** 2)
        pent_effects = (lin_effects ** 3) - np.mean(lin_effects ** 3)
        hex_effects = (lin_effects ** 4) - np.mean(lin_effects ** 4)
        sin1 = np.sin(np.arange(n) / 60)
        sin2 = np.sin(np.arange(n) / 40)
        sin3 = np.sin(np.arange(n) / 20)
        sin4 = np.sin(np.arange(n) / 10)

        # Common/Rare split analysis
        for epoch_idx, epoch in enumerate(EPOCHS):
            response = data.generate_epoch_norm(cell_idx, epoch)

            # Build design matrix with Q-values
            design_matrix = np.vstack((
                np.ones(trial_data.n),
                trial_data.q_chosen1hyb,
                trial_data.q_unchosen1hyb,
                lin_effects, quad_effects, pent_effects, hex_effects,
                sin1, sin2, sin3, sin4
            )).T

            # Separate by transition type (common vs rare)
            common_mask = trial_data.transition == 1
            rare_mask = trial_data.transition == 2

            # Regression for common trials
            betas[epoch_idx, 0, :, cell_idx] = Maths.reg(
                design_matrix[common_mask],
                response[common_mask]
            )[1:3]

            # Regression for rare trials
            betas[epoch_idx, 1, :, cell_idx] = Maths.reg(
                design_matrix[rare_mask],
                response[rare_mask]
            )[1:3]

        # Reward regressor (for context, not currently used in plots)
        design_matrix_glm = data.generate_glm1(trial_data)
        response_reward = data.generate_epoch_norm(cell_idx, D.sc_secondaryreinforceron)
        response_reward = response_reward[2:]
        betas[2, 0, 0, cell_idx] = Maths.reg(design_matrix_glm, response_reward)[15]

    all_betas.append(betas)

# %% Create Figure
fig, axes = plt.subplots(4, 2, figsize=(6, 7), dpi=FIGURE_DPI)
axes = axes.flatten()

# Format axes
for idx in range(len(axes)):
    Plot.let(axes[idx], idx, [-0.0751, -0.35][idx % 2 == 0], fontsize=FONT_SIZE_LETTER)
    axes[idx].spines['right'].set_visible(False)
    axes[idx].spines['top'].set_visible(False)

# Set titles
axes[0].set_title('Common trials')
axes[1].set_title('Rare trials')

# Set x-labels for bottom row
for ax in axes[-2:]:
    ax.set_xlabel('Time (ms) post choice 1 made\n(transition revealed)')

# %% Plot Results
for iarea, area in enumerate(D.areas[1:]):
    betas = all_betas[iarea]

    ax_common = axes[iarea * 2]
    ax_rare = axes[iarea * 2 + 1]

    # Plot chosen and unchosen Q-values for common trials
    Plot.AvgSem(betas[0, 0, 0], ax_common, label='Chosen',
                alpha=0.3, showleg=False, lw=LINE_WIDTH)
    Plot.AvgSem(betas[0, 0, 1], ax_common, label='Unchosen',
                alpha=0.3, showleg=False, lw=LINE_WIDTH)

    # Plot chosen and unchosen Q-values for rare trials
    Plot.AvgSem(betas[0, 1, 0], ax_rare, label='Chosen',
                alpha=0.3, showleg=False, lw=LINE_WIDTH)
    Plot.AvgSem(betas[0, 1, 1], ax_rare, label='Unchosen',
                alpha=0.3, showleg=False, lw=LINE_WIDTH)

    # Add legend for ACC
    if area == 'ACC':
        ax_rare.legend(loc='lower left', ncol=2)

    # Set x-axis limits
    for ax in (ax_common, ax_rare):
        Plot.set_xlim(data, ax, 0, 500, res=250, offset=-500)

    # Format y-axis
    for ax in (ax_common, ax_rare):
        ax.axhline(0, c='k', lw=1, zorder=-9999999)
        ax.set_ylim(-0.3, 0.3)

    ax_common.set_ylabel('Average\ncoefficients')
    ax_common.set_yticks([-0.3, 0, 0.3])
    ax_common.text(1, ax_common.get_ylim()[0], area, ha='left',
                   va='bottom', fontsize=12)

plt.tight_layout()
plt.show()