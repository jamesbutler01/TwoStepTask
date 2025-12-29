import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import Details as D
import Maths
import ImportData
import Plot

# Configuration
FIGURE_DPI = 300
FONT_SIZE_LABEL = 14
FONT_SIZE_AXIS = 9
FONT_SIZE_LETTER = 16
LINE_WIDTH = 2.5

# Set matplotlib font sizes
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

# Analysis parameters
EPOCHS = [D.sc_choice1on]

# %% Main Analysis Loop
all_cpds = []

for area in D.areas[1:]:
    data = ImportData.EntireArea(area, 500, 1500)
    print(f"Processing area: {area}")

    # Initialize arrays
    num_neurons = data.n
    num_epochs = len(EPOCHS)
    cpds = Maths.nans((4, num_epochs, num_neurons, data.numTimepoints))

    for cell_idx in range(num_neurons):
        print(f"{area} - Cell {cell_idx + 1}/{num_neurons}")

        trial_data = data.behavdata[cell_idx]

        for epoch_idx, epoch in enumerate(EPOCHS):
            # Get normalized epoch data
            response = data.generate_epoch_norm(cell_idx, epoch)
            n = trial_data.n

            # Prepare regressors
            c1_chosen = np.copy(trial_data.c1chosen[:n]) - 1.5
            lin_effects = np.arange(n) - n / 2
            quad_effects = (lin_effects ** 2) - np.mean(lin_effects ** 2)
            pent_effects = (lin_effects ** 3) - np.mean(lin_effects ** 3)
            hex_effects = (lin_effects ** 4) - np.mean(lin_effects ** 4)
            sin1 = np.sin(np.arange(n) / 60)
            sin2 = np.sin(np.arange(n) / 40)
            sin3 = np.sin(np.arange(n) / 20)
            sin4 = np.sin(np.arange(n) / 10)

            # Build design matrix with Q-values and nuisance regressors
            design_matrix = np.vstack((
                np.ones(trial_data.n),
                trial_data.qvals['QmbPicA'],
                trial_data.qvals['QmbPicB'],
                trial_data.qvals['QmfPicA'],
                trial_data.qvals['QmfPicB'],
                c1_chosen,
                lin_effects, quad_effects, pent_effects, hex_effects,
                sin1, sin2, sin3, sin4
            )).T

            # Skip first trial as previous coefficients are invalid
            design_matrix = design_matrix[2:]
            response = response[2:]

            # Compute CPD for Q-values
            cpds[:, epoch_idx, cell_idx] = Maths.cpd(design_matrix, response, [1, 2, 3, 4])

    all_cpds.append(cpds)

# %% Create Figure
fig, axes = plt.subplots(3, 2, figsize=(8, 7), dpi=FIGURE_DPI)
axes = axes.flatten()

# Format axes
for idx in [2, 3, 4]:
    Plot.let(axes[idx], idx, -0.15, fontsize=FONT_SIZE_LETTER)
    axes[idx].spines['right'].set_visible(False)
    axes[idx].spines['top'].set_visible(False)

# Hide panels A, B, and F
axes[0].axis('off')  # Panel A
axes[1].axis('off')  # Panel B
axes[5].axis('off')  # Panel F

# Set labels for visible panels
for ax in axes[2:4]:
    ax.set_xlabel('Time (ms) post choice 1 on')
    ax.set_ylabel('CPD (%)')

# Separate axes for CPD plots and other visualizations
axes_cpd = axes[2:4]  # Panels C and D
ax_violin = axes[4]  # Panel E
# Panel F (axes[5]) is hidden

# %% Plot CPD Over Time
peak_cpds_mb = []
peak_cpds_mf = []

for iarea, area in enumerate(D.areas[1:]):
    cpds = all_cpds[iarea]
    num_neurons = cpds.shape[2]

    titles = ['MB', 'MF']

    for icoeff in range(2):
        # Average weights for both PicA and PicB
        cpd_data = cpds[icoeff * 2:(icoeff * 2) + 2, 0, :, :]
        cpd_avg = np.mean(cpd_data, axis=0)

        # Plot population CPD (Panels C and D)
        ax_cpd = axes_cpd[icoeff]
        avg, sem = Plot.AvgSem(cpd_avg, ax_cpd, c=f'C{iarea}', label=area,
                               showleg=False, zorder=20 - iarea * 3,
                               alpha=0.3, lw=LINE_WIDTH)
        Plot.set_xlim(data, ax_cpd, -500, 1000, offset=-500)

        # Format CPD axes
        ax_cpd.set_ylim(0.15, 0.7)

        ax_cpd.set_title(['MB', 'MF', 'Hyb'][icoeff] + ' Q-values')

        # Print peak values
        peak = np.max(avg)
        peak_tp = np.argmax(avg)
        print(f"{area} {titles[icoeff]}: peak CPD = {peak:.4f} ± {sem[peak_tp]:.4f} "
              f"at {peak_tp * 10 - 500} ms")

    # Log peak CPDs for violin plot
    peak_cpds_mb.append(np.nanmax(np.nanmax(cpds[0:2, 0], axis=1), axis=0))
    peak_cpds_mf.append(np.nanmax(np.nanmax(cpds[2:4, 0], axis=1), axis=0))

# Add legend
axes_cpd[0].legend(loc='upper left', fontsize=FONT_SIZE_AXIS + 2,
                   bbox_to_anchor=[1, 1], ncol=1, framealpha=1)

# %% Violin Plot (Panel E)
v1 = ax_violin.violinplot(peak_cpds_mb, points=100)

for body in v1['bodies']:
    try:
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        # Modify paths to not go further right than center
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], -np.inf, m
        )
        body.set_alpha(0.6)
    except:
        pass

v2 = ax_violin.violinplot(peak_cpds_mf, points=100)
for body in v2['bodies']:
    try:
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        # Modify paths to not go further left than center
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], m, np.inf
        )
        body.set_alpha(0.6)
    except:
        pass

# Add median labels
for iarea in range(4):
    mb_median = np.median(peak_cpds_mb[iarea])
    ax_violin.plot([0.75 + iarea, 1 + iarea], [mb_median] * 2, c='C0')
    mf_median = np.median(peak_cpds_mf[iarea])
    ax_violin.plot([1 + iarea, 1.25 + iarea], [mf_median] * 2, c='C1')

ax_violin.legend([v1['bodies'][0], v2['bodies'][0]], ['MB', 'MF'],
                 loc='upper right', bbox_to_anchor=[1, 1.1], fontsize=9)
ax_violin.set_ylabel('Peak CPD')
ax_violin.set_xticks(range(1, 5))
ax_violin.set_xticklabels(D.areas[1:])
ax_violin.set_ylim(1, 11)

plt.tight_layout()
plt.show()