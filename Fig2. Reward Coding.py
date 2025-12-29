import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import Details as D
import Maths
import ImportData
import Plot

# Configuration
FIGURE_DPI = 200
FIGURE_SCALE = 0.9
FONT_SIZE_LABEL = 14
FONT_SIZE_AXIS = 13
FONT_SIZE_LETTER = 16
FONT_SIZE_TICK = 12
LINE_WIDTH = 2.5

# Set matplotlib font sizes
matplotlib.rc('xtick', labelsize=FONT_SIZE_TICK)
matplotlib.rc('ytick', labelsize=FONT_SIZE_TICK)

# Analysis parameters
EPOCHS = (D.sc_secondaryreinforceron, D.sc_madefixation, D.sc_choice1on,
          D.sc_choice2on, D.sc_secondaryreinforceron)
EPOCH_INDICES = [0, 100, 150, 200, 250, 300]
NUM_TIMEPOINTS = EPOCH_INDICES[-1]

#%% Main Analysis Loop
all_cpds_concatenated = []
all_cpds_per_epoch = []
all_betas_per_epoch = []
all_betas_previous = []

for area in D.areas[1:]:
    data = ImportData.EntireArea(area, 0, 1500, exclude_neurons=False)
    print(f"Processing area: {area}")

    # Initialize arrays
    num_neurons = data.n
    cpds_concatenated = Maths.nans((num_neurons, NUM_TIMEPOINTS))
    cpds_per_epoch = Maths.nans((num_neurons, data.numTimepoints))
    betas_per_epoch = Maths.nans((num_neurons, data.numTimepoints))
    betas_previous_trials = np.empty((3, num_neurons, data.numTimepoints))

    # Process each neuron
    for cell_idx in range(num_neurons):
        print(f"{area} - Cell {cell_idx + 1}/{num_neurons}")

        trial_data = data.behavdata[cell_idx]
        epoch_responses = []
        concatenated_response = np.empty((trial_data.n - 2, NUM_TIMEPOINTS))

        # Process each epoch
        for epoch_idx, epoch in enumerate(EPOCHS):
            epoch_normalized, _ = data.generate_epoch_norm(cell_idx, epoch)

            # Handle previous trial (first epoch)
            if epoch_idx == 0:
                epoch_responses.append(epoch_normalized[1:-1])
                epoch_normalized = epoch_normalized[1:-1]
            else:
                epoch_responses.append(epoch_normalized[2:])
                epoch_normalized = epoch_normalized[2:]  # Skip first trial

            # Fill concatenated response array
            start_idx = EPOCH_INDICES[epoch_idx]
            end_idx = EPOCH_INDICES[epoch_idx + 1]
            duration = end_idx - start_idx
            concatenated_response[:, start_idx:end_idx] = epoch_normalized[:, 0:duration]

        # Generate GLM design matrix
        design_matrix = data.generate_glm1(trial_data)

        # Compute CPD for concatenated data (reward coding)
        cpds_concatenated[cell_idx] = Maths.cpd(design_matrix, concatenated_response, 1)

        # Compute CPD and betas for secondary reinforcer epoch
        response = epoch_responses[-1]
        cpds_per_epoch[cell_idx] = Maths.cpd(design_matrix, response, 15)
        betas_per_epoch[cell_idx] = Maths.reg(design_matrix, response)[15]

        # Previous trials analysis (t0, t-1, t-2)
        response_prev, _ = data.generate_epoch_norm(cell_idx, D.sc_secondaryreinforceron)
        response_prev = response_prev[2:]  # Skip first trial
        previous_reward_indices = [15, 1, 14]
        betas_previous_trials[:, cell_idx] = \
            Maths.reg(design_matrix, response_prev)[previous_reward_indices]

    # Store results for this area
    all_cpds_concatenated.append(cpds_concatenated)
    all_cpds_per_epoch.append(cpds_per_epoch)
    all_betas_previous.append(betas_previous_trials)
    all_betas_per_epoch.append(betas_per_epoch)

#%% Create Figure
fig = plt.figure(figsize=(12.5 * FIGURE_SCALE, 11 * FIGURE_SCALE), dpi=FIGURE_DPI)

# Define grid layouts
height_ratios_main = [1, 0.5, 1, 0.5, 1, 0.5, 1]
width_ratios_main = [1, 0.35, 1, 0.35, 1]
width_ratios_cpd = [1, 0.2, 1]

gs_cpd = plt.GridSpec(len(height_ratios_main), len(width_ratios_cpd),
                      height_ratios=height_ratios_main, width_ratios=width_ratios_cpd,
                      hspace=0, wspace=0)
gs_mid = plt.GridSpec(len(height_ratios_main), len(width_ratios_main),
                      height_ratios=height_ratios_main, width_ratios=width_ratios_main,
                      hspace=0, wspace=0)
gs_bottom = plt.GridSpec(len(height_ratios_main), len(width_ratios_main),
                         height_ratios=height_ratios_main, width_ratios=width_ratios_main,
                         hspace=0, wspace=0)

# Create subplots
ax_cpd = fig.add_subplot(gs_cpd[0, 0])
ax_percent = fig.add_subplot(gs_cpd[0, 2])  # Hidden panel
ax_latency = fig.add_subplot(gs_mid[2, 0])
ax_reward = fig.add_subplot(gs_mid[2, 2])
ax_choice2 = fig.add_subplot(gs_mid[2, 4])  # Hidden panel
ax_prev_reward_1 = fig.add_subplot(gs_bottom[4, 0])
ax_prev_reward_2 = fig.add_subplot(gs_bottom[4, 2])
ax_prev_reward_3 = fig.add_subplot(gs_bottom[6, 0])
ax_prev_reward_4 = fig.add_subplot(gs_bottom[6, 2])

# Hide panels B and E completely
ax_percent.axis('off')
ax_choice2.axis('off')

all_axes = [ax_cpd, ax_latency, ax_reward,
            ax_prev_reward_1, ax_prev_reward_2, ax_prev_reward_3, ax_prev_reward_4]

# Format axes
for idx, ax in enumerate(all_axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.get_label().set_fontsize(FONT_SIZE_AXIS)
    ax.yaxis.get_label().set_fontsize(FONT_SIZE_AXIS)

    # Panel letters: A, C, D, F, G, H, I (skipping B and E)
    panel_indices = [0, 2, 3, 5, 6, 7, 8]
    x_positions = [-0.1, -0.2, -0.25, -0.25, -0.05, -0.25, -0.05]
    y_positions = [1, 1.1, 1.1, 1, 1, 1, 1]
    Plot.let(ax, panel_indices[idx], x=x_positions[idx], y=y_positions[idx], fontsize=FONT_SIZE_LETTER)

# Add horizontal zero lines
for ax in [ax_cpd, ax_reward,
           ax_prev_reward_1, ax_prev_reward_2, ax_prev_reward_3, ax_prev_reward_4]:
    ax.axhline(0, c='k', lw=0.5, zorder=-10)

#%% Plot CPD Over Time
all_latencies = []

for area_idx, area in enumerate(D.areas[1:]):
    cpds_concat = all_cpds_concatenated[area_idx]
    cpds_epoch = all_cpds_per_epoch[area_idx]

    # Plot CPD for each epoch
    for epoch_idx, epoch_end in enumerate(EPOCH_INDICES):
        if epoch_idx == 0:
            continue

        epoch_start = EPOCH_INDICES[epoch_idx - 1]
        cpd_data = cpds_concat[:, epoch_start:epoch_end]

        if epoch_idx == 1:
            Plot.AvgSem(cpd_data, ax_cpd, xrange=range(epoch_start, epoch_end),
                       c=f'C{area_idx}', label=area, showleg=False,
                       zorder=20 - area_idx * 3, alpha=0.3, lw=LINE_WIDTH)
        else:
            Plot.AvgSem(cpd_data, ax_cpd, xrange=range(epoch_start, epoch_end),
                       c=f'C{area_idx}', showleg=False,
                       zorder=20 - area_idx * 3, alpha=0.3, lw=LINE_WIDTH)

    # Calculate latency peaks
    latency_peaks = []
    max_time = 100

    for cell_idx in range(cpds_epoch.shape[0]):
        observed = cpds_epoch[cell_idx, :max_time]
        latency_peaks.append(np.argmax(observed) * 10)

    all_latencies.append(latency_peaks)

# Format CPD axis (Panel A)
[ax_cpd.axvline(t, c='grey', lw=0.5, zorder=-100) for t in EPOCH_INDICES[1:-1]]
ax_cpd.set_xticks(EPOCH_INDICES[:-1])
ax_cpd.set_xticklabels(
    ('Feedback\n(t₋₁) (1000 ms)', 'Fixation\n(t) (500 ms)',
     'Choice 1', 'Choice 2', 'Feedback'),
    ha='left', fontsize=10
)
ax_cpd.set_xlim(0, 300)
ax_cpd.set_ylim(0, 6)
ax_cpd.legend(loc='upper right', bbox_to_anchor=[1, 0.95],
             framealpha=1, fontsize=10)
ax_cpd.set_ylabel('CPD (%)')

#%% Plot Latency Analysis (Panel C)
latencies_combined = [
    all_latencies[0] + all_latencies[1],  # PFC
    all_latencies[2] + all_latencies[3]   # Striatum
]

for region_idx, latencies in enumerate(latencies_combined):
    means = Maths.nans(2)
    sems = Maths.nans(2)
    means[region_idx] = np.mean(latencies)
    sems[region_idx] = Maths.sem(np.array(latencies))

    ax_latency.errorbar(means, range(len(means)), xerr=sems,
                       color=['grey', 'k'][region_idx],
                       elinewidth=2, capsize=5, capthick=2, marker='o')
    print(f"{['PFC', 'Striatum'][region_idx]}: "
          f"{means[region_idx]:.2f} ± {sems[region_idx]:.2f}")

ax_latency.set_yticks(range(2))
ax_latency.set_yticklabels(['PFC', 'Striatum'])
ax_latency.set_ylim(-0.5, 1.25)
ax_latency.set_xlabel('Peak coding latency (ms)', fontsize=FONT_SIZE_AXIS)
ax_latency.set_xlim(480, 620)

#%% Plot Average Coefficients (Panel D)
for area_idx, area in enumerate(D.areas[1:]):
    betas_prev = all_betas_previous[area_idx]

    # Reward coefficients (Panel D)
    reward_betas = betas_prev[0]
    Plot.AvgSem(reward_betas, ax_reward, c=f'C{area_idx}',
               label=area, alpha=0.3, showleg=False, lw=LINE_WIDTH)
    Plot.set_xlim(data, ax_reward, 0, 600, res=300)
    ax_reward.set_ylabel('Average\ncoefficients')
    ax_reward.yaxis.set_label_coords(-0.18, 0.5)
    ax_reward.set_ylim(-0.1, 0.16)
    ax_reward.set_xlabel('Time (ms) post feedback')
    ax_reward.axhline(0, c='k', lw=1, zorder=-100)

    # Previous trial analysis (Panels F, G, H, I)
    prev_axes = [ax_prev_reward_1, ax_prev_reward_2, ax_prev_reward_3, ax_prev_reward_4]
    ax_main = prev_axes[area_idx]

    # Create inset axis
    inset_x = [0.30, 0.59, 0.30, 0.59][area_idx]
    inset_y = [0.41, 0.41, 0.21, 0.21][area_idx]
    ax_inset = fig.add_axes([inset_x, inset_y, 0.08, 0.06])

    # Plot coefficients for t0, t-1, t-2
    means = Maths.nans(3)
    labels = ['t₀', 't₋₁', 't₋₂']

    for trial_lag in range(3):
        Plot.AvgSem(betas_prev[trial_lag], ax_main, c=f'C{trial_lag + 5}',
                   label=labels[trial_lag], alpha=0.3, showleg=False, lw=LINE_WIDTH)

        # Calculate mean at peak
        peak_idx = np.argmax(np.abs(np.mean(betas_prev[trial_lag, :, :50], axis=0)))
        means[trial_lag] = np.mean(betas_prev[trial_lag, :, peak_idx])

        # Plot in inset
        ax_inset.errorbar([trial_lag], means[trial_lag], 0,
                         marker='o', c=f'C{trial_lag + 5}')

    # Format main axis
    Plot.set_xlim(data, ax_main, 0, 600, res=300)
    ax_main.set_ylim(-0.08, 0.155)
    ax_main.text(1, 0.155, area, ha='left', va='top', fontsize=15)
    ax_main.set_xlabel('Time (ms) post feedback', fontsize=FONT_SIZE_AXIS)
    ax_main.axhline(0, c='k', lw=1, zorder=-99999)

    if area == 'DLPFC':
        ax_main.legend(loc='center left', ncol=1, fontsize=15,
                      bbox_to_anchor=[1.3, 0.5])
    if area in ['ACC', 'Caudate']:
        ax_main.set_ylabel('Average\ncoefficients', fontsize=FONT_SIZE_AXIS)
        ax_main.yaxis.set_label_coords(-0.18, 0.5)
    else:
        ax_main.set_yticklabels([])

    # Format inset
    ax_inset.set_xticks([0, 1, 2])
    ax_inset.set_xticklabels(labels)
    ax_inset.axhline(0, c='grey', lw=0.5, zorder=-100)
    ax_inset.set_ylim(-0.08, 0.14)
    ax_inset.set_xlim(-0.5, 2.5)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()