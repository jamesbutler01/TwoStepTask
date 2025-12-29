import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import Details as D
import Maths
import ImportData
import Plot
import sklearn.model_selection
import sklearn.svm

# Configuration
FIGURE_DPI = 200
FONT_SIZE_AXIS = 13
FONT_SIZE_LETTER = 14
LINE_WIDTH = 2.5

# Set matplotlib font sizes
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

# Decoder parameters
N_SPLITS = 2
N_REPEATS = 1
N_SAMPLES = 20
RES = 1  # Take every nth time point

# %% Run decoder analysis for cell splits
print("Running decoder analysis with cell splits...")

decoder = sklearn.svm.LinearSVC(max_iter=25000)

# Initialize storage for all areas
data_temp = ImportData.EntireArea('ACC', 600, 600, exclude_neurons=False)
shape = (len(D.areas), data_temp.numTimepoints // RES, N_SAMPLES)
decoder_no_split = np.empty(shape)
decoder_high_split = np.empty(shape)
decoder_low_split = np.empty(shape)

for iarea, area in enumerate(D.areas[1:]):
    data = ImportData.EntireArea(area, 600, 600, exclude_neurons=False)
    print(f"Processing decoder for area: {area}, neurons: {data.n}")

    # Find minimum label count and collect data
    min_label = 10000
    responses_all = []
    labels_all = []

    fb_cpd = np.empty(data.n)
    trans_cpd = np.empty(data.n)

    for cell_idx in range(data.n):
        trial_data = data.behavdata[cell_idx]

        # Prepare regressors
        transition = np.copy(trial_data.transition) - 1.5
        c1_chosen = np.copy(trial_data.c1chosen) - 1.5
        reward = np.copy(trial_data.rewgiven) - 1
        c1_chosen *= -1
        transition *= -1

        design_matrix = np.vstack((
            np.ones(trial_data.n), reward, c1_chosen, transition,
            reward * transition, reward * c1_chosen, transition * c1_chosen,
            transition * reward * c1_chosen,
            trial_data.previouschoice1c, trial_data.previoustransition
        )).T

        # Calculate CPDs for sorting cells
        response_fb = data.generate_epoch_norm(cell_idx, D.sc_secondaryreinforceron)
        fb_cpd[cell_idx] = np.max(Maths.cpd(design_matrix, response_fb, 1))

        response_trans = data.generate_epoch_norm(cell_idx, D.sc_choice1made)
        trans_cpd[cell_idx] = np.max(Maths.cpd(design_matrix, response_trans, 3))

        # Find minimum number of trials per label
        min1 = np.min([np.sum(trial_data.c1chosen == label) for label in [1, 2]])
        cell_min_label = np.min([min1])
        if cell_min_label < min_label:
            min_label = cell_min_label

        labels_all.append(trial_data.c1chosen)
        responses_all.append(response_trans)

    # Sort cells by combined CPD
    overall_cpd = fb_cpd * trans_cpd
    sort_cpd = np.argsort(overall_cpd)

    # Run decoder for each sample
    for isample in range(N_SAMPLES):
        print(f"  Sample {isample + 1}/{N_SAMPLES}")

        split_generator = sklearn.model_selection.RepeatedStratifiedKFold(
            n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=isample
        )

        # Collect balanced data
        labels_array = np.empty((data.n, 2, min_label))
        responses_array = np.empty((data.n, 2, min_label, data.numTimepoints // RES))

        for cell_idx in range(data.n):
            response = responses_all[cell_idx]
            labels = labels_all[cell_idx]

            for i_label, label_val in enumerate(np.unique(labels)):
                rand_inds = np.random.permutation(np.sum(labels == label_val))[:min_label]
                response_for_label = response[labels == label_val][rand_inds]
                responses_array[cell_idx, i_label] = response_for_label
                labels_array[cell_idx, i_label] = i_label

        # Reshape for decoder
        x_labels = labels_array.reshape([labels_array.shape[0], -1])
        x_labels = x_labels[0, :]  # Remove redundant dimension
        responses_array = responses_array.reshape([
            responses_array.shape[0],
            responses_array.shape[1] * responses_array.shape[2],
            responses_array.shape[3]
        ])

        # Split into high and low CPD neurons
        responses_low = responses_array[sort_cpd[:len(sort_cpd) // 2]]
        responses_high = responses_array[sort_cpd[len(sort_cpd) // 2:]]

        # Run decoder at each timepoint
        for ti in range(responses_array.shape[-1]):
            responses_all_ti = responses_array[..., ti].T
            responses_low_ti = responses_low[..., ti].T
            responses_high_ti = responses_high[..., ti].T

            # Cross-validation
            acc_all = sklearn.model_selection.cross_val_score(
                decoder, responses_all_ti, x_labels,
                cv=split_generator, n_jobs=D.n_cores
            )
            acc_high = sklearn.model_selection.cross_val_score(
                decoder, responses_high_ti, x_labels,
                cv=split_generator, n_jobs=D.n_cores
            )
            acc_low = sklearn.model_selection.cross_val_score(
                decoder, responses_low_ti, x_labels,
                cv=split_generator, n_jobs=D.n_cores
            )

            decoder_no_split[iarea, ti, isample] = np.mean(acc_all)
            decoder_high_split[iarea, ti, isample] = np.mean(acc_high)
            decoder_low_split[iarea, ti, isample] = np.mean(acc_low)

# %% Firing rates and CPD analysis
all_cpds = []

for area in D.areas[1:]:
    data = ImportData.EntireArea(area, 600, 600, exclude_neurons=False)
    print(f"Processing area: {area}")

    # Initialize arrays
    num_neurons = data.n
    cpds = np.empty((2, num_neurons, data.numTimepoints))

    for cell_idx in range(num_neurons):
        print(f"{area} - Cell {cell_idx + 1}/{num_neurons}")

        trial_data = data.behavdata[cell_idx]

        # For choice 1 made epoch
        epoch = D.sc_choice1made
        response = data.generate_epoch_raw(cell_idx, epoch)

        # Generate GLM design matrix
        design_matrix = data.generate_glm1(trial_data)

        # Skip first trial as previous coefficients are invalid
        response = response[2:]

        # Compute CPD for Choice 1 and triple interaction
        cpds[:, cell_idx] = Maths.cpd(design_matrix, response, [2, 13])

    all_cpds.append(cpds)

# %% Create Figure
fig = plt.figure(figsize=(12 * 1.0, 8 * 0.6), dpi=FIGURE_DPI)

# Define grid layout
height_ratios = [1, 0.4, 1]
width_ratios_main = [1, 0.2, 1, 0.3, 1]
gs = plt.GridSpec(len(height_ratios), len(width_ratios_main),
                  height_ratios=height_ratios, width_ratios=width_ratios_main,
                  hspace=0, wspace=0)

# Create subplots
axes = np.array([
    fig.add_subplot(gs[0, 0]),  # CPD panel A
    fig.add_subplot(gs[0, 2]),  # CPD panel B
    fig.add_subplot(gs[0, 4]),  # Decoder - All neurons
    fig.add_subplot(gs[2, 0]),  # Decoder - High split
    fig.add_subplot(gs[2, 2]),  # Decoder - Low split
    fig.add_subplot(gs[2, 4])  # Decoder difference
])

# Format all axes
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Set x-axis limits for all panels
for ax in axes:
    Plot.set_xlim(data, ax, -600, 600, offset=-600, res=200)

# Set x-labels for bottom row
for ax in axes[3:]:
    ax.set_xlabel('Time (ms) post choice 1', fontsize=FONT_SIZE_AXIS)

# Add panel letters
for i in range(len(axes)):
    Plot.let(axes[i], i, -0.125, fontsize=FONT_SIZE_LETTER)

# Add vertical line at 60ms
for ax in axes:
    ax.axvline(60, ls='--', c='k', lw=0.5, zorder=-1)

# Separate axes for CPD and decoder plots
axes_cpd = axes[:2]
axes_decoder = axes[2:]

# %% Decoder Plots (Panels C, D, E, F)
for iarea, area in enumerate(D.areas[1:]):
    decoder_averages = {}

    decoder_data = [
        (decoder_no_split, 'All neurons'),
        (decoder_high_split, 'Reward/Transition coders'),
        (decoder_low_split, 'Other neurons')
    ]

    for (data_arr, label), ax in zip(decoder_data, axes_decoder[:3]):
        # Plot decoder accuracy
        avg, sem = Plot.AvgSem(data_arr[iarea].T * 100, ax, label=area,
                               showleg=False, lw=LINE_WIDTH)

        decoder_averages[label] = data_arr[iarea]

        # Print peak values
        tp = 50
        print(f"{area} {label}: {avg[tp]:.4f} ± {sem[tp]:.4f}")

        # Format axes
        ax.axhline(50, color='gray', zorder=0, lw=0.5)
        ax.set_ylim(45, 83)
        ax.set_title(label)

    # Set y-labels
    axes_decoder[0].set_ylabel('Accuracy (%)', fontsize=FONT_SIZE_AXIS)
    axes_decoder[1].set_ylabel('Accuracy (%)', fontsize=FONT_SIZE_AXIS)

    # Decoder difference plot (Panel F)
    # Compute difference at the sample level before averaging
    diff_data = (decoder_averages['Reward/Transition coders'] -
                 decoder_averages['Other neurons']) * 100
    Plot.AvgSem(diff_data.T, axes_decoder[3], label=area, showleg=False, lw=LINE_WIDTH)
    axes_decoder[3].set_ylabel('Difference', fontsize=FONT_SIZE_AXIS)
    axes_decoder[3].axhline(0, ls='-', c='k', lw=0.5, zorder=-99)

# %% CPD Plots (Panels A and B)
print('Choice 1 CPD analysis')

for iarea, area in enumerate(D.areas[1:]):
    cpds = all_cpds[iarea]

    for coeff_idx in range(2):
        ax = axes_cpd[coeff_idx]

        # Plot population CPD
        Plot.AvgSem(cpds[coeff_idx], ax, label=area, showleg=False, lw=LINE_WIDTH)
        Plot.set_xlim(data, ax, -600, 600, offset=-600, res=300)

        # Format axes
        ax.set_xlabel('Time (ms) post choice 1 on', fontsize=FONT_SIZE_AXIS)
        ax.set_ylim(0.1, 0.75)

    # Set titles and labels
    axes_cpd[0].set_ylabel('CPD (%)', fontsize=FONT_SIZE_AXIS)
    axes_cpd[0].set_title('Choice 1 coding')
    axes_cpd[1].set_title(
        'Choice 1 (t₋₁) * Reward (t₋₁) * Transition (t₋₁)',
        ha='left', x=-0.05, fontsize=10
    )
    axes_cpd[1].set_xlabel('')

# Add legend
axes_cpd[0].legend(loc='lower left', ncol=4, bbox_to_anchor=[0, 1.15],
                   fontsize=10)

plt.tight_layout()
plt.show()