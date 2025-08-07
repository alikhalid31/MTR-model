import os
import json
import glob
import matplotlib.pyplot as plt

# Directory where your JSON files are
# folder_path = './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_no_data_augmentation_debuging_single_example_mode_6/eval/epoch_30/default'
# output_path =  './output/debuging_single_example/wo_data_augmentation/mode_6'

# folder_path = './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_no_data_augmentation_debuging_mode_1/eval/epoch_30/default'
# output_path =  './output/debuging_all_data/wo_data_augmentation/mode_1'

# folder_path = './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_debuging_mode_6/eval/epoch_30/default'
# output_path =  './output/debuging_all_data/w_data_augmentation/mode_6'

folder_path = './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_debuging_single_example_mode_1/eval/epoch_30/default'
output_path =  './output/debuging_single_example/w_data_augmentation/mode_1'
# Pattern to match JSON files
file_pattern = os.path.join(folder_path, 'sliding_window_results_mode_6_timestamp_variable_*.json')
# Dict to store timestamp -> FDE at 80
for ts in range(10, 31, 10):
    current_timestamp = ts # The timehorizon you are interested in
    # max_pred_timestamp = 91  # Maximum timestamp index to consider

    fde_at_80 = {}
    confidence_at_80 = {}

    # Iterate over all matching files
    for file_path in glob.glob(file_pattern):
        # print(f"Processing file: {file_path}")
        # Extract the timestamp from the filename
        filename = os.path.basename(file_path)
        try:
            timestamp = int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            continue  # skip if it doesn't follow the format

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Assume data is a list of dictionaries, and each dict has 'FDE' as a list
        # You want FDE at predicted timestamp = 80 (i.e., index 79)
        if 'minFDE' in data and len(data['minFDE']) >= current_timestamp:
            # continue
            # print(f"Timestamp: {timestamp}")
            # print(-((max_pred_timestamp - current_timestamp)+1))
            # print(len(data['minFDE']))
            fde_value = data['minFDE'][current_timestamp-1]
            fde_at_80[timestamp] = fde_value
            confidence_at_80[timestamp] = data['confidence'][0]
        else:
            continue
            # print(f"Timestamp: {timestamp}")

    # Sort by timestamp
    # sorted_items = sorted(fde_at_80.items())
    # timestamps = [item[0] for item in sorted_items]
    # fde_values = [item[1] for item in sorted_items]

    sorted_items = sorted(fde_at_80.items())
    timestamps = [item[0] for item in sorted_items]
    fde_values = [item[1] for item in sorted_items]
    confidence_values = [confidence_at_80.get(t, None) for t in timestamps]

    print("FDE at 80:", fde_values)
    print(len(fde_values))

    # print("Timestamps:", timestamps)
    # print("FDE at 80:", fde_values)

    # # Plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(timestamps, fde_values, marker='o')
    # plt.xlabel('Inference Timestamp')
    # plt.ylabel('FDE (m)')
    # plt.title('FDE at time horizon 30 vs Inference Time')
    # plt.grid(True)
    # plt.tight_layout()

        # Plot with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Primary axis for FDE
    ax1.plot(timestamps, fde_values, marker='o', color='blue', label='FDE')
    ax1.set_xlabel('Inference Timestamp')
    ax1.set_ylabel('FDE (m)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Secondary axis for Confidence
    ax2 = ax1.twinx()
    ax2.plot(timestamps, confidence_values, marker='s', color='red', linestyle='--', label='Confidence')
    ax2.set_ylabel('Confidence', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('FDE and Confidence at Time Step 80 vs Inference Time')
    fig.tight_layout()

    # Save the plot
    output_file = os.path.join(output_path, f"fde_vs_timehorizon_with_confidence_{current_timestamp}.png")
    plt.savefig(output_file)

    print(f"Plot saved to: {output_path}")