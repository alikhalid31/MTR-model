import os
import json
import glob
import matplotlib.pyplot as plt

# Directory where your JSON files are
folder_path = './../output/waymo/mtr+sliding_window/10s_history_70s_future_debuging/eval/epoch_100/default'
output_path =  './../output/waymo/'
# Pattern to match JSON files
file_pattern = os.path.join(folder_path, 'sliding_window_results_mode_6_timestamp_variable_*.json')
# Dict to store timestamp -> FDE at 80
fde_at_80 = {}
current_timestamp = 20 # The timestamp you are interested in
max_pred_timestamp = 91  # Maximum timestamp index to consider
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
    if 'minFDE' in data and len(data['minFDE']) > max_pred_timestamp - current_timestamp:
        # continue
        # print(f"Timestamp: {timestamp}")
        # print(-((max_pred_timestamp - current_timestamp)+1))
        # print(len(data['minFDE']))
        fde_value = data['minFDE'][-((max_pred_timestamp - current_timestamp)+1)]
        fde_at_80[timestamp] = fde_value
    else:
        continue
        # print(f"Timestamp: {timestamp}")

# Sort by timestamp
sorted_items = sorted(fde_at_80.items())
timestamps = [item[0] for item in sorted_items]
fde_values = [item[1] for item in sorted_items]

# print("Timestamps:", timestamps)
# print("FDE at 80:", fde_values)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(timestamps, fde_values, marker='o')
plt.xlabel('Evaluation Timestamp')
plt.ylabel('FDE at Predicted Timestamp 90')
plt.title('FDE at Time Step 90 vs Evaluation Time')
plt.grid(True)
plt.tight_layout()

# Save the plot
output_path = os.path.join(output_path, 'fde_vs_timestamp_plot_debuging.png')
plt.savefig(output_path)

print(f"Plot saved to: {output_path}")