import os
import json
import glob
import matplotlib.pyplot as plt


# Directory where your JSON files are

folder_paths = [
    './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_debuging_mode_1_custom_eval/eval/epoch_30/default', 
    './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_debuging_mode_6_custom_eval/eval/epoch_30/default', 
    './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_no_data_augmentation_debuging_mode_1_custom_eval/eval/epoch_30/default', 
    './../output/waymo/mtr+sliding_window_training/10s_history_30s_future_no_data_augmentation_debuging_mode_6_custom_eval/eval/epoch_30/default'
]
output_paths =  [
    './output/custom_eval/debuging_all_data/w_data_augmentation/mode_1',
    './output/custom_eval/debuging_all_data/w_data_augmentation/mode_6',
    './output/custom_eval/debuging_all_data/wo_data_augmentation/mode_1',
    './output/custom_eval/debuging_all_data/wo_data_augmentation/mode_6',
]


for i, folder_path in enumerate(folder_paths):
    output_path = output_paths[i]
    os.makedirs(output_path, exist_ok=True)
    file_pattern = os.path.join(folder_path, 'sliding_window_results_mode_6_timestamp_variable_*.json')


    for ts in range(41, 92, 10):
        current_timestamp =ts # The timestamp you are interested in
        max_pred_timestamp = 91  # Maximum timestamp index to consider

        mean_fde = {}  
        median_fde = {}  
        q1_fde = {}  
        q3_fde = {} 
        confidence = {}
        # Iterate over all matching files
        for file_path in glob.glob(file_pattern):
            # print(f"Processing file: {file_path}")
            # Extract the timestamp from the filename
            filename = os.path.basename(file_path)
            try:
                inference_timestamp = int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                continue  # skip if it doesn't follow the format

            with open(file_path, 'r') as f:
                data = json.load(f)
                data=data['custom_eval']['1']

            # Assume data is a list of dictionaries, and each dict has 'FDE' as a list
            # You want FDE at predicted timestamp = 80 (i.e., index 79)

            if 'mean_fde' in data and current_timestamp > 11:
                if inference_timestamp <60:
                    if inference_timestamp +1< current_timestamp and inference_timestamp+31>=current_timestamp:
                        # print(inference_timestamp)
                        mean_fde[inference_timestamp] = data['mean_fde'][current_timestamp - inference_timestamp -1 -1]
                        median_fde[inference_timestamp] = data['median_fde'][current_timestamp - inference_timestamp -1 -1]
                        q1_fde[inference_timestamp] = data['q1_fde'][current_timestamp - inference_timestamp -1 -1]
                        q3_fde[inference_timestamp] = data['q3_fde'][current_timestamp - inference_timestamp -1 -1]
                        confidence[inference_timestamp] = data['mean_scores'][current_timestamp - inference_timestamp -1 -1]
                    else:
                        continue

                if inference_timestamp >= 60:
                    if len(data['mean_fde']) > max_pred_timestamp - current_timestamp:
                        # continue
                        # print(f"Timestamp: {timestamp}")
                        # print(-((max_pred_timestamp - current_timestamp)+1))
                        # print(len(data['minFDE']))
                        mean_fde[inference_timestamp] = data['mean_fde'][-((max_pred_timestamp - current_timestamp)+1)]
                        median_fde[inference_timestamp] = data['median_fde'][-((max_pred_timestamp - current_timestamp)+1)]
                        q1_fde[inference_timestamp] = data['q1_fde'][-((max_pred_timestamp - current_timestamp)+1)]
                        q3_fde[inference_timestamp] = data['q3_fde'][-((max_pred_timestamp - current_timestamp)+1)]
                        confidence[inference_timestamp] = data['mean_scores'][-((max_pred_timestamp - current_timestamp)+1)]
                    else:
                        continue
                        # print(f"Timestamp: {timestamp}")
                

        # Sort by timestamp
        # print(fde_at_80)
        sorted_items = sorted(mean_fde.items())
        timestamps = [item[0] for item in sorted_items]
        mean_fde = [item[1] for item in sorted_items]
        confidence_values = [confidence.get(t, None) for t in timestamps]
        median_fde = [median_fde.get(t, None) for t in timestamps]
        q1_fde = [q1_fde.get(t, None) for t in timestamps]
        q3_fde = [q3_fde.get(t, None) for t in timestamps]

        # print("Timestamps:", timestamps)
        print("FDE:", mean_fde)
        print(len(mean_fde))

        # # Plot
        # plt.figure(figsize=(10, 5))
        # plt.plot(timestamps, fde_values, marker='o')
        # # plt.ylim(0, 12)
        # plt.xlabel('Inference Timestamp')
        # plt.ylabel('FDE (m)')
        # plt.title('FDE at Time Step 90 vs Inference Time')
        # plt.grid(True)
        # plt.tight_layout()

        # # Save the plot
        # output_path = os.path.join(output_path, 'fde_vs_timestamp_plot_debuging.png')
        # plt.savefig(output_path)

        # print(f"Plot saved to: {output_path}")

        # Plot with secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Primary axis for FDE
        ax1.plot(timestamps, mean_fde, marker='o', color='blue', label='mean FDE')
        ax1.scatter(timestamps, median_fde, label='Median FDE', color='green', marker='x')
        ax1.fill_between(timestamps, q1_fde, q3_fde, color='gray', alpha=0.3, label='IQR (Q1–Q3)')
        ax1.set_xlabel('Inference Timestamp')
        ax1.set_ylabel('FDE (m)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')


        # Secondary axis for Confidence
        ax2 = ax1.twinx()
        ax2.plot(timestamps, confidence_values, marker='s', color='red', linestyle='--', label='Confidence')
        ax2.set_ylabel('Confidence', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('FDE and Confidence VS Inference Time')
        fig.tight_layout()

        # Save the plot
        output_file = os.path.join(output_path, f"fde_vs_timestamp_with_confidence_{current_timestamp}.png")
        plt.savefig(output_file)
        print(f"Plot saved to: {output_file}")