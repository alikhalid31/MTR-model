import pickle

# Replace 'file_path.pkl' with your file's path
with open('../../../data/waymo/processed_scenarios_single_example_20s_infos.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)

