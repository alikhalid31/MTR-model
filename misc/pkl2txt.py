import argparse
import pickle
from pprint import pprint
from pathlib import Path

def convert_pickle_to_text(input_path, output_folder):
    input_path = Path(input_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the pickle file
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Prepare output path with .txt extension
    output_file = output_folder / (input_path.stem + ".txt")

    # Save as readable text
    with open(output_file, 'w') as f:
        pprint(data, stream=f)

    print(f"Converted and saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a pickle file to a readable text file.")
    parser.add_argument("input_file", help="Path to the .pkl file")
    parser.add_argument("output_folder", help="Folder to save the .txt file")

    args = parser.parse_args()
    convert_pickle_to_text(args.input_file, args.output_folder)
