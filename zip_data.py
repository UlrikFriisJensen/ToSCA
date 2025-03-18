# Imports
from shutil import make_archive
import argparse

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    
    args = parser.parse_args()
    
    make_archive(args.output_file, 'zip', args.data_folder)