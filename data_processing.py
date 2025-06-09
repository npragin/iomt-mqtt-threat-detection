# Author: Noah Pragin
# Date: 2025-04-16
# Class: AI 541 - Machine Learning in the Real World - Dr. Kiri Wagstaff
# Description: This script combines multiple CSV files into one after adding labels.

import os
import glob

import pandas as pd


def combine_csv_files(directory: str, output_file: str, benign_file: str) -> None:
    """
    Combines malicious and benign network traffic data into a single labeled dataset.

    This function:
    1. Reads all CSV files starting with "MQTT" from the given directory (malicious traffic)
    2. Reads the specified benign traffic file
    3. Labels the data (0=benign, 1=malicious)
    4. Removes unnecessary columns as specified in the dataset documentation
    5. Combines everything into a single CSV file

    Args:
        directory (str): Directory containing the MQTT malicious traffic CSV files
        output_file (str): Path where the combined dataset will be saved
        benign_file (str): Filename of the benign traffic CSV file in the directory
    """
    # Read all MQTT CSV files from data/train directory
    csv_files = glob.glob(f"{directory}/MQTT*.csv")
    dfs = []
    for file in csv_files:
        df_temp = pd.read_csv(file)
        dfs.append(df_temp)
    df = pd.concat(dfs, ignore_index=True)

    # Add label to malicious data
    df["Label"] = 1

    # Add label to benign data and combine with malicious data
    df_temp = pd.read_csv(f"{directory}/{benign_file}")
    df_temp["Label"] = 0
    df = pd.concat([df, df_temp], ignore_index=True)

    # Drop columns not described in the dataset description
    # http://cicresearch.ca/IOTDataset/CICIoMT2024/Dataset/README.pdf
    df.drop(
        columns=[
            "Duration",
            "Srate",
            "Drate",
            "Magnitue",
            "Radius",
            "Covariance",
            "Weight",
        ],
        inplace=True,
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    df.to_csv(f"{output_file}", index=False)


def main():
    combine_csv_files(
        "data/csv/raw/train",
        "data/csv/compiled/train/compiled-data-train.csv",
        "Benign_train.pcap.csv",
    )
    combine_csv_files(
        "data/csv/raw/test",
        "data/csv/compiled/test/compiled-data-test.csv",
        "Benign_test.pcap.csv",
    )


if __name__ == "__main__":
    main()
