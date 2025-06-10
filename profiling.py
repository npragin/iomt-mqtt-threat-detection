# Author: Noah Pragin
# Date: 2025-04-16
# Class: AI 541 - Machine Learning in the Real World - Dr. Kiri Wagstaff
# Description: This script generates a data profile for the CICIoMT2024 dataset.

import os

import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize


def generate_numeric_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a table of numeric feature statistics for each feature in the dataframe.

    For each numeric feature, calculates summary statistics including:
    - Minimum value
    - Maximum value
    - Mean
    - Median (50th percentile)
    - Standard deviation

    Values >= 1e5 are formatted in scientific notation with 2 decimal places.
    Values < 1e5 are formatted as regular decimals with 2 decimal places.
    The 'Label' column is excluded from the analysis.

    Args:
        df (pd.DataFrame): Input dataframe containing the features

    Returns:
        pd.DataFrame: A dataframe with rows for each feature and columns for the statistics
    """

    def format_value(value):
        """Format value conditionally based on magnitude."""
        if abs(value) >= 1e5:
            return f'{value:.2e}'
        else:
            return f'{value:.2f}'

    data = []
    for feature_name in df.columns:
        if feature_name not in ["Label"]:
            stats = df[feature_name].describe()
            row = {
                "name": feature_name,
                "min": format_value(stats["min"]),
                "max": format_value(stats["max"]),
                "mean": format_value(stats["mean"]),
                "50%": format_value(stats["50%"]),
            }
            data.append(row)

    return pd.DataFrame(data)


def generate_comparative_feature_distribution_histograms(
    dataframes: dict[str, pd.DataFrame],
) -> None:
    """
    Generates comparative histogram plots for feature distributions across multiple datasets sharing features.

    This function creates overlayed histograms for each feature across different dataframes, allowing for visual analysis of how feature distributions vary between datasets.

    The function:
    1. Creates output directory if it doesn't exist
    2. Iterates through each feature/column in the dataframes
    3. Generates overlaid histograms with transparency for comparison
    4. Saves each comparison as a separate PNG file

    Args:
        dataframes (dict[str, pd.DataFrame]): Dictionary mapping dataset names to
            their corresponding pandas DataFrames. All DataFrames should have the
            same column structure.

    Output:
        - Creates directory 'histograms/feature_distribution_comparison/' if needed
        - Generates one PNG file per feature named '{feature_name}.png' in the 'histograms/feature_distribution_comparison/' directory
    """
    os.makedirs("histograms/feature_distribution_comparison", exist_ok=True)

    for feature in next(iter(dataframes.values())).columns:
        plt.figure(figsize=(10, 5))
        for df_name, df in dataframes.items():
            plt.hist(
                df[feature],
                bins=20,
                alpha=1 / len(dataframes),
                label=df_name,
                density=True,
            )
        plt.legend()
        plt.title(f"Distribution of {feature} by label")
        plt.savefig(f"histograms/feature_distribution_comparison/{feature}.png")
        plt.close()


def create_violin_plots(df: pd.DataFrame):
    """
    Create violin plots for all features except "Label" and save them to
    histograms/feature_violin_plots/

    Winsorizes the top 5% of values for Header_Length and Rate features.

    Args:
        df (pd.DataFrame): The dataframe containing the features
    """
    os.makedirs("histograms/feature_violin_plots", exist_ok=True)

    features = [feature for feature in df.columns if feature not in ["Label"]]

    for feature in features:
        plt.figure(figsize=(12, 6))

        if feature in ["Header_Length", "Rate"]:
            feature_series = pd.Series(winsorize(df[feature].to_numpy(), (0, 0.05)))
        else:
            feature_series = df[feature]

        # Create violin plot
        plt.violinplot(feature_series, showmeans=True, showmedians=True)

        # Add labels
        plt.title(
            f'Distribution of {feature}{" (winsorized, top 5%)" if feature in ["Header_Length", "Rate"] else ""}',
            fontsize=14,
        )
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"histograms/feature_violin_plots/{feature}.png")
        plt.close()


def main():
    # Dictionaries to define type schema and feature ranges for group-wise comparison
    type_schema = {
        "Header_Length": "Packet",
        "Protocol Type": "Protocol",
        "Rate": "Packet",
        "fin_flag_number": "Flag",
        "syn_flag_number": "Flag",
        "rst_flag_number": "Flag",
        "psh_flag_number": "Flag",
        "ack_flag_number": "Flag",
        "ece_flag_number": "Flag",
        "cwr_flag_number": "Flag",
        "ack_count": "Flag",
        "syn_count": "Flag",
        "fin_count": "Flag",
        "rst_count": "Flag",
        "HTTP": "Protocol",
        "HTTPS": "Protocol",
        "DNS": "Protocol",
        "Telnet": "Protocol",
        "SMTP": "Protocol",
        "SSH": "Protocol",
        "IRC": "Protocol",
        "TCP": "Protocol",
        "UDP": "Protocol",
        "DHCP": "Protocol",
        "ARP": "Protocol",
        "ICMP": "Protocol",
        "IGMP": "Protocol",
        "IPv": "Protocol",
        "LLC": "Protocol",
        "Tot sum": "Packet",
        "Min": "Packet",
        "Max": "Packet",
        "AVG": "Packet",
        "Std": "Packet",
        "Tot size": "Packet",
        "IAT": "Packet",
        "Number": "Packet",
        "Variance": "Packet",
    }

    feature_ranges = {
        "Header_Length": "Real",
        "Protocol Type": "Real",
        "Rate": "Real",
        "fin_flag_number": "Rate",
        "syn_flag_number": "Rate",
        "rst_flag_number": "Rate",
        "psh_flag_number": "Rate",
        "ack_flag_number": "Rate",
        "ece_flag_number": "Rate",
        "cwr_flag_number": "Rate",
        "ack_count": "Real",
        "syn_count": "Real",
        "fin_count": "Real",
        "rst_count": "Real",
        "HTTP": "Rate",
        "HTTPS": "Rate",
        "DNS": "Rate",
        "Telnet": "Rate",
        "SMTP": "Rate",
        "SSH": "Rate",
        "IRC": "Rate",
        "TCP": "Rate",
        "UDP": "Rate",
        "DHCP": "Rate",
        "ARP": "Rate",
        "ICMP": "Rate",
        "IGMP": "Rate",
        "IPv": "Rate",
        "LLC": "Rate",
        "Tot sum": "Real",
        "Min": "Real",
        "Max": "Real",
        "AVG": "Real",
        "Std": "Real",
        "Tot size": "Real",
        "IAT": "Real",
        "Number": "Real",
        "Variance": "Real",
    }

    df_train = pd.read_csv("data/csv/compiled/train/compiled-data-train.csv")
    df_test = pd.read_csv("data/csv/compiled/test/compiled-data-test.csv")
    df = pd.concat([df_train, df_test])

    # Generate data profiles and save them as HTML files
    train_profile = ProfileReport(df_train, pool_size=64, type_schema=type_schema)
    test_profile = ProfileReport(df_test, pool_size=64, type_schema=type_schema)
    profile = ProfileReport(df, pool_size=64, type_schema=type_schema)

    train_profile.to_file("profiles/train_profile.html")
    test_profile.to_file("profiles/test_profile.html")
    profile.to_file("profiles/full_profile.html")

    # Split data by label to compare feature distributions
    df_benign = df[df["Label"] == 0]
    df_malicious = df[df["Label"] == 1]

    # Generate feature distribution histograms to compare across labels
    generate_comparative_feature_distribution_histograms(
        {"combined": df, "benign": df_benign, "malicious": df_malicious}
    )

    # Generate violin plots to clearer view into feature distributions
    create_violin_plots(df)

    packet_features = [k for k, v in type_schema.items() if v == "Packet"]
    flag_features = [k for k, v in type_schema.items() if v == "Flag"]
    protocol_features = [k for k, v in type_schema.items() if v == "Protocol"]

    # Generate numeric feature tables for each feature type for better comparison
    stats = generate_numeric_feature_table(df)
    os.makedirs("stats", exist_ok=True)
    packet_stats = stats[stats["name"].isin(packet_features)]
    flag_stats = stats[stats["name"].isin(flag_features)]
    protocol_stats = stats[stats["name"].isin(protocol_features)]
    packet_stats.to_csv("stats/packet_stats.csv", index=False)
    flag_stats.to_csv("stats/flag_stats.csv", index=False)
    protocol_stats.to_csv("stats/protocol_stats.csv", index=False)


if __name__ == "__main__":
    main()
