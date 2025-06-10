# Author: Noah Pragin
# Date: 2025-06-09
# Class: AI 541 - Machine Learning in the Real World - Dr. Kiri Wagstaff
# Description: This script runs experiments to compare the effects of different feature reduction techniques.


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from experiments_helpers import (
    get_train_test_data,
    get_models,
    run_experiments,
    DoNothingTransformer,
    save_feature_elimination_results,
    run_sequential_feature_selection_experiments,
    run_recursive_feature_selection_experiments,
)


def get_adjusted_indices(
    original_columns: pd.Index | list[str],
    target_columns: list[str],
    removed_columns: list[str],
    num_pca_components_prepended: int,
) -> list[int]:
    """
    Get correct indices after accounting for removed columns and prepended PCA components.

    Args:
        original_columns: Original column list
        target_columns: Names of columns we want to find indices for
        removed_columns: Names of columns already removed in previous steps
        num_pca_components_prepended: Number of PCA components added so far (to the front)

    Returns:
        List of indices of the target columns in the new dataframe.
    """
    original_indices = [
        list(original_columns).index(column) for column in target_columns
    ]
    removed_indices = [
        list(original_columns).index(column) for column in removed_columns
    ]

    adjusted_indices = []
    for orig_idx in original_indices:
        columns_removed_before = sum(
            1 for removed_idx in removed_indices if removed_idx < orig_idx
        )
        adjusted_idx = orig_idx - columns_removed_before + num_pca_components_prepended
        adjusted_indices.append(adjusted_idx)

    return adjusted_indices


def main():
    X_train, y_train, X_test, y_test = get_train_test_data()

    # Use solution to Challenge 1: Feature Scaling
    scaler = MinMaxScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = get_models()

    # Remove baseline models that aren't affected by feature reduction
    models.pop("Dummy - Most Frequent")
    models.pop("Dummy - Stratified")

    # Define set of PCA experiments
    feature_reduction_experiments = {
        "Do Nothing": Pipeline([("nothinger", DoNothingTransformer())]),
        "PCA - Layer 2": Pipeline(
            [
                (
                    "data_link_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["ARP", "LLC"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "PCA - Layer 3": Pipeline(
            [
                (
                    "network_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["ICMP", "IGMP", "IPv"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "PCA - Layer 4": Pipeline(
            [
                (
                    "transport_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["TCP", "UDP"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "PCA - Layer 7": Pipeline(
            [
                (
                    "application_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                [
                                    "HTTP",
                                    "HTTPS",
                                    "DNS",
                                    "Telnet",
                                    "SMTP",
                                    "SSH",
                                    "IRC",
                                    "DHCP",
                                ],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "PCA - Protocols": Pipeline(
            [
                (
                    "data_link_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["ARP", "LLC"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "network_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["ICMP", "IGMP", "IPv"],
                                    ["ARP", "LLC"],
                                    1,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "transport_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["TCP", "UDP"],
                                    ["ARP", "LLC", "ICMP", "IGMP", "IPv"],
                                    2,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "application_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    [
                                        "HTTP",
                                        "HTTPS",
                                        "DNS",
                                        "Telnet",
                                        "SMTP",
                                        "SSH",
                                        "IRC",
                                        "DHCP",
                                    ],
                                    ["ARP", "LLC", "ICMP", "IGMP", "IPv", "TCP", "UDP"],
                                    3,
                                ),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "PCA - Flags": Pipeline(
            [
                (
                    "syn_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["syn_flag_number", "syn_count"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "fin_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["fin_flag_number", "fin_count"],
                                    ["syn_flag_number", "syn_count"],
                                    1,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "rst_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["rst_flag_number", "rst_count"],
                                    [
                                        "syn_flag_number",
                                        "syn_count",
                                        "fin_flag_number",
                                        "fin_count",
                                    ],
                                    2,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "ack_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["ack_flag_number", "ack_count"],
                                    [
                                        "syn_flag_number",
                                        "syn_count",
                                        "fin_flag_number",
                                        "fin_count",
                                        "rst_flag_number",
                                        "rst_count",
                                    ],
                                    3,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "PCA - SYN Flag": Pipeline(
            [
                (
                    "syn_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["syn_flag_number", "syn_count"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "PCA - FIN Flag": Pipeline(
            [
                (
                    "fin_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["fin_flag_number", "fin_count"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "PCA - RST Flag": Pipeline(
            [
                (
                    "rst_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["rst_flag_number", "rst_count"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "PCA - ACK Flag": Pipeline(
            [
                (
                    "ack_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["ack_flag_number", "ack_count"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "PCA - All": Pipeline(
            [
                (
                    "data_link_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                ["ARP", "LLC"],
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "network_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["ICMP", "IGMP", "IPv"],
                                    ["ARP", "LLC"],
                                    1,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "transport_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["TCP", "UDP"],
                                    ["ARP", "LLC", "ICMP", "IGMP", "IPv"],
                                    2,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "application_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    [
                                        "HTTP",
                                        "HTTPS",
                                        "DNS",
                                        "Telnet",
                                        "SMTP",
                                        "SSH",
                                        "IRC",
                                        "DHCP",
                                    ],
                                    ["ARP", "LLC", "ICMP", "IGMP", "IPv", "TCP", "UDP"],
                                    3,
                                ),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "syn_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["syn_flag_number", "syn_count"],
                                    ["ARP", "LLC", "ICMP", "IGMP", "IPv", "TCP", "UDP", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "DHCP"],
                                    4,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "fin_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["fin_flag_number", "fin_count"],
                                    ["ARP", "LLC", "ICMP", "IGMP", "IPv", "TCP", "UDP", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "DHCP", "syn_flag_number", "syn_count"],
                                    5,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "rst_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["rst_flag_number", "rst_count"],
                                    [
                                        "ARP",
                                        "LLC",
                                        "ICMP",
                                        "IGMP",
                                        "IPv",
                                        "TCP",
                                        "UDP",
                                        "HTTP",
                                        "HTTPS",
                                        "DNS",
                                        "Telnet",
                                        "SMTP",
                                        "SSH",
                                        "IRC",
                                        "DHCP",
                                        "syn_flag_number",
                                        "syn_count",
                                        "fin_flag_number",
                                        "fin_count"
                                    ],
                                    6,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "ack_flags_pca",
                    ColumnTransformer(
                        [
                            (
                                "pca",
                                PCA(n_components=1),
                                get_adjusted_indices(
                                    X_train.columns,
                                    ["ack_flag_number", "ack_count"],
                                    [
                                        "ARP",
                                        "LLC",
                                        "ICMP",
                                        "IGMP",
                                        "IPv",
                                        "TCP",
                                        "UDP",
                                        "HTTP",
                                        "HTTPS",
                                        "DNS",
                                        "Telnet",
                                        "SMTP",
                                        "SSH",
                                        "IRC",
                                        "DHCP",
                                        "syn_flag_number",
                                        "syn_count",
                                        "fin_flag_number",
                                        "fin_count",
                                        "rst_flag_number",
                                        "rst_count",
                                    ],
                                    7,
                                ),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    # Run experiments and save results
    results = run_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        models,
        feature_reduction_experiments,
        stream_results=True,
    )
    results.to_csv("results/feature_reduction_results.csv")

    # Run sequential feature selection experiments and save results
    # results = run_sequential_feature_selection_experiments(
    #     models, X_train, y_train, X_test, y_test
    # )
    # save_feature_elimination_results(results, "sequential")

    # Run recursive feature selection experiments and save results
    results = run_recursive_feature_selection_experiments(
        models,
        models["Random Forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )
    save_feature_elimination_results(results, "recursive")


if __name__ == "__main__":
    main()
