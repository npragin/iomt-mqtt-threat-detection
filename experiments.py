# Author: Noah Pragin
# Date: 2025-06-09
# Class: AI 541 - Machine Learning in the Real World - Dr. Kiri Wagstaff
# Description: This script runs several sets of experiments to address feature scaling and class imbalance.

import os

from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from experiments_helpers import (
    run_experiments,
    DoNothingTransformer,
    WinsorizerTransformer,
    SEED,
    get_train_test_data,
    get_models,
)


def main():
    X_train, y_train, X_test, y_test = get_train_test_data()

    models = get_models()

    # Define set of models that support class weighting at training time
    class_weighted_models = {
        "Logistic Regression - D1": LogisticRegression(
            class_weight="balanced", random_state=SEED
        ),
        "Logistic Regression - D3": Pipeline(
            [
                ("basis_expansion", PolynomialFeatures(degree=3)),
                (
                    "logistic_regression",
                    LogisticRegression(class_weight="balanced", random_state=SEED),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="log2",
            n_jobs=-1,
            class_weight="balanced",
            random_state=SEED,
        ),
        "SVM - RBF": SVC(shrinking=False, class_weight="balanced", random_state=SEED),
        "SVM - Polynomial D3": SVC(
            kernel="poly",
            max_iter=10000,
            class_weight="balanced",
            random_state=SEED,
        ),
    }

    # Define set of feature scaling experiments
    feature_scaling_experiments = {
        "Do Nothing": Pipeline([("nothinger", DoNothingTransformer())]),
        "Normalized": Pipeline([("scaler", MinMaxScaler())]),
        "Normalized - Winsorized": Pipeline(
            [("winsorizer", WinsorizerTransformer()), ("scaler", MinMaxScaler())]
        ),
        "Standardized": Pipeline([("scaler", StandardScaler())]),
        "Standardized - Winsorized": Pipeline(
            [("winsorizer", WinsorizerTransformer()), ("scaler", StandardScaler())]
        ),
    }

    # Define set of class imbalance experiments
    class_imbalance_experiments = {
        "Do Nothing": Pipeline([("nothinger", DoNothingTransformer())]),
        "Oversample Minority": Pipeline(
            [
                (
                    "oversampler",
                    RandomOverSampler(sampling_strategy="minority", random_state=SEED),
                )
            ]
        ),
        "Undersample Majority": Pipeline(
            [
                (
                    "undersampler",
                    RandomUnderSampler(sampling_strategy="majority", random_state=SEED),
                )
            ]
        ),
    }

    # Define set of class weighted class imbalance experiments
    class_weighted_class_imbalance_experiment = {
        "Class-Weighted Learning": Pipeline([("nothinger", DoNothingTransformer())]),
    }

    # Create dictionary of experiments to be saved separately
    experiments = {
        "feature_scaling": (models, feature_scaling_experiments),
        "class_imbalance": (models, class_imbalance_experiments),
        "class_weighted_class_imbalance": (
            class_weighted_models,
            class_weighted_class_imbalance_experiment,
        ),
    }

    # Run experiments and save results
    os.makedirs("results", exist_ok=True)
    for experiment_name, (models, experiments) in experiments.items():
        results = run_experiments(X_train, y_train, X_test, y_test, models, experiments)
        results.to_csv(f"results/{experiment_name}_results.csv")

        # Normalize the data for non-feature scaling experiments
        if experiment_name == "feature_scaling":
            X_train = MinMaxScaler().fit_transform(X_train)
            X_test = MinMaxScaler().fit_transform(X_test)


if __name__ == "__main__":
    main()
