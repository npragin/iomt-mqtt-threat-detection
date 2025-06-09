# Author: Noah Pragin
# Date: 2025-06-09
# Class: AI 541 - Machine Learning in the Real World - Dr. Kiri Wagstaff
# Description: This script contains helper functions for the experiments.py script.

import os
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold


SEED = 42
TRAIN_DATA_PATH = "data/csv/compiled/train/compiled-data-train.csv"
TEST_DATA_PATH = "data/csv/compiled/test/compiled-data-test.csv"


######################################
## Data Loading and Model Definition #
######################################


def get_X_y_df(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV file and separate features from labels.

    This function reads a CSV file containing labeled data and splits it into
    feature matrix (X) and target vector (y). The function assumes the CSV
    contains a 'Label' column that serves as the target variable, with all
    other columns treated as features.

    Args:
        csv_path (str): Path to the CSV file containing the dataset.
                       Expected to have a 'Label' column for the target variable.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): Feature matrix with all columns except 'Label'
            - y (pd.Series): Target vector containing the 'Label' column values
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return X, y


def get_train_test_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load the training and test data from the CSV files and split into feature matrices and target vectors.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            X_train, y_train, X_test, y_test
    """
    X_train, y_train = get_X_y_df(TRAIN_DATA_PATH)
    X_test, y_test = get_X_y_df(TEST_DATA_PATH)
    return X_train, y_train, X_test, y_test


def get_models() -> dict:
    """
    Get a dictionary of models to use for the experiments.

    Returns:
        dict: A dictionary of model names and model objects
    """
    models = {
        "Dummy - Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Dummy - Stratified": DummyClassifier(strategy="stratified", random_state=SEED),
        "Logistic Regression - D1": LogisticRegression(random_state=SEED),
        "Logistic Regression - D3": Pipeline(
            [
                ("basis_expansion", PolynomialFeatures(degree=3)),
                (
                    "logistic_regression",
                    LogisticRegression(random_state=SEED),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="log2", n_jobs=-1, random_state=SEED
        ),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "SVM - RBF": SVC(shrinking=False, random_state=SEED),
        "SVM - Polynomial D3": SVC(kernel="poly", max_iter=10000, random_state=SEED),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 64, 64),
            max_iter=25,
            random_state=SEED,
        ),
    }

    return models


def run_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict,
    experiments: dict,
    stream_results: bool = False,
) -> pd.DataFrame:
    """
    Run experiments on a given dataset using different preprocessing and inference pipelines.

    This function iterates through a dictionary of inference pipelines and preprocessing pipelines,
    fitting each model on the training data and evaluating its performance on the test
    data. The results are stored in a pandas DataFrame and optionally saved to a CSV file.

    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target vector
        models (dict): Dictionary of model names and models or inference pipelines
        experiments (dict): Dictionary of experiment names and preprocessing pipelines
        stream_results (bool): Whether to stream the results to a CSV file `results/streamed_results.csv`

    Returns:
        pd.DataFrame: A DataFrame containing the results of the experiments
    """
    if stream_results:
        os.makedirs("results", exist_ok=True)

    results = pd.DataFrame(index=models.keys(), columns=experiments.keys())

    for model_name, model in models.items():
        for experiment_name, experiment in experiments.items():
            print(f"Running {experiment_name} with {model_name}")

            if isinstance(model, Pipeline):
                pipeline = Pipeline(experiment.steps + model.steps)
            else:
                pipeline = Pipeline(experiment.steps + [("model", model)])

            t = time()

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=np.nan
            )
            results.loc[model_name, experiment_name] = (
                f"F: {report['weighted avg']['f1-score']*100:.2f}\nP: {report['weighted avg']['precision']*100:.2f}\nR: {report['weighted avg']['recall']*100:.2f}\nA: {report['accuracy']*100:.2f}"
            )

            print(f"Time taken: {time() - t:.2f} seconds", end="\n\n")

        if stream_results:
            results.to_csv("results/streamed_results.csv")

    return results


##########################
## Feature Scaling Stuff #
##########################


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, limits: tuple[float, float] = (0.05, 0.05)):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.apply_along_axis(
            lambda x: winsorize(x, limits=self.limits), axis=0, arr=X
        )


class DoNothingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


##############################
## Feature Elimination Stuff #
##############################


def sequential_feature_selection_with_scores(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_features: int,
    direction: str = "forward",
    cv_folds: int = 5,
    scoring: str = "accuracy",
) -> tuple[list[int], list[int], list[float]]:
    """
    Run sequential feature selection with cross-validation and return the selected features, removed features, and test scores in order of feature elimination.

    Args:
        model: The model to use for the experiments
        X_train: The training feature matrix
        y_train: The training target vector
        X_test: The test feature matrix
        y_test: The test target vector
        target_features: The number of features to select
        direction: The direction of feature selection (forward or backward)
        cv_folds: The number of folds for cross-validation
        scoring: The scoring metric to use

    Returns:
        tuple[list[int], list[int], list[float]]: The selected features, removed features, and test scores in order of feature elimination
    """
    cross_val = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    n_features = X_train.shape[1]
    selected_features = [] if direction == "forward" else list(range(n_features))
    remaining_features = list(range(n_features))
    removed_features = []
    test_scores = []

    while len(selected_features) != target_features:
        best_val_score = -np.inf
        best_feature = None

        for feature in remaining_features:
            if direction == "forward":
                current_features = selected_features + [feature]
            elif direction == "backward":
                current_features = [f for f in remaining_features if f != feature]
            else:
                raise ValueError(f"Invalid direction: {direction}")

            val_score = cross_val_score(
                model,
                X_train.iloc[:, current_features],
                y_train,
                cv=cross_val,
                scoring=scoring,
                n_jobs=-1,
            ).mean()

            if val_score > best_val_score:
                best_val_score = val_score
                best_feature = feature

        if direction == "forward":
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            remaining_features.remove(best_feature)
            selected_features.remove(best_feature)
            removed_features.append(best_feature)

        model.fit(X_train.iloc[:, selected_features], y_train)
        test_score = model.score(X_test.iloc[:, selected_features], y_test)
        test_scores.append(test_score)

        print(
            f"{'Dropping' if direction == 'backward' else 'Adding'} feature {best_feature} with score {best_val_score} on val and {test_score} on test"
        )

    if direction == "forward":
        removed_features = remaining_features

    return selected_features, removed_features, test_scores


def recursive_feature_selection_with_scores(
    model,
    feature_importance_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_features: int,
) -> tuple[list[int], list[int], list[float]]:
    """
    Run recursive feature selection using an estimator as a feature importance estimator and return the selected features, removed features, and test scores in order of feature elimination.

    Args:
        model: The model to use for the experiments
        feature_importance_estimator: The feature importance estimator to use for the experiments
        X_train: The training feature matrix
        y_train: The training target vector
        X_test: The test feature matrix
        y_test: The test target vector
        target_features: The number of features to select

    Returns:
        tuple[list[int], list[int], list[float]]: The selected features, removed features, and test scores in order of feature elimination
    """
    n_features = X_train.shape[1]
    selected_features = list(range(n_features))
    removed_features = []
    test_scores = []

    while len(selected_features) != target_features:
        feature_importance_estimator.fit(X_train.iloc[:, selected_features], y_train)

        if hasattr(feature_importance_estimator, "feature_importances_"):
            feature_importances = feature_importance_estimator.feature_importances_
        elif hasattr(feature_importance_estimator, "coef_"):
            feature_importances = feature_importance_estimator.coef_
        else:
            raise ValueError(
                "Feature importance estimator must have either feature_importances_ or coef_ attribute"
            )

        subset_idx = np.argmin(feature_importances)
        least_important_feature = selected_features[subset_idx]

        selected_features.remove(least_important_feature)
        removed_features.append(least_important_feature)

        model.fit(X_train.iloc[:, selected_features], y_train)
        test_score = model.score(X_test.iloc[:, selected_features], y_test)
        test_scores.append(test_score)

        print(
            f"Dropping feature {least_important_feature} with importance {feature_importances[subset_idx]} and {test_score} score on test"
        )

    return selected_features, removed_features, test_scores


def run_sequential_feature_selection_experiments(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Run sequential feature selection from all features to 1 and from 1 to all features and return the results for each across all models.

    Args:
        models: The models to use for the experiments
        X_train: The training feature matrix
        y_train: The training target vector
        X_test: The test feature matrix
        y_test: The test target vector

    Returns:
        pd.DataFrame: A DataFrame containing the results of the experiments
    """
    results = pd.DataFrame(
        index=models.keys(), columns=["Sequential - Forward", "Sequential - Backward"]
    )

    for model_name, model in models.items():
        for direction in ["forward", "backward"]:
            print(f"Running {model_name} - Sequential - {direction.capitalize()}")
            t = time()
            result = sequential_feature_selection_with_scores(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                target_features=1 if direction == "backward" else X_train.shape[1],
                direction=direction,
            )
            results.loc[model_name, f"Sequential - {direction.capitalize()}"] = result
            print(f"Time taken: {time() - t:.2f} seconds", end="\n\n")

    return results


def run_recursive_feature_selection_experiments(
    models: dict,
    feature_importance_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Run recursive feature selection from all features to 1 and return the results for each across all models.

    Args:
        models: The models to use for the experiments
        feature_importance_estimator: The feature importance estimator to use for the experiments
        X_train: The training feature matrix
        y_train: The training target vector
        X_test: The test feature matrix
        y_test: The test target vector

    Returns:
        pd.DataFrame: A DataFrame containing the results of the experiments
    """
    results = pd.DataFrame(index=models.keys(), columns=["Recursive - Backward"])

    for model_name, model in models.items():
        print(f"Running {model_name} - Recursive - Backward")

        t = time()

        result = recursive_feature_selection_with_scores(
            model,
            feature_importance_estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            target_features=1,
        )
        results.at[model_name, "Recursive - Backward"] = result

        print(f"Time taken: {time() - t:.2f} seconds", end="\n\n")

    return results


def save_feature_elimination_results(
    results: pd.DataFrame, experiment_name: str
) -> None:
    """
    Save the results of the feature elimination experiments to CSV files. Saves one dataframe containing the scores for each model
    as features are added/removed and one dataframe containing the scores, active features, and inactive features for each model.
    Also saves a plot of the scores for each model over feature selection steps.

    Args:
        results: The results of the feature elimination experiments
        experiment_name: The name of the experiment
    """
    score_results = pd.DataFrame(index=results.index, columns=results.columns)
    for index, row in results.iterrows():
        os.makedirs(f"results/feature_elimination_plots/{index}/", exist_ok=True)
        for col_name in results.columns:
            path = f"results/feature_elimination_plots/{index}/{col_name}.png"
            _, _, scores = row[col_name]

            plt.figure(figsize=(10, 5))
            plt.plot(scores)
            plt.title(f"{col_name} - {index}")
            plt.xlabel("Step")
            plt.ylabel("Accuracy Score")
            plt.savefig(path)
            plt.close()

            score_results.loc[index, col_name] = max(scores)

    score_results.to_csv(
        f"results/feature_elimination_score_results_{experiment_name}.csv"
    )
    results.to_csv(f"results/feature_elimination_results_{experiment_name}.csv")
