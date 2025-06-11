# Identifying Malicious IoMT Network Traffic

**Author:** Noah Pragin  
**Instructor:** Dr. Kiri Wagstaff

Read the full report [here](assets/report.pdf)!

## Project Overview

This project implements machine learning techniques to improve the classification of IoMT (Internet of Medical Things) network traffic over MQTT as benign or malicious by addressing three key challenges:

1. **Challenge 1: Feature Scaling** - Comparing different normalization and standardization techniques
2. **Challenge 2: Class Imbalance** - Comparing oversampling, undersampling, and class weighting techniques
3. **Challenge 3: Feature Selection** - Reducing dimensionality while maintaining or improving classification performance

## Dataset
**CIC IoMT 2024 Dataset** - MQTT Traffic  
[Download Link](http://cicresearch.ca/IOTDataset/CICIoMT2024/Dataset/WiFI_and_MQTT/attacks/CSV/)

### Dataset Setup
1. Download all CSV files that start with `MQTT` and `Benign`
2. Organize files in the following directory structure:
   ```
   data/
   ├── csv/
   │   └── raw/
   │       ├── train/    # Place training CSV files here
   │       └── test/     # Place test CSV files here
   ```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Data Preparation
```bash
python data_processing.py
```
This combines all raw CSV files into compiled datasets for training and testing.

### Step 2 (Optional): Exploratory Data Analysis
```bash
python profiling.py
```
Generates comprehensive data profiles and visualizations.

### Step 3: Run Feature Scaling and Class Imbalance Experiments  
```bash
python experiments.py
```
Executes experiments and saves results comparing different feature scaling and class imbalance techniques.

### Step 4: Run Feature Selection Experiments
```bash
python feature_reduction_experiments.py
```
Runs feature selection experiments to evaluate dimensionality reduction techniques and saves results and plots comparing different techniques.

## Detailed Usage

### `data_processing.py`
**Purpose:** Data preprocessing and compilation

**What it does:**
- Combines all CSV files from `data/csv/raw/train/` into `compiled-data-train.csv`
- Combines all CSV files from `data/csv/raw/test/` into `compiled-data-test.csv`  
- Saves compiled datasets to `data/csv/compiled/train/` and `data/csv/compiled/test/`

**Output:** Clean, compiled datasets ready for experiments

---

### `experiments.py`
**Purpose:** Run experiments to compare techniques for addressing feature scaling and class imbalance.

**What it does:**
- **Feature Scaling:** Tests multiple feature scaling techniques (Standardization, Normalization, Winsorization)
- **Class Imbalance:** Evaluates class imbalance handling methods (Oversampling, Undersampling, Class Weighting)
- Compares classification performance across different algorithms and techniques
- Saves accuracy, precision, recall, and F1 scores to CSV files for analysis and plotting

**Output:** Performance metrics and comparison tables for scaling and imbalance techniques

---

### `feature_reduction_experiments.py`
**Purpose:** Run experiments to compare techniques for addressing feature selection.

**What it does:**
- **Sequential Feature Selection:** Tests forward and backward feature selection using validation scores as a selection criterion
- **Recursive Feature Selection:** Tests backward feature selection using a random forest as a feature importance estimator
- **PCA:** Tests column-specific PCA for dimensionality reduction leveraging known relationships between features
- Saves accuracy scores and order of feature selection/elimination for each model and technique to CSV files for analysis and plotting
- Plots accuracy scores for each model and technique over feature selection/elimination steps
- **Note:** Sequential feature selection experiments are commented out due to runtime.

**Output:** Feature selection performance comparisons and plots

---

### `profiling.py`
**Purpose:** Exploratory Data Analysis (EDA)

**What it does:**
- Generates comprehensive HTML reports for train, test, and combined datasets using `ydata-profiling`
- Creates overlaid histograms comparing feature distributions of benign vs. malicious vs. all traffic
- Generates violin plots for feature distributions with special handling for `Header_Length` and `Rate` features (winsorized to top 5%)
- Creates numeric feature statistics table saved as CSV in `stats/` directory

**Output:** 
- HTML profiles in `profiles/` directory
- Distribution plots in `histograms/` directory  
- Statistical summaries in `stats/` directory

---

### `experiments_helpers.py`
**Purpose:** Utility functions and configuration

**What it does:**
- Contains helper functions for experiment execution
- Manages data loading and preprocessing pipelines
- Core function: `run_experiments()` orchestrates the experimental workflow
- Implements a Winsorizer transformer to handle winsorization in a `sklearn.Pipeline`
- Implements a custom sequential and recursive feature selection functions that track performance for each feature selection/elimination step

**Configuration:**
- `TRAIN_DATA_PATH` and `TEST_DATA_PATH` can be changed to the new paths if the compiled CSV files are moved to a different directory.
- `get_models` can be modified to add or remove models from the experiments.
- Want to run your own experiments with the models defined in `get_models`? Create a dictionary of experiment names and `sklearn.Pipeline` objects and pass it to `run_experiments`!
