# RF Signal Classifier: WiFi vs. Bluetooth

This project implements a machine learning pipeline to classify Radio Frequency (RF) signals as either WiFi or Bluetooth. It utilizes Power Spectral Density (PSD) data, extracts various spectral and bandwidth-related features, and trains a LightGBM classifier. The process includes event detection for signal isolation and a width-based heuristic to refine predictions.

## Project Structure

```
.
├── MLRF_1.5.py                 # Main script to run the classification pipeline
├── data/
│   └── dataset.h5              # Placeholder for the HDF5 data file
├── MLRF/
│   ├── __init__.py             # Makes MLRF a Python package
│   ├── utils.py                # Core utility functions (e.g., event_detector)
│   ├── preprocessing.py        # Data preprocessing (event detection, width adjustment)
│   ├── data_loader.py          # Loading and splitting HDF5 data
│   ├── feature_extraction_chunks.py # Worker functions for parallel feature extraction
│   ├── feature_extraction.py   # Main parallelized feature extraction logic
│   ├── model_trainer.py        # Training and evaluation of the LightGBM model
│   ├── output_generator.py     # Creating output files (model, plots, metadata)
│   └── classifier.py           # Deployable RFClassifier class
├── lightgbm_model_*.joblib     # Saved trained LightGBM model
├── confusion_matrix_*.png      # Saved confusion matrix plot
└── model_metadata_*.json       # Saved metadata including performance and parameters
```

## Pipeline Steps

The classification process involves several key stages:

### 1. Data Loading and Initial Preparation (`MLRF/data_loader.py`)

1.  **Load HDF5 Data:**
    *   The script reads PSD data from an HDF5 file (e.g., `dataset.h5`). Each key in the file typically represents a signal sample.
    *   Labels are assigned based on the key prefix (e.g., "wifi" -> 0, "bt" -> 1).
2.  **Noise Floor Calculation:**
    *   A global noise floor is calculated by finding the minimum valid (non-infinite) dB value across all samples in the dataset.
    *   `-inf` values in the data are replaced with a value 10dB below this calculated noise floor. This ensures numerical stability and provides a baseline for low-power regions.
    *   *(Note: For rigorous evaluation, this noise floor should ideally be calculated only from the training set and then applied to the test set to prevent data leakage.)*

### 2. Preprocessing with Event Detection (`MLRF/preprocessing.py`)

1.  **Event Detection (`event_detector` in `MLRF/utils.py`):**
    *   For each PSD sample, this function identifies "events" or regions where the signal power significantly exceeds a dynamic threshold.
    *   The dynamic threshold is calculated as a factor of the mean power of the entire PSD sample.
    *   A sliding window approach is used. If the average power within a window is above the threshold, an event is considered active.
    *   The start and end indices of these detected events are recorded.
2.  **Signal Masking:**
    *   A new representation of each sample is created. It's initialized with the `noise_floor - 10dB` value.
    *   Only the data within the detected event ranges is copied from the original sample into this new masked representation. Regions outside events remain at the noise floor level.
    *   The total width (in bins) of all detected events for each sample is also calculated and stored (`event_widths`).
3.  **Multiprocessing:**
    *   This preprocessing step (event detection and masking for each sample) is parallelized using `multiprocessing.Pool` to improve performance on multi-core CPUs. Each sample is processed by a separate worker.

### 3. Data Splitting (`MLRF/data_loader.py`)

1.  **Train-Test Split:**
    *   The preprocessed data (masked signals) and their corresponding labels are split into training and testing sets using `sklearn.model_selection.train_test_split`.
    *   The `stratify=labels` option is used to ensure that the class distribution (WiFi vs. Bluetooth) is approximately the same in both the training and testing sets.
    *   The `event_widths` calculated in the previous step are also split accordingly to match the train and test samples.

### 4. Feature Extraction (`MLRF/feature_extraction.py` & `MLRF/feature_extraction_chunks.py`)

This is a crucial step where meaningful characteristics are derived from the preprocessed PSD signals. Three feature extraction methods are available: `spectral_shape`, `bandwidth`, or `combined` (default). Multiprocessing is used here as well for efficiency.

1.  **Spectral Shape Features (`extract_spectral_shape_features_chunk`):**
    *   **Peak Analysis:**
        *   Detects peaks in the signal using `scipy.signal.find_peaks`.
        *   Calculates statistics of peak widths at different relative heights (e.g., 3dB, 6dB, 10dB below the peak).
        *   Counts peaks resembling typical Bluetooth and WiFi bandwidths.
        *   Calculates peak width-to-height ratios.
        *   Creates a histogram of peak widths.
        *   Identifies the widths of the top 3 most prominent peaks.
    *   **Energy Distribution:**
        *   Calculates the energy within sliding windows of typical WiFi and Bluetooth bandwidths.
        *   Computes statistics (max, std) of these energies and their ratio.
    *   **Template Correlation:**
        *   Correlates segments of the signal with simplified Gaussian templates representing WiFi and Bluetooth spectral shapes.
        *   Statistics (max, mean) of these correlation coefficients are used as features.
    *   **Basic Statistics:** Mean, standard deviation, max, min, median, and quartiles of the signal.

2.  **Segmented Bandwidth Features (`extract_bandwidth_features_chunk`):**
    *   The signal is divided into a fixed number of segments (e.g., 8).
    *   For each segment:
        *   Peak detection is performed.
        *   Counts of Bluetooth-like and WiFi-like peak widths.
        *   Average peak width.
        *   Average peak height-to-width ratio.
        *   Ratio of energy in peak regions to the total energy in the segment.

3.  **Combined Features:**
    *   If `feature_method='combined'`, both spectral shape and segmented bandwidth features are extracted and concatenated horizontally to form the final feature vector for each sample.

### 5. Model Training (`MLRF/model_trainer.py`)

1.  **LightGBM Classifier:**
    *   A LightGBM (Light Gradient Boosting Machine) classifier (`lgbm.LGBMClassifier`) is initialized with a set of pre-defined hyperparameters (e.g., `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, regularization parameters).
2.  **Training:**
    *   The model is trained using the extracted features from the training set (`X_train_features`) and the corresponding training labels (`y_train`).

### 6. Model Evaluation and Prediction Adjustment (`MLRF/model_trainer.py` & `MLRF/preprocessing.py`)

1.  **Initial Prediction:**
    *   The trained model predicts probabilities for the test set features (`X_test_features`).
    *   Initial class predictions are made by thresholding these probabilities at 0.5.
2.  **Width-Based Prediction Adjustment (`adjust_predictions_by_width`):**
    *   A heuristic is applied to potentially adjust the model's predictions based on the `event_widths` of the test samples.
    *   A `width_threshold` (e.g., corresponding to ~10 MHz) is defined.
    *   If a model's prediction confidence is not very high (e.g., probability is between 0.15 and 0.85):
        *   If the `event_width` is less than the `width_threshold`, the prediction is nudged towards Bluetooth (class 1).
        *   If the `event_width` is greater than or equal to the `width_threshold`, the prediction is nudged towards WiFi (class 0).
3.  **Performance Metrics:**
    *   Accuracy is calculated for both initial and width-adjusted predictions.
    *   A confusion matrix is generated for both.
    *   Other metrics for the adjusted predictions include:
        *   Sensitivity (True Positive Rate for Bluetooth)
        *   Specificity (True Positive Rate for WiFi)
        *   F1 Score
        *   ROC AUC (Area Under the Receiver Operating Characteristic Curve), calculated using the original prediction probabilities.
    *   The number and percentage of adjustments made by the width-based heuristic are reported.

### 7. Output Generation (`MLRF/output_generator.py`)

1.  **Save Trained Model:**
    *   The trained LightGBM model object is serialized and saved to a `.joblib` file (e.g., `lightgbm_model_{timestamp}.joblib`).
2.  **Save Confusion Matrix Plot:**
    *   A plot comparing the initial and width-adjusted confusion matrices is generated using `matplotlib` and saved as a PNG file (e.g., `confusion_matrix_{timestamp}.png`).
3.  **Save Metadata:**
    *   A JSON file (e.g., `model_metadata_{timestamp}.json`) is created containing:
        *   Timestamp of the run.
        *   Feature extraction method used.
        *   Paths to the saved model and confusion matrix files.
        *   Detailed performance metrics.
        *   Key preprocessing parameters (e.g., noise floor, width threshold).
        *   LightGBM model parameters.
        *   Top 10 most important features and their importance scores.
        *   Total number of features used.

### 8. Deployable Classifier (`MLRF/classifier.py`)

*   The `RFClassifier` class encapsulates the trained model and the necessary preprocessing and feature extraction logic (without multiprocessing for single instance predictions).
*   It provides `predict()` and `predict_proba()` methods that can take raw PSD data as input, apply the learned noise floor, perform event detection, extract features, and return predictions. This class is designed for easier deployment or use in other applications.

## How to Run

1.  **Ensure Dependencies:** Install necessary Python libraries:
    ```bash
    pip install numpy h5py tqdm scikit-learn lightgbm joblib matplotlib
    ```
2.  **Prepare Data:** Place your HDF5 data file (e.g., `dataset.h5`) in the `./data/` directory. The keys in the H5 file should be named such that WiFi samples start with "wifi" and Bluetooth samples start with "bt" (or adjust the labeling logic in `MLRF/data_loader.py`).
3.  **Run the Main Script:**
    ```bash
    python MLRF_1.5.py
    ```
4.  **Outputs:** The script will generate the model file, confusion matrix image, and metadata JSON file in the root directory, timestamped to avoid overwriting previous runs.

## Customization

*   **Feature Method:** Modify the `feature_method` variable in `MLRF_1.5.py` to 'spectral_shape', 'bandwidth', or 'combined'.
*   **Model Hyperparameters:** Adjust LightGBM parameters in `MLRF/model_trainer.py`.
*   **Data Path:** Change the `data_path` variable in `MLRF_1.5.py` if your data file is located elsewhere.
