
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def create_outputs(
    model, result, feature_method, X_test_features, noise_floor
):
    """Create the three required output files: model, confusion matrix, and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"lightgbm_model_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    cm_initial = np.array(result["confusion_matrix_initial"])
    plt.imshow(cm_initial, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Initial Confusion Matrix")
    plt.colorbar()
    classes = ["WiFi", "Bluetooth"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm_initial.max() / 2
    for i in range(cm_initial.shape[0]):
        for j in range(cm_initial.shape[1]):
            plt.text(
                j,
                i,
                format(cm_initial[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_initial[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.subplot(1, 2, 2)
    cm_adjusted = np.array(result["confusion_matrix_adjusted"])
    plt.imshow(cm_adjusted, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Width-Adjusted Confusion Matrix")
    plt.colorbar()
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm_adjusted.max() / 2
    for i in range(cm_adjusted.shape[0]):
        for j in range(cm_adjusted.shape[1]):
            plt.text(
                j,
                i,
                format(cm_adjusted[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_adjusted[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(cm_filename, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {cm_filename}")

    feature_importance = np.array(result["feature_importance"])
    if feature_method == "spectral_shape":
        feature_names = [
            "Num Peaks", "Mean 3dB Width", "Std 3dB Width",
            "Mean 6dB Width", "Std 6dB Width", "Mean 10dB Width", "Std 10dB Width",
            "BT-like Peaks", "WiFi-like Peaks", "Wide Peaks",
            "Mean Width/Height", "Max Width/Height",
        ]
        feature_names.extend([f"Width Hist {i}" for i in range(7)])
        feature_names.extend([f"Top Peak Width {i}" for i in range(3)])
        feature_names.extend([
            "Max WiFi Energy", "Std WiFi Energy", "Max BT Energy", "Std BT Energy",
            "WiFi/BT Energy Ratio",
        ])
        feature_names.extend([
            "Max WiFi Corr", "Mean WiFi Corr", "Max BT Corr", "Mean BT Corr",
        ])
        feature_names.extend([
            "Mean", "Std", "Max", "Min", "Median", "P25", "P75",
        ])
    elif feature_method == "bandwidth":
        segments = 8
        segment_features = [
            "BT Count", "WiFi Count", "Avg Width", "H/W Ratio", "Energy Ratio",
        ]
        feature_names = [
            f"Segment {i+1} {f}"
            for i in range(segments)
            for f in segment_features
        ]
    elif feature_method == "combined":
        spectral_names = [
            "Num Peaks", "Mean 3dB Width", "Std 3dB Width",
            "Mean 6dB Width", "Std 6dB Width", "Mean 10dB Width", "Std 10dB Width",
            "BT-like Peaks", "WiFi-like Peaks", "Wide Peaks",
            "Mean Width/Height", "Max Width/Height",
        ]
        spectral_names.extend([f"Width Hist {i}" for i in range(7)])
        spectral_names.extend([f"Top Peak Width {i}" for i in range(3)])
        spectral_names.extend([
            "Max WiFi Energy", "Std WiFi Energy", "Max BT Energy", "Std BT Energy",
            "WiFi/BT Energy Ratio",
        ])
        spectral_names.extend([
            "Max WiFi Corr", "Mean WiFi Corr", "Max BT Corr", "Mean BT Corr",
        ])
        spectral_names.extend([
            "Mean", "Std", "Max", "Min", "Median", "P25", "P75",
        ])
        segments = 8
        segment_features = [
            "BT Count", "WiFi Count", "Avg Width", "H/W Ratio", "Energy Ratio",
        ]
        bandwidth_names = [
            f"Segment {i+1} {f}"
            for i in range(segments)
            for f in segment_features
        ]
        feature_names = spectral_names + bandwidth_names
    else:
        feature_names = [
            f"Feature {i}" for i in range(len(feature_importance))
        ]

    if len(feature_names) > len(feature_importance):
        feature_names = feature_names[: len(feature_importance)]
    elif len(feature_names) < len(feature_importance):
        feature_names.extend(
            [
                f"Feature {i}"
                for i in range(
                    len(feature_names), len(feature_importance)
                )
            ]
        )

    top_indices = np.argsort(feature_importance)[::-1][:10]
    top_features = [
        (feature_names[i], float(feature_importance[i])) for i in top_indices
    ]

    model_params = {
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "num_leaves": model.num_leaves,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
        "reg_alpha": model.reg_alpha,
        "reg_lambda": model.reg_lambda,
    }

    metadata = {
        "timestamp": timestamp,
        "feature_method": feature_method,
        "model_file": model_filename,
        "confusion_matrix_file": cm_filename,
        "performance": {
            "accuracy_initial": float(result["accuracy_initial"]),
            "accuracy_adjusted": float(result["accuracy_adjusted"]),
            "num_predictions_adjusted": int(result["num_adjustments"]),
            "adjustment_percentage": float(result["adjustment_percentage"]),
            "sensitivity": float(result["sensitivity"]),
            "specificity": float(result["specificity"]),
            "f1_score": float(result["f1"]),
            "roc_auc": float(result["roc_auc"]),
            "training_time_seconds": float(result["training_time"]),
        },
        "preprocessing": {
            "event_detection": True,
            "noise_floor": float(noise_floor),
            "width_based_adjustment": True,
            "width_threshold_mhz": 10.0,
        },
        "model_parameters": model_params,
        "top_features": top_features,
        "feature_dimensions": X_test_features.shape[1],
    }

    metadata_filename = f"model_metadata_{timestamp}.json"
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_filename}")

    return model_filename, cm_filename, metadata_filename
