
import numpy as np
import time
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from .feature_extraction import extract_features
from .preprocessing import adjust_predictions_by_width


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    train_event_widths,
    test_event_widths,
    feature_method="combined",
    noise_floor=-190.0,
):
    """Train and evaluate LightGBM model"""
    X_train_features = extract_features(X_train, feature_method)
    X_test_features = extract_features(X_test, feature_method)

    print(
        f"Feature shapes - Train: {X_train_features.shape}, Test: {X_test_features.shape}"
    )

    print("\n--- Training LightGBM model ---")
    model = lgbm.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=15,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
    )

    start_time = time.time()
    model.fit(X_train_features, y_train)
    train_time = time.time() - start_time

    y_pred_proba = model.predict_proba(X_test_features)[:, 1]
    y_pred_initial = (y_pred_proba > 0.5).astype(int)

    y_pred_adjusted, num_adjustments = adjust_predictions_by_width(
        X_test, y_pred_proba, test_event_widths, noise_floor
    )

    accuracy_initial = np.mean(y_pred_initial == y_test) * 100
    cm_initial = confusion_matrix(y_test, y_pred_initial)

    accuracy_adjusted = np.mean(y_pred_adjusted == y_test) * 100
    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)

    tn, fp, fn, tp = cm_adjusted.ravel()
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"Initial accuracy: {accuracy_initial:.2f}%")
    print(f"Width-adjusted accuracy: {accuracy_adjusted:.2f}%")
    print(
        f"Number of predictions adjusted: {num_adjustments} ({100 * num_adjustments / len(y_test):.2f}%)"
    )
    print(f"Sensitivity (BT recall): {sensitivity:.4f}")
    print(f"Specificity (WiFi recall): {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    print("Confusion Matrix (width-adjusted):")
    print(cm_adjusted)

    feature_importance = model.feature_importances_

    result = {
        "accuracy_initial": accuracy_initial,
        "accuracy_adjusted": accuracy_adjusted,
        "num_adjustments": num_adjustments,
        "adjustment_percentage": 100 * num_adjustments / len(y_test),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix_initial": cm_initial.tolist(),
        "confusion_matrix_adjusted": cm_adjusted.tolist(),
        "roc_auc": roc_auc,
        "training_time": train_time,
        "feature_importance": feature_importance.tolist(),
    }

    return model, result, X_test_features
