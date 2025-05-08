import warnings
from MLRF.data_loader import load_and_split_data
from MLRF.model_trainer import train_and_evaluate
from MLRF.output_generator import create_outputs

warnings.filterwarnings("ignore")


def run_rf_classification(data_path, feature_method="combined"):
    """Main function to run the RF classification pipeline using LightGBM"""
    print("=" * 80)
    print("RF Signal Classifier - WiFi vs Bluetooth using LightGBM")
    print("=" * 80)

    # Load and split data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_event_widths,
        test_event_widths,
        noise_floor,
    ) = load_and_split_data(data_path)

    print("\n" + "=" * 80)
    print(f"Feature Extraction Method: {feature_method}")
    print("=" * 80)

    # Train and evaluate model
    model, result, X_test_features = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        train_event_widths,
        test_event_widths,
        feature_method,
        noise_floor,
    )

    # Create output files
    model_path, cm_path, metadata_path = create_outputs(
        model, result, feature_method, X_test_features, noise_floor
    )

    print("\n" + "=" * 80)
    print("Classification Complete")
    print("=" * 80)
    print("Output files:")
    print(f"1. Model: {model_path}")
    print(f"2. Confusion Matrix: {cm_path}")
    print(f"3. Metadata: {metadata_path}")

    return model, result


if __name__ == "__main__":
    # Define path to data file
    data_path = "./data/dataset.h5"  # Replace with your actual h5 file

    # Feature extraction method - default to combined for best performance
    feature_method = "combined"  # Options: 'spectral_shape', 'bandwidth', 'combined'

    run_rf_classification(data_path, feature_method)



