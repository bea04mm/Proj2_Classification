"""
Training Script for Classic ML Models - Part I.2.a)
Extracts hand landmarks using MediaPipe and trains SVM, KNN, and Random Forest classifiers
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class HandGestureDataExtractor:
    """Extract hand landmarks from video clips"""

    def __init__(self, dataset_path, use_z_coord=False):
        """
        Initialize the data extractor

        Args:
            dataset_path: Path to the Videos_Datasets folder
            use_z_coord: Whether to use (x,y,z) or just (x,y) coordinates
        """
        self.dataset_path = Path(dataset_path)
        self.use_z_coord = use_z_coord
        self.gesture_mapping = {
            "open_palm": 0,
            "closed_hand": 1,
            "v_shape": 2,
            "hand_right": 3,
            "hand_left": 4,
            "thumbs_up": 5,
        }

    def extract_landmarks_from_video(self, video_path):
        """
        Extract hand landmarks from a video file

        Args:
            video_path: Path to the video file

        Returns:
            List of landmark features (flattened arrays)
        """
        cap = cv2.VideoCapture(str(video_path))
        features = []

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame
                results = hands.process(frame_rgb)

                # Extract landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract coordinates
                        if self.use_z_coord:
                            landmarks = [
                                [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                            ]
                        else:
                            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                        # Flatten the landmarks array
                        features.append(np.array(landmarks).flatten())

        cap.release()
        return features

    def load_dataset(self, train_clips=[1, 2, 3, 4], test_clips=[5, 6]):
        """
        Load all video clips and extract features

        Args:
            train_clips: Indices of clips to use for training (1-6)
            test_clips: Indices of clips to use for testing (1-6)

        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        print("Loading dataset and extracting features...")

        # Iterate through all subjects
        for subject_folder in self.dataset_path.iterdir():
            if not subject_folder.is_dir() or subject_folder.name == "__pycache__":
                continue

            print(f"\nProcessing subject: {subject_folder.name}")

            # Iterate through gestures
            for gesture_name, gesture_label in self.gesture_mapping.items():
                gesture_folder = subject_folder / gesture_name

                if not gesture_folder.exists():
                    continue

                print(f"  - Gesture: {gesture_name}")

                # Get all video files in the gesture folder
                video_files = sorted(
                    list(gesture_folder.glob("*.avi"))
                    + list(gesture_folder.glob("*.mp4"))
                )

                # Process each video clip
                for idx, video_file in enumerate(video_files, start=1):
                    print(f"    Processing clip {idx}: {video_file.name}")

                    # Extract features from video
                    features = self.extract_landmarks_from_video(video_file)

                    if not features:
                        print(
                            f"    Warning: No landmarks detected in {video_file.name}"
                        )
                        continue

                    # Assign to train or test set based on clip index
                    if idx in train_clips:
                        X_train.extend(features)
                        y_train.extend([gesture_label] * len(features))
                    elif idx in test_clips:
                        X_test.extend(features)
                        y_test.extend([gesture_label] * len(features))

        print(f"\nDataset loaded:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir="models"):
    """
    Train SVM, KNN, and Random Forest models

    Args:
        X_train, X_test, y_train, y_test: Training and testing data
        output_dir: Directory to save trained models

    Returns:
        Dictionary of trained models
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, output_path / "scaler.pkl")
    print(f"Scaler saved to {output_path / 'scaler.pkl'}")

    # Define models
    models = {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42
        ),
    }

    trained_models = {}
    results = {}

    gesture_names = [
        "open_palm",
        "closed_hand",
        "v_shape",
        "hand_right",
        "hand_left",
        "thumbs_up",
    ]

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=gesture_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=gesture_names,
            yticklabels=gesture_names,
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_path / f"confusion_matrix_{model_name}.png")
        print(
            f"Confusion matrix saved to {output_path / f'confusion_matrix_{model_name}.png'}"
        )
        plt.close()

        # Save model
        model_filename = output_path / f"{model_name.lower()}_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

        trained_models[model_name] = model
        results[model_name] = {"accuracy": accuracy, "confusion_matrix": cm}

    # Compare models
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name:15s}: Accuracy = {result['accuracy']:.4f}")

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]

    plt.bar(model_names, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Accuracy")
    plt.title("Model Comparison - Test Accuracy")
    plt.ylim([0, 1])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png")
    print(f"\nComparison plot saved to {output_path / 'model_comparison.png'}")
    plt.close()

    return trained_models, scaler


def main():
    """Main function to run the training pipeline"""

    # Configuration
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "models"
    USE_Z_COORD = False  # Set to True to use (x,y,z) instead of (x,y)

    # Training/Testing split (clip indices)
    TRAIN_CLIPS = [1, 2, 3, 4]  # Clips 1-4 for training
    TEST_CLIPS = [5, 6]  # Clips 5-6 for testing

    print("=" * 60)
    print("Hand Gesture Classification - Training Script")
    print("Part I.2.a) - Classic ML Models")
    print("=" * 60)
    print(f"\nDataset path: {DATASET_PATH}")
    print(f"Using coordinates: {'(x, y, z)' if USE_Z_COORD else '(x, y)'}")
    print(f"Training clips: {TRAIN_CLIPS}")
    print(f"Testing clips: {TEST_CLIPS}")

    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"\nError: Dataset path does not exist: {DATASET_PATH}")
        print("Please ensure the Videos_Datasets folder is in the correct location.")
        return

    # Extract features from dataset
    extractor = HandGestureDataExtractor(DATASET_PATH, use_z_coord=USE_Z_COORD)
    X_train, X_test, y_train, y_test = extractor.load_dataset(
        train_clips=TRAIN_CLIPS, test_clips=TEST_CLIPS
    )

    # Check if data was loaded
    if len(X_train) == 0 or len(X_test) == 0:
        print("\nError: No data was loaded. Please check your dataset structure.")
        return

    # Save dataset statistics
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    stats = {
        "num_features": X_train.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": len(np.unique(y_train)),
        "train_class_distribution": pd.Series(y_train).value_counts().to_dict(),
        "test_class_distribution": pd.Series(y_test).value_counts().to_dict(),
    }

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Number of features: {stats['num_features']}")
    print(f"Training samples: {stats['train_samples']}")
    print(f"Testing samples: {stats['test_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print("\nTraining class distribution:")
    gesture_names = [
        "open_palm",
        "closed_hand",
        "v_shape",
        "hand_right",
        "hand_left",
        "thumbs_up",
    ]
    for label, count in stats["train_class_distribution"].items():
        # labels are 0-based indices that map directly to gesture_names
        print(f"  {gesture_names[label]}: {count} samples")

    # Train and evaluate models
    trained_models, scaler = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, output_dir=OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nAll models and artifacts saved to '{OUTPUT_DIR}/' folder:")
    print("  - scaler.pkl (feature scaler)")
    print("  - svm_model.pkl (SVM classifier)")
    print("  - knn_model.pkl (KNN classifier)")
    print("  - randomforest_model.pkl (Random Forest classifier)")
    print("  - confusion_matrix_*.png (confusion matrices)")
    print("  - model_comparison.png (accuracy comparison)")


if __name__ == "__main__":
    main()
