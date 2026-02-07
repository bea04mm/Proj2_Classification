"""
Training Script for Classic ML Models + CNN-1D - Parts I.2.a) and I.2.b)
Extracts hand landmarks using MediaPipe and trains SVM, KNN, Random Forest, and CNN-1D classifiers
"""

import os
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
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

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

    def _extract_landmarks(self, hand_landmarks):
        """Extract and flatten landmarks from a hand detection"""
        if self.use_z_coord:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        else:
            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
        return np.array(landmarks).flatten()

    def extract_landmarks_from_video(self, video_path, as_sequence=False, seq_len=30):
        """
        Extract hand landmarks from a video file

        Args:
            video_path: Path to the video file
            as_sequence: If True, returns fixed-length sequence; if False, returns all frames
            seq_len: Target sequence length (only used if as_sequence=True)

        Returns:
            List of landmark features or fixed-length sequence array
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

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    features.append(self._extract_landmarks(hand_landmarks))

        cap.release()

        if as_sequence:
            return self._resample_sequence(features, seq_len)
        return features

    def _resample_sequence(self, seq, target_len):
        """Resample/pad a sequence to a fixed length"""
        if len(seq) == 0:
            return None
        seq = np.array(seq, dtype=np.float32)
        if len(seq) == target_len:
            return seq
        if len(seq) < target_len:
            # Pad by repeating the last frame
            pad_count = target_len - len(seq)
            pad_frames = np.repeat(seq[-1][None, :], pad_count, axis=0)
            return np.concatenate([seq, pad_frames], axis=0)
        # Downsample using evenly spaced indices
        idx = np.linspace(0, len(seq) - 1, target_len).astype(int)
        return seq[idx]

    def extract_sequence_from_video(self, video_path, seq_len=30):
        """
        Extract a fixed-length landmark sequence from a video file

        Args:
            video_path: Path to the video file
            seq_len: Number of frames per clip sequence

        Returns:
            Numpy array with shape (seq_len, num_features)
        """
        return self.extract_landmarks_from_video(
            video_path, as_sequence=True, seq_len=seq_len
        )

    def _process_dataset(self, train_clips, test_clips, for_cnn=False, seq_len=30):
        """Generic method to process dataset for both classic and CNN models"""
        X_train, y_train = [], []
        X_test, y_test = [], []

        model_type = "CNN" if for_cnn else "classic models"
        print(f"Loading dataset and extracting features for {model_type}...")

        for subject_folder in self.dataset_path.iterdir():
            if not subject_folder.is_dir() or subject_folder.name == "__pycache__":
                continue

            print(f"\nProcessing subject: {subject_folder.name}")

            for gesture_name, gesture_label in self.gesture_mapping.items():
                gesture_folder = subject_folder / gesture_name
                if not gesture_folder.exists():
                    continue

                print(f"  - Gesture: {gesture_name}")

                video_files = sorted(
                    list(gesture_folder.glob("*.avi"))
                    + list(gesture_folder.glob("*.mp4"))
                )

                for idx, video_file in enumerate(video_files, start=1):
                    print(f"    Processing clip {idx}: {video_file.name}")

                    # Extract features
                    if for_cnn:
                        features = self.extract_landmarks_from_video(
                            video_file, as_sequence=True, seq_len=seq_len
                        )
                        if features is None:
                            print(
                                f"    Warning: No landmarks detected in {video_file.name}"
                            )
                            continue

                        if idx in train_clips:
                            X_train.append(features)
                            y_train.append(gesture_label)
                        elif idx in test_clips:
                            X_test.append(features)
                            y_test.append(gesture_label)
                    else:
                        features = self.extract_landmarks_from_video(video_file)
                        if not features:
                            print(
                                f"    Warning: No landmarks detected in {video_file.name}"
                            )
                            continue

                        if idx in train_clips:
                            X_train.extend(features)
                            y_train.extend([gesture_label] * len(features))
                        elif idx in test_clips:
                            X_test.extend(features)
                            y_test.extend([gesture_label] * len(features))

        print(f"\nDataset loaded:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")

        if for_cnn:
            return (
                np.array(X_train, dtype=np.float32),
                np.array(X_test, dtype=np.float32),
                np.array(y_train, dtype=np.int32),
                np.array(y_test, dtype=np.int32),
            )
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def load_both_datasets(
        self, seq_len=30, train_clips=[1, 2, 3, 4], test_clips=[5, 6]
    ):
        """
        Load all video clips ONCE and extract features for BOTH classic ML and CNN models

        Args:
            seq_len: Number of frames per clip sequence (for CNN)
            train_clips: Indices of clips to use for training (1-6)
            test_clips: Indices of clips to use for testing (1-6)

        Returns:
            Tuple: (classic_data, cnn_data) where each contains (X_train, X_test, y_train, y_test)
        """
        # Classic model data (individual frames)
        X_train_classic, y_train_classic = [], []
        X_test_classic, y_test_classic = [], []

        # CNN model data (sequences)
        X_train_cnn, y_train_cnn = [], []
        X_test_cnn, y_test_cnn = [], []

        print("Loading dataset and extracting features for ALL models (single pass)...")

        for subject_folder in self.dataset_path.iterdir():
            if not subject_folder.is_dir() or subject_folder.name == "__pycache__":
                continue

            print(f"\nProcessing subject: {subject_folder.name}")

            for gesture_name, gesture_label in self.gesture_mapping.items():
                gesture_folder = subject_folder / gesture_name
                if not gesture_folder.exists():
                    continue

                print(f"  - Gesture: {gesture_name}")

                video_files = sorted(
                    list(gesture_folder.glob("*.avi"))
                    + list(gesture_folder.glob("*.mp4"))
                )

                for idx, video_file in enumerate(video_files, start=1):
                    print(f"    Processing clip {idx}: {video_file.name}")

                    # Extract features once (all frames)
                    features_all = self.extract_landmarks_from_video(video_file)

                    if not features_all:
                        print(
                            f"    Warning: No landmarks detected in {video_file.name}"
                        )
                        continue

                    # Prepare sequence for CNN from the same features
                    sequence = self._resample_sequence(features_all, seq_len)

                    if sequence is None:
                        print(
                            f"    Warning: Could not create sequence from {video_file.name}"
                        )
                        continue

                    # Assign to train or test sets
                    if idx in train_clips:
                        # Classic models: all frames
                        X_train_classic.extend(features_all)
                        y_train_classic.extend([gesture_label] * len(features_all))
                        # CNN: sequence
                        X_train_cnn.append(sequence)
                        y_train_cnn.append(gesture_label)
                    elif idx in test_clips:
                        # Classic models: all frames
                        X_test_classic.extend(features_all)
                        y_test_classic.extend([gesture_label] * len(features_all))
                        # CNN: sequence
                        X_test_cnn.append(sequence)
                        y_test_cnn.append(gesture_label)

        print(f"\nDataset loaded (single pass):")
        print(
            f"  Classic models - Training: {len(X_train_classic)}, Testing: {len(X_test_classic)}"
        )
        print(f"  CNN model - Training: {len(X_train_cnn)}, Testing: {len(X_test_cnn)}")

        # Prepare classic data
        classic_data = (
            np.array(X_train_classic),
            np.array(X_test_classic),
            np.array(y_train_classic),
            np.array(y_test_classic),
        )

        # Prepare CNN data
        cnn_data = (
            np.array(X_train_cnn, dtype=np.float32),
            np.array(X_test_cnn, dtype=np.float32),
            np.array(y_train_cnn, dtype=np.int32),
            np.array(y_test_cnn, dtype=np.int32),
        )

        return classic_data, cnn_data

    def load_dataset_classic(self, train_clips=[1, 2, 3, 4], test_clips=[5, 6]):
        """
        Load all video clips and extract features for classic ML models

        Args:
            train_clips: Indices of clips to use for training (1-6)
            test_clips: Indices of clips to use for testing (1-6)

        Returns:
            X_train, X_test, y_train, y_test
        """
        return self._process_dataset(train_clips, test_clips, for_cnn=False)

    def load_dataset_cnn(self, seq_len=30, train_clips=[1, 2, 3, 4], test_clips=[5, 6]):
        """
        Load all video clips and extract fixed-length sequences for CNN

        Args:
            seq_len: Number of frames per clip sequence
            train_clips: Indices of clips to use for training (1-6)
            test_clips: Indices of clips to use for testing (1-6)

        Returns:
            X_train, X_test, y_train, y_test
        """
        return self._process_dataset(
            train_clips, test_clips, for_cnn=True, seq_len=seq_len
        )


# Gesture names constant
GESTURE_NAMES = [
    "open_palm",
    "closed_hand",
    "v_shape",
    "hand_right",
    "hand_left",
    "thumbs_up",
]


def plot_confusion_matrix(y_test, y_pred, model_name, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GESTURE_NAMES,
        yticklabels=GESTURE_NAMES,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    filename = (
        output_path
        / f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('-', '')}.png"
    )
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.close()
    return cm


def evaluate_model(y_test, y_pred, model_name, output_path):
    """Evaluate model and generate reports"""
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=GESTURE_NAMES))

    cm = plot_confusion_matrix(y_test, y_pred, model_name, output_path)

    return accuracy, cm


def build_cnn1d_model(input_shape, num_classes):
    """Build a simple CNN-1D model"""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_classic_models(
    X_train, X_test, y_train, y_test, output_dir="models"
):
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
    models_dict = {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42
        ),
    }

    trained_models = {}
    results = {}

    # Train and evaluate each model
    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate
        accuracy, cm = evaluate_model(y_test, y_pred, model_name, output_path)

        # Save model
        model_filename = output_path / f"{model_name.lower()}_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

        trained_models[model_name] = model
        results[model_name] = {"accuracy": accuracy, "confusion_matrix": cm}

    # Compare models
    print(f"\n{'='*60}")
    print("Classic Models Comparison Summary")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name:15s}: Accuracy = {result['accuracy']:.4f}")

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]

    plt.bar(model_names, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Accuracy")
    plt.title("Classic Models Comparison - Test Accuracy")
    plt.ylim([0, 1])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison_classic.png")
    print(f"\nComparison plot saved to {output_path / 'model_comparison_classic.png'}")
    plt.close()

    return trained_models, scaler


def plot_training_history(history, output_path):
    """Plot training history for loss and accuracy"""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path / "training_history_cnn.png")
    plt.close()


def train_and_evaluate_cnn(X_train, X_test, y_train, y_test, output_dir="models"):
    """
    Train and evaluate CNN-1D model

    Args:
        X_train, X_test, y_train, y_test: Training and testing data
        output_dir: Directory to save trained models
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("Training CNN-1D Model...")
    print(f"{'='*60}")

    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    model = build_cnn1d_model(input_shape, num_classes)
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy, cm = evaluate_model(y_test, y_pred, "CNN-1D", output_path)

    plot_training_history(history, output_path)

    model_path = output_path / "cnn1d_model.keras"
    model.save(model_path)
    print(f"\nCNN-1D Model saved to {model_path}")
    print(f"Training history saved to {output_path / 'training_history_cnn.png'}")

    return model, accuracy


def main():
    """Main function to run the complete training pipeline"""

    # Configuration
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "models"
    USE_Z_COORD = False
    SEQ_LEN = 30

    # Training/Testing split (clip indices)
    TRAIN_CLIPS = [1, 2, 3, 4]
    TEST_CLIPS = [5, 6]

    print("=" * 60)
    print("Hand Gesture Classification - Complete Training Pipeline")
    print("Parts I.2.a) and I.2.b)")
    print("=" * 60)
    print(f"\nDataset path: {DATASET_PATH}")
    print(f"Using coordinates: {'(x, y, z)' if USE_Z_COORD else '(x, y)'}")
    print(f"Sequence length (CNN): {SEQ_LEN}")
    print(f"Training clips: {TRAIN_CLIPS}")
    print(f"Testing clips: {TEST_CLIPS}")

    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"\nError: Dataset path does not exist: {DATASET_PATH}")
        print("Please ensure the Videos_Datasets folder is in the correct location.")
        return

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # Initialize extractor
    extractor = HandGestureDataExtractor(DATASET_PATH, use_z_coord=USE_Z_COORD)

    # ==================== LOAD DATA ONCE ====================
    print("\n" + "=" * 60)
    print("LOADING DATASET (Single Pass for All Models)")
    print("=" * 60)

    classic_data, cnn_data = extractor.load_both_datasets(
        seq_len=SEQ_LEN, train_clips=TRAIN_CLIPS, test_clips=TEST_CLIPS
    )

    X_train_classic, X_test_classic, y_train_classic, y_test_classic = classic_data
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = cnn_data

    if len(X_train_classic) == 0 or len(X_test_classic) == 0:
        print(
            "\nError: No data was loaded for classic models. Please check your dataset structure."
        )
        return

    # ==================== CLASSIC ML MODELS ====================
    print("\n" + "=" * 60)
    print("PHASE 1: CLASSIC ML MODELS (SVM, KNN, Random Forest)")
    print("=" * 60)

    # Save dataset statistics for classic models
    stats = {
        "num_features": X_train_classic.shape[1],
        "train_samples": len(X_train_classic),
        "test_samples": len(X_test_classic),
        "num_classes": len(np.unique(y_train_classic)),
        "train_class_distribution": pd.Series(y_train_classic).value_counts().to_dict(),
        "test_class_distribution": pd.Series(y_test_classic).value_counts().to_dict(),
    }

    print("\n" + "=" * 60)
    print("Dataset Statistics (Classic Models)")
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
        print(f"  {gesture_names[label]}: {count} samples")

    # Train classic models
    trained_models_classic, scaler = train_and_evaluate_classic_models(
        X_train_classic,
        X_test_classic,
        y_train_classic,
        y_test_classic,
        output_dir=OUTPUT_DIR,
    )

    # ==================== CNN MODEL ====================
    print("\n" + "=" * 60)
    print("PHASE 2: DEEP LEARNING MODEL (CNN-1D)")
    print("=" * 60)

    if len(X_train_cnn) == 0 or len(X_test_cnn) == 0:
        print(
            "\nError: No data was loaded for CNN. Please check your dataset structure."
        )
        return

    print("\n" + "=" * 60)
    print("Dataset Statistics (CNN Model)")
    print("=" * 60)
    print(f"Sequence length: {X_train_cnn.shape[1]}")
    print(f"Features per frame: {X_train_cnn.shape[2]}")
    print(f"Training samples: {len(X_train_cnn)}")
    print(f"Testing samples: {len(X_test_cnn)}")
    print(f"Number of classes: {len(np.unique(y_train_cnn))}")

    # Train CNN model
    cnn_model, cnn_accuracy = train_and_evaluate_cnn(
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, output_dir=OUTPUT_DIR
    )

    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"\nAll models and artifacts saved to '{OUTPUT_DIR}/' folder:")
    print("\nClassic ML Models:")
    print("  - scaler.pkl (feature scaler)")
    print("  - svm_model.pkl (SVM classifier)")
    print("  - knn_model.pkl (KNN classifier)")
    print("  - randomforest_model.pkl (Random Forest classifier)")
    print("  - confusion_matrix_*.png (confusion matrices)")
    print("  - model_comparison_classic.png (accuracy comparison)")
    print("\nDeep Learning Model:")
    print("  - cnn1d_model.keras (CNN-1D classifier)")
    print("  - confusion_matrix_cnn1d.png (confusion matrix)")
    print("  - training_history_cnn.png (training curves)")


if __name__ == "__main__":
    # Reduce TensorFlow logging noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
