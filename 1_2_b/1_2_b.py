"""
Training Script for CNN-1D Model - Part I.2.b)
Extracts hand landmarks using MediaPipe and trains a 1D CNN classifier
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands


class HandGestureSequenceExtractor:
    """Extract hand landmark sequences from video clips"""

    def __init__(self, dataset_path, use_z_coord=False, seq_len=30):
        """
        Initialize the data extractor

        Args:
                dataset_path: Path to the Videos_Datasets folder
                use_z_coord: Whether to use (x,y,z) or just (x,y) coordinates
                seq_len: Number of frames per clip sequence
        """
        self.dataset_path = Path(dataset_path)
        self.use_z_coord = use_z_coord
        self.seq_len = seq_len
        self.gesture_mapping = {
            "open_palm": 0,
            "closed_hand": 1,
            "v_shape": 2,
            "hand_right": 3,
            "hand_left": 4,
            "thumbs_up": 5,
        }

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

    def extract_sequence_from_video(self, video_path):
        """
        Extract a fixed-length landmark sequence from a video file

        Args:
                video_path: Path to the video file

        Returns:
                Numpy array with shape (seq_len, num_features)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames_features = []

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
                    if self.use_z_coord:
                        landmarks = [
                            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                        ]
                    else:
                        landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                    frames_features.append(np.array(landmarks).flatten())

        cap.release()
        return self._resample_sequence(frames_features, self.seq_len)

    def load_dataset(self, train_clips=[1, 2, 3, 4], test_clips=[5, 6]):
        """
        Load all video clips and extract fixed-length sequences

        Args:
                train_clips: Indices of clips to use for training (1-6)
                test_clips: Indices of clips to use for testing (1-6)

        Returns:
                X_train, X_test, y_train, y_test
        """
        X_train, y_train = [], []
        X_test, y_test = [], []

        print("Loading dataset and extracting sequences...")

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
                    sequence = self.extract_sequence_from_video(video_file)

                    if sequence is None:
                        print(
                            f"    Warning: No landmarks detected in {video_file.name}"
                        )
                        continue

                    if idx in train_clips:
                        X_train.append(sequence)
                        y_train.append(gesture_label)
                    elif idx in test_clips:
                        X_test.append(sequence)
                        y_test.append(gesture_label)

        print("\nDataset loaded:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")

        return (
            np.array(X_train, dtype=np.float32),
            np.array(X_test, dtype=np.float32),
            np.array(y_train, dtype=np.int32),
            np.array(y_test, dtype=np.int32),
        )


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
    plt.savefig(output_path / "training_history.png")
    plt.close()


def main():
    """Main function to run the CNN-1D training pipeline"""

    # Configuration
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "models"
    USE_Z_COORD = False
    SEQ_LEN = 30

    # Training/Testing split (clip indices)
    TRAIN_CLIPS = [1, 2, 3, 4]
    TEST_CLIPS = [5, 6]

    print("=" * 60)
    print("Hand Gesture Classification - CNN-1D")
    print("Part I.2.b) - Offline Training")
    print("=" * 60)
    print(f"\nDataset path: {DATASET_PATH}")
    print(f"Using coordinates: {'(x, y, z)' if USE_Z_COORD else '(x, y)'}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Training clips: {TRAIN_CLIPS}")
    print(f"Testing clips: {TEST_CLIPS}")

    if not DATASET_PATH.exists():
        print(f"\nError: Dataset path does not exist: {DATASET_PATH}")
        print("Please ensure the Videos_Datasets folder is in the correct location.")
        return

    extractor = HandGestureSequenceExtractor(
        DATASET_PATH, use_z_coord=USE_Z_COORD, seq_len=SEQ_LEN
    )
    X_train, X_test, y_train, y_test = extractor.load_dataset(
        train_clips=TRAIN_CLIPS, test_clips=TEST_CLIPS
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("\nError: No data was loaded. Please check your dataset structure.")
        return

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    gesture_names = [
        "open_palm",
        "closed_hand",
        "v_shape",
        "hand_right",
        "hand_left",
        "thumbs_up",
    ]
    print(classification_report(y_test, y_pred, target_names=gesture_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=gesture_names,
        yticklabels=gesture_names,
    )
    plt.title("Confusion Matrix - CNN-1D")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix_cnn1d.png")
    plt.close()

    plot_training_history(history, output_path)

    model_path = output_path / "cnn1d_model.keras"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Confusion matrix saved to {output_path / 'confusion_matrix_cnn1d.png'}")
    print(f"Training history saved to {output_path / 'training_history.png'}")


if __name__ == "__main__":
    # Reduce TensorFlow logging noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
