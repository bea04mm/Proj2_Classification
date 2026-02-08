"""
Script of Training CNN-1D - Version 2
Extracts hand landmarks relative to the wrist using MediaPipe and trains a CNN-1D.
Tests two configurations:
  1) Only (x, y) relative to the wrist
  2) (x, y, z) relative to the wrist
Compares the results and saves the best model.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import accuracy_score, confusion_matrix

mp_hands = mp.solutions.hands


class HandGestureSequenceExtractorRelWrist:
    """Extracts sequences of landmarks relative to the wrist"""

    def __init__(self, dataset_path, use_z_coord=False, seq_len=30):
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
        """Pad/Downsample sequences to have fixed length"""
        if len(seq) == 0:
            return None
        seq = np.array(seq, dtype=np.float32)
        if len(seq) == target_len:
            return seq
        if len(seq) < target_len:
            pad_count = target_len - len(seq)
            pad_frames = np.repeat(seq[-1][None, :], pad_count, axis=0)
            return np.concatenate([seq, pad_frames], axis=0)
        idx = np.linspace(0, len(seq) - 1, target_len).astype(int)
        return seq[idx]

    def extract_sequence_from_video(self, video_path):
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
                    wrist = hand_landmarks.landmark[0]
                    if self.use_z_coord:
                        landmarks = [
                            [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                            for lm in hand_landmarks.landmark
                        ]
                    else:
                        landmarks = [
                            [lm.x - wrist.x, lm.y - wrist.y]
                            for lm in hand_landmarks.landmark
                        ]
                    frames_features.append(np.array(landmarks).flatten())
        cap.release()
        return self._resample_sequence(frames_features, self.seq_len)

    def load_dataset(self, train_clips=[1, 2, 3, 4], test_clips=[5, 6]):
        X_train, y_train, X_test, y_test = [], [], [], []

        print("Loading dataset and extracting sequences relative to the wrist...")

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
                    print(f"    Clip {idx}: {video_file.name}")
                    sequence = self.extract_sequence_from_video(video_file)
                    if sequence is None:
                        print(f"    Warning: No landmarks in {video_file.name}")
                        continue

                    if idx in train_clips:
                        X_train.append(sequence)
                        y_train.append(gesture_label)
                    elif idx in test_clips:
                        X_test.append(sequence)
                        y_test.append(gesture_label)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return (
            np.array(X_train, dtype=np.float32),
            np.array(X_test, dtype=np.float32),
            np.array(y_train, dtype=np.int32),
            np.array(y_test, dtype=np.int32),
        )


def build_cnn1d_model(input_shape, num_classes, dropout1=0.3, dropout2=0.4):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout1),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_cnn(X_train, X_test, y_train, y_test, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    model = build_cnn1d_model(input_shape, num_classes)
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    gesture_names = [
        "open_palm",
        "closed_hand",
        "v_shape",
        "hand_right",
        "hand_left",
        "thumbs_up",
    ]
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
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()

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

    model_path = output_path / "cnn1d_model.keras"
    model.save(model_path)

    return {"accuracy": accuracy, "model_path": model_path, "history": history}


def remove_z_coordinate(X):
    """Removes the z coordinate from sequence feature vectors"""
    # X shape: (samples, seq_len, features)
    # features = 21 landmarks * 3 coords = 63
    n_samples, seq_len, features = X.shape
    n_landmarks = 21
    X_reshaped = X.reshape(n_samples, seq_len, n_landmarks, 3)
    X_xy = X_reshaped[:, :, :, :2]  # Keep only x,y
    return X_xy.reshape(n_samples, seq_len, -1)


def run_experiment(X_train, X_test, y_train, y_test, output_base_dir):
    results = {}
    for use_z in [False, True]:
        mode_name = "xy" if not use_z else "xyz"
        output_dir = Path(output_base_dir) / mode_name
        
        # If we don't want z, remove it from the data
        if not use_z:
            X_train_exp = remove_z_coordinate(X_train)
            X_test_exp = remove_z_coordinate(X_test)
        else:
            X_train_exp = X_train
            X_test_exp = X_test
        
        if len(X_train_exp) == 0 or len(X_test_exp) == 0:
            continue
        print(f"\nExecuting experiment with {mode_name.upper()} relative to the wrist...")
        results[mode_name] = train_and_evaluate_cnn(
            X_train_exp, X_test_exp, y_train, y_test, output_dir
        )
        print(f"Accuracy ({mode_name}): {results[mode_name]['accuracy']:.4f}")

    best_mode = max(results, key=lambda k: results[k]["accuracy"])
    print(
        f"\nBest model: {best_mode} with accuracy {results[best_mode]['accuracy']:.4f}"
    )
    return results, best_mode


def main():
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "cnn_models_rel_wrist"

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Extract dataset once with (x, y, z) relative to wrist
    print("\nExtracting dataset (only once with x,y,z relative to wrist)...")
    extractor = HandGestureSequenceExtractorRelWrist(
        DATASET_PATH, use_z_coord=True, seq_len=30
    )
    X_train, X_test, y_train, y_test = extractor.load_dataset()

    results, best_mode = run_experiment(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    print(f"\nBest model saved for mode: {best_mode}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
