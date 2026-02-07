"""
CNN-1D Training Script v2 - Coordenadas relativas ao pulso
Todos os landmarks são relativos ao pulso (landmark 0)
Inclui GridSearch para otimização de CNN-1D
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

mp_hands = mp.solutions.hands


class HandGestureSequenceExtractorRelWrist:
    """Extrai sequências de landmarks relativas ao pulso"""

    def __init__(
        self,
        dataset_path,
        use_z_coord=False,
        seq_len=30,
        include_deltas=True,
        normalize_sequence=True,
    ):
        self.dataset_path = Path(dataset_path)
        self.use_z_coord = use_z_coord
        self.seq_len = seq_len
        self.include_deltas = include_deltas
        self.normalize_sequence = normalize_sequence
        self.gesture_mapping = {
            "open_palm": 0,
            "closed_hand": 1,
            "v_shape": 2,
            "hand_right": 3,
            "hand_left": 4,
            "thumbs_up": 5,
        }

    def _resample_sequence(self, seq, target_len):
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

    def _make_time_series_features(self, seq):
        seq = np.array(seq, dtype=np.float32)
        if self.include_deltas:
            deltas = np.zeros_like(seq)
            deltas[1:] = seq[1:] - seq[:-1]
            seq = np.concatenate([seq, deltas], axis=1)
        if self.normalize_sequence:
            mean = seq.mean(axis=0, keepdims=True)
            std = seq.std(axis=0, keepdims=True) + 1e-6
            seq = (seq - mean) / std
        return seq

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
        resampled = self._resample_sequence(frames_features, self.seq_len)
        if resampled is None:
            return None
        return self._make_time_series_features(resampled)

    def load_dataset(self, train_clips=[1, 2, 3, 4], test_clips=[5, 6]):
        X_train, y_train, X_test, y_test = [], [], [], []
        for subject_folder in self.dataset_path.iterdir():
            if not subject_folder.is_dir() or subject_folder.name == "__pycache__":
                continue
            for gesture_name, gesture_label in self.gesture_mapping.items():
                gesture_folder = subject_folder / gesture_name
                if not gesture_folder.exists():
                    continue
                video_files = sorted(
                    list(gesture_folder.glob("*.avi"))
                    + list(gesture_folder.glob("*.mp4"))
                )
                for idx, video_file in enumerate(video_files, start=1):
                    sequence = self.extract_sequence_from_video(video_file)
                    if sequence is None:
                        continue
                    if idx in train_clips:
                        X_train.append(sequence)
                        y_train.append(gesture_label)
                    elif idx in test_clips:
                        X_test.append(sequence)
                        y_test.append(gesture_label)
        return (
            np.array(X_train, dtype=np.float32),
            np.array(X_test, dtype=np.float32),
            np.array(y_train, dtype=np.int32),
            np.array(y_test, dtype=np.int32),
        )


# Função para construir CNN-1D
def build_cnn1d_model(
    input_shape=(30, 63),
    num_classes=6,
    filters1=64,
    filters2=128,
    dropout1=0.3,
    dropout2=0.4,
):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(filters1, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(filters2, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout1),
            layers.Conv1D(filters2, 3, activation="relu", padding="same"),
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


def train_with_gridsearch(
    X_train, y_train, X_test, y_test, input_shape, num_classes, output_dir, mode_name
):
    """Treina CNN-1D com GridSearch"""

    def create_model(filters1=64, filters2=128, dropout1=0.3, dropout2=0.4):
        return build_cnn1d_model(
            input_shape, num_classes, filters1, filters2, dropout1, dropout2
        )

    model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=16, verbose=0)
    param_grid = {
        "filters1": [32, 64],
        "filters2": [64, 128],
        "dropout1": [0.2, 0.3],
        "dropout2": [0.3, 0.4],
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy ({mode_name}): {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    gesture_names = [
        "open_palm",
        "closed_hand",
        "v_shape",
        "hand_right",
        "hand_left",
        "thumbs_up",
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=gesture_names,
        yticklabels=gesture_names,
    )
    plt.title(f"Confusion Matrix - {mode_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(Path(output_dir) / f"confusion_matrix_{mode_name}.png")
    plt.close()

    best_model.model.save(Path(output_dir) / f"cnn1d_grid_{mode_name}.keras")
    return accuracy


def main():
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "cnn_models_v2_relative"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    results = {}
    for use_z in [False, True]:
        mode_name = "xy" if not use_z else "xyz"
        extractor = HandGestureSequenceExtractorRelWrist(
            DATASET_PATH, use_z_coord=use_z
        )
        X_train, X_test, y_train, y_test = extractor.load_dataset()
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        print(f"\nRunning experiment: {mode_name}")
        acc = train_with_gridsearch(
            X_train,
            y_train,
            X_test,
            y_test,
            X_train.shape[1:],
            len(np.unique(y_train)),
            Path(OUTPUT_DIR) / mode_name,
            mode_name,
        )
        results[mode_name] = acc

    best_mode = max(results, key=lambda k: results[k])
    print(f"\nBest model: {best_mode} Accuracy: {results[best_mode]:.4f}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
