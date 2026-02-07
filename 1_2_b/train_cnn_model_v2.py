"""
Script de Treino CNN-1D - Versão 2
Extrai landmarks da mão relativos ao pulso com MediaPipe e treina um CNN-1D.
Testa duas configurações:
  1) Apenas (x, y) relativo ao pulso
  2) (x, y, z) relativo ao pulso
Compara os resultados e guarda o melhor modelo.
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

mp_hands = mp.solutions.hands


class HandGestureSequenceExtractorRelWrist:
    """Extrai sequências de landmarks relativos ao pulso"""

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
        """Pad/Downsample sequências para ter comprimento fixo"""
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
    output_path.mkdir(exist_ok=True)

    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    model = build_cnn1d_model(input_shape, num_classes)
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


def run_experiment(dataset_path, output_base_dir):
    results = {}
    for use_z in [False, True]:
        mode_name = "xy" if not use_z else "xyz"
        output_dir = Path(output_base_dir) / mode_name
        extractor = HandGestureSequenceExtractorRelWrist(
            dataset_path, use_z_coord=use_z, seq_len=30
        )
        X_train, X_test, y_train, y_test = extractor.load_dataset()
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        print(f"\nExecutando experiência com {mode_name.upper()} relativo ao pulso...")
        results[mode_name] = train_and_evaluate_cnn(
            X_train, X_test, y_train, y_test, output_dir
        )
        print(f"Accuracy ({mode_name}): {results[mode_name]['accuracy']:.4f}")

    best_mode = max(results, key=lambda k: results[k]["accuracy"])
    print(
        f"\nMelhor modelo: {best_mode} com accuracy {results[best_mode]['accuracy']:.4f}"
    )
    return results, best_mode


def main():
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "cnn_models_rel_wrist"

    if not DATASET_PATH.exists():
        print(f"Erro: Dataset não encontrado em {DATASET_PATH}")
        return

    results, best_mode = run_experiment(DATASET_PATH, OUTPUT_DIR)
    print(f"\nMelhor modelo guardado para modo: {best_mode}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
