"""
Script of Training for Classic ML Models - Part I.2.a) - Version 2
Extracts hand landmarks relative to the wrist with MediaPipe and trains SVM, KNN, and Random Forest.
Tests two configurations:
  1) Only (x, y) relative to the wrist
  2) (x, y, z) relative to the wrist
Compares the results and saves the best model.
Includes GridSearch for hyperparameter optimization.
"""

import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

mp_hands = mp.solutions.hands


class HandGestureDataExtractorRelWrist:
    """Extracts landmarks relative to the wrist"""

    def __init__(self, dataset_path, use_z_coord=False):
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
                    for hand_landmarks in results.multi_hand_landmarks:
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
                        features.append(np.array(landmarks).flatten())

        cap.release()
        return features

    def load_dataset(self, train_clips=[1, 2, 3, 4], test_clips=[5, 6]):
        X_train, y_train, X_test, y_test = [], [], [], []

        print("Loading dataset and extracting features relative to the wrist...")

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
                    features = self.extract_landmarks_from_video(video_file)
                    if not features:
                        print(f"    Warning: No landmarks in {video_file.name}")
                        continue

                    if idx in train_clips:
                        X_train.extend(features)
                        y_train.extend([gesture_label] * len(features))
                    elif idx in test_clips:
                        X_test.extend(features)
                        y_test.extend([gesture_label] * len(features))

        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir):
    """Trains models with GridSearch and saves results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, output_path / "scaler.pkl")

    gesture_names = [
        "open_palm",
        "closed_hand",
        "v_shape",
        "hand_right",
        "hand_left",
        "thumbs_up",
    ]

    results = {}

    param_grids = {
        "SVM": {"C": [1, 10, 50], "gamma": ["scale", "auto"], "kernel": ["rbf"]},
        "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [10, 20, None]},
    }

    models = {
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(random_state=42),
    }

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} with GridSearch")
        print(f"{'='*60}")

        grid = GridSearchCV(model, param_grids[model_name], cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_

        print(f"Best parameters: {grid.best_params_}")

        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

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
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(output_path / f"confusion_matrix_{model_name}.png")
        plt.close()

        model_filename = output_path / f"{model_name.lower()}_model.pkl"
        joblib.dump(best_model, model_filename)

        results[model_name] = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "model_path": model_filename,
            "best_params": grid.best_params_,
        }

    return results


def remove_z_coordinate(X):
    """Removes the z coordinate from feature vectors"""
    # Assuming features are flattened: [x1,y1,z1, x2,y2,z2, ...]
    # We need to keep only x,y for each landmark (21 landmarks total)
    n_samples = X.shape[0]
    n_landmarks = 21
    X_reshaped = X.reshape(n_samples, n_landmarks, 3)  # (samples, 21, 3)
    X_xy = X_reshaped[:, :, :2]  # Keep only x,y
    return X_xy.reshape(n_samples, -1)  # Flatten back


def run_experiment(use_z_coord, X_train, X_test, y_train, y_test, output_base_dir):
    mode_name = "xyz" if use_z_coord else "xy"
    output_dir = Path(output_base_dir) / mode_name

    print("\n" + "=" * 60)
    print(f"Running experiment with coordinates: {mode_name.upper()}")
    print("=" * 60)

    # If we don't want z, remove it from the data
    if not use_z_coord:
        X_train_exp = remove_z_coordinate(X_train)
        X_test_exp = remove_z_coordinate(X_test)
    else:
        X_train_exp = X_train
        X_test_exp = X_test

    results = train_and_evaluate_models(
        X_train_exp, X_test_exp, y_train, y_test, output_dir
    )

    return results


def main():
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "models_comparison_rel_wrist"

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Extract dataset once with (x, y, z) relative to wrist
    print("\n" + "=" * 60)
    print("Extracting dataset (only once with x,y,z relative to wrist)")
    print("=" * 60)
    extractor = HandGestureDataExtractorRelWrist(DATASET_PATH, use_z_coord=True)
    X_train, X_test, y_train, y_test = extractor.load_dataset()

    # Experiment 1: only (x, y) - remove z from extracted data
    results_xy = run_experiment(False, X_train, X_test, y_train, y_test, OUTPUT_DIR)

    # Experiment 2: (x, y, z) - use full extracted data
    results_xyz = run_experiment(True, X_train, X_test, y_train, y_test, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)

    best_overall = {"mode": None, "model": None, "accuracy": 0, "model_path": None}

    for model_name in results_xy.keys():
        acc_xy = results_xy[model_name]["accuracy"]
        acc_xyz = results_xyz[model_name]["accuracy"]

        print(f"\nModel: {model_name}")
        print(f"  Accuracy (x,y):   {acc_xy:.4f}")
        print(f"  Accuracy (x,y,z): {acc_xyz:.4f}")

        if acc_xy > acc_xyz:
            print("  -> Better without Z")
            if acc_xy > best_overall["accuracy"]:
                best_overall.update(
                    {
                        "mode": "xy",
                        "model": model_name,
                        "accuracy": acc_xy,
                        "model_path": results_xy[model_name]["model_path"],
                    }
                )
        else:
            print("  -> Better with Z")
            if acc_xyz > best_overall["accuracy"]:
                best_overall.update(
                    {
                        "mode": "xyz",
                        "model": model_name,
                        "accuracy": acc_xyz,
                        "model_path": results_xyz[model_name]["model_path"],
                    }
                )

    print("\n" + "=" * 60)
    print("Best Overall Model")
    print("=" * 60)
    print(f"Mode: {best_overall['mode']}")
    print(f"Model: {best_overall['model']}")
    print(f"Accuracy: {best_overall['accuracy']:.4f}")

    best_model_dest = Path(OUTPUT_DIR) / "best_model.pkl"
    shutil.copy(best_overall["model_path"], best_model_dest)
    print(f"Best model saved at: {best_model_dest}")


if __name__ == "__main__":
    main()
