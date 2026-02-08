"""
Online Classification Script - Part I.2.c)
Real-time hand gesture classification using trained classic ML models (SVM, KNN, Random Forest)
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class OnlineGestureClassifier:
    """Real-time hand gesture classifier using trained models"""

    def __init__(self, model_path, scaler_path, use_z_coord=False, relative_to_wrist=False):
        """
        Initialize the online classifier

        Args:
            model_path: Path to the trained model (.pkl file)
            scaler_path: Path to the scaler (.pkl file)
            use_z_coord: Whether to use (x,y,z) or just (x,y) coordinates
            relative_to_wrist: Whether to use coordinates relative to the wrist
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.use_z_coord = use_z_coord
        self.relative_to_wrist = relative_to_wrist

        self.gesture_names = [
            "open_palm",
            "closed_hand",
            "v_shape",
            "hand_right",
            "hand_left",
            "thumbs_up",
        ]

        # Color for each gesture (BGR format)
        self.gesture_colors = {
            "open_palm": (0, 255, 0),  # Green
            "closed_hand": (0, 0, 255),  # Red
            "v_shape": (255, 0, 255),  # Magenta
            "hand_right": (255, 255, 0),  # Cyan
            "hand_left": (0, 255, 255),  # Yellow
            "thumbs_up": (255, 165, 0),  # Orange
        }

        print(f"Model loaded: {Path(model_path).name}")
        print(f"Model type: {type(self.model).__name__}")

    def extract_landmarks(self, hand_landmarks):
        """
        Extract landmarks from MediaPipe hand detection

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            Flattened numpy array of landmarks
        """
        if self.relative_to_wrist:
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
        else:
            if self.use_z_coord:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            else:
                landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

        return np.array(landmarks).flatten()

    def predict(self, landmarks):
        """
        Predict gesture from landmarks

        Args:
            landmarks: Flattened array of landmarks

        Returns:
            Predicted gesture name and class label
        """
        # Reshape for single prediction
        landmarks = landmarks.reshape(1, -1)

        # Scale features
        landmarks_scaled = self.scaler.transform(landmarks)

        # Predict
        prediction = self.model.predict(landmarks_scaled)[0]
        gesture_name = self.gesture_names[prediction]  # Labels are 0-5

        return gesture_name, prediction

    def run(self, camera_id=0, display_fps=True):
        """
        Run real-time gesture classification

        Args:
            camera_id: Camera device ID (default 0)
            display_fps: Whether to display FPS counter
        """
        cap = cv2.VideoCapture(camera_id)

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # FPS calculation
        prev_time = time.time()
        fps = 0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:

            print("\n" + "=" * 60)
            print("ONLINE HAND GESTURE CLASSIFICATION")
            print("=" * 60)
            print("Press 'q' to quit")
            print("=" * 60 + "\n")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame
                results = hands.process(frame_rgb)

                # Initialize prediction text
                prediction_text = "No hand detected"
                prediction_color = (200, 200, 200)

                # Draw hand landmarks and predict gesture
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                        # Extract landmarks and predict
                        landmarks = self.extract_landmarks(hand_landmarks)
                        gesture_name, gesture_label = self.predict(landmarks)

                        prediction_text = (
                            f"Gesture: {gesture_name.upper().replace('_', ' ')}"
                        )
                        prediction_color = self.gesture_colors.get(
                            gesture_name, (255, 255, 255)
                        )

                # Calculate FPS
                if display_fps:
                    curr_time = time.time()
                    fps = (
                        1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
                    )
                    prev_time = curr_time

                # Display information on frame
                h, w, _ = frame.shape

                # Draw semi-transparent background for text
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                # Display prediction
                cv2.putText(
                    frame,
                    prediction_text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    prediction_color,
                    3,
                )

                # Display model info
                model_info = f"Model: {type(self.model).__name__}"
                cv2.putText(
                    frame,
                    model_info,
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Display FPS
                if display_fps:
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(
                        frame,
                        fps_text,
                        (w - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Display instructions
                cv2.putText(
                    frame,
                    "Press 'Q' to quit",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Show frame
                cv2.imshow("Hand Gesture Recognition - Classic Models", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("\nClassification stopped.")


def main():
    """Main function to run online classification"""

    print("=" * 60)
    print("ONLINE HAND GESTURE CLASSIFICATION")
    print("Part I.2.c) - Classic ML Models")
    print("=" * 60)

    # Available model directories
    model_directories = {
        "1": ("Classic Models: XY Coordinates", Path("../1_2_a/models_comparison/xy")),
        "2": ("Classic Models: XYZ Coordinates", Path("../1_2_a/models_comparison/xyz")),
        "3": ("Classic Models V2: XY Coordinates", Path("../1_2_a/models_comparison_rel_wrist/xy")),
        "4": ("Classic Models V2: XYZ Coordinates", Path("../1_2_a/models_comparison_rel_wrist/xyz")),
    }

    print("\nAvailable model directories:")
    for key, (dir_name, dir_path) in model_directories.items():
        print(f"  {key}. {dir_name} - {dir_path}")

    # Select directory
    while True:
        dir_choice = input("\nSelect model directory (1-4): ").strip()
        if dir_choice in model_directories:
            selected_dir_name, MODELS_DIR = model_directories[dir_choice]
            print(f"\nUsing models from: {MODELS_DIR}")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

    SCALER_PATH = MODELS_DIR / "scaler.pkl"

    # Available models
    models = {
        "1": ("SVM", MODELS_DIR / "svm_model.pkl"),
        "2": ("KNN", MODELS_DIR / "knn_model.pkl"),
        "3": ("RandomForest", MODELS_DIR / "randomforest_model.pkl"),
    }

    print("\nAvailable models:")
    for key, (name, path) in models.items():
        print(f"  {key}. {name}")

    # Select model
    while True:
        choice = input("\nSelect model (1-3): ").strip().lower()

        if choice in models:
            name, model_path = models[choice]

            if not model_path.exists():
                print(f"Error: Model file not found: {model_path}")
                print(f"Please run the training script first from the {selected_dir_name} directory")
                return

            if not SCALER_PATH.exists():
                print(f"Error: Scaler file not found: {SCALER_PATH}")
                print(f"Please run the training script first from the {selected_dir_name} directory")
                return

            # Determine if we should use z coordinate based on the selected directory
            use_z_coord = "xyz" in str(MODELS_DIR).lower()
            relative_to_wrist = "rel_wrist" in str(MODELS_DIR)
            print(f"Using {'XYZ' if use_z_coord else 'XY'} coordinates")
            print(f"Relative to wrist: {'Yes' if relative_to_wrist else 'No'}")

            # Create classifier
            classifier = OnlineGestureClassifier(
                model_path=model_path,
                scaler_path=SCALER_PATH,
                use_z_coord=use_z_coord,
                relative_to_wrist=relative_to_wrist
            )

            # Run online classification
            classifier.run(camera_id=0, display_fps=True)
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
