"""
Online Classification Script for CNN-1D - Part I.2.c)
Real-time hand gesture classification using trained CNN-1D model
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time
import tensorflow as tf
from collections import deque
import shutil
import tempfile

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class OnlineGestureClassifierCNN:
    """Real-time hand gesture classifier using CNN-1D model"""

    def __init__(self, model_path, use_z_coord=False, seq_len=30):
        """
        Initialize the online classifier

        Args:
            model_path: Path to the trained model (.keras file)
            use_z_coord: Whether to use (x,y,z) or just (x,y) coordinates
            seq_len: Number of frames per sequence
        """
        # Workaround for TensorFlow Unicode path issue on Windows
        # Copy model to a temporary location with ASCII-only path
        temp_dir = tempfile.gettempdir()
        temp_model_path = Path(temp_dir) / "temp_cnn1d_model.keras"

        print(f"Loading model (working around path encoding issue)...")
        shutil.copy2(str(model_path), str(temp_model_path))

        self.model = tf.keras.models.load_model(str(temp_model_path))
        self.temp_model_path = temp_model_path  # Store for cleanup
        self.use_z_coord = use_z_coord
        self.seq_len = seq_len

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

        # Buffer for storing recent frames
        self.frame_buffer = deque(maxlen=seq_len)

        print(f"Model loaded: {Path(model_path).name}")
        print(f"Model type: CNN-1D")
        print(f"Sequence length: {seq_len}")

    def extract_landmarks(self, hand_landmarks):
        """
        Extract landmarks from MediaPipe hand detection

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            Flattened numpy array of landmarks
        """
        if self.use_z_coord:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        else:
            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

        return np.array(landmarks).flatten()

    def predict(self, sequence):
        """
        Predict gesture from sequence

        Args:
            sequence: Array of shape (seq_len, num_features)

        Returns:
            Predicted gesture name, class label, and confidence
        """
        # Reshape for single prediction
        sequence = sequence.reshape(1, self.seq_len, -1)

        # Predict
        predictions = self.model.predict(sequence, verbose=0)[0]
        prediction = np.argmax(predictions)
        confidence = predictions[prediction]
        gesture_name = self.gesture_names[prediction]

        return gesture_name, prediction, confidence

    def run(self, camera_id=0, display_fps=True, confidence_threshold=0.5):
        """
        Run real-time gesture classification

        Args:
            camera_id: Camera device ID (default 0)
            display_fps: Whether to display FPS counter
            confidence_threshold: Minimum confidence for prediction display
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
            print("ONLINE HAND GESTURE CLASSIFICATION - CNN-1D")
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
                confidence_text = ""

                # Draw hand landmarks and update buffer
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

                        # Extract landmarks and add to buffer
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.frame_buffer.append(landmarks)

                        # Predict only when buffer is full
                        if len(self.frame_buffer) == self.seq_len:
                            sequence = np.array(list(self.frame_buffer))
                            gesture_name, gesture_label, confidence = self.predict(
                                sequence
                            )

                            if confidence >= confidence_threshold:
                                prediction_text = (
                                    f"Gesture: {gesture_name.upper().replace('_', ' ')}"
                                )
                                prediction_color = self.gesture_colors.get(
                                    gesture_name, (255, 255, 255)
                                )
                                confidence_text = f"Confidence: {confidence:.2%}"
                            else:
                                prediction_text = "Low confidence"
                                confidence_text = f"Confidence: {confidence:.2%}"
                        else:
                            prediction_text = f"Buffering... ({len(self.frame_buffer)}/{self.seq_len})"
                else:
                    # Clear buffer when no hand is detected
                    self.frame_buffer.clear()

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
                cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
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

                # Display confidence
                if confidence_text:
                    cv2.putText(
                        frame,
                        confidence_text,
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # Display model info
                model_info = "Model: CNN-1D"
                cv2.putText(
                    frame,
                    model_info,
                    (20, 125),
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

                # Display buffer status
                buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.seq_len}"
                cv2.putText(
                    frame,
                    buffer_text,
                    (w - 200, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (150, 150, 150),
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
                cv2.imshow("Hand Gesture Recognition - CNN-1D", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("\nClassification stopped.")

        # Cleanup temporary model file
        self._cleanup()

    def _cleanup(self):
        """Clean up temporary model file"""
        try:
            if hasattr(self, "temp_model_path") and self.temp_model_path.exists():
                self.temp_model_path.unlink()
                print("Temporary model file cleaned up.")
        except Exception as e:
            print(f"Warning: Could not clean up temporary file: {e}")


def main():
    """Main function to run online classification"""

    # Configuration
    MODELS_DIR = Path("../1_2_a_and_1_2_b/models")
    MODEL_PATH = MODELS_DIR / "cnn1d_model.keras"

    print("=" * 60)
    print("ONLINE HAND GESTURE CLASSIFICATION")
    print("Part I.2.c) - CNN-1D Model")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"\nError: Model file not found: {MODEL_PATH}")
        print("Please run the training script first (1_2_a_and_1_2_b/train_models.py)")
        return

    # Resolve absolute path to handle special characters in Windows paths
    MODEL_PATH = MODEL_PATH.resolve()

    # Create classifier
    classifier = OnlineGestureClassifierCNN(
        model_path=MODEL_PATH, use_z_coord=False, seq_len=30
    )

    # Run online classification
    classifier.run(camera_id=0, display_fps=True, confidence_threshold=0.5)


if __name__ == "__main__":
    # Reduce TensorFlow logging noise
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
