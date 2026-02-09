"""
Online Facial Recognition System - Part II.2)
Real-time facial recognition using trained model
Face Detection -> Crop -> Embeddings (FaceNet) -> Classification
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pickle
import time
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")

# Deep learning for embeddings
from keras_facenet import FaceNet
from collections import deque

# Initialize face cascade for detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class OnlineFacialRecognition:
    """Real-time facial recognition system"""

    def __init__(
        self,
        model_path,
        scaler_path,
        mapping_path,
        embedding_dim=128,
        confidence_threshold=0.6,
    ):
        """
        Initialize online facial recognition

        Args:
            model_path: Path to trained classifier (.pkl)
            scaler_path: Path to embeddings scaler (.pkl)
            mapping_path: Path to subject mapping (.pkl)
            embedding_dim: Embedding dimension (128 for FaceNet)
            confidence_threshold: Minimum confidence for recognition
        """
        # Load model, scaler, and mapping
        with open(model_path, "rb") as f:
            self.classifier = pickle.load(f)

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(mapping_path, "rb") as f:
            self.subject_mapping = pickle.load(f)

        self.confidence_threshold = confidence_threshold
        self.embedding_dim = embedding_dim

        # Initialize embedding model (FaceNet)
        self._init_embedding_model()

        # Face recognition buffer for smoothing predictions
        self.prediction_buffer = deque(maxlen=10)  # Keep last 10 predictions

        print(f"Model loaded: {Path(model_path).name}")
        print(f"Subjects: {list(self.subject_mapping.values())}")
        print(f"Confidence threshold: {confidence_threshold}")

    def _init_embedding_model(self):
        """Initialize FaceNet embedding model"""
        self.embedding_model = FaceNet()
        print(f"FaceNet embedding model initialized ({self.embedding_dim}-dim)")

    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    def extract_face_crop(self, frame, face_rect, margin=20):
        """Extract face crop with margin (same as training)"""
        x, y, w, h = face_rect
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        return frame[y1:y2, x1:x2]

    def get_embedding(self, face_crop):
        """Extract FaceNet embedding from face crop (same as training)"""
        try:
            # Resize to 160x160 for FaceNet
            face_resized = cv2.resize(face_crop, (160, 160))
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Get FaceNet embedding
            embedding = self.embedding_model.embeddings([face_rgb])[0]
            
            # Normalize (L2 normalization)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def predict_subject(self, embedding):
        """
        Predict subject from embedding

        Returns:
            (subject_name, confidence, subject_id)
        """
        # Scale embedding
        embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))

        # Get prediction and probabilities
        subject_id = self.classifier.predict(embedding_scaled)[0]
        probabilities = self.classifier.predict_proba(embedding_scaled)[0]
        confidence = np.max(probabilities)

        subject_name = self.subject_mapping.get(subject_id, "Unknown")

        return subject_name, confidence, subject_id

    def smooth_prediction(self, prediction):
        """Smooth prediction using temporal buffer"""
        self.prediction_buffer.append(prediction)

        # Get most common prediction from buffer
        if len(self.prediction_buffer) > 0:
            predictions = [p[0] for p in self.prediction_buffer]
            from collections import Counter

            most_common = Counter(predictions).most_common(1)[0][0]

            # Return most common prediction with average confidence
            confidences = [p[1] for p in self.prediction_buffer if p[0] == most_common]
            avg_confidence = np.mean(confidences) if confidences else 0

            return most_common, avg_confidence

        return prediction[0], prediction[1]

    def run(self, camera_id=0, display_fps=True):
        """Run online facial recognition"""
        cap = cv2.VideoCapture(camera_id)

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # FPS calculation
        prev_time = time.time()
        fps = 0

        # Color mapping for subjects
        colors = {}
        color_list = [
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 0, 0),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 165, 0),  # Orange
            (165, 42, 42),  # Brown
        ]

        for idx, (id, name) in enumerate(self.subject_mapping.items()):
            colors[name] = color_list[idx % len(color_list)]

        print("\n" + "=" * 60)
        print("ONLINE FACIAL RECOGNITION")
        print("=" * 60)
        print("Press 'q' to quit\n")

        face_detected = False
        current_subject = "Unknown"
        current_confidence = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Detect faces
            faces = self.detect_faces(frame)

            # If no faces detected
            if len(faces) == 0:
                face_detected = False
                self.prediction_buffer.clear()
                current_subject = "Unknown"
                current_confidence = 0.0
            else:
                face_detected = True

                # Process first detected face
                x, y, w_face, h_face = faces[0]
                face_crop = self.extract_face_crop(frame, faces[0])

                # Skip very small crops
                if face_crop.shape[0] >= 40 and face_crop.shape[1] >= 40:
                    # Get embedding
                    embedding = self.get_embedding(face_crop)

                    if embedding is not None:
                        # Predict
                        subject_name, confidence, _ = self.predict_subject(embedding)

                        # Smooth prediction
                        subject_name, confidence = self.smooth_prediction(
                            (subject_name, confidence)
                        )

                        # Update current display
                        if confidence >= self.confidence_threshold:
                            current_subject = subject_name
                            current_confidence = confidence
                        else:
                            current_subject = "Low Confidence"
                            current_confidence = confidence

                # Draw face rectangle
                color = colors.get(current_subject, (200, 200, 200))
                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), color, 2)

                # Draw face label
                label = f"{current_subject} ({current_confidence:.2%})"
                cv2.putText(
                    frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

            # Calculate FPS
            if display_fps:
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
                prev_time = curr_time

            # Draw information panel
            panel_height = 120
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            # Status text
            status = "Face Detected" if face_detected else "No Face Detected"
            status_color = (0, 255, 0) if face_detected else (200, 200, 200)

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                f"Subject: {current_subject}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {current_confidence:.2%}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # FPS counter
            if display_fps:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Instructions
            cv2.putText(
                frame,
                "Press 'Q' to quit | Threshold: 0.6",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Show frame
            cv2.imshow("Facial Recognition - Online", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nFacial recognition stopped.")


def main():
    """Main function"""

    # Configuration
    MODEL_DIR = Path("../2_1/models")
    
    # Choose classifier: 'svm', 'rf', 'knn', 'mlp', etc.
    CLASSIFIER = "svm"  # Change this to use different model
    
    MODEL_PATH = MODEL_DIR / f"facial_recognition_model_{CLASSIFIER}.pkl"
    SCALER_PATH = MODEL_DIR / "embeddings_scaler.pkl"
    MAPPING_PATH = MODEL_DIR / "subject_mapping.pkl"

    print("=" * 60)
    print("ONLINE FACIAL RECOGNITION - Part II.2)")
    print("=" * 60)
    print(f"Using classifier: {CLASSIFIER.upper()}")

    # Check if model files exist
    if not MODEL_PATH.exists():
        print(f"\nError: Model file not found: {MODEL_PATH}")
        print("Please train the model first using 2_1/facial_recognition.py")
        return

    if not SCALER_PATH.exists():
        print(f"\nError: Scaler file not found: {SCALER_PATH}")
        return

    if not MAPPING_PATH.exists():
        print(f"\nError: Mapping file not found: {MAPPING_PATH}")
        return

    print(f"\nModel path: {MODEL_PATH}")
    print(f"Scaler path: {SCALER_PATH}")
    print(f"Mapping path: {MAPPING_PATH}")

    # Create recognizer and run
    try:
        recognizer = OnlineFacialRecognition(
            MODEL_PATH, SCALER_PATH, MAPPING_PATH, confidence_threshold=0.6
        )
        recognizer.run(camera_id=0, display_fps=True)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("PROGRAM ENDED")
    print("=" * 60)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
