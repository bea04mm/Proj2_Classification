"""
Facial Recognition System - Part II.1)
Implements: Face Detection + Crop + Embeddings (FaceNet) + Classic Classifier
Uses recorded video data from Videos_Datasets folder organized by subject names
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# Deep learning & embeddings
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cosine

# Classical classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Initialize face cascade for detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class FaceDetectorAndEmbedding:
    """Detect faces and extract embeddings from video frames"""

    def __init__(self, model_type="vgg16", embedding_dim=512):
        """
        Initialize face detector and embedding model

        Args:
            model_type: 'vgg16' for VGG16-based embeddings
            embedding_dim: Dimension of embeddings
        """
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Initialize a pre-trained model for embeddings"""
        if self.model_type == "vgg16":
            # Load VGG16 without top classification layer
            base_model = VGG16(
                weights="imagenet", input_shape=(224, 224, 3), include_top=False
            )
            # Take the last conv layer output as embeddings
            self.embedding_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=base_model.get_layer("block5_pool").output,
            )
            # Flatten to get embeddings
            self.embedding_model = tf.keras.Sequential(
                [
                    self.embedding_model,
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(self.embedding_dim),
                ]
            )

        self.embedding_model.trainable = False
        print(f"Embedding model initialized: {self.model_type}")

    def detect_faces(self, frame):
        """
        Detect faces in a frame using Haar Cascade

        Args:
            frame: Input video frame

        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    def extract_face_crop(self, frame, face_rect, margin=10):
        """
        Extract and crop face region from frame

        Args:
            frame: Input frame
            face_rect: (x, y, w, h) tuple from detector
            margin: Margin around face for context

        Returns:
            Cropped face image
        """
        x, y, w, h = face_rect
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        return frame[y1:y2, x1:x2]

    def get_embedding(self, face_crop):
        """
        Extract embedding from face crop

        Args:
            face_crop: Cropped face image

        Returns:
            Embedding vector (1D array)
        """
        try:
            # Resize to VGG16 input size
            face_resized = cv2.resize(face_crop, (224, 224))

            # Preprocess
            face_array = keras_image.img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = preprocess_input(face_array)

            # Get embedding
            embedding = self.embedding_model.predict(face_array, verbose=0)
            embedding = embedding.flatten()

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def process_video(self, video_path, max_frames=None, skip_frames=5):
        """
        Process video and extract face embeddings

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            skip_frames: Skip N frames between detections

        Returns:
            List of embeddings detected in video
        """
        embeddings = []
        cap = cv2.VideoCapture(str(video_path))

        frame_count = 0
        processed_count = 0

        while cap.isOpened() and (max_frames is None or processed_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            frame_count += 1

            # Detect faces
            faces = self.detect_faces(frame)

            for face_rect in faces:
                face_crop = self.extract_face_crop(frame, face_rect)

                # Skip very small crops
                if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                    continue

                embedding = self.get_embedding(face_crop)
                if embedding is not None:
                    embeddings.append(embedding)

            processed_count += 1

            if processed_count % 10 == 0:
                print(
                    f"  Processed {processed_count} frames, extracted {len(embeddings)} embeddings"
                )

        cap.release()
        return np.array(embeddings)


class FacialRecognitionPipeline:
    """Complete facial recognition pipeline"""

    def __init__(self, dataset_path, output_dir="models"):
        """
        Initialize the pipeline

        Args:
            dataset_path: Path to Videos_Datasets folder
            output_dir: Output directory for models
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.detector = FaceDetectorAndEmbedding()
        self.classifier = None
        self.scaler = StandardScaler()
        self.subject_mapping = {}
        self.reverse_mapping = {}

    def load_embeddings_from_videos(self):
        """
        Load all videos organized by subject and extract embeddings

        Returns:
            X: Array of embeddings
            y: Array of subject labels
            subject_names: List of unique subject names
        """
        print("Loading embeddings from videos...")
        print(f"Dataset path: {self.dataset_path}\n")

        embeddings_list = []
        labels_list = []
        subject_names = []

        subject_id = 0

        # Iterate over subject folders
        for subject_folder in sorted(self.dataset_path.iterdir()):
            if not subject_folder.is_dir() or subject_folder.name.startswith("__"):
                continue

            subject_name = subject_folder.name.replace("subject_", "")
            subject_names.append(subject_name)
            self.subject_mapping[subject_id] = subject_name
            self.reverse_mapping[subject_name] = subject_id

            print(f"\nProcessing subject: {subject_name}")

            # Collect all video files from subject
            video_files = list(subject_folder.glob("*/*.avi")) + list(
                subject_folder.glob("*/*.mp4")
            )

            if not video_files:
                print(f"  No video files found for {subject_name}")
                subject_id += 1
                continue

            print(f"  Found {len(video_files)} video files")

            # Process each video
            for video_file in video_files:
                print(f"  Processing: {video_file.name}")
                embeddings = self.detector.process_video(
                    video_file, max_frames=50, skip_frames=3
                )

                if len(embeddings) > 0:
                    embeddings_list.extend(embeddings)
                    labels_list.extend([subject_id] * len(embeddings))
                    print(f"    Extracted {len(embeddings)} embeddings")
                else:
                    print(f"    No faces detected")

            subject_id += 1

        print(f"\n{'='*60}")
        print(f"Total embeddings extracted: {len(embeddings_list)}")
        print(f"Total subjects: {len(subject_names)}")

        if embeddings_list:
            X = np.array(embeddings_list)
            y = np.array(labels_list)
            return X, y, subject_names
        else:
            raise ValueError("No embeddings extracted from videos!")

    def train(self, test_size=0.2, classifier_type="svm"):
        """
        Train facial recognition classifier

        Args:
            test_size: Proportion of data for testing
            classifier_type: 'svm' or 'rf' (RandomForest)
        """
        print("\n" + "=" * 60)
        print("TRAINING FACIAL RECOGNITION SYSTEM")
        print("=" * 60)

        # Load embeddings
        X, y, subject_names = self.load_embeddings_from_videos()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale embeddings
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train classifier
        print(f"\nTraining {classifier_type.upper()} classifier...")
        if classifier_type == "svm":
            self.classifier = SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True, verbose=1
            )
        elif classifier_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100, max_depth=30, random_state=42, n_jobs=-1, verbose=1
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.classifier.fit(X_train_scaled, y_train)
        print("Training complete!")

        # Evaluate
        y_pred = self.classifier.predict(X_test_scaled)
        y_pred_proba = self.classifier.predict_proba(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=subject_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot results
        self._plot_results(cm, subject_names, accuracy)

        # Save model
        self._save_model(classifier_type)

        return X_test_scaled, y_test, y_pred, y_pred_proba

    def _plot_results(self, cm, subject_names, accuracy):
        """Plot confusion matrix and save"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=subject_names,
            yticklabels=subject_names,
            cbar_kws={"label": "Count"},
        )
        plt.title(f"Facial Recognition - Confusion Matrix (Accuracy: {accuracy:.4f})")
        plt.ylabel("True Subject")
        plt.xlabel("Predicted Subject")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")

    def _save_model(self, classifier_type):
        """Save trained model and scaler"""
        model_path = self.output_dir / f"facial_recognition_model_{classifier_type}.pkl"
        scaler_path = self.output_dir / "embeddings_scaler.pkl"
        mapping_path = self.output_dir / "subject_mapping.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.subject_mapping, f)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Subject mapping saved to {mapping_path}")


def main():
    """Main function"""

    # Configuration
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "models"
    CLASSIFIER_TYPE = "svm"  # 'svm' or 'rf'

    print("=" * 60)
    print("FACIAL RECOGNITION SYSTEM - Part II.1)")
    print("Face Detection + Crop + Embeddings + Classifier")
    print("=" * 60)
    print(f"\nDataset path: {DATASET_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Classifier: {CLASSIFIER_TYPE}")

    if not DATASET_PATH.exists():
        print(f"\nError: Dataset path does not exist: {DATASET_PATH}")
        print("Please ensure the Videos_Datasets folder is in the correct location.")
        return

    # Create pipeline and train
    pipeline = FacialRecognitionPipeline(DATASET_PATH, OUTPUT_DIR)
    pipeline.train(test_size=0.2, classifier_type=CLASSIFIER_TYPE)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
