"""
System of Facial Recognition - Part II.1
Face Detection + Crop + Embeddings (FaceNet) + Classifier
Uses videos from the Videos_Datasets folder organized by subject
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

# Deep learning & embeddings
from keras_facenet import FaceNet

# Classic classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class FaceDetectorAndEmbedding:
    """Detect faces and extract embeddings"""

    def __init__(self, model_type="facenet", embedding_dim=128):
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Initialize FaceNet"""
        if self.model_type == "facenet":
            self.embedding_model = FaceNet()
        print(f"Embedding model initialized: {self.model_type}")

    def detect_faces(self, frame):
        """Detect faces with optimized parameters for better quality"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More precise
            minNeighbors=8,     # Only well-detected faces
            minSize=(80, 80),   # Larger minimum size for better quality
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    def extract_face_crop(self, frame, face_rect, margin=20):
        """Extract face with larger margin to capture full context"""
        x, y, w, h = face_rect
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        return frame[y1:y2, x1:x2]
    
    def is_blurry(self, image, threshold=100):
        """Detect blurry images using variance of Laplacian"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold

    def get_embedding(self, face_crop):
        try:
            face_resized = cv2.resize(face_crop, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            embedding = self.embedding_model.embeddings([face_rgb])[0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def process_video(self, video_path, max_frames=None, skip_frames=2):
        """Process video with quality filters to extract embeddings"""
        embeddings = []
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        processed_count = 0
        
        # Statistics
        blurry_count = 0
        small_face_count = 0

        while cap.isOpened() and (max_frames is None or processed_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            frame_count += 1
            faces = self.detect_faces(frame)

            for face_rect in faces:
                face_crop = self.extract_face_crop(frame, face_rect)
                
                # Filter 1: More stringent minimum size
                if face_crop.shape[0] < 80 or face_crop.shape[1] < 80:
                    small_face_count += 1
                    continue
                
                # Filter 2: Reject blurry images
                if self.is_blurry(face_crop, threshold=120):
                    blurry_count += 1
                    continue
                
                embedding = self.get_embedding(face_crop)
                if embedding is not None:
                    embeddings.append(embedding)

            processed_count += 1
            if processed_count % 20 == 0:
                print(
                    f"  Processed {processed_count} frames, extracted {len(embeddings)} embeddings "
                    f"(rejected: {blurry_count} blurry, {small_face_count} small)"
                )

        cap.release()
        print(f"  Total: {len(embeddings)} valid embeddings from {frame_count} frames")
        return np.array(embeddings)


class FacialRecognitionPipeline:
    """Complete facial recognition pipeline"""

    def __init__(self, dataset_path, output_dir="models"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.detector = FaceDetectorAndEmbedding()
        self.classifier = None
        self.scaler = StandardScaler()
        self.subject_mapping = {}
        self.reverse_mapping = {}

    def load_embeddings_from_videos(self):
        embeddings_list = []
        labels_list = []
        subject_names = []
        subject_id = 0

        for subject_folder in sorted(self.dataset_path.iterdir()):
            if not subject_folder.is_dir() or subject_folder.name.startswith("__"):
                continue

            subject_name = subject_folder.name.replace("subject_", "")
            subject_names.append(subject_name)
            self.subject_mapping[subject_id] = subject_name
            self.reverse_mapping[subject_name] = subject_id

            print(f"\nProcessing subject: {subject_name}")

            video_files = list(subject_folder.glob("*/*.avi")) + list(
                subject_folder.glob("*/*.mp4")
            )
            if not video_files:
                print(f"  No videos for {subject_name}")
                subject_id += 1
                continue

            for video_file in video_files:
                print(f"  Processing: {video_file.name}")
                embeddings = self.detector.process_video(
                    video_file, max_frames=150, skip_frames=2  # More data
                )
                if len(embeddings) > 0:
                    embeddings_list.extend(embeddings)
                    labels_list.extend([subject_id] * len(embeddings))
                    print(f"    Extracted {len(embeddings)} quality embeddings")
                else:
                    print(f"    No quality faces detected")

            subject_id += 1

        print(f"\n{'='*60}")
        print(f"Total embeddings extracted: {len(embeddings_list)}")
        print(f"Total subjects: {len(subject_names)}")

        if embeddings_list:
            X = np.array(embeddings_list)
            y = np.array(labels_list)
            return X, y, subject_names
        else:
            raise ValueError("No embeddings were extracted!")

    def train(self, test_size=0.2, classifier_type="svm"):
        print("\n" + "=" * 60)
        print("FACIAL RECOGNITION SYSTEM TRAINING")
        print("=" * 60)

        X, y, subject_names = self.load_embeddings_from_videos()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTraining {classifier_type.upper()} classifier...")
        if classifier_type == "svm":
            self.classifier = SVC(
                kernel="rbf", 
                C=10.0,              # Increased for better separation
                gamma="scale", 
                probability=True, 
                class_weight='balanced',  # Balance classes
                verbose=1
            )
        elif classifier_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=200,    # More trees
                max_depth=None,      # No limit (more expressive)
                min_samples_split=5, # Avoid overfitting
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42, 
                n_jobs=-1, 
                verbose=1
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

        self.classifier.fit(X_train_scaled, y_train)
        print("Training complete!")

        # Predictions with confidence analysis
        y_pred = self.classifier.predict(X_test_scaled)
        y_proba = self.classifier.predict_proba(X_test_scaled)
        
        # Confidence analysis
        confidence_scores = np.max(y_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)
        low_confidence_count = np.sum(confidence_scores < 0.7)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{'='*60}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"Low confidence predictions (<0.7): {low_confidence_count}/{len(y_test)} ({100*low_confidence_count/len(y_test):.1f}%)")
        print(f"Minimum confidence: {np.min(confidence_scores):.4f}")
        print(f"Maximum confidence: {np.max(confidence_scores):.4f}")
        print(f"{'='*60}")
        
        # Filter subject names to only those present in test set
        unique_test_labels = np.unique(y_test)
        test_subject_names = [subject_names[i] for i in unique_test_labels]
        
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, labels=unique_test_labels, target_names=test_subject_names))
        
        # Show low confidence cases
        if low_confidence_count > 0:
            print("\n  Examples of low confidence predictions:")
            low_conf_indices = np.where(confidence_scores < 0.7)[0][:5]
            for idx in low_conf_indices:
                true_label = subject_names[y_test[idx]]
                pred_label = subject_names[y_pred[idx]]
                conf = confidence_scores[idx]
                print(f"  Real: {true_label:20s} | Predicted: {pred_label:20s} | Confidence: {conf:.4f}")

        cm = confusion_matrix(y_test, y_pred, labels=unique_test_labels)
        self._plot_results(cm, test_subject_names, accuracy)
        self._save_model(classifier_type)

        return X_test_scaled, y_test, y_pred

    def _plot_results(self, cm, subject_names, accuracy):
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
        plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
        plt.ylabel("True Subject")
        plt.xlabel("Predicted Subject")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")

    def _save_model(self, classifier_type):
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
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "models"
    CLASSIFIER_TYPE = "svm"  # 'svm' or 'rf' - SVM usually better for faces

    if not DATASET_PATH.exists():
        print(f"\nError: Dataset does not exist: {DATASET_PATH}")
        return

    pipeline = FacialRecognitionPipeline(DATASET_PATH, OUTPUT_DIR)
    pipeline.train(test_size=0.2, classifier_type=CLASSIFIER_TYPE)
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
