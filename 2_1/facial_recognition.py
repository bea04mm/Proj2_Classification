"""
Sistema de Reconhecimento Facial - Parte II.1
Detecção de faces + Recorte + Embeddings (FaceNet) + Classificador
Usa vídeos da pasta Videos_Datasets organizada por sujeito
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

# Classificadores clássicos
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Inicializar Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class FaceDetectorAndEmbedding:
    """Deteta faces e extrai embeddings"""

    def __init__(self, model_type="facenet", embedding_dim=128):
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Inicializa FaceNet"""
        if self.model_type == "facenet":
            self.embedding_model = FaceNet()
        print(f"Modelo de embeddings inicializado: {self.model_type}")

    def detect_faces(self, frame):
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
        x, y, w, h = face_rect
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        return frame[y1:y2, x1:x2]

    def get_embedding(self, face_crop):
        try:
            face_resized = cv2.resize(face_crop, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            embedding = self.embedding_model.embeddings([face_rgb])[0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            return embedding
        except Exception as e:
            print(f"Erro ao extrair embedding: {e}")
            return None

    def process_video(self, video_path, max_frames=None, skip_frames=5):
        embeddings = []
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        processed_count = 0

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
                if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                    continue
                embedding = self.get_embedding(face_crop)
                if embedding is not None:
                    embeddings.append(embedding)

            processed_count += 1
            if processed_count % 10 == 0:
                print(
                    f"  Processadas {processed_count} frames, extraídas {len(embeddings)} embeddings"
                )

        cap.release()
        return np.array(embeddings)


class FacialRecognitionPipeline:
    """Pipeline completo de reconhecimento facial"""

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

            print(f"\nProcessando sujeito: {subject_name}")

            video_files = list(subject_folder.glob("*/*.avi")) + list(
                subject_folder.glob("*/*.mp4")
            )
            if not video_files:
                print(f"  Sem vídeos para {subject_name}")
                subject_id += 1
                continue

            for video_file in video_files:
                print(f"  Processando: {video_file.name}")
                embeddings = self.detector.process_video(
                    video_file, max_frames=50, skip_frames=3
                )
                if len(embeddings) > 0:
                    embeddings_list.extend(embeddings)
                    labels_list.extend([subject_id] * len(embeddings))
                    print(f"    Extraídas {len(embeddings)} embeddings")
                else:
                    print(f"    Nenhuma face detectada")

            subject_id += 1

        print(f"\n{'='*60}")
        print(f"Total embeddings extraídas: {len(embeddings_list)}")
        print(f"Total sujeitos: {len(subject_names)}")

        if embeddings_list:
            X = np.array(embeddings_list)
            y = np.array(labels_list)
            return X, y, subject_names
        else:
            raise ValueError("Não foram extraídas embeddings!")

    def train(self, test_size=0.2, classifier_type="svm"):
        print("\n" + "=" * 60)
        print("TREINO DO SISTEMA DE RECONHECIMENTO FACIAL")
        print("=" * 60)

        X, y, subject_names = self.load_embeddings_from_videos()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTreinando classificador {classifier_type.upper()}...")
        if classifier_type == "svm":
            self.classifier = SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True, verbose=1
            )
        elif classifier_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100, max_depth=30, random_state=42, n_jobs=-1, verbose=1
            )
        else:
            raise ValueError(f"Classificador desconhecido: {classifier_type}")

        self.classifier.fit(X_train_scaled, y_train)
        print("Treino completo!")

        y_pred = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAcurácia de teste: {accuracy:.4f}")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, target_names=subject_names))

        cm = confusion_matrix(y_test, y_pred)
        self._plot_results(cm, subject_names, accuracy)
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
            cbar_kws={"label": "Contagem"},
        )
        plt.title(f"Matriz de Confusão (Acurácia: {accuracy:.4f})")
        plt.ylabel("Sujeito Real")
        plt.xlabel("Sujeito Previsto")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Matriz de confusão salva em {self.output_dir / 'confusion_matrix.png'}")

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

        print(f"Modelo salvo em {model_path}")
        print(f"Scaler salvo em {scaler_path}")
        print(f"Mapeamento de sujeitos salvo em {mapping_path}")


def main():
    DATASET_PATH = Path("../Videos_Datasets")
    OUTPUT_DIR = "models"
    CLASSIFIER_TYPE = "rf"  # 'svm' ou 'rf'

    if not DATASET_PATH.exists():
        print(f"\nErro: Dataset não existe: {DATASET_PATH}")
        return

    pipeline = FacialRecognitionPipeline(DATASET_PATH, OUTPUT_DIR)
    pipeline.train(test_size=0.2, classifier_type=CLASSIFIER_TYPE)
    print("\nTREINO CONCLUÍDO")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
