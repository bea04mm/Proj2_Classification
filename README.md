# Hand Gesture Classification Project - Part I

Static hand gesture classification using MediaPipe Hands, classic Machine Learning models, and Deep Learning (CNN-1D).

## Project Structure

```
Proj2_Classification/
├── Videos_Datasets/          # Dataset with recorded videos
│   ├── subject_*/
│   │   ├── open_palm/
│   │   ├── closed_hand/
│   │   ├── v_shape/
│   │   ├── hand_right/
│   │   ├── hand_left/
│   │   └── thumbs_up/
├── 1_2_a/                    # Part I.2.a) - Classic ML offline training
│   ├── train_classic_models.py
│   └── models/              # Trained classic models (generated)
├── 1_2_b/                    # Part I.2.b) - CNN-1D offline training
│   ├── train_cnn_model.py
│   └── models/              # Trained CNN model (generated)
├── 1_2_a_and_1_2_b/         # Combined training script (all models)
│   ├── train_models.py
│   └── models/              # All trained models (generated)
├── 1_2_c/                    # Part I.2.c) - Online classification
│   ├── online_classic_models.py  # Real-time with classic ML
│   └── online_cnn_model.py       # Real-time with CNN-1D
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Initial Setup (Anaconda)

### 1. Create Conda environment

```bash
conda create -n hand_gesture python=3.10 -y
```

### 2. Activate the environment

```bash
conda activate hand_gesture
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.10
- MediaPipe 0.10.9
- OpenCV 4.8.1.78
- protobuf 3.20.3
- scikit-learn 1.3.2
- NumPy 1.24.3
- Matplotlib 3.8.2
- Seaborn 0.13.0
- TensorFlow 2.10.0

See [requirements.txt](requirements.txt) for complete list.

## Recognized Gestures

The project recognizes 6 static hand gestures:

1. **open_palm** - Open palm hand (green)
2. **closed_hand** - Closed hand/fist (red)
3. **v_shape** - V-shape hand/peace sign (magenta)
4. **hand_right** - Hand pointing right (cyan)
5. **hand_left** - Hand pointing left (yellow)
6. **thumbs_up** - Thumbs up (orange)

---

## Part I.2.a) - Model Training (Offline)

### Overview

Trains 3 classic Machine Learning models (SVM, KNN, Random Forest) using hand landmarks extracted from video clips.

### Running the Training Script

```bash
cd 1_2_a
python train_classic_models.py
```

### What the Script Does

1. **Loads video clips** from `Videos_Datasets/` folder
2. **Extracts hand landmarks** using MediaPipe Hands (21 landmarks per hand)
3. **Uses (x,y) coordinates** by default (can be changed to (x,y,z) in the script)
4. **Splits data**: 
   - Training: clips 1-4 (first 4 clips per gesture per subject)
   - Testing: clips 5-6 (last 2 clips per gesture per subject)
5. **Trains three models**:
   - Support Vector Machine (SVM) with RBF kernel
   - K-Nearest Neighbors (KNN) with k=5
   - Random Forest with 100 trees
6. **Evaluates models** with:
   - Accuracy scores
   - Classification reports (precision, recall, f1-score)
   - Confusion matrices (saved as PNG images)
7. **Saves all trained models** in the `1_2_a/models/` folder

### Output Files

After running, you'll find in `1_2_a/models/`:

- `scaler.pkl` - StandardScaler for feature normalization
- `svm_model.pkl` - Trained SVM model
- `knn_model.pkl` - Trained KNN model
- `randomforest_model.pkl` - Trained Random Forest model
- `confusion_matrix_SVM.png` - SVM confusion matrix visualization
- `confusion_matrix_KNN.png` - KNN confusion matrix visualization
- `confusion_matrix_RandomForest.png` - Random Forest confusion matrix visualization
- `model_comparison.png` - Bar chart comparing model accuracies

### Configuration

You can modify parameters in `train_classic_models.py`:

```python
# In the main() function:
DATASET_PATH = Path('../Videos_Datasets')  # Path to dataset
USE_Z_COORD = False  # Set to True to use (x,y,z) coordinates
TRAIN_CLIPS = [1, 2, 3, 4]  # Training clip indices
TEST_CLIPS = [5, 6]  # Testing clip indices
```

### Model Hyperparameters

```python
# SVM
SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

# KNN
KNeighborsClassifier(n_neighbors=5, weights='distance')

# Random Forest
RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
```

---

## Part I.2.b) - CNN-1D Model Training (Offline)

### Overview

Trains a 1D Convolutional Neural Network (CNN-1D) using temporal sequences of hand landmarks extracted from video clips. Unlike classic ML models that use averaged features, CNN-1D processes sequences of frames to capture temporal patterns.

### Running the CNN Training Script

```bash
cd 1_2_b
python train_cnn_model.py
```

**Alternative:** Train all models (classic + CNN) together:
```bash
cd 1_2_a_and_1_2_b
python train_models.py
```

### What the Script Does

1. **Loads video clips** from `Videos_Datasets/` folder
2. **Extracts hand landmark sequences** using MediaPipe Hands
3. **Processes temporal sequences**:
   - Each clip is processed as a sequence of 30 frames
   - 21 landmarks per frame with (x,y) coordinates = 42 features per frame
   - Final input shape: (30, 42) per clip
4. **Splits data**:
   - Training: clips 1-4 (first 4 clips per gesture per subject)
   - Testing: clips 5-6 (last 2 clips per gesture per subject)
5. **Builds CNN-1D architecture**:
   - 3 Conv1D layers with batch normalization and dropout
   - MaxPooling for temporal downsampling
   - Dense layers for classification
6. **Trains the model**:
   - Adam optimizer with learning rate scheduling
   - Early stopping to prevent overfitting
   - Model checkpointing to save best model
7. **Evaluates performance**:
   - Accuracy scores on test set
   - Classification report (precision, recall, f1-score)
   - Confusion matrix visualization
   - Training history plots (loss & accuracy curves)
8. **Saves trained model** as `1_2_b/models/cnn1d_model.keras`

### CNN-1D Architecture

```python
# Input: (30 frames, 42 features)
Conv1D(64 filters, kernel=3) → BatchNorm → ReLU → Dropout(0.3) → MaxPool
Conv1D(128 filters, kernel=3) → BatchNorm → ReLU → Dropout(0.3) → MaxPool
Conv1D(256 filters, kernel=3) → BatchNorm → ReLU → Dropout(0.3)
GlobalAveragePooling1D
Dense(128) → BatchNorm → ReLU → Dropout(0.5)
Dense(64) → BatchNorm → ReLU → Dropout(0.5)
Dense(6, softmax) # Output: 6 gesture classes
```

### Output Files

After running, you'll find in `1_2_b/models/`:

- `cnn1d_model.keras` - Trained CNN-1D model
- `confusion_matrix_CNN1D.png` - Confusion matrix visualization
- `training_history.png` - Loss and accuracy curves

### Configuration

You can modify parameters in `train_cnn_model.py`:

```python
# In the main() function:
DATASET_PATH = Path('../Videos_Datasets')  # Path to dataset
USE_Z_COORD = False  # Set to True to use (x,y,z) coordinates
SEQ_LEN = 30  # Number of frames per sequence
TRAIN_CLIPS = [1, 2, 3, 4]  # Training clip indices
TEST_CLIPS = [5, 6]  # Testing clip indices
BATCH_SIZE = 32
EPOCHS = 100  # With early stopping
```

### Training Hyperparameters

```python
# Model compilation
optimizer = Adam(learning_rate=0.001)
loss = 'sparse_categorical_crossentropy'

# Callbacks
EarlyStopping(patience=15, restore_best_weights=True)
ReduceLROnPlateau(factor=0.5, patience=5)
ModelCheckpoint(save_best_only=True)
```

---

## Part I.2.c) - Online Classification (Real-time)

### Overview

Real-time hand gesture classification using webcam input. Supports both classic ML models (Part I.2.a) and CNN-1D model (Part I.2.b).

### Prerequisites

**For Classic Models:**
```bash
cd 1_2_a
python train_classic_models.py
```

**For CNN Model:**
```bash
cd 1_2_b
python train_cnn_model.py
```

---

### Option 1: Classic ML Models (SVM, KNN, Random Forest)

### Running Online Classification

```bash
cd 1_2_c
python online_classic_models.py
```

### Usage

1. Select which model to test:
   - **1** - SVM (typically best performance: ~98.75% accuracy)
   - **2** - KNN (~95.90% accuracy)
   - **3** - Random Forest (~96.00% accuracy)
   - **all** - Test all models sequentially

2. The webcam will open and start classifying gestures in real-time

3. Press **'q'** to quit

### How It Works

1. **Captures video** from webcam (1280x720 @ 30fps)
2. **Detects hand** using MediaPipe Hands
3. **Extracts 21 landmarks** (x,y coordinates)
4. **Normalizes features** using saved scaler
5. **Predicts gesture** using selected model
6. **Displays result** on screen with colored labels

### Display Information

The video window shows:
- **Gesture prediction** with colored label (see gesture colors above)
- **Model type** being used (SVM, KNN, or RandomForest)
- **FPS counter** (frames per second)
- **Hand landmarks** overlaid on detected hand
- **Instructions** (press 'Q' to quit)

---

### Option 2: CNN-1D Model

### Running Online Classification with CNN

```bash
cd 1_2_c
python online_cnn_model.py
```

### How It Works (CNN Version)

1. **Captures video** from webcam (1280x720 @ 30fps)
2. **Detects hand** using MediaPipe Hands
3. **Extracts landmarks** and builds a **sliding window buffer**:
   - Maintains a sequence of 30 frames
   - Adds new landmarks to the buffer each frame
   - Once buffer is full (30 frames), makes predictions
4. **Predicts gesture** using CNN-1D model
5. **Smooths predictions** using majority voting over recent predictions
6. **Displays result** with colored labels and confidence scores

### Display Information (CNN)

The video window shows:
- **Gesture prediction** with colored label
- **Confidence score** (softmax probability)
- **Buffer status** (filling/ready for prediction)
- **Model type** "CNN-1D"
- **FPS counter**
- **Hand landmarks** overlaid on detected hand
- **Instructions** (press 'Q' to quit)

### Key Differences: Classic ML vs CNN-1D

| Feature | Classic ML | CNN-1D |
|---------|-----------|--------|
| **Input** | Single frame (averaged landmarks) | Sequence of 30 frames |
| **Prediction** | Immediate (every frame) | After buffer fills (30 frames) |
| **Temporal Info** | No | Yes |
| **Smoothing** | Optional averaging | Built-in + majority voting |
| **Latency** | Very low (~30ms) | Slightly higher (~50ms) |
| **Accuracy** | ~95-99% | ~97-99% |

---

### General Tips for Best Results

- Ensure good **lighting** conditions
- Keep hand within **camera frame**
- Position hand at **medium distance** from camera (30-50cm)
- Make **clear, distinct gestures**
- The **SVM model** typically gives best results
- Avoid cluttered backgrounds

---

## Dataset Organization

Videos should be organized as follows:

```
Videos_Datasets/
├── subject_name_1/
│   ├── open_palm/
│   │   ├── clip_1.avi
│   │   ├── clip_2.avi
│   │   ├── ...
│   │   └── clip_6.avi
│   ├── closed_hand/
│   │   └── (6 clips)
│   ├── v_shape/
│   ├── hand_right/
│   ├── hand_left/
│   └── thumbs_up/
├── subject_name_2/
│   └── (same structure)
└── ...
```

**Recording Guidelines:**
- 6 clips per gesture per subject
- 10 seconds per clip @ 15fps
- Upper body visible (head, right hand, trunk)
- Varied hand positions and distances
- Supported formats: `.avi`, `.mp4`

---

## Troubleshooting

### Environment Issues

**Environment not found:**
```bash
conda env list
conda activate hand_gesture
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

### Training Issues (Part I.2.a)

**No landmarks detected:**
- Check if videos contain visible hands
- Ensure proper lighting in videos
- Verify video file formats (.avi or .mp4)

**Low accuracy:**
- Try adjusting model hyperparameters
- Use (x,y,z) coordinates instead of (x,y)
- Increase training data
- Check data quality and variability

**Import errors:**
```bash
pip install -r requirements.txt
```

### Online Classification Issues (Part I.2.c)

**Models not found:**
```bash
# For classic models:
cd 1_2_a
python train_classic_models.py

# For CNN model:
cd 1_2_b
python train_cnn_model.py

# For all models:
cd 1_2_a_and_1_2_b
python train_models.py
```

**No hand detected:**
- Check lighting and hand position
- Ensure camera is working
- Keep hand visible in frame

**Wrong predictions:**
- Try the **SVM model** for better accuracy (classic ML)
- Try the **CNN-1D model** for temporal pattern recognition
- Ensure clear, distinct gestures
- Improve lighting conditions
- **For CNN:** Keep gesture stable for 1-2 seconds while buffer fills

**Camera not found:**
- Change `camera_id` parameter in code (default is 0)
- Check camera permissions
- Verify camera is not in use by another application

**Low FPS:**
- Close other applications
- Use a faster computer
- Reduce camera resolution in code

---

## Performance Results

Based on typical training results:

### Classic Machine Learning Models

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM** | ~98.75% | Best overall performance, recommended |
| **Random Forest** | ~96.00% | Good balance of speed and accuracy |
| **KNN** | ~95.90% | Fast inference, slightly lower accuracy |

### Deep Learning Model

| Model | Accuracy | Notes |
|-------|----------|-------|
| **CNN-1D** | ~97-99% | Captures temporal patterns, robust to variations |

### Comparison

- **Classic ML**: Best for instant single-frame predictions, lower latency
- **CNN-1D**: Best for temporal pattern recognition, smoother predictions, requires sequence buffer
- All models show excellent performance (>95% accuracy) for the 6-gesture classification task

### Recommended Use Cases

- **SVM**: Best overall classic ML model, use when immediate response is critical
- **CNN-1D**: Best for continuous gesture recognition with temporal context
- **Random Forest**: Good alternative with balanced performance
- **KNN**: Fastest inference, suitable for resource-constrained scenarios

