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
│   ├── train_classic_models.py         # Absolute coordinates
│   ├── train_classic_models_v2.py      # Relative to wrist
│   └── models_comparison/              # Trained models (generated)
├── 1_2_b/                    # Part I.2.b) - CNN-1D offline training
│   ├── train_cnn_model.py              # Absolute coordinates
│   ├── train_cnn_model_v2.py           # Relative to wrist
│   ├── cnn_models_comparison/          # Models with absolute coords
│   └── cnn_models_rel_wrist/           # Models relative to wrist
├── 1_2_c/                    # Part I.2.c) - Online classification
│   ├── online_classic_models.py        # Real-time with classic ML
│   └── online_cnn_model.py             # Real-time with CNN-1D
├── 1_3_a/                    # Part I.3.a) - CNN with temporal features
│   ├── train_cnn_time_model.py         # Absolute coordinates + deltas
│   └── train_cnn_time_model_v2.py      # Relative to wrist + deltas
├── 1_3_b/                    # Part I.3.b) - Online with temporal features
│   └── online_cnn_time_model.py        # Real-time with temporal CNN
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
- scikeras 0.12.0
- NumPy 1.24.3
- Matplotlib 3.8.2
- Seaborn 0.13.0
- TensorFlow 2.15.0

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

## Part I.2.a) - Classic ML Model Training (Offline)

### Overview

Trains 3 classic Machine Learning models (SVM, KNN, Random Forest) using hand landmarks extracted from video clips. Two versions available:
- **Version 1** (`train_classic_models.py`): Uses absolute landmark coordinates
- **Version 2** (`train_classic_models_v2.py`): Uses coordinates relative to wrist (more robust to hand position)

### Running the Training Scripts

**Absolute coordinates:**
```bash
cd 1_2_a
python train_classic_models.py
```

**Relative to wrist (recommended):**
```bash
cd 1_2_a
python train_classic_models_v2.py
```

### What the Scripts Do

1. **Loads video clips** from `Videos_Datasets/` folder
2. **Extracts hand landmarks** using MediaPipe Hands (21 landmarks per hand)
3. **Processes coordinates**:
   - Version 1: Uses absolute (x,y) or (x,y,z) coordinates
   - Version 2: Normalizes to wrist position (more translation-invariant)
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
7. **Saves all trained models** in `models_comparison/xy/` or `models_comparison/xyz/`

### Output Files

After running, you'll find in `1_2_a/models_comparison/xy/` (or `xyz/`):

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

Trains a 1D Convolutional Neural Network (CNN-1D) using temporal sequences of hand landmarks extracted from video clips. Unlike classic ML models that use averaged features, CNN-1D processes sequences of frames to capture temporal patterns. Two versions available:
- **Version 1** (`train_cnn_model.py`): Uses absolute landmark coordinates
- **Version 2** (`train_cnn_model_v2.py`): Uses coordinates relative to wrist

### Running the CNN Training Scripts

**Absolute coordinates:**
```bash
cd 1_2_b
python train_cnn_model.py
```

**Relative to wrist (recommended):**
```bash
cd 1_2_b
python train_cnn_model_v2.py
```

Both scripts test XY and XYZ coordinate configurations and save the best models.

### What the Scripts Do

1. **Loads video clips** from `Videos_Datasets/` folder
2. **Extracts hand landmark sequences** using MediaPipe Hands
3. **Processes temporal sequences**:
   - Each clip is processed as a sequence of 30 frames
   - 21 landmarks per frame with (x,y) or (x,y,z) coordinates
   - Final input shape: (30, 42) for XY or (30, 63) for XYZ
4. **Splits data**:
   - Training: clips 1-4 (first 4 clips per gesture per subject)
   - Testing: clips 5-6 (last 2 clips per gesture per subject)
5. **Builds CNN-1D architecture**:
   - 3 Conv1D layers with batch normalization and dropout
   - MaxPooling for temporal downsampling
   - Dense layers for classification
6. **Trains the model**:
   - Adam optimizer with learning rate scheduling (ReduceLROnPlateau)
   - Early stopping to prevent overfitting (patience=15)
   - Up to 100 epochs
7. **Evaluates performance**:
   - Accuracy scores on test set
   - Classification report (precision, recall, f1-score)
   - Confusion matrix visualization
   - Training history plots (loss & accuracy curves)
8. **Saves trained models** in `cnn_models_comparison/` or `cnn_models_rel_wrist/`

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

After running, you'll find in `1_2_b/cnn_models_comparison/xy/` or `xyz/`:

- `cnn1d_model.keras` - Trained CNN-1D model
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Loss and accuracy curves

For version 2 (relative to wrist), files are in `1_2_b/cnn_models_rel_wrist/xy/` or `xyz/`

### Configuration

Scripts automatically test both XY and XYZ configurations. Key parameters:

```python
# In the scripts:
DATASET_PATH = Path('../Videos_Datasets')  # Path to dataset
SEQ_LEN = 30  # Number of frames per sequence
TRAIN_CLIPS = [1, 2, 3, 4]  # Training clip indices
TEST_CLIPS = [5, 6]  # Testing clip indices
BATCH_SIZE = 16
EPOCHS = 100  # With early stopping
```

### Training Hyperparameters

```python
# Model compilation
optimizer = Adam(learning_rate=0.001)
loss = 'sparse_categorical_crossentropy'

# Callbacks
EarlyStopping(patience=15, restore_best_weights=True)
ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
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

### Usage

1. Select which CNN model to use:
   - **1** - CNN-1D: XY Coordinates (absolute)
   - **2** - CNN-1D: XYZ Coordinates (absolute)
   - **3** - CNN-1D V2: XY Coordinates (relative to wrist)
   - **4** - CNN-1D V2: XYZ Coordinates (relative to wrist)

2. The webcam will open and start classifying gestures in real-time

3. Press **'q'** to quit

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

## Part I.3.a) - CNN with Temporal Features Training (Offline)

### Overview

Extends CNN-1D models with temporal features (velocity/delta information) to capture hand motion dynamics. Uses GridSearchCV for hyperparameter optimization. Two versions available:
- **Version 1** (`train_cnn_time_model.py`): Absolute coordinates + temporal deltas
- **Version 2** (`train_cnn_time_model_v2.py`): Relative to wrist + temporal deltas

### Running the Training Scripts

**Absolute coordinates with temporal features:**
```bash
cd 1_3_a
python train_cnn_time_model.py
```

**Relative to wrist with temporal features:**
```bash
cd 1_3_a
python train_cnn_time_model_v2.py
```

### What the Scripts Do

1. **Extracts landmark sequences** with temporal information
2. **Computes frame-to-frame deltas** (velocity/motion features)
3. **Normalizes sequences** for better training stability
4. **Performs GridSearchCV** to find optimal hyperparameters:
   - Number of filters (32, 64, 128)
   - Dropout rates (0.2, 0.3, 0.4)
5. **Trains CNN model** with best parameters
6. **Evaluates on test set** and saves best model

### Key Differences from Part I.2.b

- **Temporal features**: Includes frame-to-frame differences (motion)
- **GridSearchCV**: Automated hyperparameter tuning
- **Normalized sequences**: Better handling of different hand sizes
- **Enhanced architecture**: Optimized for temporal dynamics

### Output Files

After running, you'll find models with optimized hyperparameters and performance metrics.

---

## Part I.3.b) - Online Classification with Temporal Features

### Overview

Real-time gesture classification using CNN models trained with temporal features (Part I.3.a).

### Running Online Classification

```bash
cd 1_3_b
python online_cnn_time_model.py
```

### Usage

1. Select which temporal CNN model to use
2. The webcam will open and start classifying with motion-aware predictions
3. Press **'q'** to quit

### Key Advantages

- **Motion-aware**: Captures hand movement dynamics
- **More robust**: Better handling of gesture transitions
- **Optimized**: Uses GridSearchCV-tuned hyperparameters

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
# For classic models (absolute coords):
cd 1_2_a
python train_classic_models.py

# For classic models (relative to wrist):
cd 1_2_a
python train_classic_models_v2.py

# For CNN model (absolute coords):
cd 1_2_b
python train_cnn_model.py

# For CNN model (relative to wrist):
cd 1_2_b
python train_cnn_model_v2.py
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

### Training Issues (Part I.3.a)

**ModuleNotFoundError: No module named 'tensorflow.keras.wrappers':**
```bash
pip install scikeras
```
The `tensorflow.keras.wrappers` module was removed in newer TensorFlow versions. Use `scikeras` instead.

**GridSearchCV taking too long:**
- Reduce the parameter grid size
- Decrease `cv` value (e.g., cv=2 instead of cv=3)
- Use fewer parameter combinations
- Consider training without GridSearch first

**Out of memory errors:**
- Reduce batch size
- Close other applications
- Use fewer filters in the parameter grid

---

## Performance Results

Based on typical training results:

### Classic Machine Learning Models (Part I.2.a)

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM** | ~98-99% | Best overall performance, recommended |
| **Random Forest** | ~96-97% | Good balance of speed and accuracy |
| **KNN** | ~95-96% | Fast inference, slightly lower accuracy |

### Deep Learning Models

| Model | Accuracy | Notes |
|-------|----------|-------|
| **CNN-1D (I.2.b)** | ~97-99% | Captures temporal patterns, robust to variations |
| **CNN-1D with Temporal Features (I.3.a)** | ~97-99%+ | Motion-aware, GridSearchCV optimized |

### Coordinate System Comparison

| Approach | Advantages | Disadvantages |
|----------|------------|---------------|
| **Absolute Coords** | Simple, direct | Sensitive to hand position/distance |
| **Relative to Wrist** | Translation-invariant, more robust | Slightly more complex processing |
| **XY Coordinates** | Faster processing, smaller models | Loses depth information |
| **XYZ Coordinates** | Full 3D information | Larger models, more data needed |

### Comparison

- **Classic ML (SVM)**: Best for instant single-frame predictions, lowest latency, excellent accuracy
- **CNN-1D**: Best for temporal pattern recognition, smoother predictions, requires sequence buffer
- **CNN with Temporal Features**: Best for motion-based gestures, captures dynamics, optimized hyperparameters
- **Relative to Wrist**: More robust to hand position variations, recommended for real-world deployment
- All models show excellent performance (>95% accuracy) for the 6-gesture classification task

### Recommended Use Cases

- **SVM (Absolute)**: Best overall classic ML model, use when immediate response is critical
- **SVM (Relative to Wrist)**: Most robust to position variations
- **CNN-1D (Relative to Wrist)**: Best for continuous gesture recognition with temporal context
- **CNN with Temporal Features**: Best for dynamic, motion-based gestures
- **Random Forest**: Good alternative with balanced performance
- **KNN**: Fastest inference, suitable for resource-constrained scenarios

