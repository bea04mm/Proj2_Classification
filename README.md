# Hand Gesture Classification Project - Part I

Static hand gesture classification using MediaPipe Hands and classic Machine Learning models.

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
├── 1_2_a/                    # Part I.2.a) - Offline training
│   ├── train_classic_models.py
│   └── models/              # Trained models (generated)
├── 1_2_c/                    # Part I.2.c) - Online classification
│   └── online_classic_models.py
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
- scikit-learn 1.3.2
- NumPy 1.24.3
- Pandas 2.1.4
- Matplotlib 3.8.2
- Seaborn 0.13.0
- Joblib 1.3.2

See [requirements.txt](requirements.txt) for complete list.

## Recognized Gestures

The project recognizes 6 static hand gestures:

1. 🖐️ **open_palm** - Open palm hand (green)
2. ✊ **closed_hand** - Closed hand/fist (red)
3. ✌️ **v_shape** - V-shape hand/peace sign (magenta)
4. 👉 **hand_right** - Hand pointing right (cyan)
5. 👈 **hand_left** - Hand pointing left (yellow)
6. 👍 **thumbs_up** - Thumbs up (orange)

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

## Part I.2.c) - Online Classification (Real-time)

### Overview

Real-time hand gesture classification using the trained models from Part I.2.a) with webcam input.

### Prerequisites

**Trained models** from Part I.2.a) must exist in `1_2_a/models/`:
- `scaler.pkl`
- `svm_model.pkl`
- `knn_model.pkl`
- `randomforest_model.pkl`

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

### Tips for Best Results

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
cd 1_2_a
python train_classic_models.py
```

**No hand detected:**
- Check lighting and hand position
- Ensure camera is working
- Keep hand visible in frame

**Wrong predictions:**
- Try the SVM model for better accuracy
- Ensure clear, distinct gestures
- Improve lighting conditions

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

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM** | ~98.75% | Best overall performance, recommended |
| **Random Forest** | ~96.00% | Good balance of speed and accuracy |
| **KNN** | ~95.90% | Fast inference, slightly lower accuracy |

All models show excellent performance (>95% accuracy) for the 6-gesture classification task.

