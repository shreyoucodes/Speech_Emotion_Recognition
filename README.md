
# Speech Emotion Recognition Project

## Overview
This project implements a Speech Emotion Recognition system using the RAVDESS dataset. It extracts audio features (MFCC, chromagram, mel spectrogram, spectral contrast, tonnetz), balances the dataset, trains a neural network (`MLPClassifier`) to classify emotions (e.g., happy, sad, angry), and visualizes results with feature plots and confusion matrices.

---

## Prerequisites

- **Python**: Version 3.11.x or higher (3.11.5 used in the notebook)
- **Virtual Environment**: Recommended for isolating dependencies
- **Dataset**: RAVDESS audio dataset ([available here](https://zenodo.org/record/1188976))
- **Dependencies**: Install required libraries using:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Create a Virtual Environment:

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### 2. Install Dependencies:

```bash
pip install ipykernel librosa numpy matplotlib scikit-learn joblib pandas seaborn
```

Alternatively, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Download RAVDESS Dataset:
- Place the dataset in a directory (e.g., `C:\ML 2024\speech-emotion-recognition-ravdess-data`)
- Update the dataset path in the `load_data` function if necessary

---

## Usage

### Run the Jupyter Notebook:
Open `Speech_Emotion_Recognition.ipynb` in Jupyter Notebook or JupyterLab.

Execute cells sequentially to:
- Install libraries
- Load and process audio files
- Extract features and balance the dataset
- Train the `MLPClassifier`
- Visualize features and model performance

---

## Key Files

- **`orange_problem_annotated.ipynb`**: Main notebook with implementation
- **`requirements.txt`**: List of required Python libraries

---

## Project Structure

- **Feature Extraction**: Extracts MFCC (13), chromagram (12), mel spectrogram (100), spectral contrast (7), and tonnetz (6) using Librosa
- **Dataset Balancing**: Ensures equal samples per emotion to avoid bias
- **Model Training**: Uses `MLPClassifier` from `scikit-learn` for emotion classification
- **Visualization**: Plots feature distributions and confusion matrix for model evaluation

---

## Dataset

**RAVDESS**: Contains `.wav` files labeled with emotions:
- Neutral, calm, happy, sad, angry, fearful, disgust, surprised

**File Naming**:
- Emotion codes (e.g., `'03'` for happy) are extracted from file names

**Path**:
- Update the `glob.glob` path in `load_data` to match your dataset location

---

## Key Features

- **Audio Processing**: Librosa for robust feature extraction
- **Balanced Dataset**: Equal samples per emotion for fair training
- **Neural Network**: `MLPClassifier` for multi-class emotion classification
- **Evaluation**: Confusion matrix to assess model performance

---

## Results

- **Feature Visualizations**: Plots show average feature values (MFCC, chromagram, etc.) across the dataset
- **Model Performance**: Confusion matrix (initially for KNN, assumed similar for MLP) shows prediction accuracy
- **Output**: Trained model can predict emotions from new audio files (after preprocessing)

---

## Notes

- **Missing Code**: Notebook references a KNN model and `LabelEncoder` but lacks training code for KNN. Focus is on `MLPClassifier`.
- **Customization**: Adjust feature extraction (e.g., number of MFCCs) or model parameters (e.g., hidden layers)
- **Extensions**: Add more features, try CNNs, or integrate real-time audio processing

---

## Requirements

See `requirements.txt` or install manually:

- `ipykernel`
- `librosa`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `pandas`
- `seaborn`

---

## Acknowledgments

- **RAVDESS Dataset**: For providing a standardized dataset for emotion recognition
- **Librosa**: For powerful audio processing tools
- **Scikit-learn**: For accessible machine learning utilities
