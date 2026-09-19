# VOIXSPLIT (VoiceSense)

VOIXSPLIT is an advanced audio processing, speaker diarization, speaker recognition, and gender classification toolkit.

## Features

- **Speaker Diarization**: Multi-speaker segment detection, clustering (spectral, agglomerative), and change detection.
- **Deep Speaker Embeddings**: Custom CNN architectures (v6, v7, v8) for learning speaker representations from MFCC and spectrogram features.
- **Gender Classification**: MFCC feature extraction with SVM classification trained on Common Voice data.
- **Speech Transcription**: Whisper-based transcription pipeline aligned with speaker turns.
- **Evaluation & Visualization**: Dimensionality reduction (PCA), embedding evaluation, and MFCC visualization scripts.

## Project Structure

```
VOIXSPLIT/
├── data/
│   └── audio/              # Sample test audio files (.m4a)
├── models/
│   └── gender/             # Pretrained gender classification model (.pkl)
├── results/                # Trained speaker embedding checkpoints (.pth) and logs
├── src/
│   ├── audio/              # Audio analysis utilities
│   ├── diarization/        # Diarization pipelines (pyannote.audio & custom)
│   ├── gender/             # Gender classification training & inference
│   ├── speech/             # Whisper transcription integration
│   ├── clustering_experiment.py
│   ├── diarization_v2.py - v5.py
│   ├── evaluate_speaker_embedding_v6.py
│   ├── feature_experiment.py
│   ├── pipeline.py
│   ├── prepare_speaker_dataset.py
│   ├── speaker_change_detection.py
│   ├── speaker_embedding_v6.py / v8.py
│   ├── train_speaker_embedding_v6.py / v7.py / v8_train.py
│   └── visualize_mfcc.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/riyakarn1604-bot/VOIXSPLIT.git
cd VOIXSPLIT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Speaker Diarization
Run speaker diarization on sample audio:
```bash
python src/diarization/speaker_diarization.py
```

### 2. Full End-to-End Pipeline
Run the complete pipeline (audio analysis, diarization, transcription):
```bash
python src/pipeline.py
```

### 3. Gender Detection
Train or predict gender from voice:
```bash
python src/gender/prediction_gender.py
```

### 4. Speaker Embeddings
Train or evaluate speaker embedding models:
```bash
python src/train_speaker_embedding_v7.py
python src/evaluate_speaker_embedding_v6.py
```
