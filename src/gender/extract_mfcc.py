print("STARTING MFCC EXTRACTION...")

import librosa
import numpy as np
from pathlib import Path
import joblib

# Dataset folders
DATA_DIR = Path("data/gender")
MALE_DIR = DATA_DIR / "male"
FEMALE_DIR = DATA_DIR / "female"

# File where extracted features will be saved
OUTPUT_FILE = DATA_DIR / "mfcc_features.pkl"

# Number of MFCC features
N_MFCC = 40


def extract_mfcc(file_path):
    """Extract 40 MFCC features from one audio file."""

    # Load audio at 16 kHz
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    # Take mean so every audio gets exactly 40 features
    return np.mean(mfcc, axis=1)


# Lists to store our dataset
features = []
labels = []

print("\n========== MFCC EXTRACTION ==========")

# ---------------- MALE AUDIO ----------------

male_files = list(MALE_DIR.glob("*.wav"))
print("Male files found:", len(male_files))

for i, file_path in enumerate(male_files):

    try:
        mfcc = extract_mfcc(file_path)

        features.append(mfcc)
        labels.append(0)  # 0 = Male

        print(f"Male [{i + 1}/{len(male_files)}]: {file_path.name}")

    except Exception as e:
        print(f"ERROR: {file_path.name} -> {e}")


# ---------------- FEMALE AUDIO ----------------

female_files = list(FEMALE_DIR.glob("*.wav"))
print("\nFemale files found:", len(female_files))

for i, file_path in enumerate(female_files):

    try:
        mfcc = extract_mfcc(file_path)

        features.append(mfcc)
        labels.append(1)  # 1 = Female

        print(f"Female [{i + 1}/{len(female_files)}]: {file_path.name}")

    except Exception as e:
        print(f"ERROR: {file_path.name} -> {e}")


# Convert lists into NumPy arrays
X = np.array(features)
y = np.array(labels)

print("\n========== RESULTS ==========")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Male samples:", np.sum(y == 0))
print("Female samples:", np.sum(y == 1))


# Save extracted features
data = {
    "X": X,
    "y": y
}

joblib.dump(data, OUTPUT_FILE)

print("\nMFCC features saved successfully!")
print("Location:", OUTPUT_FILE)
print("=============================")