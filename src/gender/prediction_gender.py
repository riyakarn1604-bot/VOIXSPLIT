import librosa
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = Path("models/gender/gender_svm.pkl")
AUDIO_FILE = Path("data/audio/test.m4a")

N_MFCC = 40


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading gender model...")

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
scaler = model_data["scaler"]

print("Model loaded successfully.")


# ============================================================
# LOAD AUDIO
# ============================================================

print("\nLoading audio...")

audio, sample_rate = librosa.load(
    AUDIO_FILE,
    sr=16000,
    mono=True
)

print("Audio loaded.")
print("Duration:", round(len(audio) / sample_rate, 2), "seconds")


# ============================================================
# EXTRACT MFCC
# ============================================================

print("\nExtracting MFCC...")

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=N_MFCC
)

# Same operation used during training
mfcc_mean = np.mean(mfcc, axis=1)

# Convert to model input shape
X = mfcc_mean.reshape(1, -1)


# ============================================================
# SCALE FEATURES
# ============================================================

X_scaled = scaler.transform(X)


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(X_scaled)[0]


# ============================================================
# RESULT
# ============================================================

if prediction == 0:
    gender = "Male"
else:
    gender = "Female"


print("\n========== RESULT ==========")
print("Predicted gender:", gender)
print("============================")