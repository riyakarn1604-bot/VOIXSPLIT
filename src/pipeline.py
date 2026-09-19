
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import librosa
import joblib
import torch

from pyannote.audio import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = Path("data/audio/test5.m4a")

GENDER_MODEL_FILE = Path(
    "models/gender/gender_svm.pkl"
)

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# CHECK FILES
# ============================================================

if not AUDIO_FILE.exists():
    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_FILE}"
    )

if not GENDER_MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Gender model not found:\n{GENDER_MODEL_FILE}"
    )

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set.\n\n"
        "Set it using:\n"
        '$env:HF_TOKEN="hf_YOUR_TOKEN"'
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("==========================================")
print("          VOICESENSE PIPELINE")
print("==========================================")

print("Audio:", AUDIO_FILE)
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK FFMPEG
# ============================================================

print()
print("Checking FFmpeg...")

ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:

    raise RuntimeError(
        "FFmpeg was not found in PATH."
    )

print(
    "FFmpeg:",
    ffmpeg_path
)


# ============================================================
# CONVERT M4A → WAV
# ============================================================

print()
print("Converting input audio to WAV...")

temp_wav = tempfile.NamedTemporaryFile(
    suffix=".wav",
    delete=False
)

temp_wav_path = temp_wav.name

temp_wav.close()


ffmpeg_command = [
    ffmpeg_path,
    "-y",
    "-i",
    str(AUDIO_FILE),
    "-vn",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-sample_fmt",
    "s16",
    temp_wav_path
]


try:

    result = subprocess.run(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            "FFmpeg failed to convert the audio."
        )

except Exception:

    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

    raise


print(
    "WAV created successfully."
)


# ============================================================
# LOAD GENDER MODEL
# ============================================================

print()
print("Loading gender model...")

model_data = joblib.load(
    GENDER_MODEL_FILE
)


# Your training script saved:
# model
# scaler

gender_model = model_data["model"]
scaler = model_data["scaler"]

print(
    "Gender model loaded."
)


# ============================================================
# LOAD DIARIZATION MODEL
# ============================================================

print()
print("Loading speaker diarization model...")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

diarization_pipeline.to(device)

print(
    "Diarization model loaded."
)


# ============================================================
# RUN DIARIZATION
# ============================================================

print()
print("Running speaker diarization...")

output = diarization_pipeline(
    str(AUDIO_FILE)
)


# ============================================================
# HANDLE PYANNOTE OUTPUT
# ============================================================

if hasattr(
    output,
    "speaker_diarization"
):

    diarization = (
        output.speaker_diarization
    )

else:

    diarization = output


# ============================================================
# COLLECT SPEAKER SEGMENTS
# ============================================================

speaker_segments = {}


for turn, _, speaker in diarization.itertracks(
    yield_label=True
):

    start = float(turn.start)
    end = float(turn.end)

    if speaker not in speaker_segments:

        speaker_segments[speaker] = []

    speaker_segments[speaker].append(
        (start, end)
    )


# ============================================================
# DISPLAY SPEAKERS
# ============================================================

print()
print(
    "========== SPEAKERS DETECTED =========="
)


for speaker, segments in speaker_segments.items():

    duration = sum(
        end - start
        for start, end in segments
    )

    print(
        f"{speaker}: "
        f"{len(segments)} segments, "
        f"{duration:.2f}s"
    )


number_of_speakers = len(
    speaker_segments
)

print()
print(
    "Number of speakers:",
    number_of_speakers
)


# ============================================================
# LOAD CONVERTED WAV
# ============================================================

print()
print(
    "Loading converted WAV for gender analysis..."
)

audio, sample_rate = librosa.load(
    temp_wav_path,
    sr=16000,
    mono=True
)

print(
    f"Audio loaded: "
    f"{len(audio) / sample_rate:.2f}s"
)


# ============================================================
# GENDER ANALYSIS
# ============================================================

print()
print(
    "========== GENDER ANALYSIS =========="
)


results = {}


for speaker, segments in speaker_segments.items():

    print()
    print(
        f"Processing {speaker}..."
    )


    # --------------------------------------------------------
    # COLLECT SPEAKER AUDIO
    # --------------------------------------------------------

    speaker_parts = []


    for start, end in segments:

        start_sample = int(
            start * sample_rate
        )

        end_sample = int(
            end * sample_rate
        )

        segment = audio[
            start_sample:end_sample
        ]


        if len(segment) > 0:

            speaker_parts.append(
                segment
            )


    # --------------------------------------------------------
    # CHECK SEGMENTS
    # --------------------------------------------------------

    if not speaker_parts:

        print(
            "No usable audio found."
        )

        continue


    # --------------------------------------------------------
    # COMBINE SPEAKER SEGMENTS
    # --------------------------------------------------------

    speaker_audio = np.concatenate(
        speaker_parts
    )


    duration = (
        len(speaker_audio)
        / sample_rate
    )


    print(
        f"Usable speech: "
        f"{duration:.2f}s"
    )


    # --------------------------------------------------------
    # MINIMUM AUDIO
    # --------------------------------------------------------

    if duration < 1.0:

        print(
            "Not enough speech "
            "for reliable gender prediction."
        )

        continue


    # --------------------------------------------------------
    # MFCC EXTRACTION
    # --------------------------------------------------------

    print(
        "Extracting 40 MFCC features..."
    )


    mfcc = librosa.feature.mfcc(
        y=speaker_audio,
        sr=sample_rate,
        n_mfcc=40
    )


    print(
        "MFCC matrix:",
        mfcc.shape
    )


    # --------------------------------------------------------
    # SAME FEATURE PROCESSING AS TRAINING
    # --------------------------------------------------------

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )


    features = mfcc_mean.reshape(
        1,
        -1
    )


    print(
        "Feature vector:",
        features.shape
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    features_scaled = scaler.transform(
        features
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = gender_model.predict(
        features_scaled
    )[0]


    # --------------------------------------------------------
    # HANDLE MODEL LABEL
    # --------------------------------------------------------

    if isinstance(
        prediction,
        str
    ):

        prediction_lower = (
            prediction.lower()
        )

        if "female" in prediction_lower:

            gender = "Female"

        elif "male" in prediction_lower:

            gender = "Male"

        else:

            gender = prediction

    else:

        # Current model:
        # 0 = Male
        # 1 = Female

        if int(prediction) == 0:

            gender = "Male"

        else:

            gender = "Female"


    # --------------------------------------------------------
    # CONFIDENCE IF AVAILABLE
    # --------------------------------------------------------

    confidence = None


    if hasattr(
        gender_model,
        "predict_proba"
    ):

        probabilities = (
            gender_model.predict_proba(
                features_scaled
            )[0]
        )

        confidence = float(
            np.max(probabilities)
        ) * 100


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results[speaker] = {
        "gender": gender,
        "duration": duration,
        "confidence": confidence
    }


    if confidence is not None:

        print(
            f"Prediction: {gender}"
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
        )

    else:

        print(
            f"Prediction: {gender}"
        )


# ============================================================
# DELETE TEMP WAV
# ============================================================

try:

    if os.path.exists(
        temp_wav_path
    ):

        os.remove(
            temp_wav_path
        )

        print()
        print(
            "Temporary WAV deleted."
        )

except Exception as e:

    print(
        "Could not delete temporary WAV:",
        e
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print(
    "=========================================="
)

print(
    "          VOICESENSE RESULT"
)

print(
    "=========================================="
)

print(
    "Total speakers detected:",
    number_of_speakers
)

print(
    "Speakers classified:",
    len(results)
)

print()


for speaker, info in results.items():

    gender = info["gender"]
    duration = info["duration"]
    confidence = info["confidence"]


    if confidence is not None:

        print(
            f"{speaker} → "
            f"{gender} "
            f"({confidence:.2f}% confidence) "
            f"[{duration:.2f}s]"
        )

    else:

        print(
            f"{speaker} → "
            f"{gender} "
            f"[{duration:.2f}s]"
        )


print()
print(
    "=========================================="
)

print(
    "          PIPELINE COMPLETE"
)

print(
    "=========================================="
)

