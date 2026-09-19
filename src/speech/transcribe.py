import whisper
import torch
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = Path("data/audio/test1.m4a")


# ============================================================
# CHECK AUDIO
# ============================================================

if not AUDIO_FILE.exists():

    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_FILE}"
    )


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


print("==========================================")
print("        VOICESENSE SPEECH-TO-TEXT")
print("==========================================")

print("Audio:", AUDIO_FILE)
print("Device:", device)


# ============================================================
# LOAD WHISPER
# ============================================================

print("\nLoading Whisper model...")

# Start with base.
# It is much faster than large models.

model = whisper.load_model(
    "base",
    device=device
)

print("Whisper loaded.")


# ============================================================
# TRANSCRIBE
# ============================================================

print("\nTranscribing audio...")

result = model.transcribe(
    str(AUDIO_FILE),
    fp16=(device == "cuda")
)


# ============================================================
# RESULT
# ============================================================

text = result["text"].strip()

language = result.get(
    "language",
    "unknown"
)


print("\n========== TRANSCRIPTION ==========")

print(text)

print("\nDetected language:", language)

print("===================================")


# ============================================================
# SAVE TEXT
# ============================================================

OUTPUT_FILE = Path(
    "data/transcription.txt"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE.write_text(
    text,
    encoding="utf-8"
)


print(
    f"\nTranscription saved to: {OUTPUT_FILE}"
)