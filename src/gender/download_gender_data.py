from datasets import load_dataset
from pathlib import Path
import soundfile as sf

# ============================================================
# SETTINGS
# ============================================================

MAX_PER_GENDER = 500

BASE_DIR = Path("data/gender")
MALE_DIR = BASE_DIR / "male"
FEMALE_DIR = BASE_DIR / "female"

# Create folders
MALE_DIR.mkdir(parents=True, exist_ok=True)
FEMALE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading gender dataset...")

dataset = load_dataset(
    "saeedzou/common-voice-17-en-age-gender-accent-sampled",
    split="train"
)

print("Dataset loaded.")
print("Total samples:", len(dataset))

# ============================================================
# COUNTERS
# ============================================================

male_count = 0
female_count = 0

# ============================================================
# PROCESS DATA
# ============================================================

for sample in dataset:

    gender = sample["gender"]

    # Dataset uses:
    # male_masculine
    # female_feminine

    if gender == "male_masculine":

        # Stop if we already have enough male samples
        if male_count >= MAX_PER_GENDER:
            continue

        audio = sample["audio"]

        output = MALE_DIR / f"male_{male_count:03d}.wav"

        sf.write(
            output,
            audio["array"],
            audio["sampling_rate"]
        )

        male_count += 1

        print(f"Saved male: {output}")

    elif gender == "female_feminine":

        # Stop if we already have enough female samples
        if female_count >= MAX_PER_GENDER:
            continue

        audio = sample["audio"]

        output = FEMALE_DIR / f"female_{female_count:03d}.wav"

        sf.write(
            output,
            audio["array"],
            audio["sampling_rate"]
        )

        female_count += 1

        print(f"Saved female: {output}")

    # ========================================================
    # STOP AFTER GETTING 100 OF EACH
    # ========================================================

    if (
        male_count >= MAX_PER_GENDER
        and female_count >= MAX_PER_GENDER
    ):
        break

# ============================================================
# FINAL RESULT
# ============================================================

print("\n========== DONE ==========")
print("Male samples:", male_count)
print("Female samples:", female_count)
print("Total samples:", male_count + female_count)
print("==========================")