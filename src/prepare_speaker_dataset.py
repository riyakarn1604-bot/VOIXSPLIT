from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import numpy as np
import random


# ============================================================
# VOXSPLIT V6 - SPEAKER DATASET PREPARATION
# ============================================================

OUTPUT_DIR = Path("data/speaker_dataset")

NUM_SPEAKERS = 30
SAMPLES_PER_SPEAKER = 20

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42

random.seed(SEED)


print()
print("==========================================")
print("      VOXSPLIT V6 SPEAKER DATASET")
print("==========================================")

print()
print("Loading SLUE VoxCeleb dataset...")

dataset = load_dataset(
    "asapp/slue",
    "voxceleb"
)

print()
print(dataset)


# ============================================================
# USE TRAIN SPLIT
# ============================================================

data = dataset["train"]

print()
print("Training samples:", len(data))
print("Columns:", data.column_names)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "audio",
    "speaker_id",
    "start_second",
    "end_second"
]

for column in required_columns:

    if column not in data.column_names:

        raise RuntimeError(
            f"Required column missing: {column}"
        )


print()
print("Speaker ID column found.")
print("Audio column found.")


# ============================================================
# GROUP SAMPLES BY SPEAKER
# ============================================================

speaker_samples = {}

for index in range(len(data)):

    speaker = str(
        data[index]["speaker_id"]
    )

    if speaker not in speaker_samples:

        speaker_samples[speaker] = []

    speaker_samples[speaker].append(index)


print()
print(
    "Unique speakers:",
    len(speaker_samples)
)


# ============================================================
# FIND ELIGIBLE SPEAKERS
# ============================================================

eligible_speakers = [

    speaker

    for speaker, indices
    in speaker_samples.items()

    if len(indices) >= SAMPLES_PER_SPEAKER
]


print(
    "Speakers with at least",
    SAMPLES_PER_SPEAKER,
    "samples:",
    len(eligible_speakers)
)


if len(eligible_speakers) < NUM_SPEAKERS:

    raise RuntimeError(
        "Not enough speakers available."
    )


# ============================================================
# SELECT SPEAKERS
# ============================================================

selected_speakers = random.sample(
    eligible_speakers,
    NUM_SPEAKERS
)


print()
print("Selected speakers:")

for i, speaker in enumerate(
    selected_speakers
):

    print(
        f"{i:02d} -> {speaker}"
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    for speaker_index in range(
        NUM_SPEAKERS
    ):

        directory = (

            OUTPUT_DIR
            / split
            / f"speaker_{speaker_index:03d}"

        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS DATA
# ============================================================

print()
print("Creating WAV files...")


for speaker_index, speaker in enumerate(
    selected_speakers
):

    indices = speaker_samples[
        speaker
    ].copy()

    random.shuffle(indices)

    # exactly 20 recordings
    indices = indices[
        :SAMPLES_PER_SPEAKER
    ]


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_count = int(
        len(indices) * TRAIN_RATIO
    )

    val_count = int(
        len(indices) * VAL_RATIO
    )


    train_indices = indices[
        :train_count
    ]

    val_indices = indices[
        train_count:
        train_count + val_count
    ]

    test_indices = indices[
        train_count + val_count:
    ]


    splits = {

        "train": train_indices,

        "val": val_indices,

        "test": test_indices
    }


    print(
        f"Speaker {speaker_index:03d}"
        f" | original ID: {speaker}"
        f" | total: {len(indices)}"
    )


    # ========================================================
    # SAVE AUDIO
    # ========================================================

    for split, sample_indices in splits.items():

        output_dir = (

            OUTPUT_DIR
            / split
            / f"speaker_{speaker_index:03d}"

        )


        for local_index, dataset_index in enumerate(
            sample_indices
        ):

            item = data[
                dataset_index
            ]


            # ----------------------------------------------
            # AUDIO
            # ----------------------------------------------

            audio = item["audio"]

            samples = np.asarray(
                audio["array"],
                dtype=np.float32
            )

            sample_rate = audio[
                "sampling_rate"
            ]


            # ----------------------------------------------
            # CROP TO ANNOTATED SEGMENT
            # ----------------------------------------------

            start_second = float(
                item["start_second"]
            )

            end_second = float(
                item["end_second"]
            )


            start_sample = int(
                start_second * sample_rate
            )

            end_sample = int(
                end_second * sample_rate
            )


            samples = samples[
                start_sample:end_sample
            ]


            # ----------------------------------------------
            # MONO
            # ----------------------------------------------

            if samples.ndim > 1:

                samples = np.mean(
                    samples,
                    axis=0
                )


            # ----------------------------------------------
            # NORMALIZE
            # ----------------------------------------------

            max_value = np.max(
                np.abs(samples)
            )

            if max_value > 0:

                samples = (
                    samples / max_value
                )


            # ----------------------------------------------
            # SAVE
            # ----------------------------------------------

            output_file = (

                output_dir
                / f"{split}_{local_index:03d}.wav"

            )


            sf.write(
                output_file,
                samples,
                sample_rate
            )


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print("       DATASET PREPARATION COMPLETE")
print("==========================================")


for split in [
    "train",
    "val",
    "test"
]:

    count = sum(

        1

        for _ in (

            OUTPUT_DIR
            / split

        ).rglob("*.wav")

    )

    print(
        f"{split.upper():5s}: {count} files"
    )


print()
print(
    "Dataset:",
    OUTPUT_DIR
)

print()
print("==========================================")