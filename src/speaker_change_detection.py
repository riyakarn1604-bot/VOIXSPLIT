import numpy as np
import librosa
import webrtcvad

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = Path("data/audio/test1.m4a")

SAMPLE_RATE = 16000

FRAME_DURATION = 30
VAD_AGGRESSIVENESS = 2

# Small windows used to observe voice changes
WINDOW_DURATION = 1.0

# Move forward by this amount each time
HOP_DURATION = 0.5

N_MFCC = 20

MIN_SPEECH_DURATION = 0.5

# Higher threshold = fewer speaker changes
CHANGE_THRESHOLD = 2.5


# ============================================================
# HEADER
# ============================================================

print()
print("==========================================")
print("    VOXSPLIT SPEAKER CHANGE DETECTION")
print("==========================================")


# ============================================================
# LOAD AUDIO
# ============================================================

print()
print("Loading audio...")

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

audio = audio.astype(np.float32)

print(f"Audio duration: {len(audio) / sr:.2f}s")
print(f"Sample rate: {sr}")


# ============================================================
# NORMALIZE AUDIO
# ============================================================

maximum = np.max(np.abs(audio))

if maximum > 0:
    audio = audio / maximum


# ============================================================
# VAD
# ============================================================

print()
print("Running Voice Activity Detection...")

vad = webrtcvad.Vad(
    VAD_AGGRESSIVENESS
)

frame_size = int(
    SAMPLE_RATE * FRAME_DURATION / 1000
)

frames = []


for start in range(
    0,
    len(audio) - frame_size,
    frame_size
):

    frame = audio[
        start:start + frame_size
    ]

    pcm = (
        frame * 32767
    ).astype(np.int16).tobytes()

    is_speech = vad.is_speech(
        pcm,
        SAMPLE_RATE
    )

    frames.append(
        (start, is_speech)
    )


# ============================================================
# FIND SPEECH REGIONS
# ============================================================

speech_regions = []

speech_start = None


for position, is_speech in frames:

    if is_speech:

        if speech_start is None:
            speech_start = position

    else:

        if speech_start is not None:

            speech_end = position

            duration = (
                speech_end - speech_start
            ) / SAMPLE_RATE

            if duration >= MIN_SPEECH_DURATION:

                speech_regions.append(
                    (
                        speech_start,
                        speech_end
                    )
                )

            speech_start = None


# Handle speech until end of file

if speech_start is not None:

    speech_end = len(audio)

    duration = (
        speech_end - speech_start
    ) / SAMPLE_RATE

    if duration >= MIN_SPEECH_DURATION:

        speech_regions.append(
            (
                speech_start,
                speech_end
            )
        )


print(f"Speech regions detected: {len(speech_regions)}")


# ============================================================
# CREATE OVERLAPPING WINDOWS
# ============================================================

print()
print("Creating overlapping analysis windows...")

window_samples = int(
    WINDOW_DURATION * SAMPLE_RATE
)

hop_samples = int(
    HOP_DURATION * SAMPLE_RATE
)


windows = []


for region_start, region_end in speech_regions:

    position = region_start

    while position + window_samples <= region_end:

        window = audio[
            position:
            position + window_samples
        ]

        windows.append(
            {
                "start": position,
                "end": position + window_samples,
                "audio": window
            }
        )

        position += hop_samples


print(f"Total analysis windows: {len(windows)}")


# ============================================================
# MFCC FEATURE EXTRACTION
# ============================================================

print()
print("Extracting MFCC features...")


features = []


for index, window in enumerate(windows):

    mfcc = librosa.feature.mfcc(
        y=window["audio"],
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    # 20 MFCC means
    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    # 20 MFCC standard deviations
    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    feature_vector = np.concatenate(
        [
            mfcc_mean,
            mfcc_std
        ]
    )

    features.append(
        feature_vector
    )

    print(
        f"\rProcessing window "
        f"{index + 1}/{len(windows)}",
        end=""
    )


print()


features = np.array(features)


# ============================================================
# CHECK DATA
# ============================================================

if len(features) < 2:

    print()
    print("Not enough speech windows for analysis.")

    raise SystemExit


print()
print("Feature matrix shape:", features.shape)


# ============================================================
# SCALE FEATURES
# ============================================================

print()
print("Scaling features...")

scaler = StandardScaler()

features_scaled = scaler.fit_transform(
    features
)


# ============================================================
# COMPARE CONSECUTIVE WINDOWS
# ============================================================

print()
print("Comparing consecutive windows...")


change_scores = []


for i in range(
    len(features_scaled) - 1
):

    current_feature = features_scaled[i]

    next_feature = features_scaled[i + 1]

    # Euclidean distance between two voice representations
    distance = np.linalg.norm(
        current_feature - next_feature
    )

    change_scores.append(distance)


change_scores = np.array(change_scores)


print()
print(
    f"Average change score: "
    f"{np.mean(change_scores):.4f}"
)

print(
    f"Maximum change score: "
    f"{np.max(change_scores):.4f}"
)


# ============================================================
# AUTOMATIC CHANGE THRESHOLD
# ============================================================

# Instead of blindly trusting a fixed number,
# calculate a threshold from this audio itself.

mean_change = np.mean(change_scores)

std_change = np.std(change_scores)


adaptive_threshold = (
    mean_change +
    1.0 * std_change
)


print()
print("Change statistics:")

print(f"Mean: {mean_change:.4f}")

print(f"Standard deviation: {std_change:.4f}")

print(
    f"Adaptive threshold: "
    f"{adaptive_threshold:.4f}"
)


# ============================================================
# DETECT SPEAKER CHANGES
# ============================================================

print()
print("==========================================")
print("      DETECTED SPEAKER CHANGES")
print("==========================================")


speaker_changes = []


for i, score in enumerate(change_scores):

    if score >= adaptive_threshold:

        change_time = (
            windows[i + 1]["start"]
            / SAMPLE_RATE
        )

        speaker_changes.append(
            {
                "time": change_time,
                "score": score
            }
        )

        print(
            f"Possible change at "
            f"{change_time:.2f}s "
            f"| Score = {score:.4f}"
        )


if len(speaker_changes) == 0:

    print(
        "No strong speaker changes detected."
    )


# ============================================================
# CREATE ADAPTIVE SEGMENTS
# ============================================================

print()
print("==========================================")
print("       ADAPTIVE SPEAKER SEGMENTS")
print("==========================================")


adaptive_segments = []


for region_start, region_end in speech_regions:

    region_changes = []

    for change in speaker_changes:

        change_sample = int(
            change["time"] *
            SAMPLE_RATE
        )

        if (
            region_start <
            change_sample <
            region_end
        ):

            region_changes.append(
                change_sample
            )


    boundaries = (
        [region_start]
        +
        region_changes
        +
        [region_end]
    )


    for i in range(
        len(boundaries) - 1
    ):

        start = boundaries[i]

        end = boundaries[i + 1]

        duration = (
            end - start
        ) / SAMPLE_RATE


        if duration >= MIN_SPEECH_DURATION:

            adaptive_segments.append(
                (
                    start,
                    end
                )
            )


for index, (
    start,
    end
) in enumerate(adaptive_segments):

    print()

    print(
        f"Segment {index + 1}"
    )

    print(
        f"Start: "
        f"{start / SAMPLE_RATE:.2f}s"
    )

    print(
        f"End: "
        f"{end / SAMPLE_RATE:.2f}s"
    )

    print(
        f"Duration: "
        f"{(end - start) / SAMPLE_RATE:.2f}s"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/speaker_change_results.txt"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "VoxSplit Speaker Change Detection\n"
    )

    file.write(
        "==================================\n\n"
    )

    file.write(
        f"Audio duration: "
        f"{len(audio) / SAMPLE_RATE:.2f}s\n"
    )

    file.write(
        f"Speech regions: "
        f"{len(speech_regions)}\n"
    )

    file.write(
        f"Analysis windows: "
        f"{len(windows)}\n"
    )

    file.write(
        f"Adaptive threshold: "
        f"{adaptive_threshold:.4f}\n\n"
    )


    file.write(
        "DETECTED CHANGES\n"
    )

    file.write(
        "----------------\n"
    )


    for change in speaker_changes:

        file.write(
            f"{change['time']:.2f}s "
            f"| Score = "
            f"{change['score']:.4f}\n"
        )


    file.write(
        "\nADAPTIVE SEGMENTS\n"
    )

    file.write(
        "-----------------\n"
    )


    for index, (
        start,
        end
    ) in enumerate(adaptive_segments):

        file.write(
            f"Segment {index + 1}: "
            f"{start / SAMPLE_RATE:.2f}s"
            f" → "
            f"{end / SAMPLE_RATE:.2f}s\n"
        )


print()
print()
print("Results saved to:")
print(OUTPUT_FILE)


print()
print("==========================================")
print("    SPEAKER CHANGE DETECTION COMPLETE")
print("==========================================")