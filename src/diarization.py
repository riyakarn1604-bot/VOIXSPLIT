
import numpy as np
import librosa
import webrtcvad

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

# Input audio
AUDIO_FILE = Path(
    "data/audio/test1.m4a"
)

# Audio settings
SAMPLE_RATE = 16000

# WebRTC VAD
FRAME_DURATION = 30
VAD_AGGRESSIVENESS = 2

# MFCC
N_MFCC = 20

# Speaker segmentation
SEGMENT_DURATION = 1.5
MIN_SPEECH_DURATION = 0.5

# Maximum number of speakers we are willing to detect
# Increase this for larger meetings/classes.
MAX_SPEAKERS = 10

# Minimum number of speakers to test
MIN_SPEAKERS = 1


# ============================================================
# HEADER
# ============================================================

print()
print("==========================================")
print("     VOICESENSE SPEAKER DIARIZATION")
print("==========================================")
print()


# ============================================================
# LOAD AUDIO
# ============================================================

print("Loading audio...")

try:

    audio, sr = librosa.load(
        AUDIO_FILE,
        sr=SAMPLE_RATE,
        mono=True
    )

except Exception as e:

    print()
    print("ERROR: Could not load audio.")
    print("File:", AUDIO_FILE)
    print("Reason:", e)

    raise SystemExit


duration = len(audio) / sr

print(
    f"Audio duration: {duration:.2f} seconds"
)

print(
    f"Sample rate: {sr} Hz"
)


# ============================================================
# NORMALIZE AUDIO
# ============================================================

audio = audio.astype(
    np.float32
)

max_value = np.max(
    np.abs(audio)
)

if max_value > 0:

    audio = audio / max_value


# ============================================================
# VOICE ACTIVITY DETECTION
# ============================================================

print()
print("Running Voice Activity Detection...")


vad = webrtcvad.Vad(
    VAD_AGGRESSIVENESS
)


frame_size = int(
    SAMPLE_RATE *
    FRAME_DURATION /
    1000
)


frames = []


for start in range(
    0,
    len(audio) - frame_size,
    frame_size
):

    frame = audio[
        start:
        start + frame_size
    ]


    pcm = (
        frame * 32767
    ).astype(
        np.int16
    ).tobytes()


    try:

        is_speech = vad.is_speech(
            pcm,
            SAMPLE_RATE
        )

    except Exception:

        is_speech = False


    frames.append(
        (
            start,
            is_speech
        )
    )


# ============================================================
# FIND SPEECH REGIONS
# ============================================================

speech_regions = []

speech_start = None


for sample_position, is_speech in frames:

    if is_speech:

        if speech_start is None:

            speech_start = sample_position

    else:

        if speech_start is not None:

            speech_end = sample_position

            region_duration = (
                speech_end -
                speech_start
            ) / SAMPLE_RATE


            if region_duration >= MIN_SPEECH_DURATION:

                speech_regions.append(
                    (
                        speech_start,
                        speech_end
                    )
                )


            speech_start = None


# Handle speech continuing until the end

if speech_start is not None:

    speech_end = len(audio)

    region_duration = (
        speech_end -
        speech_start
    ) / SAMPLE_RATE


    if region_duration >= MIN_SPEECH_DURATION:

        speech_regions.append(
            (
                speech_start,
                speech_end
            )
        )


print()
print(
    "Speech regions detected:",
    len(speech_regions)
)


if len(speech_regions) == 0:

    print()
    print(
        "No speech was detected."
    )

    raise SystemExit


# ============================================================
# DISPLAY SPEECH REGIONS
# ============================================================

print()
print("========== SPEECH REGIONS ==========")


for i, (
    start,
    end
) in enumerate(
    speech_regions
):

    start_time = (
        start / SAMPLE_RATE
    )

    end_time = (
        end / SAMPLE_RATE
    )


    print(
        f"Region {i + 1:03d}: "
        f"{start_time:.2f}s → "
        f"{end_time:.2f}s"
    )


# ============================================================
# CREATE FIXED-LENGTH SEGMENTS
# ============================================================

print()
print("Creating speaker segments...")


segments = []


segment_samples = int(
    SEGMENT_DURATION *
    SAMPLE_RATE
)


for region_start, region_end in speech_regions:

    position = region_start


    while (
        position +
        segment_samples
        <= region_end
    ):

        segment = audio[
            position:
            position +
            segment_samples
        ]


        if len(segment) == segment_samples:

            segments.append(
                (
                    position,
                    position +
                    segment_samples,
                    segment
                )
            )


        position += segment_samples


print(
    "Usable segments:",
    len(segments)
)


if len(segments) < 2:

    print()
    print(
        "Not enough speech segments "
        "for speaker detection."
    )

    raise SystemExit


# ============================================================
# EXTRACT MFCC FEATURES
# ============================================================

print()
print("Extracting MFCC speaker features...")


features = []


for i, (
    start,
    end,
    segment
) in enumerate(
    segments
):

    print(
        f"Processing segment "
        f"{i + 1}/{len(segments)}",
        end="\r"
    )


    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )


    # Mean of every MFCC coefficient
    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )


    # Standard deviation of every coefficient
    mfcc_std = np.std(
        mfcc,
        axis=1
    )


    # Combine them
    feature_vector = np.concatenate(
        [
            mfcc_mean,
            mfcc_std
        ]
    )


    features.append(
        feature_vector
    )


features = np.array(
    features
)


print()
print(
    "Feature matrix:",
    features.shape
)


# ============================================================
# SCALE FEATURES
# ============================================================

print()
print("Scaling features...")


scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    features
)


# ============================================================
# DETERMINE POSSIBLE NUMBER OF SPEAKERS
# ============================================================

# We cannot create more clusters than
# available speech segments.

maximum_possible = min(
    MAX_SPEAKERS,
    len(features)
)


print()
print(
    "Searching for number of speakers..."
)

print(
    f"Testing from "
    f"{MIN_SPEAKERS} to "
    f"{maximum_possible} speakers."
)


# ============================================================
# SINGLE SPEAKER CASE
# ============================================================

if maximum_possible == 1:

    best_k = 1

    best_score = None


else:

    best_k = 1
    best_score = -1


    # Test different numbers of speakers

    for k in range(
        2,
        maximum_possible + 1
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )


        labels = model.fit_predict(
            scaled_features
        )


        # Silhouette score requires
        # at least two clusters.

        if len(
            np.unique(labels)
        ) < 2:

            continue


        score = silhouette_score(
            scaled_features,
            labels
        )


        print(
            f"K = {k:2d} "
            f"| Silhouette Score = "
            f"{score:.4f}"
        )


        if score > best_score:

            best_score = score

            best_k = k


# ============================================================
# SELECTED NUMBER OF SPEAKERS
# ============================================================

print()
print(
    "=========================================="
)

print(
    "Estimated number of speakers:",
    best_k
)

if best_score is not None:

    print(
        "Best silhouette score:",
        f"{best_score:.4f}"
    )

print(
    "=========================================="
)


# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

print()
print("Clustering speaker segments...")


kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


labels = kmeans.fit_predict(
    scaled_features
)


# ============================================================
# ORGANIZE RESULTS BY SPEAKER
# ============================================================

speaker_segments = {}


for i, label in enumerate(labels):

    speaker_name = (
        f"SPEAKER_{label:02d}"
    )


    if speaker_name not in speaker_segments:

        speaker_segments[
            speaker_name
        ] = []


    start_time = (
        segments[i][0]
        / SAMPLE_RATE
    )


    end_time = (
        segments[i][1]
        / SAMPLE_RATE
    )


    speaker_segments[
        speaker_name
    ].append(
        (
            start_time,
            end_time
        )
    )


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print()
print("========== SPEAKER RESULTS ==========")


for speaker, times in sorted(
    speaker_segments.items()
):

    total_duration = sum(
        end - start
        for start, end in times
    )


    print()
    print(
        f"{speaker}"
    )

    print(
        f"Segments: {len(times)}"
    )

    print(
        f"Speech duration: "
        f"{total_duration:.2f}s"
    )


    for start, end in times:

        print(
            f"   {start:.2f}s → "
            f"{end:.2f}s"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print("           VOICESENSE RESULT")
print("==========================================")

print(
    f"Total speakers detected: {best_k}"
)

print(
    f"Total speech segments: "
    f"{len(segments)}"
)

print(
    f"Audio duration: "
    f"{duration:.2f}s"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/diarization_results.txt"
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
        "VoiceSense Speaker Diarization\n"
    )

    file.write(
        "================================\n\n"
    )


    file.write(
        f"Audio: {AUDIO_FILE}\n"
    )

    file.write(
        f"Audio duration: "
        f"{duration:.2f}s\n"
    )

    file.write(
        f"Detected speakers: "
        f"{best_k}\n\n"
    )


    for speaker, times in sorted(
        speaker_segments.items()
    ):

        total_duration = sum(
            end - start
            for start, end in times
        )


        file.write(
            f"{speaker}\n"
        )

        file.write(
            f"Total duration: "
            f"{total_duration:.2f}s\n"
        )


        for start, end in times:

            file.write(
                f"   {start:.2f}s → "
                f"{end:.2f}s\n"
            )


        file.write(
            "\n"
        )


print()
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)


print()
print("==========================================")
print("       DIARIZATION COMPLETE")
print("==========================================")

