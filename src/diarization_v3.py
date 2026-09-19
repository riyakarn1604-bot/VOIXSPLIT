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

AUDIO_FILE = Path("data/audio/test1.m4a")

SAMPLE_RATE = 16000

FRAME_DURATION = 30
VAD_AGGRESSIVENESS = 2

WINDOW_DURATION = 1.0
HOP_DURATION = 0.5

N_MFCC = 20

MIN_SPEECH_DURATION = 0.5

MIN_SEGMENT_DURATION = 1.0

MAX_SPEAKERS = 4


# ============================================================
# HEADER
# ============================================================

print()
print("==========================================")
print("         VOXSPLIT DIARIZATION V3")
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

duration = len(audio) / SAMPLE_RATE

print(f"Audio duration: {duration:.2f}s")
print(f"Sample rate: {sr}")


# ============================================================
# NORMALIZE
# ============================================================

maximum = np.max(np.abs(audio))

if maximum > 0:
    audio = audio / maximum


# ============================================================
# VAD
# ============================================================

print()
print("Running Voice Activity Detection...")

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

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


print(
    f"Speech regions detected: "
    f"{len(speech_regions)}"
)


# ============================================================
# CREATE OVERLAPPING WINDOWS
# ============================================================

print()
print("Creating overlapping windows...")

window_samples = int(
    WINDOW_DURATION *
    SAMPLE_RATE
)

hop_samples = int(
    HOP_DURATION *
    SAMPLE_RATE
)


windows = []


for region_start, region_end in speech_regions:

    position = region_start

    while (
        position +
        window_samples
        <= region_end
    ):

        window_audio = audio[
            position:
            position +
            window_samples
        ]

        windows.append(
            {
                "start": position,
                "end": position + window_samples,
                "audio": window_audio
            }
        )

        position += hop_samples


print(
    f"Analysis windows: "
    f"{len(windows)}"
)


if len(windows) < 2:

    print()
    print(
        "Not enough analysis windows."
    )

    raise SystemExit


# ============================================================
# EXTRACT WINDOW MFCC FEATURES
# ============================================================

print()
print("Extracting window features...")


window_features = []


for window in windows:

    mfcc = librosa.feature.mfcc(
        y=window["audio"],
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    feature = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ]
    )

    window_features.append(feature)


window_features = np.array(
    window_features
)


# ============================================================
# SCALE WINDOW FEATURES
# ============================================================

window_scaler = StandardScaler()

window_features_scaled = (
    window_scaler.fit_transform(
        window_features
    )
)


# ============================================================
# CALCULATE CHANGE SCORES
# ============================================================

print()
print("Calculating speaker change scores...")


change_scores = []


for i in range(
    len(window_features_scaled) - 1
):

    distance = np.linalg.norm(
        window_features_scaled[i]
        -
        window_features_scaled[i + 1]
    )

    change_scores.append(
        distance
    )


change_scores = np.array(
    change_scores
)


mean_change = np.mean(
    change_scores
)

std_change = np.std(
    change_scores
)


adaptive_threshold = (
    mean_change +
    std_change
)


print(
    f"Mean change score: "
    f"{mean_change:.4f}"
)

print(
    f"Change threshold: "
    f"{adaptive_threshold:.4f}"
)


# ============================================================
# DETECT SPEAKER CHANGE TIMES
# ============================================================

print()
print("Detecting speaker changes...")


speaker_change_samples = []


for i, score in enumerate(
    change_scores
):

    if score >= adaptive_threshold:

        change_sample = (
            windows[i + 1]["start"]
        )

        speaker_change_samples.append(
            change_sample
        )

        print(
            f"Possible change: "
            f"{change_sample / SAMPLE_RATE:.2f}s "
            f"| Score: {score:.4f}"
        )


# ============================================================
# CREATE ADAPTIVE SEGMENTS
# ============================================================

print()
print("Creating adaptive segments...")


adaptive_segments = []


for region_start, region_end in speech_regions:

    region_changes = []

    for change_sample in speaker_change_samples:

        if (
            region_start
            <
            change_sample
            <
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

        segment_duration = (
            end - start
        ) / SAMPLE_RATE


        # Skip segments that are too short
        # for reliable speaker features.

        if (
            segment_duration
            >= MIN_SEGMENT_DURATION
        ):

            adaptive_segments.append(
                (
                    start,
                    end
                )
            )


print(
    f"Adaptive segments retained: "
    f"{len(adaptive_segments)}"
)


for i, (
    start,
    end
) in enumerate(
    adaptive_segments
):

    print(
        f"Segment {i + 1}: "
        f"{start / SAMPLE_RATE:.2f}s "
        f"→ {end / SAMPLE_RATE:.2f}s "
        f"| {(end - start) / SAMPLE_RATE:.2f}s"
    )


if len(adaptive_segments) < 2:

    print()
    print(
        "Not enough adaptive segments "
        "for clustering."
    )

    raise SystemExit


# ============================================================
# EXTRACT SEGMENT MFCC FEATURES
# ============================================================

print()
print("Extracting adaptive segment features...")


segment_features = []


for start, end in adaptive_segments:

    segment = audio[
        start:end
    ]

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    feature = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ]
    )

    segment_features.append(
        feature
    )


X = np.array(
    segment_features
)


X = np.nan_to_num(
    X,
    nan=0,
    posinf=0,
    neginf=0
)


print(
    "Feature matrix:",
    X.shape
)


# ============================================================
# SCALE SEGMENT FEATURES
# ============================================================

print()
print("Scaling features...")


scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# FIND BEST NUMBER OF SPEAKERS
# ============================================================

print()
print("==========================================")
print("        TESTING NUMBER OF SPEAKERS")
print("==========================================")


maximum_k = min(
    MAX_SPEAKERS,
    len(X_scaled) - 1
)


best_k = 1
best_score = -1


for k in range(
    2,
    maximum_k + 1
):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(
        X_scaled
    )


    if len(
        np.unique(labels)
    ) < 2:

        continue


    score = silhouette_score(
        X_scaled,
        labels
    )


    print(
        f"K={k} "
        f"→ Silhouette = "
        f"{score:.4f}"
    )


    if score > best_score:

        best_score = score
        best_k = k


# ============================================================
# FINAL CLUSTERING
# ============================================================

print()
print(
    f"Selected speakers: "
    f"{best_k}"
)


kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)


labels = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# BUILD SPEAKER RESULTS
# ============================================================

speaker_segments = {}


for i, label in enumerate(
    labels
):

    speaker_name = (
        f"SPEAKER_{label:02d}"
    )


    if speaker_name not in speaker_segments:

        speaker_segments[
            speaker_name
        ] = []


    start, end = adaptive_segments[i]


    speaker_segments[
        speaker_name
    ].append(
        (
            start / SAMPLE_RATE,
            end / SAMPLE_RATE
        )
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("==========================================")
print("         VOXSPLIT V3 RESULTS")
print("==========================================")


print(
    f"Estimated speakers: "
    f"{best_k}"
)

print(
    f"Silhouette score: "
    f"{best_score:.4f}"
)


for speaker, times in speaker_segments.items():

    total_duration = sum(
        end - start
        for start, end in times
    )


    print()
    print(speaker)

    print(
        f"Segments: "
        f"{len(times)}"
    )

    print(
        f"Speech duration: "
        f"{total_duration:.2f}s"
    )


    for start, end in times:

        print(
            f"   {start:.2f}s "
            f"→ {end:.2f}s"
        )


# ============================================================
# COMPARISON
# ============================================================

BASELINE_SCORE = 0.3007


print()
print("==========================================")
print("       BASELINE VS V3 COMPARISON")
print("==========================================")


print(
    f"Baseline V2: "
    f"{BASELINE_SCORE:.4f}"
)

print(
    f"V3 Adaptive: "
    f"{best_score:.4f}"
)


difference = (
    best_score -
    BASELINE_SCORE
)


if difference > 0:

    print(
        f"Improvement: "
        f"+{difference:.4f}"
    )

else:

    print(
        f"Difference: "
        f"{difference:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/diarization_v3_results.txt"
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
        "VoxSplit Diarization V3\n"
    )

    file.write(
        "======================\n\n"
    )

    file.write(
        f"Estimated speakers: "
        f"{best_k}\n"
    )

    file.write(
        f"Silhouette score: "
        f"{best_score:.4f}\n"
    )

    file.write(
        f"Baseline score: "
        f"{BASELINE_SCORE:.4f}\n"
    )

    file.write(
        f"Difference: "
        f"{difference:.4f}\n\n"
    )


    for speaker, times in speaker_segments.items():

        file.write(
            f"{speaker}\n"
        )


        for start, end in times:

            file.write(
                f"   {start:.2f}s "
                f"→ {end:.2f}s\n"
            )


        file.write("\n")


print()
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)


print()
print("==========================================")
print("      DIARIZATION V3 COMPLETE")
print("==========================================")