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

# MFCC is still used ONLY for speaker-change detection
N_MFCC_CHANGE = 20

# Log-Mel configuration
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160

MIN_SPEECH_DURATION = 0.5
MIN_SEGMENT_DURATION = 1.0

MAX_SPEAKERS = 4


# ============================================================
# HEADER
# ============================================================

print()
print("==========================================")
print("         VOXSPLIT DIARIZATION V5")
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
# MFCC FOR CHANGE DETECTION
# ============================================================

print()
print("Extracting MFCC features for boundary detection...")


window_features = []


for window in windows:

    mfcc = librosa.feature.mfcc(
        y=window["audio"],
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC_CHANGE
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
# LOG-MEL FEATURE EXTRACTION
# ============================================================

print()
print("Extracting LOG-MEL speaker features...")


segment_features = []


for i, (start, end) in enumerate(
    adaptive_segments
):

    print(
        f"Segment [{i + 1}/"
        f"{len(adaptive_segments)}]"
    )

    segment = audio[
        start:end
    ]


    # --------------------------------------------------------
    # Mel Spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=50,
        fmax=8000
    )


    # Convert power spectrogram to decibels

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # --------------------------------------------------------
    # Statistical representation
    # --------------------------------------------------------
    #
    # We summarize the temporal Log-Mel representation
    # using mean and standard deviation.
    #
    # 40 mean values
    # + 40 standard deviation values
    # = 80-dimensional feature vector
    #

    mean_features = np.mean(
        log_mel,
        axis=1
    )

    std_features = np.std(
        log_mel,
        axis=1
    )


    feature = np.concatenate(
        [
            mean_features,
            std_features
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


print()
print(
    "Log-Mel feature matrix:",
    X.shape
)


# ============================================================
# SCALE FEATURES
# ============================================================

print()
print("Scaling Log-Mel features...")


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
print("         VOXSPLIT V5 RESULTS")
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

V3_SCORE = 0.3723
V4_SCORE = 0.3755


print()
print("==========================================")
print("       V3 / V4 / V5 COMPARISON")
print("==========================================")


print(
    f"V3 MFCC: "
    f"{V3_SCORE:.4f}"
)

print(
    f"V4 MFCC + Energy: "
    f"{V4_SCORE:.4f}"
)

print(
    f"V5 Log-Mel: "
    f"{best_score:.4f}"
)


difference_v3 = (
    best_score -
    V3_SCORE
)

difference_v4 = (
    best_score -
    V4_SCORE
)


print(
    f"V5 vs V3: "
    f"{difference_v3:+.4f}"
)

print(
    f"V5 vs V4: "
    f"{difference_v4:+.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/diarization_v5_results.txt"
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
        "VoxSplit Diarization V5\n"
    )

    file.write(
        "======================\n\n"
    )

    file.write(
        "Feature representation: Log-Mel\n"
    )

    file.write(
        f"Estimated speakers: "
        f"{best_k}\n"
    )

    file.write(
        f"V3 MFCC score: "
        f"{V3_SCORE:.4f}\n"
    )

    file.write(
        f"V4 MFCC + Energy score: "
        f"{V4_SCORE:.4f}\n"
    )

    file.write(
        f"V5 Log-Mel score: "
        f"{best_score:.4f}\n"
    )

    file.write(
        f"V5 vs V3: "
        f"{difference_v3:+.4f}\n"
    )

    file.write(
        f"V5 vs V4: "
        f"{difference_v4:+.4f}\n\n"
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
print("       DIARIZATION V5 COMPLETE")
print("==========================================")