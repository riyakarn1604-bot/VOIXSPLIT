
import numpy as np
import librosa
import webrtcvad

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = Path("data/audio/test1.m4a")

SAMPLE_RATE = 16000

FRAME_DURATION = 30

VAD_AGGRESSIVENESS = 2

SEGMENT_DURATION = 1.5

SEGMENT_HOP = 0.75

MIN_SPEECH_DURATION = 0.5

MAX_SPEAKERS = 4

N_MFCC = 20


# ============================================================
# HEADER
# ============================================================

print()
print("==========================================")
print("      VOICESENSE CLUSTERING EXPERIMENT")
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

print(
    f"Audio duration: {len(audio) / sr:.2f}s"
)


# ============================================================
# NORMALIZE
# ============================================================

audio = audio.astype(np.float32)

maximum = np.max(np.abs(audio))

if maximum > 0:
    audio = audio / maximum


# ============================================================
# VOICE ACTIVITY DETECTION
# ============================================================

print()
print("Running VAD...")

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
        start:start + frame_size
    ]

    pcm = (
        frame * 32767
    ).astype(
        np.int16
    ).tobytes()

    speech = vad.is_speech(
        pcm,
        SAMPLE_RATE
    )

    frames.append(
        (
            start,
            speech
        )
    )


# ============================================================
# FIND SPEECH REGIONS
# ============================================================

speech_regions = []

speech_start = None

for position, speech in frames:

    if speech:

        if speech_start is None:
            speech_start = position

    else:

        if speech_start is not None:

            speech_end = position

            duration = (
                speech_end -
                speech_start
            ) / SAMPLE_RATE

            if duration >= MIN_SPEECH_DURATION:

                speech_regions.append(
                    (
                        speech_start,
                        speech_end
                    )
                )

            speech_start = None


# Handle speech until EOF

if speech_start is not None:

    speech_end = len(audio)

    duration = (
        speech_end -
        speech_start
    ) / SAMPLE_RATE

    if duration >= MIN_SPEECH_DURATION:

        speech_regions.append(
            (
                speech_start,
                speech_end
            )
        )


print(
    "Speech regions:",
    len(speech_regions)
)


# ============================================================
# CREATE OVERLAPPING SEGMENTS
# ============================================================

print()
print("Creating overlapping segments...")

segments = []

segment_samples = int(
    SEGMENT_DURATION *
    SAMPLE_RATE
)

hop_samples = int(
    SEGMENT_HOP *
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
                segment
            )

        position += hop_samples


print(
    "Total segments:",
    len(segments)
)


# ============================================================
# MFCC FEATURE EXTRACTION
# ============================================================

print()
print("Extracting MFCC features...")

features = []


for i, segment in enumerate(segments):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

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


X = np.array(
    features
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
# SCALE FEATURES
# ============================================================

print()
print("Scaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# FUNCTION TO TEST CLUSTERING
# ============================================================

def evaluate_clustering(
    method_name,
    labels
):

    unique_labels = np.unique(
        labels
    )

    if len(unique_labels) < 2:

        return -1

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"{method_name:<25}"
        f"→ {score:.4f}"
    )

    return score


# ============================================================
# EXPERIMENTS
# ============================================================

print()
print("==========================================")
print("        RUNNING CLUSTERING TESTS")
print("==========================================")


results = []


# ============================================================
# K-MEANS
# ============================================================

print()
print("------------------------------------------")
print("K-MEANS")
print("------------------------------------------")


for k in range(
    2,
    MAX_SPEAKERS + 1
):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = evaluate_clustering(
        f"K-Means K={k}",
        labels
    )

    results.append(
        (
            "K-Means",
            k,
            score,
            labels
        )
    )


# ============================================================
# AGGLOMERATIVE CLUSTERING
# ============================================================

print()
print("------------------------------------------")
print("AGGLOMERATIVE CLUSTERING")
print("------------------------------------------")


for k in range(
    2,
    MAX_SPEAKERS + 1
):

    model = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage="ward"
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = evaluate_clustering(
        f"Agglomerative K={k}",
        labels
    )

    results.append(
        (
            "Agglomerative",
            k,
            score,
            labels
        )
    )


# ============================================================
# FIND BEST RESULT
# ============================================================

best_result = max(
    results,
    key=lambda x: x[2]
)


best_method = best_result[0]

best_k = best_result[1]

best_score = best_result[2]

best_labels = best_result[3]


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("==========================================")
print("          FINAL COMPARISON")
print("==========================================")

print()

print(
    f"{'Method':<25}"
    f"{'K':<10}"
    f"{'Silhouette'}"
)

print(
    "-" * 50
)


for method, k, score, _ in results:

    print(
        f"{method:<25}"
        f"{k:<10}"
        f"{score:.4f}"
    )


print()
print("==========================================")
print("          BEST CLUSTERING METHOD")
print("==========================================")

print(
    "Method:",
    best_method
)

print(
    "Number of speakers:",
    best_k
)

print(
    "Silhouette score:",
    f"{best_score:.4f}"
)


# ============================================================
# SPEAKER DISTRIBUTION
# ============================================================

print()
print("========== SPEAKER DISTRIBUTION ==========")


for speaker_id in range(best_k):

    count = np.sum(
        best_labels == speaker_id
    )

    duration = (
        count *
        SEGMENT_DURATION
    )

    print()

    print(
        f"SPEAKER_{speaker_id:02d}"
    )

    print(
        f"Segments: {count}"
    )

    print(
        f"Approx speech duration: "
        f"{duration:.2f}s"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/clustering_experiment_results.txt"
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
        "VoiceSense Clustering Experiment\n"
    )

    file.write(
        "================================\n\n"
    )

    file.write(
        f"Audio: {AUDIO_FILE}\n"
    )

    file.write(
        f"Segments: {len(segments)}\n\n"
    )

    for method, k, score, _ in results:

        file.write(
            f"{method} | "
            f"K={k} | "
            f"Silhouette={score:.4f}\n"
        )

    file.write("\n")

    file.write(
        "BEST RESULT\n"
    )

    file.write(
        f"Method: {best_method}\n"
    )

    file.write(
        f"K: {best_k}\n"
    )

    file.write(
        f"Silhouette: {best_score:.4f}\n"
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
print("       CLUSTERING EXPERIMENT COMPLETE")
print("==========================================")

