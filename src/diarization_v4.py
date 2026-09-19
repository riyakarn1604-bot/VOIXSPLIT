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
print("         VOXSPLIT DIARIZATION V4")
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
print(f"Sample rate: {sr} Hz")


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
                speech_end - speech_start
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
        speech_end - speech_start
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
        position + window_samples
        <= region_end
    ):

        windows.append(
            {
                "start": position,
                "end": position + window_samples,
                "audio": audio[
                    position:
                    position + window_samples
                ]
            }
        )

        position += hop_samples


print(
    f"Analysis windows: "
    f"{len(windows)}"
)


if len(windows) < 2:

    print()
    print("Not enough analysis windows.")
    raise SystemExit


# ============================================================
# MFCC FEATURES FOR CHANGE DETECTION
# ============================================================

print()
print("Extracting window MFCC features...")


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


window_scaler = StandardScaler()

window_features_scaled = (
    window_scaler.fit_transform(
        window_features
    )
)


# ============================================================
# SPEAKER CHANGE DETECTION
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

    change_scores.append(distance)


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


speaker_change_samples = []

print()
print("Detecting speaker changes...")


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


for i, (start, end) in enumerate(
    adaptive_segments
):

    print(
        f"Segment {i + 1}: "
        f"{start / SAMPLE_RATE:.2f}s "
        f"→ "
        f"{end / SAMPLE_RATE:.2f}s "
        f"| "
        f"{(end - start) / SAMPLE_RATE:.2f}s"
    )


if len(adaptive_segments) < 3:

    print()
    print(
        "Not enough adaptive segments "
        "for feature evaluation."
    )

    raise SystemExit


# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def safe_mean(values):
    return float(np.mean(values))


def safe_std(values):
    return float(np.std(values))


def extract_mfcc(segment):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ]
    )


def extract_delta_mfcc(segment):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(
        mfcc
    )

    return np.concatenate(
        [
            np.mean(delta, axis=1),
            np.std(delta, axis=1)
        ]
    )


def extract_delta2_mfcc(segment):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    return np.concatenate(
        [
            np.mean(delta2, axis=1),
            np.std(delta2, axis=1)
        ]
    )


def extract_pitch(segment):

    f0, voiced_flag, voiced_probs = (
        librosa.pyin(
            segment,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SAMPLE_RATE
        )
    )

    valid_pitch = f0[
        ~np.isnan(f0)
    ]

    if len(valid_pitch) == 0:

        return np.array(
            [0.0, 0.0]
        )

    return np.array(
        [
            safe_mean(valid_pitch),
            safe_std(valid_pitch)
        ]
    )


def extract_energy(segment):

    rms = librosa.feature.rms(
        y=segment
    )[0]

    return np.array(
        [
            safe_mean(rms),
            safe_std(rms)
        ]
    )


def extract_spectral(segment):

    centroid = librosa.feature.spectral_centroid(
        y=segment,
        sr=SAMPLE_RATE
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=segment,
        sr=SAMPLE_RATE
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        segment
    )[0]

    return np.array(
        [
            safe_mean(centroid),
            safe_std(centroid),
            safe_mean(bandwidth),
            safe_std(bandwidth),
            safe_mean(zcr),
            safe_std(zcr)
        ]
    )


# ============================================================
# EXTRACT ALL FEATURE GROUPS
# ============================================================

print()
print("Extracting speaker feature groups...")


mfcc_features = []
delta_features = []
delta2_features = []
pitch_features = []
energy_features = []
spectral_features = []


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


    mfcc_features.append(
        extract_mfcc(segment)
    )

    delta_features.append(
        extract_delta_mfcc(segment)
    )

    delta2_features.append(
        extract_delta2_mfcc(segment)
    )

    pitch_features.append(
        extract_pitch(segment)
    )

    energy_features.append(
        extract_energy(segment)
    )

    spectral_features.append(
        extract_spectral(segment)
    )


mfcc_features = np.array(
    mfcc_features
)

delta_features = np.array(
    delta_features
)

delta2_features = np.array(
    delta2_features
)

pitch_features = np.array(
    pitch_features
)

energy_features = np.array(
    energy_features
)

spectral_features = np.array(
    spectral_features
)


# ============================================================
# CLEAN FEATURES
# ============================================================

def clean_features(X):

    return np.nan_to_num(
        X,
        nan=0,
        posinf=0,
        neginf=0
    )


mfcc_features = clean_features(
    mfcc_features
)

delta_features = clean_features(
    delta_features
)

delta2_features = clean_features(
    delta2_features
)

pitch_features = clean_features(
    pitch_features
)

energy_features = clean_features(
    energy_features
)

spectral_features = clean_features(
    spectral_features
)


# ============================================================
# FEATURE EXPERIMENTS
# ============================================================

feature_sets = {

    "MFCC":
        mfcc_features,

    "MFCC + DELTA":
        np.hstack(
            [
                mfcc_features,
                delta_features
            ]
        ),

    "MFCC + DELTA + DELTA2":
        np.hstack(
            [
                mfcc_features,
                delta_features,
                delta2_features
            ]
        ),

    "MFCC + PITCH":
        np.hstack(
            [
                mfcc_features,
                pitch_features
            ]
        ),

    "MFCC + ENERGY":
        np.hstack(
            [
                mfcc_features,
                energy_features
            ]
        ),

    "MFCC + SPECTRAL":
        np.hstack(
            [
                mfcc_features,
                spectral_features
            ]
        ),

    "MFCC + PITCH + ENERGY":
        np.hstack(
            [
                mfcc_features,
                pitch_features,
                energy_features
            ]
        ),

    "ALL FEATURES":
        np.hstack(
            [
                mfcc_features,
                delta_features,
                delta2_features,
                pitch_features,
                energy_features,
                spectral_features
            ]
        )
}


# ============================================================
# CLUSTERING FUNCTION
# ============================================================

def evaluate_features(
    feature_matrix,
    feature_name
):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        feature_matrix
    )

    maximum_k = min(
        MAX_SPEAKERS,
        len(X_scaled) - 1
    )

    best_k = None
    best_score = -1


    print()
    print("------------------------------------------")
    print(feature_name)
    print("------------------------------------------")


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


    return best_k, best_score


# ============================================================
# RUN FEATURE EXPERIMENTS
# ============================================================

print()
print("==========================================")
print("       V4 FEATURE EXPERIMENTS")
print("==========================================")


experiment_results = []


for feature_name, feature_matrix in (
    feature_sets.items()
):

    best_k, best_score = evaluate_features(
        feature_matrix,
        feature_name
    )

    experiment_results.append(
        (
            feature_name,
            best_k,
            best_score
        )
    )


# ============================================================
# FIND BEST FEATURE SET
# ============================================================

best_result = max(
    experiment_results,
    key=lambda x: x[2]
)


best_feature_name = best_result[0]
best_k = best_result[1]
best_score = best_result[2]


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("==========================================")
print("          VOXSPLIT V4 RESULTS")
print("==========================================")


for (
    feature_name,
    k,
    score
) in experiment_results:

    print(
        f"{feature_name:<30} "
        f"K={k} "
        f"Score={score:.4f}"
    )


print()
print(
    f"Best feature set: "
    f"{best_feature_name}"
)

print(
    f"Best K: "
    f"{best_k}"
)

print(
    f"Best silhouette: "
    f"{best_score:.4f}"
)


# ============================================================
# V3 COMPARISON
# ============================================================

V3_SCORE = 0.3723


difference = (
    best_score -
    V3_SCORE
)


relative_change = (
    difference /
    V3_SCORE
) * 100


print()
print("==========================================")
print("          V3 VS V4 COMPARISON")
print("==========================================")


print(
    f"V3 MFCC: "
    f"{V3_SCORE:.4f}"
)

print(
    f"V4 Best: "
    f"{best_score:.4f}"
)

print(
    f"Difference: "
    f"{difference:+.4f}"
)

print(
    f"Relative change: "
    f"{relative_change:+.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/diarization_v4_results.txt"
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
        "VoxSplit Diarization V4\n"
    )

    file.write(
        "======================\n\n"
    )

    file.write(
        f"V3 baseline: "
        f"{V3_SCORE:.4f}\n\n"
    )


    for (
        feature_name,
        k,
        score
    ) in experiment_results:

        file.write(
            f"{feature_name} | "
            f"K={k} | "
            f"Silhouette={score:.4f}\n"
        )


    file.write("\n")

    file.write(
        f"Best feature set: "
        f"{best_feature_name}\n"
    )

    file.write(
        f"Best K: "
        f"{best_k}\n"
    )

    file.write(
        f"Best silhouette: "
        f"{best_score:.4f}\n"
    )

    file.write(
        f"Difference from V3: "
        f"{difference:+.4f}\n"
    )

    file.write(
        f"Relative change: "
        f"{relative_change:+.2f}%\n"
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
print("       DIARIZATION V4 COMPLETE")
print("==========================================")