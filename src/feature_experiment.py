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
print("       VOICESENSE FEATURE EXPERIMENT")
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
# VAD
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
# CREATE SEGMENTS
# ============================================================

# ============================================================
# CREATE OVERLAPPING SEGMENTS
# ============================================================

print()
print("Creating overlapping segments...")

segments = []

segment_samples = int(
    SEGMENT_DURATION * SAMPLE_RATE
)

hop_samples = int(
    SEGMENT_HOP * SAMPLE_RATE
)

for region_start, region_end in speech_regions:

    position = region_start

    while (
        position + segment_samples
        <= region_end
    ):

        segment = audio[
            position:
            position + segment_samples
        ]

        if len(segment) == segment_samples:

            segments.append(segment)

        position += hop_samples


print(
    "Total overlapping segments:",
    len(segments)
)

# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def mfcc_features(segment):

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


# ------------------------------------------------------------

def delta_features(segment):

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


# ------------------------------------------------------------

def delta_delta_features(segment):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    return np.concatenate(
        [
            np.mean(delta_delta, axis=1),
            np.std(delta_delta, axis=1)
        ]
    )


# ------------------------------------------------------------

def pitch_features(segment):

    try:

        f0, _, _ = librosa.pyin(
            segment,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SAMPLE_RATE
        )

        valid = f0[
            ~np.isnan(f0)
        ]

        if len(valid) == 0:

            return np.array(
                [0, 0, 0, 0]
            )

        return np.array(
            [
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid)
            ]
        )

    except Exception:

        return np.array(
            [0, 0, 0, 0]
        )


# ------------------------------------------------------------

def energy_features(segment):

    rms = librosa.feature.rms(
        y=segment
    )

    return np.array(
        [
            np.mean(rms),
            np.std(rms)
        ]
    )


# ------------------------------------------------------------

def spectral_features(segment):

    centroid = (
        librosa.feature.spectral_centroid(
            y=segment,
            sr=SAMPLE_RATE
        )
    )

    bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=segment,
            sr=SAMPLE_RATE
        )
    )

    rolloff = (
        librosa.feature.spectral_rolloff(
            y=segment,
            sr=SAMPLE_RATE
        )
    )

    zcr = (
        librosa.feature.zero_crossing_rate(
            segment
        )
    )

    return np.array(
        [
            np.mean(centroid),
            np.std(centroid),

            np.mean(bandwidth),
            np.std(bandwidth),

            np.mean(rolloff),
            np.std(rolloff),

            np.mean(zcr),
            np.std(zcr)
        ]
    )


# ============================================================
# BUILD DIFFERENT FEATURE SETS
# ============================================================

print()
print("Preparing feature combinations...")


experiments = {}


# ------------------------------------------------------------
# EXPERIMENT 1
# ------------------------------------------------------------

experiments["MFCC"] = []

for segment in segments:

    experiments["MFCC"].append(
        mfcc_features(segment)
    )


# ------------------------------------------------------------
# EXPERIMENT 2
# ------------------------------------------------------------

experiments["MFCC + Pitch"] = []

for segment in segments:

    feature = np.concatenate(
        [
            mfcc_features(segment),
            pitch_features(segment)
        ]
    )

    experiments[
        "MFCC + Pitch"
    ].append(feature)


# ------------------------------------------------------------
# EXPERIMENT 3
# ------------------------------------------------------------

experiments["MFCC + Delta"] = []

for segment in segments:

    feature = np.concatenate(
        [
            mfcc_features(segment),
            delta_features(segment)
        ]
    )

    experiments[
        "MFCC + Delta"
    ].append(feature)


# ------------------------------------------------------------
# EXPERIMENT 4
# ------------------------------------------------------------

experiments[
    "MFCC + Delta + DeltaDelta"
] = []

for segment in segments:

    feature = np.concatenate(
        [
            mfcc_features(segment),
            delta_features(segment),
            delta_delta_features(segment)
        ]
    )

    experiments[
        "MFCC + Delta + DeltaDelta"
    ].append(feature)


# ------------------------------------------------------------
# EXPERIMENT 5
# ------------------------------------------------------------

experiments[
    "MFCC + Pitch + Energy"
] = []

for segment in segments:

    feature = np.concatenate(
        [
            mfcc_features(segment),
            pitch_features(segment),
            energy_features(segment)
        ]
    )

    experiments[
        "MFCC + Pitch + Energy"
    ].append(feature)


# ------------------------------------------------------------
# EXPERIMENT 6
# ------------------------------------------------------------

experiments[
    "ALL FEATURES"
] = []

for segment in segments:

    feature = np.concatenate(
        [
            mfcc_features(segment),
            delta_features(segment),
            delta_delta_features(segment),
            pitch_features(segment),
            energy_features(segment),
            spectral_features(segment)
        ]
    )

    experiments[
        "ALL FEATURES"
    ].append(feature)


# ============================================================
# RUN EXPERIMENTS
# ============================================================

print()
print("==========================================")
print("          RUNNING EXPERIMENTS")
print("==========================================")


results = []


for experiment_name, feature_list in experiments.items():

    print()
    print(
        "------------------------------------------"
    )

    print(
        "Experiment:",
        experiment_name
    )

    X = np.array(
        feature_list
    )

    X = np.nan_to_num(
        X,
        nan=0,
        posinf=0,
        neginf=0
    )

    print(
        "Feature dimensions:",
        X.shape[1]
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )


    # --------------------------------------------------------
    # FIND BEST K
    # --------------------------------------------------------

    maximum_k = min(
        MAX_SPEAKERS,
        len(X) - 1
    )

    best_k = 1

    best_score = -1


    if maximum_k >= 2:

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
                f"K={k} → "
                f"{score:.4f}"
            )


            if score > best_score:

                best_score = score
                best_k = k


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        (
            experiment_name,
            X.shape[1],
            best_k,
            best_score
        )
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print("==========================================")
print("             FINAL RESULTS")
print("==========================================")

print()

print(
    f"{'Feature Set':<32}"
    f"{'Dimensions':<12}"
    f"{'Best K':<10}"
    f"{'Silhouette'}"
)

print(
    "-" * 70
)


for name, dimensions, k, score in results:

    print(
        f"{name:<32}"
        f"{dimensions:<12}"
        f"{k:<10}"
        f"{score:.4f}"
    )


# ============================================================
# BEST FEATURE SET
# ============================================================

best_result = max(
    results,
    key=lambda x: x[3]
)


print()
print("==========================================")
print("           BEST FEATURE SET")
print("==========================================")

print(
    "Feature set:",
    best_result[0]
)

print(
    "Dimensions:",
    best_result[1]
)

print(
    "Best number of speakers:",
    best_result[2]
)

print(
    "Silhouette score:",
    f"{best_result[3]:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/feature_experiment_results.txt"
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
        "VoiceSense Feature Experiment\n"
    )

    file.write(
        "==============================\n\n"
    )

    for name, dimensions, k, score in results:

        file.write(
            f"{name}\n"
        )

        file.write(
            f"Dimensions: {dimensions}\n"
        )

        file.write(
            f"Best K: {k}\n"
        )

        file.write(
            f"Silhouette: {score:.4f}\n\n"
        )

    file.write(
        "BEST FEATURE SET\n"
    )

    file.write(
        f"{best_result[0]}\n"
    )

    file.write(
        f"Silhouette: "
        f"{best_result[3]:.4f}\n"
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
print("          EXPERIMENT COMPLETE")
print("==========================================")