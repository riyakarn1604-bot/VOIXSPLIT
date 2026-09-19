
import numpy as np
import librosa
import webrtcvad
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

N_MFCC = 20


# ============================================================
# HEADER
# ============================================================

print()
print("==========================================")
print("       VOICESENSE MFCC VISUALIZATION")
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

audio = audio.astype(
    np.float32
)

maximum = np.max(
    np.abs(audio)
)

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
# EXTRACT MFCC
# ============================================================

print()
print("Extracting MFCC features...")

features = []

for segment in segments:

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
# SCALE
# ============================================================

print()
print("Scaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# PCA
# ============================================================

print()
print("Applying PCA...")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


print(
    "PCA shape:",
    X_pca.shape
)

print(
    "Variance explained by PC1:",
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    "Variance explained by PC2:",
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    "Total variance explained:",
    f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
)


# ============================================================
# SAVE PCA VALUES
# ============================================================

OUTPUT_FILE = Path(
    "data/mfcc_pca_results.txt"
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
        "VoiceSense MFCC PCA Results\n"
    )

    file.write(
        "===========================\n\n"
    )

    file.write(
        f"Segments: {len(segments)}\n"
    )

    file.write(
        f"Original dimensions: {X.shape[1]}\n"
    )

    file.write(
        f"PCA dimensions: 2\n\n"
    )

    for i, point in enumerate(X_pca):

        file.write(
            f"Segment {i + 1:02d}: "
            f"X={point[0]:.4f}, "
            f"Y={point[1]:.4f}\n"
        )


# ============================================================
# VISUALIZATION
# ============================================================

print()
print("Creating visualization...")


plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    s=80
)


# Label every segment

for i, (
    x,
    y
) in enumerate(X_pca):

    plt.annotate(
        f"S{i + 1}",
        (
            x,
            y
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "VoiceSense - MFCC Feature Distribution"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


IMAGE_FILE = Path(
    "data/mfcc_pca_visualization.png"
)

plt.savefig(
    IMAGE_FILE,
    dpi=150
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("       PCA VISUALIZATION COMPLETE")
print("==========================================")

print()
print(
    "PCA values saved to:",
    OUTPUT_FILE
)

print(
    "Visualization saved to:",
    IMAGE_FILE
)

print()
