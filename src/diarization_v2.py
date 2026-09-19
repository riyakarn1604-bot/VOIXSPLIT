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

AUDIO_FILE = Path("data/audio/test2.m4a")

SAMPLE_RATE = 16000

FRAME_DURATION = 30  # milliseconds

VAD_AGGRESSIVENESS = 2

SEGMENT_DURATION = 1.5

MIN_SPEECH_DURATION = 0.5

MAX_SPEAKERS = 4

N_MFCC = 20


# ============================================================
# HEADER
# ============================================================

print()
print("==========================================")
print("      VOICESENSE DIARIZATION V2")
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

print(f"Audio duration: {len(audio) / sr:.2f} seconds")
print(f"Sample rate: {sr} Hz")


# ============================================================
# NORMALIZE AUDIO
# ============================================================

audio = audio.astype(np.float32)

max_value = np.max(np.abs(audio))

if max_value > 0:
    audio = audio / max_value


# ============================================================
# VOICE ACTIVITY DETECTION
# ============================================================

print()
print("Running Voice Activity Detection...")

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

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
    ).astype(
        np.int16
    ).tobytes()

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

for sample_position, is_speech in frames:

    if is_speech:

        if speech_start is None:
            speech_start = sample_position

    else:

        if speech_start is not None:

            speech_end = sample_position

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


print()
print(
    "Speech regions detected:",
    len(speech_regions)
)


# ============================================================
# CREATE FIXED-LENGTH SEGMENTS
# ============================================================

print()
print("Creating speaker segments...")

segments = []

segment_samples = int(
    SEGMENT_DURATION * SAMPLE_RATE
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

            segments.append(
                (
                    position,
                    position + segment_samples,
                    segment
                )
            )

        position += segment_samples


print(
    "Usable segments:",
    len(segments)
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(segment):

    features = []


    # --------------------------------------------------------
    # 1. MFCC
    # --------------------------------------------------------

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

    features.extend(mfcc_mean)
    features.extend(mfcc_std)


    # --------------------------------------------------------
    # 2. DELTA MFCC
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    delta_mean = np.mean(
        delta,
        axis=1
    )

    delta_std = np.std(
        delta,
        axis=1
    )

    features.extend(delta_mean)
    features.extend(delta_std)


    # --------------------------------------------------------
    # 3. DELTA-DELTA MFCC
    # --------------------------------------------------------

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    delta_delta_mean = np.mean(
        delta_delta,
        axis=1
    )

    delta_delta_std = np.std(
        delta_delta,
        axis=1
    )

    features.extend(
        delta_delta_mean
    )

    features.extend(
        delta_delta_std
    )


    # --------------------------------------------------------
    # 4. RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=segment
    )

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    features.append(rms_mean)
    features.append(rms_std)


    # --------------------------------------------------------
    # 5. ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        segment
    )

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    features.append(zcr_mean)
    features.append(zcr_std)


    # --------------------------------------------------------
    # 6. SPECTRAL CENTROID
    # --------------------------------------------------------

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=segment,
            sr=SAMPLE_RATE
        )
    )

    centroid_mean = np.mean(
        spectral_centroid
    )

    centroid_std = np.std(
        spectral_centroid
    )

    features.append(centroid_mean)
    features.append(centroid_std)


    # --------------------------------------------------------
    # 7. SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=segment,
            sr=SAMPLE_RATE
        )
    )

    bandwidth_mean = np.mean(
        spectral_bandwidth
    )

    bandwidth_std = np.std(
        spectral_bandwidth
    )

    features.append(
        bandwidth_mean
    )

    features.append(
        bandwidth_std
    )


    # --------------------------------------------------------
    # 8. SPECTRAL ROLLOFF
    # --------------------------------------------------------

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            y=segment,
            sr=SAMPLE_RATE
        )
    )

    rolloff_mean = np.mean(
        spectral_rolloff
    )

    rolloff_std = np.std(
        spectral_rolloff
    )

    features.append(
        rolloff_mean
    )

    features.append(
        rolloff_std
    )


    # --------------------------------------------------------
    # 9. PITCH / FUNDAMENTAL FREQUENCY
    # --------------------------------------------------------

    try:

        f0, voiced_flag, voiced_prob = (
            librosa.pyin(
                segment,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=SAMPLE_RATE
            )
        )

        valid_f0 = f0[
            ~np.isnan(f0)
        ]

        if len(valid_f0) > 0:

            pitch_mean = np.mean(
                valid_f0
            )

            pitch_std = np.std(
                valid_f0
            )

            pitch_min = np.min(
                valid_f0
            )

            pitch_max = np.max(
                valid_f0
            )

        else:

            pitch_mean = 0
            pitch_std = 0
            pitch_min = 0
            pitch_max = 0

    except Exception:

        pitch_mean = 0
        pitch_std = 0
        pitch_min = 0
        pitch_max = 0


    features.append(pitch_mean)
    features.append(pitch_std)
    features.append(pitch_min)
    features.append(pitch_max)


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

print()
print("Extracting acoustic features...")

features = []

for i, (
    start,
    end,
    segment
) in enumerate(segments):

    print(
        f"Segment [{i + 1}/{len(segments)}]",
        end="\r"
    )

    feature_vector = extract_features(
        segment
    )

    features.append(
        feature_vector
    )


features = np.array(
    features
)


print()
print()
print(
    "Feature matrix shape:",
    features.shape
)


# ============================================================
# CHECK FEATURES
# ============================================================

if len(features) < 2:

    print()
    print(
        "Not enough segments."
    )

    raise SystemExit


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

print()
print("Checking feature values...")

features = np.nan_to_num(
    features,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

print("Feature validation complete.")


# ============================================================
# STANDARDIZATION
# ============================================================

print()
print("Scaling features...")

scaler = StandardScaler()

scaled_features = (
    scaler.fit_transform(
        features
    )
)


# ============================================================
# SEARCH FOR BEST NUMBER OF SPEAKERS
# ============================================================

print()
print("==========================================")
print("Searching for number of speakers...")
print("==========================================")


maximum_possible = min(
    MAX_SPEAKERS,
    len(features) - 1
)

best_k = 1

best_score = -1


if maximum_possible >= 2:

    for k in range(
        2,
        maximum_possible + 1
    ):

        print(
            f"Testing K = {k}..."
        )

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(
            scaled_features
        )

        if len(
            set(labels)
        ) < 2:

            continue

        score = silhouette_score(
            scaled_features,
            labels
        )

        print(
            f"K = {k} | "
            f"Silhouette Score = "
            f"{score:.4f}"
        )

        if score > best_score:

            best_score = score
            best_k = k


else:

    best_k = 1
    best_score = 0


# ============================================================
# FINAL CLUSTERING
# ============================================================

print()
print("==========================================")
print(
    f"Estimated number of speakers: {best_k}"
)
print(
    f"Best silhouette score: {best_score:.4f}"
)
print("==========================================")


kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(
    scaled_features
)


# ============================================================
# CREATE SPEAKER TIMELINE
# ============================================================

print()
print("Clustering speaker segments...")

speaker_segments = {}


for i, label in enumerate(labels):

    speaker_name = (
        f"SPEAKER_{label:02d}"
    )

    if speaker_name not in speaker_segments:

        speaker_segments[
            speaker_name
        ] = []

    start = (
        segments[i][0]
        / SAMPLE_RATE
    )

    end = (
        segments[i][1]
        / SAMPLE_RATE
    )

    speaker_segments[
        speaker_name
    ].append(
        (
            start,
            end
        )
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("========== SPEAKER RESULTS ==========")


for speaker, times in speaker_segments.items():

    duration = sum(
        end - start
        for start, end in times
    )

    print()
    print(speaker)

    print(
        f"Segments: {len(times)}"
    )

    print(
        f"Speech duration: "
        f"{duration:.2f}s"
    )

    for start, end in times:

        print(
            f"   {start:.2f}s → "
            f"{end:.2f}s"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = Path(
    "data/diarization_v2_results.txt"
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
        "VoiceSense Diarization V2\n"
    )

    file.write(
        "=========================\n\n"
    )

    file.write(
        f"Audio: {AUDIO_FILE}\n"
    )

    file.write(
        f"Audio duration: "
        f"{len(audio) / SAMPLE_RATE:.2f}s\n"
    )

    file.write(
        f"Number of speakers: "
        f"{best_k}\n"
    )

    file.write(
        f"Silhouette score: "
        f"{best_score:.4f}\n\n"
    )

    for speaker, times in speaker_segments.items():

        duration = sum(
            end - start
            for start, end in times
        )

        file.write(
            f"{speaker}\n"
        )

        file.write(
            f"Segments: {len(times)}\n"
        )

        file.write(
            f"Speech duration: "
            f"{duration:.2f}s\n"
        )

        for start, end in times:

            file.write(
                f"{start:.2f}s - "
                f"{end:.2f}s\n"
            )

        file.write("\n")


# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("       DIARIZATION V2 COMPLETE")
print("==========================================")

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Compare this silhouette score with:"
)

print(
    "Version 1 → 0.2753"
)

print(
    f"Version 2 → {best_score:.4f}"
)

print()