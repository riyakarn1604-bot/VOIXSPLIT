import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from speaker_embedding_v6 import SpeakerEmbeddingCNN


# ============================================================
# VOXSPLIT V6
# SPEAKER EMBEDDING EVALUATION
# ============================================================

DATA_DIR = Path(
    "data/speaker_dataset/test"
)

MODEL_PATH = Path(
    "results/speaker_embedding_v6.pth"
)

RESULTS_DIR = Path(
    "results/v6_embedding_evaluation"
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

AUDIO_DURATION = 4

TARGET_LENGTH = (
    SAMPLE_RATE * AUDIO_DURATION
)

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256

NUM_SPEAKERS = 30

EMBEDDING_DIM = 128

BATCH_SIZE = 16


# ============================================================
# DEVICE
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print()

print("==========================================")

print("   VOXSPLIT V6 EMBEDDING EVALUATION")

print("==========================================")

print()

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ============================================================
# MEL SPECTROGRAM
# ============================================================

mel_transform = (

    torchaudio.transforms.MelSpectrogram(

        sample_rate=SAMPLE_RATE,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        n_mels=N_MELS

    )

)


# ============================================================
# DATASET
# ============================================================

class TestSpeakerDataset(Dataset):

    def __init__(
        self,
        root_dir
    ):

        self.root_dir = Path(
            root_dir
        )

        self.files = []

        self.labels = []


        speaker_dirs = sorted(

            [

                directory

                for directory
                in self.root_dir.iterdir()

                if directory.is_dir()

            ]

        )


        print()

        print(
            "Speakers found:",
            len(speaker_dirs)
        )


        for label, speaker_dir in enumerate(
            speaker_dirs
        ):

            wav_files = sorted(

                speaker_dir.glob(
                    "*.wav"
                )

            )


            for wav_file in wav_files:

                self.files.append(
                    wav_file
                )

                self.labels.append(
                    label
                )


        print(
            "Test files:",
            len(self.files)
        )


    def __len__(self):

        return len(
            self.files
        )


    def __getitem__(
        self,
        index
    ):

        file_path = self.files[
            index
        ]

        label = self.labels[
            index
        ]


        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        waveform, sample_rate = (

            torchaudio.load(
                file_path
            )

        )


        # ----------------------------------------------------
        # MONO
        # ----------------------------------------------------

        if waveform.shape[0] > 1:

            waveform = waveform.mean(

                dim=0,
                keepdim=True

            )


        # ----------------------------------------------------
        # RESAMPLE
        # ----------------------------------------------------

        if sample_rate != SAMPLE_RATE:

            waveform = (

                torchaudio.functional.resample(

                    waveform,

                    sample_rate,

                    SAMPLE_RATE

                )

            )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        max_value = waveform.abs().max()

        if max_value > 0:

            waveform = (

                waveform
                / max_value

            )


        # ----------------------------------------------------
        # FIXED LENGTH
        # ----------------------------------------------------

        current_length = waveform.shape[1]


        if current_length < TARGET_LENGTH:

            padding = (

                TARGET_LENGTH
                - current_length

            )

            waveform = (

                torch.nn.functional.pad(

                    waveform,

                    (0, padding)

                )

            )


        elif current_length > TARGET_LENGTH:

            start = (

                current_length
                - TARGET_LENGTH
            ) // 2

            waveform = (

                waveform[
                    :,
                    start:
                    start + TARGET_LENGTH
                ]

            )


        # ----------------------------------------------------
        # MEL
        # ----------------------------------------------------

        mel = mel_transform(
            waveform
        )


        # ----------------------------------------------------
        # LOG MEL
        # ----------------------------------------------------

        mel = torch.log(
            mel + 1e-9
        )


        return (
            mel,
            label,
            str(file_path)
        )


# ============================================================
# LOAD DATA
# ============================================================

dataset = TestSpeakerDataset(
    DATA_DIR
)


loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ============================================================
# LOAD MODEL
# ============================================================

print()

print("Loading trained V6 model...")


model = SpeakerEmbeddingCNN(

    num_speakers=NUM_SPEAKERS,

    embedding_dim=EMBEDDING_DIM

).to(device)


checkpoint = torch.load(

    MODEL_PATH,

    map_location=device,

    weights_only=False

)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)


model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

print()

print("Extracting 128-D speaker embeddings...")


all_embeddings = []

all_labels = []

all_paths = []

all_predictions = []


with torch.no_grad():

    for mel, labels, paths in loader:

        mel = mel.to(
            device
        )


        # ----------------------------------------------------
        # GET EMBEDDING
        # ----------------------------------------------------

        embeddings = model.embedding(

            model.features(
                mel
            )

        )


        # ----------------------------------------------------
        # NORMALIZE EMBEDDINGS
        # ----------------------------------------------------

        embeddings = (

            torch.nn.functional.normalize(

                embeddings,

                p=2,

                dim=1

            )

        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        logits = model.classifier(

            embeddings
        )


        predictions = (

            logits.argmax(
                dim=1
            )

        )


        all_embeddings.append(

            embeddings.cpu()

        )

        all_labels.append(

            labels
        )

        all_predictions.append(

            predictions.cpu()

        )

        all_paths.extend(paths)


# ============================================================
# COMBINE RESULTS
# ============================================================

embeddings = torch.cat(

    all_embeddings,

    dim=0

).numpy()


labels = torch.cat(

    all_labels,

    dim=0

).numpy()


predictions = torch.cat(

    all_predictions,

    dim=0

).numpy()


print()

print(
    "Embedding matrix shape:",
    embeddings.shape
)


# ============================================================
# CLASSIFICATION ACCURACY
# ============================================================

accuracy = (

    100.0
    * np.mean(
        predictions == labels
    )

)


print()

print(
    "Test classification accuracy:",
    f"{accuracy:.2f}%"
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

print()

print(
    "Calculating speaker similarity..."
)


same_scores = []

different_scores = []


num_samples = len(
    embeddings
)


for i in range(
    num_samples
):

    for j in range(
        i + 1,
        num_samples
    ):


        similarity = float(

            np.dot(

                embeddings[i],

                embeddings[j]

            )

        )


        if labels[i] == labels[j]:

            same_scores.append(
                similarity
            )

        else:

            different_scores.append(
                similarity
            )


same_scores = np.array(
    same_scores
)

different_scores = np.array(
    different_scores
)


print()

print("Same-speaker pairs:")

print(
    "Count:",
    len(same_scores)
)

print(
    "Mean similarity:",
    f"{same_scores.mean():.4f}"
)

print(
    "Std:",
    f"{same_scores.std():.4f}"
)


print()

print("Different-speaker pairs:")

print(
    "Count:",
    len(different_scores)
)

print(
    "Mean similarity:",
    f"{different_scores.mean():.4f}"
)

print(
    "Std:",
    f"{different_scores.std():.4f}"
)


# ============================================================
# SEPARATION
# ============================================================

separation = (

    same_scores.mean()
    - different_scores.mean()

)


print()

print(
    "Speaker separation:",
    f"{separation:.4f}"
)


# ============================================================
# SIMPLE THRESHOLD EVALUATION
# ============================================================

print()

print(
    "Finding best similarity threshold..."
)


all_scores = np.concatenate(

    [
        same_scores,
        different_scores
    ]

)


minimum = all_scores.min()

maximum = all_scores.max()


thresholds = np.linspace(

    minimum,

    maximum,

    200

)


best_threshold = 0.0

best_accuracy = 0.0


for threshold in thresholds:

    same_predictions = (

        same_scores >= threshold

    )

    different_predictions = (

        different_scores < threshold

    )


    correct = (

        same_predictions.sum()
        +
        different_predictions.sum()

    )


    total = (

        len(same_scores)
        +
        len(different_scores)

    )


    threshold_accuracy = (

        100.0
        * correct
        / total

    )


    if threshold_accuracy > best_accuracy:

        best_accuracy = (

            threshold_accuracy

        )

        best_threshold = threshold


print()

print(
    "Best threshold:",
    f"{best_threshold:.4f}"
)

print(
    "Pair classification accuracy:",
    f"{best_accuracy:.2f}%"
)


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

results_file = (

    RESULTS_DIR
    / "embedding_results.txt"

)


with open(

    results_file,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "VOXSPLIT V6 EMBEDDING EVALUATION\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        f"Test samples: {num_samples}\n"
    )

    file.write(
        f"Embedding dimension: "
        f"{EMBEDDING_DIM}\n"
    )

    file.write(
        f"Test classification accuracy: "
        f"{accuracy:.2f}%\n\n"
    )

    file.write(
        f"Same-speaker mean similarity: "
        f"{same_scores.mean():.4f}\n"
    )

    file.write(
        f"Same-speaker std: "
        f"{same_scores.std():.4f}\n\n"
    )

    file.write(
        f"Different-speaker mean similarity: "
        f"{different_scores.mean():.4f}\n"
    )

    file.write(
        f"Different-speaker std: "
        f"{different_scores.std():.4f}\n\n"
    )

    file.write(
        f"Speaker separation: "
        f"{separation:.4f}\n"
    )

    file.write(
        f"Best threshold: "
        f"{best_threshold:.4f}\n"
    )

    file.write(
        f"Pair classification accuracy: "
        f"{best_accuracy:.2f}%\n"
    )


# ============================================================
# PCA
# ============================================================

print()

print(
    "Creating PCA visualization..."
)


# PCA implemented with NumPy SVD

X = embeddings.copy()


X = X - X.mean(
    axis=0,
    keepdims=True
)


U, S, Vt = np.linalg.svd(

    X,

    full_matrices=False

)


pca_2d = (

    X @ Vt[:2].T

)


# ============================================================
# PCA PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)


for speaker_id in sorted(
    np.unique(labels)
):

    mask = (
        labels == speaker_id
    )


    plt.scatter(

        pca_2d[mask, 0],

        pca_2d[mask, 1],

        label=f"Speaker {speaker_id}"

    )


plt.title(
    "VoxSplit V6 Speaker Embeddings - PCA"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.legend(
    fontsize=7,
    ncol=3
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


pca_file = (

    RESULTS_DIR
    / "speaker_embeddings_pca.png"

)


plt.savefig(
    pca_file,
    dpi=300
)

plt.close()


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

embedding_file = (

    RESULTS_DIR
    / "test_embeddings.npz"

)


np.savez(

    embedding_file,

    embeddings=embeddings,

    labels=labels,

    predictions=predictions,

    paths=np.array(
        all_paths
    )

)


# ============================================================
# COMPLETE
# ============================================================

print()

print("==========================================")

print("       EVALUATION COMPLETE")

print("==========================================")

print()

print(
    f"Test classification accuracy: "
    f"{accuracy:.2f}%"
)

print(

    f"Same-speaker similarity: "
    f"{same_scores.mean():.4f}"

)

print(

    f"Different-speaker similarity: "
    f"{different_scores.mean():.4f}"

)

print(

    f"Speaker separation: "
    f"{separation:.4f}"

)

print(

    f"Best threshold: "
    f"{best_threshold:.4f}"

)

print(

    f"Pair accuracy: "
    f"{best_accuracy:.2f}%"

)

print()

print(
    "Results:",
    results_file
)

print(
    "PCA:",
    pca_file
)

print(
    "Embeddings:",
    embedding_file
)

print()

print("==========================================")
