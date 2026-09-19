import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    silhouette_score,
    accuracy_score
)

from sklearn.metrics.pairwise import cosine_similarity

from speaker_embedding_v6 import SpeakerEmbeddingCNN


# ============================================================
# VOXSPLIT V6 — EMBEDDING EVALUATION
# ============================================================

DATA_DIR = Path("data/speaker_dataset")
MODEL_PATH = Path("results/speaker_embedding_v6.pth")

SAMPLE_RATE = 16000
AUDIO_DURATION = 4
TARGET_AUDIO_LENGTH = SAMPLE_RATE * AUDIO_DURATION

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

NUM_SPEAKERS = 30
EMBEDDING_DIM = 128

BATCH_SIZE = 16

SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
print("     VOXSPLIT V6 EMBEDDING EVALUATION")
print("==========================================")

print()
print("Device:", device)


# ============================================================
# MEL SPECTROGRAM
# ============================================================

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)


# ============================================================
# DATASET
# ============================================================

class SpeakerDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.files = []
        self.labels = []

        speaker_dirs = sorted(
            [
                directory
                for directory in self.root_dir.iterdir()
                if directory.is_dir()
            ]
        )

        if len(speaker_dirs) != NUM_SPEAKERS:

            raise RuntimeError(
                f"Expected {NUM_SPEAKERS} speakers, "
                f"but found {len(speaker_dirs)}"
            )

        for label, speaker_dir in enumerate(speaker_dirs):

            wav_files = sorted(
                speaker_dir.glob("*.wav")
            )

            for wav_file in wav_files:

                self.files.append(wav_file)
                self.labels.append(label)

        print()
        print(
            f"Dataset: {self.root_dir}"
        )

        print(
            f"Files: {len(self.files)}"
        )

        print(
            f"Speakers: {len(speaker_dirs)}"
        )


    def __len__(self):

        return len(self.files)


    def __getitem__(self, index):

        file_path = self.files[index]
        label = self.labels[index]

        waveform, sample_rate = torchaudio.load(
            file_path
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

            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                SAMPLE_RATE
            )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        max_amplitude = waveform.abs().max()

        if max_amplitude > 0:

            waveform = (
                waveform / max_amplitude
            )


        # ----------------------------------------------------
        # FIXED LENGTH
        # ----------------------------------------------------

        current_length = waveform.shape[1]

        if current_length < TARGET_AUDIO_LENGTH:

            padding = (
                TARGET_AUDIO_LENGTH
                - current_length
            )

            waveform = torch.nn.functional.pad(
                waveform,
                (0, padding)
            )

        elif current_length > TARGET_AUDIO_LENGTH:

            max_start = (
                current_length
                - TARGET_AUDIO_LENGTH
            )

            # Deterministic center crop
            start = max_start // 2

            waveform = waveform[
                :,
                start:
                start + TARGET_AUDIO_LENGTH
            ]


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


        return mel, label, str(file_path)


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("Loading test dataset...")

test_dataset = SpeakerDataset(
    DATA_DIR / "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading V6 model...")

model = SpeakerEmbeddingCNN(
    num_speakers=NUM_SPEAKERS,
    embedding_dim=EMBEDDING_DIM
).to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(
    "Checkpoint epoch:",
    checkpoint.get("epoch", "unknown")
)

print(
    "Validation accuracy:",
    checkpoint.get(
        "val_accuracy",
        "unknown"
    )
)


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

print()
print("Extracting CNN embeddings...")

all_embeddings = []
all_labels = []
all_predictions = []
all_files = []


with torch.no_grad():

    for mel, labels, files in test_loader:

        mel = mel.to(device)
        labels = labels.to(device)

        # ----------------------------------------------------
        # CLASSIFICATION OUTPUT
        # ----------------------------------------------------

        logits = model(mel)

        predictions = logits.argmax(
            dim=1
        )


        # ----------------------------------------------------
        # 128-D EMBEDDING
        # ----------------------------------------------------

        embeddings = model(
            mel,
            return_embedding=True
        )


        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_files.extend(files)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

embeddings = torch.cat(
    all_embeddings
).numpy()

labels = torch.cat(
    all_labels
).numpy()

predictions = torch.cat(
    all_predictions
).numpy()


print()
print("Embedding matrix:")
print(
    embeddings.shape
)


# ============================================================
# CLASSIFICATION ACCURACY
# ============================================================

classification_accuracy = (
    accuracy_score(
        labels,
        predictions
    )
    * 100
)


print()
print("==========================================")
print("        CLASSIFICATION RESULT")
print("==========================================")

print(
    f"Test Classification Accuracy: "
    f"{classification_accuracy:.2f}%"
)


# ============================================================
# L2 NORMALIZATION
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

normalized_embeddings = (
    embeddings
    / np.maximum(norms, 1e-12)
)


# ============================================================
# SILHOUETTE SCORE
# ============================================================

print()
print("Calculating silhouette score...")

silhouette = silhouette_score(
    normalized_embeddings,
    labels,
    metric="cosine"
)


print()
print("==========================================")
print("        EMBEDDING QUALITY")
print("==========================================")

print(
    f"CNN Embedding Silhouette: "
    f"{silhouette:.4f}"
)


# ============================================================
# SAME / DIFFERENT SPEAKER SIMILARITY
# ============================================================

print()
print(
    "Calculating speaker similarities..."
)

similarities = cosine_similarity(
    normalized_embeddings
)


same_scores = []
different_scores = []


n = len(labels)

for i in range(n):

    for j in range(i + 1, n):

        score = similarities[i, j]

        if labels[i] == labels[j]:

            same_scores.append(score)

        else:

            different_scores.append(score)


same_scores = np.array(
    same_scores
)

different_scores = np.array(
    different_scores
)


# ============================================================
# SIMILARITY STATISTICS
# ============================================================

same_mean = same_scores.mean()
same_std = same_scores.std()

different_mean = different_scores.mean()
different_std = different_scores.std()

separation_gap = (
    same_mean
    - different_mean
)


print()
print("==========================================")
print("       SPEAKER SIMILARITY")
print("==========================================")

print(
    f"Same-speaker similarity: "
    f"{same_mean:.4f}"
)

print(
    f"Same-speaker std: "
    f"{same_std:.4f}"
)

print(
    f"Different-speaker similarity: "
    f"{different_mean:.4f}"
)

print(
    f"Different-speaker std: "
    f"{different_std:.4f}"
)

print(
    f"Separation gap: "
    f"{separation_gap:.4f}"
)


# ============================================================
# PAIR ACCURACY
# ============================================================

# Threshold selected from the midpoint of
# same-speaker and different-speaker means.

threshold = (
    same_mean
    + different_mean
) / 2


pair_scores = np.concatenate(
    [
        same_scores,
        different_scores
    ]
)


pair_labels = np.concatenate(
    [
        np.ones(
            len(same_scores)
        ),

        np.zeros(
            len(different_scores)
        )
    ]
)


pair_predictions = (
    pair_scores >= threshold
).astype(int)


pair_accuracy = (
    accuracy_score(
        pair_labels,
        pair_predictions
    )
    * 100
)


print()
print("==========================================")
print("          PAIR EVALUATION")
print("==========================================")

print(
    f"Threshold: "
    f"{threshold:.4f}"
)

print(
    f"Pair Accuracy: "
    f"{pair_accuracy:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("==========================================")
print("       VOXSPLIT V6 FINAL EVALUATION")
print("==========================================")

print(
    f"Test Classification Accuracy : "
    f"{classification_accuracy:.2f}%"
)

print(
    f"CNN Embedding Silhouette     : "
    f"{silhouette:.4f}"
)

print(
    f"Same Speaker Similarity      : "
    f"{same_mean:.4f}"
)

print(
    f"Different Speaker Similarity : "
    f"{different_mean:.4f}"
)

print(
    f"Speaker Separation Gap       : "
    f"{separation_gap:.4f}"
)

print(
    f"Pair Accuracy                : "
    f"{pair_accuracy:.2f}%"
)

print()
print("==========================================")
print("             EVALUATION DONE")
print("==========================================")