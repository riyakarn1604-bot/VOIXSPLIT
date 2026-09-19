import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# IMPORTANT:
# Change this ONLY if your audio files are somewhere else.
#
# The script searches recursively inside this directory.
# ------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results" / "speaker"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "speaker_embedding_v7.pth"
HISTORY_PATH = RESULTS_DIR / "speaker_embedding_v7_history.txt"


# Audio configuration
SAMPLE_RATE = 16000
AUDIO_SECONDS = 2.0
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_SECONDS)

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

EPOCHS = 80

BATCH_SPEAKERS = 6
SAMPLES_PER_SPEAKER = 4

BATCH_SIZE = BATCH_SPEAKERS * SAMPLES_PER_SPEAKER

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

# Relative weight of the two objectives
SUPCON_WEIGHT = 0.65
CE_WEIGHT = 0.35

VALIDATION_RATIO = 0.20

NUM_WORKERS = 0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("V7.1 IMPROVED SPEAKER EMBEDDING TRAINING")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# FIND AUDIO FILES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".WAV",
    ".flac",
    ".FLAC",
    ".mp3",
    ".MP3",
    ".ogg",
    ".OGG",
    ".m4a",
    ".M4A",
}


def find_audio_files(root):
    files = []

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{root}"
        )

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


# ============================================================
# SPEAKER LABEL EXTRACTION
# ============================================================

def get_speaker_name(path):
    """
    Assumes speaker identity is represented by the immediate
    parent folder.

    Example:

        data/audio/speaker_01/file1.wav
        data/audio/speaker_01/file2.wav

    speaker = speaker_01
    """

    return path.parent.name


# ============================================================
# LOAD DATASET INFORMATION
# ============================================================

all_files = find_audio_files(DATA_ROOT)

print(f"\nAudio files found: {len(all_files)}")


if len(all_files) == 0:
    raise RuntimeError(
        f"No supported audio files found inside:\n{DATA_ROOT}"
    )


speaker_files = defaultdict(list)

for file_path in all_files:
    speaker = get_speaker_name(file_path)
    speaker_files[speaker].append(file_path)


# Remove speakers with too few samples
MIN_SAMPLES_PER_SPEAKER = 4

speaker_files = {
    speaker: files
    for speaker, files in speaker_files.items()
    if len(files) >= MIN_SAMPLES_PER_SPEAKER
}


speakers = sorted(speaker_files.keys())


print(f"Speakers found: {len(speakers)}")

for speaker in speakers:
    print(
        f"  {speaker}: "
        f"{len(speaker_files[speaker])} files"
    )


if len(speakers) < 2:
    raise RuntimeError(
        "Need at least 2 speakers for speaker embedding training."
    )


# ============================================================
# SPEAKER -> INTEGER LABEL
# ============================================================

speaker_to_id = {
    speaker: idx
    for idx, speaker in enumerate(speakers)
}

id_to_speaker = {
    idx: speaker
    for speaker, idx in speaker_to_id.items()
}


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_items = []
val_items = []


for speaker in speakers:

    files = speaker_files[speaker].copy()

    random.shuffle(files)

    n_val = max(
        1,
        int(len(files) * VALIDATION_RATIO)
    )

    val_files = files[:n_val]
    train_files = files[n_val:]

    # Make sure the speaker has enough training examples
    if len(train_files) < SAMPLES_PER_SPEAKER:
        print(
            f"WARNING: {speaker} has only "
            f"{len(train_files)} training files."
        )

    speaker_id = speaker_to_id[speaker]

    for path in train_files:
        train_items.append(
            (path, speaker_id)
        )

    for path in val_files:
        val_items.append(
            (path, speaker_id)
        )


print("\nDataset split")
print("-" * 60)
print(f"Training files:   {len(train_items)}")
print(f"Validation files: {len(val_items)}")


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def load_audio(path):

    try:

        audio, sr = librosa.load(
            str(path),
            sr=SAMPLE_RATE,
            mono=True
        )

    except Exception as e:

        raise RuntimeError(
            f"\nCould not load audio:\n{path}\n"
            f"Error: {e}"
        )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value > 1e-8:
        audio = audio / max_value

    # Fix length
    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )

    else:

        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


def audio_to_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Per-sample normalization
    mean = mel.mean()
    std = mel.std() + 1e-6

    mel = (mel - mean) / std

    return mel.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class SpeakerDataset(Dataset):

    def __init__(self, items, augment=False):

        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        path, label = self.items[index]

        audio = load_audio(path)

        # ----------------------------------------------------
        # Lightweight augmentation
        # ----------------------------------------------------

        if self.augment:

            # Random gain
            if random.random() < 0.5:

                gain = random.uniform(
                    0.85,
                    1.15
                )

                audio = audio * gain

            # Small Gaussian noise
            if random.random() < 0.4:

                noise = np.random.normal(
                    0,
                    0.003,
                    size=audio.shape
                ).astype(np.float32)

                audio = audio + noise

            # Time shift
            if random.random() < 0.3:

                shift = random.randint(
                    -int(0.05 * SAMPLE_RATE),
                    int(0.05 * SAMPLE_RATE)
                )

                audio = np.roll(
                    audio,
                    shift
                )

        mel = audio_to_mel(audio)

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # CNN expects [channel, height, width]
        mel = mel.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel, label


# ============================================================
# BALANCED BATCH SAMPLER
# ============================================================

class BalancedSpeakerSampler(Sampler):

    """
    Creates batches containing:

        BATCH_SPEAKERS speakers
        x
        SAMPLES_PER_SPEAKER samples

    Example:

        6 speakers x 4 samples
        = 24 samples/batch

    This is important for Supervised Contrastive Learning
    because every anchor needs positive samples from the
    same speaker inside the batch.
    """

    def __init__(
        self,
        items,
        speakers_per_batch,
        samples_per_speaker
    ):

        self.items = items

        self.P = speakers_per_batch
        self.K = samples_per_speaker

        self.by_speaker = defaultdict(list)

        for idx, (_, label) in enumerate(items):

            self.by_speaker[label].append(idx)

        self.speaker_ids = sorted(
            self.by_speaker.keys()
        )

    def __iter__(self):

        # Shuffle samples for each speaker
        speaker_indices = {}

        for speaker in self.speaker_ids:

            indices = self.by_speaker[speaker].copy()

            random.shuffle(indices)

            speaker_indices[speaker] = indices

        # ----------------------------------------------------
        # Each speaker gets chunks of K samples
        # ----------------------------------------------------

        speaker_chunks = {}

        for speaker in self.speaker_ids:

            indices = speaker_indices[speaker]

            chunks = []

            for i in range(
                0,
                len(indices),
                self.K
            ):

                chunk = indices[
                    i:i + self.K
                ]

                # If last chunk is short,
                # sample additional examples
                # with replacement.
                while len(chunk) < self.K:

                    chunk.append(
                        random.choice(indices)
                    )

                chunks.append(chunk)

            speaker_chunks[speaker] = chunks

        # ----------------------------------------------------
        # Build batches
        # ----------------------------------------------------

        active_speakers = self.speaker_ids.copy()

        random.shuffle(active_speakers)

        max_chunks = max(
            len(chunks)
            for chunks in speaker_chunks.values()
        )

        for chunk_idx in range(max_chunks):

            available = [
                speaker
                for speaker in active_speakers
                if chunk_idx < len(
                    speaker_chunks[speaker]
                )
            ]

            if len(available) < self.P:
                break

            random.shuffle(available)

            selected = available[:self.P]

            batch = []

            for speaker in selected:

                batch.extend(
                    speaker_chunks[speaker][chunk_idx]
                )

            random.shuffle(batch)

            yield batch

    def __len__(self):

        if len(self.speaker_ids) < self.P:
            return 0

        max_chunks = max(
            int(
                np.ceil(
                    len(
                        self.by_speaker[speaker]
                    ) / self.K
                )
            )
            for speaker in self.speaker_ids
        )

        return max_chunks


# ============================================================
# DATASETS
# ============================================================

train_dataset = SpeakerDataset(
    train_items,
    augment=True
)

val_dataset = SpeakerDataset(
    val_items,
    augment=False
)


# ============================================================
# DATALOADERS
# ============================================================

train_sampler = BalancedSpeakerSampler(
    train_items,
    speakers_per_batch=BATCH_SPEAKERS,
    samples_per_speaker=SAMPLES_PER_SPEAKER
)


train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\nBatch configuration")
print("-" * 60)
print(f"Speakers per batch: {BATCH_SPEAKERS}")
print(f"Samples per speaker: {SAMPLES_PER_SPEAKER}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Batches per epoch: {len(train_loader)}")


# ============================================================
# CNN SPEAKER EMBEDDING MODEL
# ============================================================

class SpeakerEmbeddingCNN(nn.Module):

    def __init__(
        self,
        num_speakers,
        embedding_dim=128
    ):

        super().__init__()

        # ----------------------------------------------------
        # CNN feature extractor
        # ----------------------------------------------------

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # ------------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # ------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            # Adaptive pooling removes dependency
            # on exact mel/time dimensions.
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        # ----------------------------------------------------
        # Embedding layer
        # ----------------------------------------------------

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.25),

            nn.Linear(
                256,
                embedding_dim
            )
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            embedding_dim,
            num_speakers
        )

    def forward(self, x):

        x = self.features(x)

        embedding = self.embedding(x)

        # L2 normalized embedding
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        logits = self.classifier(
            embedding
        )

        return embedding, logits


model = SpeakerEmbeddingCNN(
    num_speakers=len(speakers),
    embedding_dim=128
).to(DEVICE)


print("\nModel")
print("-" * 60)
print(model)


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

class SupConLoss(nn.Module):

    def __init__(
        self,
        temperature=0.07
    ):

        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        features,
        labels
    ):

        """
        features:
            [batch, embedding_dim]

        labels:
            [batch]
        """

        device = features.device

        features = F.normalize(
            features,
            dim=1
        )

        similarity = torch.matmul(
            features,
            features.T
        ) / self.temperature

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(
            labels,
            labels.T
        ).float().to(device)

        # Remove self-comparisons
        logits_mask = torch.ones_like(mask)

        logits_mask.fill_diagonal_(0)

        mask = mask * logits_mask

        # Numerical stability
        logits = similarity - similarity.max(
            dim=1,
            keepdim=True
        ).values.detach()

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = (
            logits -
            torch.log(
                exp_logits.sum(
                    dim=1,
                    keepdim=True
                ) + 1e-8
            )
        )

        positive_count = mask.sum(
            dim=1
        )

        mean_log_prob_pos = (
            (mask * log_prob).sum(dim=1)
            /
            (positive_count + 1e-8)
        )

        loss = -mean_log_prob_pos.mean()

        return loss


supcon_loss_fn = SupConLoss(
    temperature=TEMPERATURE
)

ce_loss_fn = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_supcon = 0.0
    total_ce = 0.0

    correct = 0
    total = 0

    batches = 0

    for mel, labels in train_loader:

        mel = mel.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        embeddings, logits = model(
            mel
        )

        loss_supcon = supcon_loss_fn(
            embeddings,
            labels
        )

        loss_ce = ce_loss_fn(
            logits,
            labels
        )

        loss = (
            SUPCON_WEIGHT * loss_supcon
            +
            CE_WEIGHT * loss_ce
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_supcon += (
            loss_supcon.item()
        )

        total_ce += (
            loss_ce.item()
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        batches += 1

    return (
        total_loss / max(batches, 1),
        total_supcon / max(batches, 1),
        total_ce / max(batches, 1),
        100.0 * correct / max(total, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    correct = 0
    total = 0

    total_loss = 0.0
    batches = 0

    embeddings_all = []
    labels_all = []

    for mel, labels in val_loader:

        mel = mel.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        embeddings, logits = model(
            mel
        )

        loss = ce_loss_fn(
            logits,
            labels
        )

        total_loss += loss.item()

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        embeddings_all.append(
            embeddings.cpu()
        )

        labels_all.append(
            labels.cpu()
        )

        batches += 1

    accuracy = (
        100.0 * correct / max(total, 1)
    )

    avg_loss = (
        total_loss / max(batches, 1)
    )

    embeddings_all = torch.cat(
        embeddings_all,
        dim=0
    )

    labels_all = torch.cat(
        labels_all,
        dim=0
    )

    return (
        avg_loss,
        accuracy,
        embeddings_all,
        labels_all
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_checkpoint(
    epoch,
    val_accuracy,
    val_loss
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "val_accuracy":
            val_accuracy,

        "val_loss":
            val_loss,

        "speaker_to_id":
            speaker_to_id,

        "id_to_speaker":
            id_to_speaker,

        "embedding_dim":
            128,

        "sample_rate":
            SAMPLE_RATE,

        "n_mels":
            N_MELS
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history = []

best_val_accuracy = -1.0

best_val_loss = float("inf")

patience = 15

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(1, EPOCHS + 1):

    train_loss, train_supcon, train_ce, train_acc = (
        train_one_epoch()
    )

    (
        val_loss,
        val_acc,
        _,
        _
    ) = validate()

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_line = (
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"SupCon: {train_supcon:.4f} | "
        f"CE: {train_ce:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}% | "
        f"LR: {current_lr:.7f}"
    )

    print(history_line)

    history.append(
        history_line
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    improved = False

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc
        best_val_loss = val_loss

        epochs_without_improvement = 0

        save_checkpoint(
            epoch,
            val_acc,
            val_loss
        )

        improved = True

    else:

        epochs_without_improvement += 1

    if improved:

        print(
            "  -> BEST MODEL SAVED"
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= patience:

        print(
            f"\nEarly stopping triggered after "
            f"{patience} epochs without validation improvement."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(history)
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("V7.1 TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"\nModel saved to:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nHistory saved to:"
    f"\n{HISTORY_PATH}"
)

print("=" * 60)