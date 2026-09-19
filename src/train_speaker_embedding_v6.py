import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import Dataset, DataLoader

from speaker_embedding_v6 import SpeakerEmbeddingCNN


# ============================================================
# VOXSPLIT V6
# FROM-SCRATCH SPEAKER EMBEDDING TRAINING
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(
    "data/speaker_dataset"
)

RESULTS_DIR = Path(
    "results"
)

MODEL_OUTPUT = (
    RESULTS_DIR
    / "speaker_embedding_v6.pth"
)

HISTORY_OUTPUT = (
    RESULTS_DIR
    / "speaker_embedding_v6_history.txt"
)


# Audio
SAMPLE_RATE = 16000

AUDIO_DURATION = 4

TARGET_AUDIO_LENGTH = (
    SAMPLE_RATE * AUDIO_DURATION
)


# Mel spectrogram
N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256


# Training
NUM_SPEAKERS = 30

EMBEDDING_DIM = 128

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 0.001


# Reproducibility
SEED = 42


# ============================================================
# RANDOM SEEDS
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


# ============================================================
# HEADER
# ============================================================

print()

print("==========================================")

print("       VOXSPLIT V6 TRAINING")

print("==========================================")


print()

print(
    "Device:",
    device
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version:",
        torch.version.cuda
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

class SpeakerDataset(Dataset):

    def __init__(
        self,
        root_dir,
        training=False
    ):

        self.root_dir = Path(
            root_dir
        )

        self.training = training

        self.files = []

        self.labels = []


        # ----------------------------------------------------
        # FIND SPEAKER DIRECTORIES
        # ----------------------------------------------------

        speaker_dirs = sorted(

            [

                directory

                for directory
                in self.root_dir.iterdir()

                if directory.is_dir()

            ]

        )


        if len(speaker_dirs) != NUM_SPEAKERS:

            raise RuntimeError(

                f"Expected "
                f"{NUM_SPEAKERS} speakers, "
                f"but found "
                f"{len(speaker_dirs)} "
                f"in {self.root_dir}"

            )


        # ----------------------------------------------------
        # LOAD FILE LIST
        # ----------------------------------------------------

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


        print()

        print(

            f"{self.root_dir}: "
            f"{len(self.files)} files"

        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.files
        )


    # ========================================================
    # GET ITEM
    # ========================================================

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
        # LOAD AUDIO
        # ----------------------------------------------------

        waveform, sample_rate = (

            torchaudio.load(
                file_path
            )

        )


        # ----------------------------------------------------
        # CONVERT TO MONO
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

        max_amplitude = (

            waveform.abs().max()

        )


        if max_amplitude > 0:

            waveform = (

                waveform
                / max_amplitude

            )


        # ----------------------------------------------------
        # FIXED LENGTH
        # ----------------------------------------------------

        current_length = (

            waveform.shape[1]

        )


        # ----------------------------------------------------
        # SHORT AUDIO
        # ----------------------------------------------------

        if current_length < TARGET_AUDIO_LENGTH:

            padding = (

                TARGET_AUDIO_LENGTH
                - current_length

            )


            waveform = (

                torch.nn.functional.pad(

                    waveform,

                    (0, padding)

                )

            )


        # ----------------------------------------------------
        # LONG AUDIO
        # ----------------------------------------------------

        elif current_length > TARGET_AUDIO_LENGTH:

            max_start = (

                current_length
                - TARGET_AUDIO_LENGTH

            )


            if self.training:

                # Random crop

                start = torch.randint(

                    0,

                    max_start + 1,

                    (1,)

                ).item()

            else:

                # Center crop for validation

                start = (

                    max_start // 2

                )


            waveform = (

                waveform[
                    :,
                    start:
                    start + TARGET_AUDIO_LENGTH
                ]

            )


        # ----------------------------------------------------
        # MEL SPECTROGRAM
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


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return mel, label


# ============================================================
# CREATE DATASETS
# ============================================================

print()

print("Loading datasets...")


train_dataset = SpeakerDataset(

    DATA_DIR / "train",

    training=True

)


val_dataset = SpeakerDataset(

    DATA_DIR / "val",

    training=False

)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    drop_last=True

)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ============================================================
# CHECK ONE BATCH
# ============================================================

print()

print("Checking training batch...")


sample_mel, sample_labels = next(

    iter(train_loader)

)


print(
    "Mel batch shape:",
    sample_mel.shape
)


print(
    "Label batch shape:",
    sample_labels.shape
)


# Expected:

# [batch, 1, 128, time]


# ============================================================
# MODEL
# ============================================================

print()

print("Creating neural network...")


model = SpeakerEmbeddingCNN(

    num_speakers=NUM_SPEAKERS,

    embedding_dim=EMBEDDING_DIM

).to(device)


# ============================================================
# LOSS
# ============================================================

criterion = (

    nn.CrossEntropyLoss()

)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = []


best_val_accuracy = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

print()

print("==========================================")

print("           STARTING TRAINING")

print("==========================================")


for epoch in range(
    EPOCHS
):


    # ========================================================
    # TRAINING MODE
    # ========================================================

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for mel, labels in train_loader:


        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        mel = mel.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        outputs = model(
            mel
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(

            outputs,

            labels

        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # UPDATE WEIGHTS
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        running_loss += (

            loss.item()

        )


        predictions = (

            outputs.argmax(
                dim=1
            )

        )


        correct += (

            (predictions == labels)
            .sum()
            .item()

        )


        total += (

            labels.size(0)

        )


    # ========================================================
    # TRAINING METRICS
    # ========================================================

    train_loss = (

        running_loss
        / len(train_loader)

    )


    train_accuracy = (

        100.0
        * correct
        / total

    )


    # ========================================================
    # VALIDATION MODE
    # ========================================================

    model.eval()


    val_loss_total = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():


        for mel, labels in val_loader:


            mel = mel.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                mel
            )


            loss = criterion(

                outputs,

                labels

            )


            val_loss_total += (

                loss.item()

            )


            predictions = (

                outputs.argmax(
                    dim=1
                )

            )


            val_correct += (

                (predictions == labels)
                .sum()
                .item()

            )


            val_total += (

                labels.size(0)

            )


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    val_loss = (

        val_loss_total
        / len(val_loader)

    )


    val_accuracy = (

        100.0
        * val_correct
        / val_total

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    result = (

        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} | "

        f"Train Loss: "
        f"{train_loss:.4f} | "

        f"Train Acc: "
        f"{train_accuracy:.2f}% | "

        f"Val Loss: "
        f"{val_loss:.4f} | "

        f"Val Acc: "
        f"{val_accuracy:.2f}%"

    )


    print()

    print(result)


    history.append(
        result
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (

            val_accuracy

        )


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "num_speakers":
                NUM_SPEAKERS,

            "embedding_dim":
                EMBEDDING_DIM,

            "sample_rate":
                SAMPLE_RATE,

            "audio_duration":
                AUDIO_DURATION,

            "n_mels":
                N_MELS,

            "n_fft":
                N_FFT,

            "hop_length":
                HOP_LENGTH,

            "seed":
                SEED

        }


        torch.save(

            checkpoint,

            MODEL_OUTPUT

        )


        print(
            "  -> Best model saved."
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(

    HISTORY_OUTPUT,

    "w",

    encoding="utf-8"

) as file:


    file.write(

        "VOXSPLIT V6 TRAINING HISTORY\n"

    )

    file.write(

        "============================\n\n"

    )


    for line in history:

        file.write(
            line + "\n"
        )


    file.write("\n")

    file.write(

        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%\n"

    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("==========================================")

print("          TRAINING COMPLETE")

print("==========================================")


print()

print(

    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"

)


print()

print(
    "Model saved to:"
)

print(
    MODEL_OUTPUT
)


print()

print(
    "Training history saved to:"
)

print(
    HISTORY_OUTPUT
)


print()

print("==========================================")