import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader, Sampler

from speaker_embedding_v8 import SpeakerEmbeddingCNN


# ============================================================
# VOXSPLIT V8
# FROM-SCRATCH SUPERVISED SPEAKER EMBEDDINGS
# ============================================================


DATA_DIR = Path(
    "data/speaker_dataset/train"
)

RESULTS_DIR = Path(
    "results"
)

MODEL_OUTPUT = (
    RESULTS_DIR /
    "speaker_embedding_v8.pth"
)

HISTORY_OUTPUT = (
    RESULTS_DIR /
    "speaker_embedding_v8_history.txt"
)


# ============================================================
# AUDIO
# ============================================================

SAMPLE_RATE = 16000

AUDIO_DURATION = 4

TARGET_LENGTH = (
    SAMPLE_RATE *
    AUDIO_DURATION
)


# ============================================================
# MEL
# ============================================================

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256


# ============================================================
# SPEAKERS
# ============================================================

NUM_SPEAKERS = 30

EMBEDDING_DIM = 128


# ============================================================
# BATCH
# ============================================================

SPEAKERS_PER_BATCH = 8

SAMPLES_PER_SPEAKER = 2

BATCH_SIZE = (
    SPEAKERS_PER_BATCH *
    SAMPLES_PER_SPEAKER
)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 80

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.15

CONTRASTIVE_WEIGHT = 0.7

CLASSIFICATION_WEIGHT = 0.3


SEED = 42


# ============================================================
# SEED
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
print("        VOXSPLIT V8 TRAINING")
print("==========================================")
print()

print("Method:")
print("From-scratch CNN + Supervised Contrastive Learning")

print()

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MEL
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

        self.speaker_to_indices = {}


        speaker_dirs = sorted([

            d

            for d in self.root_dir.iterdir()

            if d.is_dir()

        ])


        print()

        print(
            "Speakers found:",
            len(speaker_dirs)
        )


        if len(speaker_dirs) != NUM_SPEAKERS:

            raise RuntimeError(

                f"Expected {NUM_SPEAKERS} speakers "
                f"but found {len(speaker_dirs)}"

            )


        for label, speaker_dir in enumerate(
            speaker_dirs
        ):

            wav_files = sorted(
                speaker_dir.glob("*.wav")
            )


            print(
                f"Speaker {label}: "
                f"{len(wav_files)} files"
            )


            if len(wav_files) < 2:

                raise RuntimeError(

                    f"{speaker_dir} has "
                    f"less than 2 files"

                )


            self.speaker_to_indices[label] = []


            for wav_file in wav_files:

                index = len(self.files)

                self.files.append(
                    wav_file
                )

                self.labels.append(
                    label
                )

                self.speaker_to_indices[
                    label
                ].append(index)


        print()

        print(
            "Total training files:",
            len(self.files)
        )


    def __len__(self):

        return len(self.files)


    def __getitem__(self, index):

        file_path = self.files[index]

        label = self.labels[index]


        waveform, sr = torchaudio.load(
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

        if sr != SAMPLE_RATE:

            waveform = torchaudio.functional.resample(

                waveform,

                sr,

                SAMPLE_RATE

            )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        maximum = waveform.abs().max()

        if maximum > 0:

            waveform = waveform / maximum


        # ----------------------------------------------------
        # LENGTH
        # ----------------------------------------------------

        length = waveform.shape[1]


        if length < TARGET_LENGTH:

            padding = (
                TARGET_LENGTH -
                length
            )

            waveform = F.pad(
                waveform,
                (0, padding)
            )


        elif length > TARGET_LENGTH:

            max_start = (
                length -
                TARGET_LENGTH
            )

            start = torch.randint(

                0,

                max_start + 1,

                (1,)

            ).item()


            waveform = waveform[
                :,
                start:
                start + TARGET_LENGTH
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
            mel + 1e-6
        )


        # ----------------------------------------------------
        # STANDARDIZE MEL
        # ----------------------------------------------------

        mean = mel.mean()

        std = mel.std()

        mel = (
            mel - mean
        ) / (
            std + 1e-6
        )


        return mel, label


# ============================================================
# BATCH SAMPLER
# ============================================================

class SpeakerBatchSampler(Sampler):

    def __init__(
        self,
        speaker_to_indices
    ):

        self.speaker_to_indices = (
            speaker_to_indices
        )

        self.speakers = list(
            speaker_to_indices.keys()
        )


    def __iter__(self):

        speakers = self.speakers.copy()

        random.shuffle(
            speakers
        )


        for start in range(

            0,

            len(speakers),

            SPEAKERS_PER_BATCH

        ):

            selected = speakers[
                start:
                start + SPEAKERS_PER_BATCH
            ]


            if len(selected) < SPEAKERS_PER_BATCH:

                continue


            batch = []


            for speaker in selected:

                indices = (
                    self.speaker_to_indices[
                        speaker
                    ]
                )


                chosen = random.sample(

                    indices,

                    SAMPLES_PER_SPEAKER

                )


                batch.extend(
                    chosen
                )


            random.shuffle(
                batch
            )


            yield batch


    def __len__(self):

        return (
            len(self.speakers)
            //
            SPEAKERS_PER_BATCH
        )


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

class SupConLoss(nn.Module):

    def __init__(
        self,
        temperature=0.15
    ):

        super().__init__()

        self.temperature = temperature


    def forward(
        self,
        embeddings,
        labels
    ):

        embeddings = F.normalize(
            embeddings,
            dim=1
        )


        similarity = torch.matmul(

            embeddings,

            embeddings.T

        )


        similarity = (
            similarity /
            self.temperature
        )


        batch_size = labels.size(0)


        self_mask = torch.eye(

            batch_size,

            device=labels.device,

            dtype=torch.bool

        )


        similarity = similarity.masked_fill(

            self_mask,

            -1e9

        )


        positive_mask = (

            labels.unsqueeze(0)
            ==
            labels.unsqueeze(1)

        )


        positive_mask = (

            positive_mask
            &
            ~self_mask

        )


        log_prob = F.log_softmax(

            similarity,

            dim=1

        )


        positive_count = positive_mask.sum(
            dim=1
        )


        mean_log_prob = (

            (
                positive_mask *
                log_prob
            ).sum(dim=1)

            /

            positive_count.clamp(
                min=1
            )

        )


        loss = -mean_log_prob.mean()


        return loss


# ============================================================
# DATA
# ============================================================

dataset = SpeakerDataset(
    DATA_DIR
)


batch_sampler = SpeakerBatchSampler(

    dataset.speaker_to_indices

)


loader = DataLoader(

    dataset,

    batch_sampler=batch_sampler,

    num_workers=0

)


print()

print(
    "Batches per epoch:",
    len(loader)
)

print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# MODEL
# ============================================================

model = SpeakerEmbeddingCNN(

    num_speakers=NUM_SPEAKERS,

    embedding_dim=EMBEDDING_DIM

).to(device)


print()

print(
    "CNN model created."
)


# ============================================================
# LOSSES
# ============================================================

supcon = SupConLoss(
    TEMPERATURE
)

cross_entropy = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS

)


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)


history = []

best_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

print()

print("==========================================")
print("          STARTING V8 TRAINING")
print("==========================================")


for epoch in range(EPOCHS):


    model.train()


    total_loss = 0.0

    total_supcon = 0.0

    total_ce = 0.0

    correct = 0

    total = 0


    for mel, labels in loader:


        mel = mel.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        # ------------------------------------------------
        # CNN
        # ------------------------------------------------

        features = model.features(
            mel
        )


        # ------------------------------------------------
        # POOL
        # ------------------------------------------------

        pooled = model.pool(
            features
        )


        pooled = torch.flatten(
            pooled,
            1
        )


        # ------------------------------------------------
        # EMBEDDING
        # ------------------------------------------------

        embeddings = model.embedding(
            pooled
        )


        # ------------------------------------------------
        # NORMALIZED EMBEDDING
        # ------------------------------------------------

        normalized = F.normalize(

            embeddings,

            p=2,

            dim=1

        )


        # ------------------------------------------------
        # CONTRASTIVE
        # ------------------------------------------------

        loss_supcon = supcon(

            normalized,

            labels

        )


        # ------------------------------------------------
        # CLASSIFICATION
        # ------------------------------------------------

        logits = model.classifier(
            normalized
        )


        loss_ce = cross_entropy(

            logits,

            labels

        )


        # ------------------------------------------------
        # TOTAL
        # ------------------------------------------------

        loss = (

            CONTRASTIVE_WEIGHT *
            loss_supcon

            +

            CLASSIFICATION_WEIGHT *
            loss_ce

        )


        loss.backward()


        # ------------------------------------------------
        # GRADIENT CLIPPING
        # ------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0

        )


        optimizer.step()


        # ------------------------------------------------
        # METRICS
        # ------------------------------------------------

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


    scheduler.step()


    # ====================================================
    # EPOCH
    # ====================================================

    avg_loss = (
        total_loss /
        len(loader)
    )


    avg_supcon = (
        total_supcon /
        len(loader)
    )


    avg_ce = (
        total_ce /
        len(loader)
    )


    accuracy = (
        100.0 *
        correct /
        total
    )


    lr = optimizer.param_groups[0]["lr"]


    result = (

        f"Epoch "
        f"{epoch + 1:03d}/{EPOCHS} | "

        f"Loss: "
        f"{avg_loss:.4f} | "

        f"SupCon: "
        f"{avg_supcon:.4f} | "

        f"CE: "
        f"{avg_ce:.4f} | "

        f"Acc: "
        f"{accuracy:.2f}% | "

        f"LR: "
        f"{lr:.7f}"

    )


    print()

    print(result)


    history.append(
        result
    )


    # ====================================================
    # SAVE BEST
    # ====================================================

    if avg_loss < best_loss:

        best_loss = avg_loss


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch + 1,

            "loss":
                avg_loss,

            "accuracy":
                accuracy,

            "embedding_dim":
                EMBEDDING_DIM,

            "num_speakers":
                NUM_SPEAKERS,

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

            "temperature":
                TEMPERATURE

        }


        torch.save(

            checkpoint,

            MODEL_OUTPUT

        )


        print(
            "  -> Best V8 model saved."
        )


# ============================================================
# HISTORY
# ============================================================

with open(

    HISTORY_OUTPUT,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "VOXSPLIT V8 TRAINING HISTORY\n"
    )

    f.write(
        "============================\n\n"
    )


    for line in history:

        f.write(
            line + "\n"
        )


    f.write("\n")

    f.write(
        f"Best loss: {best_loss:.6f}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("==========================================")
print("          V8 TRAINING COMPLETE")
print("==========================================")

print()

print(
    "Best loss:",
    best_loss
)

print()

print(
    "Model:",
    MODEL_OUTPUT
)

print()

print(
    "History:",
    HISTORY_OUTPUT
)

print()

print("==========================================")
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader, Sampler

from speaker_embedding_v8 import SpeakerEmbeddingCNN


# ============================================================
# VOXSPLIT V8
# FROM-SCRATCH SUPERVISED SPEAKER EMBEDDINGS
# ============================================================


DATA_DIR = Path(
    "data/speaker_dataset/train"
)

RESULTS_DIR = Path(
    "results"
)

MODEL_OUTPUT = (
    RESULTS_DIR /
    "speaker_embedding_v8.pth"
)

HISTORY_OUTPUT = (
    RESULTS_DIR /
    "speaker_embedding_v8_history.txt"
)


# ============================================================
# AUDIO
# ============================================================

SAMPLE_RATE = 16000

AUDIO_DURATION = 4

TARGET_LENGTH = (
    SAMPLE_RATE *
    AUDIO_DURATION
)


# ============================================================
# MEL
# ============================================================

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256


# ============================================================
# SPEAKERS
# ============================================================

NUM_SPEAKERS = 30

EMBEDDING_DIM = 128


# ============================================================
# BATCH
# ============================================================

SPEAKERS_PER_BATCH = 8

SAMPLES_PER_SPEAKER = 2

BATCH_SIZE = (
    SPEAKERS_PER_BATCH *
    SAMPLES_PER_SPEAKER
)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 80

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.15

CONTRASTIVE_WEIGHT = 0.7

CLASSIFICATION_WEIGHT = 0.3


SEED = 42


# ============================================================
# SEED
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
print("        VOXSPLIT V8 TRAINING")
print("==========================================")
print()

print("Method:")
print("From-scratch CNN + Supervised Contrastive Learning")

print()

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MEL
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

        self.speaker_to_indices = {}


        speaker_dirs = sorted([

            d

            for d in self.root_dir.iterdir()

            if d.is_dir()

        ])


        print()

        print(
            "Speakers found:",
            len(speaker_dirs)
        )


        if len(speaker_dirs) != NUM_SPEAKERS:

            raise RuntimeError(

                f"Expected {NUM_SPEAKERS} speakers "
                f"but found {len(speaker_dirs)}"

            )


        for label, speaker_dir in enumerate(
            speaker_dirs
        ):

            wav_files = sorted(
                speaker_dir.glob("*.wav")
            )


            print(
                f"Speaker {label}: "
                f"{len(wav_files)} files"
            )


            if len(wav_files) < 2:

                raise RuntimeError(

                    f"{speaker_dir} has "
                    f"less than 2 files"

                )


            self.speaker_to_indices[label] = []


            for wav_file in wav_files:

                index = len(self.files)

                self.files.append(
                    wav_file
                )

                self.labels.append(
                    label
                )

                self.speaker_to_indices[
                    label
                ].append(index)


        print()

        print(
            "Total training files:",
            len(self.files)
        )


    def __len__(self):

        return len(self.files)


    def __getitem__(self, index):

        file_path = self.files[index]

        label = self.labels[index]


        waveform, sr = torchaudio.load(
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

        if sr != SAMPLE_RATE:

            waveform = torchaudio.functional.resample(

                waveform,

                sr,

                SAMPLE_RATE

            )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        maximum = waveform.abs().max()

        if maximum > 0:

            waveform = waveform / maximum


        # ----------------------------------------------------
        # LENGTH
        # ----------------------------------------------------

        length = waveform.shape[1]


        if length < TARGET_LENGTH:

            padding = (
                TARGET_LENGTH -
                length
            )

            waveform = F.pad(
                waveform,
                (0, padding)
            )


        elif length > TARGET_LENGTH:

            max_start = (
                length -
                TARGET_LENGTH
            )

            start = torch.randint(

                0,

                max_start + 1,

                (1,)

            ).item()


            waveform = waveform[
                :,
                start:
                start + TARGET_LENGTH
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
            mel + 1e-6
        )


        # ----------------------------------------------------
        # STANDARDIZE MEL
        # ----------------------------------------------------

        mean = mel.mean()

        std = mel.std()

        mel = (
            mel - mean
        ) / (
            std + 1e-6
        )


        return mel, label


# ============================================================
# BATCH SAMPLER
# ============================================================

class SpeakerBatchSampler(Sampler):

    def __init__(
        self,
        speaker_to_indices
    ):

        self.speaker_to_indices = (
            speaker_to_indices
        )

        self.speakers = list(
            speaker_to_indices.keys()
        )


    def __iter__(self):

        speakers = self.speakers.copy()

        random.shuffle(
            speakers
        )


        for start in range(

            0,

            len(speakers),

            SPEAKERS_PER_BATCH

        ):

            selected = speakers[
                start:
                start + SPEAKERS_PER_BATCH
            ]


            if len(selected) < SPEAKERS_PER_BATCH:

                continue


            batch = []


            for speaker in selected:

                indices = (
                    self.speaker_to_indices[
                        speaker
                    ]
                )


                chosen = random.sample(

                    indices,

                    SAMPLES_PER_SPEAKER

                )


                batch.extend(
                    chosen
                )


            random.shuffle(
                batch
            )


            yield batch


    def __len__(self):

        return (
            len(self.speakers)
            //
            SPEAKERS_PER_BATCH
        )


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

class SupConLoss(nn.Module):

    def __init__(
        self,
        temperature=0.15
    ):

        super().__init__()

        self.temperature = temperature


    def forward(
        self,
        embeddings,
        labels
    ):

        embeddings = F.normalize(
            embeddings,
            dim=1
        )


        similarity = torch.matmul(

            embeddings,

            embeddings.T

        )


        similarity = (
            similarity /
            self.temperature
        )


        batch_size = labels.size(0)


        self_mask = torch.eye(

            batch_size,

            device=labels.device,

            dtype=torch.bool

        )


        similarity = similarity.masked_fill(

            self_mask,

            -1e9

        )


        positive_mask = (

            labels.unsqueeze(0)
            ==
            labels.unsqueeze(1)

        )


        positive_mask = (

            positive_mask
            &
            ~self_mask

        )


        log_prob = F.log_softmax(

            similarity,

            dim=1

        )


        positive_count = positive_mask.sum(
            dim=1
        )


        mean_log_prob = (

            (
                positive_mask *
                log_prob
            ).sum(dim=1)

            /

            positive_count.clamp(
                min=1
            )

        )


        loss = -mean_log_prob.mean()


        return loss


# ============================================================
# DATA
# ============================================================

dataset = SpeakerDataset(
    DATA_DIR
)


batch_sampler = SpeakerBatchSampler(

    dataset.speaker_to_indices

)


loader = DataLoader(

    dataset,

    batch_sampler=batch_sampler,

    num_workers=0

)


print()

print(
    "Batches per epoch:",
    len(loader)
)

print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# MODEL
# ============================================================

model = SpeakerEmbeddingCNN(

    num_speakers=NUM_SPEAKERS,

    embedding_dim=EMBEDDING_DIM

).to(device)


print()

print(
    "CNN model created."
)


# ============================================================
# LOSSES
# ============================================================

supcon = SupConLoss(
    TEMPERATURE
)

cross_entropy = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS

)


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)


history = []

best_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

print()

print("==========================================")
print("          STARTING V8 TRAINING")
print("==========================================")


for epoch in range(EPOCHS):


    model.train()


    total_loss = 0.0

    total_supcon = 0.0

    total_ce = 0.0

    correct = 0

    total = 0


    for mel, labels in loader:


        mel = mel.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        # ------------------------------------------------
        # CNN
        # ------------------------------------------------

        features = model.features(
            mel
        )


        # ------------------------------------------------
        # POOL
        # ------------------------------------------------

        pooled = model.pool(
            features
        )


        pooled = torch.flatten(
            pooled,
            1
        )


        # ------------------------------------------------
        # EMBEDDING
        # ------------------------------------------------

        embeddings = model.embedding(
            pooled
        )


        # ------------------------------------------------
        # NORMALIZED EMBEDDING
        # ------------------------------------------------

        normalized = F.normalize(

            embeddings,

            p=2,

            dim=1

        )


        # ------------------------------------------------
        # CONTRASTIVE
        # ------------------------------------------------

        loss_supcon = supcon(

            normalized,

            labels

        )


        # ------------------------------------------------
        # CLASSIFICATION
        # ------------------------------------------------

        logits = model.classifier(
            normalized
        )


        loss_ce = cross_entropy(

            logits,

            labels

        )


        # ------------------------------------------------
        # TOTAL
        # ------------------------------------------------

        loss = (

            CONTRASTIVE_WEIGHT *
            loss_supcon

            +

            CLASSIFICATION_WEIGHT *
            loss_ce

        )


        loss.backward()


        # ------------------------------------------------
        # GRADIENT CLIPPING
        # ------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0

        )


        optimizer.step()


        # ------------------------------------------------
        # METRICS
        # ------------------------------------------------

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


    scheduler.step()


    # ====================================================
    # EPOCH
    # ====================================================

    avg_loss = (
        total_loss /
        len(loader)
    )


    avg_supcon = (
        total_supcon /
        len(loader)
    )


    avg_ce = (
        total_ce /
        len(loader)
    )


    accuracy = (
        100.0 *
        correct /
        total
    )


    lr = optimizer.param_groups[0]["lr"]


    result = (

        f"Epoch "
        f"{epoch + 1:03d}/{EPOCHS} | "

        f"Loss: "
        f"{avg_loss:.4f} | "

        f"SupCon: "
        f"{avg_supcon:.4f} | "

        f"CE: "
        f"{avg_ce:.4f} | "

        f"Acc: "
        f"{accuracy:.2f}% | "

        f"LR: "
        f"{lr:.7f}"

    )


    print()

    print(result)


    history.append(
        result
    )


    # ====================================================
    # SAVE BEST
    # ====================================================

    if avg_loss < best_loss:

        best_loss = avg_loss


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch + 1,

            "loss":
                avg_loss,

            "accuracy":
                accuracy,

            "embedding_dim":
                EMBEDDING_DIM,

            "num_speakers":
                NUM_SPEAKERS,

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

            "temperature":
                TEMPERATURE

        }


        torch.save(

            checkpoint,

            MODEL_OUTPUT

        )


        print(
            "  -> Best V8 model saved."
        )


# ============================================================
# HISTORY
# ============================================================

with open(

    HISTORY_OUTPUT,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "VOXSPLIT V8 TRAINING HISTORY\n"
    )

    f.write(
        "============================\n\n"
    )


    for line in history:

        f.write(
            line + "\n"
        )


    f.write("\n")

    f.write(
        f"Best loss: {best_loss:.6f}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("==========================================")
print("          V8 TRAINING COMPLETE")
print("==========================================")

print()

print(
    "Best loss:",
    best_loss
)

print()

print(
    "Model:",
    MODEL_OUTPUT
)

print()

print(
    "History:",
    HISTORY_OUTPUT
)

print()

print("==========================================")