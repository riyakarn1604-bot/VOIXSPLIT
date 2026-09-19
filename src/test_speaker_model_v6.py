import torch

from speaker_embedding_v6 import SpeakerEmbeddingCNN


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# MODEL
# ============================================================

model = SpeakerEmbeddingCNN(
    num_speakers=30,
    embedding_dim=128
).to(device)


print()
print(model)


# ============================================================
# TEST INPUT
# ============================================================

# Simulated Mel Spectrogram
#
# batch = 4
# channels = 1
# mel bins = 128
# time frames = 200

x = torch.randn(
    4,
    1,
    128,
    200
).to(device)


# ============================================================
# FORWARD PASS
# ============================================================

logits = model(x)

embeddings = model(
    x,
    return_embedding=True
)


print()
print("Input shape:", x.shape)

print(
    "Logits shape:",
    logits.shape
)

print(
    "Embedding shape:",
    embeddings.shape
)