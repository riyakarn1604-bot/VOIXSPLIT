import torch
import torch.nn as nn


# ============================================================
# VOXSPLIT V6
# SPEAKER EMBEDDING NETWORK
# ============================================================


class SpeakerEmbeddingCNN(nn.Module):

    def __init__(
        self,
        num_speakers=30,
        embedding_dim=128
    ):

        super().__init__()


        # ----------------------------------------------------
        # CNN FEATURE EXTRACTOR
        # ----------------------------------------------------

        self.features = nn.Sequential(

            # Input:
            # [batch, 1, mel_bins, time]

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # ----------------------------------------------------
        # EMBEDDING LAYER
        # ----------------------------------------------------

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                embedding_dim
            ),

            nn.ReLU()
        )


        # ----------------------------------------------------
        # SPEAKER CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            embedding_dim,
            num_speakers
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        return_embedding=False
    ):

        x = self.features(x)

        embedding = self.embedding(x)

        logits = self.classifier(
            embedding
        )


        if return_embedding:

            return embedding

        return logits