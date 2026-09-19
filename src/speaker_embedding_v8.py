import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEmbeddingCNN(nn.Module):

    def __init__(
        self,
        num_speakers=30,
        embedding_dim=128
    ):

        super().__init__()

        # ====================================================
        # CNN FEATURE EXTRACTOR
        # ====================================================

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # Block 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )


        # ====================================================
        # GLOBAL POOLING
        # ====================================================

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # ====================================================
        # EMBEDDING LAYER
        # ====================================================

        self.embedding = nn.Sequential(

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                embedding_dim
            )

        )


        # ====================================================
        # SPEAKER CLASSIFIER
        # ====================================================

        self.classifier = nn.Linear(

            embedding_dim,

            num_speakers

        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        embedding = self.embedding(x)

        normalized_embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        logits = self.classifier(
            normalized_embedding
        )

        return logits


    # ========================================================
    # GET EMBEDDING
    # ========================================================

    def get_embedding(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        embedding = self.embedding(x)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding