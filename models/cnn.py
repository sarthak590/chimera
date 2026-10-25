"""Small CNN for 28x28 grayscale doodle classification."""
import torch
from torch import nn

from config.settings import IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES


class DoodleCNN(nn.Module):
    """Three conv blocks followed by a small classifier head.

    Shape walk-through for a 28x28 input:
        (1, 28, 28) -> conv block 1 -> (32, 14, 14)
                    -> conv block 2 -> (64, 7, 7)
                    -> conv block 3 -> (128, 3, 3)
                    -> flatten (1152) -> fc (128) -> logits (num_classes)

    The model returns raw logits (no softmax); use ``predict_proba`` or
    ``torch.softmax`` to get probabilities.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS,
                 dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            self._block(in_channels, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        feat_dim = self._feature_dim(in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def _feature_dim(self, in_channels: int) -> int:
        """Compute the flattened feature size with a dummy forward pass."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.features(torch.zeros(1, in_channels, IMAGE_SIZE, IMAGE_SIZE))
        self.train(was_training)
        return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities for a batch (switches to eval mode temporarily)."""
        was_training = self.training
        self.eval()
        probs = torch.softmax(self(x), dim=1)
        self.train(was_training)
        return probs


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
