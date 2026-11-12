"""The MLPs and the small CNN."""
from torch import nn

from image_processing.features import FEATURE_SIZE

IMAGE_SIZE = 32
CLASSES = 14
CLASSES_WITH_LIGHTS = 17


class MLP(nn.Module):
    def __init__(self, inputs, classes=CLASSES, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers = [nn.Flatten()]
        size = inputs
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
            size = width
        layers.append(nn.Linear(size, classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SmallCNN(nn.Module):
    def __init__(self, classes=CLASSES):
        super().__init__()

        def block(channels_in, channels_out):
            return [
                nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(),
                nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]

        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, classes=CLASSES):
    if name == "mlp":
        return MLP(IMAGE_SIZE * IMAGE_SIZE * 3, classes)
    if name == "feature_mlp":
        return MLP(FEATURE_SIZE, classes)
    if name == "cnn":
        return SmallCNN(classes)
    raise ValueError(f"Unknown model: {name}")


def uses_features(name):
    return name == "feature_mlp"
