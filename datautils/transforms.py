from __future__ import annotations
import torchvision.transforms as T

default_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRANSFORM_REGISTRY = {
    "default": default_transform,
}
