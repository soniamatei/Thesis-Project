import pathlib

import torch
from torchvision.transforms import v2

# ROOT_DIR = pathlib.Path("/home/sonia/Documents/Licenta/Thesis-Project/Thesis-Project")
# ROOT_DIR = pathlib.Path("C:\\Users\\sonia\\Desktop\\Thesis-Project\\Thesis-Project")
ROOT_DIR = pathlib.Path("/home/sonia2oo2soia/projects/Thesis-Project/Thesis-Project")
BASIC_TRANSFORMS = v2.Compose(
    [
        v2.Resize(size=(360, 360)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ]
)

MODEL_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "vit_hidden_size": {"type": "integer"},
        "nhead": {"type": "integer"},
        "num_layers": {"type": "integer"},
        "fc1_features": {"type": "integer"},
        "fc2_features": {"type": "integer"},
        "final_output_features": {"type": "integer"},
    },
    "required": [
        "vit_hidden_size",
        "nhead",
        "num_layers",
        "fc1_features",
        "fc2_features",
        "final_output_features",
    ],
}

AUGMENTATION_VALUES_SCHEMA = {
    "type": "object",
    "properties": {
        "resize": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "horizontal_flip": {
            "type": "number",
        },
        "vertical_flip": {
            "type": "number",
        },
        "rotation": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "color_jitter_params": {
            "type": "object",
            "properties": {
                "brightness": {
                    "type": "number",
                },
                "contrast": {
                    "type": "number",
                },
                "saturation": {
                    "type": "number",
                },
                "hue": {
                    "type": "number",
                },
            },
            "required": ["brightness", "contrast", "saturation", "hue"],
        },
        "mean": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
        "std": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": [
        "resize",
        "horizontal_flip",
        "vertical_flip",
        "rotation",
        "color_jitter_params",
        "mean",
        "std",
    ],
}
