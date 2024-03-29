import pathlib
from torchvision.transforms import v2
import torch

ROOT_DIR = pathlib.Path("/home/sonia/Documents/Licenta/Thesis-Project/Thesis-Project")
BASIC_TRANSFORMS = v2.Compose(
    [
        v2.Resize(size=(360, 360)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ]
)
