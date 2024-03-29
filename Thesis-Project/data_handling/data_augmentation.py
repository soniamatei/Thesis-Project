import json
from typing import Any

import config
import PIL.Image
import torch
from torchvision.transforms import v2

# TODO: see about temporal crop and frame sampling for adding augmentations in temporal space
# TODO: maybe save data every run aside from wandb


class VideoTransform:
    def __init__(
        self,
        dataset: str,
        json_file: str,
    ):
        """
        :param dataset: name of the dataset
        :param json_file: json file with settings for augmentations
        """
        self._aug_val = {}

        base_folder = config.ROOT_DIR / "data_handling"
        self._json_file = base_folder / json_file

        # check if json path is correct
        if not self._json_file.exists():
            raise ValueError("Json not found. Please check and try again.")

        # check if dataset exists in json
        with open(self._json_file, "r") as file:
            temporary_fetch = json.load(file)

            if dataset not in temporary_fetch.keys():
                raise ValueError("Dataset not found in json.")
            del temporary_fetch
        self._dataset = dataset

        self._load_augmentation_values()

        self._transforms = v2.Compose(
            [
                v2.Resize(size=self._aug_val["resize"]),
                v2.RandomHorizontalFlip(p=self._aug_val["horizontal_flip"]),
                v2.RandomVerticalFlip(p=self._aug_val["vertical_flip"]),
                v2.RandomRotation(degrees=self._aug_val["rotation"]),
                v2.ColorJitter(**self._aug_val["color_jitter_params"]),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean=self._aug_val["mean"], std=self._aug_val["std"]),
            ]
        )

    def __call__(self, frames: list[PIL.Image]) -> torch.Tensor:
        """
        Apply the augmentations with the same random torch seed for torch for all frames given.
        :param frames: multiple frames representing a single video
        :return: the transformed frames
        """
        # save the current state of torch
        state = torch.get_rng_state()

        transformed_frames = []
        for frame in frames:
            # reset state each time
            torch.set_rng_state(state)
            transformed_frame = self._transforms(frame)
            transformed_frames.append(transformed_frame)

        return torch.stack(transformed_frames)

    def json(self, load: bool = False) -> dict[Any, Any]:
        """
        Return the 'augmentation values dict.' plus an additional option of loading the settings again
        from the json file.
        :param load: flag for loading the settings again
        :return: 'augmentation values dict.'
        """
        if not isinstance(load, bool):
            raise ValueError("Please give 'load' a boolean value.")
        if load:
            self._load_augmentation_values()
        return self._aug_val

    def save(self, aug_values: dict[Any, Any], dump: bool = False):
        """
        Save the 'augmentation values dict.' given as parameter in the object plus an additional option
        of saving them in the json file. ValueError is raised if the dictionary doesn't respect the structure.
        :param aug_values: settings for the augmentations
        :param dump: flag for saving the settings in json file
        """
        if not self._check_structure(self._aug_val, aug_values):
            raise ValueError("Dictionary doesn't match the required structure.")
        if not isinstance(dump, bool):
            raise ValueError("Please give 'dump' a boolean value.")

        self._aug_val = aug_values
        if dump:
            self._dump_augmentation_values()

    def _load_augmentation_values(self):
        """
        Load the augmentation for the dataset from the json file.
        """
        with open(self._json_file, mode="r") as file:
            self._aug_val = json.load(file)[self._dataset]

    def _dump_augmentation_values(self):
        """
        Dump the augmentation values for the dataset to the json file.
        """
        with open(self._json_file, mode="r+") as file:
            temporary_fetch = json.load(file)
            temporary_fetch[self._dataset] = self._aug_val
            # go to start
            file.seek(0)
            # truncate file
            file.truncate()
            json.dump(obj=temporary_fetch, fp=file, indent=4)

    def _check_structure(
        self, self_object: dict[Any, Any], object: dict[Any, Any]
    ) -> bool:
        """
        Checks if the structure of 2 dictionaries are the same looking at keys only.
        Nesting search for nested dicts.
        :param self_object: dictionary object of the class
        :param object: outside object
        :return: True -> they match; False otherwise
        """
        if isinstance(object, dict) and isinstance(self_object, dict):
            # check if both dictionaries have the same keys
            if self_object.keys() != object.keys():
                return False

            # recursively check the structure of nested dictionaries
            for key in object:
                if not self._check_structure(self_object[key], object[key]):
                    return False
            return True

        else:
            return True
