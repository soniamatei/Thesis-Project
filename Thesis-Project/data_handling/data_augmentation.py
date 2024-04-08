import json

import PIL.Image
import jsonschema
import torch
from torchvision.transforms import v2

import config


# TODO: see about temporal crop and frame sampling for adding augmentations in temporal space
# TODO: when tuned, do get(.. []) from json dict


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
        self._json_schema = config.AUGMENTATION_VALUES_SCHEMA

        self._json_path = config.ROOT_DIR / "data_handling" / json_file
        if not self._json_path.exists():
            raise ValueError("No json file found at {}".format(self._json_path))

        # check if dataset exists in json
        with open(self._json_path, "r") as file:
            temporary_fetch = json.load(file)

            if dataset not in temporary_fetch.keys():
                raise ValueError("Dataset {} not found in json.".format(dataset))
            del temporary_fetch
        self._dataset = dataset

        self._load_augmentation_values()
        self._set_transforms()

    def __call__(self, frames: list[PIL.Image]) -> torch.Tensor:
        """
        Apply the augmentations with the same random torch seed for torch for all images given.
        :param frames: multiple frames representing a single video
        :return: the transformed frames [no. PIL images, channels, height, width]
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

    def json(self, load: bool = False) -> dict[any, any]:
        """
        Return the 'augmentation values dict.' plus an additional option of loading the settings again
        from the json file.
        :param load: flag for loading the settings from file
        :return: 'augmentation values dict.'
        """
        if not isinstance(load, bool):
            raise ValueError("Please give 'load' a boolean value.")
        if load:
            self._load_augmentation_values()
            self._set_transforms()
        return self._aug_val

    def save(self, aug_values: dict[any, any] = None, dump: bool = False):
        """
        Save the 'augmentation values dict.' given as parameter in the object json file an additional option
        of saving reloading the object. ValueError is raised if the dictionary doesn't respect the structure.
        :param aug_values: settings for the augmentations
        :param dump: flag for dumping the settings to json file
        """
        self._aug_val = aug_values
        self._set_transforms()

        if not isinstance(dump, bool):
            raise ValueError("Please give 'dump' a boolean value.")

        if dump:
            self._dump_augmentation_values(aug_values)

    def _set_transforms(self):
        """
        Set the transforms according to the json file.
        """
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

    def _load_augmentation_values(self):
        """
        Load the augmentation for the dataset from the json file.
        """
        with open(self._json_path, mode="r") as file:
            temporary_fetch = json.load(fp=file)[self._dataset]

            # no need for if, validate raises error
            jsonschema.validate(temporary_fetch, schema=self._json_schema)
            self._aug_val = temporary_fetch

    def _dump_augmentation_values(self, aug_values: dict[any, any]):
        """
        Dump the augmentation values for the dataset to the json file.
        :param aug_values: settings for the augmentations
        """
        jsonschema.validate(aug_values, schema=self._json_schema)

        with open(self._json_path, mode="r+") as file:
            temporary_fetch = json.load(file)
            temporary_fetch[self._dataset] = self._aug_val
            # go to start
            file.seek(0)
            # truncate file
            file.truncate()
            json.dump(obj=temporary_fetch, fp=file, indent=4)
