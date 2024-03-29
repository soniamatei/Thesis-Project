from pathlib import Path

import config
import cv2
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data_augmentation import VideoTransform

# TODO: add required structure for dataset folder in readme
# TODO: decide final size for images time height width


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        transformations: VideoTransform = None,
        val: bool = False,
    ) -> None:
        """
        :param dataset: name of the folder containing the dataset of choice
        :param transformations: transformations you want to apply to the dataset
        :param val: flag for validation dataset
        """
        # get root project dir
        base_folder = config.ROOT_DIR / "datasets"

        if transformations is not None:
            self._transforms = transformations
        else:
            self._transforms = config.BASIC_TRANSFORMS

        # flag for using the val data
        self.flag = val

        self._fight_folder_path = base_folder / dataset / "Fight"
        self._non_fight_folder_path = base_folder / dataset / "NonFight"
        self._dataset = dataset

        if (
            not self._fight_folder_path.exists()
            or not self._non_fight_folder_path.exists()
        ):
            raise ValueError(
                "Paths to the folders of the dataset not found. Please check the path or structure of the folder."
            )

        self._train, self._val = self._split_data(fetch=True)

    def __len__(self) -> int:
        """
        :return: Number of videos in the dataset
        """
        return len(self._train if not self._flag else self._val)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Getter for an item in the dataset based on index.
        :param idx: index
        :return: the sample and the corresponding label
        """
        video_path, label, path = self._get_by_flag(idx)
        capture = cv2.VideoCapture(str(path / video_path))

        # capture frames
        frames = []
        while capture.isOpened():
            ret, frame = capture.read()

            if ret:
                # convert frame to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # convert to PIL image so transforms are applied
                frame = Image.fromarray(frame)
                frames.append(frame)

            else:
                capture.release()

        if isinstance(self._transforms, VideoTransform):
            frames = self._transforms(frames)
        else:
            frames = torch.stack([self._transforms(frame) for frame in frames])
        label = torch.tensor(label)

        return frames, label

    @property
    def flag(self) -> bool:
        return self._flag

    @flag.setter
    def flag(self, val: bool):
        if not isinstance(val, bool):
            raise ValueError("Please provide a boolean value for 'val' flag.")
        self._flag = val

    @property
    def dataset(self):
        return self._dataset

    def shuffle(self):
        """
        Shuffle the dataset (to reuse it in training for example).
        """
        self._train, self._val = self._split_data()

    def _get_by_flag(self, idx: int) -> tuple[str, bool, Path]:
        """
        Get an item based on the 'val' flag and label of the video.
        :param idx: index
        :return: (video name, corresponding label, path to parent folder)
        """
        if not isinstance(idx, int):
            raise ValueError("Please provide an integer value for 'idx'.")

        video_name, label = self._train[idx] if not self._flag else self._val[idx]
        path = self._fight_folder_path if label else self._non_fight_folder_path

        return video_name, label, path

    def _split_data(self, fetch: bool = False):
        """
        Split the dataset into train and validation sets.
        :param fetch: if set, fetches the data from the local memory
        """
        if fetch:
            # get sample names and their label (corresp. to the folder location)
            x, y = [], []
            for label, folder_path in [
                [1, self._fight_folder_path],
                [0, self._non_fight_folder_path],
            ]:
                for item in folder_path.iterdir():
                    x.append(item.parts[-1])
                    y.append(label)
        else:
            x, y = zip(*(self._train + self._val))

        # make a balanced split of the data
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.25, shuffle=True
        )
        return list(zip(x_train, y_train)), list(zip(x_val, y_val))
