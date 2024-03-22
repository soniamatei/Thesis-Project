import pathlib
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split


# TODO: add required structure for dataset folder in readme
# TODO: load something to see if labels are right


class CustomDataset(Dataset):
    def __init__(
            self,
            dataset: str,
            transformations: transforms.Compose = None,
            val: bool = False,
    ) -> None:
        """
        :param dataset: name of the folder containing the dataset of choice
        :param transformations: transformations you want to apply to the dataset
        :param val: flag for validation dataset
        """
        # get root project dir
        base_folder = pathlib.Path(__file__).parent.resolve() / "datasets"
        self._transforms = transformations

        if not isinstance(val, bool):
            raise ValueError("Please provide a good value for 'val' flag.")

        # flag for using the val data
        self._flag = val

        self._fight_folder_path = base_folder / dataset / "Fight"
        self._non_fight_folder_path = base_folder / dataset / "NonFight"

        if not self._fight_folder_path.exists() or not self._non_fight_folder_path.exists():
            raise ValueError("Can't find path to dataset folders. Please check the name or structure of the folder.")

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
        if not self._flag:
            video_path, label = self._train[idx]
        else:
            video_path, label = self._val[idx]

        if label:
            capture = cv2.VideoCapture(str(self._fight_folder_path / video_path))
        else:
            capture = cv2.VideoCapture(str(self._non_fight_folder_path / video_path))

        # capture frames
        frames = []
        while capture.isOpened():
            ret, frame = capture.read()

            if ret:
                # convert frame to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # convert to PIL image so transforms are applied
                frame = Image.fromarray(frame)
                if self._transforms:
                    frame = self._transforms(frame)

                frames.append(frame)
            else:
                capture.release()

        # make a tensor out of frames list
        frames = torch.stack(frames)
        label = torch.tensor(label)

        return frames, label

    @property
    def flag(self) -> bool:
        return self._flag

    @flag.setter
    def flag(self, val: bool):
        if not isinstance(val, bool):
            raise ValueError("Please provide a good value for 'val' flag.")

        self._flag = val

    def shuffle(self):
        """
        Shuffle the dataset (to reuse it in training for example).
        """
        self._train, self._val = self._split_data()

    def _split_data(self, fetch: bool = False):
        """
        Split the dataset into train and validation sets.
        :param fetch: if set, fetches the data from the local memory
        """
        if fetch:
            # get sample names and their label (corresp. to the folder location)
            x, y = [], []
            for label, folder_path in [
                [True, self._fight_folder_path],
                [False, self._non_fight_folder_path],
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


if __name__ == "__main__":
    transform = v2.Compose([v2.Resize((225, 225)), v2.ToTensor()])

    dataset = CustomDataset(dataset="Fight-Surveillance", transformations=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # test the videos and their label to match
    for i, batch in enumerate(data_loader):
        video, label = batch
        if label:
            assert dataset._fight_folder_path.joinpath(dataset._train[i][0]).exists()
        else:
            assert dataset._non_fight_folder_path.joinpath(dataset._train[i][0]).exists()
        print(i)

    dataset.flag = True

    for i, batch in enumerate(data_loader):
        video, label = batch
        if label:
            assert dataset._fight_folder_path.joinpath(dataset._val[i][0]).exists()
        else:
            assert dataset._non_fight_folder_path.joinpath(dataset._val[i][0]).exists()
        print(i)
