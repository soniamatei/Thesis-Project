import json

import jsonschema
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights

import config


# TODO: do default features after training with the best ones .get(.., ..)


class ViolenceDetectionModel(nn.Module):
    def __init__(self, json_file: str):
        super(ViolenceDetectionModel, self).__init__()
        self._settings = {}
        self._json_schema = config.MODEL_SETTINGS_SCHEMA

        self._json_path = config.ROOT_DIR / "model" / json_file
        if not self._json_path.exists():
            raise ValueError("No json file found at {}.".format(self._json_path))

        self._load_model_settings()

        self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features,
            out_features=self._settings["resnet_lstm_features"],
        )

        self.lstm = nn.LSTM(
            input_size=self._settings["resnet_lstm_features"],
            hidden_size=self._settings["lstm_hidden_size"],
            num_layers=self._settings["lstm_num_layers"],
            batch_first=True,
        )

        # TODO: test:
        #     - dropout/batch norm
        #     - reg layer before/after activations
        #     - reg layer places
        #     - gru/lstm
        #     - unfreezing params
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=self._settings["lstm_hidden_size"],
                out_features=self._settings["fc1_features"],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self._settings["fc1_features"]),
            nn.Linear(
                in_features=self._settings["fc1_features"],
                out_features=self._settings["fc2_features"],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self._settings["fc2_features"]),
            nn.Linear(
                in_features=self._settings["fc2_features"],
                out_features=self._settings["final_output_features"],
            ),
            nn.Sigmoid(),
        )

    def forward(self, videos: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, frames, channels, height, width = videos.size()

        # get features for each frame
        videos = videos.view(batch_size * frames, channels, height, width)
        videos = self.resnet(videos)
        videos = videos.view(batch_size, frames, -1)

        videos = nn.utils.rnn.pack_padded_sequence(
            videos, lengths, batch_first=True, enforce_sorted=False
        )
        videos, _ = self.lstm(videos)
        videos, _ = nn.utils.rnn.pad_packed_sequence(videos, batch_first=True)

        # take only last output from LSTM
        # [all videos from batch, last output, shape of output (hidden_size)]
        videos = self.fc(videos[:, -1, :]).squeeze(-1)

        return videos

    def _load_model_settings(self):
        with open(file=self._json_path, mode="r") as settings_file:
            temporary_fetch = json.load(fp=settings_file)

            jsonschema.validate(instance=temporary_fetch, schema=self._json_schema)
            self._settings = temporary_fetch

    def _dump_model_settings(self, settings: dict[any, any]):
        with open(file=self._json_path, mode="w") as settings_file:
            jsonschema.validate(instance=settings, schema=self._json_schema)
            json.dump(obj=settings, fp=settings_file)
