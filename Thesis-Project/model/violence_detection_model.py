import json

import jsonschema
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
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

        vit_config = ViTConfig(hidden_size=self._settings["vit_hidden_size"])
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config=vit_config)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.transformer_encoder = TransformerModel(
            d_model=self._settings["vit_hidden_size"],
            nhead=self._settings["nhead"],
            nlayers=self._settings["num_layers"],
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=self._settings["vit_hidden_size"],
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
        videos = self.vit(videos).last_hidden_state[:, 0, :]  # Use [CLS] token representation
        videos = videos.view(batch_size, frames, -1)

        videos = videos.permute(1, 0, 2)  # (frames, batch_size, feature_dim)
        # Transformer Encoder for temporal modeling
        videos = self.transformer_encoder(videos, lengths)

        videos = videos.permute(1, 0, 2) # (batch_size, frames, feature_dim)

        # take the mean output from the encoder
        videos = videos.mean(dim=1)
        videos = self.fc(videos).squeeze(-1)

        return videos

    def _load_model_settings(self):
        with open(file=self._json_path, mode="r") as settings_file:
            temporary_fetch = json.load(fp=settings_file)

            jsonschema.validate(instance=temporary_fetch, schema=self._json_schema)
            self._settings = temporary_fetch

    # def _dump_model_settings(self, settings: dict[any, any]):
    #     with open(file=self._json_path, mode="w") as settings_file:
    #         jsonschema.validate(instance=settings, schema=self._json_schema)
    #         json.dump(obj=settings, fp=settings_file)


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, d_model]``
            lengths: Tensor, shape ``[batch_size]`` containing the lengths of the sequences

        Returns:
            output Tensor of shape ``[seq_len, batch_size, d_model]``
        """
        # create padding mask
        mask = self.create_padding_mask(src, lengths).to(src.get_device())

        # apply positional encoding
        src = self.pos_encoder(src)

        # apply transformer encoder with mask
        output = self.transformer_encoder(src=src, src_key_padding_mask=mask)
        return output

    def create_padding_mask(self, src: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Creates a padding mask for the transformer encoder.

        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, d_model]``
            lengths: Tensor, shape ``[batch_size]`` containing the lengths of the sequences

        Returns:
            mask: Tensor, shape ``[batch_size, seq_len]``
        """
        seq_len, batch_size, _ = src.size()
        mask = torch.full(size=(batch_size, seq_len), fill_value=False)

        for i, length in enumerate(lengths):
            mask[i, length:] = True

        return mask
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)