import io

import numpy as np
import torch
import matplotlib.pyplot as plt
from data_handling.video_dataset import VideoDataset


def collate_fn_pad(batch: list[tuple[torch.Tensor]]) -> tuple[torch.Tensor]:
    """
    Pads batches of variable length. Dataloader helper function.
    :param batch: list of returned tuples from dataset as (frames, label)
    :return: - the padded videos [frames, channels, height, width]
             - the corresponding labels [label]
             - the lengths of each sequence in the batch (no. frames) [length]
    """
    # separate videos and labels
    videos, labels = zip(*batch)

    # get number of frames
    lengths = torch.tensor([t.size(0) for t in videos], dtype=torch.int)

    # pad after max length tensor
    videos = torch.nn.utils.rnn.pad_sequence(videos, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)

    return videos, labels, lengths


def compute_mean_std(
    dataset: VideoDataset, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and std for a dataset.
    :param dataset: the desired dataset
    :param device: option for calc. for optimization. defaults to 'cpu'
    :return: the calculated mean and standard deviation [channel_1, channel_2, channel_3]
    """
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"Device should be 'cpu' or 'cuda', got {device}.")

    channel_sums = torch.zeros(3, device=device)
    channel_sums_squared = torch.zeros(3, device=device)
    pixel_count = 0

    for i in range(2):
        dataset.flag = not dataset.flag
        for batch in dataset:
            video, _ = batch
            video = video.to(device)
            # make [channels, frames, height, width]
            video = video.permute(1, 0, 2, 3)

            batch_pixel_count = video.size(1) * video.size(2) * video.size(3)
            pixel_count += batch_pixel_count

            channel_sums += video.sum(dim=[1, 2, 3])
            channel_sums_squared += (video**2).sum(dim=[1, 2, 3])

            if device == "cuda":
                del video
                torch.cuda.empty_cache()

    mean = channel_sums / pixel_count
    variance = (channel_sums_squared / pixel_count) - (mean**2)
    std = torch.sqrt(variance)

    return mean, std


def plot_to_image_buff(x: np.array, label: str = "") -> io.BytesIO:
    fig, ax = plt.subplots()
    ax.plot(x, label=label)
    ax.legend()
    # create a buffer
    img_buf = io.BytesIO()
    # save the figure to the buffer
    fig.savefig(img_buf, format="png")
    # rewind the buffer to the beginning, so it can be read from
    img_buf.seek(0)
    plt.close(fig)

    return img_buf
