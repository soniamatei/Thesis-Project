import torch
from data_handling.video_dataset import VideoDataset


def collate_fn_pad(batch: list[tuple[torch.Tensor]]) -> tuple[torch.Tensor]:
    """
    Pads batches of variable length. Dataloader helper function.
    :param batch: list of returned tuples from dataset as (frames, label)
    :return: the padded videos, the corresponding labels, the lengths of each sequence in the batch,
             and the mask indicating the padded elements
    """
    # separate videos and labels
    video_list, labels = zip(*batch)

    # get number of frames
    lengths = torch.tensor([t.size(0) for t in video_list], dtype=torch.int)

    # pad after max length tensor
    video_list = torch.nn.utils.rnn.pad_sequence(video_list, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.int)

    # mask to ignore padded data
    mask = (video_list != 0).float()

    return video_list, labels, lengths, mask


def compute_mean_std(
    dataset: VideoDataset, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and std for a dataset.
    :param dataset: the desired dataset
    :param device: option for calc. for optimization. defaults to 'cpu'
    :return: the calculated mean and standard deviation
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
            # make [channels, batch, frames, height, width]
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
