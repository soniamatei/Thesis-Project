import torch
from torch.nn import BCELoss
import wandb
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils import data
import numpy as np

from data_handling.data_augmentation import VideoTransform
from data_handling.video_dataset import VideoDataset
from model.violence_detection_model import ViolenceDetectionModel
from utils import collate_fn_pad

sweep_config = {
    # search strategy
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epoch': {
            'values': [2]
        },
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'values': [8, 12, 16]
        },
        'n_folds': {
            'values': [5, 7, 4, 10]
        },
        'brightness': {
            'min': 0.0,
            'max': 0.5
        },
        'contrast': {
            'min': 0.0,
            'max': 0.5
        },
        'saturation': {
            'min': 0.0,
            'max': 0.5
        },
        'hue': {
            'min': 0.0,
            'max': 0.1
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="vd_model_training")


def train():
    wandb.init(project="vd_model_training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("device: {}".format(device))

    # TODO: maybe make a constant from database names

    print("set transforms for each dataset")
    hf_transforms = VideoTransform(dataset="HockeyFights", json_file="augmentation_values.json")
    rwf_transforms = VideoTransform(dataset="RWF-2000", json_file="augmentation_values.json")

    for transforms in [hf_transforms, rwf_transforms]:
        aug_val = transforms.json()
        aug_val["color_jitter_params"]["brightness"] = wandb.config.brightness
        aug_val["color_jitter_params"]["contrast"] = wandb.config.contrast
        aug_val["color_jitter_params"]["saturation"] = wandb.config.saturation
        aug_val["color_jitter_params"]["hue"] = wandb.config.hue

        transforms.save(aug_val)

    print("initialize datasets")
    hf_dataset = VideoDataset(dataset="HockeyFights", transformations=hf_transforms)
    rwf_dataset = VideoDataset(dataset="RWF-2000", transformations=rwf_transforms)

    print("create model")
    vd_model = ViolenceDetectionModel(json_file="model_settings.json")
    vd_model = vd_model.to(device)

    criterion = BCELoss()
    optimizer = Adam(params=vd_model.parameters(), lr=wandb.config.learning_rate)

    for epoch in range(wandb.config.epoch):
        print("epoch {}".format(epoch))
        for dataset in [hf_dataset, rwf_dataset]:
            print("dataset {}".format(dataset.dataset))
            for n, fold in enumerate(dataset.k_fold(n_folds=wandb.config.n_folds)):
                print("fold {}".format(n))
                print("training")
                vd_model.train()
                fold.flag = False

                dataloader = data.DataLoader(fold, batch_size=wandb.config.batch_size, collate_fn=collate_fn_pad)
                for batch in dataloader:
                    print("batch")
                    optimizer.zero_grad()

                    videos, labels, lengths = batch
                    videos, labels = videos.to(device), labels.to(device)

                    outputs = vd_model(videos, lengths)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # delete to not occupy space in cuda
                    if device == "cuda":
                        del videos, labels
                        torch.cuda.empty_cache()

                print("evaluating")
                vd_model.eval()
                fold.flag = True

                # track loss and f1 for the entire val dataset
                cumulating_loss = 0.0
                cumulating_outputs_labels = []

                for batch in dataloader:
                    print("batch")
                    videos, labels, lengths = batch
                    videos, labels = videos.to(device), labels.to(device)

                    outputs = vd_model(videos, lengths)
                    loss = criterion(outputs, labels)

                    cumulating_loss += loss.cpu().item()
                    cumulating_outputs_labels.append((outputs.cpu().detach(), labels.cpu().detach()))

                    if device == "cuda":
                        del videos, labels
                        torch.cuda.empty_cache()

                avg_loss = cumulating_loss / len(dataloader)
                all_outputs, all_labels = zip(*cumulating_outputs_labels)
                all_outputs, all_labels = (np.array(all_outputs) >= 0.5), np.array(all_labels)
                f1_score_result = f1_score(all_labels, all_outputs, average="binary")

                wandb.log({dataset.dataset + "_loss": avg_loss, dataset.dataset + "_f1score": f1_score_result})


if __name__ == "__main__":
    wandb.agent(sweep_id, function=train)



