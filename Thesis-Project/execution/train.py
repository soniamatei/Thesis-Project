import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, animation

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import wandb
from sklearn.metrics import f1_score
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils import data

from data_handling.data_augmentation import VideoTransform
from data_handling.video_dataset import VideoDataset
from model.violence_detection_model import ViolenceDetectionModel
from utils import collate_fn_pad, plot_to_image_buff

# plt.switch_backend('TkAgg')

sweep_config = {
    # search strategy
    'method': 'bayes',
    'metric': {
        'name': 'mean_val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epoch': {
            'values': [2]
        },
        'learning_rate': {
            'min': 1e-7,
            'max': 6e-3
        },
        'factor': {
            # 'values': [0.5, 0.7, 0.1, 0.3]
            'min': 1e-7,
            'max': 1e-1
        },
        'batch_size': {
            'values': [16, 24]
        },
        'n_folds': {
            'values': [4, 5]
        },
    },
}

sweep_id = wandb.sweep(sweep_config, project="vd_model_training")


def train():
    print("initialize wandb")
    wandb.init(project="vd_model_training")
    loss_per_batch_columns = ["Fold", "Dataset", "Loss Plot"]
    loss_per_batch_table = wandb.Table(columns=loss_per_batch_columns)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("device: {}".format(device))

    # TODO: maybe make a constant from database names

    print("set transforms for each dataset")
    hf_transforms = VideoTransform(dataset="HockeyFights", json_file="augmentation_values.json")
    rwf_transforms = VideoTransform(dataset="RWF-2000", json_file="augmentation_values.json")

    print("initialize datasets")
    hf_dataset = VideoDataset(dataset="HockeyFights", transformations=hf_transforms)
    rwf_dataset = VideoDataset(dataset="RWF-2000", transformations=rwf_transforms)
    # dataloader = data.DataLoader(hf_dataset, batch_size=wandb.config.batch_size, collate_fn=collate_fn_pad)

    # for i, batch in enumerate(dataloader):
    #     fig, ax = plt.subplots(figsize=(3, 3))
    #     frames = batch[0][0].permute(0, 2, 3, 1)
    #
    #     # define an animation to play th video
    #     def animate(frame):
    #         ax.clear()
    #         ax.imshow(frame.float().clamp(0, 1))
    #         ax.axis("off")
    #
    #     ani = animation.FuncAnimation(fig, animate, frames=frames, interval=30)
    #     plt.show()

    # helper list for calculating the final val_loss
    val_losses = []

    for dataset in [hf_dataset]:
        print("dataset {}".format(dataset.dataset))

        for n, fold in enumerate(dataset.k_fold(n_folds=wandb.config.n_folds)):
            print("fold {}".format(n))
            dataloader = data.DataLoader(fold, batch_size=wandb.config.batch_size, collate_fn=collate_fn_pad, shuffle=True)

            print("create model")
            vd_model = ViolenceDetectionModel(json_file="model_settings.json")
            vd_model = vd_model.to(device)

            criterion = BCELoss()
            optimizer = Adam(params=vd_model.parameters(), lr=wandb.config.learning_rate)
            scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=wandb.config.factor)
            # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=wandb.config.factor, patience=0)

            # track losses per batch in training along epochs
            losses_per_batch = np.array([])

            for epoch in range(wandb.config.epoch):
                print("epoch {}".format(epoch))

                # ------------train------------
                print("training")

                vd_model.train()
                # use training part of dataset
                fold.flag = False

                # track loss for entire train dataset
                cumulating_loss = 0.0

                print(len(fold))
                print(len(dataloader))
                for batch in dataloader:
                    print("batch")
                    optimizer.zero_grad()

                    videos, labels, lengths = batch
                    videos, labels = videos.to(device), labels.to(device)

                    outputs = vd_model(videos, lengths)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    losses_per_batch = np.append(losses_per_batch, loss.detach().cpu().item())
                    cumulating_loss += loss.cpu().item()

                # log the cumulating train loss to wandb
                avg_loss = cumulating_loss / len(dataloader)
                wandb.log({dataset.dataset + "_train_loss": avg_loss})

                scheduler.step()
                # ------------val------------
                print("evaluating")

                vd_model.eval()
                fold.flag = True

                # lists for tracking loss and f1 for the entire val dataset
                cumulating_loss = 0.0
                cumulating_outputs = []
                cumulating_labels = []

                print(len(fold))
                print(len(dataloader))
                for batch in dataloader:
                    print("batch")
                    videos, labels, lengths = batch
                    videos, labels = videos.to(device), labels.to(device)

                    outputs = vd_model(videos, lengths)
                    loss = criterion(outputs, labels)

                    cumulating_loss += loss.cpu().item()
                    cumulating_outputs.extend(outputs.cpu().tolist())
                    cumulating_labels.extend(labels.cpu().tolist())

                avg_loss = cumulating_loss / len(dataloader)
                all_outputs, all_labels = (np.array(cumulating_outputs) >= 0.5), np.array(cumulating_labels)
                f1_score_result = f1_score(y_true=all_labels, y_pred=all_outputs, average="binary")

                val_losses.append(avg_loss)

                wandb.log({dataset.dataset + "_val_loss": avg_loss, dataset.dataset + "_f1score": f1_score_result})

            # plot batch losses for each epoch and save to wandb
            image_buffer = plot_to_image_buff(x=losses_per_batch, label='Loss per batch')
            image = Image.open(image_buffer)
            loss_per_batch_table.add_data(n, dataset.dataset, wandb.Image(image))
            image_buffer.close()

            break  # fold

    # TODO: maybe add weights to importance
    # log the loss which needs to be minimized
    wandb.log({'mean_val_loss': np.array(val_losses).mean()})

    # log the table
    wandb.log({"Loss per Batch Table in Training": loss_per_batch_table})


if __name__ == "__main__":
    wandb.agent(sweep_id, function=train)
