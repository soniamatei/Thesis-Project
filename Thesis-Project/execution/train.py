import sys
sys.path.insert(0, '/home/sonia2oo2soia/projects/Thesis-Project/Thesis-Project')  

from dotenv import load_dotenv
load_dotenv('/home/sonia2oo2soia/projects/Thesis-Project/Thesis-Project/.env')

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, animation

from torch.optim.lr_scheduler import StepLR
import wandb
from sklearn.metrics import f1_score
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils import data

from data_handling.data_augmentation import VideoTransform
from data_handling.video_dataset import VideoDataset
from model.violence_detection_model import ViolenceDetectionModel
from utils import collate_fn_pad, plot_to_image_buff

sweep_config = {
    # search strategy
    'method': 'random',
    'metric': {
        'name': 'mean_f1_score',
        'goal': 'maximize'
    },
    'parameters': {
        'epoch': {
            'value': 2
        },
        'learning_rate': {
            'min': 0.001,
            'max': 0.1
            # 'value': 8.971e-5
        },
        'factor': {
            'min': 0.001,
            'max': 0.1
            # 'value': 0.03526
        },
        'batch_size': {
            'value': 24
        },
        'n_folds': {
            'value': 4
        },
        'step_size': {
            'values': [1, 2]
            # 'value': 2
        },
        'treshold': {
            'value': 0.5
        },
    },
}

sweep_id = wandb.sweep(sweep_config, project="vd_model_training")

def train():
    print("initialize wandb")
    run = wandb.init(project="vd_model_training")
    loss_per_batch_columns = ["Fold", "Dataset", "Loss Plot"]
    loss_per_batch_table = wandb.Table(columns=loss_per_batch_columns)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("device: {}".format(device))

    print("set transforms for each dataset")
    hf_transforms = VideoTransform(dataset="HockeyFights", json_file="augmentation_values.json")
    # rwf_transforms = VideoTransform(dataset="RWF-2000", json_file="augmentation_values.json")

    print("initialize datasets")
    hf_dataset = VideoDataset(dataset="HockeyFights", transformations=hf_transforms)
    # rwf_dataset = VideoDataset(dataset="RWF-300", transformations=rwf_transforms)
    # dataloader = data.DataLoader(rwf_dataset, batch_size=1, collate_fn=collate_fn_pad)

    # for i, batch in enumerate(dataloader):
    #     fig, ax = plt.subplots(figsize=(3, 3))
    #     frames = batch[0][0].permute(0, 2, 3, 1)
    
    #     # define an animation to play th video
    #     def animate(frame):
    #         ax.clear()
    #         ax.imshow(frame.float().clamp(0, 1))
    #         ax.axis("off")
    
    #     ani = animation.FuncAnimation(fig, animate, frames=frames, interval=30)
    #     ani.save(f'animation_{i}.gif', writer='imagemagick')
    #     plt.close(fig)
    #     return

    # helper list for calculating the final val_loss
    val_losses = []
    f1_scores = []

    for dataset in [hf_dataset]:
        print("dataset {}".format(dataset.dataset))

        for n, fold in enumerate(dataset.k_fold(n_folds=wandb.config.n_folds)):
            print("fold {}".format(n))
            wandb_name_log = "fold_" + str(n) + "_" + dataset.dataset
            dataloader = data.DataLoader(fold, batch_size=wandb.config.batch_size, collate_fn=collate_fn_pad, shuffle=True, num_workers=2) # faster loading

            print("create model")
            vd_model = ViolenceDetectionModel(json_file="model_settings.json")
            vd_model = vd_model.to(device)

            criterion = BCELoss()
            optimizer = Adam(params=vd_model.parameters(), lr=wandb.config.learning_rate)
            scheduler = StepLR(optimizer=optimizer, step_size=wandb.config.step_size, gamma=wandb.config.factor)

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
                wandb.log({wandb_name_log + "_train_loss": avg_loss, "epoch": epoch})

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
                with torch.no_grad():
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
                all_outputs, all_labels = (np.array(cumulating_outputs) >= wandb.config.treshold), np.array(cumulating_labels)
                f1_score_result = f1_score(y_true=all_labels, y_pred=all_outputs, average="binary")

                val_losses.append(avg_loss)
                f1_scores.append(f1_score_result)

                wandb.log({wandb_name_log + "_val_loss": avg_loss, wandb_name_log + "_f1score": f1_score_result, "epoch": epoch})

            # plot batch losses for each epoch and save to wandb
            image_buffer = plot_to_image_buff(x=losses_per_batch, label='Loss per batch')
            image = Image.open(image_buffer)
            loss_per_batch_table.add_data(n, dataset.dataset, wandb.Image(image))
            image_buffer.close()

            # model_path = "vd_model.pth"
            # torch.save(vd_model.state_dict(), model_path)

            # # create a wandb artifact and log it
            # artifact = wandb.Artifact(name="vd_model_fold_" + str(n), type="model")
            # artifact.add_file(model_path)
            # wandb.log_artifact(artifact)

            break  # fold

    # log the loss which needs to be minimized
    wandb.log({'mean_val_loss': np.array(val_losses).mean()})
    wandb.log({'mean_f1_score': np.array(f1_scores).mean()})

    # log the table
    wandb.log({"Loss per Batch Table in Training": loss_per_batch_table})
    
    # finish run
    run.finish()


if __name__ == "__main__":
    wandb.agent(sweep_id, function=train)
    train()
