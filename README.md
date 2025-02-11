# Thesis Project: Violence Detection in Videos

This project aims to detect violence within a video. It is structured into three main parts:  

1. **Dataset Transformation and Manipulation**  
2. **Model Structure**
3. **Execution (Training) and Appliation (GUI)**  

Each component is implemented in `.py` files and tested within `.ipynb` notebooks to ensure correctness before training. The project follows PyTorch standards for implementation.  

## Dataset Handling  

The dataset module is responsible for loading and preparing video data for training.  

- **Video Dataset**  
  - loads videos into the required format from memory  
  - supports functionalities such as: k-Fold splitting, train/test splitting with percentage ratios or data shuffling
  - 1000 videos from hockey games that contain violent and non-violent interactions between the players ([Hockey Fights](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes/data)).

- **Video Transforms**  
  - ensures transformations are applied to each video frame  
  - transformations can be: loaded from a JSON file (validated for correctness) and modified in the code and saved to a JSON file if desired

## Model Structure  

The violence detection model is implemented using `torch.nn.Module` as a base class.  

- **Model Architecture**  
  - uses a **Vision Transformer (ViT)** for spatial feature extraction  [Hugging Face ViT Documentation](https://huggingface.co/docs/transformers/en/model_doc/vit)
  - a **Transformer Encoder** is applied for temporal analysis  (inspired by official PyTorch recommendations)

## Training & Execution  

- **Training Loop**  
  - uses [Weights & Biases (WandB)](https://wandb.ai/soniamatei/vd_model_training/overview) to track performance (the **Sweep** functionality for hyperparameter optimization (which compares each training run based on a metric (loss, accuracy, aso)))

- **Application**  
  - a GUI built with [Gradio](https://www.gradio.app/) for real-time violence detection which allows users to upload a video and receive predictions  
