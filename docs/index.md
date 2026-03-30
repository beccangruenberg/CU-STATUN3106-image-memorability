# STATUN3106 - Applied Machine Learning - Final Project
## What makes an Image Memorable?
##### Assesing if a CNN relies on human-interpretable features in the MIT Large-scale Image Memorability (LaMem) dataset

### GitHub Repository
[github.com/beccangruenberg/CU-STATUN3106-image-memorability](https://github.com/beccangruenberg/CU-STATUN3106-image-memorability/)
### Data Folder
[Google Drive Folder (LionMail Only)](https://drive.google.com/drive/folders/1Fg0yI3b-sTj-eAipAjZon-CvyojZuq23?usp=sharing)
### Project Webpage
[STATUN3106-LaMem.bg2782.net](https://STATUN3106-LaMem.bg2782.net)

## Prototype

### Data

### Initial ML Pipeline

We trained a ResNet50-based convolutional neural network on a subset of the LaMem dataset to predict image memorability scores. The pipeline includes data loading, an 80/20 train/validation split, and a regression head that outputs a memorability score between 0 and 1. After 3 epochs of training, the model achieved a validation loss of 0.0139, with loss decreasing consistently across epochs, demonstrating that the model is successfully learning to predict memorability from visual features. We have also prepared three ablation image sets (cropped, blurred, and color-modified), each containing each applied to all 500 images of our subset, which are ready to be passed through the model in the next phase. We will add two more ablation sets (desaturated, faces-removed) in future sets. Going forward, we will train the model for additional epochs to improve performance, implement Grad-CAM to visualize which regions of an image the model attends to when making predictions, and run the prepared ablation experiments to measure how specific visual elements influence memorability scores. We will also build a VAE latent space visualization to explore whether images with similar memorability scores cluster together in representation space, and develop an interactive tool that allows users to upload an image and receive a predicted memorability score alongside a Grad-CAM attention heatmap.

### Blog Post