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

### Data Set

500 random images were sampled from the LaMem dataset, and by matching each filename to the provided train/val/test split files, we retained their original memorability scores. We then created a CSV (image_list_with_scores.csv) containing the file path and memorability score for each image, which is used by the experiment interface. In addition to the original images, we then generated three ablation sets:

1. Blurred images: A Gaussian blur was applied to each image to remove fine details and textures while keeping the overall scene structure.
   - This tests whether memorability depends on high-frequency visual details.
2. Put images into black and white (gray-scaled images): Converted the images to grayscale to remove color information while preserving shapes and contrast.
   - This tests whether color plays a role in memorability.
3. Cropped images: Each image was cropped to retain only the central portion of the image (approximately the middle 50% of the width and height).
   - We removed the surrounding background or contextual information while preserving the subject, assuming the subject is centered.
   - This tests if memorability is driven more by the central object image or by the broader context of the scene of the image
All of the preprocessing steps were implemented with Python scripts to ensure that the dataset can be reproduced.

Our future work will extend this by testing more semantically meaningful features, such as selectively removing faces or salient objects, to better understand whether memorability predictions rely on human-interpretable content rather than low-level visual cues.


### Initial ML Pipeline

We trained a ResNet50-based convolutional neural network on a subset of the LaMem dataset to predict image memorability scores. The pipeline includes data loading, an 80/20 train/validation split, and a regression head that outputs a memorability score between 0 and 1. After 3 epochs of training, the model achieved a validation loss of 0.0139, with loss decreasing consistently across epochs, demonstrating that the model is successfully learning to predict memorability from visual features. We have also prepared three ablation image sets (cropped, blurred, and color-modified), each containing each applied to all 500 images of our subset, which are ready to be passed through the model in the next phase. We will add two more ablation sets (desaturated, faces-removed) in future sets. Going forward, we will train the model for additional epochs to improve performance, implement Grad-CAM to visualize which regions of an image the model attends to when making predictions, and run the prepared ablation experiments to measure how specific visual elements influence memorability scores. We will also build a VAE latent space visualization to explore whether images with similar memorability scores cluster together in representation space, and develop an interactive tool that allows users to upload an image and receive a predicted memorability score alongside a Grad-CAM attention heatmap.

### Blog Post