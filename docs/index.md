# STATUN3106 - Applied Machine Learning - Final Project
## What makes an Image Memorable?
##### Assesing if a CNN relies on human-interpretable features in the MIT Large-scale Image Memorability (LaMem) dataset

### GitHub Repository
[github.com/beccangruenberg/CU-STATUN3106-image-memorability](https://github.com/beccangruenberg/CU-STATUN3106-image-memorability/)
### Data Folder
[Google Drive Folder (LionMail Only)](https://drive.google.com/drive/folders/1Fg0yI3b-sTj-eAipAjZon-CvyojZuq23?usp=sharing)
### Project Webpage
[github.beccan.gruenbergs.net/CU-STATUN3106-image-memorability/](https://github.beccan.gruenbergs.net/CU-STATUN3106-image-memorability/)

## Prototype

### Dataset

500 random images were sampled from the LaMem dataset, and by matching each filename to the provided train/val/test split files, we retained their original memorability scores. We then created a CSV (image_list_with_scores.csv) containing the file path and memorability score for each image, which is used by the experiment interface. In addition to the original images, we then generated three ablation sets:

1. **Blurred images:** A Gaussian blur was applied to each image to remove fine details and textures while keeping the overall scene structure.
   - This tests whether memorability depends on high-frequency visual details.
2. **Put images into black and white (gray-scaled images):** Converted the images to grayscale to remove color information while preserving shapes and contrast.
   - This tests whether color plays a role in memorability.
3. **Cropped images:** Each image was cropped to retain only the central portion of the image (approximately the middle 50% of the width and height).
   - We removed the surrounding background or contextual information while preserving the subject, assuming the subject is centered.
   - This tests if memorability is driven more by the central object image or by the broader context of the scene of the image
All of the preprocessing steps were implemented with Python scripts to ensure that the dataset can be reproduced.

For the final project, we will extend this by testing even more semantically meaningful features, such as selectively removing faces or salient objects, to better understand whether memorability predictions rely on human-interpretable content rather than low-level visual cues.

### Initial ML Pipeline

We trained a ResNet50-based convolutional neural network on a subset of the LaMem dataset to predict image memorability scores. The pipeline includes data loading, an 80/20 train/validation split, and a regression head that outputs a memorability score between 0 and 1. After 3 epochs of training, the model achieved a validation loss of 0.0139, with loss decreasing consistently across epochs, demonstrating that the model is successfully learning to predict memorability from visual features. We have also prepared three ablation image sets (cropped, blurred, and color-modified), each containing each applied to all 500 images of our subset, which are ready to be passed through the model in the next phase. We will add two more ablation sets (desaturated, faces-removed) in future sets. Going forward, we will train the model for additional epochs to improve performance, implement Grad-CAM to visualize which regions of an image the model attends to when making predictions, and run the prepared ablation experiments to measure how specific visual elements influence memorability scores. We will also build a VAE latent space visualization to explore whether images with similar memorability scores cluster together in representation space, and develop an interactive tool that allows users to upload an image and receive a predicted memorability score alongside a Grad-CAM attention heatmap.

### Blog Post

#### Final Project (A Quick Recap):

Our project investigates whether a convolutional neural network (CNN) learns the same visual features that make images memorable to humans. Using the LaMem dataset (60,000 images with human memorability scores), we train a ResNet-based model to predict memorability from image inputs. We then analyze the model’s behavior using Grad-CAM heatmaps and controlled ablation experiments (grayscale, blurred, and cropped) to determine which visual features drive its predictions. The central goal is to test whether the model relies on human-interpretable features (such as faces, salient objects, emotional content) or instead achieves accurate predictions using non-human visual cues (such as color distributions, background patterns, and low-level textures).

#### Why is this Problem Important?

Numerous previous studies have convincingly demonstrated that CNNs often achieve high accuracy by relying on seemingly arbitrary shortcut features – a phenomenon known as “shortcut learning.” In 2018, Zech et al. showed that a CNN trained to detect pneumonia from medical imaging relied on embedded text markers and hospital-specific artifacts rather than clinically meaningful lung pathology, leading to detrimentally incorrect results when applied in new hospitals with new image data. While shortcut learning is well established in standard vision tasks (Geirhos et al., 2020), it remains completely unknown whether the same issue occurs when models are trained explicitly on human memorability, where the labels themselves reflect real human cognitive behavior. This distinction is critical: if a CNN trained on LaMem relies on these shortcut cues, then it is not representative of how humans remember images, but instead depend on dataset-specific correlations. Any conclusions about “what makes an image memorable” would therefore be misleading.

If, on the other hand, the model relies on human-interpretable features and produces a clear ranking of their importance (e.g. face > color contrast > salient objects) similar to that of humans, then we can both validate existing findings (Isola et al., 2011) and directly translate these results into real-world decisions. From a practical perspective, advertisers could select campaign images with the highest-ranked features to improve brand recognition after exposure. Educators could design visual materials that maximize student engagement and retention by prioritizing the most memory-relevant elements. Computer-aided diagnosis could improve in accurately identifying illnesses from medical imaging, behaving with a physician’s eye for detail. Crucially, distinguishing between human-aligned visual features and mere shortcuts ensures that these applications are based on genuine drivers of human memory rather than fabricated associations, preventing systems that perform well in testing from failing in real-world deployment.

In a broader sense, this problem also poses some philosophical questions regarding our relationship with artificial intelligence and the direction that it should take. How exactly do machine learning models “see”? How do they encode information? In turn, how might these processes inform us about how humans do the same? Should we strive to tune models to directly replicate how humans interpret the visual world, or should we aim for the opposite? Understanding the sorts of visual features that these models rely on and comparing them to the visual features humans rely on will inevitably answer these questions to some degree.

#### What Prior Work has Been Done in this Area?

The scientific study of image memorability began in earnest with Isola et al.’s landmark 2011 paper, “What Makes an Image Memorable?”. In the paper, the researchers established that memorability is not subjective but rather highly consistent across individuals, and identified specific semantic features, such as the presence of people, objects, and scene meaning, as strong predictors of what humans remember. They found that images of enclosed spaces containing people with visible faces are highly memorable, while images of vistas and peaceful landscapes are among the least memorable. Their work also produced the first computational model capable of predicting memorability scores from image features, demonstrating for the first time that memorability was a measurable, predictable quantity rather than a purely subjective experience. However, this model relied on hand-engineered features and its predictive accuracy was poor.
 
Building directly on this foundation, the same MIT group later introduced the LaMem dataset (Khosla et al., 2015), scaling the problem from roughly 10,000 to over 60,000 images, along with MemNet - one of the first deep convolutional neural networks trained to predict memorability directly from raw images. A key finding of this work was that CNNs can closely match human-rated memorability scores and outperform traditional models based on hand-engineered features. However, the study did not investigate which visual features the CNN relied on to make its predictions, leaving open the question of whether the model learned human-interpretable signals or exploited non-human shortcuts.

One consistent and important finding across all of this early work is that humans are surprisingly bad at predicting memorability themselves. When people are asked to guess which images will be memorable, their guesses track how beautiful or interesting an image looks – not how memorable it actually turns out to be. In other words, an image can be stunning and immediately forgettable, or visually unremarkable and highly memorable. This disconnect between what people think makes images stick and what actually makes them stick is one of the main motivations for why our group wanted to use machine learning to study this problem.
 
More recently, Kramer et al. challenged two longstanding assumptions in the field. First, they overturned the prior emphasis on brightness and color as drivers of memorability, demonstrating instead that semantic properties of an image, such as whether it depicts a living thing or a man-made object, are far stronger predictors of what people remember. Second, they challenged the decades-old assumption that unusual or atypical objects are the most memorable. Using over one million memory ratings across 26,000 real-world object images, they found the opposite: more pro-typical looking objects tend to be better remembered. It will be interesting to see whether our results replicate the findings from Kramer et al.
 
Even more recently, Lahner et al. provided the first unified account of where and when memorability is encoded across the whole brain, using a combination of fMRI and MEG neuroimaging. They found that highly memorable images recruit a distributed network of ventral visual regions, including the inferior temporal cortex and fusiform gyrus, with this memorability response beginning approximately 300 milliseconds after image onset, later in visual processing rather than early. This timing is significant: the fact that memorability is encoded later (after the brain has already carried out high-level semantic processing) is consistent with the behavioral finding from Kramer et al. that semantic content, not low-level visual features, is the primary driver of what humans remember.
 
Taken together, this literature establishes a coherent picture: memorable images are those that engage the brain’s high-level semantic and social processing systems most deeply, with faces and people being the clearest example of stimuli that reliably trigger this response. Whether CNNs trained on memorability data have learned to approximate this same high-level semantic sensitivity, or whether they are picking up on something else entirely, is the core question our project investigates.

#### Current Hypotheses:

For our project, we have two potential outcomes:

- Case A: The CNN achieves accurate predictions using human-interpretable features
- Case B: The CNN achieves accurate predictions using non-human visual cues.

Our central hypothesis is that memorability predictions will depend primarily on high-level semantic content rather than low-level visual properties, but that the model may partially exploit shortcut cues. In Case A (human-aligned learning), we expect the CNN to rely on features identified in prior work: faces, people, and meaningful objects will rank highest (and we hypothesis the ranking to be face > person/action > salient object > text), while backgrounds and landscapes will contribute minimally. Under this scenario, ablations that remove faces or people should cause the largest drop in predicted memorability, whereas ablations that affect background regions will have little effect; grayscale conversion should have minimal impact, since color is weakly related to memorability; and blurring should moderately reduce scores by degrading semantic clarity (e.g. obscuring faces). In Case B (shortcut learning), we expect the model to rely on non-human cues such as texture frequency, background patterns, color distributions, or image statistics. In this case the feature ranking would shift (and we predict this shift to be in the order of texture/pattern > color contrast > spatial frequency > semantic content), and the model’s responses to these ablations would behave differently: cropping faces would not significantly reduce memorability if background statistics remain intact, grayscale may cause a noticeable drop if color distributions are being used, and blurring would produce the largest decline by disrupting high-frequency texture signals. Comparing these outcomes allows us to directly test whether the model is learning meaningful, human-relevant features or exploiting spurious correlations.

#### Limitations

A key limitation of our project is that it uses a relatively simple CNN trained on the LaMem dataset, whereas real-world systems (e.g. social media engagement models or medical imaging pipelines) are far more complex and incorporate additional signals such as user behavior, temporal dynamics, and feedback loops.

A second limitation stems from the dataset itself. LaMem provides memorability scores averaged across many participants and does not include demographic information, meaning the model assumes memorability is consistent across individuals. In reality, factors such as age, cultural background, and prior experience may influence what people remember, which our model cannot capture.

Finally, our definition of memorability is restricted to short-term recognition of static images, which differs from real-world memory processes that depend on context, repetition, and sustained attention over time. As a result, even if we identify important visual features, these reflect correlations within this constrained setting rather than comprehensive causal drivers of human memory.