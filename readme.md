# STATUN3106 - Applied Machine Learning - Final Project
### What makes an Image Memorable?
#### Assesing if a CNN relies on human-interpretable features in the MIT Large-scale Image Memorability (LaMem) dataset

## Prototype

### Blog Post Webpage

[STATUN3106-LaMem.bg2782.com](https://STATUN3106-LaMem.bg2782.com/)

### Data

Data for prototype found here: [Google Drive Folder (LionMail Only)](https://drive.google.com/drive/folders/1Fg0yI3b-sTj-eAipAjZon-CvyojZuq23?usp=sharing)

### ML Pipeline

[Prototype Notebook](https://github.com/beccangruenberg/CU-STATUN3106-image-memorability/tree/main/prototype)

## Setup
Add this cell to the top of every notebook before running. 
Change `DATA_PATH` to your own Google Drive path.
```python
from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.exists('/content/STATUN3106-image-memorability'):
    !git clone https://github.com/beccangruenberg/STATUN3106-image-memorability.git /content/STATUN3106-image-memorability
else:
    !git -C /content/STATUN3106-image-memorability pull

# ── SET YOUR DATA PATH HERE ────────────────────────────────
DATA_PATH = "/content/drive/MyDrive/YOUR/PATH/HERE"
print(f"Data path set to: {DATA_PATH}")
```

Create a local individual_data_paths.txt file for easy path access - already added to .gitignore
