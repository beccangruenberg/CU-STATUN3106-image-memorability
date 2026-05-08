# Columbia University - STATUN3106 - Applied Machine Learning - Spring 2026 - Final Project
### Benji Barnes, Beccan Gruenberg, Kayla Jiang, Mason Lau
## What makes an Image Memorable?
#### Assesing if a CNN relies on human-interpretable features in the MIT Large-scale Image Memorability (LaMem) dataset

## Final Submission

### Blog Post Webpage 

[github.beccan.gruenbergs.net/CU-STATUN3106-image-memorability](https://github.beccan.gruenbergs.net/CU-STATUN3106-image-memorability/)

### Data

Data for final submission found here: [Google Drive Folder (LionMail Only)](https://drive.google.com/drive/folders/1l1U0WZRJIZfdvxtxu5VsTpBUQYYwxd_C)

### Combined Notebook

[Final Notebook](https://github.com/beccangruenberg/CU-STATUN3106-image-memorability/blob/main/final_submission/final_combined.ipynb)

### Image Memorability Analyzer

[Webpage](https://beccangruenberg-cu-statun3106-image-memorability.hf.space/)

## Prototype

### Prototype Blog Post Webpage

[github.beccan.gruenbergs.net/CU-STATUN3106-image-memorability/prototype](https://github.beccan.gruenbergs.net/CU-STATUN3106-image-memorability/prototype/#blog-post)

### Data

Data for prototype found here: [Google Drive Folder (LionMail Only)](https://drive.google.com/drive/folders/1bZ7mYIZ1_bSef4brn3s91gUYeG4JEruR)

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
