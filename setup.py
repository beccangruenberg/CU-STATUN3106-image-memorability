# setup.py
# ── Run this once at the start of every session ───────────
from google.colab import drive
drive.mount('/content/drive')

import os, sys

if not os.path.exists('/content/STATUN3106-image-memorability'):
    os.system('git clone https://github.com/beccangruenberg/STATUN3106-image-memorability.git /content/STATUN3106-image-memorability')
else:
    os.system('git -C /content/STATUN3106-image-memorability pull')

sys.path.append('/content/STATUN3106-image-memorability')

# ── SET YOUR DATA PATH HERE ────────────────────────────────
DATA_PATH = "/content/drive/MyDrive/STATUN3106 - Applied Machine Learning/Final Project/STATUN3104_Final_Project"
# ──────────────────────────────────────────────────────────