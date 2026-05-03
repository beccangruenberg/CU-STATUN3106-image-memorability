import os
import random
import shutil

source_folder = "."          # folder with all LaMem images
target_folder = "lamem_subset"    # folder for the 500 sampled images

os.makedirs(target_folder, exist_ok=True)

images = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")]

subset = random.sample(images, 500)

for img in subset:
    shutil.copy(os.path.join(source_folder, img),
                os.path.join(target_folder, img))

print("500 images copied into lamem_subset.")