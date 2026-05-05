import os
import random
import shutil

random.seed(42)

source_folder = "/Users/kayla/Downloads/lamem/images"
target_folder = "lamem_final"    # folder for the 500 sampled images

os.makedirs(target_folder, exist_ok=True)

images = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")]
print(len(images))

subset = random.sample(images, 1000)

for img in subset:
    shutil.copy(os.path.join(source_folder, img),
                os.path.join(target_folder, img))

print("1000 images copied into lamem_final.")