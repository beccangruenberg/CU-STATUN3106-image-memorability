import os
from PIL import Image, ImageFilter

input_folder = "lamem_subset"

blur_folder = "ablations_blurred"
gray_folder = "ablations_grayscale"
crop_folder = "ablations_cropped"

os.makedirs(blur_folder, exist_ok=True)
os.makedirs(gray_folder, exist_ok=True)
os.makedirs(crop_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path)

    # blurred
    blurred = img.filter(ImageFilter.GaussianBlur(5))
    blurred.save(os.path.join(blur_folder, img_name))

    # grayscale
    gray = img.convert("L")
    gray.save(os.path.join(gray_folder, img_name))

    # cropped
    w, h = img.size
    crop = img.crop((w*0.25, h*0.25, w*0.75, h*0.75))
    crop.save(os.path.join(crop_folder, img_name))

print("Ablations created")