import os
import csv

folder = "lamem_subset"

images = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])

with open("image_list.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path"])

    for img in images:
        writer.writerow([f"lamem_subset/{img}"])

print("image_list.csv created")