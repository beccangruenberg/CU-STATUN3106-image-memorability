import os
import csv

subset_folder = "lamem_subset"

split_files = [
    "/Users/kayla/Downloads/lamem/splits/train_1.txt",
    "/Users/kayla/Downloads/lamem/splits/val_1.txt",
    "/Users/kayla/Downloads/lamem/splits/test_1.txt"
]

output_file = "image_list_with_scores.csv"

score_map = {}

for file in split_files:
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, score = parts
                score_map[filename] = score

images = sorted([f for f in os.listdir(subset_folder) if f.endswith(".jpg")])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "memorability_score"])

    for img in images:
        score = score_map.get(img, "")
        writer.writerow([f"lamem_subset/{img}", score])

print("image_list_with_scores.csv created")
