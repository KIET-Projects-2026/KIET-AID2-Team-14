

import os
import shutil
import random

# Paths (UPDATED for your real system)
SOURCE_DIR = r"C:\Users\nawyah11112004\Downloads\plant-disease-main (1)\plant-disease-main\archive\PlantVillage"

DEST_DIR = r"C:\Users\nawyah11112004\Downloads\plant-disease-main (1)\plant-disease-main\dataset"
TRAIN_DIR = os.path.join(DEST_DIR, "train")
VAL_DIR = os.path.join(DEST_DIR, "val")

SPLIT_RATIO = 0.8  # 80% train, 20% val

# Create destination folders if not exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Loop through each class folder inside COLOR
for cls in os.listdir(SOURCE_DIR):
    src_class_dir = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(src_class_dir):
        continue

    # Create class folders inside train/val
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_point = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_point]
    val_imgs = images[split_point:]

    # Copy images
    for img in train_imgs:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(TRAIN_DIR, cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(VAL_DIR, cls, img))

print("[INFO] Dataset split completed successfully!")
