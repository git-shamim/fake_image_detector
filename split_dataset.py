import os
import shutil
import random
from glob import glob

# Define source directories (raw data)
SOURCE_REAL = "data/real"
SOURCE_FAKE = "data/fake"

# Destination base directory
DEST_DIR = "data_split"
TRAIN_RATIO = 0.8  # 80% training, 20% validation

def prepare_split():
    for label in ['real', 'fake']:
        src_folder = os.path.join("data", label)
        # Consider both jpg and png files
        images = glob(os.path.join(src_folder, "*.jpg")) + glob(os.path.join(src_folder, "*.png"))
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for set_name, files in [('train', train_images), ('val', val_images)]:
            dest_folder = os.path.join(DEST_DIR, set_name, label)
            os.makedirs(dest_folder, exist_ok=True)
            for img_path in files:
                shutil.copy(img_path, dest_folder)

    print("✅ Data split complete. Directory structure created under 'data_split'.")

if __name__ == "__main__":
    prepare_split()
