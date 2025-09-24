import os
import shutil
import random

# Source directories
negatives_dir = r"C:\Users\hbsu\fiftyone\coco-2017\train\negatives"
positives_dir = r"C:\Users\hbsu\fiftyone\coco-2017\train\positives"

# Destination root
dataset_dir = r"C:\Users\hbsu\Desktop\dataset"

# Splits
splits = {
    "training": 0.75,
    "validation": 0.15,
    "test": 0.10
}

# Make sure output dirs exist (positives and negatives inside each split)
for split in splits.keys():
    for cls in ["positives", "negatives"]:
        out_dir = os.path.join(dataset_dir, split, cls)
        os.makedirs(out_dir, exist_ok=True)

def split_and_copy(src_dir, cls_name):
    # Get all files
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    random.shuffle(files)

    total = len(files)
    train_end = int(total * splits["training"])
    val_end = train_end + int(total * splits["validation"])

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # Copy files to new directories
    for f in train_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dataset_dir, "training", cls_name, f))
    for f in val_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dataset_dir, "validation", cls_name, f))
    for f in test_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dataset_dir, "test", cls_name, f))

# Run for both positives and negatives
split_and_copy(positives_dir, "positives")
split_and_copy(negatives_dir, "negatives")

print("Dataset split completed!")
