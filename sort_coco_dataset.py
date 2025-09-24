# sort person vs nonperson
# #What it does:
# Walk images under the image directory
# Match each image to coco train annnotations
# Transform 640 * 480 grayscale (crop then resize to avoid distortion)
# Save into train_out_positives or train_out_negatives accordingly
#
# Run:
#     python sort_coco_dataset.py
#
#     or :
#     python sort_coco_dataset.py --train_img_dir "C:\Users\hbsu\fiftyone\coco-2017\train\data" --raw_dir "C:\Users\hbsu\fiftyone\coco-2017\raw"

#!/usr/bin/env python3

import os
import json
import argparse
from glob import glob
from collections import defaultdict

from PIL import Image
# >>> ADDED: robust image loading + CSV logging
from PIL import ImageFile            # >>> ADDED
ImageFile.LOAD_TRUNCATED_IMAGES = True  # >>> ADDED
import csv                           # >>> ADDED

from torchvision import transforms

# -------- defaults (edit if you want) ----------
TRAIN_IMG_DIR_DEFAULT = r"C:\Users\hbsu\fiftyone\coco-2017\train\data"
RAW_DIR_DEFAULT       = r"C:\Users\hbsu\fiftyone\coco-2017\raw"  # contains instances_train2017.json
OUT_POS_DEFAULT       = r"C:\Users\hbsu\fiftyone\coco-2017\train\positives"
OUT_NEG_DEFAULT       = r"C:\Users\hbsu\fiftyone\coco-2017\train\negatives"
# ------------------------------------------------


def load_coco_indices(instances_json_path):
    """Build quick lookups from COCO instances JSON."""
    with open(instances_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])

    # Map category_id -> name (so we can test 'person')
    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}

    # Map image_id -> [annotations...]
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    # Map basename -> image dict (COCO file_name may contain "train2017/xxx.jpg")
    img_by_basename = {}
    for im in images:
        fname = os.path.basename(im.get("file_name", ""))
        if fname:
            img_by_basename[fname] = im

    return img_by_basename, anns_by_img, cat_name


def make_transform():
    """Crop to square (center) then resize to 480x640 and grayscale."""
    return transforms.Compose([
        transforms.CenterCrop,  # we’ll bind at runtime with side=min(img.size)
        transforms.Resize((480, 640), antialias=True),
        transforms.Grayscale(num_output_channels=1),
    ])


def process_one_image(img_path, img_entry, anns_by_img, cat_name, out_pos_dir, out_neg_dir):
    """
    - Open + transform image
    - Decide positive/negative based on whether any ann is 'person'
    - Save to appropriate folder
    Returns (has_person, out_path) or (None, None) if failed/missing.
    """
    # >>> ADDED: early existence check
    if not os.path.isfile(img_path):             # >>> ADDED
        print(f"[MISSING] {img_path}")           # >>> ADDED
        return None, None                        # >>> ADDED

    # Gather anns for this image
    img_id = img_entry["id"]
    anns   = anns_by_img.get(img_id, [])

    has_person = any(cat_name.get(a["category_id"], "") == "person" for a in anns)

    # >>> CHANGED: wrap image open/transform in try/except and print the file path on error
    try:                                         # >>> ADDED
        with Image.open(img_path) as pil_im:     # >>> CHANGED (no .convert here)
            pil_im = pil_im.convert("RGB")       # >>> ADDED

            # CenterCrop requires a size; use min(H,W) for square crop
            crop_side = min(pil_im.size)
            tf = transforms.Compose([
                transforms.CenterCrop(crop_side),
                transforms.Resize((480, 640), antialias=True),
                transforms.Grayscale(num_output_channels=1),
            ])
            out_im = tf(pil_im)
    except Exception as e:                       # >>> ADDED
        print(f"[ERROR] Failed on image: {img_path}")   # >>> ADDED
        print(f"        Reason: {e}")                   # >>> ADDED
        return None, None                                # >>> ADDED

    # Decide output
    target_dir = out_pos_dir if has_person else out_neg_dir
    os.makedirs(target_dir, exist_ok=True)

    out_path = os.path.join(target_dir, os.path.basename(img_path))
    out_im.save(out_path)
    return has_person, out_path


def main(train_img_dir, raw_dir, out_pos_dir, out_neg_dir, limit=None):
    instances_json = os.path.join(raw_dir, "instances_train2017.json")
    if not os.path.isfile(instances_json):
        raise FileNotFoundError(f"Could not find instances JSON at: {instances_json}")

    img_by_base, anns_by_img, cat_name = load_coco_indices(instances_json)

    # Collect real files on disk under train_img_dir (recursive)
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob(os.path.join(train_img_dir, "**", ext), recursive=True))
    files.sort()

    processed = 0
    skipped_no_ann = 0
    pos_count = 0
    neg_count = 0

    # >>> ADDED: CSV log for debugging
    log_path = os.path.join(os.getcwd(), "sort_log.csv")   # >>> ADDED
    logf = open(log_path, "w", newline="", encoding="utf-8")  # >>> ADDED
    wlog = csv.writer(logf)                                   # >>> ADDED
    wlog.writerow(["img_path", "status", "saved_path"])       # >>> ADDED

    for img_path in files:
        base = os.path.basename(img_path)
        img_entry = img_by_base.get(base)
        if img_entry is None:
            # Not present in annotations (either different split or not downloaded)
            skipped_no_ann += 1
            wlog.writerow([img_path, "no_annotation_match", ""])  # >>> ADDED
            continue

        try:
            has_person, out_path = process_one_image(
                img_path, img_entry, anns_by_img, cat_name, out_pos_dir, out_neg_dir
            )
            if has_person is None:
                # missing/corrupt/error already reported in process_one_image
                wlog.writerow([img_path, "error_or_missing", ""])  # >>> ADDED
                continue

            pos_count += int(has_person)
            neg_count += int(not has_person)
            processed += 1
            wlog.writerow([img_path, "positive" if has_person else "negative", out_path])  # >>> ADDED

        except Exception as e:
            # Final safety net — print the file that caused it
            print(f"[WARN] Failed on {img_path}: {e}")
            wlog.writerow([img_path, f"exception:{e}", ""])  # >>> ADDED

        if limit and processed >= limit:
            break

    logf.close()  # >>> ADDED

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"  Positives (has person): {pos_count}")
    print(f"  Negatives (no person):  {neg_count}")
    print(f"Skipped (no annotation match): {skipped_no_ann}")
    print(f"Positives saved to: {out_pos_dir}")
    print(f"Negatives saved to: {out_neg_dir}")
    print(f"Log written to: {log_path}")  # >>> ADDED


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_img_dir", type=str, default=TRAIN_IMG_DIR_DEFAULT,
                    help=r"Path to images, e.g. C:\Users\hbsu\fiftyone\coco-2017\train\data")
    ap.add_argument("--raw_dir", type=str, default=RAW_DIR_DEFAULT,
                    help=r"Path containing instances_train2017.json, e.g. C:\Users\hbsu\fiftyone\coco-2017\raw")
    ap.add_argument("--out_pos", type=str, default=OUT_POS_DEFAULT,
                    help=r"Output folder for positives (has person)")
    ap.add_argument("--out_neg", type=str, default=OUT_NEG_DEFAULT,
                    help=r"Output folder for negatives (no person)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional limit on number of images to process (for quick tests)")
    args = ap.parse_args()

    main(args.train_img_dir, args.raw_dir, args.out_pos, args.out_neg, args.limit)
