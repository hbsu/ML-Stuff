# pip install fiftyone
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",                      # "train" or "validation"
    label_types=["detections"],
    classes=["person"],                 # only these classes
    # max_samples=500,                  # optional limit
)
