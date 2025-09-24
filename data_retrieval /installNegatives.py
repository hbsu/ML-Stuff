import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(

    "coco-2017",
    splits=["train"],
    classes=["tv", "chair", 
            "potted plant", "toilet", "sink", 
            "refrigerator", "book", "bed", "bench"],
    max_samples = 7000,
    drop_exisiting_dataset=True,
    overwrite=True,

)
