Setup Guide for ML-Stuff Project
Prerequisites
Python 3.9 or later but do not use 3.13 if on windows as pytorch is not supported for that version
pip (Python package manager)
Step 1: Clone/Download the Project
Download the project files or clone the repository to your local machine
Navigate to the project directory in your terminal
Step 2: Setting up Python Virtual Environment (Recommended)
Step 3: Install Required Dependencies
The project requires several Python packages. Install them using pip:

Step 4: Project Structure
The project has the following structure:

data_retrieval/ - Contains scripts for dataset management
installCoCo.py - Script to download and set up COCO dataset
dataset_split.py - Handles dataset splitting
sort_coco_dataset.py - Organizes COCO dataset
6layer_mobilenet.py - Contains the MobileNet model architecture
train.py - Main training script
Step 5: Download and Prepare Dataset
First, run the COCO dataset installation script:
This will download the COCO-2017 dataset (train split) with person detection annotations, it is recommended to limit the samples.

Then run installNegatives.py to install the negative data. You can customize the classes you want to include

Then run sort_coco_dataset.py. make sure the directories match where you pulling the pictures from: python sort_coco_dataset.py --train_img_dir ""C:\Users\anish\fiftyone\coco-2017\train\data"" --raw_dir ""C:\Users\anish\fiftyone\coco-2017\raw"" for example

Then run dataset_split.py. Here you can split the training, validation and test data. It is recommended to set training from 70-80% and allocate 15-20% for validation and set the rest to test.


Step 6: Training the Model
After the dataset is prepared, you can run the training script:

The training script follows this process:

Initializes dataset and epochs
Performs cross-validation splits
Executes training loop (if epochs > 0)
Saves and evaluates the model
Plots validation curves and cross-validation error summary
Notes
The project uses MobileNetV1 architecture for binary classification
Cross-validation is implemented for robust model evaluation
The training script includes optional quantization features
Troubleshooting
If you encounter any issues:

Make sure all dependencies are correctly installed
Verify that your Python environment is properly set up
Check that the COCO dataset was downloaded successfully
Ensure you have sufficient disk space for the dataset
This project appears to be focused on implementing a MobileNetV1-based binary classifier with cross-validation capabilities and optional model quantization features.
