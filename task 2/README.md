Task 2: Text-Image Mismatch Pipeline

This project implements a comprehensive ML pipeline to verify whether the text correctly describes the animal shown in the image. The system combines two models:
NER (Transformer): for extracting the animal's name from the text query.
CNN (ResNet18): for classifying the animal in the image.
The result is a boolean decision (True/False) regarding the match between the text and the image.

1. Architecture and Components

pipeline.py - Combines NER and CV, returns the final result.

ner_train.py - Trains DistilBERT for Named Entity Recognition (animals).

cv_train.py - Trains ResNet18 for image classification.

split_dataset.py - Splits source images into train (80%) and val (20%) folders.

eda_dataset.ipynb.md - Notebook for checking class balance and data quality.

2. Dataset Setup

Create Directory: Create the ./dataset/ folder.
Structure: Place images in corresponding class subfolders within ./dataset/ (e.g., ./dataset/dog/, ./dataset/cat/).
10 Classes: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, spider.

3. Execution Instructions (3 Steps)

Step 1: Data Preparation and Analysis (EDA)

    # 1. Split data into train (80%) and val (20%). Creates necessary folders.
    python split_dataset.py --ratio 0.8 

    # 2. Check class balance and image quality
    jupyter notebook eda_dataset.ipynb.md


Step 2: Model Training (CV and NER)

    Both models must be trained and saved.

    # 1. Train the image classifier (CV)
    # Saves weights to ./cv_model.pth
    python cv_train.py --epochs 10 --model_path ./cv_model.pth

    # 2. Train NER (NLP)
    # Saves model and tokenizer to ./ner_model/
    python ner_train.py --epochs 5 --output_dir ./ner_model


Step 3: Run the Pipeline

    Perform the final check. Update the IMAGE_TO_TEST and TEXT_TO_TEST variables in the pipeline.py file before running.
    python pipeline.py
