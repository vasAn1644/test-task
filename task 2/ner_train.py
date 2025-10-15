import argparse
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any

# Using Auto* for flexibility, as in the user's code
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- Class and Argument Configuration ---
# Classes we want to identify as ANIMAL
ANIMAL_CLASSES = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]
# IOB (Inside, Outside, Beginning) Scheme
LABELS = ["O", "B-ANIMAL", "I-ANIMAL"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS) # 3 labels

# --- Synthetic Data Generation ---
def create_synthetic_dataset(num_samples: int = 2000) -> List[Dict[str, Any]]:
    """Creates a simplified synthetic dataset for NER with IOB labels."""
    data = []
    
    # Sentence templates
    templates = [
        "I saw a beautiful [ANIMAL] at the zoo.",
        "The best pet is a [ANIMAL].",
        "Look at that huge [ANIMAL] running.",
        "A picture of a [ANIMAL].",
        "It is not a [ANIMAL], it is something else.",
        "My favorite animal is the [ANIMAL] and the color is brown.",
        "The [ANIMAL] is in the picture, but I don't see anything else.",
        "There is a [ANIMAL] in the picture.",
        "Is this a [ANIMAL]?",
        "This is [ANIMAL]",
        # Examples provided by the user
        "The small [ANIMAL] jumped over the fence.",
        "We need to find the [ANIMAL] in the forest.",
        "Do you have a [ANIMAL] at home?",
        "Watching the [ANIMAL] in its natural habitat was amazing."
    ]

    for i in range(num_samples):
        # Randomly select an animal and a template
        animal = np.random.choice(ANIMAL_CLASSES)
        template = np.random.choice(templates)
        
        sentence = template.replace("[ANIMAL]", animal)
        tokens = sentence.split()
        
        # Creating IOB labels
        labels = []
        
        for token in tokens:
            # Remove punctuation for correct comparison
            if token.strip(".,!?").lower() == animal:
                # Beginning of entity (B-ANIMAL)
                labels.append(LABEL2ID["B-ANIMAL"])
            else:
                # Outside of entity (O)
                labels.append(LABEL2ID["O"])
        
        # Adding data (format required for Hugging Face)
        data.append({
            "id": str(i),
            "tokens": tokens,
            "ner_tags": labels
        })
        
    return data

# --- Data Preprocessing ---
def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes the inputs and aligns IOB labels with WordPiece tokens.
    FIX: Correctly converts B-ANIMAL -> I-ANIMAL for sub-token parts.
    """
    # Do not use padding here; DataCollatorForTokenClassification handles this dynamically
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Special tokens (CLS, SEP, PAD) receive the label -100
            if word_idx is None:
                label_ids.append(-100)
            
            # This is the first subtoken of a new word
            elif word_idx != previous_word_idx:
                # Take the original label (O or B-ANIMAL)
                label_ids.append(label[word_idx])
            
            # This is a continuation subtoken of the same word (e.g., "##phant")
            else:
                original_label_id = label[word_idx]
                
                if original_label_id == LABEL2ID["B-ANIMAL"]:
                    # If the original label was B-ANIMAL, we use I-ANIMAL for the word continuation
                    label_ids.append(LABEL2ID["I-ANIMAL"])
                else:
                    # If the original label was O, we leave it as O
                    label_ids.append(original_label_id)
            
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    """Computes NER metrics using seqeval."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove tokens with the special label -100
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # seqeval requires labels to be strings
    results = {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions)
    }
    return results

# --- Main Training Function ---
def train_ner_model():
    """Initializes data, model, Trainer, and runs NER training."""
    parser = argparse.ArgumentParser(description="Train a NER model for animal entity extraction.")
    parser.add_argument('--model_checkpoint', type=str, default="distilbert-base-uncased",
                        help='Pre-trained transformer model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='./ner_model',
                        help='Output directory for the trained model.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training.')
    args = parser.parse_args()

    # 1. Data Creation and Splitting
    print("1. Creating and splitting synthetic dataset (2000 samples)...")
    synthetic_data = create_synthetic_dataset(num_samples=2000)
    dataset = Dataset.from_list(synthetic_data)
    
    # Splitting into training and test sets
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    raw_datasets = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test'],
    })
    
    # 2. Tokenizer and Model Initialization
    print("2. Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 3. Tokenization and Label Alignment
    print("3. Tokenizing and aligning labels...")
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        # Safely remove columns that are no longer needed
        remove_columns=raw_datasets["train"].column_names
    )

    # 4. Training Parameter Setup
    print("4. Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    # 5. Creating Trainer and starting training
    print("5. Initializing Trainer...")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting NER model training...")
    trainer.train()

    # Saving the model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("-" * 50)
    print(f"Training complete. NER model saved to: {args.output_dir}")
    print("-" * 50)

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")
        
    train_ner_model()