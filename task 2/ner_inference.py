import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple, Dict
import argparse
import os

# --- Class Configuration ---
# These variables are no longer used here as we load ID2LABEL from the model configuration.
# LABELS = ["O", "B-ANIMAL", "I-ANIMAL"]
# ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# --- Main Inference Function ---
def extract_animal_entities(text: str, model_path: str = 'ner_model') -> List[str]:
    """
    Loads the trained NER model and extracts ANIMAL entities from the text.

    Args:
        text: The input text string.
        model_path: Path to the directory containing the saved model and tokenizer.

    Returns:
        A list of extracted animal names.
    """
    # 1. Device Selection (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Check if the directory exists before loading
        full_model_path = os.path.abspath(model_path)
        if not os.path.exists(full_model_path):
            print(f"Error loading model. Directory {full_model_path} not found.")
            print("Ensure that training (ner_train.py) is complete.")
            return []

        # 2. Loading Tokenizer and Model
        print(f"Loading NER model from: {model_path} onto {device}...")
        # Use the provided path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        model.eval()

        # !!! FIX: LOAD ID2LABEL DIRECTLY FROM MODEL CONFIGURATION !!!
        ID2LABEL = model.config.id2label
        
    except Exception as e:
        # This catches other errors, e.g., corrupted files
        print("Error during model initialization.")
        print(f"Error details: {e}")
        return []

    # 3. Tokenizing the Input Text
    # Splitting by space simplifies word alignment for synthetic data
    tokenized_input = tokenizer(
        text.split(), 
        is_split_into_words=True, 
        return_tensors="pt", 
        truncation=True
    ).to(device)

    # Get word_ids for alignment
    word_ids = tokenized_input.word_ids()

    # 4. Model Inference
    with torch.no_grad():
        outputs = model(**tokenized_input)

    # 5. Output Processing
    predictions = torch.argmax(outputs.logits, dim=2)
    # ID2LABEL is now guaranteed to be correct for converting numeric ID to string label.
    predicted_labels = [ID2LABEL[p.item()] for p in predictions[0]]

    # 6. Entity Alignment and Reconstruction
    entities = []
    current_entity = []
    previous_word_idx = None

    # Get tokens corresponding to the input_ids
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])

    for token_idx, label_idx in enumerate(predicted_labels):
        word_idx = word_ids[token_idx]
        
        # Ignore special tokens (CLS, SEP, PAD) and internal None indices
        if word_idx is None:
            continue
            
        # current_label is the string label (B-ANIMAL, O, I-ANIMAL)
        current_label = label_idx
        current_token = tokens[token_idx]

        # Check if this is the start of a new word
        if word_idx != previous_word_idx:
            # If it's the beginning of an entity (B-ANIMAL)
            if current_label.startswith("B-"):
                # Save the previous entity if one was being tracked
                if current_entity:
                    entities.append("".join(current_entity).replace("##", ""))
                # Start of a new entity
                current_entity = [current_token]
            # If it's "O" and we just finished an entity
            elif current_label == "O" and current_entity:
                entities.append("".join(current_entity).replace("##", ""))
                current_entity = []
            # If it's "O" and we are not in an entity, do nothing
            else:
                current_entity = []

        # If we are inside the same word (subtoken) and it is part of an I-ANIMAL entity
        elif current_entity and current_label.startswith("I-"):
            current_entity.append(current_token)

        previous_word_idx = word_idx

    # Save the last entity if it ends at the end of the sentence
    if current_entity:
        entities.append("".join(current_entity).replace("##", ""))

    # Filter and return unique animal names
    # Remove [CLS], [SEP] and other metatokens that might have crept in due to alignment error
    final_entities = [e for e in entities if e and not e.startswith("[")]
    return list(set(final_entities))

# --- Command Line Arguments ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NER inference on a given text.")
    parser.add_argument('--text', type=str, required=True,
                        help='Input text message for entity extraction (e.g., "I see a dog and a horse").')
    # Use default value
    parser.add_argument('--model_path', type=str, default='ner_model',
                        help='Path to the directory containing the trained model.')
    
    args = parser.parse_args()
    
    # Example usage
    extracted_animals = extract_animal_entities(args.text, args.model_path)
    
    print("-" * 50)
    print(f"Input text: {args.text}")
    print(f"Extracted ANIMAL entities: {extracted_animals}")
    print("-" * 50)

# python ner_inference.py --text "There is a small cat hiding in the bushes."