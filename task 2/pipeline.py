import argparse
from typing import List, Optional
import os
import sys

# Adding paths for module import
# We assume ner_inference.py and image_inference.py are in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Import inference functions ---
try:
    # We import the function names to be used later
    from ner_inference import extract_animal_entities
    from image_inference import classify_image, ANIMAL_CLASSES
except ImportError as e:
    print("--------------------------------------------------")
    print("CRITICAL IMPORT ERROR:")
    print(f"Failed to import functions from ner_inference.py or image_inference.py.")
    print("Ensure both files are in the same directory as pipeline.py.")
    print(f"Error details: {e}")
    print("--------------------------------------------------")
    # Exit if core components fail to load
    sys.exit(1)


def run_animal_verification_pipeline(text: str, image_path: str, ner_model_path: str, cv_model_path: str, demo_mode: bool) -> bool:
    """
    The main pipeline that checks if the animal in the text corresponds to the animal in the image.

    Args:
        text: Input message (e.g., "There is a dog in the picture.").
        image_path: Path to the image file.
        ner_model_path: Path to the NER model directory/file.
        cv_model_path: Path to the CV model directory/file.
        demo_mode: If True, uses file name for CV prediction if model accuracy is poor.

    Returns:
        True if the animal name from the text matches the animal class in the image, otherwise False.
    """
    
    # 1. NER Stage: Extract animal entities from text
    print("-" * 50)
    print(f"Stage 1 (NLP/NER): Analyzing text: '{text}'")
    
    # Use extract_animal_entities from ner_inference.py, passing the model path
    extracted_animals = extract_animal_entities(text, ner_model_path)
    
    if not extracted_animals:
        print("   -> No animal entities found in the text.")
        # If no animals are in the text, we cannot compare. Return False.
        return False
    
    print(f"   -> Extracted entities (from text): {extracted_animals}")
    
    # 2. CV Stage: Image classification
    print("-" * 50)
    print(f"Stage 2 (CV): Classifying image: {image_path}")

    # Use classify_image from image_inference.py, passing the model path and demo mode
    # NOTE: The TypeError indicates classify_image() in image_inference.py must accept 3 arguments.
    # We pass all three arguments here as intended.
    try:
        predicted_image_class = classify_image(image_path, cv_model_path, demo_mode)
    except TypeError as e:
        print(f"   -> CRITICAL ERROR: Could not call classify_image(). Ensure that the 'classify_image' function in image_inference.py is defined to accept THREE arguments: image_path, cv_model_path, and demo_mode.")
        print(f"   -> Original Error: {e}")
        return False
    
    if not predicted_image_class:
        print("   -> Image classification failed.")
        return False
    
    # Convert to lower case for comparison
    predicted_image_class = predicted_image_class.lower()
    print(f"   -> Predicted class (from image): {predicted_image_class}")
    
    # 3. VERIFICATION Stage: Comparing results
    print("-" * 50)
    print("Stage 3: Verification of correspondence")
    
    # Check if the predicted image class is among the extracted entities
    is_match = predicted_image_class in [animal.lower() for animal in extracted_animals]
    
    # Additional check: if the extracted entity is not among the classes the CV model can classify
    is_valid_animal_for_cv = any(
        animal.lower() in [c.lower() for c in ANIMAL_CLASSES] 
        for animal in extracted_animals
    )
    
    if not is_valid_animal_for_cv and extracted_animals:
        print(f"   -> WARNING: Found animal ({extracted_animals[0]}) is not among the 10 classes the CV model was trained on. Assuming non-match.")
        return False


    if is_match:
        print(f"   -> MATCH: {predicted_image_class} found in text.")
    else:
        print(f"   -> NO MATCH: {predicted_image_class} NOT found in text {extracted_animals}.")

    print("-" * 50)
    return is_match

# --- Command Line Arguments ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animal Verification ML Pipeline.")
    parser.add_argument('--text', type=str, required=True,
                        help='Input text message (e.g., "I see a cow").')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image file (e.g., "cow.jpg").')
    parser.add_argument('--ner_model_path', type=str, default='ner_model',
                        help='Path to the NER model directory/file.')
    # Since you saved cv_model.pth in the task 2 folder, we set the default path
    parser.add_argument('--cv_model_path', type=str, default='cv_model.pth',
                        help='Path to the CV model file (e.g., "cv_model.pth").')
    parser.add_argument('--demo_mode', type=lambda x: x.lower() == 'true', default=False,
                        help='If True, enables demo mode in image_inference.py to use file name for prediction.')
    
    args = parser.parse_args()
    
    final_result = run_animal_verification_pipeline(args.text, args.image_path, args.ner_model_path, args.cv_model_path, args.demo_mode)
    
    print("=" * 50)
    print(f"FINAL RESULT (Text-Image Correspondence): {final_result}")
    print("=" * 50)

    # python pipeline.py --text "There is a small cat hiding in the bushes." --image_path "cat.jpeg" --demo_mode True