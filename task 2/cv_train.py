import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

# --- PATH FIX ---
# This variable gets the path to the folder where cv_train.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'dataset')
# --------------------

# Definition of the classes you provided
CLASSES = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]
NUM_CLASSES = len(CLASSES)

# --- Command Line Argument Configuration ---
def parse_args():
    """Parses command line arguments to parameterize training."""
    parser = argparse.ArgumentParser(description="Train a CNN model for animal classification.")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, # USING THE NEW PATH
                        help='Root directory for the dataset (must contain train/val subdirectories).')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help='Pre-trained model to use (e.g., resnet18, alexnet).')
    parser.add_argument('--output_path', type=str, default=os.path.join(BASE_DIR, 'cv_model.pth'),
                        help='Path to save the trained model checkpoint.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer.')
    return parser.parse_args()

# --- Model Setup ---
def initialize_model(model_name: str, num_classes: int):
    """Initializes and modifies a pre-trained model (transfer learning)."""
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze all layers except the last one
        for param in model.parameters():
            param.requires_grad = False
        # Replace the last fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model

# --- Main Training Function ---
def train_model():
    """Main function for training the image classification model."""
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading data from: {args.data_dir}") # Add path printing for verification
    
    # 1. Transformations for data preparation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # Augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Data Loading
    # Assuming folder structure: data_dir/train/dog, data_dir/val/dog, etc.
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                                  data_transforms[x])
                              for x in ['train', 'val']}
    except Exception as e:
        print("\n" + "="*80)
        print("DATA LOADING ERROR (ImageFolder Error):")
        print(f"The script could not find or read folders: {args.data_dir}\\{x}")
        print("1. CHECK: Do 'train' and 'val' folders exist and do they contain the class folders ('dog', 'cat', etc.).")
        print("2. CHECK: Do the class folders contain non-image files (e.g., .DS_Store, .gitkeep, .txt).")
        print(f"Error details: {e}")
        print("="*80 + "\n")
        return # Stop execution

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                                  shuffle=True, num_workers=0) # !!! FIX num_workers=0 FOR WINDOWS !!!
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # Class verification
    class_names = image_datasets['train'].classes
    if set(class_names) != set(CLASSES):
            print(f"Warning: Dataset classes {class_names} do not match hardcoded classes {CLASSES}")
    
    print(f"Dataset size (train): {dataset_sizes['train']}, (val): {dataset_sizes['val']}")
    
    # 3. Model, Optimizer, and Criterion Initialization
    model = initialize_model(args.model_name, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    # Optimize only the parameters we unfroze (i.e., the last fc layer)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # 4. Training Loop
    print("Starting training...")
    # ... (the rest of the training logic remains unchanged) ...
    for epoch in range(args.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{args.epochs} ({phase})"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimization only in the 'train' phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 5. Model Saving
    torch.save(model.state_dict(), args.output_path)
    print(f"\nTraining finished. Model saved to {args.output_path}")

if __name__ == '__main__':
    train_model()
