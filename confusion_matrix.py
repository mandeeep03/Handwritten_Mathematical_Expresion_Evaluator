import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

from model import SymbolCNN, NUM_CLASSES, SYMBOL_CLASSES
from train import EMNISTDigitDataset, EMNISTLetterDataset, OperatorDataset

def generate_and_plot_confusion_matrix():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = 'symbol_cnn.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model weights '{model_path}' not found. Please train the model first.")
        return

    # Initialize and load model
    print("Loading model...")
    model = SymbolCNN(num_classes=NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("\n" + "="*60)
            print("ERROR: Model Architecture Mismatch")
            print("="*60)
            print(f"The saved model '{model_path}' was trained with a different number of classes")
            print(f"than what is currently defined in model.py (NUM_CLASSES = {NUM_CLASSES}).")
            print("Please retrain your model by running: python train.py")
            print("="*60 + "\n")
            return
        else:
            raise e
    model.eval()

    # Load validation data
    print("Loading validation datasets (this might take a moment)...")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    val_digits = EMNISTDigitDataset(train=False, transform=val_transform, max_per_digit=800)
    val_letters = EMNISTLetterDataset(train=False, transform=val_transform, max_per_letter=600)
    val_operators = OperatorDataset(num_samples_per_class=600, transform=val_transform)
    
    val_dataset = ConcatDataset([val_digits, val_letters, val_operators])
    print(f"Total validation samples: {len(val_dataset)}")
    
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Calculating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))

    # Plot the confusion matrix
    plt.figure(figsize=(14, 12))
    
    # We use a white background for seaborn heatmap
    sns.set_theme(style="whitegrid")
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=SYMBOL_CLASSES, yticklabels=SYMBOL_CLASSES,
                     cbar_kws={'label': 'Number of Samples'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Symbol Classifier', fontsize=16, fontweight='bold')
    
    # Rotate tick labels for better visibility
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Confusion matrix saved as '{save_path}'")
    
    # Show the plot if running interactively
    try:
        plt.show()
    except Exception as e:
        print("Note: Could not display plot interactively in this environment. Please check the saved image file.")

if __name__ == '__main__':
    generate_and_plot_confusion_matrix()
