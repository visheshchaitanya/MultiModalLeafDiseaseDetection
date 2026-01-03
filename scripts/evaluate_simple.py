"""
Simplified evaluation script matching train_simple.py approach.
Evaluates the trained model on the test set.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import MultiModalLeafDataset
from src.data.transforms import get_validation_transforms
from src.models.multimodal_model import MultiModalLeafDiseaseModel
from src.templates.vocabulary import load_vocabulary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")

def evaluate_model(model, test_loader, vocabulary, class_names, device):
    """Evaluate model on test set."""
    model.eval()

    all_predictions = []
    all_labels = []
    all_class_probs = []
    total_loss = 0.0

    vocab_size = len(vocabulary)
    classification_criterion = nn.CrossEntropyLoss()
    text_criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.get_word_index('<PAD>'))

    logger.info("Running evaluation...")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for batch in pbar:
            images = batch['image'].to(device)
            sensors = batch['sensors'].to(device)
            class_labels = batch['class_label'].to(device)
            text_targets = batch['text_ids'].to(device)

            # Forward pass
            outputs = model(images, sensors, text_targets[:, :-1])

            # Calculate losses
            class_loss = classification_criterion(outputs['class_logits'], class_labels)
            text_logits = outputs['text_logits']
            text_loss = text_criterion(
                text_logits.reshape(-1, vocab_size),
                text_targets[:, 1:].reshape(-1)
            )
            loss = 0.3 * class_loss + 0.7 * text_loss

            total_loss += loss.item()

            # Get predictions
            class_probs = torch.softmax(outputs['class_logits'], dim=1)
            _, predicted = outputs['class_logits'].max(1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(class_labels.cpu().numpy())
            all_class_probs.extend(class_probs.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_class_probs = np.array(all_class_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'class_probs': all_class_probs
    }

def main():
    # Fixed parameters (matching train_simple.py)
    BATCH_SIZE = 8
    DEVICE = 'cpu'  # Change to 'cuda' if GPU available
    IMAGE_SIZE = 224
    NUM_WORKERS = 0

    logger.info("=" * 80)
    logger.info("Multi-Modal Leaf Disease Detection - Simple Evaluation")
    logger.info("=" * 80)

    # Setup device
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path('outputs/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    logger.info("Loading vocabulary...")
    vocab_path = Path('data/processed/vocabulary.pkl')
    vocabulary = load_vocabulary(vocab_path)
    vocab_size = len(vocabulary)
    logger.info(f"Vocabulary loaded: {vocab_size} tokens")

    # Define class names
    class_names = ['Healthy', 'Alternaria', 'Stemphylium', 'Marssonina']
    num_classes = len(class_names)

    # Create transforms
    logger.info("Creating data transforms...")
    test_transform = get_validation_transforms(image_size=IMAGE_SIZE)

    # Load test CSV data
    logger.info("Loading test CSV data...")
    test_df = pd.read_csv('data/processed/test.csv')

    # Create test dataset
    logger.info("Creating test dataset...")
    test_dataset = MultiModalLeafDataset(
        data_df=test_df,
        vocabulary=vocabulary,
        image_transform=test_transform
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    logger.info(f"Test batches: {len(test_loader)}")

    # Create model (same architecture as train_simple.py)
    logger.info("Creating model...")
    model = MultiModalLeafDiseaseModel(
        vocab_size=vocab_size,
        num_classes=num_classes,
        image_encoder_config={
            'backbone': 'resnet50',
            'pretrained': True,
            'freeze_layers': 7,
            'output_dim': 512
        },
        tabular_encoder_config={
            'input_dim': 3,
            'hidden_dims': [64, 128],
            'output_dim': 128,
            'activation': 'relu',
            'dropout': 0.1
        },
        fusion_config={
            'image_dim': 512,
            'sensor_dim': 128,
            'hidden_dim': 512,
            'output_dim': 512,
            'dropout': 0.1
        },
        decoder_config={
            'num_layers': 4,
            'num_heads': 8,
            'embed_dim': 512,
            'ff_dim': 2048,
            'dropout': 0.1
        },
        classifier_config={
            'input_dim': 512,
            'hidden_dims': [256],
            'dropout': 0.3
        },
        pad_idx=vocabulary.get_word_index('<PAD>')
    )

    # Load checkpoint
    checkpoint_path = Path('outputs/checkpoints/best_model.pt')
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"  Training accuracy: {checkpoint['train_acc']:.2f}%")
    logger.info(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Starting Test Set Evaluation")
    logger.info("=" * 80 + "\n")

    results = evaluate_model(model, test_loader, vocabulary, class_names, device)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Test Loss:       {results['loss']:.4f}")
    logger.info(f"  Accuracy:        {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"  Precision (macro): {results['precision']:.4f}")
    logger.info(f"  Recall (macro):    {results['recall']:.4f}")
    logger.info(f"  F1-Score (macro):  {results['f1_macro']:.4f}")

    logger.info(f"\nPer-Class Metrics:")
    logger.info("-" * 80)
    logger.info(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    logger.info("-" * 80)
    for i, class_name in enumerate(class_names):
        logger.info(
            f"{class_name:<15} "
            f"{results['precision_per_class'][i]:<12.4f} "
            f"{results['recall_per_class'][i]:<12.4f} "
            f"{results['f1_per_class'][i]:<12.4f} "
            f"{results['support_per_class'][i]:<10.0f}"
        )
    logger.info("-" * 80)

    # Plot confusion matrix
    logger.info("\nGenerating confusion matrix plot...")
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)

    # Save detailed results
    results_path = output_dir / 'results.txt'
    with open(results_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Test Loss:         {results['loss']:.4f}\n")
        f.write(f"  Accuracy:          {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision (macro): {results['precision']:.4f}\n")
        f.write(f"  Recall (macro):    {results['recall']:.4f}\n")
        f.write(f"  F1-Score (macro):  {results['f1_macro']:.4f}\n\n")

        f.write(f"Per-Class Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(
                f"{class_name:<15} "
                f"{results['precision_per_class'][i]:<12.4f} "
                f"{results['recall_per_class'][i]:<12.4f} "
                f"{results['f1_per_class'][i]:<12.4f} "
                f"{results['support_per_class'][i]:<10.0f}\n"
            )
        f.write("-" * 80 + "\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
        f.write("\n")

    logger.info(f"Results saved to {results_path}")

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'true_class': [class_names[i] for i in results['labels']],
        'predicted_class': [class_names[i] for i in results['predictions']],
        'correct': results['labels'] == results['predictions']
    })

    # Add probability columns
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = results['class_probs'][:, i]

    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation completed successfully!")
    logger.info(f"All results saved to {output_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
