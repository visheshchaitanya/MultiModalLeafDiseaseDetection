"""
Simplified training script with direct initialization.
This bypasses complex config loading to get training running quickly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from datetime import datetime
import pandas as pd

from src.data.dataset import MultiModalLeafDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.multimodal_model import MultiModalLeafDiseaseModel
from src.templates.vocabulary import load_vocabulary
from src.utils.seed import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    # Fixed parameters
    EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    DEVICE = 'cpu'  # Change to 'cuda' if GPU available
    IMAGE_SIZE = 224
    NUM_WORKERS = 0  # Set to 0 for Windows compatibility

    logger.info("=" * 80)
    logger.info("Multi-Modal Leaf Disease Detection - Simplified Training")
    logger.info("=" * 80)

    # Set seed
    set_seed(42)
    logger.info("Random seed set to 42")

    # Setup device
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path('outputs')
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    train_transform = get_training_transforms(image_size=IMAGE_SIZE)
    val_transform = get_validation_transforms(image_size=IMAGE_SIZE)

    # Load CSV data
    logger.info("Loading CSV data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = MultiModalLeafDataset(
        data_df=train_df,
        vocabulary=vocabulary,
        image_transform=train_transform
    )

    val_dataset = MultiModalLeafDataset(
        data_df=val_df,
        vocabulary=vocabulary,
        image_transform=val_transform
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model with direct parameters
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

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    text_criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.get_word_index('<PAD>'))

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        logger.info("=" * 80)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        logger.info("=" * 80)

        # Training phase
        model.train()
        train_loss = 0.0
        train_class_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(device)
            sensors = batch['sensors'].to(device)
            class_labels = batch['class_label'].to(device)
            text_targets = batch['text_ids'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, sensors, text_targets[:, :-1])

            # Calculate losses
            class_loss = classification_criterion(outputs['class_logits'], class_labels)

            # Text generation loss
            text_logits = outputs['text_logits']
            text_loss = text_criterion(
                text_logits.reshape(-1, vocab_size),
                text_targets[:, 1:].reshape(-1)
            )

            # Combined loss
            loss = 0.3 * class_loss + 0.7 * text_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs['class_logits'].max(1)
            train_total += class_labels.size(0)
            train_class_correct += predicted.eq(class_labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * train_class_correct / train_total:.2f}%"
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_class_correct / train_total

        logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_class_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for batch in pbar:
                images = batch['image'].to(device)
                sensors = batch['sensors'].to(device)
                class_labels = batch['class_label'].to(device)
                text_targets = batch['text_ids'].to(device)

                outputs = model(images, sensors, text_targets[:, :-1])

                class_loss = classification_criterion(outputs['class_logits'], class_labels)
                text_logits = outputs['text_logits']
                text_loss = text_criterion(
                    text_logits.reshape(-1, vocab_size),
                    text_targets[:, 1:].reshape(-1)
                )

                loss = 0.3 * class_loss + 0.7 * text_loss

                val_loss += loss.item()
                _, predicted = outputs['class_logits'].max(1)
                val_total += class_labels.size(0)
                val_class_correct += predicted.eq(class_labels).sum().item()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * val_class_correct / val_total:.2f}%"
                })

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_class_correct / val_total

        logger.info(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Update scheduler
        scheduler.step()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, checkpoint_path)
            logger.info(f"✓ Best model saved to {checkpoint_path}")

        # Save last checkpoint
        last_checkpoint_path = checkpoint_dir / 'last_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, last_checkpoint_path)

        logger.info("-" * 80)

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved at: {checkpoint_dir / 'best_model.pt'}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
