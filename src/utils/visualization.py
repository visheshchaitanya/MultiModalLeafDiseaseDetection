"""
Visualization utilities for the multi-modal leaf disease detection system.
Provides functions for plotting training curves, images, attention maps, and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        train_metrics: Dictionary of training metrics {metric_name: [values]}
        val_metrics: Dictionary of validation metrics {metric_name: [values]}
        save_path: Path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    num_plots = 1 + (len(train_metrics) if train_metrics else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metrics
    if train_metrics:
        for idx, (metric_name, train_values) in enumerate(train_metrics.items(), start=1):
            axes[idx].plot(epochs, train_values, 'b-', label=f'Train {metric_name}', linewidth=2)

            if val_metrics and metric_name in val_metrics:
                val_values = val_metrics[metric_name]
                axes[idx].plot(epochs, val_values, 'r-', label=f'Val {metric_name}', linewidth=2)

            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name.capitalize())
            axes[idx].set_title(f'{metric_name.capitalize()} over Epochs')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_image_grid(
    images: Union[torch.Tensor, np.ndarray],
    titles: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    predictions: Optional[List[str]] = None,
    nrows: int = 2,
    ncols: int = 4,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a grid of images.

    Args:
        images: Tensor or array of images (N, C, H, W) or (N, H, W, C)
        titles: Optional list of titles for each image
        labels: Optional list of ground truth labels
        predictions: Optional list of predicted labels
        nrows: Number of rows in grid
        ncols: Number of columns in grid
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    # Handle different image formats
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        # (N, C, H, W) -> (N, H, W, C)
        images = np.transpose(images, (0, 2, 3, 1))

    # Normalize to [0, 1] if needed
    if images.max() > 1.0:
        images = images / 255.0

    num_images = min(len(images), nrows * ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num_images):
        img = images[idx]

        # Handle grayscale
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            axes[idx].imshow(img, cmap='gray')
        else:
            axes[idx].imshow(img)

        # Set title
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])
        elif labels and predictions:
            if idx < len(labels) and idx < len(predictions):
                color = 'green' if labels[idx] == predictions[idx] else 'red'
                axes[idx].set_title(f'GT: {labels[idx]}\nPred: {predictions[idx]}', color=color)
        elif labels and idx < len(labels):
            axes[idx].set_title(f'Label: {labels[idx]}')

        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Image grid saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,
    tokens: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weight matrix (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        tokens: Optional list of token labels
        save_path: Path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if attention_weights.ndim == 3:
        # Average across attention heads
        attention_weights = attention_weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        attention_weights,
        cmap='viridis',
        xticklabels=tokens if tokens else range(attention_weights.shape[1]),
        yticklabels=tokens if tokens else range(attention_weights.shape[0]),
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Attention Weights Heatmap')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Attention weights saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_grad_cam(
    image: Union[torch.Tensor, np.ndarray],
    heatmap: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.4,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot Grad-CAM visualization overlaid on the original image.

    Args:
        image: Original image (H, W, C) or (C, H, W)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Transparency for overlay
        save_path: Path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()

    # Handle image format
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # Normalize image
    if image.max() > 1.0:
        image = image / 255.0

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap='jet', alpha=alpha)
    axes[2].set_title('Grad-CAM Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Grad-CAM visualization saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_sensor_distribution(
    sensor_data: Dict[str, np.ndarray],
    class_labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot sensor data distribution by class.

    Args:
        sensor_data: Dictionary of sensor features {feature_name: values}
        class_labels: Array of class labels
        class_names: List of class names
        save_path: Path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    num_features = len(sensor_data)
    fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 5))

    if num_features == 1:
        axes = [axes]

    for idx, (feature_name, values) in enumerate(sensor_data.items()):
        for class_idx, class_name in enumerate(class_names):
            class_mask = class_labels == class_idx
            class_values = values[class_mask]

            axes[idx].hist(class_values, alpha=0.5, label=class_name, bins=30)

        axes[idx].set_xlabel(feature_name.capitalize())
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{feature_name.capitalize()} Distribution by Class')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sensor distribution saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


if __name__ == "__main__":
    # Test visualization functions
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing visualization utilities...")

    # Test training curves
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
    val_losses = [0.9, 0.7, 0.5, 0.4, 0.3]
    train_metrics = {'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9]}
    val_metrics = {'accuracy': [0.55, 0.65, 0.75, 0.8, 0.85]}

    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, show=False)
    print("Training curves plotted successfully")

    # Test confusion matrix
    cm = np.array([[50, 2, 3, 1], [5, 45, 3, 2], [2, 3, 48, 1], [1, 2, 2, 50]])
    class_names = ['Healthy', 'Alternaria', 'Stemphylium', 'Marssonina']

    plot_confusion_matrix(cm, class_names, show=False)
    print("Confusion matrix plotted successfully")

    # Test image grid
    images = np.random.rand(8, 224, 224, 3)
    labels = ['Healthy', 'Alternaria', 'Healthy', 'Stemphylium', 'Marssonina', 'Healthy', 'Alternaria', 'Marssonina']
    predictions = ['Healthy', 'Alternaria', 'Alternaria', 'Stemphylium', 'Marssonina', 'Healthy', 'Alternaria', 'Healthy']

    plot_image_grid(images, labels=labels, predictions=predictions, show=False)
    print("Image grid plotted successfully")

    print("\nAll visualization tests passed!")
