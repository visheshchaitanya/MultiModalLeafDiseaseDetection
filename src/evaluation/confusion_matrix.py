"""
Confusion matrix utilities and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot confusion matrix with annotations.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        normalize: Whether to normalize by true labels
        title: Plot title
        cmap: Colormap name
        figsize: Figure size (width, height)
        save_path: Path to save figure (optional)
        show: Whether to display plot
    """
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2%'
        vmax = 1.0
    else:
        cm_display = cm
        fmt = 'd'
        vmax = None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_normalized_confusion_matrices(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot both raw and normalized confusion matrices side by side.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save figure (optional)
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw confusion matrix
    cm_raw = cm
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Raw Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        vmin=0,
        vmax=1.0,
        linewidths=0.5,
        linecolor='gray',
        ax=axes[1]
    )
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_title('Normalized Confusion Matrix (by True Label)', fontsize=14, fontweight='bold', pad=20)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def analyze_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> dict:
    """
    Analyze confusion matrix and extract insights.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names

    Returns:
        Dictionary with analysis results
    """
    num_classes = len(class_names)
    total_samples = cm.sum()

    # Per-class statistics
    per_class_stats = {}

    for i, class_name in enumerate(class_names):
        true_positives = cm[i, i]
        false_negatives = cm[i, :].sum() - true_positives
        false_positives = cm[:, i].sum() - true_positives
        true_negatives = total_samples - true_positives - false_negatives - false_positives

        # Metrics
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        specificity = true_negatives / (true_negatives + false_positives + 1e-10)

        per_class_stats[class_name] = {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'support': int(cm[i, :].sum())
        }

    # Most common misclassifications
    misclassifications = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(cm[i, j]),
                    'percentage': float(cm[i, j] / cm[i, :].sum() * 100)
                })

    # Sort by count
    misclassifications.sort(key=lambda x: x['count'], reverse=True)

    # Overall accuracy
    accuracy = np.trace(cm) / total_samples

    analysis = {
        'accuracy': float(accuracy),
        'total_samples': int(total_samples),
        'per_class_stats': per_class_stats,
        'top_misclassifications': misclassifications[:10],  # Top 10
    }

    return analysis


def print_confusion_matrix_analysis(analysis: dict):
    """
    Print confusion matrix analysis in a readable format.

    Args:
        analysis: Analysis dictionary from analyze_confusion_matrix
    """
    print("=" * 80)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 80)

    print(f"\nOverall Accuracy: {analysis['accuracy']:.4f}")
    print(f"Total Samples: {analysis['total_samples']}")

    print("\n" + "-" * 80)
    print("PER-CLASS STATISTICS")
    print("-" * 80)

    for class_name, stats in analysis['per_class_stats'].items():
        print(f"\n{class_name}:")
        print(f"  Support: {stats['support']}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall: {stats['recall']:.4f}")
        print(f"  F1-Score: {stats['f1_score']:.4f}")
        print(f"  Specificity: {stats['specificity']:.4f}")
        print(f"  True Positives: {stats['true_positives']}")
        print(f"  False Positives: {stats['false_positives']}")
        print(f"  False Negatives: {stats['false_negatives']}")

    if analysis['top_misclassifications']:
        print("\n" + "-" * 80)
        print("TOP MISCLASSIFICATIONS")
        print("-" * 80)

        for i, misc in enumerate(analysis['top_misclassifications'][:5], 1):
            print(f"\n{i}. {misc['true_class']} → {misc['predicted_class']}")
            print(f"   Count: {misc['count']} ({misc['percentage']:.2f}% of true {misc['true_class']})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test confusion matrix utilities
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Confusion Matrix Utilities...")
    print("=" * 80)

    # Create sample confusion matrix
    class_names = ['Healthy', 'Alternaria', 'Stemphylium', 'Marssonina']
    num_classes = len(class_names)

    # Simulated confusion matrix (good performance)
    cm = np.array([
        [45, 2, 1, 2],   # Healthy
        [3, 42, 3, 2],   # Alternaria
        [1, 2, 44, 3],   # Stemphylium
        [2, 1, 2, 45]    # Marssonina
    ])

    print("\nConfusion Matrix:")
    print(cm)

    # Test 1: Analyze confusion matrix
    print("\n\nTest 1: Analyze confusion matrix")
    print("-" * 80)

    analysis = analyze_confusion_matrix(cm, class_names)
    print_confusion_matrix_analysis(analysis)

    # Test 2: Plot confusion matrix
    print("\n\nTest 2: Plot confusion matrix")
    print("-" * 80)

    output_dir = Path("outputs/test_cm_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPlotting raw confusion matrix...")
    plot_confusion_matrix(
        cm,
        class_names,
        normalize=False,
        title='Raw Confusion Matrix',
        save_path=output_dir / 'cm_raw.png',
        show=False
    )

    print("Plotting normalized confusion matrix...")
    plot_confusion_matrix(
        cm,
        class_names,
        normalize=True,
        title='Normalized Confusion Matrix',
        save_path=output_dir / 'cm_normalized.png',
        show=False
    )

    # Test 3: Plot both matrices side by side
    print("\n\nTest 3: Plot side-by-side comparison")
    print("-" * 80)

    plot_normalized_confusion_matrices(
        cm,
        class_names,
        save_path=output_dir / 'cm_comparison.png',
        show=False
    )

    print(f"\nPlots saved to {output_dir}")

    # Test 4: Poor performance example
    print("\n\nTest 4: Analyze poor performance confusion matrix")
    print("-" * 80)

    # Simulated confusion matrix (poor performance with common misclassifications)
    cm_poor = np.array([
        [30, 10, 5, 5],   # Healthy
        [8, 25, 10, 7],   # Alternaria
        [6, 12, 22, 10],  # Stemphylium
        [4, 8, 8, 30]     # Marssonina
    ])

    analysis_poor = analyze_confusion_matrix(cm_poor, class_names)
    print_confusion_matrix_analysis(analysis_poor)

    print("\n" + "=" * 80)
    print("Confusion matrix utilities tests completed!")
