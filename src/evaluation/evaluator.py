"""
Main evaluator for multi-modal leaf disease detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import json
import logging

from .metrics import ClassificationMetrics
from .bleu_scorer import BLEUScorer
from .rouge_scorer import ROUGEScorer
from .confusion_matrix import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrices,
    analyze_confusion_matrix,
    print_confusion_matrix_analysis
)
from ..utils.device import get_device, move_to_device

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for multi-modal model.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        vocabulary,
        class_names: List[str],
        device: Optional[torch.device] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            test_loader: Test data loader
            vocabulary: Vocabulary for text decoding
            class_names: List of class names
            device: Device to run evaluation on
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.test_loader = test_loader
        self.vocabulary = vocabulary
        self.class_names = class_names
        self.device = device or get_device()
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/evaluation')

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.classification_metrics = ClassificationMetrics(
            num_classes=len(class_names),
            class_names=class_names
        )
        self.bleu_scorer = BLEUScorer(max_n=4, smooth=True)
        self.rouge_scorer = ROUGEScorer()

        logger.info(f"Evaluator initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output directory: {self.output_dir}")

    @torch.no_grad()
    def evaluate(
        self,
        compute_text_metrics: bool = True,
        save_predictions: bool = True,
        plot_cm: bool = True
    ) -> Dict:
        """
        Run full evaluation.

        Args:
            compute_text_metrics: Whether to compute text generation metrics
            save_predictions: Whether to save predictions to file
            plot_cm: Whether to plot confusion matrix

        Returns:
            Dictionary of all metrics
        """
        logger.info("=" * 80)
        logger.info("Starting evaluation")
        logger.info("=" * 80)

        # Reset metrics
        self.classification_metrics.reset()
        self.bleu_scorer.reset()
        self.rouge_scorer.reset()

        # Storage for predictions
        all_predictions = []

        # Run inference
        logger.info("Running inference on test set...")
        pbar = tqdm(self.test_loader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = move_to_device(batch, self.device)

            # Forward pass
            outputs = self.model(
                images=batch['images'],
                sensors=batch['sensors'],
                text_ids=batch.get('text_ids')
            )

            # Classification predictions
            class_predictions = torch.argmax(outputs['class_logits'], dim=1)

            # Update classification metrics
            self.classification_metrics.update(
                predictions=class_predictions,
                targets=batch['class_labels']
            )

            # Text generation (if enabled)
            if compute_text_metrics and 'text_logits' in outputs:
                # Decode generated text
                generated_texts = self._decode_predictions(outputs['text_logits'])

                # Decode reference text
                reference_texts = self._decode_targets(batch['text_ids'])

                # Update text metrics
                for gen_text, ref_text in zip(generated_texts, reference_texts):
                    self.bleu_scorer.add(gen_text, ref_text)
                    self.rouge_scorer.add(gen_text, ref_text)

                # Store predictions
                if save_predictions:
                    for i in range(len(class_predictions)):
                        all_predictions.append({
                            'true_class': self.class_names[batch['class_labels'][i].item()],
                            'predicted_class': self.class_names[class_predictions[i].item()],
                            'generated_text': generated_texts[i],
                            'reference_text': reference_texts[i]
                        })
            else:
                # Store predictions (classification only)
                if save_predictions:
                    for i in range(len(class_predictions)):
                        all_predictions.append({
                            'true_class': self.class_names[batch['class_labels'][i].item()],
                            'predicted_class': self.class_names[class_predictions[i].item()]
                        })

        # Compute metrics
        logger.info("\nComputing metrics...")
        results = self._compute_all_metrics(compute_text_metrics)

        # Confusion matrix
        cm = self.classification_metrics.get_confusion_matrix()
        cm_analysis = analyze_confusion_matrix(cm, self.class_names)
        results['confusion_matrix_analysis'] = cm_analysis

        # Print results
        self._print_results(results)

        # Save results
        self._save_results(results, all_predictions if save_predictions else None)

        # Plot confusion matrix
        if plot_cm:
            self._plot_confusion_matrix(cm)

        logger.info("=" * 80)
        logger.info("Evaluation complete!")
        logger.info("=" * 80)

        return results

    def _compute_all_metrics(self, compute_text_metrics: bool) -> Dict:
        """
        Compute all metrics.

        Args:
            compute_text_metrics: Whether to compute text metrics

        Returns:
            Dictionary of all metrics
        """
        results = {}

        # Classification metrics
        classification_results = self.classification_metrics.compute(average='macro')
        results.update(classification_results)

        # Per-class metrics
        per_class_metrics = self.classification_metrics.compute_per_class_metrics()
        results['per_class_metrics'] = per_class_metrics

        # Text generation metrics
        if compute_text_metrics:
            bleu_scores = self.bleu_scorer.compute(return_all=True)
            rouge_scores = self.rouge_scorer.compute()

            results.update(bleu_scores)
            results.update(rouge_scores)

        return results

    def _decode_predictions(self, text_logits: torch.Tensor) -> List[str]:
        """
        Decode text predictions from logits.

        Args:
            text_logits: Text logits (B, seq_len, vocab_size)

        Returns:
            List of decoded texts
        """
        # Get predicted indices
        predicted_indices = torch.argmax(text_logits, dim=-1)  # (B, seq_len)

        # Decode each sequence
        decoded_texts = []
        for indices in predicted_indices:
            text = self.vocabulary.decode(indices.cpu().tolist(), skip_special_tokens=True)
            decoded_texts.append(text)

        return decoded_texts

    def _decode_targets(self, text_ids: torch.Tensor) -> List[str]:
        """
        Decode target text from token IDs.

        Args:
            text_ids: Target token IDs (B, seq_len)

        Returns:
            List of decoded texts
        """
        decoded_texts = []
        for indices in text_ids:
            text = self.vocabulary.decode(indices.cpu().tolist(), skip_special_tokens=True)
            decoded_texts.append(text)

        return decoded_texts

    def _print_results(self, results: Dict):
        """
        Print evaluation results.

        Args:
            results: Results dictionary
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        # Classification metrics
        print("\nClassification Metrics:")
        print("-" * 80)
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision (macro): {results['precision_macro']:.4f}")
        print(f"  Recall (macro): {results['recall_macro']:.4f}")
        print(f"  F1-Score (macro): {results['f1_macro']:.4f}")

        # Per-class metrics
        print("\nPer-Class Metrics:")
        print("-" * 80)
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")

        # Text generation metrics
        if 'bleu_1' in results:
            print("\nText Generation Metrics:")
            print("-" * 80)
            print(f"  BLEU-1: {results['bleu_1']:.4f}")
            print(f"  BLEU-2: {results['bleu_2']:.4f}")
            print(f"  BLEU-3: {results['bleu_3']:.4f}")
            print(f"  BLEU-4: {results['bleu_4']:.4f}")
            print(f"  ROUGE-1: {results['rouge1']:.4f}")
            print(f"  ROUGE-2: {results['rouge2']:.4f}")
            print(f"  ROUGE-L: {results['rougeL']:.4f}")

        # Confusion matrix analysis
        if 'confusion_matrix_analysis' in results:
            print("\n")
            print_confusion_matrix_analysis(results['confusion_matrix_analysis'])

        print("=" * 80)

    def _save_results(self, results: Dict, predictions: Optional[List[Dict]] = None):
        """
        Save evaluation results to file.

        Args:
            results: Results dictionary
            predictions: List of predictions (optional)
        """
        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'

        # Prepare results for JSON (remove non-serializable items)
        results_to_save = {
            k: v for k, v in results.items()
            if k != 'confusion_matrix_analysis'
        }

        # Add confusion matrix analysis (convert to serializable format)
        if 'confusion_matrix_analysis' in results:
            results_to_save['confusion_matrix_analysis'] = results['confusion_matrix_analysis']

        with open(metrics_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

        # Save classification report
        report_path = self.output_dir / 'classification_report.txt'
        report = self.classification_metrics.get_classification_report()

        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Classification report saved to {report_path}")

        # Save predictions
        if predictions is not None:
            predictions_path = self.output_dir / 'predictions.json'

            with open(predictions_path, 'w') as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Predictions saved to {predictions_path}")

    def _plot_confusion_matrix(self, cm):
        """
        Plot and save confusion matrix.

        Args:
            cm: Confusion matrix
        """
        # Plot both raw and normalized
        cm_path = self.output_dir / 'confusion_matrix.png'

        plot_normalized_confusion_matrices(
            cm,
            self.class_names,
            save_path=cm_path,
            show=False
        )

        logger.info(f"Confusion matrix plot saved to {cm_path}")


if __name__ == "__main__":
    # Test evaluator
    print("Evaluator module loaded successfully")
    print("To test, please run with actual model and data loader")
    print("\nExample usage:")
    print("""
    from src.evaluation.evaluator import Evaluator
    from src.models.multimodal_model import MultiModalLeafDiseaseModel
    from src.data.dataloader import DataLoaderFactory

    # Load model
    model = MultiModalLeafDiseaseModel(...)
    checkpoint = torch.load('outputs/checkpoints/best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create test loader
    loader_factory = DataLoaderFactory(...)
    _, _, test_loader = loader_factory.create_all_dataloaders()

    # Evaluate
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        vocabulary=vocabulary,
        class_names=['Healthy', 'Alternaria', 'Stemphylium', 'Marssonina'],
        output_dir='outputs/evaluation'
    )

    results = evaluator.evaluate(
        compute_text_metrics=True,
        save_predictions=True,
        plot_cm=True
    )
    """)
