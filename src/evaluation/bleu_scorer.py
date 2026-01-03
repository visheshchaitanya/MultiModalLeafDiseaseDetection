"""
BLEU score calculation for text generation evaluation.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)


class BLEUScorer:
    """
    Calculate BLEU scores for generated text.

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
    between generated and reference texts.
    """

    def __init__(self, max_n: int = 4, smooth: bool = True):
        """
        Initialize BLEU scorer.

        Args:
            max_n: Maximum n-gram order (typically 4 for BLEU-4)
            smooth: Whether to apply smoothing for zero counts
        """
        self.max_n = max_n
        self.smooth = smooth
        self.reset()

    def reset(self):
        """Reset accumulated candidates and references."""
        self.candidates = []
        self.references = []

    def add(self, candidate: str, reference: Union[str, List[str]]):
        """
        Add a candidate-reference pair.

        Args:
            candidate: Generated text
            reference: Reference text(s) - can be single string or list
        """
        # Ensure reference is a list
        if isinstance(reference, str):
            reference = [reference]

        self.candidates.append(candidate)
        self.references.append(reference)

    def compute(self, return_all: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute BLEU score.

        Args:
            return_all: If True, return all n-gram BLEU scores

        Returns:
            BLEU-4 score or dictionary of all BLEU scores
        """
        if len(self.candidates) == 0:
            logger.warning("No candidates to compute BLEU score")
            return 0.0 if not return_all else {f'bleu_{i}': 0.0 for i in range(1, self.max_n + 1)}

        # Compute corpus-level BLEU
        bleu_scores = {}

        for n in range(1, self.max_n + 1):
            bleu_n = self._compute_bleu_n(n)
            bleu_scores[f'bleu_{n}'] = bleu_n

        if return_all:
            return bleu_scores
        else:
            # Return BLEU-4 by default
            return bleu_scores.get(f'bleu_{self.max_n}', 0.0)

    def _compute_bleu_n(self, n: int) -> float:
        """
        Compute BLEU score for specific n-gram order.

        Args:
            n: N-gram order

        Returns:
            BLEU-n score
        """
        total_matches = 0
        total_candidate_count = 0
        total_reference_length = 0
        total_candidate_length = 0

        for candidate, references in zip(self.candidates, self.references):
            # Tokenize
            candidate_tokens = self._tokenize(candidate)
            reference_tokens_list = [self._tokenize(ref) for ref in references]

            # Count n-grams
            candidate_ngrams = self._get_ngrams(candidate_tokens, n)

            # Get maximum counts from all references
            max_reference_counts = Counter()
            for ref_tokens in reference_tokens_list:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                for ngram in ref_ngrams:
                    max_reference_counts[ngram] = max(
                        max_reference_counts[ngram],
                        ref_ngrams[ngram]
                    )

            # Clipped counts
            clipped_counts = {
                ngram: min(count, max_reference_counts.get(ngram, 0))
                for ngram, count in candidate_ngrams.items()
            }

            total_matches += sum(clipped_counts.values())
            total_candidate_count += sum(candidate_ngrams.values())

            # Closest reference length
            candidate_len = len(candidate_tokens)
            closest_ref_len = min(
                [len(ref_tokens) for ref_tokens in reference_tokens_list],
                key=lambda ref_len: abs(ref_len - candidate_len)
            )

            total_candidate_length += candidate_len
            total_reference_length += closest_ref_len

        # Precision
        if total_candidate_count == 0:
            precision = 0.0
        else:
            if self.smooth and total_matches == 0:
                # Add-one smoothing
                precision = 1.0 / (total_candidate_count + 1)
            else:
                precision = total_matches / total_candidate_count

        # Brevity penalty
        if total_candidate_length >= total_reference_length:
            bp = 1.0
        else:
            bp = np.exp(1 - total_reference_length / (total_candidate_length + 1e-10))

        # BLEU score
        bleu = bp * precision

        return float(bleu)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase and split
        tokens = text.lower().strip().split()
        return tokens

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """
        Extract n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram order

        Returns:
            Counter of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)

        return Counter(ngrams)

    def compute_sentence_bleu(
        self,
        candidate: str,
        reference: Union[str, List[str]]
    ) -> float:
        """
        Compute BLEU score for a single sentence.

        Args:
            candidate: Generated text
            reference: Reference text(s)

        Returns:
            BLEU-4 score
        """
        # Temporarily save current state
        saved_candidates = self.candidates
        saved_references = self.references

        # Compute for single sentence
        self.reset()
        self.add(candidate, reference)
        bleu = self.compute()

        # Restore state
        self.candidates = saved_candidates
        self.references = saved_references

        return bleu


def compute_bleu(
    candidates: List[str],
    references: List[Union[str, List[str]]],
    max_n: int = 4,
    smooth: bool = True
) -> Dict[str, float]:
    """
    Convenience function to compute BLEU scores.

    Args:
        candidates: List of generated texts
        references: List of reference texts (each can be string or list)
        max_n: Maximum n-gram order
        smooth: Whether to apply smoothing

    Returns:
        Dictionary of BLEU scores
    """
    scorer = BLEUScorer(max_n=max_n, smooth=smooth)

    for candidate, reference in zip(candidates, references):
        scorer.add(candidate, reference)

    return scorer.compute(return_all=True)


if __name__ == "__main__":
    # Test BLEU scorer
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing BLEUScorer...")
    print("=" * 80)

    # Test data
    candidates = [
        "the cat is on the mat",
        "there is a cat on the mat",
        "the dog is under the table"
    ]

    references = [
        "the cat is sitting on the mat",
        "there is a cat on the mat",
        "the dog is beneath the table"
    ]

    # Test 1: Corpus-level BLEU
    print("\nTest 1: Corpus-level BLEU")
    print("-" * 80)

    scorer = BLEUScorer(max_n=4, smooth=True)

    for candidate, reference in zip(candidates, references):
        scorer.add(candidate, reference)

    bleu_scores = scorer.compute(return_all=True)

    print("BLEU Scores:")
    for metric, value in bleu_scores.items():
        print(f"  {metric}: {value:.4f}")

    # Test 2: Sentence-level BLEU
    print("\n\nTest 2: Sentence-level BLEU")
    print("-" * 80)

    for i, (candidate, reference) in enumerate(zip(candidates, references)):
        bleu = scorer.compute_sentence_bleu(candidate, reference)
        print(f"\nSentence {i + 1}:")
        print(f"  Candidate: {candidate}")
        print(f"  Reference: {reference}")
        print(f"  BLEU-4: {bleu:.4f}")

    # Test 3: Multiple references
    print("\n\nTest 3: Multiple references")
    print("-" * 80)

    candidate = "the cat is on the mat"
    references_multi = [
        "the cat is sitting on the mat",
        "a cat is on the mat",
        "the cat sits on a mat"
    ]

    bleu = scorer.compute_sentence_bleu(candidate, references_multi)
    print(f"Candidate: {candidate}")
    print(f"References: {references_multi}")
    print(f"BLEU-4: {bleu:.4f}")

    # Test 4: Perfect match
    print("\n\nTest 4: Perfect match")
    print("-" * 80)

    candidate = "the quick brown fox jumps"
    reference = "the quick brown fox jumps"

    bleu = scorer.compute_sentence_bleu(candidate, reference)
    print(f"Candidate: {candidate}")
    print(f"Reference: {reference}")
    print(f"BLEU-4: {bleu:.4f} (should be ~1.0)")

    # Test 5: No match
    print("\n\nTest 5: No match")
    print("-" * 80)

    candidate = "hello world"
    reference = "goodbye universe"

    bleu = scorer.compute_sentence_bleu(candidate, reference)
    print(f"Candidate: {candidate}")
    print(f"Reference: {reference}")
    print(f"BLEU-4: {bleu:.4f} (should be ~0.0)")

    # Test 6: Convenience function
    print("\n\nTest 6: Convenience function")
    print("-" * 80)

    bleu_scores = compute_bleu(candidates, references, max_n=4)
    print("BLEU Scores (convenience function):")
    for metric, value in bleu_scores.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("BLEUScorer tests completed!")
