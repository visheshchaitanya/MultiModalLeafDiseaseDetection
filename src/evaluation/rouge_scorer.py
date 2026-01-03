"""
ROUGE score calculation for text generation evaluation.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class ROUGEScorer:
    """
    Calculate ROUGE scores for generated text.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    recall of n-grams and longest common subsequences.
    """

    def __init__(self):
        """Initialize ROUGE scorer."""
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

    def compute(self, rouge_types: List[str] = None) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Args:
            rouge_types: List of ROUGE types to compute
                         (e.g., ['rouge1', 'rouge2', 'rougeL'])
                         If None, computes all types

        Returns:
            Dictionary of ROUGE scores
        """
        if len(self.candidates) == 0:
            logger.warning("No candidates to compute ROUGE score")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']

        scores = {}

        for rouge_type in rouge_types:
            if rouge_type == 'rouge1':
                scores['rouge1'] = self._compute_rouge_n(1)
            elif rouge_type == 'rouge2':
                scores['rouge2'] = self._compute_rouge_n(2)
            elif rouge_type == 'rougeL':
                scores['rougeL'] = self._compute_rouge_l()
            else:
                logger.warning(f"Unknown ROUGE type: {rouge_type}")

        return scores

    def _compute_rouge_n(self, n: int) -> float:
        """
        Compute ROUGE-N score.

        Args:
            n: N-gram order

        Returns:
            ROUGE-N F1 score
        """
        total_precision = 0.0
        total_recall = 0.0
        count = 0

        for candidate, references in zip(self.candidates, self.references):
            # Tokenize
            candidate_tokens = self._tokenize(candidate)

            # Get best reference
            best_f1 = 0.0
            best_precision = 0.0
            best_recall = 0.0

            for reference in references:
                reference_tokens = self._tokenize(reference)

                # Get n-grams
                candidate_ngrams = self._get_ngrams(candidate_tokens, n)
                reference_ngrams = self._get_ngrams(reference_tokens, n)

                # Count matches
                matches = sum((candidate_ngrams & reference_ngrams).values())

                # Precision and Recall
                if sum(candidate_ngrams.values()) == 0:
                    precision = 0.0
                else:
                    precision = matches / sum(candidate_ngrams.values())

                if sum(reference_ngrams.values()) == 0:
                    recall = 0.0
                else:
                    recall = matches / sum(reference_ngrams.values())

                # F1 score
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

                # Keep best
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall

            total_precision += best_precision
            total_recall += best_recall
            count += 1

        # Average F1
        avg_precision = total_precision / count
        avg_recall = total_recall / count

        if avg_precision + avg_recall == 0:
            avg_f1 = 0.0
        else:
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

        return float(avg_f1)

    def _compute_rouge_l(self) -> float:
        """
        Compute ROUGE-L score (Longest Common Subsequence).

        Returns:
            ROUGE-L F1 score
        """
        total_precision = 0.0
        total_recall = 0.0
        count = 0

        for candidate, references in zip(self.candidates, self.references):
            # Tokenize
            candidate_tokens = self._tokenize(candidate)

            # Get best reference
            best_f1 = 0.0
            best_precision = 0.0
            best_recall = 0.0

            for reference in references:
                reference_tokens = self._tokenize(reference)

                # Compute LCS length
                lcs_length = self._lcs_length(candidate_tokens, reference_tokens)

                # Precision and Recall
                if len(candidate_tokens) == 0:
                    precision = 0.0
                else:
                    precision = lcs_length / len(candidate_tokens)

                if len(reference_tokens) == 0:
                    recall = 0.0
                else:
                    recall = lcs_length / len(reference_tokens)

                # F1 score
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

                # Keep best
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall

            total_precision += best_precision
            total_recall += best_recall
            count += 1

        # Average F1
        avg_precision = total_precision / count
        avg_recall = total_recall / count

        if avg_precision + avg_recall == 0:
            avg_f1 = 0.0
        else:
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

        return float(avg_f1)

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

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Compute length of longest common subsequence.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Length of LCS
        """
        m, n = len(seq1), len(seq2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def compute_sentence_rouge(
        self,
        candidate: str,
        reference: Union[str, List[str]],
        rouge_types: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute ROUGE score for a single sentence.

        Args:
            candidate: Generated text
            reference: Reference text(s)
            rouge_types: ROUGE types to compute

        Returns:
            Dictionary of ROUGE scores
        """
        # Temporarily save current state
        saved_candidates = self.candidates
        saved_references = self.references

        # Compute for single sentence
        self.reset()
        self.add(candidate, reference)
        scores = self.compute(rouge_types)

        # Restore state
        self.candidates = saved_candidates
        self.references = saved_references

        return scores


def compute_rouge(
    candidates: List[str],
    references: List[Union[str, List[str]]],
    rouge_types: List[str] = None
) -> Dict[str, float]:
    """
    Convenience function to compute ROUGE scores.

    Args:
        candidates: List of generated texts
        references: List of reference texts (each can be string or list)
        rouge_types: ROUGE types to compute

    Returns:
        Dictionary of ROUGE scores
    """
    scorer = ROUGEScorer()

    for candidate, reference in zip(candidates, references):
        scorer.add(candidate, reference)

    return scorer.compute(rouge_types)


if __name__ == "__main__":
    # Test ROUGE scorer
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing ROUGEScorer...")
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

    # Test 1: Corpus-level ROUGE
    print("\nTest 1: Corpus-level ROUGE")
    print("-" * 80)

    scorer = ROUGEScorer()

    for candidate, reference in zip(candidates, references):
        scorer.add(candidate, reference)

    rouge_scores = scorer.compute()

    print("ROUGE Scores:")
    for metric, value in rouge_scores.items():
        print(f"  {metric}: {value:.4f}")

    # Test 2: Sentence-level ROUGE
    print("\n\nTest 2: Sentence-level ROUGE")
    print("-" * 80)

    for i, (candidate, reference) in enumerate(zip(candidates, references)):
        scores = scorer.compute_sentence_rouge(candidate, reference)
        print(f"\nSentence {i + 1}:")
        print(f"  Candidate: {candidate}")
        print(f"  Reference: {reference}")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")

    # Test 3: Multiple references
    print("\n\nTest 3: Multiple references")
    print("-" * 80)

    candidate = "the cat is on the mat"
    references_multi = [
        "the cat is sitting on the mat",
        "a cat is on the mat",
        "the cat sits on a mat"
    ]

    scores = scorer.compute_sentence_rouge(candidate, references_multi)
    print(f"Candidate: {candidate}")
    print(f"References: {references_multi}")
    print("ROUGE Scores:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

    # Test 4: Perfect match
    print("\n\nTest 4: Perfect match")
    print("-" * 80)

    candidate = "the quick brown fox jumps"
    reference = "the quick brown fox jumps"

    scores = scorer.compute_sentence_rouge(candidate, reference)
    print(f"Candidate: {candidate}")
    print(f"Reference: {reference}")
    print("ROUGE Scores (should be ~1.0):")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

    # Test 5: No match
    print("\n\nTest 5: No match")
    print("-" * 80)

    candidate = "hello world"
    reference = "goodbye universe"

    scores = scorer.compute_sentence_rouge(candidate, reference)
    print(f"Candidate: {candidate}")
    print(f"Reference: {reference}")
    print("ROUGE Scores (should be ~0.0):")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

    # Test 6: ROUGE-L test
    print("\n\nTest 6: ROUGE-L specific test")
    print("-" * 80)

    candidate = "police killed the gunman"
    reference = "police kill the gunman"

    scores = scorer.compute_sentence_rouge(candidate, reference, rouge_types=['rougeL'])
    print(f"Candidate: {candidate}")
    print(f"Reference: {reference}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")

    # Test 7: Convenience function
    print("\n\nTest 7: Convenience function")
    print("-" * 80)

    rouge_scores = compute_rouge(candidates, references)
    print("ROUGE Scores (convenience function):")
    for metric, value in rouge_scores.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("ROUGEScorer tests completed!")
