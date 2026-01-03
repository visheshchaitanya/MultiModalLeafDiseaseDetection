"""
Vocabulary builder for text generation.
Creates word-to-index and index-to-word mappings from template-generated explanations.
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging

from .template_engine import TemplateEngine
from .template_definitions import get_all_diseases

logger = logging.getLogger(__name__)


class Vocabulary:
    """Vocabulary class for managing word-index mappings."""

    # Special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"  # Start of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence
    UNK_TOKEN = "<UNK>"  # Unknown word

    def __init__(self):
        """Initialize vocabulary."""
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()

        # Add special tokens
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        for token in special_tokens:
            self.add_word(token)

    def add_word(self, word: str) -> int:
        """
        Add a word to the vocabulary.

        Args:
            word: Word to add

        Returns:
            Index of the word
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.word_counts[word] += 1
        return self.word2idx[word]

    def add_sentence(self, sentence: str) -> List[int]:
        """
        Add all words from a sentence to the vocabulary.

        Args:
            sentence: Sentence to process

        Returns:
            List of word indices
        """
        words = self.tokenize(sentence)
        indices = []

        for word in words:
            idx = self.add_word(word)
            indices.append(idx)

        return indices

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Replace specific patterns
        text = re.sub(r'(\d+\.?\d*)°c', r'\1 degrees celsius', text)
        text = re.sub(r'(\d+\.?\d*)%', r'\1 percent', text)

        # Split on whitespace and punctuation, but keep decimal numbers together
        tokens = re.findall(r'\b\w+\.?\w*\b|[.,!?;]', text)

        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a sentence to indices.

        Args:
            sentence: Sentence to encode
            add_special_tokens: Whether to add SOS/EOS tokens

        Returns:
            List of indices
        """
        words = self.tokenize(sentence)
        indices = []

        if add_special_tokens:
            indices.append(self.word2idx[self.SOS_TOKEN])

        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx[self.UNK_TOKEN])

        if add_special_tokens:
            indices.append(self.word2idx[self.EOS_TOKEN])

        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode indices back to sentence.

        Args:
            indices: List of word indices
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded sentence
        """
        special_tokens = {self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN}
        words = []

        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]

                if skip_special_tokens and word in special_tokens:
                    continue

                words.append(word)

        return ' '.join(words)

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.word2idx)

    def get_word_index(self, word: str) -> int:
        """Get index for a word."""
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])

    def get_index_word(self, idx: int) -> str:
        """Get word for an index."""
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def save(self, path: Path):
        """
        Save vocabulary to file.

        Args:
            path: Path to save the vocabulary
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': dict(self.word_counts)
        }

        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)

        logger.info(f"Vocabulary saved to {path}")

    def load(self, path: Path):
        """
        Load vocabulary from file.

        Args:
            path: Path to the vocabulary file
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)

        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_counts = Counter(vocab_data['word_counts'])

        logger.info(f"Vocabulary loaded from {path} (size: {len(self)})")

    def get_most_common(self, n: int = 20) -> List[Tuple[str, int]]:
        """
        Get most common words.

        Args:
            n: Number of words to return

        Returns:
            List of (word, count) tuples
        """
        return self.word_counts.most_common(n)


def build_vocabulary_from_templates(
    num_samples_per_disease: int = 100,
    seed: int = 42,
    save_path: Optional[Path] = None
) -> Vocabulary:
    """
    Build vocabulary from template-generated explanations.

    Args:
        num_samples_per_disease: Number of explanation samples to generate per disease
        seed: Random seed for reproducibility
        save_path: Optional path to save the vocabulary

    Returns:
        Built Vocabulary object
    """
    logger.info("Building vocabulary from templates...")

    vocab = Vocabulary()
    engine = TemplateEngine(seed=seed)
    diseases = get_all_diseases()

    total_explanations = 0

    for disease in diseases:
        logger.info(f"Generating explanations for {disease}...")

        # Generate diverse explanations
        explanations = engine.get_all_possible_explanations(
            disease,
            num_sensor_samples=num_samples_per_disease
        )

        # Add to vocabulary
        for explanation in explanations:
            vocab.add_sentence(explanation)

        total_explanations += len(explanations)
        logger.info(f"  Added {len(explanations)} explanations")

    logger.info(f"Vocabulary built with {len(vocab)} unique words from {total_explanations} explanations")

    # Save if path provided
    if save_path:
        vocab.save(save_path)

    return vocab


def load_vocabulary(path: Path) -> Vocabulary:
    """
    Load vocabulary from file.

    Args:
        path: Path to the vocabulary file

    Returns:
        Loaded Vocabulary object
    """
    vocab = Vocabulary()
    vocab.load(path)
    return vocab


if __name__ == "__main__":
    # Test vocabulary building
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Vocabulary Builder...")
    print("=" * 80)

    # Test basic vocabulary operations
    print("\nTest 1: Basic vocabulary operations")
    print("-" * 80)

    vocab = Vocabulary()

    # Test tokenization
    test_sentence = "Temperature 24.5°C and humidity 82.3% create favorable conditions."
    tokens = vocab.tokenize(test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Tokens: {tokens}")

    # Test encoding/decoding
    indices = vocab.add_sentence(test_sentence)
    print(f"Indices: {indices}")

    encoded = vocab.encode(test_sentence)
    decoded = vocab.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Test building from templates
    print("\n\nTest 2: Build vocabulary from templates")
    print("-" * 80)

    vocab = build_vocabulary_from_templates(num_samples_per_disease=10, seed=42)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"\nMost common words:")
    for word, count in vocab.get_most_common(20):
        print(f"  {word}: {count}")

    # Test save/load
    print("\n\nTest 3: Save and load vocabulary")
    print("-" * 80)

    save_path = Path("outputs/vocabulary.pkl")
    vocab.save(save_path)

    loaded_vocab = load_vocabulary(save_path)
    print(f"Loaded vocabulary size: {len(loaded_vocab)}")

    # Test encoding with loaded vocabulary
    test_text = "Alternaria leaf spot detected with brown lesions."
    encoded = loaded_vocab.encode(test_text)
    decoded = loaded_vocab.decode(encoded)

    print(f"\nTest encoding:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
