"""
Transformer decoder for generating text explanations.
Uses cross-attention to condition on fused multimodal features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
            x: Input tensor (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        if x.dim() == 3 and x.size(0) != x.size(1):
            # Assume (batch_size, seq_len, d_model)
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].unsqueeze(0)
        else:
            # Assume (seq_len, batch_size, d_model)
            seq_len = x.size(0)
            x = x + self.pe[:seq_len, :].unsqueeze(1)

        return self.dropout(x)


class TransformerTextDecoder(nn.Module):
    """
    Transformer decoder for text generation.
    Conditions on multimodal features to generate explanations.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_seq_len: int = 100,
        pad_idx: int = 0
    ):
        """
        Initialize transformer decoder.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            max_seq_len: Maximum sequence length
            pad_idx: Padding token index
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_idx = pad_idx

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # Initialize weights
        self._init_weights()

        logger.info(f"TransformerTextDecoder initialized: vocab_size={vocab_size}, "
                   f"embed_dim={embed_dim}, num_layers={num_layers}, num_heads={num_heads}")

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.

        Args:
            sz: Sequence length

        Returns:
            Mask tensor
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tgt_tokens: Target token indices (B, seq_len)
            memory: Encoded multimodal features (B, 1, embed_dim) or (B, embed_dim)
            tgt_mask: Target attention mask (seq_len, seq_len)
            tgt_key_padding_mask: Target padding mask (B, seq_len)

        Returns:
            Logits over vocabulary (B, seq_len, vocab_size)
        """
        batch_size, seq_len = tgt_tokens.size()

        # Embed tokens
        tgt_embedded = self.token_embedding(tgt_tokens)  # (B, seq_len, embed_dim)

        # Add positional encoding
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # Prepare memory (expand if needed)
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)  # (B, 1, embed_dim)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt_tokens.device)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (B, seq_len, embed_dim)

        # Project to vocabulary
        logits = self.output_projection(output)  # (B, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        start_token_idx: int,
        end_token_idx: int,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            memory: Encoded multimodal features (B, embed_dim)
            start_token_idx: Start-of-sequence token index
            end_token_idx: End-of-sequence token index
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold

        Returns:
            Generated token indices (B, generated_len)
        """
        batch_size = memory.size(0)
        device = memory.device

        # Start with SOS token
        generated = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # Forward pass
            logits = self.forward(generated, memory)  # (B, cur_len, vocab_size)

            # Get logits for next token
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k sampling
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have generated EOS
            if (next_token == end_token_idx).all():
                break

        return generated


if __name__ == "__main__":
    # Test transformer decoder
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing TransformerTextDecoder...")
    print("=" * 80)

    # Parameters
    vocab_size = 1000
    batch_size = 8
    seq_len = 50
    embed_dim = 512

    # Create decoder
    decoder = TransformerTextDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=4,
        num_heads=8,
        ff_dim=2048,
        max_seq_len=100
    )

    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test forward pass
    print("\nTest 1: Forward pass")
    print("-" * 80)

    tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    memory = torch.randn(batch_size, embed_dim)

    logits = decoder(tgt_tokens, memory)

    print(f"Target tokens shape: {tgt_tokens.shape}")
    print(f"Memory shape: {memory.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Test generation
    print("\nTest 2: Autoregressive generation")
    print("-" * 80)

    memory = torch.randn(batch_size, embed_dim)
    start_token = 1
    end_token = 2

    generated = decoder.generate(
        memory=memory,
        start_token_idx=start_token,
        end_token_idx=end_token,
        max_len=30,
        temperature=0.8
    )

    print(f"Generated sequence shape: {generated.shape}")
    print(f"Sample generated sequence: {generated[0].tolist()}")

    print("\n" + "=" * 80)
    print("TransformerTextDecoder tests completed!")
