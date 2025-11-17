"""Inference Kit - Utility functions for working with causal language models."""

from .inference_kit import (
    init_model,
    generate,
    sum_seq_log_prob,
    mean_seq_log_prob,
    get_k_next_tokens,
    branch_prompts,
    evaluate,
    sequence_embedding_similarity,
)

__version__ = "0.1.0"

__all__ = [
    "init_model",
    "generate",
    "sum_seq_log_prob",
    "mean_seq_log_prob",
    "get_k_next_tokens",
    "branch_prompts",
    "evaluate",
    "sequence_embedding_similarity",
]
