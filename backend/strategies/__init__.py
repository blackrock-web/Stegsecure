"""Embedding strategy abstractions for benchmarking."""

from .base import EmbeddingStrategy, StrategyResult
from .registry import get_strategy, list_strategies

__all__ = ["EmbeddingStrategy", "StrategyResult", "get_strategy", "list_strategies"]
