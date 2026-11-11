"""Deterministic numeric-grounding scorer for Inspect AI."""

from ._numbers import NumberMention, extract_numbers, is_grounded, ungrounded_numbers
from ._scorer import collect_evidence, numeric_grounding

__all__ = [
    "NumberMention",
    "collect_evidence",
    "extract_numbers",
    "is_grounded",
    "numeric_grounding",
    "ungrounded_numbers",
]

__version__ = "0.1.0"
