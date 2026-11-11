# Imported by Inspect through the entry point in pyproject.toml, so the scorer can be used by name:
#   inspect score log.eval --scorer inspect_grounding/numeric_grounding
from ._scorer import numeric_grounding

__all__ = ["numeric_grounding"]
