"""Find the numbers in a piece of text and check them against evidence. Pure functions, no Inspect imports.

A number in model output is *grounded* when the same value appears somewhere in the evidence, allowing for
formatting differences (``1,200`` vs ``1200``, full-width digits), percentages (``45%`` vs ``0.45``),
``k`` suffixes (``12k`` vs ``12000``) and, optionally, rounding (``7.8`` grounded by ``7.849``).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# A number: digits with optional comma thousands groups and an optional decimal part, or a bare decimal (.5).
# The lookbehind keeps digits glued to letters (E12, H2O) or to another digit/decimal point out of the match;
# identifiers like gpt-4.1 or COVID-19 are dropped afterwards (see _is_identifier).
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-−])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<suffix>%|[kK](?![A-Za-z]))?"
)


@dataclass(frozen=True)
class NumberMention:
    """One number as written in the text."""

    text: str
    """The number exactly as it appears (after Unicode normalization), e.g. ``"1,200"`` or ``"45%"``."""

    value: Decimal
    """The plain value: ``45%`` -> 45, ``12k`` -> 12000."""

    decimals: int
    """Decimal places as written: ``7.80`` -> 2. Used for rounding."""

    start: int
    """Offset in the normalized text."""

    candidates: tuple[Decimal, ...]
    """Every value this mention may stand for: ``45%`` -> (45, 0.45), ``12k`` -> (12, 12000)."""

    suffix: str = ""
    """``"%"``, ``"k"`` or ``""``."""

    def places(self, index: int) -> int:
        """Decimal places of ``candidates[index]`` as written: ``7.5%`` is 1 place as 7.5 and 3 as 0.075."""
        if index == 0:
            return self.decimals
        if self.suffix == "%":
            return self.decimals + 2
        return max(self.decimals - 3, 0)


def normalize(text: str) -> str:
    """NFKC-normalize so full-width digits and punctuation (４２，０００) read like ASCII."""
    return unicodedata.normalize("NFKC", text)


def _is_identifier(text: str, start: int, end: int) -> bool:
    """True for version numbers and names (gpt-4.1, COVID-19, v2): a letter right before an optional hyphen."""
    i = start - 1
    if i >= 0 and text[i] in "-−":
        i -= 1
    return i >= 0 and (text[i].isalpha() or text[i] == "_")


def extract_numbers(text: str, ignore_patterns: Sequence[str] = ()) -> list[NumberMention]:
    """Every number in ``text``, in order.

    ``ignore_patterns`` are regular expressions removed before extraction, for tokens that look like numbers but
    are not claims (evidence ids such as ``E12`` are already skipped; ``r"\\[\\d+\\]"`` would skip citation
    markers like ``[3]``).
    """
    text = normalize(text)
    for pattern in ignore_patterns:
        text = re.sub(pattern, lambda m: " " * len(m.group(0)), text)

    mentions: list[NumberMention] = []
    for m in _NUMBER.finditer(text):
        if _is_identifier(text, m.start(), m.end()):
            continue
        raw = m.group("num")
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:  # pragma: no cover - the regex only admits valid decimals
            continue
        sign = m.group("sign")
        # A minus sign right after a digit or letter is a range or a hyphen (1.8-2.3, 3-day), not a sign.
        before = text[m.start() - 1] if m.start() > 0 else " "
        negative = bool(sign) and not (before.isalnum())
        if negative:
            value = -value
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        suffix = m.group("suffix") or ""

        candidates = [value]
        if suffix == "%":
            candidates.append(value / 100)
        elif suffix.lower() == "k":
            candidates.append(value * 1000)
        start = m.start() if negative else m.start("num")
        mentions.append(
            NumberMention(
                text=text[start : m.end()],
                value=candidates[-1] if suffix.lower() == "k" else value,
                decimals=decimals,
                start=start,
                candidates=tuple(candidates),
                suffix=suffix.lower(),
            )
        )
    return mentions


def _rounded(value: Decimal, decimals: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def evidence_values(evidence: Iterable[str], ignore_patterns: Sequence[str] = ()) -> set[Decimal]:
    """All values the evidence supports, including the percentage and ``k`` readings."""
    values: set[Decimal] = set()
    for text in evidence:
        for mention in extract_numbers(text, ignore_patterns):
            values.update(mention.candidates)
    return values


def is_grounded(mention: NumberMention, evidence: set[Decimal], rounding: bool = True) -> bool:
    """True when any reading of ``mention`` equals a value in ``evidence``, or a rounding of one.

    With ``rounding``, each evidence value is rounded to the precision the mention was written with, so 7.849
    grounds 7.85, 7.8 and 8, while evidence of 7.85 does not ground a claim of 7.849 (more precise than the
    evidence).
    """
    for i, candidate in enumerate(mention.candidates):
        if candidate in evidence:
            return True
        if rounding:
            places = mention.places(i)
            if any(_rounded(e, places) == candidate for e in evidence):
                return True
    return False


def ungrounded_numbers(
    text: str,
    evidence: Iterable[str],
    *,
    rounding: bool = True,
    ignore: Iterable[float | int | str] = (),
    ignore_below: float | None = None,
    ignore_patterns: Sequence[str] = (),
) -> list[NumberMention]:
    """The numbers in ``text`` that the evidence does not support.

    Args:
        text: The claim to check, typically a model's answer.
        evidence: The texts the answer may draw numbers from (input, tool results, retrieved documents).
        rounding: Accept a number that rounds an evidence value to the precision it is written with.
        ignore: Values that never need grounding, e.g. ``[1, 2, 3]`` for list numbering.
        ignore_below: Skip numbers whose absolute value is below this, e.g. ``10`` for small counts.
        ignore_patterns: Regular expressions removed from both the text and the evidence before checking.
    """
    supported = evidence_values(evidence, ignore_patterns)
    skip = {Decimal(str(v)) for v in ignore}
    bad = []
    for mention in extract_numbers(text, ignore_patterns):
        if mention.value in skip or any(c in skip for c in mention.candidates):
            continue
        if ignore_below is not None and abs(mention.value) < Decimal(str(ignore_below)):
            continue
        if not is_grounded(mention, supported, rounding):
            bad.append(mention)
    return bad
