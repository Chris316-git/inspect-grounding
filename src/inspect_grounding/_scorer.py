"""The ``numeric_grounding`` scorer for Inspect."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from ._numbers import extract_numbers, ungrounded_numbers

Source = Literal["input", "tools", "metadata", "target"]
EvidenceFn = Callable[[TaskState, Target], Iterable[str]]


def collect_evidence(state: TaskState, target: Target, sources: Sequence[Source]) -> list[str]:
    """The texts an answer is allowed to take numbers from.

    * ``input``: system and user messages, i.e. everything the model was given.
    * ``tools``: the results of tool calls made during the sample.
    * ``metadata``: the sample metadata, serialized as JSON.
    * ``target``: the target text (useful when the target holds the reference document).

    Assistant messages are never evidence: a number the model invented earlier in the conversation does not
    become grounded by being repeated.
    """
    texts: list[str] = []
    for message in state.messages:
        role = message.role
        if (role in ("system", "user") and "input" in sources) or (role == "tool" and "tools" in sources):
            texts.append(message.text)
    if "metadata" in sources and state.metadata:
        texts.append(json.dumps(state.metadata, ensure_ascii=False, default=str))
    if "target" in sources:
        texts.append(target.text)
    return texts


def _answer_text(state: TaskState, check: Literal["completion", "all_assistant"]) -> str:
    if check == "all_assistant":
        return "\n".join(m.text for m in state.messages if isinstance(m, ChatMessageAssistant))
    return state.output.completion


@scorer(metrics=[accuracy(), stderr()])
def numeric_grounding(
    sources: Sequence[Source] = ("input", "tools", "metadata"),
    evidence: EvidenceFn | None = None,
    check: Literal["completion", "all_assistant"] = "completion",
    rounding: bool = True,
    ignore: Sequence[float | int | str] = (),
    ignore_below: float | None = None,
    ignore_patterns: Sequence[str] = (),
) -> Scorer:
    """Score CORRECT when every number in the answer appears in the evidence the model was given.

    A deterministic check for invented numbers: no judge model, no API calls. The answer passes when each
    number in it matches a value in the evidence, allowing for formatting (``1,200`` / ``1200``, full-width
    digits), percentages (``45%`` / ``0.45``), ``k`` suffixes and, by default, rounding (``7.8`` from
    ``7.849``). Ungrounded numbers are listed in the score's ``answer``, ``explanation`` and ``metadata``.

    Args:
        sources: Where evidence comes from: ``"input"``, ``"tools"``, ``"metadata"`` and/or ``"target"``.
            Ignored when ``evidence`` is given.
        evidence: A function ``(state, target) -> texts`` that returns the evidence instead, e.g. a
            retrieved document stored in the sample store.
        check: Check only the final completion (default) or every assistant message in the sample.
        rounding: Accept numbers that round an evidence value to the precision they are written with.
        ignore: Values that never need grounding, e.g. ``[1, 2, 3]`` for list numbering.
        ignore_below: Skip numbers whose absolute value is below this, e.g. ``10`` for small counts.
        ignore_patterns: Regular expressions removed from answer and evidence before checking, e.g.
            ``[r"\\[\\d+\\]"]`` for citation markers like ``[3]``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = _answer_text(state, check)
        texts = list(evidence(state, target)) if evidence else collect_evidence(state, target, sources)
        checked = extract_numbers(answer, ignore_patterns)
        bad = ungrounded_numbers(
            answer,
            texts,
            rounding=rounding,
            ignore=ignore,
            ignore_below=ignore_below,
            ignore_patterns=ignore_patterns,
        )
        meta: dict[str, Any] = {
            "numbers_checked": len(checked),
            "ungrounded": [m.text for m in bad],
        }
        if bad:
            listed = ", ".join(m.text for m in bad)
            explanation = f"{len(bad)} of {len(checked)} numbers not found in the evidence: {listed}"
            return Score(value=INCORRECT, answer=listed, explanation=explanation, metadata=meta)
        explanation = f"All {len(checked)} numbers found in the evidence." if checked else "No numbers to check."
        return Score(value=CORRECT, answer="", explanation=explanation, metadata=meta)

    return score
