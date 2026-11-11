# inspect-grounding

A deterministic scorer for [Inspect AI](https://inspect.aisi.org.uk/) that flags **numbers in a model's answer
that do not appear in the evidence it was given**: the input, tool results or sample metadata.

```
Evidence (tool result): {"hrv_ms": 61.7, "resting_hr": 54}

"Your HRV is about 62 ms and resting HR 54."   -> C   (62 rounds 61.7)
"Your HRV jumped to 75 ms, up 22%."            -> I   ungrounded: 75, 22%
```

No judge model and no API calls: the check is plain Python, so it is fast, free, reproducible and safe to run
in CI on every prompt or model change.

## Why

Agents that report on data (health metrics, financials, analytics, retrieved documents) often get the wording
right and a number wrong. LLM-judged faithfulness scorers can miss this and cost a model call per sample. A
number either appears in the evidence or it does not, so this part of grounding can be checked exactly.

The check started as the numeric validator in [Recovery Lab](https://github.com/Chris316-git/WHOOP_RecoveryLab),
a personal health agent where every number in a reply must come from data computed in Python. This package
generalizes it into a reusable Inspect scorer.

## Install

```bash
pip install inspect-grounding
```

## Use

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_grounding import numeric_grounding

@task
def summarize() -> Task:
    return Task(
        dataset=[Sample(input="Q3 revenue was $4,215,000, up 12.4%. Summarize it.")],
        solver=generate(),
        scorer=numeric_grounding(),
    )
```

It is registered with Inspect, so it also works by name, e.g. to re-score an existing log:

```bash
inspect score logs/my-eval.eval --scorer inspect_grounding/numeric_grounding -S ignore_below=10 --action append
```

Each score lists what failed: `answer` holds the ungrounded numbers, `explanation` reads like
`2 of 5 numbers not found in the evidence: 75, 22%`, and `metadata` has `numbers_checked` and `ungrounded`.

A runnable example is in [`examples/grounded_report.py`](examples/grounded_report.py).

## Options

| Argument | Default | Meaning |
|---|---|---|
| `sources` | `("input", "tools", "metadata")` | Where evidence comes from. Add `"target"` to use the target text. Assistant messages are never evidence, so a number the model invented earlier does not become grounded by repetition. |
| `evidence` | `None` | A function `(state, target) -> list[str]` returning the evidence instead, e.g. retrieved documents kept in the sample store. |
| `check` | `"completion"` | Check the final completion, or `"all_assistant"` to check every assistant message. |
| `rounding` | `True` | Accept a number that rounds an evidence value to the precision it is written with. |
| `ignore` | `()` | Values that never need grounding, e.g. `[1, 2, 3]` for list numbering. |
| `ignore_below` | `None` | Skip numbers whose absolute value is below this, e.g. `10` for small counts in prose. |
| `ignore_patterns` | `()` | Regular expressions removed from answer and evidence first, e.g. `[r"\[\d+\]"]` for citation markers. |

The text helpers are exported too, for use outside a scorer: `extract_numbers(text)` and
`ungrounded_numbers(text, evidence, ...)`.

## What counts as grounded

| Answer | Evidence | Grounded? |
|---|---|---|
| `1,200` / `1200` / `1.2k` | `1200` | yes, formatting and `k` suffix |
| `45%` | `0.45` (or the reverse) | yes |
| `7.8`, `8` | `7.849` | yes with `rounding=True` |
| `7.849` | `7.85` | no: more precise than the evidence |
| `-3` | `3` | no: the sign matters |
| `４２，０００` | `42000` | yes: full-width digits are normalized |
| `gpt-4.1`, `COVID-19`, `E12`, `v2` | | not treated as numbers |

## Limitations

Read these before relying on the score:

- **It checks values, not meaning.** "HRV 54" passes when 54 is in the evidence as the resting heart rate.
  Pair it with a semantic or judge-based scorer if attribution matters.
- **Derived numbers fail.** A correct sum or difference (`up 3.5 from 58.2` when the evidence has 58.2 and
  61.7) is flagged unless it is in the evidence. Compute such numbers in your tools and pass them in.
- **Dates and years in the answer** must appear in the evidence too; use `ignore` or `ignore_patterns` if not.
- Spelled-out numbers (`seven`), Chinese numerals (`十二`) and European decimal commas (`1.200,5`) are not
  parsed.
- With many evidence values and `rounding=True`, a small integer can match by chance. `ignore_below` and
  `rounding=False` make the check stricter.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests && mypy
```

Built with AI assistance (Claude); every change is reviewed and tested by the author.

## License

MIT
