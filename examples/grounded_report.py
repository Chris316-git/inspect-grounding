"""Ask a model to summarize a short report, then check that every number it writes comes from the report.

Run with any model Inspect supports:

    inspect eval examples/grounded_report.py --model openai/gpt-4o-mini
    inspect eval examples/grounded_report.py --model anthropic/claude-haiku-4-5

or offline, to see the scorer wiring without an API key:

    inspect eval examples/grounded_report.py --model mockllm/model
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message

from inspect_grounding import numeric_grounding

REPORTS = [
    (
        "sleep",
        "Last night: 7 h 24 min asleep (7.4 h), sleep performance 86%, HRV 61.7 ms (30-day average 58.2 ms), "
        "resting heart rate 54 bpm.",
    ),
    (
        "sales",
        "Q3 revenue was $4,215,000, up 12.4% from Q2. 1,872 new customers; churn fell from 3.1% to 2.6%.",
    ),
    (
        "trial",
        "In the pilot, 48 of 60 participants (80%) completed all sessions. Mean score rose from 61.5 to 68.2.",
    ),
]

PROMPT = "Summarize this report in two sentences. Quote the key numbers.\n\n{report}"


@task
def grounded_report() -> Task:
    return Task(
        dataset=[Sample(id=name, input=PROMPT.format(report=report)) for name, report in REPORTS],
        solver=[system_message("Use only numbers that appear in the report."), generate()],
        # ignore_below=10 skips small counts such as "two sentences" or "3 points".
        scorer=numeric_grounding(ignore_below=10),
    )
