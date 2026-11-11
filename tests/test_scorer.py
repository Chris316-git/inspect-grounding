from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import Tool, tool

from inspect_grounding import collect_evidence, numeric_grounding

MODEL = "mockllm/model"


def make_state(
    messages: list[ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool],
    completion: str,
    metadata: dict[str, object] | None = None,
) -> TaskState:
    state = TaskState(
        model=ModelName(MODEL), sample_id=1, epoch=1, input="q", messages=messages, metadata=metadata or {}
    )
    state.output = ModelOutput.from_content(MODEL, completion)
    return state


async def test_grounded_in_input() -> None:
    state = make_state([ChatMessageUser(content="Sleep was 7.4 hours, HRV 58 ms.")], "You slept 7.4 h; HRV 58.")
    score = await numeric_grounding()(state, Target(""))
    assert score is not None
    assert score.value == CORRECT
    assert score.metadata == {"numbers_checked": 2, "ungrounded": []}


async def test_invented_number_is_incorrect() -> None:
    state = make_state([ChatMessageUser(content="HRV 58 ms.")], "HRV jumped to 71 ms, up 22%.")
    score = await numeric_grounding()(state, Target(""))
    assert score is not None
    assert score.value == INCORRECT
    assert score.answer == "71, 22%"
    assert score.metadata is not None and score.metadata["ungrounded"] == ["71", "22%"]


async def test_tool_results_and_metadata_are_evidence() -> None:
    messages = [
        ChatMessageUser(content="How did I sleep?"),
        ChatMessageAssistant(content=""),
        ChatMessageTool(content='{"sleep_hours": 7.4}', function="sleep"),
    ]
    state = make_state(messages, "7.4 hours, recovery 62.", metadata={"recovery": 62})
    score = await numeric_grounding()(state, Target(""))
    assert score is not None and score.value == CORRECT

    only_input = await numeric_grounding(sources=["input"])(state, Target(""))
    assert only_input is not None and only_input.value == INCORRECT


async def test_assistant_messages_are_not_evidence() -> None:
    messages = [
        ChatMessageUser(content="What was my HRV?"),
        ChatMessageAssistant(content="It was 99 ms."),  # invented earlier
    ]
    state = make_state(messages, "As I said, 99 ms.")
    score = await numeric_grounding()(state, Target(""))
    assert score is not None and score.value == INCORRECT


async def test_check_all_assistant_messages() -> None:
    messages = [
        ChatMessageUser(content="HRV 58."),
        ChatMessageAssistant(content="It was 99 ms."),
    ]
    state = make_state(messages, "HRV 58.")
    final_only = await numeric_grounding()(state, Target(""))
    every_turn = await numeric_grounding(check="all_assistant")(state, Target(""))
    assert final_only is not None and final_only.value == CORRECT
    assert every_turn is not None and every_turn.value == INCORRECT


async def test_custom_evidence_function() -> None:
    state = make_state([ChatMessageUser(content="no numbers here")], "Revenue was 4.2 million.")

    def from_target(state: TaskState, target: Target) -> list[str]:
        return [target.text]

    score = await numeric_grounding(evidence=from_target)(state, Target("revenue 4.2 million"))
    assert score is not None and score.value == CORRECT


def test_collect_evidence_sources() -> None:
    messages = [
        ChatMessageSystem(content="sys 1"),
        ChatMessageUser(content="user 2"),
        ChatMessageAssistant(content="assistant 3"),
        ChatMessageTool(content="tool 4", function="f"),
    ]
    state = make_state(messages, "", metadata={"m": 5})
    joined = " ".join(collect_evidence(state, Target("target 6"), ["input", "tools", "metadata", "target"]))
    for word in ("sys 1", "user 2", "tool 4", '"m": 5', "target 6"):
        assert word in joined
    assert "assistant 3" not in joined


@tool
def lookup() -> Tool:
    async def execute(metric: str) -> str:
        """Look up a metric.

        Args:
            metric: Name of the metric.
        """
        return '{"resting_hr": 54, "hrv_ms": 61.7}'

    return execute


def test_end_to_end_eval_with_tool_call(tmp_path: Path) -> None:
    outputs = [
        ModelOutput.for_tool_call(MODEL, "lookup", {"metric": "hrv"}),
        ModelOutput.from_content(MODEL, "Your HRV is about 62 ms and resting HR 54."),
        ModelOutput.for_tool_call(MODEL, "lookup", {"metric": "hrv"}),
        ModelOutput.from_content(MODEL, "Your HRV is 75 ms."),
    ]
    task = Task(
        dataset=[Sample(input="What is my HRV?", id="grounded"), Sample(input="What is my HRV?", id="invented")],
        solver=[use_tools(lookup()), generate()],
        scorer=numeric_grounding(),
    )
    model = get_model(MODEL, custom_outputs=outputs)
    [log] = eval(task, model=model, display="none", max_samples=1, log_dir=str(tmp_path))
    assert log.status == "success"
    assert log.samples is not None
    by_id = {s.id: s.scores["numeric_grounding"].value for s in log.samples if s.scores}
    assert by_id == {"grounded": CORRECT, "invented": INCORRECT}
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 0.5
