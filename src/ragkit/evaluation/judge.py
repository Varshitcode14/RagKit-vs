"""LLM-as-a-judge scoring (graded 1-5 rubric, normalized to [0, 1]).

Uses any :class:`BaseLLM`, so it works with the multi-provider manager or a
mock. Ported from the original ``llm_judge`` with an injected LLM.
"""

from __future__ import annotations

import re

from ragkit.core.interfaces import BaseLLM

_CORRECTNESS_PROMPT = """
You are a strict evaluation judge for a question-answering system.

Question:
{question}

Gold Answer:
{gold}

Predicted Answer:
{prediction}

Rate how well the predicted answer matches the gold answer on a 1-5 scale,
judging substance, not wording or length.

5 = Fully correct.   4 = Mostly correct.   3 = Partially correct.
2 = Mostly incorrect. 1 = Completely incorrect.

Return ONLY the integer rating (1-5).
"""

_FAITHFULNESS_PROMPT = """
You are judging whether an answer is grounded in the provided context.

Context:
{context}

Answer:
{prediction}

Rate 1-5 how well the answer is supported by the context.
5 = fully supported, 1 = not supported at all.
Return ONLY the integer rating (1-5).
"""


def _rating_to_score(text: str) -> float | None:
    m = re.search(r"[1-5]", text or "")
    if not m:
        return None
    return (int(m.group(0)) - 1) / 4.0


def judge_correctness(llm: BaseLLM, question: str, gold: str, prediction: str) -> float | None:
    out = llm.generate(
        _CORRECTNESS_PROMPT.format(question=question, gold=gold, prediction=prediction)
    )
    return _rating_to_score(out)


def judge_faithfulness(llm: BaseLLM, context: str, prediction: str) -> float | None:
    out = llm.generate(_FAITHFULNESS_PROMPT.format(context=context, prediction=prediction))
    return _rating_to_score(out)


def batch_judge(
    llm: BaseLLM,
    predictions: list[dict],
    run_faithfulness: bool = True,
) -> list[dict]:
    judged: list[dict] = []
    for row in predictions:
        r = dict(row)
        c = judge_correctness(llm, row["question"], row["gold"], row["prediction"])
        if c is not None:
            r["judge_correctness_score"] = c
        if run_faithfulness and row.get("context"):
            f = judge_faithfulness(llm, row["context"], row["prediction"])
            if f is not None:
                r["judge_faithfulness_score"] = f
        judged.append(r)
    return judged
