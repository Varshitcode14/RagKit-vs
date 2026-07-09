"""Prompt templates ported from the original project.

Kept as module constants so they can be tuned independently of logic and
overridden per-pipeline via the PromptBuilder.
"""

# Standardized response used across pipelines when the retrieved context does
# not contain the answer. Keeping it consistent makes "grounded-only" behavior
# predictable and easy to detect downstream.
NOT_FOUND_RESPONSE = "The answer is not available in the provided documents."

TRADITIONAL_RAG_PROMPT = """
You are a question answering system.

Answer the question using ONLY the information in the context below.
You must NOT use any outside knowledge, prior training, or assumptions.

If the answer is not contained in the context, reply EXACTLY with:
The answer is not available in the provided documents.

Rules:
- Use ONLY facts explicitly stated in the context. Do NOT infer beyond it.
- Do NOT hallucinate or add information from memory.
- Be concise and direct. Return ONLY the answer, with no preamble.

Context:
{context}

Question:
{question}

Answer:
"""

PLANNER_PROMPT = """
You are the Planner Agent in a Multi-Agent RAG system.

Your job is to decompose a user question into the MINIMUM reasoning steps needed.

Rules:
- Simple factual questions (one fact needed) -> ONE step only.
- Questions requiring two pieces of information from different sources -> TWO steps.
- Comparison questions -> TWO steps (one per subject being compared).
- NEVER create more than 3 steps. More steps = more error accumulation.
- Each step must retrieve ONE specific piece of information.

Return ONLY a Python list of strings. No explanation.

Examples:

Question: What year did AlexNet achieve a breakthrough?
Output:
["Find the year CNNs achieved a breakthrough with AlexNet."]

Question: What technique is used to train the Claude model family?
Output:
["Find the specific training technique used for the Claude model family."]

Question: How do GPT and BERT differ in their training objectives?
Output:
["Find the training objective of GPT.", "Find the training objective of BERT."]

Question: What is a key advantage of Multi-Agent RAG over single-pass RAG?
Output:
["Identify the key advantage of Multi-Agent RAG systems that single-pass RAG cannot achieve."]

Question:
{question}
"""

STEP_DEFINER_PROMPT = """
You are the Step Definer Agent of a Multi-Agent RAG system.

Your job is to convert one reasoning step into a detailed retrieval query.

You are given

Original Question:
{question}

Current Step:
{step}

Previous Answers:
{history}

Rules:

1. Use previous answers whenever necessary.

2. If the step depends on a previous answer,
incorporate it into the retrieval query.

3. Produce ONE retrieval query.

Return ONLY the query.

Do not explain.
"""

EXTRACTOR_PROMPT = """
You are the Evidence Extraction Agent in a Multi-Agent RAG system.

Your ONLY responsibility is to extract evidence that helps solve the CURRENT GOAL.

CURRENT GOAL:
{goal}

You are given several retrieved documents.

Rules:
1. Extract ONLY useful evidence.
2. Ignore unrelated information.
3. Do NOT summarize the whole document.
4. Do NOT answer the overall question.
5. Mention which document each fact comes from.

Return EXACTLY in this format:

FACT 1
Source: DOCUMENT X
Evidence:
...

FACT 2
Source: DOCUMENT X
Evidence:
...

Retrieved Documents:

{documents}
"""

QA_PROMPT = """
You are the QA Agent in a Multi-Agent RAG system.

Your task is to answer ONLY the CURRENT GOAL.

You are given:
1. The current reasoning goal.
2. Previously solved reasoning steps.
3. Evidence extracted from retrieved documents.

Rules:
- Use ONLY the provided evidence. Do NOT use outside or prior knowledge.
- Use previous answers only if they help.
- Do NOT hallucinate or guess.
- Be concise - give the shortest correct answer.
- If the answer cannot be determined from the evidence, reply exactly: Unknown
- Return ONLY the answer. Do NOT explain or add preamble.

CURRENT GOAL:
{goal}

PREVIOUS STEPS:
{history}

EXTRACTED EVIDENCE:
{evidence}
"""

FINAL_ANSWER_PROMPT = """
You are the Final Answer Agent in a Multi-Agent RAG system.

You are given:
1. The user's original question.
2. The reasoning history (each step's goal and its answer).

Your task: synthesize ONE complete, self-contained answer to the
original question using ONLY the reasoning history.

RULES:
- Use ONLY the reasoning history below. Do NOT use outside knowledge,
  prior training, or assumptions.
- If the reasoning history does not contain enough information to answer,
  reply EXACTLY with:
  The answer is not available in the provided documents.
- Address EVERY part of the question. Multi-part questions require an
  answer that covers all parts.
- Match the expected answer format and length.
- Be factual and grounded ONLY in the reasoning history. Do not speculate.
- Do NOT include preamble such as "The answer is", "Based on the
  history", or "According to". Start directly with the answer.

Examples:

Question: What year did AlexNet achieve a breakthrough?
Answer: 2012

Question: The architecture in 'Attention Is All You Need' eliminated a
sequential mechanism. Name it and the component that replaced it.
Answer: It eliminated recurrence; self-attention replaced it, allowing all
positions to be processed in parallel.

Original Question:
{question}

Reasoning History:
{history}

Final Answer:
"""

__all__ = [
    "NOT_FOUND_RESPONSE",
    "TRADITIONAL_RAG_PROMPT",
    "PLANNER_PROMPT",
    "STEP_DEFINER_PROMPT",
    "EXTRACTOR_PROMPT",
    "QA_PROMPT",
    "FINAL_ANSWER_PROMPT",
]
