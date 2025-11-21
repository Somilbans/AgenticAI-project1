"""
Tiny helper around the OpenAI Responses API so we can summarize Chroma hits.
"""

import os
import time
from typing import Iterable, List, Tuple

from openai import OpenAI

from mini_project.llm_metrics import LLMMetrics, extract_usage_from_response


SYSTEM_PROMPT = (
    "You are a helpful recruiting analyst. Use the provided context rows to "
    "answer the user question. If the answer is not in the context, say so."
)


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment before querying.")
    return OpenAI(api_key=api_key)


def build_context(chunks: Iterable[str]) -> str:
    """
    Join the retrieved Chroma documents into a readable block.
    """
    cleaned = [chunk.strip() for chunk in chunks if chunk]
    if not cleaned:
        return "No matching records were retrieved."
    bullet_list = "\n".join(f"- {chunk}" for chunk in cleaned)
    return f"Relevant rows:\n{bullet_list}"


def answer_question(question: str, context_chunks: List[str], model: str = "gpt-4o-mini") -> Tuple[str, LLMMetrics]:
    """
    Use OpenAI to answer the question with the supplied context.
    
    Returns:
        Tuple of (answer_text, metrics)
    """
    answer_start_time = time.time()
    client = _get_client()
    context = build_context(context_chunks)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    )
    
    # Extract metrics
    answer_end_time = time.time()
    answer_time = answer_end_time - answer_start_time
    metrics = extract_usage_from_response(response, model)
    metrics.time_taken = answer_time
    
    print(f"[LLM QUERY] Answer Question", flush=True)
    print(f"  Model: {model}", flush=True)
    print(f"  Time Taken: {answer_time:.3f} seconds", flush=True)
    print(f"  Prompt Tokens: {metrics.prompt_tokens}", flush=True)
    print(f"  Completion Tokens: {metrics.completion_tokens}", flush=True)
    print(f"  Total Tokens: {metrics.total_tokens}", flush=True)
    
    answer_text = response.output[0].content[0].text.strip()
    return answer_text, metrics

