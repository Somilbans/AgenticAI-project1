"""
Tiny helper around the OpenAI Responses API so we can summarize Chroma hits.
"""

import os
from typing import Iterable, List

from openai import OpenAI


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


def answer_question(question: str, context_chunks: List[str], model: str = "gpt-4o-mini") -> str:
    """
    Use OpenAI to answer the question with the supplied context.
    """
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
    return response.output[0].content[0].text.strip()

