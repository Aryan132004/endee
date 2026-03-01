"""
utils/llm.py – LLM caller using the new google-genai SDK.
"""
from __future__ import annotations
import config

_SYSTEM_PROMPT = """You are a helpful Personal Knowledge Assistant.
Answer the user's question using ONLY the context chunks provided below.
If the answer is not in the context, say "I couldn't find that in your documents."
Be concise and cite the source file when relevant.
"""

def _build_user_message(question: str, context_chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[{i}] (source: {chunk['source']}, similarity: {chunk['similarity']})\n{chunk['text']}"
        )
    context_str = "\n\n".join(context_parts)
    return f"Context from your documents:\n\n{context_str}\n\nQuestion: {question}"


def _answer_gemini(user_message: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


def _answer_openai_compatible(user_message: str) -> str:
    from openai import OpenAI
    _URLS = {
        "groq":   "https://api.groq.com/openai/v1",
        "openai": "https://api.openai.com/v1",
    }
    provider = config.LLM_PROVIDER.lower()
    api_key = config.GROQ_API_KEY if provider == "groq" else config.OPENAI_API_KEY
    client = OpenAI(api_key=api_key, base_url=_URLS[provider])
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def answer(question: str, context_chunks: list[dict]) -> str:
    user_message = _build_user_message(question, context_chunks)
    provider = config.LLM_PROVIDER.lower()
    if provider == "gemini":
        return _answer_gemini(user_message)
    elif provider in ("groq", "openai"):
        return _answer_openai_compatible(user_message)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")