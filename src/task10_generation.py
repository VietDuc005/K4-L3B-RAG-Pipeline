"""Grounded answer generation with verifiable chunk citations."""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ các nguồn hiện có."
REFUSAL_TOKEN = "KHÔNG ĐỦ BẰNG CHỨNG"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-4-5-20251001",
}
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

SYSTEM_PROMPT = f"""Bạn là trợ lý tra cứu tài liệu Toyota. Chỉ trả lời từ các đoạn tài liệu được đưa vào.
Mỗi khẳng định thực tế phải có trích dẫn dạng [1], [2], ... trỏ đúng số đoạn tài liệu.
Không dùng kiến thức ngoài, không đoán, không trích dẫn đoạn không hỗ trợ khẳng định.
Nếu không đủ bằng chứng để trả lời, chỉ trả về chính xác: {REFUSAL_TOKEN}"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Put high-ranked chunks at the edges without mutating retrieval order."""
    ordered = list(chunks)
    result = [None] * len(ordered)
    for rank, chunk in enumerate(ordered):
        position = rank // 2 if rank % 2 == 0 else len(ordered) - 1 - rank // 2
        result[position] = chunk
    return result


def format_context(
    chunks: list[dict], citation_numbers: list[int] | None = None
) -> str:
    """Number the exact chunks supplied to the model and expose source metadata."""
    parts = []
    for position, chunk in enumerate(chunks):
        index = citation_numbers[position] if citation_numbers is not None else position + 1
        metadata = chunk["metadata"]
        parts.append(
            f"[{index}] {metadata.get('title', 'Không rõ tiêu đề')} | "
            f"Nguồn: {metadata.get('source', 'Không rõ nguồn')} | "
            f"Chunk ID: {chunk['id']}\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Call the configured provider; credentials are read at call time."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER không được hỗ trợ: {provider}")
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    key_name = PROVIDER_KEYS[provider]
    key = os.getenv(key_name, "").strip()
    if not key:
        raise ValueError(f"Thiếu {key_name} trong .env")

    if provider == "openai":
        from openai import OpenAI

        response = OpenAI(api_key=key, timeout=30).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""

    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=30000)
        )
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return response.text or ""

    from anthropic import Anthropic

    response = Anthropic(api_key=key, timeout=30).messages.create(
        model=model,
        max_tokens=1024,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_from_sources(
    query: str, chunks: list[dict], *, raise_on_error: bool = False
) -> dict:
    """Generate from a fixed retrieval result; also used by the A/B evaluator."""
    refusal = {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    if not query.strip() or not chunks:
        return refusal

    reordered = reorder_for_llm(chunks)
    source_numbers = {chunk["id"]: index for index, chunk in enumerate(chunks, 1)}
    context = format_context(
        reordered, [source_numbers[chunk["id"]] for chunk in reordered]
    )
    try:
        answer = call_llm(SYSTEM_PROMPT, f"Tài liệu:\n{context}\n\nCâu hỏi: {query}")
    except Exception:
        if raise_on_error:
            raise
        return refusal

    answer = answer.strip()
    citations = [int(number) for number in re.findall(r"\[(\d+)\]", answer)]
    if (
        not answer
        or REFUSAL_TOKEN in answer.upper()
        or not citations
        or any(number < 1 or number > len(chunks) for number in citations)
    ):
        return refusal

    return {
        "answer": answer,
        "sources": list(chunks),
        "retrieval_source": "pageindex" if chunks[0]["retrieval_method"] == "pageindex" else "hybrid",
    }


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve and answer, returning the public GenerationResult contract."""
    if not query.strip() or top_k <= 0:
        return generate_from_sources(query, [])
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []
    return generate_from_sources(query, chunks)


if __name__ == "__main__":
    print(generate_with_citation("What does Toyota's anti-bribery policy prohibit?"))
