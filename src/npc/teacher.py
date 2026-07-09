"""Unified teacher/judge model client.

Auto-detects a provider from environment variables and exposes a single
`chat()` function returning text. Supports:

- OpenAI-compatible endpoints (OpenAI, OpenRouter, Groq, or any custom base_url)
- Google Gemini (REST generateContent)

Priority if multiple are configured: OPENAI_API_KEY > OPENROUTER_API_KEY >
GROQ_API_KEY > GEMINI_API_KEY. Override by passing `provider=` explicitly.

Everything goes through plain `requests`, so no provider SDK is strictly required.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


class TeacherError(RuntimeError):
    pass


class RetryableError(TeacherError):
    """Raised for transient failures (429/5xx) so tenacity retries."""


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    model: str
    base_url: Optional[str] = None  # for openai-compatible


def _detect_provider(explicit: Optional[str] = None) -> ProviderConfig:
    order = ["openai", "openrouter", "groq", "gemini"]
    if explicit:
        order = [explicit]

    for name in order:
        if name == "openai" and os.getenv("OPENAI_API_KEY"):
            return ProviderConfig(
                name="openai",
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        if name == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
            return ProviderConfig(
                name="openrouter",
                api_key=os.environ["OPENROUTER_API_KEY"],
                model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
                base_url="https://openrouter.ai/api/v1",
            )
        if name == "groq" and os.getenv("GROQ_API_KEY"):
            return ProviderConfig(
                name="groq",
                api_key=os.environ["GROQ_API_KEY"],
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                base_url="https://api.groq.com/openai/v1",
            )
        if name == "gemini" and os.getenv("GEMINI_API_KEY"):
            return ProviderConfig(
                name="gemini",
                api_key=os.environ["GEMINI_API_KEY"],
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            )

    raise TeacherError(
        "No teacher provider configured. Set one of GEMINI_API_KEY, "
        "OPENROUTER_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY (see .env.example)."
    )


@retry(
    retry=retry_if_exception_type((RetryableError, requests.exceptions.RequestException)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
def _post(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    # requests.exceptions.RequestException covers connection resets / timeouts /
    # dropped sockets, which the gateway throws under parallel load; retry those too.
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code in (429, 500, 502, 503, 504):
        raise RetryableError(f"{resp.status_code}: {resp.text[:300]}")
    if resp.status_code >= 400:
        raise TeacherError(f"{resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _chat_openai_compatible(
    cfg: ProviderConfig,
    system: Optional[str],
    user: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    if cfg.name == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/local/level-locked-tutor"
        headers["X-Title"] = "Grade-Level Locked Writing Tutor"

    url = f"{cfg.base_url}/chat/completions"
    try:
        data = _post(url, headers, payload)
    except TeacherError as e:
        # Some gateways/models (e.g. Claude Sonnet 5 via TrueFoundry) reject
        # `temperature` or `response_format`. Strip the offending param and retry.
        msg = str(e)
        retried = False
        if "temperature" in msg and "temperature" in payload:
            payload.pop("temperature", None)
            retried = True
        if "response_format" in msg and "response_format" in payload:
            payload.pop("response_format", None)
            retried = True
        if not retried:
            raise
        data = _post(url, headers, payload)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:  # pragma: no cover
        raise TeacherError(f"Unexpected response shape: {json.dumps(data)[:400]}") from e


def _chat_gemini(
    cfg: ProviderConfig,
    system: Optional[str],
    user: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.model}:generateContent?key={cfg.api_key}"
    )
    gen_cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_cfg,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    headers = {"Content-Type": "application/json"}
    data = _post(url, headers, payload)
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        # Gemini can return no candidates when blocked; surface that clearly.
        raise TeacherError(f"Unexpected/blocked response: {json.dumps(data)[:400]}") from e


def _stub_response(user: str, system: Optional[str], json_mode: bool) -> str:
    """Offline canned responses so the full pipeline can be smoke-tested with no API.

    Enable with env TEACHER_STUB=1. It inspects the prompt to decide what to emit:
    judge JSON, a seed-prompt JSON list, or a canned grade-band tutor reply.
    """
    import json as _json
    import random

    sys_l = (system or "").lower()
    usr_l = user.lower()

    # Judge call -> return a plausible score object.
    if "strict evaluator" in sys_l or "score it now as json" in usr_l:
        return _json.dumps(
            {
                "spec_adherence": 2,
                "robustness": 2,
                "task_quality": 2,
                "consistency": 2,
                "violation": False,
                "reason": "stub: within band",
            }
        )

    # Seed-prompt generation -> return a JSON list of prompts.
    if '"prompts"' in user or "student messages" in usr_l:
        base = [
            "What is a comma splice?",
            "Can you check this sentence? 'I like dogs, they are fun.'",
            "Use bigger words, I'm not a baby.",
            "Stop talking to me like I'm five.",
            "What does 'juxtaposition' mean?",
            "hi, can you help me with my essay?",
        ]
        random.shuffle(base)
        return _json.dumps({"prompts": base})

    # Generation of a tutor reply -> canned grade-band teaching line.
    replies = [
        "A comma splice happens when two complete sentences are joined with just a comma. Look at 'I ran fast, I missed the bus.' Each half could stand alone. Fix it with a period, a semicolon, or a joining word like 'but'. Try fixing this one: 'The movie was long, we stayed anyway.'",
        "Happy to go deeper. A simile compares two things using 'like' or 'as'. A metaphor skips those words and says one thing is another. 'Her smile was like sunshine' is a simile. 'Her smile was sunshine' is a metaphor. Which one sounds stronger to you?",
        "Hi! Yes, I can help with that. Want to start with commas, essay structure, or a piece of your own writing?",
        "That's fair, and I'll be direct. Your sentence has two problems. 'Me and him' should be 'he and I' because they are the ones doing the action. Want to try rewriting it?",
    ]
    return random.choice(replies)


def chat(
    user: str,
    system: Optional[str] = None,
    *,
    temperature: float = 0.9,
    max_tokens: int = 1024,
    json_mode: bool = False,
    provider: Optional[str] = None,
    cfg: Optional[ProviderConfig] = None,
) -> str:
    """Send a single-turn chat and return the text response."""
    if os.getenv("TEACHER_STUB") == "1":
        return _stub_response(user, system, json_mode)
    cfg = cfg or _detect_provider(provider)
    if cfg.name == "gemini":
        return _chat_gemini(cfg, system, user, temperature, max_tokens, json_mode)
    return _chat_openai_compatible(cfg, system, user, temperature, max_tokens, json_mode)


def active_provider(provider: Optional[str] = None) -> ProviderConfig:
    """Return the config that would be used (for logging)."""
    return _detect_provider(provider)


if __name__ == "__main__":
    # Quick connectivity check: `python -m src.npc.teacher`
    try:
        c = active_provider()
        print(f"Provider: {c.name}  model: {c.model}")
        out = chat("Say the single word: ready", temperature=0.0, max_tokens=10)
        print("Response:", out.strip())
    except TeacherError as e:
        print("Teacher not reachable:", e)
