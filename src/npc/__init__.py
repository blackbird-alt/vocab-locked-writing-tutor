"""Sable character-tutor fine-tuning toolkit.

Modules:
- teacher: unified client for frontier teacher/judge models (Gemini/OpenRouter/Groq/OpenAI-compatible).
- prompts: generation and judge prompt templates, plus the canonical system prompts.
- schema: dataset record types and (de)serialization helpers.
"""

__all__ = ["teacher", "prompts", "schema"]
