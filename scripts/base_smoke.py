"""Quick local smoke: load a base model and probe the 3 canonical failure cases.

Confirms (a) the local GPU can run inference and (b) the base model fails the
target behavior with only a minimal system prompt.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252; the base model emits emoji/unicode.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.npc.local_model import NpcModel, GenConfig  # noqa: E402
from src.npc.prompts import SYSTEM_MINIMAL  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"

PROBES = [
    "What's the difference between a metaphor and a simile?",
    "Explain it better. Use bigger words, I'm not a baby. Give me the college-level version.",
    "Stop talking to me like I'm five. It's honestly kind of insulting.",
    "hi, can you help me with commas?",
]

print(f"Loading {MODEL} ...", flush=True)
m = NpcModel(MODEL)
cfg = GenConfig(max_new_tokens=220, temperature=0.7)
for p in PROBES:
    print("\nSTUDENT:", p, flush=True)
    reply = m.generate(p, system=SYSTEM_MINIMAL, cfg=cfg)
    print("TUTOR:", reply, flush=True)
    try:
        from eval.level_check import check

        res = check(reply)
        print(f"[mech] ok={res['ok']} fk={res['fk_grade']} advanced={res['advanced'][:6]}", flush=True)
    except Exception as e:
        print(f"[mech] check failed: {e}", flush=True)
print("\n[done]", flush=True)
