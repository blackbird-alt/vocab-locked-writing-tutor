"""Gradio chat demo for the vocabulary-locked tutor (runs locally on the 4GB GPU).

Usage:
    python demo/app.py --model your-username/qwen3-0.6b-leveled-tutor
    python demo/app.py --model Qwen/Qwen3-0.6B --adapter outputs/tutor-0.6b-v5

Set --compare to load BOTH base and tuned side by side so viewers can see the
base model escalate/cave while the tuned model holds the level - great for the
demo video.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.npc.local_model import NpcModel, GenConfig  # noqa: E402
from src.npc.prompts import SYSTEM_PROMPTS  # noqa: E402


def build_ui(tuned: NpcModel, base: NpcModel | None, system: str, cfg: GenConfig):
    import gradio as gr

    def respond(message, history):
        # Gradio supplies history as [{"role","content"}, ...]; pass the last 8
        # turns so follow-ups ("wait, was I right?") see the conversation.
        hist = [{"role": h["role"], "content": h["content"]}
                for h in (history or [])][-8:]
        reply = tuned.generate(message, system=system, cfg=cfg, history=hist)
        return reply

    if base is None:
        return gr.ChatInterface(
            respond,
            title="Writing Tutor (grade 7-8 locked)",
            description="A writing tutor that never escalates above your level - no matter how hard you push.",
        )

    # Comparison mode: show base vs tuned for the same message.
    with gr.Blocks(title="Leveled tutor - base vs tuned") as demo:
        gr.Markdown("# Writing tutor, base vs tuned\nSame prompt, both models. Watch the base escalate.")
        with gr.Row():
            inp = gr.Textbox(label="Ask your writing tutor", scale=4)
            btn = gr.Button("Send", scale=1)
        with gr.Row():
            base_out = gr.Textbox(label="BASE (untuned)")
            tuned_out = gr.Textbox(label="TUNED (ours)")

        def compare(msg):
            return (
                base.generate(msg, system=system, cfg=cfg),
                tuned.generate(msg, system=system, cfg=cfg),
            )

        btn.click(compare, inputs=inp, outputs=[base_out, tuned_out])
        inp.submit(compare, inputs=inp, outputs=[base_out, tuned_out])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Tuned model id/path")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--compare", default=None, help="Base model id to compare against")
    ap.add_argument("--system", default="minimal", choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    system = SYSTEM_PROMPTS[args.system]
    cfg = GenConfig()

    print("Loading tuned model ...")
    tuned = NpcModel(args.model, adapter_id=args.adapter, load_in_4bit=args.load_in_4bit)
    base = None
    if args.compare:
        print("Loading base model for comparison ...")
        base = NpcModel(args.compare, load_in_4bit=args.load_in_4bit)

    ui = build_ui(tuned, base, system, cfg)
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
