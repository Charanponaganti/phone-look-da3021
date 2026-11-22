"""
Hugging Face Spaces / local entrypoint for the Phone Look Gradio app.
"""

from __future__ import annotations

import gradio as gr

from src.app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
