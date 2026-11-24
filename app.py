"""
Hugging Face Spaces / local entrypoint for Phone Look Studio.
"""

from __future__ import annotations

from src.app import _studio_theme, build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=_studio_theme())
