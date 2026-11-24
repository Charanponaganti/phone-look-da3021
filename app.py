"""
Hugging Face Spaces / Render / local entrypoint for Phone Look Studio.
"""

from __future__ import annotations

import os

from src.app import _studio_theme, build_demo

demo = build_demo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=_studio_theme(),
        show_error=True,
    )
