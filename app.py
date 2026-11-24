"""
Render / local entrypoint for Phone Look Studio.
"""

from __future__ import annotations

import os
import traceback


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    try:
        import gradio as gr

        from src.app import _studio_theme, build_demo

        demo = build_demo()
        launch_kw: dict = {
            "server_name": "0.0.0.0",
            "server_port": port,
            "show_error": True,
        }
        if int(gr.__version__.split(".", 1)[0]) >= 6:
            launch_kw["theme"] = _studio_theme()
        demo.launch(**launch_kw)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
