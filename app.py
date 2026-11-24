"""
Render / local entrypoint for Phone Look Studio.
"""

from __future__ import annotations

import os
import traceback


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    try:
        from src.app import build_demo, launch_kwargs

        demo = build_demo()
        demo.launch(**launch_kwargs(port))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
