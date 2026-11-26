import argparse
import os
from pathlib import Path

project_dir = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LayerDivider Gradio app.")
    parser.add_argument(
        "--reload", action="store_true", help="Reload the UI when app source files change."
    )
    # Keep compatibility with the existing Colab command: python demo.py share.
    parser.add_argument("legacy_share", nargs="?", choices=("share",), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reload:
        os.environ["GRADIO_WATCH_DIRS"] = ",".join(
            str(project_dir / name) for name in ("ldivider", "scripts")
        )
        os.environ["GRADIO_WATCH_FILE"] = "demo"
        os.environ["GRADIO_WATCH_DEMO_NAME"] = "demo"
        print("Reload enabled: watching ldivider and scripts.", flush=True)

# Gradio reads reload settings at import time, so configure them first.
from ldivider.ld_ui import LayerDividerUI

ui = LayerDividerUI(project_dir / "output", project_dir / "segment_model")
demo = ui.demo
demo.queue()

if __name__ == "__main__":
    demo.launch(share=True)