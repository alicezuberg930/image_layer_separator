import sys
from pathlib import Path

from ldivider.ld_ui import LayerDividerUI


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    ui = LayerDividerUI(project_dir / "output", project_dir / "segment_model")
    ui.demo.queue()
    ui.demo.launch(share="share" in sys.argv[1:])
