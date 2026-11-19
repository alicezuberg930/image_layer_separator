from pathlib import Path

import modules.scripts as scripts
from modules import script_callbacks

from ldivider.ld_ui import LayerDividerUI


class Script(scripts.Script):
    def title(self):
        return "LayerDivider"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def on_ui_tabs():
    project_dir = Path(__file__).resolve().parent.parent
    ui = LayerDividerUI(project_dir / "output", project_dir / "segment_model")
    return [(ui.demo, "LayerDivider", "layerdivider")]


script_callbacks.on_ui_tabs(on_ui_tabs)
