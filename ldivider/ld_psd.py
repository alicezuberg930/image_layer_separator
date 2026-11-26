import uuid
from pathlib import Path
from typing import Sequence

from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Compression


def export_layers_to_psd(
    layer_paths: Sequence[str | Path], output_dir: str | Path
) -> str:
    """Combine separated PNGs into a PSD with one editable pixel layer per image."""
    if not layer_paths:
        raise ValueError("Click Create Layers before generating a PSD.")

    with Image.open(layer_paths[0]) as first_layer:
        canvas_size = first_layer.size
    # PSDImage.new otherwise stores the full-canvas merged preview as raw pixels.
    psd = PSDImage.new(
        "RGBA", canvas_size, color=(0, 0, 0, 0), depth=8, compression=Compression.RLE
    )

    # PNG order is background to foreground, matching PSD layer insertion order.
    for layer_path in layer_paths:
        layer_path = Path(layer_path)
        with Image.open(layer_path) as layer_image:
            if layer_image.size != canvas_size:
                raise ValueError("All layers must have the same canvas size.")
            psd.create_pixel_layer(
                layer_image.convert("RGBA"), name=layer_path.stem, top=0, left=0
            )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    psd_path = output_dir / f"_{uuid.uuid4()}.psd"
    psd.save(psd_path)
    return str(psd_path)
