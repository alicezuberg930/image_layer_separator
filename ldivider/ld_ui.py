import os

import gradio as gr

from ldivider.ld_segment import (
    export_mask_layers,
    get_masks,
    get_sam3_processor,
    load_masks,
    show_anns,
)


class LayerDividerUI:
    """The shared SAM 3 interface for the standalone app and WebUI extension."""

    def __init__(self, output_dir, model_dir):
        self.output_dir = os.fspath(output_dir)
        self.model_dir = os.fspath(model_dir)
        self.demo = gr.Blocks(analytics_enabled=False)
        self._build()

    def segment_image(
        self,
        input_image,
        text_prompt,
        confidence_threshold,
        mask_threshold,
        max_masks,
        min_mask_area,
    ):
        if input_image is None:
            raise ValueError("Upload an image before running SAM 3 segmentation.")
        processor = get_sam3_processor(confidence_threshold, self.model_dir)
        masks = get_masks(
            input_image,
            processor,
            text_prompt,
            mask_threshold,
            min_mask_area,
            max_masks,
        )
        return show_anns(input_image, masks, self.output_dir)

    def create_layers(self, input_image):
        if input_image is None:
            raise ValueError("Upload an image before creating PNG layers.")
        masks = load_masks(self.output_dir)
        layer_paths = export_mask_layers(input_image, masks, self.output_dir)
        previews = [(path, os.path.basename(path)) for path in layer_paths]
        return previews, layer_paths

    def _build(self):
        with self.demo:
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Input image")
                    text_prompt = gr.Textbox(
                        value="object",
                        lines=6,
                        label="SAM 3 prompts (one concept per line)",
                        show_label=True,
                    )
                    with gr.Accordion("SAM 3 settings", open=True):
                        confidence_threshold = gr.Slider(
                            0,
                            1,
                            value=0.5,
                            step=0.01,
                            label="SAM 3 confidence threshold",
                            show_label=True,
                        )
                        mask_threshold = gr.Slider(
                            0,
                            1,
                            value=0.5,
                            step=0.01,
                            label="mask threshold",
                            show_label=True,
                        )
                        max_masks = gr.Slider(
                            1,
                            200,
                            value=50,
                            step=1,
                            label="maximum masks",
                            show_label=True,
                        )
                        min_mask_area = gr.Slider(
                            0,
                            100000,
                            value=100,
                            step=10,
                            label="minimum mask area",
                            show_label=True,
                        )
                    segment = gr.Button(value="Segment with SAM 3", variant="primary")

                with gr.Column():
                    sam_output = gr.Image(type="pil", label="Segmentation preview")
                    create_layers = gr.Button(value="Create Layers")
                    layer_gallery = gr.Gallery(
                        label="Layer previews",
                        show_label=True,
                        columns=3,
                        object_fit="contain",
                    )
                    layer_files = gr.Files(label="Transparent PNG layers")

            segment.click(
                self.segment_image,
                inputs=[
                    input_image,
                    text_prompt,
                    confidence_threshold,
                    mask_threshold,
                    max_masks,
                    min_mask_area,
                ],
                outputs=[sam_output],
            )
            create_layers.click(
                self.create_layers,
                inputs=[input_image],
                outputs=[layer_gallery, layer_files],
            )
