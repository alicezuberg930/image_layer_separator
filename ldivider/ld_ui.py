import functools
import json
import os
from pathlib import Path

import gradio as gr

from layerdivider_runtime import sam3_cache
from ldivider.ld_jobs import LayerJobStore, matching_mask, sam3_settings
from ldivider.ld_psd import export_layers_to_psd
from ldivider.ld_segment import get_masks, get_sam3_processor, mask_to_image, parse_text_prompts, show_anns


SETTINGS = ("confidence_threshold", "mask_threshold", "max_masks", "min_mask_area")
VIEW = ("job_id", "sam_output", "gallery", "files", "psd_file", "selected", "editor", "draft")
EDIT_PENCILS_JS = Path(__file__).with_name("layer_edit.js").read_text(encoding="utf-8")
CSS = """
#layer-preview-group { position: relative; }
#layer-edit-bridge-container { display: none !important; }
#layer-gallery .layer-edit-tile { position: relative; isolation: isolate; z-index: 0; min-width: 0; min-height: 0; aspect-ratio: 1; }
#layer-gallery .layer-edit-tile > .thumbnail-lg { width: 100%; height: 100%; }
#layer-gallery .layer-action-button { position: absolute; top: 6px; right: 6px; z-index: 2; display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; min-width: 32px; padding: 0; border: 1px solid var(--border-color-primary); border-radius: 50%; background: var(--background-fill-primary); color: var(--body-text-color); font-size: 21px; line-height: 1; cursor: pointer; box-shadow: 0 2px 8px #0004; }
#layer-gallery .layer-edit-pencil { right: 44px; }
#layer-gallery .layer-action-button:hover { color: var(--color-accent); }
#layer-gallery .layer-delete-button:hover { color: #ef4444; }
#layer-gallery .layer-action-button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
#layer-gallery .layer-action-button--preview { position: static; margin: 4px; flex: none; }
#layer-editor-modal { position: fixed !important; inset: 0 !important; z-index: 1000; width: 100% !important; height: 100vh !important; height: 100dvh !important; max-width: none !important; overflow-y: auto !important; background: rgba(0,0,0,.7) !important; border: none !important; padding: 20px !important; box-sizing: border-box; }
/* Gradio's Group wrappers otherwise shrink and clip the dialog's contents. */
#layer-editor-modal > .styler, #layer-editor-dialog > .styler { overflow: visible; flex-shrink: 0; }
#layer-editor-dialog { width: min(1100px, 100%); margin: 0 auto; max-height: calc(100vh - 40px); max-height: calc(100dvh - 40px); overflow-y: auto !important; overscroll-behavior-y: contain; flex-shrink: 0; box-sizing: border-box; padding: 20px; border: 1px solid var(--border-color-primary); border-radius: 12px; background: var(--background-fill-primary); box-shadow: 0 20px 80px #0008; }
#layer-editor-dialog .layer-editor-image { height: auto; min-width: 0; }
#layer-editor-dialog .layer-editor-image > button { display: block; width: 100%; }
#layer-editor-dialog .layer-editor-image img { display: block; width: 100%; height: auto; max-height: none; object-fit: contain; }
@media (max-width: 650px) { #layer-editor-modal { padding: 8px !important; } #layer-editor-dialog { padding: 12px; max-height: calc(100vh - 16px); max-height: calc(100dvh - 16px); } }
"""
READ_JOB_JS = """() => {
  if (!window.__layerdividerEditorKeys) {
    document.addEventListener('keydown', (event) => {
      const modal = document.getElementById('layer-editor-modal');
      if (event.key === 'Escape' && modal && getComputedStyle(modal).display !== 'none') {
        document.getElementById('cancel-layer-editor')?.click();
      }
    });
    window.__layerdividerEditorKeys = true;
  }
  try { return [localStorage.getItem('layerdivider.current-job') || '']; }
  catch (_) { return ['']; }
}"""
SAVE_JOB_JS = """(id) => {
  try { if (id) localStorage.setItem('layerdivider.current-job', id);
        else localStorage.removeItem('layerdivider.current-job'); } catch (_) {}
  return [];
}"""
OPEN_DIALOG_JS = """() => new Promise(resolve => requestAnimationFrame(() => {
  const modal = document.getElementById('layer-editor-modal');
  if (!modal || getComputedStyle(modal).display === 'none') { resolve([]); return; }
  const dialog = document.getElementById('layer-editor-dialog');
  if (dialog) { dialog.setAttribute('role', 'dialog'); dialog.setAttribute('aria-modal', 'true'); dialog.setAttribute('aria-label', 'Edit selected layer'); }
  document.querySelector('#close-layer-editor')?.focus();
  resolve([]);
}))"""


def ui_action(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
    return wrapped


class LayerDividerUI:
    """SAM 3 interface with persistent browser jobs and per-layer editing."""

    def __init__(self, output_dir, model_dir):
        self.output_dir = os.fspath(output_dir)
        self.model_dir = os.fspath(model_dir)
        self.jobs = LayerJobStore(output_dir)
        self.components = {}
        self.demo = gr.Blocks(analytics_enabled=False, css=CSS)
        self._build()

    def _updates(self, **values):
        return {self.components[name]: value for name, value in values.items()}

    def _outputs(self, *names):
        return [self.components[name] for name in names]

    def _view(self, job_id):
        paths = self.jobs.layer_paths(job_id) if job_id else []
        return self._updates(
            job_id=job_id,
            sam_output=str(self.jobs.preview_path(job_id)) if job_id else None,
            gallery=[(path, Path(path).name) for path in paths], files=paths,
            psd_file=None, selected=None, draft=None,
            editor=gr.update(visible=False),
        )

    def _infer(self, source, prompt, settings):
        if not parse_text_prompts(prompt):
            raise ValueError("Enter at least one SAM 3 prompt.")
        with sam3_cache.inference_lock:
            processor = get_sam3_processor(settings["confidence_threshold"], self.model_dir)
            return get_masks(
                source, processor, prompt, settings["mask_threshold"],
                settings["min_mask_area"], settings["max_masks"],
            )

    @ui_action
    def segment_image(self, input_image, text_prompt, confidence_threshold, mask_threshold, max_masks, min_mask_area):
        if input_image is None:
            raise ValueError("Upload an image before running SAM 3 segmentation.")
        settings = sam3_settings(confidence_threshold, mask_threshold, max_masks, min_mask_area)
        masks = self._infer(input_image, text_prompt, settings)
        job_id = self.jobs.create_job(input_image, masks, text_prompt, settings)
        return self._view(job_id)

    @ui_action
    def create_layers(self, job_id):
        self.jobs.create_layers(job_id)
        return self._view(job_id)

    @ui_action
    def generate_psd(self, job_id):
        return export_layers_to_psd(
            self.jobs.layer_paths(job_id, required=True), Path(self.output_dir) / "jobs" / "psd"
        )

    def restore_job(self, job_id):
        try:
            job = self.jobs.job(job_id)
            updates = self._view(job_id)
            updates.update(self._updates(input_image=self.jobs.source(job_id), text_prompt=job["text_prompt"], **job["settings"]))
            return updates
        except (ValueError, FileNotFoundError):
            return self._view("")

    @ui_action
    def select_layer(self, job_id, evt: gr.SelectData):
        if not evt.selected:
            return self._updates(selected=None, editor=gr.update(visible=False), draft=None)
        self.jobs.layer_paths(job_id, required=True)
        index = evt.index
        self.jobs.layer(job_id, index)
        return self._updates(selected=index, editor=gr.update(visible=False), draft=None)

    @ui_action
    def open_layer_from_preview(self, job_id, edit_request):
        try:
            request = json.loads(edit_request)
            index = request["index"]
            action = request.get("action", "edit")
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Use the buttons on a layer preview.") from exc
        if action == "delete":
            self.jobs.delete_layer(job_id, request.get("filename"))
            return self._view(job_id)
        if action != "edit":
            raise ValueError("Choose a valid layer action.")
        index = self.jobs.layer_index(job_id, index, request.get("filename"))
        updates = self.open_layer_editor(job_id, index)
        updates.update(self._updates(selected=index))
        return updates

    @ui_action
    def open_layer_editor(self, job_id, index):
        self.jobs.layer_paths(job_id, required=True)
        layer = self.jobs.layer(job_id, index)
        mask, settings = self.jobs.mask(job_id, index)
        values = {"edit_" + name: value for name, value in settings.items()}
        return self._updates(
            editor=gr.update(visible=True), edit_prompt=layer["prompt"],
            editor_original=show_anns(self.jobs.source(job_id), [mask]),
            editor_preview=str(self.jobs.layers_dir / job_id / layer["filename"]),
            draft={"job_id": job_id, "filename": layer["filename"]}, **values,
        )

    def _draft(self, job_id, index, settings):
        layer = self.jobs.layer(job_id, index)
        previous, _ = self.jobs.mask(job_id, index)
        source = self.jobs.source(job_id)
        candidates = self._infer(source, layer["prompt"], settings)
        mask = matching_mask(candidates, previous)
        return {"job_id": job_id, "index": index, "filename": layer["filename"], "settings": settings, "mask": mask}

    def _edit_index(self, job_id, index, draft):
        filename = draft.get("filename") if draft and draft.get("job_id") == job_id else None
        return self.jobs.layer_index(job_id, index, filename)

    @ui_action
    def preview_layer(self, job_id, index, confidence_threshold, mask_threshold, max_masks, min_mask_area, draft=None):
        settings = sam3_settings(confidence_threshold, mask_threshold, max_masks, min_mask_area)
        index = self._edit_index(job_id, index, draft)
        draft = self._draft(job_id, index, settings)
        source = self.jobs.source(job_id)
        return self._updates(
            editor_original=show_anns(source, [draft["mask"]]),
            editor_preview=mask_to_image(source, draft["mask"]), draft=draft,
        )

    @ui_action
    def apply_layer_edit(self, job_id, index, confidence_threshold, mask_threshold, max_masks, min_mask_area, draft):
        settings = sam3_settings(confidence_threshold, mask_threshold, max_masks, min_mask_area)
        index = self._edit_index(job_id, index, draft)
        if not draft or draft.get("job_id") != job_id or draft.get("settings") != settings or "mask" not in draft:
            draft = self._draft(job_id, index, settings)
        self.jobs.replace_layer(job_id, index, draft["mask"], settings, expected_filename=draft["filename"])
        index = self.jobs.layer_index(job_id, index, draft["filename"])
        updates = self._view(job_id)
        updates.update(self._updates(selected=index))
        return updates

    def close_editor(self):
        return self._updates(editor=gr.update(visible=False), draft=None)

    def _build(self):
        c = self.components
        def add(name, component):
            c[name] = component
            return component

        def settings_sliders(prefix=""):
            definitions = (
                ("confidence_threshold", 0, 1, 0.5, 0.01, "SAM 3 confidence threshold"),
                ("mask_threshold", 0, 1, 0.5, 0.01, "mask threshold"),
                ("max_masks", 1, 200, 50, 1, "maximum masks"),
                ("min_mask_area", 0, 100000, 100, 10, "minimum mask area"),
            )
            for name, minimum, maximum, value, step, label in definitions:
                add(prefix + name, gr.Slider(minimum, maximum, value=value, step=step, label=label, show_label=True))

        with self.demo:
            add("job_id", gr.Textbox(value="", visible=False, elem_id="active-job-id"))
            add("selected", gr.State(None))
            add("draft", gr.State(None))
            with gr.Group(elem_id="layer-edit-bridge-container"):
                add("edit_request", gr.Textbox(value="", interactive=True, show_label=False, container=False, elem_id="layer-edit-request"))
            with gr.Row():
                with gr.Column():
                    add("input_image", gr.Image(type="pil", image_mode="RGBA", label="Input image", elem_id="source-image"))
                    add("text_prompt", gr.Textbox(value="object", lines=6, label="SAM 3 prompts (one concept per line)", show_label=True))
                    with gr.Accordion("SAM 3 settings", open=True):
                        settings_sliders()
                    segment = gr.Button(value="Segment with SAM 3", variant="primary")
                with gr.Column():
                    add("sam_output", gr.Image(type="pil", label="Segmentation preview"))
                    create = gr.Button(value="Create Layers")
                    with gr.Group(elem_id="layer-preview-group"):
                        add("gallery", gr.Gallery(label="Layer previews", show_label=True, columns=3, object_fit="contain", elem_id="layer-gallery"))
                    add("files", gr.Files(label="Transparent PNG layers"))
                    generate = gr.Button(value="Generate PSD")
                    add("psd_file", gr.File(label="Layered PSD"))

            with gr.Group(visible=False, elem_id="layer-editor-modal") as editor:
                add("editor", editor)
                with gr.Group(elem_id="layer-editor-dialog"):
                    with gr.Row():
                        gr.Markdown("### Edit selected layer")
                        close = gr.Button(value="Close", min_width=70, scale=0, elem_id="close-layer-editor")
                    add("edit_prompt", gr.Textbox(label="Part name", interactive=False))
                    with gr.Row(equal_height=False):
                        add("editor_original", gr.Image(type="pil", label="Original image (mask preview)", interactive=False, elem_classes="layer-editor-image"))
                        add("editor_preview", gr.Image(type="pil", label="Layer preview", interactive=False, elem_classes="layer-editor-image"))
                    with gr.Accordion("SAM 3 settings", open=True):
                        settings_sliders("edit_")
                    with gr.Row():
                        cancel = gr.Button(value="Cancel", elem_id="cancel-layer-editor")
                        preview = gr.Button(value="Preview")
                        apply = gr.Button(value="Replace layer", variant="primary")

            segmented = segment.click(
                self.segment_image,
                inputs=self._outputs("input_image", "text_prompt", *SETTINGS), outputs=self._outputs(*VIEW),
            )
            segmented.success(fn=None, inputs=[c["job_id"]], outputs=[], _js=SAVE_JOB_JS, queue=False)
            create.click(self.create_layers, inputs=[c["job_id"]], outputs=self._outputs(*VIEW))
            generate.click(self.generate_psd, inputs=[c["job_id"]], outputs=[c["psd_file"]])
            c["gallery"].select(self.select_layer, inputs=[c["job_id"]], outputs=self._outputs("selected", "editor", "draft"))
            opened = c["edit_request"].input(
                self.open_layer_from_preview, inputs=self._outputs("job_id", "edit_request"),
                outputs=self._outputs(*VIEW, "edit_prompt", "editor_original", "editor_preview", *("edit_" + name for name in SETTINGS)),
            )
            opened.success(fn=None, inputs=[], outputs=[], _js=OPEN_DIALOG_JS, queue=False)
            edit_inputs = self._outputs("job_id", "selected", *("edit_" + name for name in SETTINGS))
            preview.click(self.preview_layer, inputs=edit_inputs + [c["draft"]], outputs=self._outputs("editor_original", "editor_preview", "draft"))
            apply.click(self.apply_layer_edit, inputs=edit_inputs + [c["draft"]], outputs=self._outputs(*VIEW))
            for button in (close, cancel):
                button.click(self.close_editor, inputs=[], outputs=self._outputs("editor", "draft"), queue=False)
            restored = self.demo.load(
                self.restore_job, inputs=[c["job_id"]],
                outputs=self._outputs(*VIEW, "input_image", "text_prompt", *SETTINGS),
                _js=READ_JOB_JS,
            )
            restored.success(fn=None, inputs=[c["job_id"]], outputs=[], _js=SAVE_JOB_JS, queue=False)
            self.demo.load(fn=None, inputs=[], outputs=[], _js=EDIT_PENCILS_JS, queue=False)
