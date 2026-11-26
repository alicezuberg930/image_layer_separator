import json
import re
import uuid
from pathlib import Path

import numpy as np
from filelock import FileLock
from PIL import Image

from ldivider.ld_segment import export_mask_layers, layer_filename, mask_to_image, save_png, show_anns


def sam3_settings(confidence_threshold, mask_threshold, max_masks, min_mask_area):
    settings = {
        "confidence_threshold": float(confidence_threshold),
        "mask_threshold": float(mask_threshold),
        "max_masks": int(max_masks),
        "min_mask_area": int(min_mask_area),
    }
    if not all(np.isfinite(value) for value in settings.values()):
        raise ValueError("SAM 3 settings must be finite numbers.")
    if not 0 <= settings["confidence_threshold"] <= 1 or not 0 <= settings["mask_threshold"] <= 1:
        raise ValueError("Thresholds must be between 0 and 1.")
    if not 1 <= settings["max_masks"] <= 200 or settings["min_mask_area"] < 0:
        raise ValueError("Maximum masks must be 1–200 and minimum area cannot be negative.")
    return settings


class LayerJobStore:
    """Persist original images, per-layer masks/settings, and stable PNG job paths."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).resolve()
        self.layers_dir = self.output_dir / "jobs" / "layers"
        self.previews_dir = self.output_dir / "jobs" / "previews"
        self.state_dir = self.output_dir / "jobs" / "state"
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Segment an image first to start a job.")
        return self.state_dir / job_id

    def job(self, job_id):
        try:
            return json.loads((self._job_dir(job_id) / "job.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError("This job is unavailable. Click Segment to start again.") from exc

    def source(self, job_id):
        self.job(job_id)
        with Image.open(self._job_dir(job_id) / "source.png") as image:
            return image.convert("RGBA")

    def layer(self, job_id, index):
        job = self.job(job_id)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(job["layers"]):
            raise ValueError("Select a layer photo first.")
        return job["layers"][index]

    def layer_index(self, job_id, index, filename=None):
        if filename is None:
            self.layer(job_id, index)
            return index
        for position, layer in enumerate(self.job(job_id)["layers"]):
            if layer["filename"] == filename:
                return position
        raise ValueError("This layer has been deleted. Choose another layer.")

    def _mask_path(self, job_id, index, layer=None):
        layer = layer if layer is not None else self.layer(job_id, index)
        filename = layer.get("mask_file", f"mask_{index:03d}.npz")
        if not isinstance(filename, str) or not re.fullmatch(r"mask_[0-9]+\.npz", filename):
            raise ValueError("The saved mask filename is invalid.")
        return self._job_dir(job_id) / filename

    def _save_job(self, job_id, job):
        path = self._job_dir(job_id) / "job.json"
        temporary = path.with_name(f".job.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_mask(self, job_id, index, mask, settings, mask_file=None):
        layer = {"mask_file": mask_file} if mask_file is not None else None
        path = self._mask_path(job_id, index, layer)
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as file:
                np.savez_compressed(
                    file, segmentation=np.asarray(mask["segmentation"], dtype=bool),
                    score=float(mask.get("score", 0)), settings=json.dumps(settings),
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def mask(self, job_id, index):
        layer = self.layer(job_id, index)
        with np.load(self._mask_path(job_id, index, layer), allow_pickle=False) as data:
            segmentation = data["segmentation"].copy()
            return {
                "segmentation": segmentation, "area": int(segmentation.sum()),
                "score": float(data["score"]), "prompt": layer["prompt"],
            }, json.loads(str(data["settings"]))

    def create_job(self, source, masks, text_prompt, settings):
        if not masks:
            raise ValueError("SAM 3 found no usable masks.")
        source = source.convert("RGBA")
        masks = sorted(masks, key=lambda mask: mask["area"], reverse=True)
        if any(np.asarray(mask["segmentation"]).shape != (source.height, source.width) for mask in masks):
            raise ValueError("The masks do not match the original image.")
        job_id = uuid.uuid4().hex
        directory = self._job_dir(job_id)
        directory.mkdir(parents=True)
        (self.layers_dir / job_id).mkdir()
        save_png(source, directory / "source.png")
        layers = []
        for index, mask in enumerate(masks):
            prompt = mask.get("prompt", "mask")
            mask_file = f"mask_{index:03d}.npz"
            layers.append({"prompt": prompt, "filename": layer_filename(index + 1, prompt), "mask_file": mask_file})
            self._save_mask(job_id, index, mask, settings, mask_file=mask_file)
        job = {"id": job_id, "text_prompt": text_prompt, "settings": settings, "layers": layers}
        self._save_job(job_id, job)
        self._save_preview(job_id)
        return job_id

    def _save_preview(self, job_id):
        masks = [self.mask(job_id, index)[0] for index in range(len(self.job(job_id)["layers"]))]
        path = self.previews_dir / job_id / "segmentation.png"
        save_png(show_anns(self.source(job_id), masks), path)
        return path

    def preview_path(self, job_id):
        self.job(job_id)
        path = self.previews_dir / job_id / "segmentation.png"
        if not path.is_file():
            with FileLock(str(self._job_dir(job_id) / "job.lock")):
                return self._save_preview(job_id)
        return path

    def layer_paths(self, job_id, required=False):
        paths = [str(self.layers_dir / job_id / layer["filename"]) for layer in self.job(job_id)["layers"]]
        if required and not paths:
            raise ValueError("No layers remain in this job. Segment again to create new layers.")
        if not all(Path(path).is_file() for path in paths):
            if required:
                raise ValueError("Click Create Layers before exporting or editing layers.")
            return []
        return paths

    def create_layers(self, job_id):
        self.job(job_id)
        with FileLock(str(self._job_dir(job_id) / "job.lock")):
            job = self.job(job_id)
            if not job["layers"]:
                return []
            masks = [self.mask(job_id, index)[0] for index in range(len(job["layers"]))]
            return export_mask_layers(
                self.source(job_id), masks, self.layers_dir / job_id,
                filenames=[layer["filename"] for layer in job["layers"]],
            )

    def replace_layer(self, job_id, index, mask, settings, expected_filename=None):
        self.job(job_id)
        with FileLock(str(self._job_dir(job_id) / "job.lock")):
            self.layer_paths(job_id, required=True)
            index = self.layer_index(job_id, index, expected_filename)
            layer = self.layer(job_id, index)
            image = mask_to_image(self.source(job_id), mask)
            save_png(image, self.layers_dir / job_id / layer["filename"])
            self._save_mask(job_id, index, mask, settings)
            self._save_preview(job_id)
        return self.layer_paths(job_id, required=True)

    def delete_layer(self, job_id, filename):
        """Permanently remove one PNG and its saved mask without renumbering others."""
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".png"):
            raise ValueError("Choose a valid layer to delete.")
        self.job(job_id)
        with FileLock(str(self._job_dir(job_id) / "job.lock")):
            job = self.job(job_id)
            index = next((i for i, layer in enumerate(job["layers"]) if layer["filename"] == filename), None)
            if index is None:
                return self.layer_paths(job_id)
            # Older jobs used list positions for mask filenames. Persist those
            # original identities before removing an entry and shifting indices.
            for position, layer in enumerate(job["layers"]):
                layer.setdefault("mask_file", f"mask_{position:03d}.npz")
            layer = job["layers"][index]
            path = self.layers_dir / job_id / layer["filename"]
            if not path.resolve().is_relative_to(self.layers_dir.resolve()):
                raise ValueError("The layer is outside its job folder.")
            path.unlink(missing_ok=True)
            self._mask_path(job_id, index, layer).unlink(missing_ok=True)
            job["layers"].pop(index)
            self._save_job(job_id, job)
            self._save_preview(job_id)
        return self.layer_paths(job_id)


def matching_mask(candidates, previous):
    """Keep the selected instance when its prompt detects several objects."""
    original = np.asarray(previous["segmentation"], dtype=bool)
    def rank(mask):
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        if segmentation.shape != original.shape:
            raise ValueError("The new mask does not match the original image.")
        union = np.logical_or(original, segmentation).sum()
        overlap = float(np.logical_and(original, segmentation).sum()) / union if union else 0
        return overlap, float(mask.get("score", 0))
    if not candidates:
        raise ValueError("No matching mask was found. Adjust the sliders and try again.")
    return max(candidates, key=rank)
