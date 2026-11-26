"""Standalone SAM 3 HTTP API. Run with: python scripts/api.py"""

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_DIR))

from ldivider.ld_jobs import LayerJobStore, matching_mask, sam3_settings
from ldivider.ld_psd import export_layers_to_psd
from ldivider.ld_segment import get_masks, get_sam3_processor, parse_text_prompts
from layerdivider_runtime import sam3_cache

# SAM 3 caches a mutable processor, so inference calls must run one at a time.
_inference_lock = sam3_cache.inference_lock


class ImageResponse(BaseModel):
    segmentation_id: str
    images: list[str] = Field(description="Absolute URLs of the generated PNG images.")


class CreateLayerRequest(BaseModel):
    segmentation_id: str = Field(
        ...,
        regex=r"^[0-9a-f]{32}$",
        description="The segmentation_id returned by POST /segment.",
    )


class LayerRequest(CreateLayerRequest):
    filename: str = Field(
        ...,
        regex=r"^[^/\\]+\.png$",
        description="A layer filename returned in an image URL by POST /create-layer.",
    )


class EditLayerRequest(LayerRequest):
    confidence_threshold: float | None = Field(
        None, ge=0, le=1, description="Defaults to this layer's saved setting."
    )
    mask_threshold: float | None = Field(
        None, ge=0, le=1, description="Defaults to this layer's saved setting."
    )
    max_masks: int | None = Field(
        None, ge=1, le=200, description="Defaults to this layer's saved setting."
    )
    min_mask_area: int | None = Field(
        None, ge=0, description="Defaults to this layer's saved setting."
    )


class PSDResponse(BaseModel):
    segmentation_id: str
    psd: str = Field(description="Absolute URL of the generated layered PSD file.")


def create_app(
    output_dir: str | Path = PROJECT_DIR / "output",
    model_dir: str | Path = PROJECT_DIR / "segment_model",
) -> FastAPI:
    output_dir = Path(output_dir).resolve()
    model_dir = Path(model_dir).resolve()
    jobs = LayerJobStore(output_dir)
    psd_dir = output_dir / "jobs" / "psd"
    psd_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="LayerDivider API",
        description=(
            "Segment an image with SAM 3, edit or delete its transparent PNG "
            "layers, and export a layered PSD."
        ),
        version="1.1.0",
    )
    # Only generated images are public; source images and saved masks stay private.
    app.state.jobs = jobs
    app.mount("/images", StaticFiles(directory=jobs.layers_dir), name="images")
    app.mount("/previews", StaticFiles(directory=jobs.previews_dir), name="previews")
    app.mount("/psd", StaticFiles(directory=psd_dir), name="psd")

    def image_urls(
        request: Request, paths: list[Path], directory=None, route="images"
    ) -> list[str]:
        directory = directory or jobs.layers_dir
        return [
            str(request.url_for(route, path=path.relative_to(directory).as_posix()))
            for path in paths
        ]

    def layer_response(
        request: Request, segmentation_id: str, paths: list[str]
    ) -> ImageResponse:
        return ImageResponse(
            segmentation_id=segmentation_id,
            images=image_urls(request, [Path(path) for path in paths]),
        )

    def layer_settings(payload: EditLayerRequest, saved: dict) -> dict:
        return sam3_settings(
            payload.confidence_threshold
            if payload.confidence_threshold is not None
            else saved["confidence_threshold"],
            payload.mask_threshold
            if payload.mask_threshold is not None
            else saved["mask_threshold"],
            payload.max_masks if payload.max_masks is not None else saved["max_masks"],
            payload.min_mask_area
            if payload.min_mask_area is not None
            else saved["min_mask_area"],
        )

    @app.post(
        "/segment",
        response_model=ImageResponse,
        summary="Segment an uploaded image with SAM 3",
        description=(
            "Send an image and prompts as multipart/form-data. Returns a combined "
            "segmentation preview in images and a segmentation_id for /create-layer."
        ),
    )
    def segment(
        request: Request,
        image: UploadFile = File(..., description="The image to segment."),
        text_prompt: str = Form("object", description="One concept per line."),
        confidence_threshold: float = Form(0.5, ge=0, le=1),
        mask_threshold: float = Form(0.5, ge=0, le=1),
        max_masks: int = Form(50, ge=1, le=200),
        min_mask_area: int = Form(100, ge=0),
    ) -> ImageResponse:
        if not parse_text_prompts(text_prompt):
            raise HTTPException(
                status_code=422, detail="Enter at least one SAM 3 prompt."
            )
        try:
            with Image.open(image.file) as uploaded:
                source = ImageOps.exif_transpose(uploaded).convert("RGBA")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(
                status_code=400, detail="Upload a valid, supported image file."
            ) from exc

        try:
            with _inference_lock:
                processor = get_sam3_processor(confidence_threshold, str(model_dir))
                masks = get_masks(
                    source,
                    processor,
                    text_prompt,
                    mask_threshold,
                    min_mask_area,
                    max_masks,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        settings = sam3_settings(
            confidence_threshold, mask_threshold, max_masks, min_mask_area
        )
        segmentation_id = jobs.create_job(source, masks, text_prompt, settings)
        return ImageResponse(
            segmentation_id=segmentation_id,
            images=image_urls(
                request,
                [jobs.preview_path(segmentation_id)],
                jobs.previews_dir,
                "previews",
            ),
        )

    @app.post(
        "/create-layer",
        response_model=ImageResponse,
        summary="Export transparent layers from a segmentation",
        description=(
            "Send the segmentation_id from /segment as JSON. Returns one PNG URL "
            "per retained mask, preserving the source image's size and colors."
        ),
    )
    def create_layer(request: Request, payload: CreateLayerRequest) -> ImageResponse:
        try:
            paths = jobs.create_layers(payload.segmentation_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Segmentation not found."
            ) from exc
        return layer_response(request, payload.segmentation_id, paths)

    @app.post(
        "/edit-layer",
        response_model=ImageResponse,
        summary="Rerun SAM 3 and replace one layer",
        description=(
            "Select a layer by its stable PNG filename. Omitted SAM 3 settings "
            "reuse that layer's saved values. The best-overlapping new mask "
            "replaces only the selected layer."
        ),
    )
    @app.patch(
        "/edit-layer",
        response_model=ImageResponse,
        summary="Rerun SAM 3 and replace one layer",
        description=(
            "Select a layer by its stable PNG filename. Omitted SAM 3 settings "
            "reuse that layer's saved values. The best-overlapping new mask "
            "replaces only the selected layer."
        ),
    )
    def edit_layer(request: Request, payload: EditLayerRequest) -> ImageResponse:
        try:
            jobs.layer_paths(payload.segmentation_id, required=True)
            index = jobs.layer_index(payload.segmentation_id, 0, payload.filename)
            layer = jobs.layer(payload.segmentation_id, index)
            previous, saved_settings = jobs.mask(payload.segmentation_id, index)
            settings = layer_settings(payload, saved_settings)
            source = jobs.source(payload.segmentation_id)
            with _inference_lock:
                processor = get_sam3_processor(
                    settings["confidence_threshold"], str(model_dir)
                )
                candidates = get_masks(
                    source,
                    processor,
                    layer["prompt"],
                    settings["mask_threshold"],
                    settings["min_mask_area"],
                    settings["max_masks"],
                )
            mask = matching_mask(candidates, previous)
            paths = jobs.replace_layer(
                payload.segmentation_id,
                index,
                mask,
                settings,
                expected_filename=payload.filename,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Segmentation not found."
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return layer_response(request, payload.segmentation_id, paths)

    @app.post(
        "/delete-layer",
        response_model=ImageResponse,
        summary="Permanently delete one layer",
        description=(
            "Delete a layer PNG and its saved mask by stable filename. Returns "
            "the URLs of the remaining layers; repeating a deletion is safe."
        ),
    )
    @app.delete(
        "/delete-layer",
        response_model=ImageResponse,
        summary="Permanently delete one layer",
        description=(
            "Delete a layer PNG and its saved mask by stable filename. Returns "
            "the URLs of the remaining layers; repeating a deletion is safe."
        ),
    )
    def delete_layer(request: Request, payload: LayerRequest) -> ImageResponse:
        try:
            paths = jobs.delete_layer(payload.segmentation_id, payload.filename)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Segmentation not found."
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return layer_response(request, payload.segmentation_id, paths)

    @app.post(
        "/create-psd",
        response_model=PSDResponse,
        summary="Create a layered PSD from the current PNG layers",
        description=(
            "Export the job's current, generated PNG layers to a new PSD. "
            "Generate layers first; edits and deletions are reflected in the export."
        ),
    )
    def create_psd(request: Request, payload: CreateLayerRequest) -> PSDResponse:
        try:
            layer_paths = jobs.layer_paths(payload.segmentation_id, required=True)
            path = Path(export_layers_to_psd(layer_paths, psd_dir))
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Segmentation not found."
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PSDResponse(
            segmentation_id=payload.segmentation_id,
            psd=str(request.url_for("psd", path=path.relative_to(psd_dir).as_posix())),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the LayerDivider SAM 3 API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
