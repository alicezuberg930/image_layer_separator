import copy
import os
import re
import uuid
from contextlib import nullcontext
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from PIL import Image

from layerdivider_runtime import sam3_cache

if not hasattr(sam3_cache, "inference_lock"):
    sam3_cache.inference_lock = RLock()

SAM3_REPO_ID = "facebook/sam3"
SAM3_CHECKPOINT_NAME = "sam3.pt"


def parse_text_prompts(text_prompt):
    """Return unique SAM 3 concepts from lines, commas, or semicolons."""
    prompts = []
    seen = set()
    for value in re.split(r"[\n,;]+", text_prompt or ""):
        prompt = value.strip()
        key = prompt.casefold()
        if prompt and key not in seen:
            prompts.append(prompt)
            seen.add(key)
    return prompts


def _download_sam3_checkpoint(model_dir):
    """Download the gated official SAM 3 checkpoint into the project model folder."""
    from huggingface_hub import hf_hub_download

    os.makedirs(model_dir, exist_ok=True)
    try:
        return hf_hub_download(
            repo_id=SAM3_REPO_ID,
            filename=SAM3_CHECKPOINT_NAME,
            local_dir=model_dir,
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(token in message for token in ("gated", "401", "403", "access")):
            raise RuntimeError(
                "SAM 3 checkpoint access is required. Accept Meta's license at "
                "https://huggingface.co/facebook/sam3, then run `hf auth login` "
                "inside this project's venv and try Segment again."
            ) from exc
        raise RuntimeError(f"Unable to download the SAM 3 checkpoint: {exc}") from exc


def get_sam3_processor(confidence_threshold, model_dir, device="cuda"):
    """Load and cache the official SAM 3 image model and processor."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SAM 3 requires a CUDA-capable GPU, but CUDA is unavailable."
        )

    torch_version = tuple(
        int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
    )
    if torch_version < (2, 7):
        raise RuntimeError(
            f"SAM 3 requires PyTorch 2.7 or newer; found {torch.__version__}. "
            "Run the updated installer before using Segment."
        )

    if sam3_cache.processor is None:
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "The SAM 3 package is not installed. Run the updated installer or "
                "install `git+https://github.com/facebookresearch/sam3.git`."
            ) from exc

        sam3_cache.checkpoint_path = _download_sam3_checkpoint(model_dir)
        sam3_cache.model = build_sam3_image_model(
            checkpoint_path=sam3_cache.checkpoint_path,
            load_from_HF=False,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
        )
        sam3_cache.processor = Sam3Processor(
            sam3_cache.model,
            device=device,
            confidence_threshold=float(confidence_threshold),
        )
    else:
        sam3_cache.processor.set_confidence_threshold(float(confidence_threshold))

    return sam3_cache.processor


def get_masks(
    image,
    processor,
    text_prompt,
    mask_threshold=0.5,
    min_mask_area=100,
    max_masks=50,
):
    """Run one or more SAM 3 concepts and return combined LayerDivider masks."""
    prompts = parse_text_prompts(text_prompt)
    if not prompts:
        raise ValueError(
            "Enter at least one short SAM 3 prompt. Put separate concepts on "
            "separate lines, such as `hair`, `face`, and `shoe`."
        )

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")

    device_type = str(processor.device).split(":", 1)[0]
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast_context = torch.autocast("cuda", dtype=dtype)
    else:
        autocast_context = nullcontext()

    masks = []
    with torch.inference_mode(), autocast_context:
        state = processor.set_image(image)
        for prompt in prompts:
            state = processor.set_text_prompt(prompt=prompt, state=state)
            scores = state["scores"].detach().float().cpu().numpy()
            probabilities = state["masks_logits"][:, 0].detach().float().cpu().numpy()
            order = np.argsort(scores)[::-1]

            for index in order[: int(max_masks)]:
                segmentation = probabilities[index] >= float(mask_threshold)
                area = int(segmentation.sum())
                if area < int(min_mask_area):
                    continue
                masks.append(
                    {
                        "segmentation": segmentation,
                        "area": area,
                        "score": float(scores[index]),
                        "prompt": prompt,
                    }
                )

    masks.sort(key=lambda item: item["score"], reverse=True)
    masks = masks[: int(max_masks)]

    if not masks:
        raise ValueError(
            "SAM 3 found no usable masks. Try a more specific noun phrase, lower the "
            "confidence or mask threshold, or reduce the minimum mask area."
        )

    return masks


def layer_filename(index, prompt):
    safe_prompt = re.sub(r"[^A-Za-z0-9_-]+", "_", prompt).strip("_") or "mask"
    return f"{index:03d}_{safe_prompt}.png"


def mask_to_image(image, mask):
    """Render one full-size transparent layer without changing its source."""
    if image is None:
        raise ValueError("Upload an image before creating PNG layers.")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    source = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    segmentation = np.asarray(mask["segmentation"], dtype=bool)
    if segmentation.shape != source.shape[:2]:
        raise ValueError("The saved mask does not match the original image.")
    layer = np.zeros_like(source)
    layer[segmentation] = source[segmentation]
    return Image.fromarray(layer, mode="RGBA")


def save_png(image, path):
    """Replace an image atomically so clients never receive a partial PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_mask_layers(image, masks, output_dir, filenames=None):
    """Write layers directly to an existing job folder, keeping stable filenames."""
    if not masks:
        raise ValueError("Run SAM 3 segmentation before creating PNG layers.")
    filenames = filenames or [
        layer_filename(index, mask.get("prompt", "mask"))
        for index, mask in enumerate(masks, start=1)
    ]
    if len(filenames) != len(masks):
        raise ValueError("Every mask must have a layer filename.")
    paths = []
    for mask, filename in zip(masks, filenames):
        path = Path(output_dir) / filename
        save_png(mask_to_image(image, mask), path)
        paths.append(str(path))
    return paths


def show_anns(image, masks):
    """Render a mask overlay without mutating the job's saved masks."""
    if len(masks) == 0:
        return image.convert("RGBA")

    sorted_masks = sorted(masks, key=lambda item: item["area"], reverse=True)
    base_mask = image.convert("RGBA")
    for mask in sorted_masks:
        segmentation = mask["segmentation"]
        overlay = np.ones((segmentation.shape[0], segmentation.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for channel in range(3):
            overlay[:, :, channel] = color_mask[channel]
        overlay = np.dstack((overlay * 255, segmentation * 255 * 0.35))
        base_mask = Image.alpha_composite(
            base_mask, Image.fromarray(overlay.astype(np.uint8))
        )

    return base_mask


def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)
