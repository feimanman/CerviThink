"""Image operations for evidence regions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter


BBox = tuple[int, int, int, int]
DEFAULT_TRANSFORM_SCALE_RANGE = (1.0, 3.0)
DEFAULT_TRANSFORM_CONTRAST_RANGE = (0.8, 1.5)
DEFAULT_INPAINT_RADIUS = 5
OPENCV_INPAINT_METHOD = "navier-stokes"


@dataclass(frozen=True)
class VisualVariants:
    focus: Image.Image
    transform: Image.Image
    ignore: Image.Image
    bbox: BBox
    zoom: float = 1.0
    contrast: float = 1.0


def load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def clamp_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    min_size: int = 1,
) -> BBox:
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))

    if x2 - x1 < min_size:
        pad = min_size - (x2 - x1)
        x1 = max(0, x1 - pad // 2)
        x2 = min(width, x2 + pad - pad // 2)
    if y2 - y1 < min_size:
        pad = min_size - (y2 - y1)
        y1 = max(0, y1 - pad // 2)
        y2 = min(height, y2 + pad - pad // 2)

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def scale_bbox(bbox: BBox, width: int, height: int, scale: float) -> BBox:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, (x2 - x1) * scale)
    bh = max(1.0, (y2 - y1) * scale)
    return clamp_bbox((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), width, height)


def focus_operation(
    image: str | Path | Image.Image,
    bbox: tuple[float, float, float, float],
    min_size: int = 28,
    context_scale: float = 1.0,
) -> Image.Image:
    img = load_image(image)
    box = clamp_bbox(bbox, img.width, img.height, min_size=min_size)
    if context_scale != 1.0:
        box = scale_bbox(box, img.width, img.height, context_scale)
    return img.crop(box)


def transform_operation(
    image: str | Path | Image.Image,
    bbox: Optional[tuple[float, float, float, float]] = None,
    zoom: float = 2.0,
    contrast: float = 1.25,
    sharpness: float = 1.0,
    output_size: Optional[tuple[int, int]] = None,
) -> Image.Image:
    """Zoom and enhance a crop to mimic microscope magnification."""

    img = load_image(image)
    if bbox is not None:
        img = focus_operation(img, bbox, context_scale=1.0)

    if output_size is None:
        output_size = (max(28, int(img.width * zoom)), max(28, int(img.height * zoom)))
    img = img.resize(output_size, Image.Resampling.BICUBIC)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def sample_transform_params(
    rng: random.Random | None = None,
    scale_range: tuple[float, float] = DEFAULT_TRANSFORM_SCALE_RANGE,
    contrast_range: tuple[float, float] = DEFAULT_TRANSFORM_CONTRAST_RANGE,
) -> tuple[float, float]:
    sampler = rng if rng is not None else random
    return sampler.uniform(*scale_range), sampler.uniform(*contrast_range)


def _pil_ignore(image: Image.Image, bbox: BBox, blur_radius: float) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = image.copy()
    patch = blurred.crop(bbox)
    out.paste(patch, bbox)
    return out


def _opencv_inpaint(image: Image.Image, bbox: BBox, radius: int) -> Optional[Image.Image]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    arr = np.array(image)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255
    inpainted = cv2.inpaint(arr, mask, radius, cv2.INPAINT_NS)
    return Image.fromarray(inpainted).convert("RGB")


def ignore_operation(
    image: str | Path | Image.Image,
    bbox: tuple[float, float, float, float],
    dilation: float = 1.08,
    inpaint_radius: int = DEFAULT_INPAINT_RADIUS,
    blur_radius: float = 12.0,
    prefer_opencv: bool = True,
    require_opencv: bool = False,
) -> Image.Image:
    """Mask suspected abnormal cells and reconstruct the area."""

    img = load_image(image)
    box = clamp_bbox(bbox, img.width, img.height, min_size=2)
    box = scale_bbox(box, img.width, img.height, dilation)
    if prefer_opencv:
        inpainted = _opencv_inpaint(img, box, radius=inpaint_radius)
        if inpainted is not None:
            return inpainted
    if require_opencv:
        raise RuntimeError("Paper workflow requires OpenCV NS inpainting; blur fallback is disabled")
    return _pil_ignore(img, box, blur_radius=blur_radius)


def make_visual_variants(
    image: str | Path | Image.Image,
    bbox: tuple[float, float, float, float],
    focus_context_scale: float = 1.15,
    transform_zoom: float | None = None,
    transform_contrast: float | None = None,
    rng: random.Random | None = None,
    require_opencv: bool = True,
) -> VisualVariants:
    img = load_image(image)
    box = clamp_bbox(bbox, img.width, img.height, min_size=28)
    if transform_zoom is None or transform_contrast is None:
        sampled_zoom, sampled_contrast = sample_transform_params(rng)
        if transform_zoom is None:
            transform_zoom = sampled_zoom
        if transform_contrast is None:
            transform_contrast = sampled_contrast
    focus = focus_operation(img, box, context_scale=focus_context_scale)
    transformed = transform_operation(focus, zoom=transform_zoom, contrast=transform_contrast)
    ignored = ignore_operation(img, box, require_opencv=require_opencv)
    return VisualVariants(focus=focus, transform=transformed, ignore=ignored, bbox=box,
                          zoom=transform_zoom, contrast=transform_contrast)


def save_variants(variants: VisualVariants, output_dir: str | Path, prefix: str = "sample") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "focus": out / f"{prefix}_focus.png",
        "transform": out / f"{prefix}_transform.png",
        "ignore": out / f"{prefix}_ignore.png",
    }
    variants.focus.save(paths["focus"])
    variants.transform.save(paths["transform"])
    variants.ignore.save(paths["ignore"])
    return {key: str(value) for key, value in paths.items()}
