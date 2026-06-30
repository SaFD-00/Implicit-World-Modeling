"""Screenshot luminance fingerprint for the Stage-0 identical-page prefilter.

Ports MobileGPT-V2's ``memory_manager`` luminance prefilter (BT.601 luma,
width-100 resize, per-pixel ``|ΔY| > threshold`` fraction) using **Pillow only** —
no numpy. ``PIL.Image.convert("L")`` applies the exact ITU-R BT.601 transform
``L = R*299/1000 + G*587/1000 + B*114/1000``, matching MobileGPT-V2's
``0.299/0.587/0.114`` weights. The luma conversion is done *after* the resize so
it operates on the down-scaled RGB, mirroring the reference implementation.

The pixel-difference count is computed with ``ImageChops.difference`` + the L-mode
histogram: ``sum(hist[threshold+1:])`` reproduces ``np.sum(|a-b| > threshold)``
exactly (strict ``>``), so a per-pixel delta equal to the threshold does NOT
count as changed.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image, ImageChops

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def extract_luminance_features(
    screenshot: bytes, low_res_width: int = 100
) -> PILImage | None:
    """Decode screenshot bytes → low-res BT.601 luminance fingerprint (L-mode image).

    Returns ``None`` on any failure (undecodable bytes, zero-width image) so the
    caller can degrade gracefully — the matcher must never break on a bad frame.
    The input is whatever the device sent (JPEG in practice); ``Image.open``
    sniffs the format from the buffer, so the on-disk ``.png`` naming quirk is
    irrelevant.
    """
    try:
        img = Image.open(io.BytesIO(screenshot)).convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return None
        target_h = max(1, int(h * low_res_width / w))
        img = img.resize((low_res_width, target_h), Image.Resampling.LANCZOS)
        return img.convert("L")  # BT.601 luma AFTER resize (V2 parity)
    except Exception as e:  # noqa: BLE001 — never break the matcher on a bad frame
        logger.warning(f"luminance feature extraction failed: {e}")
        return None


def luminance_diff(a: PILImage, b: PILImage, threshold: int) -> float:
    """Fraction of pixels whose ``|ΔY|`` exceeds *threshold* (0–255), in [0.0, 1.0].

    Size mismatch (e.g. orientation change) returns ``1.0`` (definitely different),
    matching MobileGPT-V2's bitmap-size-mismatch behaviour.
    """
    if a.size != b.size:
        return 1.0
    # L-mode difference → 256-bin histogram; bins above the threshold are the
    # "changed" pixels (strict >, so bin == threshold is NOT changed).
    hist = ImageChops.difference(a, b).histogram()
    differing = sum(hist[threshold + 1:])
    total = a.size[0] * a.size[1]
    return differing / total if total else 1.0
