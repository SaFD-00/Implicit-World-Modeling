"""Tests for the Stage-0 luminance prefilter primitives (pure-PIL, no numpy)."""

import io

from PIL import Image

from monkey_collector.pipeline.screen_matching.luminance import (
    extract_luminance_features,
    luminance_diff,
)


def _img_bytes(color, size=(40, 80), fmt="JPEG"):
    """Encode a solid-colour RGB image to bytes (format inferred on decode)."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


# ── extract ──

def test_extract_returns_low_res_l_image():
    feat = extract_luminance_features(_img_bytes((10, 20, 30), size=(200, 400)), low_res_width=100)
    assert feat is not None
    assert feat.mode == "L"
    assert feat.size[0] == 100          # width pinned to low_res_width
    assert feat.size[1] == 200          # aspect-preserved: 400 * 100/200


def test_extract_custom_width():
    feat = extract_luminance_features(_img_bytes((0, 0, 0), size=(60, 60)), low_res_width=20)
    assert feat is not None and feat.size == (20, 20)


def test_extract_png_bytes_also_decode():
    # The decode sniffs the format from the buffer, so PNG works too.
    feat = extract_luminance_features(_img_bytes((5, 5, 5), fmt="PNG"))
    assert feat is not None and feat.mode == "L"


def test_extract_garbage_returns_none():
    assert extract_luminance_features(b"not an image") is None


def test_extract_empty_returns_none():
    assert extract_luminance_features(b"") is None


# ── diff ──

def test_identical_images_diff_zero():
    a = Image.new("L", (10, 10), 128)
    b = Image.new("L", (10, 10), 128)
    assert luminance_diff(a, b, threshold=10) == 0.0


def test_black_vs_white_diff_one():
    a = Image.new("L", (10, 10), 0)
    b = Image.new("L", (10, 10), 255)
    assert luminance_diff(a, b, threshold=10) == 1.0


def test_threshold_is_strict_greater_than():
    a = Image.new("L", (8, 8), 100)
    # |Δ| == threshold → NOT counted as changed (strict >, V2 parity).
    assert luminance_diff(a, Image.new("L", (8, 8), 110), threshold=10) == 0.0
    # |Δ| == threshold + 1 → every pixel changed.
    assert luminance_diff(a, Image.new("L", (8, 8), 111), threshold=10) == 1.0


def test_size_mismatch_is_max_diff():
    a = Image.new("L", (10, 10), 100)
    b = Image.new("L", (10, 12), 100)
    assert luminance_diff(a, b, threshold=10) == 1.0


def test_partial_diff_fraction():
    # Half the rows differ beyond threshold → fraction ≈ 0.5.
    a = Image.new("L", (10, 10), 0)
    b = a.copy()
    for y in range(5):
        for x in range(10):
            b.putpixel((x, y), 255)
    assert luminance_diff(a, b, threshold=10) == 0.5
