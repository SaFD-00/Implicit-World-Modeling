#!/usr/bin/env python3
"""
Extract AndroidControl screenshots from GCS TFRecord files.
No tensorflow required — uses only Python standard library.

Usage:
    python scripts/extract_androidcontrol_images.py \
        --output data/AndroidControl/images/ \
        --skip-existing --verbose

    # Test with a few episodes first:
    python scripts/extract_androidcontrol_images.py \
        --output data/AndroidControl/images/ \
        --max-episodes 3 --verbose
"""

import argparse
import gzip
import io
import json
import os
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request

from PIL import Image

GCS_BUCKET = "gresearch"
GCS_PREFIX = "android_control/android_control"
GCS_API = "https://storage.googleapis.com/storage/v1"
GCS_MEDIA = "https://storage.googleapis.com"

PNG_MAGIC = b"\x89PNG"
JPEG_QUALITY = 95


# ---------------------------------------------------------------------------
# GCS helpers (public bucket, no auth needed)
# ---------------------------------------------------------------------------


def gcs_list_objects(bucket: str, prefix: str) -> list[str]:
    """List object names in a public GCS bucket by prefix."""
    objects = []
    page_token = None
    while True:
        url = f"{GCS_API}/b/{bucket}/o?prefix={prefix}&maxResults=1000"
        if page_token:
            url += f"&pageToken={page_token}"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
        for item in data.get("items", []):
            objects.append(item["name"])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return sorted(objects)


def gcs_download_to_file(bucket: str, obj_name: str, dest_path: str) -> None:
    """Download a GCS object to a local file."""
    safe_name = urllib.request.quote(obj_name, safe="")
    url = f"{GCS_MEDIA}/storage/v1/b/{bucket}/o/{safe_name}?alt=media"
    urllib.request.urlretrieve(url, dest_path)


# ---------------------------------------------------------------------------
# TFRecord reader (GZIP compressed)
# ---------------------------------------------------------------------------


def iter_tfrecord_gzip(path: str):
    """Yield raw record bytes from a GZIP-compressed TFRecord file."""
    with gzip.open(path, "rb") as f:
        while True:
            # uint64 length
            buf = f.read(8)
            if len(buf) < 8:
                break
            length = struct.unpack("<Q", buf)[0]
            # uint32 masked CRC of length
            f.read(4)
            # byte[length] data
            data = f.read(length)
            if len(data) < length:
                break
            # uint32 masked CRC of data
            f.read(4)
            yield data


# ---------------------------------------------------------------------------
# Minimal protobuf wire format parser for tf.train.Example
# ---------------------------------------------------------------------------


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _read_field(data: bytes, pos: int) -> tuple[int, int, object, int]:
    """Returns (field_number, wire_type, value, new_pos)."""
    tag, pos = _read_varint(data, pos)
    field_number = tag >> 3
    wire_type = tag & 0x07

    if wire_type == 0:  # varint
        value, pos = _read_varint(data, pos)
    elif wire_type == 1:  # 64-bit fixed
        value = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
    elif wire_type == 2:  # length-delimited
        length, pos = _read_varint(data, pos)
        value = data[pos : pos + length]
        pos += length
    elif wire_type == 5:  # 32-bit fixed
        value = struct.unpack_from("<I", data, pos)[0]
        pos += 4
    else:
        raise ValueError(f"Unknown wire type {wire_type}")

    return field_number, wire_type, value, pos


def _parse_int64_list(data: bytes) -> list[int]:
    """Parse Int64List message: repeated int64 value = 1;"""
    result = []
    pos = 0
    while pos < len(data):
        fn, wt, val, pos = _read_field(data, pos)
        if fn == 1:
            if wt == 2:  # packed
                inner_pos = 0
                while inner_pos < len(val):
                    v, inner_pos = _read_varint(val, inner_pos)
                    if v > 0x7FFFFFFFFFFFFFFF:
                        v -= 0x10000000000000000
                    result.append(v)
            elif wt == 0:  # non-packed
                if val > 0x7FFFFFFFFFFFFFFF:
                    val -= 0x10000000000000000
                result.append(val)
    return result


def _parse_bytes_list(data: bytes) -> list[bytes]:
    """Parse BytesList message: repeated bytes value = 1;"""
    result = []
    pos = 0
    while pos < len(data):
        fn, wt, val, pos = _read_field(data, pos)
        if fn == 1 and wt == 2:
            result.append(val)
    return result


def _parse_feature(data: bytes) -> tuple[str, list]:
    """Parse Feature message. Returns ('bytes_list', [...]) or ('int64_list', [...])."""
    pos = 0
    while pos < len(data):
        fn, wt, val, pos = _read_field(data, pos)
        if fn == 1 and wt == 2:
            return "bytes_list", _parse_bytes_list(val)
        elif fn == 2 and wt == 2:
            return "float_list", []
        elif fn == 3 and wt == 2:
            return "int64_list", _parse_int64_list(val)
    return "unknown", []


def _parse_map_entry(data: bytes) -> tuple[str, tuple[str, list]]:
    """Parse a map<string, Feature> entry."""
    key = ""
    feature = ("unknown", [])
    pos = 0
    while pos < len(data):
        fn, wt, val, pos = _read_field(data, pos)
        if fn == 1 and wt == 2:
            key = val.decode("utf-8", errors="replace")
        elif fn == 2 and wt == 2:
            feature = _parse_feature(val)
    return key, feature


def parse_example(data: bytes) -> dict[str, tuple[str, list]]:
    """Parse tf.train.Example → dict of feature_name → (type, values)."""
    features = {}
    # Example: field 1 = Features message
    pos = 0
    while pos < len(data):
        fn, wt, val, pos = _read_field(data, pos)
        if fn == 1 and wt == 2:
            # Features: field 1 = map<string, Feature> entries (repeated)
            fpos = 0
            while fpos < len(val):
                ffn, fwt, fval, fpos = _read_field(val, fpos)
                if ffn == 1 and fwt == 2:
                    name, feat = _parse_map_entry(fval)
                    if name:
                        features[name] = feat
    return features


def get_int64(features: dict, key: str) -> int | None:
    feat = features.get(key)
    if feat and feat[0] == "int64_list" and feat[1]:
        return feat[1][0]
    return None


def get_bytes_list(features: dict, key: str) -> list[bytes]:
    feat = features.get(key)
    if feat and feat[0] == "bytes_list":
        return feat[1]
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Extract AndroidControl screenshots from GCS TFRecord(GZIP) files."
    )
    ap.add_argument(
        "--output",
        default="data/AndroidControl/images/",
        help="Output directory for images (default: data/AndroidControl/images/)",
    )
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Limit extraction to N episodes (0 = unlimited)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist on disk (enables resume)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Verbose per-episode logging"
    )
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Output: {args.output}")
    print(f"Skip existing: {args.skip_existing}")
    if args.max_episodes > 0:
        print(f"Max episodes: {args.max_episodes}")
    print()

    # List TFRecord files from GCS
    print("Listing TFRecord files from GCS...")
    try:
        obj_names = gcs_list_objects(GCS_BUCKET, GCS_PREFIX)
    except urllib.error.URLError as e:
        print(f"ERROR: Failed to list GCS objects: {e}", file=sys.stderr)
        sys.exit(1)

    if not obj_names:
        print("ERROR: No TFRecord files found in GCS bucket", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(obj_names)} TFRecord files\n")

    total_episodes = 0
    total_saved = 0
    total_skipped = 0
    total_errors = 0
    t_start = time.time()
    done = False

    for file_idx, obj_name in enumerate(obj_names):
        if done:
            break

        file_name = os.path.basename(obj_name)
        print(f"[{file_idx + 1}/{len(obj_names)}] Downloading {file_name} ...")

        # Download to temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tfrecord.gz")
        os.close(tmp_fd)

        try:
            gcs_download_to_file(GCS_BUCKET, obj_name, tmp_path)
            tmp_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            print(f"  Downloaded {tmp_size_mb:.1f} MB, parsing...")

            file_saved = 0
            file_skipped = 0

            for record_data in iter_tfrecord_gzip(tmp_path):
                try:
                    features = parse_example(record_data)
                    episode_id = get_int64(features, "episode_id")
                    if episode_id is None:
                        total_errors += 1
                        continue
                except Exception as e:
                    total_errors += 1
                    if args.verbose:
                        print(f"  [WARN] Failed to parse record: {e}")
                    continue

                screenshots = get_bytes_list(features, "screenshots")
                ep_saved = 0
                ep_skipped = 0

                for step_idx, png_bytes in enumerate(screenshots):
                    filename = f"episode_{episode_id:06d}_step_{step_idx:04d}.jpg"
                    filepath = os.path.join(args.output, filename)

                    if args.skip_existing and os.path.exists(filepath):
                        ep_skipped += 1
                        continue

                    if len(png_bytes) < 4 or png_bytes[:4] != PNG_MAGIC:
                        total_errors += 1
                        if args.verbose:
                            print(
                                f"  [WARN] episode {episode_id} step {step_idx}: "
                                f"invalid PNG ({len(png_bytes)} bytes)"
                            )
                        continue

                    with Image.open(io.BytesIO(png_bytes)) as im:
                        if im.mode != "RGB":
                            im = im.convert("RGB")
                        im.save(filepath, "JPEG", quality=JPEG_QUALITY)
                    ep_saved += 1

                file_saved += ep_saved
                file_skipped += ep_skipped
                total_episodes += 1

                if args.verbose:
                    print(
                        f"  episode {episode_id}: "
                        f"{ep_saved} saved, {ep_skipped} skipped, "
                        f"{len(screenshots)} total steps"
                    )

                if args.max_episodes > 0 and total_episodes >= args.max_episodes:
                    done = True
                    break

            total_saved += file_saved
            total_skipped += file_skipped
            elapsed = time.time() - t_start
            print(
                f"  -> {file_saved} saved, {file_skipped} skipped "
                f"(cumulative: {total_saved} images, {total_episodes} episodes, "
                f"{elapsed:.0f}s)\n"
            )
        except urllib.error.URLError as e:
            print(f"  [ERROR] Download failed: {e}")
            total_errors += 1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"Done! {elapsed:.1f}s elapsed")
    print(f"Episodes processed: {total_episodes}")
    print(f"Images saved:       {total_saved}")
    print(f"Images skipped:     {total_skipped}")
    print(f"Errors:             {total_errors}")


if __name__ == "__main__":
    main()
