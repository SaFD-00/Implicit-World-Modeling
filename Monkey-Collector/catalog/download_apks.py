"""Download APKs for apps in catalog/apps.csv from F-Droid and Google Play.

F-Droid uses the public HTTP API (https://f-droid.org/api/v1/packages/<pkg>).
Play Store uses `gplaydl` v2 as a subprocess (Aurora Store token dispenser,
anonymous auth).

Output layout (compatible with MobileGPT-V2's /setup-emulator):

    Monkey-Collector/catalog/apks/{package_id}.apk   # base APK only
    Monkey-Collector/catalog/apks/MISSING.json        # cumulative ledger (source of truth)
    Monkey-Collector/catalog/apks/MISSING.md          # rendered view of the ledger

The ledger is cumulative: a partial run (``--only`` / ``--source``) updates only
the packages inside its own scope and never drops out-of-scope records.

Run:

    python -m catalog.download_apks --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from monkey_collector.pipeline.app_catalog import AppCatalog, AppJob  # noqa: E402

FDROID_INDEX_URL = "https://f-droid.org/repo/index-v2.json"
FDROID_REPO_BASE = "https://f-droid.org/repo"
DEFAULT_ABI = "x86_64"
DEFAULT_PLAYSTORE_ARCH = "arm64"
GPLAYDL_TIMEOUT_SEC = 300
HTTP_CHUNK_BYTES = 65536

MISSING_JSON_NAME = "MISSING.json"
MISSING_MD_NAME = "MISSING.md"
MISSING_SCHEMA_VERSION = 1
MISSING_ENTRY_KEYS = ("source", "reason", "first_seen", "last_seen")
SYSTEM_REASON = "platform built-in, not downloadable"
_MISSING_SECTIONS = ("F-Droid", "PlayStore", "System")


class DownloadError(Exception):
    """Download failed for a recorded reason (logged and added to the ledger)."""


class LedgerError(Exception):
    """The MISSING.json ledger could not be read (never silently reset)."""


@dataclass
class DownloadResult:
    package_id: str
    source: str
    status: str
    reason: str = ""
    path: Path | None = None


# ── F-Droid ────────────────────────────────────────────────────────────────
# F-Droid exposes rich per-version metadata (nativecode, sha256, file path) only
# through the ~47 MB ``index-v2.json`` feed. The small ``/api/v1/packages/<pkg>``
# endpoint returns versionCode+versionName only. We fetch the index once and
# cache it in-process for all F-Droid lookups.


def fetch_fdroid_index(session: requests.Session) -> dict:
    logger.info(f"fetching F-Droid index ({FDROID_INDEX_URL})…")
    resp = session.get(FDROID_INDEX_URL, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    count = len(data.get("packages") or {})
    logger.info(f"F-Droid index loaded: {count} packages")
    return data


def pick_fdroid_build(package_entry: dict, abi: str) -> dict:
    """Select the best F-Droid version entry for the target ABI.

    ``package_entry`` is a node from ``index-v2.json`` → ``packages[<pkg>]``
    shaped as ``{"metadata": {...}, "versions": {<key>: {"manifest": {...}, "file": {...}}}}``.

    A manifest with empty or absent ``nativecode`` is universal and always
    eligible. Otherwise ``nativecode`` must include ``abi``. The highest
    ``versionCode`` wins.

    Returns a dict with keys: ``versionCode``, ``file_name`` (leading slash as
    stored in the index), ``sha256`` (optional), ``size`` (optional).
    """
    versions = (package_entry or {}).get("versions") or {}
    if not versions:
        raise DownloadError("no versions in F-Droid index")

    candidates: list[dict] = []
    seen_archs: set[str] = set()
    for v in versions.values():
        manifest = v.get("manifest") or {}
        file_info = v.get("file") or {}
        native = manifest.get("nativecode") or []
        seen_archs.update(native or ["universal"])
        if native and abi not in native:
            continue
        name = file_info.get("name")
        if not name:
            continue
        candidates.append(
            {
                "versionCode": manifest.get("versionCode", 0),
                "file_name": name,
                "sha256": file_info.get("sha256"),
                "size": file_info.get("size"),
            }
        )

    if not candidates:
        raise DownloadError(f"no {abi} build in F-Droid (available: {sorted(seen_archs)})")

    candidates.sort(key=lambda b: b["versionCode"], reverse=True)
    return candidates[0]


def download_fdroid(
    session: requests.Session,
    package_id: str,
    abi: str,
    dest: Path,
    *,
    index: dict,
) -> DownloadResult:
    try:
        pkgs = index.get("packages") or {}
        entry = pkgs.get(package_id)
        if entry is None:
            raise DownloadError(f"not in F-Droid index: {package_id}")
        build = pick_fdroid_build(entry, abi)
        file_name = build["file_name"]
        expected_sha = build.get("sha256")

        url = f"{FDROID_REPO_BASE}{file_name}" if file_name.startswith("/") else f"{FDROID_REPO_BASE}/{file_name}"
        logger.info(f"[fdroid] {package_id} → {file_name} (vc={build['versionCode']})")
        resp = session.get(url, timeout=180, stream=True)
        resp.raise_for_status()

        tmp = dest.with_suffix(".apk.part")
        hasher = hashlib.sha256() if expected_sha else None
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=HTTP_CHUNK_BYTES):
                fh.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)

        if hasher is not None and hasher.hexdigest().lower() != expected_sha.lower():
            tmp.unlink(missing_ok=True)
            raise DownloadError(
                f"sha256 mismatch: expected {expected_sha[:12]}…, got {hasher.hexdigest()[:12]}…"
            )
        tmp.replace(dest)
        return DownloadResult(package_id, "F-Droid", "downloaded", path=dest)
    except DownloadError as exc:
        return DownloadResult(package_id, "F-Droid", "failed", reason=str(exc))
    except requests.RequestException as exc:
        return DownloadResult(package_id, "F-Droid", "failed", reason=f"http error: {exc}")


# ── PlayStore (gplaydl) ────────────────────────────────────────────────────

def build_gplaydl_command(package_id: str, arch: str, output_dir: Path) -> list[str]:
    return [
        "gplaydl",
        "download",
        package_id,
        "--output",
        str(output_dir),
        "--arch",
        arch,
        "--no-extras",
    ]


def run_gplaydl(package_id: str, arch: str, output_dir: Path) -> None:
    cmd = build_gplaydl_command(package_id, arch, output_dir)
    logger.debug(f"[playstore] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=GPLAYDL_TIMEOUT_SEC, check=False
    )
    if proc.returncode != 0:
        combined = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = " | ".join(combined[-5:]) if combined else "no output"
        raise DownloadError(f"gplaydl rc={proc.returncode}: {tail}")


_SPLIT_PREFIXES = ("split_", "split.", "config.")


def finalize_playstore_output(tmp_dir: Path, dest: Path) -> tuple[Path, bool]:
    """Copy the base APK out of the gplaydl output directory.

    Returns (dest, had_splits). ``had_splits`` flags that split APKs existed
    alongside the base, so the caller can record a warning in MISSING.md.
    """
    apks = sorted(tmp_dir.rglob("*.apk"))
    if not apks:
        raise DownloadError("gplaydl produced no APK files")

    base_candidates = [p for p in apks if not p.name.startswith(_SPLIT_PREFIXES)]
    if not base_candidates:
        base_candidates = apks
    base = max(base_candidates, key=lambda p: p.stat().st_size)
    had_splits = any(p.name.startswith(_SPLIT_PREFIXES) for p in apks)
    shutil.copy2(base, dest)
    return dest, had_splits


def download_playstore(package_id: str, arch: str, dest: Path) -> DownloadResult:
    try:
        with tempfile.TemporaryDirectory(prefix="gplaydl_") as tmp:
            tmp_path = Path(tmp)
            run_gplaydl(package_id, arch, tmp_path)
            dest_path, had_splits = finalize_playstore_output(tmp_path, dest)
        reason = "base-only saved (split APKs present, may need manual fallback)" if had_splits else ""
        return DownloadResult(package_id, "PlayStore", "downloaded", reason=reason, path=dest_path)
    except subprocess.TimeoutExpired:
        return DownloadResult(package_id, "PlayStore", "failed", reason="gplaydl timeout")
    except DownloadError as exc:
        return DownloadResult(package_id, "PlayStore", "failed", reason=str(exc))


# ── Orchestration ──────────────────────────────────────────────────────────

def partition_jobs(
    jobs: list[AppJob],
    sources: set[str],
    only: set[str] | None,
) -> tuple[list[AppJob], list[AppJob], list[AppJob]]:
    """Split jobs into (fdroid, playstore, system_skipped) lists.

    `sources` is a subset of {"fdroid", "playstore"}. `only`, if set, filters
    to those package_ids. Unknown-source jobs are logged and dropped.
    """
    fdroid: list[AppJob] = []
    playstore: list[AppJob] = []
    system: list[AppJob] = []
    for job in jobs:
        if only is not None and job.package_id not in only:
            continue
        src = (job.source or "").strip().lower()
        if src == "f-droid":
            if "fdroid" in sources:
                fdroid.append(job)
        elif src == "playstore":
            if "playstore" in sources:
                playstore.append(job)
        elif src == "system":
            system.append(job)
        else:
            logger.warning(f"{job.package_id}: unknown source '{job.source}'")
    return fdroid, playstore, system


# ── Missing-APK ledger ─────────────────────────────────────────────────────
# MISSING.json is the source of truth; MISSING.md is a rendered view of it.
# F-Droid drops apps from its index without notice, so the local APK cache is
# the only defence and this ledger is the record of what is *not* cached. It
# must therefore survive partial runs: a `--only`/`--source` invocation may
# only touch packages inside its own scope.


def empty_ledger() -> dict:
    return {"schema_version": MISSING_SCHEMA_VERSION, "entries": {}, "last_run": {}}


def load_missing_ledger(path: Path) -> dict:
    """Read the cumulative ledger.

    An absent file is normal (fresh ledger). A corrupt file raises
    ``LedgerError``: silently falling back to an empty ledger would reproduce
    the very clobber this ledger exists to prevent. Validation reaches into
    every entry, not just the top level, because syntactically valid JSON with
    a broken entry would otherwise pass here and only blow up in
    ``update_missing_ledger`` — i.e. *after* the downloads have written files.
    Repair is deliberately manual: no defaults are filled in.
    """
    if not path.exists():
        return empty_ledger()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LedgerError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        raise LedgerError(f"{path}: unexpected shape (no 'entries' object)")
    for package_id, entry in data["entries"].items():
        if not isinstance(package_id, str):
            raise LedgerError(f"{path}: entry key {package_id!r} is not a string")
        if not isinstance(entry, dict):
            raise LedgerError(
                f"{path}: entry '{package_id}' is {type(entry).__name__}, expected an object"
            )
        for key in MISSING_ENTRY_KEYS:
            if key not in entry:
                raise LedgerError(f"{path}: entry '{package_id}' is missing '{key}'")
            if not isinstance(entry[key], str):
                raise LedgerError(
                    f"{path}: entry '{package_id}' field '{key}' is "
                    f"{type(entry[key]).__name__}, expected a string"
                )
    data.setdefault("schema_version", MISSING_SCHEMA_VERSION)
    data.setdefault("last_run", {})
    return data


def update_missing_ledger(
    ledger: dict,
    *,
    results: list[DownloadResult],
    system_jobs: list[AppJob],
    scope_ids: set[str],
    apks_dir: Path,
    now: datetime | None = None,
) -> dict:
    """Merge one run's outcome into the ledger, touching in-scope packages only."""
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    entries: dict[str, dict] = {k: dict(v) for k, v in (ledger.get("entries") or {}).items()}

    def upsert(package_id: str, source: str, reason: str) -> None:
        prev = entries.get(package_id) or {}
        entries[package_id] = {
            "source": source,
            "reason": reason,
            "first_seen": prev.get("first_seen") or stamp,
            "last_seen": stamp,
        }

    for r in results:
        if r.status == "failed" and r.package_id in scope_ids:
            upsert(r.package_id, r.source, r.reason)

    system_ids = {j.package_id for j in system_jobs}
    for job in system_jobs:
        upsert(job.package_id, "System", SYSTEM_REASON)

    # Resolution. The ledger means "not held locally", not "last attempt
    # failed" — any in-scope package whose APK exists at run end is dropped,
    # which covers fresh downloads and already-present skips alike. System
    # entries are never resolvable.
    for package_id in sorted(scope_ids - system_ids):
        if package_id in entries and (apks_dir / f"{package_id}.apk").exists():
            del entries[package_id]

    out = dict(ledger)
    out["schema_version"] = MISSING_SCHEMA_VERSION
    out["entries"] = entries
    return out


def render_missing_md(ledger: dict) -> str:
    """Render the whole ledger; sections are omitted when empty."""
    entries = ledger.get("entries") or {}
    run = ledger.get("last_run") or {}
    scope = run.get("scope") or {}
    only = list(scope.get("only") or [])

    by_source: dict[str, list[tuple[str, dict]]] = {}
    for package_id, entry in entries.items():
        by_source.setdefault(entry.get("source") or "Unknown", []).append((package_id, entry))

    lines: list[str] = ["# Missing APKs", ""]
    lines.append(f"- Generated: {run.get('generated', '')}")
    lines.append(f"- Target ABI (F-Droid filter): `{run.get('abi', '')}`")
    lines.append(f"- PlayStore gplaydl arch: `{run.get('playstore_arch', '')}`")
    lines.append(
        f"- Last run scope: source=`{scope.get('source', 'all')}`, "
        f"only=`{','.join(only) if only else '(none)'}`"
    )
    lines.append(f"- Total targets: {run.get('total_targets', 0)}")
    lines.append(f"- Downloaded: {run.get('downloaded', 0)}")
    lines.append(f"- Skipped (already present): {run.get('skipped', 0)}")
    lines.append(f"- Failed: {run.get('failed', 0)}")
    lines.append(f"- Missing (ledger total): {len(entries)}")
    lines.append("")

    section_names = list(_MISSING_SECTIONS) + sorted(set(by_source) - set(_MISSING_SECTIONS))
    for src in section_names:
        items = sorted(by_source.get(src, []), key=lambda kv: kv[0])
        if not items:
            continue
        lines.append(f"## {src} ({len(items)})")
        for package_id, entry in items:
            first = (entry.get("first_seen") or "")[:10]
            last = (entry.get("last_seen") or "")[:10]
            lines.append(
                f"- `{package_id}` — {entry.get('reason', '')} "
                f"(first seen: {first}, last seen: {last})"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        # A successful replace() already moved it; this only fires on failure,
        # where a stale .tmp would otherwise linger next to the ledger.
        tmp.unlink(missing_ok=True)


def write_missing_outputs(apks_dir: Path, ledger: dict) -> tuple[Path, Path]:
    """Persist the ledger (JSON first, then its rendered view)."""
    json_path = apks_dir / MISSING_JSON_NAME
    md_path = apks_dir / MISSING_MD_NAME
    _atomic_write_text(json_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(md_path, render_missing_md(ledger))
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="download-apks",
        description="Download APKs from F-Droid and Google Play based on apps.csv",
    )
    p.add_argument("--source", choices=["all", "fdroid", "playstore"], default="all")
    p.add_argument(
        "--abi",
        default=DEFAULT_ABI,
        help=f"F-Droid native-lib ABI filter (default: {DEFAULT_ABI})",
    )
    p.add_argument(
        "--playstore-arch",
        choices=["arm64", "armv7"],
        default=DEFAULT_PLAYSTORE_ARCH,
        help="gplaydl device profile (default: arm64; x86_64 AVDs use ARM translation)",
    )
    p.add_argument("--only", default="", help="Comma-separated package_id allowlist")
    p.add_argument("--force", action="store_true", help="Re-download even if apk exists")
    p.add_argument("--dry-run", action="store_true", help="Print targets only, no downloads")
    p.add_argument("--apks-dir", default=None, help="Override apks directory (default: ./catalog/apks)")
    p.add_argument("--csv", default=None, help="Override apps.csv path (default: ./catalog/apps.csv)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    apks_dir = Path(args.apks_dir) if args.apks_dir else _REPO_ROOT / "catalog" / "apks"
    csv_path = Path(args.csv) if args.csv else _REPO_ROOT / "catalog" / "apps.csv"
    apks_dir.mkdir(parents=True, exist_ok=True)

    sources: set[str] = {"fdroid", "playstore"} if args.source == "all" else {args.source}
    only = {p.strip() for p in args.only.split(",") if p.strip()} or None

    catalog = AppCatalog.load(csv_path)
    fdroid_jobs, playstore_jobs, system_jobs = partition_jobs(catalog.filter(), sources, only)
    total = len(fdroid_jobs) + len(playstore_jobs)

    logger.info(
        f"targets: fdroid={len(fdroid_jobs)} playstore={len(playstore_jobs)} "
        f"system-skipped={len(system_jobs)} abi={args.abi} ps-arch={args.playstore_arch}"
    )

    if args.dry_run:
        for job in fdroid_jobs + playstore_jobs:
            print(f"{job.source}\t{job.package_id}")
        return 0

    # Load before downloading: a corrupt ledger must abort the run before any
    # file is written, rather than reset the record of what is missing.
    try:
        ledger = load_missing_ledger(apks_dir / MISSING_JSON_NAME)
    except LedgerError as exc:
        logger.error(f"refusing to run: {exc}")
        logger.error("fix or delete the ledger by hand; no files were written")
        return 2

    def needs_download(job: AppJob) -> bool:
        dest = apks_dir / f"{job.package_id}.apk"
        if dest.exists() and not args.force:
            logger.info(f"[skip] {job.package_id} already present")
            return False
        return True

    fdroid_pending = [j for j in fdroid_jobs if needs_download(j)]
    playstore_pending = [j for j in playstore_jobs if needs_download(j)]
    skipped = total - len(fdroid_pending) - len(playstore_pending)

    results: list[DownloadResult] = []

    if fdroid_pending:
        with requests.Session() as session:
            try:
                index = fetch_fdroid_index(session)
            except requests.RequestException as exc:
                logger.error(f"failed to fetch F-Droid index: {exc}")
                for job in fdroid_pending:
                    results.append(
                        DownloadResult(
                            job.package_id, "F-Droid", "failed", reason=f"index fetch failed: {exc}"
                        )
                    )
                index = None
            if index is not None:
                for job in fdroid_pending:
                    r = download_fdroid(
                        session,
                        job.package_id,
                        args.abi,
                        apks_dir / f"{job.package_id}.apk",
                        index=index,
                    )
                    results.append(r)
                    marker = "[+]" if r.status == "downloaded" else "[-]"
                    suffix = f" — {r.reason}" if r.reason else ""
                    print(f"{marker} F-Droid   {r.package_id}{suffix}")

    for job in playstore_pending:
        dest = apks_dir / f"{job.package_id}.apk"
        r = download_playstore(job.package_id, args.playstore_arch, dest)
        results.append(r)
        marker = "[+]" if r.status == "downloaded" else "[-]"
        suffix = f" — {r.reason}" if r.reason else ""
        print(f"{marker} PlayStore {r.package_id}{suffix}")

    downloaded = sum(1 for r in results if r.status == "downloaded")
    failed = sum(1 for r in results if r.status == "failed")

    scope_ids = {j.package_id for j in fdroid_jobs + playstore_jobs + system_jobs}
    ledger = update_missing_ledger(
        ledger,
        results=results,
        system_jobs=system_jobs,
        scope_ids=scope_ids,
        apks_dir=apks_dir,
    )
    ledger["last_run"] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "abi": args.abi,
        "playstore_arch": args.playstore_arch,
        "scope": {"source": args.source, "only": sorted(only) if only else []},
        "total_targets": total,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
    missing_json, missing_md = write_missing_outputs(apks_dir, ledger)

    print()
    print("=== download-apks summary ===")
    print(f"targets:     {total}   (fdroid {len(fdroid_jobs)} + playstore {len(playstore_jobs)})")
    print(f"downloaded:  {downloaded}")
    print(f"skipped:     {skipped}   (already present)")
    print(f"failed:      {failed}")
    print(f"system:      {len(system_jobs)}   (not downloadable)")
    print(f"missing:     {len(ledger['entries'])} in ledger ({missing_json})")
    print(f"missing log: {missing_md}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
