"""ffmpeg discovery, vendoring and probing.

ffmpeg is the one hard external dependency of the renderer. It is not
preinstalled on macOS and there is no Homebrew on some machines, so this module
can vendor a static build into `vendor/ffmpeg/` inside the project. That keeps
the tool self-contained and, critically, means nothing is installed system-wide
and no `sudo` is ever needed.

Resolution order: FFMPEG_PATH -> vendored build -> PATH.
"""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..config import ROOT, get_settings
from ..logging_setup import get_logger
from ..util import retry

log = get_logger("render.ffmpeg")

VENDOR_DIR = ROOT / "vendor" / "ffmpeg"
# osxexperts.net publishes self-contained static macOS builds, one per CPU
# architecture. The previous source (evermeet.cx) began returning 410 Gone in
# 2026, and its archive shipped an x86_64 binary that fails with "Bad CPU
# type" on Apple Silicon machines.
ARP_RELEASE = "https://www.osxexperts.net/ffmpeg9arm.zip"  # arm64
INT_RELEASE = "https://www.osxexperts.net/ffmpeg80intel.zip"  # x86_64


def _mac_release() -> str:
    import platform

    return ARP_RELEASE if platform.machine() == "arm64" else INT_RELEASE

DEFAULT_TIMEOUT = 900  # a 10-minute 1080p render is not unusual


class FFmpegMissing(RuntimeError):
    pass


def _mac_version() -> str:
    try:
        with open("/System/Library/CoreServices/SystemVersion.plist", "rb") as fh:
            return str(plistlib.load(fh).get("ProductVersion", ""))
    except Exception:  # noqa: BLE001 - not macOS
        return ""


def _executable(path: Path) -> Optional[Path]:
    return path if path.exists() and os.access(str(path), os.X_OK) else None


def find_ffmpeg() -> Optional[Path]:
    s = get_settings()
    if s.ffmpeg_path:
        p = Path(s.ffmpeg_path).expanduser()
        hit = _executable(p)
        if hit:
            return hit
        log.warning("FFMPEG_PATH=%s is not executable; ignoring", s.ffmpeg_path)
    for name in ("ffmpeg",):
        vendored = _executable(VENDOR_DIR / name)
        if vendored:
            return vendored
        on_path = shutil.which(name)
        if on_path:
            return Path(on_path)
    return None


def find_ffprobe() -> Optional[Path]:
    vendored = _executable(VENDOR_DIR / "ffprobe")
    if vendored:
        return vendored
    on_path = shutil.which("ffprobe")
    return Path(on_path) if on_path else None


def require() -> Path:
    bin_path = find_ffmpeg()
    if bin_path is None:
        raise FFmpegMissing(
            "ffmpeg not found. Run `python -m ytauto setup-ffmpeg` to vendor a "
            "static build into vendor/ffmpeg/, or set FFMPEG_PATH."
        )
    return bin_path


# ------------------------------------------------------------- bootstrap
def install_macos(*, force: bool = False) -> Path:
    """Download + unzip a static ffmpeg into vendor/ffmpeg/.

    The upstream archive contains `ffmpeg` only, not `ffprobe`. That is fine:
    `probe_duration` falls back to parsing ffmpeg's own stderr, so nothing in
    the renderer requires ffprobe.
    """
    import platform

    if platform.system() != "Darwin":
        raise FFmpegMissing(
            f"auto-install is only implemented for macOS (this is {platform.system()}). "
            "Install ffmpeg with your package manager and set FFMPEG_PATH."
        )
    if find_ffmpeg() and not force:
        log.info("ffmpeg already available at %s", find_ffmpeg())
        return require()

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = VENDOR_DIR / "ffmpeg.zip"
    release = _mac_release()
    log.info("downloading static ffmpeg (%s macOS)…", _mac_version() or "unknown")
    log.info("release url: %s", release)

    def _download() -> Path:
        import requests

        with requests.get(release, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return zip_path

    retry(_download, attempts=3, base_delay=3.0, on_error=lambda n, e: log.warning("download attempt %d failed: %s", n, e))

    log.info("extracting %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(VENDOR_DIR)
    zip_path.unlink(missing_ok=True)

    for name in ("ffmpeg", "ffprobe"):
        p = VENDOR_DIR / name
        if p.exists():
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            # Gatekeeper blocks unsigned binaries that carry the quarantine
            # attribute; clear it so the binary can run.
            subprocess.run(
                ["/usr/bin/xattr", "-d", "com.apple.quarantine", str(p)],
                capture_output=True,
            )

    bin_path = find_ffmpeg()
    if bin_path is None:
        raise FFmpegMissing("download finished but no ffmpeg binary appeared in vendor/ffmpeg/")
    version = version_of(bin_path)
    log.info("ffmpeg installed: %s (%s)", bin_path, version)
    return bin_path


def version_of(bin_path: Optional[Path] = None) -> str:
    bin_path = bin_path or find_ffmpeg()
    if bin_path is None:
        return "not installed"
    try:
        out = subprocess.run(
            [str(bin_path), "-version"], capture_output=True, text=True, timeout=30
        ).stdout
        return out.splitlines()[0] if out else "unknown"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unrunnable ({exc})"


# ------------------------------------------------------------------- run
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")
_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.?)")


def run(args: Sequence[str], *, timeout: int = DEFAULT_TIMEOUT) -> subprocess.CompletedProcess:
    """Run ffmpeg with the shared preamble. Raises on non-zero exit."""
    bin_path = require()
    cmd = [str(bin_path), "-hide_banner", "-nostdin", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args[:12]))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stderr or "").strip().splitlines()[-25:]
        raise FFmpegMissing(
            f"ffmpeg timed out after {timeout}s\n  " + "\n  ".join(tail)
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise FFmpegMissing(
            f"ffmpeg exited {proc.returncode}\n  " + "\n  ".join(tail)
        )
    return proc


def probe_duration(path: Path) -> Optional[float]:
    """Duration in seconds. Prefers ffprobe; falls back to parsing stderr."""
    if not Path(path).exists():
        return None
    probe = find_ffprobe()
    if probe:
        try:
            out = subprocess.run(
                [
                    str(probe), "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nw=1:nk=1", str(path),
                ],
                capture_output=True, text=True, timeout=60,
            ).stdout.strip()
            if out:
                return float(out)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    # ffmpeg prints duration on stderr even when it cannot do anything else.
    try:
        proc = subprocess.run(
            [str(require()), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        m = _DURATION_RE.search(proc.stderr or "")
        if m:
            h, mnt, sec = m.groups()
            return int(h) * 3600 + int(mnt) * 60 + float(sec)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def has_filter(name: str) -> bool:
    """True when this ffmpeg build has the filter (drawtext, zoompan, …)."""
    try:
        proc = subprocess.run(
            [str(require()), "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=60,
        )
        return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", proc.stdout or "", re.M) is not None
    except (OSError, subprocess.SubprocessError):
        return False


# ------------------------------------------------------------------ fonts
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def find_font(bold: bool = True) -> Optional[Path]:
    """A TTF ffmpeg's drawtext can use, for burned-in text.

    ffmpeg refuses .ttc on many builds, so .ttf candidates are listed first.
    """
    order = _FONT_CANDIDATES if bold else tuple(reversed(_FONT_CANDIDATES))
    for candidate in order:
        p = Path(candidate)
        if p.exists() and p.suffix.lower() in (".ttf", ".otf"):
            return p
    for candidate in _FONT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None
