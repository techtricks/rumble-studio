r"""Video composition.

One ffmpeg pass per video: an animated gradient background, burned-in captions
timed against the real voiceover, a title card over the opening seconds, and a
light vignette so the flat background does not band on compression.

Captions are burned from files (`textfile=`) rather than inline `text=`.
Inline text needs escaping for `\ : % , ' [ ] ;` and a single unescaped
apostrophe — extremely common in narration — silently kills the whole filter
graph. A caption file per line is not worth worrying about.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..util import clip
from . import ffmpeg as ff
from .voice import Caption

log = get_logger("render.video")

# Vertical Shorts and 16:9 long-form, at YouTube's common delivery resolutions.
SHORTS_SIZE = (1080, 1920)
LONGFORM_SIZE = (1920, 1080)
FPS = 30

# Opening title card duration (seconds) before the first caption takes over.
TITLE_CARD_S = 4.0

PALETTES = {
    "dark navy background, cyan accent": ("0x0B1020", "0x14304F"),
    "dark background, orange accent": ("0x140B0B", "0x4F1F14"),
    "dark background, green accent": ("0x08140B", "0x14401F"),
    "dark background, purple accent": ("0x120B1A", "0x2F144F"),
    "dark background, white accent": ("0x0A0A0A", "0x2A2A2A"),
}


@dataclass
class Style:
    width: int
    height: int
    caption_size: int
    title_size: int
    caption_y: str
    title_y: str
    max_chars_per_line: int


def style_for(kind: str) -> Style:
    if kind == "shorts":
        return Style(
            width=SHORTS_SIZE[0], height=SHORTS_SIZE[1],
            # Big enough to read on a phone held at arm's length.
            caption_size=88, title_size=104,
            # Vertical: caption block sits in the optical centre, clear of the
            # channel name and the Shorts action rail on the right.
            caption_y="(h-text_h)/2+120", title_y="(h-text_h)/2-260",
            max_chars_per_line=16,
        )
    return Style(
        width=LONGFORM_SIZE[0], height=LONGFORM_SIZE[1],
        caption_size=54, title_size=78,
        # Landscape: lower third, above YouTube's progress bar and caption UI.
        caption_y="h-260", title_y="(h-text_h)/2",
        max_chars_per_line=34,
    )


class RenderError(RuntimeError):
    pass


def _wrap(text: str, max_chars: int) -> str:
    """Balance a caption across at most two lines.

    Split on word boundaries, preferring the midpoint so one line does not run
    to 30 characters while the other holds 3.
    """
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    words = clean.split()
    if len(words) < 2:
        return clean
    best, best_delta = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > max_chars or len(b) > max_chars:
            continue
        delta = abs(len(a) - len(b))
        if best_delta is None or delta < best_delta:
            best, best_delta = (a, b), delta
    return "\n".join(best) if best else clean


def _caption_filters(
    captions: Sequence[Caption],
    style: Style,
    font: Path,
    tmpdir: Path,
) -> List[str]:
    """One drawtext per caption, chained, each with a time window."""
    filters: List[str] = []
    for i, cap in enumerate(captions):
        f = tmpdir / f"cap{i:04d}.txt"
        f.write_text(_wrap(cap.text, style.max_chars_per_line), encoding="utf-8")
        enable = f"between(t\\,{cap.start:.3f}\\,{cap.end:.3f})"
        filters.append(
            f"drawtext=fontfile={font}:textfile={f}:"
            f"fontsize={style.caption_size}:fontcolor=white:line_spacing={style.caption_size // 3}:"
            f"box=1:boxcolor=black@0.62:boxborderw={max(10, style.caption_size // 3)}:"
            f"x=(w-text_w)/2:y={style.caption_y}:"
            f"enable={enable}"
        )
    return filters


def _title_filters(title: str, style: Style, font: Path, tmpdir: Path) -> List[str]:
    f = tmpdir / "title.txt"
    f.write_text(_wrap(title, style.max_chars_per_line + 6), encoding="utf-8")
    return [
        f"drawtext=fontfile={font}:textfile={f}:"
        f"fontsize={style.title_size}:fontcolor=white:line_spacing={style.title_size // 3}:"
        f"box=1:boxcolor=0x0B1020@0.75:boxborderw={max(14, style.title_size // 2)}:"
        f"x=(w-text_w)/2:y={style.title_y}:"
        f"enable=between(t\\,0\\,{TITLE_CARD_S - 0.4:.2f})"
    ]


def _pick_palette(plan: Optional[dict]) -> tuple:
    if plan:
        key = str(plan.get("palette", "")).strip().lower()
        for name, colors in PALETTES.items():
            if key and (key in name or name.split(",")[0] in key):
                return colors
    return PALETTES["dark navy background, cyan accent"]


class _Chain:
    """Builds a labelled filtergraph: `[0:v]f1[v1];[v1]f2[out]`.

    `source` may be a bare input label like `[0:v]`, or a whole subgraph that
    already ends in a label. In the second case the caller passes the label
    that subgraph produces: chaining onto the subgraph *text* rather than its
    output label gives `[slides]vignette=...`, which is not valid filter syntax
    and makes ffmpeg reject the entire graph.
    """

    def __init__(self, source: str = "[0:v]", label: Optional[str] = None) -> None:
        self.stages: List[str] = []
        if label is not None:
            self.stages = [source]
            self.label = label
        else:
            self.label = source

    def add(self, filt: str, out: str = "vout") -> "_Chain":
        self.stages.append(f"{self.label}{filt}[{out}]")
        self.label = f"[{out}]"
        return self

    def build(self) -> str:
        return ";".join(self.stages)


def _slideshow_filter(
    images: Sequence[Path],
    style: "Style",
    duration: float,
) -> str:
    """Cinematic camera-work over topic pictures, as one filter chain.

    Not a slideshow: every still becomes a *shot* with its own programmed
    camera move, the way a documentary cuts between moving frames that all
    happen to come from photographs.

    Each photo is pre-scaled to 1.35x the frame with `force_original_aspect_
    ratio=increase` + crop, so the virtual camera always has headroom: pans
    and zooms never reveal an edge. `zoompan` then drives a per-shot move:

      mode 0  push-in, centred          zoom 1.04 -> 1.30
      mode 1  pull-back, centred        zoom 1.30 -> 1.04
      mode 2  lateral drift left->right gentle push 1.08 -> 1.14
      mode 3  lateral drift right->left gentle push 1.08 -> 1.14

    Shot modes cycle, so the same photo reader never sees two identical
    moves back to back. Cuts are long dissolves/fades (up to 1s) rather than
    hard slideshow slides, also on an even split of the narration: beats do
    not line up with shot counts, and a cut mid-sentence on a fixed cadence
    reads as intentional editing instead of a glitch.
    """
    n = len(images)
    seg = duration / n
    fade = min(1.0, seg / 3.0)
    W, H = style.width, style.height
    segf = max(2.0, seg * FPS)  # frames per shot, for per-frame expressions
    # Camera headroom: the still is upscaled past the frame so a pan inside
    # zoompan always lands on real pixels.
    F = 1.35
    sw = int(W * F) // 2 * 2
    sh = int(H * F) // 2 * 2
    parts: List[str] = []

    for i in range(n):
        mode = i % 4
        zmin, zmax = 1.04, 1.30
        # Eased + drifting camera. Zoompan expressions are per output frame,
        # so a smoothstep (p*p*(3-2p)) replaces the linear slide's robotic
        # feel — moves accelerate out of the cut and settle at the end. A
        # tiny sin-wave wobble on zoom and framing, at two incommensurable
        # periods, reads as a handheld operator breathing rather than a
        # computer dial: the difference between "stock Ken Burns" and
        # original documentary footage.
        p = f"on/{segf:.4f}"        # raw normalized progress
        pe = f"(pow({p}\\,2)*(3-2*pow({p}\\,1)))"  # eased progress (0->1)
        wob_z = "0.005*sin(on/9)"
        wob_x = "0.02*sin(on/13)"
        wob_y = "0.015*sin(on/11)"
        if mode == 0:      # push-in, centred
            z = f"{zmin}+{zmax - zmin:.4f}*{pe}+{wob_z}"
            fx, fy = f"0.5+{wob_x}", f"0.5+{wob_y}"
        elif mode == 1:    # pull-back, centred
            z = f"{zmax}-{zmax - zmin:.4f}*{pe}+{wob_z}"
            fx, fy = f"0.5+{wob_x}", f"0.5+{wob_y}"
        elif mode == 2:    # drift left -> right, gentle push-in
            z = f"1.08+0.06*{pe}+{wob_z}"
            fx = f"min(1\\,max(0\\,0.12+0.62*{pe}))+{wob_x}"
            fy = f"0.5+{wob_y}"
        else:              # drift right -> left, gentle push-in
            z = f"1.14-0.06*{pe}+{wob_z}"
            fx = f"min(1\\,max(0\\,0.88-0.62*{pe}))+{wob_x}"
            fy = f"0.5+{wob_y}"
        # Every 6th shot gets a vertical tilt instead of the flat centre-line,
        # so long videos keep some variety in the move vocabulary.
        if i % 6 == 5:
            fy = f"min(1\\,max(0\\,0.72-0.55*{pe}))+{wob_y}"
        offset = i * seg
        parts.append(
            f"[{i}:v]scale={sw}:{sh}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={sw}:{sh},setsar=1,fps={FPS},format=yuv420p,"
            f"zoompan=z='{z}':d=1:"
            f"x='(iw-iw/zoom)*{fx}':y='(ih-ih/zoom)*{fy}':"
            f"s={W}x{H}:fps={FPS},format=yuv420p,"
            f"trim=duration={seg + fade:.3f},setpts=PTS-STARTPTS+{offset:.3f}/TB[v{i}]"
        )

    if n == 1:
        parts.append("[v0]null[slides]")
        return ";".join(parts)

    prev = "v0"
    xfade_of = ("fade", "dissolve")
    for i in range(1, n):
        # xfade blends `duration` seconds starting at `offset`, so the next
        # segment has to begin that much before its nominal slice.
        out = f"x{i}"
        parts.append(
            f"[{prev}][v{i}]xfade=transition={xfade_of[i % 2]}:duration={fade:.3f}:"
            f"offset={i * seg - fade:.3f}[{out}]"
        )
        prev = out
    parts.append(f"[{prev}]null[slides]")
    return ";".join(parts)


def render_video(
    *,
    audio_path: Path,
    captions: Sequence[Caption],
    out_path: Path,
    kind: str,
    title: str = "",
    thumbnail_plan: Optional[dict] = None,
    brand: str = "",
    settings: Optional[Settings] = None,
    duration_s: Optional[float] = None,
    images: Optional[Sequence[Path]] = None,
) -> Path:
    """Render a finished MP4. Returns the output path.

    `images` turns the background into a slideshow timed to the narration. With
    none, this keeps the original animated-gradient behaviour, which is the
    right fallback when no picture could be fetched rather than a reason to
    fail the video.
    """
    s = settings or get_settings()
    ff.require()  # fail fast, before writing temp files
    font = ff.find_font(bold=True)
    if font is None:
        raise RenderError(
            "no usable TTF font found for burned-in captions. ffmpeg's drawtext "
            "cannot use the system .ttc files; install DejaVu or set one in "
            "/System/Library/Fonts/Supplemental/."
        )

    audio_path = Path(audio_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = duration_s or ff.probe_duration(audio_path)
    if not duration or duration <= 1:
        raise RenderError(f"audio duration unusable ({duration}s) for {audio_path.name}")

    style = style_for(kind)
    c0, c1 = _pick_palette(thumbnail_plan)
    usable = [Path(i) for i in (images or []) if Path(i).exists()]
    if usable:
        source = None
    else:
        source = (
            f"gradients=s={style.width}x{style.height}:c0={c0}:c1={c1}:"
            f"speed=0.012:d={duration:.3f}:r={FPS}"
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="ytauto-cap-"))
    try:
        if usable:
            chain = _Chain(_slideshow_filter(usable, style, duration), label="[slides]")
            # Filmic pass over the camera moves: a touch of contrast and
            # saturation, teal-orange cast so the palette stops looking like
            # unprocessed stock, gentle sharpening against the zoompan
            # rescale, live film grain, and 2.35:1 letterboxing on longform —
            # the single strongest "this was shot for cinema" cue. In/out
            # fades close the video like a documentary, not a slide.
            bar = 0 if kind == "shorts" else int((style.height - style.width / 2.35) // 2)
            grade = (
                "eq=contrast=1.06:saturation=1.14:gamma=1.0,"
                "colorbalance=rs=-0.04:bs=0.05,"
                "unsharp=5:5:0.35:5:5:0.0,"
                "noise=alls=5:allf=t,"
                "vignette=PI/4.6,"
            )
            if bar > 40:
                grade += (
                    f"drawbox=x=0:y=0:w={style.width}:h={bar}:color=black:t=fill,"
                    f"drawbox=x=0:y={style.height - bar}:w={style.width}:h={bar}:color=black:t=fill,"
                )
            grade += (
                "fade=t=in:st=0:d=0.8,"
                f"fade=t=out:st={max(0.0, duration - 0.9):.3f}:d=0.9"
            )
            chain.add(grade, "v1")
        else:
            # The gradient is a real input, so the graph starts from [0:v] and
            # every drawtext is chained onto a label.
            chain = _Chain("[0:v]")
            chain.add("vignette=PI/5", "v1")
        for i, filt in enumerate(_title_filters(clip(title, 120), style, font, tmpdir)):
            chain.add(filt, f"vt{i}")
        for i, filt in enumerate(_caption_filters(captions, style, font, tmpdir)):
            chain.add(filt, f"vc{i}")
        if brand:
            bf = tmpdir / "brand.txt"
            bf.write_text(clip(brand, 40), encoding="utf-8")
            y = "h-70" if kind != "shorts" else "h-190"
            x = "60" if kind != "shorts" else "(w-text_w)/2"
            chain.add(
                f"drawtext=fontfile={font}:textfile={bf}:fontsize={style.caption_size // 2}:"
                f"fontcolor=white@0.55:box=1:boxcolor=black@0.35:boxborderw=8:"
                f"x={x}:y={y}:enable=between(t\\,{TITLE_CARD_S:.2f}\\,{duration:.3f})",
                "vbrand",
            )

        log.info(
            "rendering %s %.0fs %dx%d, %d captions, %d image(s)",
            kind, duration, style.width, style.height, len(captions), len(usable),
        )
        cmd = []
        if source is not None:
            cmd += ["-f", "lavfi", "-i", source]
        else:
            for img in usable:
                cmd += ["-loop", "1", "-t", str(duration), "-i", str(img)]
        cmd += [
                "-i", str(audio_path),
                "-filter_complex", chain.build(),
                "-map", chain.label, "-map", f"{len(usable) if usable else 1}:a",
                "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                "-pix_fmt", "yuv420p", "-r", str(FPS),
                # 8-second GOP keeps YouTube's seek bar responsive.
                "-g", str(FPS * 8), "-keyint_min", str(FPS * 8),
                "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
                "-movflags", "+faststart",
                "-shortest",
                str(out_path),
        ]
        # A stuck filter chain would otherwise block the whole cycle for an
        # hour; CI caps this with RENDER_TIMEOUT_S to fit the runner budget.
        ff.run(cmd, timeout=int(os.getenv("RENDER_TIMEOUT_S", "3600")))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not out_path.exists() or out_path.stat().st_size < 10_000:
        raise RenderError(f"ffmpeg produced no usable video at {out_path}")
    size_mb = out_path.stat().st_size / 1e6
    log.info("rendered %s (%.1f MB, %.1fs)", out_path.name, size_mb, duration)
    return out_path


def render_thumbnail(
    out_path: Path,
    *,
    text: str,
    subtitle: str = "",
    kind: str = "longform",
    thumbnail_plan: Optional[dict] = None,
    settings: Optional[Settings] = None,
) -> Path:
    """A burned-in text thumbnail. 1280x720 (or 1080x1920 for Shorts).

    Deliberately typographic: no stock photos to license, nothing scraped, and
    it renders in well under a second.
    """
    ff.require()
    font = ff.find_font(bold=True)
    if font is None:
        raise RenderError("no usable TTF font for the thumbnail")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    style = style_for(kind)
    c0, c1 = _pick_palette(thumbnail_plan)
    width = 1280 if kind != "shorts" else 1080
    height = 720 if kind != "shorts" else 1920

    tmpdir = Path(tempfile.mkdtemp(prefix="ytauto-thumb-"))
    try:
        main = tmpdir / "main.txt"
        main.write_text(_wrap(text, 18 if kind != "shorts" else 12), encoding="utf-8")
        vf = [
            f"gradients=s={width}x{height}:c0={c0}:c1={c1}:speed=0.01:d=3:r=1",
            f"vignette=PI/4",
            f"drawtext=fontfile={font}:textfile={main}:fontsize={116 if kind != 'shorts' else 150}:"
            f"fontcolor=white:line_spacing=24:box=1:boxcolor=0x000000@0.35:boxborderw=24:"
            f"x=(w-text_w)/2:y=(h-text_h)/2",
        ]
        if subtitle:
            sub = tmpdir / "sub.txt"
            sub.write_text(_wrap(subtitle, 40), encoding="utf-8")
            vf.append(
                f"drawtext=fontfile={font}:textfile={sub}:fontsize={44 if kind != 'shorts' else 58}:"
                f"fontcolor=white@0.8:box=1:boxcolor=0x000000@0.4:boxborderw=14:"
                f"x=(w-text_w)/2:y=h-220"
            )
        ff.run(
            ["-f", "lavfi", "-i", vf[0], "-vf", ",".join(vf[1:]),
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            timeout=180,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RenderError(f"thumbnail render produced nothing at {out_path}")
    log.info("thumbnail -> %s (%s)", out_path.name, thumbnail_plan.get("emotion", "") if thumbnail_plan else "")
    return out_path
