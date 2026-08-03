"""
Core analysis logic — download a Spotify playlist with spotdl, analyze each
track with librosa, and save a PNG chart per song.

Can also be run directly as a CLI:
    python analyzer.py <spotify_playlist_url> [--songs-dir songs] [--reports-dir static/reports]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import quality

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#e94560"
TEXT      = "#eaeaea"
ENERGY_HI = "#ff6b6b"
ENERGY_LO = "#4ecdc4"
WAVEFORM  = "#a8dadc"
BEAT_LINE = "#457b9d"

LogFn = Callable[[str], None]

# ── download staging ──────────────────────────────────────────────────────────
# spotdl writes into STAGING_DIRNAME; files only reach the library after passing
# quality.verify_download.  Failures land in QUARANTINE_DIRNAME.
STAGING_DIRNAME    = ".staging"
QUARANTINE_DIRNAME = "quarantine"
ARCHIVE_FILENAME   = ".spotdl-archive"

# Three low-bitrate results in a row means the session lost Premium partway
# through, not three unlucky tracks — stop rather than quarantining the rest of
# the playlist one file at a time.
LOW_BITRATE_STREAK_LIMIT = 3

AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".flac", ".wav"}

# Staged filenames carry the Spotify track id so each file can be tied back to
# its metadata; the suffix is stripped when the file is promoted.
_TRACK_ID_RE  = re.compile(r"^(?P<name>.+) \[(?P<track_id>[A-Za-z0-9]{16,})\]$")
_DOWNLOADED_RE = re.compile(r'Downloaded "(?P<name>.+)": (?P<url>\S+)')


def _audio_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _ensure_ffmpeg(log: LogFn):
    """Auto-install ffmpeg via spotdl if it's not already on PATH."""
    if shutil.which("ffmpeg"):
        return
    log("ffmpeg not found — downloading via spotdl (one-time setup) ...")
    subprocess.run(["spotdl", "--download-ffmpeg"], check=True)
    log("ffmpeg installed.")


def _read_analyzed_tracks(reports_dir: Path) -> dict:
    """Load the set of already-analyzed track filenames."""
    analyzed_file = reports_dir / "analyzed_tracks.json"
    if analyzed_file.exists():
        try:
            return json.loads(analyzed_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_analyzed_tracks(reports_dir: Path, analyzed: dict):
    """Save the set of analyzed track filenames."""
    analyzed_file = reports_dir / "analyzed_tracks.json"
    analyzed_file.write_text(json.dumps(analyzed, indent=2), encoding="utf-8")


def _mark_analyzed(reports_dir: Path, filename: str, metadata: dict):
    """Record that a track has been analyzed."""
    analyzed = _read_analyzed_tracks(reports_dir)
    analyzed[filename] = {
        "analyzed_at": datetime.now().isoformat(),
        "tempo": metadata.get("tempo"),
        "key": metadata.get("key"),
        "duration": metadata.get("duration"),
        "name": metadata.get("name"),
    }
    _write_analyzed_tracks(reports_dir, analyzed)


def _fetch_track_metadata(url: str, staging_dir: Path) -> dict[str, dict]:
    """Ask spotdl for the playlist's Spotify metadata before downloading anything.

    Returns {spotify_track_id: {"duration": seconds, "name": str, "url": str}}.
    spotdl's Song.duration is Spotify's duration_ms floored to whole seconds,
    which is comfortably inside the ±4 s tolerance used when verifying.
    """
    save_file = staging_dir / "playlist.spotdl"
    proc = subprocess.run(
        ["spotdl", "save", url, "--save-file", str(save_file)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not save_file.exists():
        detail = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        raise RuntimeError(
            f"Could not read Spotify metadata for {url} (spotdl save exited "
            f"{proc.returncode}). Track durations are required to verify "
            f"downloads.\n{detail}"
        )

    raw = json.loads(save_file.read_text(encoding="utf-8"))
    songs = raw.get("songs", []) if isinstance(raw, dict) else raw

    meta: dict[str, dict] = {}
    for song in songs:
        track_id = song.get("song_id")
        duration = song.get("duration")
        if not track_id or not duration:
            continue
        artist = song.get("artist") or (song.get("artists") or [""])[0]
        meta[track_id] = {
            "duration": float(duration),
            "name": f"{artist} - {song.get('name', '')}".strip(" -"),
            "url": song.get("url", ""),
        }
    return meta


def download_playlist(url: str, download_dir: Path, log: LogFn = print) -> list[Path]:
    """Download all tracks from a Spotify playlist URL using spotdl.

    spotdl downloads into a staging directory; only files that pass
    quality.verify_download are promoted into download_dir.  Failures are moved
    to download_dir/quarantine with a .json sidecar and never returned.
    """
    _ensure_ffmpeg(log)
    download_dir.mkdir(parents=True, exist_ok=True)
    staging_dir    = download_dir / STAGING_DIRNAME
    quarantine_dir = download_dir / QUARANTINE_DIRNAME
    staging_dir.mkdir(parents=True, exist_ok=True)

    log("Reading Spotify track metadata ...")
    track_meta = _fetch_track_metadata(url, staging_dir)
    log(f"Got Spotify durations for {len(track_meta)} track(s).")

    # The archive holds the Spotify URLs of tracks already verified and in the
    # library, so they aren't re-downloaded.  Quarantined tracks stay out of it
    # and are retried on the next run.
    archive = download_dir / ARCHIVE_FILENAME
    verified_urls: set[str] = set()
    if archive.exists():
        verified_urls = {
            line.strip()
            for line in archive.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    log(f"Starting download → {staging_dir}")
    process = subprocess.Popen(
        [
            "spotdl", url,
            # All four of these are required together for 256 kbps AAC.
            # Dropping any one of them silently yields 128 kbps.
            "--audio", "youtube-music",
            "--format", "m4a",
            "--bitrate", "disable",
            "--only-verified-results",
            # Without the cookie file spotdl's yt-dlp is unauthenticated and
            # gets 128 kbps regardless of the premium gate passing.
            "--cookie-file", str(quality.COOKIE_FILE),
            "--archive", str(archive),
            # The [track-id] suffix ties each staged file back to its Spotify
            # metadata; it is stripped on promotion into the library.
            "--output", str(staging_dir / "{artists} - {title} [{track-id}].{output-ext}"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    source_urls: dict[str, str] = {}
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        log(line)
        matched = _DOWNLOADED_RE.search(line)
        if matched:
            source_urls[matched.group("name")] = matched.group("url")
    process.wait()

    # Everything left in staging is unverified, including leftovers from a run
    # that stopped early.
    staged = _audio_files(staging_dir)
    log(f"\nVerifying {len(staged)} downloaded file(s) ...")

    promoted: list[Path] = []
    quarantined: list[tuple[str, str]] = []
    low_bitrate_streak = 0
    stopped_early = False

    for path in staged:
        matched  = _TRACK_ID_RE.match(path.stem)
        track_id = matched.group("track_id") if matched else None
        display  = matched.group("name") if matched else path.stem
        info     = track_meta.get(track_id or "")
        source   = source_urls.get(display) or source_urls.get(path.stem)

        reason, probe = quality.verify_download(path, info["duration"] if info else None)

        if reason is None:
            dest = download_dir / f"{display}{path.suffix}"
            if dest.exists():
                log(f"  ! replacing existing library file {dest.name}")
            os.replace(path, dest)
            promoted.append(dest)
            if info and info.get("url"):
                verified_urls.add(info["url"])
            log(f"  ✓ {display}")
            low_bitrate_streak = 0
            continue

        dest = quality.quarantine_file(path, quarantine_dir, reason, probe, source)
        quarantined.append((display, reason))
        log(f"  ✗ {reason}: {display}  →  {QUARANTINE_DIRNAME}/{dest.name}")

        low_bitrate_streak = low_bitrate_streak + 1 if reason == "LOW_BITRATE" else 0
        if low_bitrate_streak >= LOW_BITRATE_STREAK_LIMIT:
            stopped_early = True
            break

    # Only verified tracks are archived, so quarantined ones get another chance.
    archive.write_text("\n".join(sorted(verified_urls)) + "\n", encoding="utf-8")

    log("\n── Download summary ─────────────────────────────")
    log(f"  Verified into library: {len(promoted)}")
    log(f"  Quarantined:           {len(quarantined)}")
    for name, reason in quarantined:
        log(f"    • {name}  [{reason}]")
    if stopped_early:
        remaining = len(_audio_files(staging_dir))
        log(f"  Left unverified in {STAGING_DIRNAME}/: {remaining}")
        raise quality.QualityGateError(
            f"Stopped after {LOW_BITRATE_STREAK_LIMIT} consecutive LOW_BITRATE "
            "results — the YouTube Music session is probably no longer Premium "
            "(expired cookies, or the wrong Google account).\n"
            f"Refresh cookies at {quality.COOKIE_FILE} and re-run; the "
            f"{remaining} untouched file(s) are still in {staging_dir}."
        )

    return promoted


def analyze_track(path: Path) -> dict:
    """Extract musical features from an audio file."""
    y, sr = librosa.load(str(path), sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Beat tracking & global tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Dynamic tempo (8-second windows)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    dynamic_tempo = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop, aggregate=None
    )
    tempo_times = librosa.frames_to_time(
        np.arange(len(dynamic_tempo)), sr=sr, hop_length=hop
    )

    # RMS energy (loudness)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    rms_norm = rms / (rms.max() + 1e-9)

    # Waveform downsampled for display
    factor = max(1, sr // 200)
    wave_display = y[::factor]
    wave_times = np.linspace(0, duration, len(wave_display))

    # Key detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = key_names[int(np.argmax(chroma.mean(axis=1)))]

    # Structural section boundaries
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    bounds = librosa.segment.agglomerative(mfcc, k=6)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)

    return {
        "name": path.stem,
        "source_file": path.name,
        "duration": duration,
        "tempo": float(np.mean(tempo)) if np.ndim(tempo) > 0 else float(tempo),
        "beat_times": beat_times.tolist(),
        "dynamic_tempo": dynamic_tempo.tolist(),
        "tempo_times": tempo_times.tolist(),
        "rms": rms_norm.tolist(),
        "rms_times": rms_times.tolist(),
        "wave": wave_display.tolist(),
        "wave_times": wave_times.tolist(),
        "key": key,
        "bound_times": bound_times.tolist(),
        "hi_threshold": float(np.percentile(rms_norm, 80)),
        "lo_threshold": float(np.percentile(rms_norm, 30)),
    }


def _fmt_time(seconds: float) -> str:
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m}:{s:02d}"


def generate_chart(data: dict, output_path: Path):
    """Render a three-panel PNG chart for a single track."""
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    gs = GridSpec(3, 1, figure=fig, hspace=0.45, top=0.88, bottom=0.07,
                  left=0.06, right=0.97)

    ax_wave   = fig.add_subplot(gs[0])
    ax_energy = fig.add_subplot(gs[1])
    ax_tempo  = fig.add_subplot(gs[2])

    duration = data["duration"]
    bpm      = data["tempo"]

    fig.text(0.5, 0.95, data["name"], ha="center", fontsize=16,
             fontweight="bold", color=TEXT)
    fig.text(0.5, 0.915,
             f"BPM: {bpm:.1f}   ·   Key: {data['key']}   ·   Duration: {_fmt_time(duration)}",
             ha="center", fontsize=12, color=ACCENT)

    def style_ax(ax, title, ylabel):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT, fontsize=11, pad=6, loc="left")
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
        ax.set_xlim(0, duration)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        spacing = max(15, int(duration / 12 / 15) * 15)
        xticks = np.arange(0, duration, spacing)
        ax.set_xticks(xticks)
        ax.set_xticklabels([_fmt_time(t) for t in xticks], color=TEXT, fontsize=8)

    def section_lines(ax):
        for t in data["bound_times"]:
            if 0 < t < duration:
                ax.axvline(t, color=ACCENT, alpha=0.35, linewidth=1, linestyle="--")

    # Panel 1 — waveform
    style_ax(ax_wave, "Waveform", "Amplitude")
    ax_wave.plot(data["wave_times"], data["wave"], color=WAVEFORM, linewidth=0.4, alpha=0.85)
    ax_wave.set_ylim(-1, 1)
    ax_wave.axhline(0, color="#333355", linewidth=0.5)
    section_lines(ax_wave)

    # Panel 2 — RMS energy
    style_ax(ax_energy, "Energy / Loudness  —  find your drops here", "Normalised RMS")
    rms_arr   = np.array(data["rms"])
    rms_times = np.array(data["rms_times"])
    hi = data["hi_threshold"]
    lo = data["lo_threshold"]

    ax_energy.fill_between(rms_times, rms_arr, where=rms_arr >= hi,
                           color=ENERGY_HI, alpha=0.75)
    ax_energy.fill_between(rms_times, rms_arr, where=rms_arr < lo,
                           color=ENERGY_LO, alpha=0.55)
    ax_energy.fill_between(rms_times, rms_arr,
                           where=(rms_arr >= lo) & (rms_arr < hi),
                           color="#a8c0cc", alpha=0.4)
    ax_energy.plot(rms_times, rms_arr, color=TEXT, linewidth=0.6, alpha=0.6)
    ax_energy.axhline(hi, color=ENERGY_HI, linestyle=":", linewidth=1, alpha=0.8)
    ax_energy.axhline(lo, color=ENERGY_LO, linestyle=":", linewidth=1, alpha=0.8)

    for bt in data["beat_times"]:
        ax_energy.axvline(bt, color=BEAT_LINE, alpha=0.08, linewidth=0.5)

    section_lines(ax_energy)
    ax_energy.set_ylim(0, 1.05)

    ax_energy.legend(
        handles=[
            mpatches.Patch(color=ENERGY_HI, alpha=0.75, label="High energy (top 20%)"),
            mpatches.Patch(color=ENERGY_LO, alpha=0.55, label="Low energy (bottom 30%)"),
        ],
        loc="upper right", fontsize=8, framealpha=0.3,
        labelcolor=TEXT, facecolor=PANEL_BG,
    )

    # Panel 3 — dynamic BPM
    style_ax(ax_tempo, "Tempo over time", "BPM")
    dt = np.array(data["dynamic_tempo"])
    tt = np.array(data["tempo_times"])
    median_bpm = np.median(dt)
    dt = np.clip(dt, median_bpm * 0.7, median_bpm * 1.3)
    ax_tempo.plot(tt, dt, color=ACCENT, linewidth=1.2, alpha=0.9)
    ax_tempo.fill_between(tt, dt, alpha=0.2, color=ACCENT)
    ax_tempo.axhline(bpm, color=TEXT, linestyle="--", linewidth=0.8, alpha=0.5,
                     label=f"Avg {bpm:.1f} BPM")
    section_lines(ax_tempo)
    ax_tempo.legend(loc="upper right", fontsize=8, framealpha=0.3,
                    labelcolor=TEXT, facecolor=PANEL_BG)
    ax_tempo.set_xlabel("Time", color=TEXT, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Finalise layout so axis transforms are accurate, then measure where the
    # actual data area sits inside the image.  We save these fractions so the
    # browser player can draw a pixel-perfect playhead cursor.
    fig.canvas.draw()
    fig_w = fig.get_figwidth()  * fig.dpi
    fig_h = fig.get_figheight() * fig.dpi

    # x: where t=0 and t=duration land in the top axes
    pt0  = ax_wave.transData.transform((0,                data["duration"] * 0))
    ptD  = ax_wave.transData.transform((data["duration"], 0))
    cl   = float(pt0[0] / fig_w)
    cr   = float(ptD[0] / fig_w)

    # y: top of first panel and bottom of last panel (display coords are from
    # the figure bottom; convert to image fraction which counts from the top)
    top_disp = ax_wave.transAxes.transform((0, 1))
    bot_disp = ax_tempo.transAxes.transform((0, 0))
    cyt = float(1 - top_disp[1] / fig_h)
    cyb = float(1 - bot_disp[1] / fig_h)

    # Save without tight cropping so figure dimensions stay predictable
    fig.savefig(output_path, dpi=130, facecolor=BG)
    plt.close(fig)

    meta = {
        "name":        data["name"],
        "source_file": data.get("source_file", ""),
        "duration":    data["duration"],
        "tempo":       data["tempo"],
        "key":         data["key"],
        "cl":  cl,
        "cr":  cr,
        "cyt": cyt,
        "cyb": cyb,
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name)


def compute_chart_bounds() -> dict:
    """Return pixel-accurate axis-spine positions for the standard chart layout.

    Creates a minimal figure with the same GridSpec/styling as generate_chart
    (no audio data needed), renders it, and reads the real axis positions.
    Used as a server-side fallback for charts whose JSON predates this field.
    """
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.45, top=0.88, bottom=0.07,
                   left=0.06, right=0.97)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    for ax, lbl in [(ax0, "Amplitude"), (ax1, "Normalised RMS"), (ax2, "BPM")]:
        ax.set_facecolor(PANEL_BG)
        ax.set_ylabel(lbl, color=TEXT, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.set_xlim(0, 240)

    ax0.set_ylim(-1, 1)
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(80, 180)   # typical BPM range — drives tick-label width

    fig.canvas.draw()

    W = fig.get_figwidth()  * fig.dpi
    H = fig.get_figheight() * fig.dpi

    pt0 = ax0.transData.transform((0,   0))
    ptD = ax0.transData.transform((240, 0))
    top = ax0.transAxes.transform((0, 1))
    bot = ax2.transAxes.transform((0, 0))

    bounds = dict(
        cl  = float(pt0[0] / W),
        cr  = float(ptD[0] / W),
        cyt = float(1 - top[1] / H),
        cyb = float(1 - bot[1] / H),
    )
    plt.close(fig)
    return bounds


def run_analysis(url: str, songs_dir: Path, reports_dir: Path, log: LogFn = print):
    """Full pipeline: download playlist, analyze each track, write PNG charts."""
    # Raises QualityGateError before anything is fetched if 256 kbps AAC cannot
    # be confirmed.  There is no bypass.
    quality.ensure_premium_access(log)

    audio_files = download_playlist(url, songs_dir, log)

    if not audio_files:
        log("No audio files found after download.")
        return

    analyzed = _read_analyzed_tracks(reports_dir)
    new_tracks = [f for f in audio_files if f.name not in analyzed]
    skip_count = len(audio_files) - len(new_tracks)

    if skip_count > 0:
        log(f"Skipping {skip_count} already-analyzed track(s).")

    if not new_tracks:
        log("All tracks already analyzed.")
        return

    log(f"\nAnalyzing {len(new_tracks)} new track(s) ...")
    for path in new_tracks:
        log(f"  Analyzing: {path.name}")
        try:
            data = analyze_track(path)
            out = reports_dir / f"{_safe_name(data['name'])}.png"
            generate_chart(data, out)
            _mark_analyzed(reports_dir, path.name, data)
            log(f"  ✓ {data['name']}  ({data['tempo']:.1f} BPM, key {data['key']})")
        except Exception as exc:
            log(f"  ✗ {path.name}: {exc}")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotimine CLI analyzer")
    parser.add_argument("url", nargs="?", help="Spotify playlist or track URL")
    parser.add_argument("--songs-dir",   default="songs",          help="Where to save downloads")
    parser.add_argument("--reports-dir", default="static/reports", help="Where to save PNG charts")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip spotdl; analyze existing files in --songs-dir")
    parser.add_argument("--check-premium", action="store_true",
                        help="Only run the YouTube Music Premium gate, then exit")
    args = parser.parse_args()

    if args.check_premium:
        try:
            quality.ensure_premium_access()
        except quality.QualityGateError as exc:
            print(f"\nPremium gate FAILED:\n{exc}", file=sys.stderr)
            sys.exit(1)
        print("\nPremium gate passed — 256 kbps AAC is available.")
        sys.exit(0)

    if not args.url and not args.skip_download:
        parser.error("a Spotify URL is required unless --skip-download is given")

    songs_dir   = Path(args.songs_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_download:
        files = sorted(
            p for p in songs_dir.iterdir()
            if p.suffix.lower() in {".mp3", ".m4a", ".opus", ".flac", ".wav"}
        )
        print(f"Skipping download — {len(files)} file(s) in {songs_dir}")
        for path in files:
            print(f"  Analyzing: {path.name}")
            try:
                data = analyze_track(path)
                out = reports_dir / f"{_safe_name(data['name'])}.png"
                generate_chart(data, out)
                print(f"  ✓ {data['name']}")
            except Exception as exc:
                print(f"  ✗ {path.name}: {exc}")
    else:
        try:
            run_analysis(args.url, songs_dir, reports_dir)
        except quality.QualityGateError as exc:
            print(f"\n{exc}", file=sys.stderr)
            sys.exit(1)

    print(f"\nDone. Charts saved to: {reports_dir.resolve()}")
