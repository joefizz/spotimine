"""
Soulseek as a second download source, for tracks YouTube Music cannot supply at
256 kbps.

The gap this fills: remixes and edits often exist on YouTube Music only as
*video* entries rather than catalog songs, and those are served at 128 kbps. No
amount of tuning the YouTube Music path helps, because the 256 kbps stream does
not exist for them. Soulseek frequently has the same track as FLAC or 320 kbps.

Drives ``sockseek`` (formerly ``sldl`` / ``slsk-batchdl``) as a subprocess and
reads its NDJSON progress stream, which reports per track what was downloaded and
from whom. Nothing here decides whether a file is good enough — every file still
goes through quality.verify_download under the "soulseek" profile.

Deliberately fails soft: if Soulseek is unconfigured or unavailable the pass is
skipped with an explanation, because losing the YouTube Music results already in
hand would be a far worse outcome.
"""

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import quality

LogFn = Callable[[str], None]


# ── settings ──────────────────────────────────────────────────────────────────

BINARY = os.environ.get("SPOTIMINE_SOCKSEEK", "sockseek")

# Credentials live in a config file rather than on the command line: argv is
# visible in the process table to every user on the box.
CONFIG_FILE = quality.DATA_DIR / "sockseek.conf"
USERNAME = os.environ.get("SPOTIMINE_SLSK_USERNAME", "")
PASSWORD = os.environ.get("SPOTIMINE_SLSK_PASSWORD", "")

# Incoming connections. Without an open inbound port you can only reach peers who
# have one open themselves, which shrinks the pool — and the pool is the point.
LISTEN_PORT = os.environ.get("SPOTIMINE_SLSK_PORT", "49998")

# Pre-filter only. Soulseek search results advertise attributes rather than
# measuring them, and the stock client broadcasts no bitrate at all, so these are
# advisory: ffprobe remains the authority. Deliberately NOT --strict-conditions,
# which would reject every file with unknown properties and cut out all
# stock-client peers — exactly the peers holding the obscure tracks.
FORMATS          = "mp3,flac"
MIN_BITRATE_KBPS = "256"

# Spotify durations differ from CD rips by more than sockseek's default 3 s often
# enough to lose good matches. Our own ±4 s ffprobe check is the real gate.
LENGTH_TOL_S = "10"

# sockseek's rate limit is roughly 34 searches per 220 s; exceeding it earns a
# 30-minute server ban. A second pass over a few tracks is well inside that, but
# the timeout has to be generous because searches wait on peers.
TIMEOUT_S = int(os.environ.get("SPOTIMINE_SLSK_TIMEOUT", "900"))


def is_configured() -> tuple[bool, str]:
    """Whether the Soulseek pass can run. Returns (ok, reason_if_not)."""
    if not shutil.which(BINARY) and not Path(BINARY).is_file():
        return False, (
            f"{BINARY!r} is not on PATH. Soulseek is optional — install sockseek "
            "(https://github.com/fiso64/sockseek) or set SPOTIMINE_SOCKSEEK to "
            "its path to enable the second pass."
        )
    if not (USERNAME and PASSWORD) and not CONFIG_FILE.exists():
        return False, (
            "no Soulseek credentials. Set SPOTIMINE_SLSK_USERNAME and "
            f"SPOTIMINE_SLSK_PASSWORD, or write {CONFIG_FILE}. An account is free "
            "and is created the first time you log in with an unused name."
        )
    return True, ""


def _ensure_config() -> Path:
    """Write the sockseek config from env vars if we have them.

    Never logged and never returned by any endpoint; the file lives under the
    already-gitignored data directory.
    """
    if not (USERNAME and PASSWORD):
        return CONFIG_FILE

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        "# Written by spotimine from SPOTIMINE_SLSK_* environment variables.\n"
        f"username = {USERNAME}\n"
        f"password = {PASSWORD}\n"
        f"listen-port = {LISTEN_PORT}\n",
        encoding="utf-8",
    )
    try:
        os.chmod(CONFIG_FILE, 0o600)   # credentials
    except OSError:
        pass
    return CONFIG_FILE


def write_track_list(songs: list[dict], path: Path) -> int:
    """Write the tracks to fetch as a CSV sockseek can consume.

    We pass an explicit track list rather than the Spotify playlist URL on
    purpose: sockseek's Spotify input needs its own Spotify developer app *and* a
    Premium subscription on that app's owner, and we already resolved this
    metadata for the first pass. This also guarantees the second pass targets
    exactly the tracks the first pass failed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Artist", "Title", "Length"])
        for song in songs:
            artist = song.get("artist") or (song.get("artists") or [""])[0]
            writer.writerow([artist, song.get("name", ""), int(song.get("duration") or 0)])
    return len(songs)


def _argv(track_csv: Path, out_dir: Path, index_path: Path,
          mock_dir: Path | None) -> list[str]:
    argv = [
        BINARY, str(track_csv),
        "--config", str(CONFIG_FILE),
        "-o", str(out_dir),
        # Hard conditions — a cheap pre-filter, not the authority.
        "--format", FORMATS,
        "--min-bitrate", MIN_BITRATE_KBPS,
        "--length-tol", LENGTH_TOL_S,
        "--strict-title", "--strict-artist",
        # Ranking only; never removes candidates.
        "--pref-format", "flac,mp3",
        "--pref-min-bitrate", MIN_BITRATE_KBPS,
        "--name-format", "{artist( - )title|filename}",
        "--index-path", str(index_path),
        # One NDJSON object per event on stdout: this is the integration point.
        "--progress-json", "--no-progress",
    ]
    if mock_dir:
        # Simulate results from a local directory: lets the whole pass, including
        # its failure paths, be tested with no account and no network traffic.
        argv += ["--mock-files-dir", str(mock_dir)]
    return argv


def _iter_events(stream) -> Iterator[dict]:
    """Yield parsed NDJSON events, ignoring anything that isn't valid JSON."""
    for line in stream:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _key(artist: str, title: str) -> tuple[str, str]:
    return (artist or "").strip().casefold(), (title or "").strip().casefold()


def download(
    songs: list[dict],
    staging_dir: Path,
    log: LogFn = print,
    mock_dir: Path | None = None,
) -> dict[tuple[str, str], dict]:
    """Fetch the given tracks from Soulseek into staging_dir.

    Returns {(artist, title): result} keyed case-insensitively, where result
    carries "path", "provenance", "outcome" and "failure". Raises nothing: a
    failure to run yields an empty mapping and a logged explanation.
    """
    ok, reason = is_configured()
    if not ok:
        log(f"  Soulseek pass skipped: {reason}")
        return {}

    _ensure_config()
    staging_dir.mkdir(parents=True, exist_ok=True)
    track_csv  = staging_dir / "wanted.csv"
    index_path = staging_dir / "sockseek-index.csv"
    write_track_list(songs, track_csv)

    argv = _argv(track_csv, staging_dir, index_path, mock_dir)
    log(f"  Searching Soulseek for {len(songs)} track(s) ...")

    results: dict[tuple[str, str], dict] = {}
    try:
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except OSError as exc:
        log(f"  Soulseek pass skipped: could not run {BINARY}: {exc}")
        return {}

    try:
        for event in _iter_events(process.stdout):
            data = event.get("data") or {}
            kind = event.get("type")

            if kind == "search_start":
                log(f"    searching: {data.get('artist')} - {data.get('title')}")
                continue
            if kind != "track_state":
                continue

            outcome = data.get("terminalOutcome")
            failure = data.get("failureReason")
            key = _key(data.get("artist", ""), data.get("title", ""))
            path = data.get("downloadPath")

            if outcome in ("Succeeded", "PartialSuccess") and path:
                user = data.get("username") or "unknown"
                results[key] = {
                    "path": Path(path),
                    # Provenance for the sidecar: who supplied it and as what.
                    "provenance": f"soulseek:{user}:{data.get('filename', '')}",
                    "advertised_bitrate": data.get("bitRate"),
                    "outcome": outcome,
                    "failure": None,
                }
                log(f"    got: {data.get('artist')} - {data.get('title')} "
                    f"from {user}")
            else:
                results[key] = {
                    "path": None, "provenance": None,
                    "advertised_bitrate": None,
                    "outcome": outcome, "failure": failure,
                }
                # These mean different things and deserve different reactions.
                if failure == "NoSearchResults":
                    log(f"    absent from Soulseek: {data.get('artist')} - "
                        f"{data.get('title')}")
                elif failure == "NoMatchingResults":
                    log(f"    on Soulseek but nothing met the quality bar: "
                        f"{data.get('artist')} - {data.get('title')}")
        process.wait(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        log(f"  Soulseek pass timed out after {TIMEOUT_S}s; keeping what arrived.")
    except Exception as exc:
        log(f"  Soulseek pass failed: {exc}")

    return results
