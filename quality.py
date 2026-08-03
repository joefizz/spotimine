"""
Quality enforcement for downloads.

Two halves:

  * ``ensure_premium_access`` — a gate that confirms a live YouTube Music
    Premium session before anything is fetched, with two tiers of cookie
    recovery, and
  * ``verify_download`` / ``quarantine_file`` — a post-download ffprobe pass
    that rejects anything which isn't genuine 256 kbps AAC.

There are deliberately no fallbacks here.  If Premium cannot be confirmed the
run stops; it never degrades to 128 kbps and continues.

Can be run directly to check the gate without downloading anything:
    python quality.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

LogFn = Callable[[str], None]


# ── settings ──────────────────────────────────────────────────────────────────

def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


# Cookies and the interactive-login browser profile live here.
DATA_DIR    = _env_path("SPOTIMINE_DATA_DIR",    Path(".spotimine"))
COOKIE_FILE = _env_path("SPOTIMINE_COOKIE_FILE", DATA_DIR / "cookies.txt")
PROFILE_DIR = DATA_DIR / "ytm-profile"

# Firefox is the supported browser for cookie extraction: it stores cookies in
# plain SQLite.  Chrome 127+ binds its cookie decryption key to the Chrome
# binary, so extraction from Chrome, Edge, Brave and other Chromium browsers
# fails — that is by design upstream, not a bug to work around.
COOKIE_BROWSER    = os.environ.get("SPOTIMINE_COOKIE_BROWSER", "firefox")
CHROMIUM_BROWSERS = {"chrome", "chromium", "edge", "brave", "opera", "vivaldi"}

# Any stable YouTube Music track works — it is only ever used to read a format
# list, never downloaded.
REFERENCE_YTM_TRACK_URL = os.environ.get(
    "SPOTIMINE_REFERENCE_TRACK",
    "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
)

# itag 141 is the 256 kbps AAC stream, served only to Premium sessions.
PREMIUM_ITAG = "141"

SESSION_COOKIE_NAMES = {"SAPISID", "__Secure-3PAPISID"}
YOUTUBE_DOMAIN       = ".youtube.com"

# Interactive login (tier 2) waits this long for the user to sign in.
LOGIN_TIMEOUT_S = 300
LOGIN_POLL_S    = 3

# Verification thresholds.  The bitrate floor is 200000 rather than 256000
# because itag 141 is not strictly CBR; real 128k results land near 128000 and
# fail unambiguously either way.
MIN_BITRATE          = 200_000
VALID_SAMPLE_RATES   = {44100, 48000}
MIN_FILE_BYTES       = 1024 * 1024
DURATION_TOLERANCE_S = 4.0

FFPROBE_ARGS = [
    "-v", "error",
    "-select_streams", "a:0",
    "-show_entries", "stream=codec_name,bit_rate,sample_rate,channels",
    "-show_entries", "format=duration",
    "-of", "json",
]


class QualityGateError(RuntimeError):
    """Premium access could not be confirmed — the run must not continue."""


# ── binaries ──────────────────────────────────────────────────────────────────

def _require_binaries():
    missing = [b for b in ("yt-dlp", "ffmpeg", "ffprobe") if not shutil.which(b)]
    if missing:
        raise QualityGateError(
            f"Required binaries not on PATH: {', '.join(missing)}.\n"
            "Note that `spotdl --download-ffmpeg` installs only ffmpeg, into "
            "~/.spotdl/, and no ffprobe — that is not enough. Install a full "
            "ffmpeg build (which includes ffprobe) and put it on PATH; yt-dlp "
            "comes with spotdl but must also be reachable as `yt-dlp`."
        )


# ── cookie file validation ────────────────────────────────────────────────────

def _iter_netscape_cookies(text: str):
    """Yield (domain, name, value) for each cookie line in a Netscape file."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        yield parts[0], parts[5], parts[6]


def _check_cookie_file(path: Path) -> str | None:
    """Validate the cookie file.

    Returns None if it looks usable, or a reason string describing what is
    wrong.  Reasons starting with "format:" are not recoverable automatically —
    the user placed the wrong kind of file there and we must not overwrite it.
    """
    if not path.exists():
        return f"missing: no cookie file at {path}"
    if path.stat().st_size == 0:
        return f"missing: cookie file {path} is empty"

    text = path.read_text(encoding="utf-8", errors="replace")

    if text.lstrip().startswith(("{", "[")):
        return (
            f"format: {path} is a JSON cookie export, not a Netscape cookie file.\n"
            "Re-export your cookies in Netscape format (most browser cookie "
            "extensions offer 'Netscape' or 'cookies.txt' as an export option) "
            "and save them to that path."
        )

    cookies = list(_iter_netscape_cookies(text))
    if not cookies:
        return (
            f"format: {path} is not in Netscape cookie format — no tab-separated "
            "cookie lines found. Re-export your cookies in Netscape "
            "(cookies.txt) format."
        )

    for domain, name, _value in cookies:
        if name in SESSION_COOKIE_NAMES and domain.endswith(YOUTUBE_DOMAIN):
            return None

    return (
        f"stale: {path} has no {' or '.join(sorted(SESSION_COOKIE_NAMES))} cookie "
        f"for a {YOUTUBE_DOMAIN} domain — the session is not signed in"
    )


# ── the probe that actually matters ───────────────────────────────────────────

def _probe_premium(cookie_file: Path, log: LogFn) -> bool:
    """Return True if itag 141 (256 kbps AAC) is offered for the reference track.

    A cookie file existing proves nothing — expired sessions and the wrong
    Google account both look fine on disk.  This is the only real confirmation.
    """
    log(f"Probing YouTube Music for itag {PREMIUM_ITAG} (256 kbps AAC) ...")
    proc = subprocess.run(
        ["yt-dlp", "--cookies", str(cookie_file), "-F", REFERENCE_YTM_TRACK_URL],
        capture_output=True,
        text=True,
    )
    output = f"{proc.stdout}\n{proc.stderr}"

    if proc.returncode != 0:
        log(f"  yt-dlp exited {proc.returncode} while listing formats:")
        for line in (proc.stderr or "").strip().splitlines()[-5:]:
            log(f"    {line}")
        return False

    for line in proc.stdout.splitlines():
        token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        # "141" and DRC variants such as "141-drc" both indicate the Premium
        # stream is available.
        if token.split("-", 1)[0] == PREMIUM_ITAG:
            log(f"  itag {PREMIUM_ITAG} present — Premium session confirmed.")
            return True

    log(f"  itag {PREMIUM_ITAG} absent from the format list.")
    if "Sign in" in output or "not a bot" in output:
        log("  yt-dlp reports the request was not authenticated.")
    return False


# ── tier 1: pull cookies from a browser profile ───────────────────────────────

def _refresh_from_browser(cookie_file: Path, log: LogFn) -> bool:
    """Extract a fresh cookie file from a local browser profile. No interaction."""
    targets = []
    # Prefer the profile left behind by a previous interactive login, so tier 2
    # is only ever needed once.
    if PROFILE_DIR.exists():
        targets.append(f"{COOKIE_BROWSER}:{PROFILE_DIR}")
    targets.append(COOKIE_BROWSER)

    cookie_file.parent.mkdir(parents=True, exist_ok=True)

    for target in targets:
        log(f"Refreshing cookies from browser profile ({target}) ...")
        proc = subprocess.run(
            [
                "yt-dlp",
                "--cookies-from-browser", target,
                "--cookies", str(cookie_file),
                "--skip-download",
                REFERENCE_YTM_TRACK_URL,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            log("  Cookies extracted.")
            return True

        err = (proc.stderr or "").strip()
        for line in err.splitlines()[-4:]:
            log(f"    {line}")
        if COOKIE_BROWSER.lower() in CHROMIUM_BROWSERS:
            log(
                f"  '{COOKIE_BROWSER}' is a Chromium browser. Chrome 127+ binds "
                "its cookie decryption key to the Chrome binary, so cookie "
                "extraction from Chrome, Edge, Brave and friends cannot work. "
                "Set SPOTIMINE_COOKIE_BROWSER=firefox and sign in to YouTube "
                "Music in Firefox — Firefox stores cookies in plain SQLite."
            )
            return False

    return False


# ── tier 2: interactive login in a real browser ───────────────────────────────

def _is_interactive_session() -> bool:
    """True only when there is a user and a display to show a browser to."""
    if os.environ.get("SPOTIMINE_UNATTENDED", "").strip() not in ("", "0"):
        return False
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    return bool(sys.stdin and sys.stdin.isatty())


def _write_netscape(cookies: list[dict], path: Path) -> int:
    """Write browser-context cookies out as a Netscape cookie file."""
    lines = [
        "# Netscape HTTP Cookie File",
        "# Written by spotimine after an interactive YouTube Music login.",
    ]
    count = 0
    for c in cookies:
        domain = c.get("domain") or ""
        if "youtube.com" not in domain and "google.com" not in domain:
            continue
        expires = c.get("expires")
        expiry = int(expires) if expires and expires > 0 else 0
        lines.append("\t".join([
            domain,
            "TRUE" if domain.startswith(".") else "FALSE",
            c.get("path") or "/",
            "TRUE" if c.get("secure") else "FALSE",
            str(expiry),
            c.get("name") or "",
            c.get("value") or "",
        ]))
        count += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def _has_session_cookie(cookies: list[dict]) -> bool:
    return any(
        c.get("name") in SESSION_COOKIE_NAMES
        and (c.get("domain") or "").endswith(YOUTUBE_DOMAIN)
        for c in cookies
    )


def _interactive_login(cookie_file: Path, log: LogFn) -> bool:
    """Open a visible browser and wait for the user to sign in themselves.

    The app never sees a password: the user types it into a real browser and we
    only read the resulting session.  The browser profile persists in the data
    dir so later refreshes can go through tier 1 against it.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log(
            "  Playwright is not installed, so interactive login is unavailable.\n"
            "  Install it with:  pip install playwright && playwright install firefox"
        )
        return False

    log("Opening a browser for you to sign in to YouTube Music ...")
    log(f"  Waiting up to {LOGIN_TIMEOUT_S // 60} minutes. Sign in, then leave the tab open.")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as pw:
            ctx = pw.firefox.launch_persistent_context(str(PROFILE_DIR), headless=False)
            try:
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto("https://music.youtube.com/", wait_until="domcontentloaded")

                deadline = time.monotonic() + LOGIN_TIMEOUT_S
                signed_in = False
                while time.monotonic() < deadline:
                    if _has_session_cookie(ctx.cookies()):
                        signed_in = True
                        break
                    time.sleep(LOGIN_POLL_S)

                if not signed_in:
                    log(
                        f"  Timed out after {LOGIN_TIMEOUT_S}s without a signed-in "
                        "YouTube Music session."
                    )
                    return False

                written = _write_netscape(ctx.cookies(), cookie_file)
                log(f"  Signed in — exported {written} cookie(s) to {cookie_file}.")
                return True
            finally:
                ctx.close()
    except Exception as exc:
        log(f"  Interactive login failed: {exc}")
        return False


# ── the gate ──────────────────────────────────────────────────────────────────

def _unattended_abort(detail: str) -> QualityGateError:
    return QualityGateError(
        f"{detail}\n"
        "This looks like an unattended run (no display / not a TTY), so the "
        "interactive login was skipped.\n"
        "Refresh cookies manually on a machine with a browser:\n"
        f"  yt-dlp --cookies-from-browser firefox --cookies {COOKIE_FILE} "
        f"--skip-download {REFERENCE_YTM_TRACK_URL}\n"
        f"then copy {COOKIE_FILE} to this machine."
    )


def ensure_premium_access(log: LogFn = print) -> None:
    """Confirm a live YouTube Music Premium session, or raise QualityGateError.

    Recovery is expected, not exceptional — cookies expire every few weeks — so
    a failing probe triggers two tiers of refresh before giving up.  What never
    happens is continuing at a lower bitrate.
    """
    _require_binaries()

    problem = _check_cookie_file(COOKIE_FILE)
    if problem and problem.startswith("format:"):
        # A wrong-format file was placed deliberately; refreshing would silently
        # overwrite it, so stop and say what to fix.
        raise QualityGateError(problem[len("format:"):].strip())

    if problem is None and _probe_premium(COOKIE_FILE, log):
        return

    if problem:
        log(f"Cookie check failed ({problem.split(':', 1)[1].strip()}).")
    log("Premium not confirmed — attempting cookie recovery.")

    # Tier 1 — automatic, no interaction.
    if _refresh_from_browser(COOKIE_FILE, log):
        if _check_cookie_file(COOKIE_FILE) is None and _probe_premium(COOKIE_FILE, log):
            log("Cookies refreshed from the browser profile; continuing.")
            return
        log("Refreshed cookies still do not grant Premium access.")

    # Tier 2 — interactive, only if tier 1 failed.
    if not _is_interactive_session():
        raise _unattended_abort("Could not confirm YouTube Music Premium access.")

    if _interactive_login(COOKIE_FILE, log):
        if _check_cookie_file(COOKIE_FILE) is None and _probe_premium(COOKIE_FILE, log):
            log("Signed in interactively; continuing.")
            return

    raise QualityGateError(
        "Could not confirm YouTube Music Premium access.\n"
        f"itag {PREMIUM_ITAG} (256 kbps AAC) is not offered for "
        f"{REFERENCE_YTM_TRACK_URL} with the cookies at {COOKIE_FILE}, after "
        "both an automatic browser refresh and an interactive login.\n"
        "Check that the Google account you signed in with is the one that has "
        "YouTube Music Premium."
    )


# ── post-download verification ────────────────────────────────────────────────

def probe_audio(path: Path) -> dict:
    """Run ffprobe over a file. Raises RuntimeError if it cannot be read."""
    proc = subprocess.run(
        ["ffprobe", *FFPROBE_ARGS, str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or "").strip() or f"ffprobe exited {proc.returncode}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"unparseable ffprobe output: {exc}") from exc


def verify_download(path: Path, expected_duration_s: float | None) -> tuple[str | None, dict]:
    """Check a downloaded file. Returns (failure_reason, probe_output).

    ``failure_reason`` is None when the file passes.  Anything ambiguous — an
    ffprobe error, a missing field, an unparseable value — counts as a failure;
    nothing ambiguous is ever treated as a pass.
    """
    # Size first, so a half-written file reports TRUNCATED rather than the
    # ffprobe error that truncation causes.
    size = path.stat().st_size if path.exists() else 0
    if size <= MIN_FILE_BYTES:
        return "TRUNCATED", {"file_size": size}

    try:
        probe = probe_audio(path)
    except RuntimeError as exc:
        return "PROBE_FAILED", {"file_size": size, "error": str(exc)}

    probe = {**probe, "file_size": size}
    streams = probe.get("streams") or []
    if not streams:
        return "PROBE_FAILED", {**probe, "error": "no audio stream reported"}
    stream = streams[0]

    if stream.get("codec_name") != "aac":
        return "WRONG_CODEC", probe

    try:
        bit_rate = int(stream["bit_rate"])
    except (KeyError, TypeError, ValueError):
        return "PROBE_FAILED", {**probe, "error": "stream bit_rate missing or unparseable"}
    if bit_rate < MIN_BITRATE:
        return "LOW_BITRATE", probe

    try:
        sample_rate = int(stream["sample_rate"])
    except (KeyError, TypeError, ValueError):
        return "PROBE_FAILED", {**probe, "error": "stream sample_rate missing or unparseable"}
    if sample_rate not in VALID_SAMPLE_RATES:
        return "BAD_SAMPLE_RATE", probe

    try:
        channels = int(stream["channels"])
    except (KeyError, TypeError, ValueError):
        return "PROBE_FAILED", {**probe, "error": "stream channels missing or unparseable"}
    if channels != 2:
        return "NOT_STEREO", probe

    # The duration check is what catches wrong recordings — a remaster, radio
    # edit, live take or sped-up re-upload downloads at full 256k and tags
    # correctly, so every check above passes.
    if expected_duration_s is None:
        return "NO_SPOTIFY_DURATION", {
            **probe,
            "error": "no Spotify duration available for this track",
        }
    try:
        actual = float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return "PROBE_FAILED", {**probe, "error": "format duration missing or unparseable"}

    probe["expected_duration"] = expected_duration_s
    if abs(actual - expected_duration_s) > DURATION_TOLERANCE_S:
        return "DURATION_MISMATCH", probe

    return None, probe


def quarantine_file(
    path: Path,
    quarantine_dir: Path,
    reason: str,
    probe: dict,
    source_url: str | None,
) -> Path:
    """Move a failed file to quarantine with a sidecar describing the failure."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    dest = quarantine_dir / path.name
    n = 1
    while dest.exists():
        dest = quarantine_dir / f"{path.stem} ({n}){path.suffix}"
        n += 1

    shutil.move(str(path), str(dest))
    sidecar = dest.with_suffix(dest.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "reason": reason,
                "original_name": path.name,
                "quarantined_at": datetime.now().isoformat(),
                "source_url": source_url,
                "ffprobe": probe,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest


# ── standalone gate check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        ensure_premium_access()
    except QualityGateError as exc:
        print(f"\nPremium gate FAILED:\n{exc}", file=sys.stderr)
        sys.exit(1)
    print("\nPremium gate passed — 256 kbps AAC is available.")
