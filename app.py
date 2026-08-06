"""
Spotimine web server.
Run:  python app.py
      docker-compose up
"""

import json
import os
import queue
import threading
import uuid
from pathlib import Path

from flask import (Flask, Response, after_this_request, jsonify, render_template,
                   request, send_file, send_from_directory)

import quality

SONGS_DIR   = Path("songs")
REPORTS_DIR = Path("static") / "reports"
AUDIO_EXTS  = {".mp3", ".m4a", ".opus", ".flac", ".wav"}
TAGS_FILE       = REPORTS_DIR / "tags.json"
PLAYLISTS_FILE  = REPORTS_DIR / "playlists.json"

SONGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# job_id → queue.Queue  (None sentinel = stream finished)
_jobs: dict[str, queue.Queue] = {}
_jobs_lock = threading.Lock()

# Cached accurate chart bounds (computed once, avoids re-importing matplotlib repeatedly)
_chart_bounds: dict | None = None
_chart_bounds_lock = threading.Lock()


def _get_chart_bounds() -> dict:
    global _chart_bounds
    with _chart_bounds_lock:
        if _chart_bounds is None:
            from analyzer import compute_chart_bounds
            _chart_bounds = compute_chart_bounds()
    return _chart_bounds


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name)


def _read_quality() -> dict:
    """Measured properties per library file, written by the download pass."""
    path = SONGS_DIR / "quality.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _measure_unrecorded(records: dict) -> dict:
    """Fill in measurements for library files that have no record yet.

    Files that predate verification, or were added by hand, still deserve to show
    their real codec and bitrate. They get facts but no verdict: without Spotify's
    duration there is nothing to judge them against, and inventing a pass or fail
    would be worse than admitting they were never checked.

    Results are cached in quality.json, so this costs one ffprobe per new file
    rather than one per page load.
    """
    fresh = {}
    for f in SONGS_DIR.iterdir():
        if f.suffix.lower() not in AUDIO_EXTS or f.name in records:
            continue
        try:
            probe = quality.probe_audio(f)
            probe["file_size"] = f.stat().st_size
            stream = (probe.get("streams") or [{}])[0]
            fresh[f.name] = {
                "ok": None,
                "verified": False,
                "reason": None,
                "codec": stream.get("codec_name"),
                "bitrate": quality.effective_bitrate(probe),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels"),
                "lossless": stream.get("codec_name") in quality.LOSSLESS_CODECS,
                "source": None,
            }
        except Exception:
            fresh[f.name] = {"ok": None, "verified": False, "reason": None,
                             "codec": None, "bitrate": None}

    if fresh:
        records = {**records, **fresh}
        (SONGS_DIR / "quality.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8")
    return records


# ── chart helpers ─────────────────────────────────────────────────────────────

def _audio_index() -> dict[str, str]:
    """Return {safe_name: audio_filename} for every file in songs/."""
    idx = {}
    for f in SONGS_DIR.iterdir():
        if f.suffix.lower() in AUDIO_EXTS:
            idx[_safe_name(f.stem)] = f.name
    return idx


def _load_charts() -> list[dict]:
    audio_idx = _audio_index()
    result = []
    for png in sorted(REPORTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = {}
        meta_path = png.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        audio_file = meta.get("source_file", "") or audio_idx.get(png.stem, "")
        # Use per-chart bounds if present; otherwise use server-computed accurate fallback
        fb = _get_chart_bounds()
        result.append({
            "name":       meta.get("name", png.stem.replace("_", " ")),
            "file":       png.name,
            "audio_file": audio_file,
            "duration":   meta.get("duration"),
            "tempo":      meta.get("tempo"),
            "key":        meta.get("key"),
            "cl":         meta.get("cl")  if meta.get("cl")  is not None else fb["cl"],
            "cr":         meta.get("cr")  if meta.get("cr")  is not None else fb["cr"],
            "cyt":        meta.get("cyt") if meta.get("cyt") is not None else fb["cyt"],
            "cyb":        meta.get("cyb") if meta.get("cyb") is not None else fb["cyb"],
        })
    return result


# ── main routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def start_analyze():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    job_id = uuid.uuid4().hex[:8]
    q: queue.Queue = queue.Queue()
    with _jobs_lock:
        _jobs[job_id] = q

    thread = threading.Thread(target=_run_job, args=(job_id, url, q), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        def _err():
            yield 'data: {"type":"error","msg":"Job not found"}\n\n'
        return Response(_err(), mimetype="text/event-stream")

    def generate():
        while True:
            try:
                item = q.get(timeout=30)
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        with _jobs_lock:
            _jobs.pop(job_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/reports")
def list_reports():
    return jsonify(_load_charts())


@app.route("/chart-bounds")
def chart_bounds():
    """Return pixel-accurate chart axis bounds for the JS playhead."""
    return jsonify(_get_chart_bounds())


# ── songs library routes ──────────────────────────────────────────────────────

@app.route("/songs")
def list_songs():
    """List all downloaded audio files with chart/metadata if available."""
    # Build an index of charts by safe-name so we can match to audio files
    chart_index: dict[str, dict] = {}
    for png in REPORTS_DIR.glob("*.png"):
        meta = {}
        meta_path = png.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        chart_index[png.stem] = {"chart_file": png.name, **meta}

    all_tags = _read_tags()
    all_quality = _measure_unrecorded(_read_quality())
    songs = []
    for f in sorted(SONGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        info = chart_index.get(_safe_name(f.stem), {})
        songs.append({
            "name":       f.stem,
            "file":       f.name,
            "chart_file": info.get("chart_file"),
            "duration":   info.get("duration"),
            "tempo":      info.get("tempo"),
            "key":        info.get("key"),
            "tags":       all_tags.get(f.name, []),
            # What the file actually measured as, so a flagged track is visible
            # in the listing rather than only in a run log.
            "quality":    all_quality.get(f.name),
        })
    return jsonify(songs)


@app.route("/songs/play/<path:filename>")
def play_song(filename: str):
    """Stream an audio file to the browser (supports range requests for seeking)."""
    return send_from_directory(SONGS_DIR, filename)


@app.route("/songs/download/<path:filename>")
def download_song(filename: str):
    """Serve an audio file as a download attachment."""
    return send_from_directory(SONGS_DIR, filename, as_attachment=True)


@app.route("/songs/<path:filename>", methods=["DELETE"])
def delete_song(filename: str):
    """Delete a song and its associated chart PNG and metadata JSON."""
    audio_path = SONGS_DIR / filename
    if audio_path.exists():
        audio_path.unlink()
    safe = _safe_name(audio_path.stem)
    for ext in (".png", ".json"):
        p = REPORTS_DIR / f"{safe}{ext}"
        if p.exists():
            p.unlink()
    # Drop its quality record too, so a re-download is measured afresh rather
    # than inheriting the old verdict.
    records = _read_quality()
    if records.pop(filename, None) is not None:
        (SONGS_DIR / "quality.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/songs", methods=["DELETE"])
def purge_all():
    """Delete all songs and all chart files."""
    for f in SONGS_DIR.iterdir():
        if f.is_file():
            f.unlink()
    for f in REPORTS_DIR.iterdir():
        if f.is_file():
            f.unlink()
    return jsonify({"ok": True})


@app.route("/static/reports/<path:filename>")
def serve_report(filename: str):
    return send_from_directory(REPORTS_DIR, filename)


# ── youtube music session cookies ─────────────────────────────────────────────
# These endpoints let cookies be pasted in from the browser, for hosts where
# dropping a file next to the app isn't practical.  They deliberately never
# return cookie contents — only whether what is stored works.

def _reason(problem: str | None) -> str | None:
    """Strip the machine-readable prefix off a quality.py problem string."""
    return None if problem is None else problem.split(":", 1)[1].strip()


@app.route("/cookies/status")
def cookies_status():
    """Report whether usable cookies are stored. Never returns their contents."""
    problem = quality.check_cookie_file(quality.COOKIE_FILE)
    return jsonify({
        "present": quality.COOKIE_FILE.exists(),
        "valid":   problem is None,
        "reason":  _reason(problem),
        "path":    str(quality.COOKIE_FILE),
    })


@app.route("/cookies", methods=["PUT"])
def save_cookies():
    """Accept pasted Netscape cookie text and store it.

    Validation happens before the write, so a bad paste can't clobber cookies
    that currently work.
    """
    data = request.get_json(silent=True) or {}
    try:
        quality.save_cookie_text(data.get("text", ""))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"ok": False, "error": f"Could not write cookie file: {exc}"}), 500
    return jsonify({"ok": True, "path": str(quality.COOKIE_FILE)})


@app.route("/cookies/check", methods=["POST"])
def check_cookies():
    """Run the real itag 141 probe against the stored cookies.

    Storing cookies proves nothing — an expired session or the wrong Google
    account both look fine on disk — so this is what actually confirms Premium.
    """
    problem = quality.check_cookie_file(quality.COOKIE_FILE)
    if problem:
        return jsonify({"premium": False, "log": [_reason(problem)]})

    lines: list[str] = []
    try:
        ok = quality.probe_premium(quality.COOKIE_FILE, lines.append)
    except Exception as exc:
        return jsonify({"premium": False, "log": lines + [str(exc)]}), 500
    return jsonify({"premium": ok, "log": lines})


# ── soulseek credentials ──────────────────────────────────────────────────────
# Same rules as the cookie endpoints: the password goes in, and nothing ever
# comes back out. The username is returned so you can confirm which account is
# configured, but there is no route that reads the password back.

@app.route("/soulseek/status")
def soulseek_status():
    """What Soulseek configuration is in effect. Never returns the password."""
    import soulseek
    return jsonify(soulseek.credentials_status())


@app.route("/soulseek/credentials", methods=["PUT"])
def save_soulseek_credentials():
    """Store Soulseek credentials for the second download pass."""
    import soulseek
    data = request.get_json(silent=True) or {}
    try:
        soulseek.save_credentials(data.get("username", ""), data.get("password", ""))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"ok": False, "error": f"Could not write config: {exc}"}), 500
    return jsonify({"ok": True, **soulseek.credentials_status()})


@app.route("/soulseek/credentials", methods=["DELETE"])
def delete_soulseek_credentials():
    """Forget the stored credentials, disabling the Soulseek pass."""
    import soulseek
    removed = soulseek.clear_credentials()
    return jsonify({"ok": True, "removed": removed, **soulseek.credentials_status()})


# ── tags ──────────────────────────────────────────────────────────────────────

def _read_tags() -> dict:
    if TAGS_FILE.exists():
        try:
            return json.loads(TAGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_tags(tags: dict):
    TAGS_FILE.write_text(json.dumps(tags, indent=2), encoding="utf-8")


@app.route("/tags")
def get_tags():
    return jsonify(_read_tags())


@app.route("/tags/<path:filename>", methods=["PUT"])
def set_tags(filename: str):
    data = request.get_json(silent=True) or {}
    new_tags = [str(t).strip() for t in data.get("tags", []) if str(t).strip()]
    tags = _read_tags()
    if new_tags:
        tags[filename] = new_tags
    else:
        tags.pop(filename, None)
    _write_tags(tags)
    return jsonify({"ok": True, "tags": new_tags})


# ── playlists ─────────────────────────────────────────────────────────────────

def _read_playlists() -> list:
    if PLAYLISTS_FILE.exists():
        try:
            return json.loads(PLAYLISTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _write_playlists(data: list):
    PLAYLISTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _mm_ss(seconds) -> str:
    s = int(seconds or 0)
    return f"{s // 60}:{s % 60:02d}"


@app.route("/playlists")
def list_playlists():
    return jsonify(_read_playlists())


@app.route("/playlists", methods=["POST"])
def create_playlist():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "My Class").strip()
    pl = {"id": uuid.uuid4().hex[:8], "name": name, "tracks": []}
    pls = _read_playlists()
    pls.append(pl)
    _write_playlists(pls)
    return jsonify(pl)


@app.route("/playlists/<pl_id>", methods=["PUT"])
def update_playlist(pl_id: str):
    data = request.get_json(silent=True) or {}
    pls  = _read_playlists()
    for pl in pls:
        if pl["id"] == pl_id:
            if "name"   in data: pl["name"]   = data["name"]
            if "tracks" in data: pl["tracks"] = data["tracks"]
            _write_playlists(pls)
            return jsonify(pl)
    return jsonify({"error": "not found"}), 404


@app.route("/playlists/<pl_id>", methods=["DELETE"])
def delete_playlist(pl_id: str):
    pls = [p for p in _read_playlists() if p["id"] != pl_id]
    _write_playlists(pls)
    return jsonify({"ok": True})


@app.route("/playlists/<pl_id>/download")
def download_playlist_audio(pl_id: str):
    """Zip every audio file in a class, numbered in running order."""
    import tempfile
    import zipfile

    pl = next((p for p in _read_playlists() if p["id"] == pl_id), None)
    if not pl:
        return jsonify({"error": "not found"}), 404

    wanted = []
    missing = []
    for position, track in enumerate(pl.get("tracks", []), start=1):
        if isinstance(track, str):
            track = {"file": track}
        # Playlist entries are user-supplied, so take only the bare filename —
        # never let one reach outside the songs directory.
        name = Path(track.get("file", "")).name
        path = SONGS_DIR / name
        if name and path.is_file():
            wanted.append((position, path))
        elif name:
            missing.append(name)

    if not wanted:
        return jsonify({"error": "no audio files found for this class"}), 404

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    # Stored, not deflated: audio is already compressed, so deflating spends CPU
    # for essentially no saving.
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
        for position, path in wanted:
            zf.write(path, arcname=f"{position:02d} - {path.name}")
        if missing:
            zf.writestr(
                "MISSING.txt",
                "These tracks are in the class but not in the library:\n\n"
                + "\n".join(f"  {n}" for n in missing) + "\n",
            )

    @after_this_request
    def _cleanup(response):
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return response

    safe = _safe_name(pl["name"]).strip() or "class"
    return send_file(tmp.name, mimetype="application/zip",
                     as_attachment=True, download_name=f"{safe}.zip")


@app.route("/playlists/<pl_id>/export")
def export_playlist(pl_id: str):
    fmt = request.args.get("format", "json")
    pl  = next((p for p in _read_playlists() if p["id"] == pl_id), None)
    if not pl:
        return jsonify({"error": "not found"}), 404

    # Build a lookup: safe_name → chart metadata
    chart_meta: dict[str, dict] = {}
    for png in REPORTS_DIR.glob("*.png"):
        mp = png.with_suffix(".json")
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                chart_meta[_safe_name(m.get("name", png.stem))] = m
            except Exception:
                pass

    all_tags  = _read_tags()
    tracks_out = []
    total_sec  = 0
    bpms: list[float] = []

    import urllib.parse
    for i, t in enumerate(pl.get("tracks", [])):
        # tracks are stored as {file, notes} objects or plain strings (legacy)
        if isinstance(t, str):
            t = {"file": t, "notes": ""}
        filename = t.get("file", "")
        notes    = t.get("notes", "")
        stem     = Path(filename).stem
        m        = chart_meta.get(_safe_name(stem), {})
        tags     = all_tags.get(filename, [])
        dur      = m.get("duration") or 0
        bpm      = m.get("tempo")
        total_sec += dur
        if bpm:
            bpms.append(bpm)
        search_url = "https://open.spotify.com/search/" + urllib.parse.quote(stem)
        tracks_out.append({
            "position":         i + 1,
            "name":             stem,
            "file":             filename,
            "bpm":              round(bpm, 1) if bpm else None,
            "key":              m.get("key"),
            "duration":         _mm_ss(dur) if dur else None,
            "duration_seconds": round(dur) if dur else None,
            "tags":             tags,
            "notes":            notes,
            "spotify_search":   search_url,
        })

    avg_bpm = round(sum(bpms) / len(bpms), 1) if bpms else None
    summary = {
        "playlist":               pl["name"],
        "track_count":            len(tracks_out),
        "total_duration":         _mm_ss(total_sec),
        "total_duration_seconds": round(total_sec),
        "average_bpm":            avg_bpm,
    }

    if fmt == "json":
        payload = {**summary, "tracks": tracks_out}
        resp = Response(
            json.dumps(payload, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{pl["name"]}.json"'},
        )
        return resp

    if fmt == "markdown":
        md = _build_markdown(summary, tracks_out)
        return Response(
            md,
            mimetype="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{pl["name"]}.md"'},
        )

    if fmt == "csv":
        csv = _build_csv(summary, tracks_out)
        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{pl["name"]}.csv"'},
        )

    return jsonify({"error": "unknown format"}), 400


def _build_csv(summary: dict, tracks: list) -> str:
    """Build CSV for Spotlistr or other Spotify playlist import tools."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["Track Name", "Artist Name", "Album Name", "BPM", "Key"])
    writer.writeheader()

    for t in tracks:
        # Try to extract artist name from the track name (usually "Artist - Track Name" format)
        name_parts = t["name"].split(" - ")
        artist = name_parts[0] if len(name_parts) > 1 else ""
        track_name = name_parts[-1] if len(name_parts) > 1 else t["name"]

        writer.writerow({
            "Track Name": track_name.strip(),
            "Artist Name": artist.strip(),
            "Album Name": "",
            "BPM": t["bpm"] or "",
            "Key": t["key"] or "",
        })

    return output.getvalue()


def _build_markdown(summary: dict, tracks: list) -> str:
    name     = summary["playlist"]
    total    = summary["total_duration"]
    n        = summary["track_count"]
    avg_bpm  = summary["average_bpm"]

    lines = [
        f"# Spin Class: {name}",
        "",
        f"**Total duration:** {total}  |  **Tracks:** {n}  |  **Avg BPM:** {avg_bpm or 'N/A'}",
        "",
        "---",
        "",
        "## Track List",
        "",
        "| # | Track | BPM | Key | Duration | Tags | Notes |",
        "|---|-------|-----|-----|----------|------|-------|",
    ]

    for t in tracks:
        tags_str  = ", ".join(t["tags"]) if t["tags"] else "—"
        lines.append(
            f"| {t['position']} | {t['name']} | {t['bpm'] or '—'} | "
            f"{t['key'] or '—'} | {t['duration'] or '—'} | {tags_str} | {t['notes'] or ''} |"
        )

    lines += ["", "---", "", "## Track Details", ""]

    for t in tracks:
        tags_str = ", ".join(t["tags"]) if t["tags"] else "None"
        lines += [
            f"### {t['position']}. {t['name']}",
            f"- **Duration:** {t['duration'] or 'N/A'}",
            f"- **BPM:** {t['bpm'] or 'N/A'}",
            f"- **Key:** {t['key'] or 'N/A'}",
            f"- **Tags:** {tags_str}",
            f"- **Notes:** {t['notes'] or '—'}",
            f"- **Spotify:** [Search on Spotify]({t['spotify_search']})",
            "",
        ]

    lines += [
        "---",
        "",
        "## Instructions for Claude",
        "",
        "I am a spin class instructor. Using the playlist above, please design a complete,"
        f" structured spin class program for a class lasting approximately {total}.",
        "",
        "For **each track** provide:",
        "",
        "1. **Riding position** — seated or standing",
        "2. **Effort type** — sprint / climb / tempo / recovery / attack",
        "3. **Resistance** — light / moderate / heavy (or 1–10)",
        "4. **Cadence (RPM)** — use the track BPM as a guide "
        "(e.g. 1:1 = BPM RPM, 2:1 = BPM/2 RPM for climbs)",
        "5. **RPE** — Rate of Perceived Exertion 1–10",
        "6. **Instructor cues** — 2–3 motivational phrases or coaching points",
        "7. **Transition** — how to set up the next track",
        "",
        "Also provide:",
        "- An overall class arc (warm-up → build → peaks → recovery → cool-down)",
        "- Any suggested music structure notes (where the drop is, tempo changes, etc.)",
        "",
        "Tag meanings used in this playlist:",
        "- **Warmup / Cooldown** — easy effort, low resistance",
        "- **Sprint** — high cadence, low–moderate resistance, seated",
        "- **Attack** — short burst of all-out effort",
        "- **Climb** — low cadence, heavy resistance, seated or standing",
        "- **Tempo** — sustained moderate-high effort",
        "- **Recovery** — active rest between harder efforts",
        "- **Standing** — out of the saddle",
        "",
        "Format the output as a clear section per track, easy to read at a glance during class planning.",
    ]

    return "\n".join(lines)


# ── job runner ────────────────────────────────────────────────────────────────

def _run_job(job_id: str, url: str, q: queue.Queue):
    from analyzer import run_analysis

    def log(msg: str):
        q.put({"type": "log", "msg": msg})

    try:
        run_analysis(url, SONGS_DIR, REPORTS_DIR, log)
        q.put({"type": "done", "charts": _load_charts()})
    except Exception as exc:
        q.put({"type": "error", "msg": str(exc)})
    finally:
        q.put(None)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
