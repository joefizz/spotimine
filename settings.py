"""
Which download sources are in play.

Two sources exist — YouTube Music and Soulseek — and either can be switched off.

Disabled is not the same as unconfigured, and the difference matters to what the
run does.  An unconfigured source is a problem to fix: the run says so and tells
you how.  A disabled one is a decision you made, so the run skips it silently and
never nags about a missing cookie file for a source you deliberately turned off.

The one consequence worth knowing: turning YouTube Music off also skips the
Premium gate.  That gate is a hard abort by design — it exists so a run can never
quietly degrade to 128 kbps — but it has nothing to say about a Soulseek-only
run, and letting an expired cookie kill a run that was never going to touch
YouTube Music would be an abort with no purpose.

Environment variables win over anything saved through the web UI, so a
deployment that pins SPOTIMINE_ENABLE_* stays authoritative.
"""

import json
import os
import threading
from pathlib import Path

import quality

# Order matters: this is the order the sources are tried in, and the order the
# UI lists them.
SOURCES = ("ytmusic", "soulseek")

LABELS = {
    "ytmusic":  "YouTube Music",
    "soulseek": "Soulseek",
}

ENV_VARS = {
    "ytmusic":  "SPOTIMINE_ENABLE_YTMUSIC",
    "soulseek": "SPOTIMINE_ENABLE_SOULSEEK",
}

# Both on, which is the behaviour every existing install already has.
DEFAULTS = {"ytmusic": True, "soulseek": True}

CONFIG_FILE = quality.DATA_DIR / "sources.json"

_lock = threading.Lock()

_TRUE  = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}


class UnknownSource(ValueError):
    """A source name that is not one of SOURCES."""


def _env_value(source: str) -> bool | None:
    """What the environment says about a source, or None if it is silent.

    An unrecognised value is treated as silence rather than as False: guessing
    that SPOTIMINE_ENABLE_SOULSEEK=maybe means "off" would disable a source
    behind your back over a typo.
    """
    raw = os.environ.get(ENV_VARS[source])
    if raw is None:
        return None
    value = raw.strip().casefold()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _load() -> dict:
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                return saved
        except Exception:
            pass
    return {}


def _save(config: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_name(CONFIG_FILE.name + ".tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(tmp, CONFIG_FILE)   # atomic: a crash cannot leave a half-file


def _resolve(source: str, saved: dict) -> bool:
    env = _env_value(source)
    if env is not None:
        return env
    value = saved.get(source)
    return DEFAULTS[source] if not isinstance(value, bool) else value


def is_enabled(source: str) -> bool:
    """Whether a source should be used. Unknown names are never enabled."""
    if source not in SOURCES:
        return False
    with _lock:
        return _resolve(source, _load())


def enabled_sources() -> list[str]:
    """The enabled sources, in the order they are tried."""
    with _lock:
        saved = _load()
        return [s for s in SOURCES if _resolve(s, saved)]


def set_enabled(source: str, enabled: bool) -> bool:
    """Turn a source on or off. Returns whether it is enabled afterwards.

    Both sources off is allowed and saved: it is a legitimate thing to want
    while you sort something out, and refusing it would make the two toggles
    fight each other.  A run with nothing enabled stops immediately and says so,
    which is a far clearer place to find out than a rejected click.
    """
    if source not in SOURCES:
        raise UnknownSource(f"unknown download source {source!r}")
    with _lock:
        config = _load()
        config[source] = bool(enabled)
        _save(config)
        return _resolve(source, config)


def status() -> dict:
    """What is enabled and why, for the UI."""
    with _lock:
        saved = _load()
        return {
            "sources": [
                {
                    "name":    source,
                    "label":   LABELS[source],
                    "enabled": _resolve(source, saved),
                    # So the UI can explain why a toggle will not stick.
                    "env_override": _env_value(source) is not None,
                    "env_var": ENV_VARS[source],
                }
                for source in SOURCES
            ],
            "any_enabled": any(_resolve(s, saved) for s in SOURCES),
        }


def describe_enabled() -> str:
    """A readable list of what is on, for the run log."""
    names = [LABELS[s] for s in enabled_sources()]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return " and ".join(names)
