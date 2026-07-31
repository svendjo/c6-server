"""Environment configuration, loaded from config/<APP_ENV>.yaml.

One YAML file per environment (config/local-dev.yaml, config/aws-prod.yaml); pick
one with the APP_ENV env var (default "local-dev"), e.g.:

    python server.py                      # local-dev
    APP_ENV=aws-prod python server.py     # aws-prod

The file carries the run flags (reload), the model filenames, the classifier's
confidence threshold, the results-storage backend, and the /predict rate limit, so
the run command itself stays just `python server.py` and a retrained model is a
config change rather than an edit in the source.
"""
import os
from pathlib import Path

import yaml

APP_ENV = os.environ.get("APP_ENV", "local-dev")
_CONFIG_DIR = Path(__file__).resolve().parent / "config"


class ConfigError(Exception):
    """The selected environment's config is missing something it needs.

    A plain exception rather than SystemExit: these are raised from lazily-called
    code (model paths, the results store), so they can surface inside a request --
    where SystemExit would slip past `except Exception` and fail the request with a
    bare ASGI error instead of a useful 503. `validate()` runs the same checks at
    startup so a bad YAML stops the server before it serves anything.
    """


def _load(env):
    path = _CONFIG_DIR / f"{env}.yaml"
    if not path.exists():
        have = ", ".join(sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))) or "none"
        raise SystemExit(f"APP_ENV={env!r}: missing config {path} (available: {have})")
    with open(path) as f:
        return yaml.safe_load(f) or {}


CONFIG = _load(APP_ENV)

RELOAD = bool(CONFIG.get("reload"))
CONFIDENCE_THRESHOLD = float(CONFIG.get("confidence_threshold", 0.8))
RESULTS = CONFIG.get("results") or {}
MODELS = CONFIG.get("models") or {}

# Absent block == no rate limiting, which is why this is a plain `or {}` rather
# than a default of 20/hour: an environment that says nothing about rate limiting
# gets none, and turning it off is deleting the block rather than finding the
# magic number that means "unlimited".
RATE_LIMIT = CONFIG.get("rate_limit") or {}


def rate_limit_settings():
    """(requests, window_seconds) for the limiter, or None when it is switched off."""
    if not RATE_LIMIT:
        return None
    requests = RATE_LIMIT.get("requests")
    if requests is None:
        return None
    return int(requests), int(RATE_LIMIT.get("window_seconds", 3600))


def model_path(kind):
    """Filesystem path to the `kind` ("counting" / "classification") model.

    The filename lives in the env YAML rather than in code, so a retrain is a config
    change (plus the Dockerfile's COPY line) instead of an edit in the source.
    Resolved relative to THIS directory, not the working directory, so the server
    finds its models however the process was started.
    """
    name = MODELS.get(kind)
    if not name:
        raise ConfigError(
            f"APP_ENV={APP_ENV!r}: config/{APP_ENV}.yaml has no models.{kind} entry."
        )
    return Path(__file__).resolve().parent / name


def validate():
    """Check everything the server needs from the config, at startup.

    Raises ConfigError listing every problem at once (rather than one per restart):
    both model entries must be named and present on disk, the confidence threshold
    must be a probability, and the results backend must be one the store understands.
    Called from the app's lifespan, so a bad YAML fails the boot loudly instead of
    surfacing on the first prediction.
    """
    problems = []
    for kind in ("counting", "classification"):
        try:
            path = model_path(kind)
        except ConfigError as e:
            problems.append(str(e))
            continue
        if not path.exists():
            problems.append(f"models.{kind}: {path.name} is not in {path.parent}")
    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        problems.append(
            f"confidence_threshold {CONFIDENCE_THRESHOLD} is not between 0 and 1")
    backend = RESULTS.get("backend", "local")
    if backend not in ("local", "s3"):
        problems.append(f"results.backend {backend!r} is not 'local' or 's3'")
    if backend == "s3" and not RESULTS.get("bucket"):
        problems.append("results.backend is 's3' but results.bucket is missing")
    if RATE_LIMIT:
        try:
            requests, window = rate_limit_settings() or (None, None)
            if requests is not None:
                if requests < 1:
                    problems.append(
                        f"rate_limit.requests {requests} is not at least 1 "
                        "(delete the rate_limit block to switch limiting off)")
                if window < 1:
                    problems.append(
                        f"rate_limit.window_seconds {window} is not at least 1")
        except (TypeError, ValueError):
            problems.append(
                "rate_limit.requests and rate_limit.window_seconds must be whole "
                f"numbers, got {RATE_LIMIT.get('requests')!r} and "
                f"{RATE_LIMIT.get('window_seconds')!r}")
    if problems:
        raise ConfigError(
            f"config/{APP_ENV}.yaml is not usable:\n  - " + "\n  - ".join(problems)
        )
