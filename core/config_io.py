"""Externalised configuration loading for APGI.

Resolves a run configuration from three layers, lowest precedence first:

1. the built-in defaults in :mod:`config`;
2. an optional TOML or JSON file (per environment: dev / staging / prod, or
   per experiment);
3. ``APGI_*`` environment variables.

Every resolved configuration is validated through the Pydantic schema in
:mod:`core.config_schema`, so a typo in a file or an env var is caught at load
time rather than producing a silently different run.

TOML is read with the standard library's ``tomllib`` (Python 3.11+), falling
back to ``tomli`` if installed; JSON needs no dependency at all. No third-party
YAML requirement is introduced.

Examples
--------
>>> import os
>>> os.environ["APGI_SEED"] = "123"
>>> cfg = load_config()
>>> cfg["seed"]
123
>>> del os.environ["APGI_SEED"]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["load_config", "load_config_file", "apply_env_overrides", "ENV_PREFIX"]

#: Prefix for environment-variable overrides. ``APGI_SEED=7`` sets ``seed``;
#: ``APGI_USE_RESERVOIR=true`` sets ``use_reservoir``.
ENV_PREFIX = "APGI_"


def _coerce(raw: str) -> Any:
    """Coerce an environment-variable string to a Python scalar.

    Environment variables are always strings, but the configuration is typed.
    JSON is tried first so lists, floats, booleans and ``null`` all round-trip
    with their natural syntax; anything else stays a string.

    Args:
        raw: Raw environment-variable value.

    Returns:
        Parsed value, or the original string when it is not valid JSON.
    """
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", ""}:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a configuration file (``.toml``, ``.json``).

    Args:
        path: Path to the file.

    Returns:
        Parsed configuration mapping. A TOML file may nest its settings under
        an ``[apgi]`` table; that table is unwrapped if present.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is unsupported.
        ImportError: If a TOML file is given on Python < 3.11 without ``tomli``.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"config file not found: {target}")
    suffix = target.suffix.lower()
    if suffix == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
    elif suffix in {".toml", ".tml"}:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover - only on 3.10
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ImportError(
                    "reading TOML on Python < 3.11 requires the 'tomli' package; "
                    "install it or use a .json config file"
                ) from exc
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported config format {suffix!r}; expected .toml or .json")
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping, got {type(data).__name__}")
    # Allow settings to live under an [apgi] table so a config file can be
    # shared with other tooling.
    inner = data.get("apgi")
    return dict(inner) if isinstance(inner, dict) else data


def apply_env_overrides(
    config: dict[str, Any], environ: dict[str, str] | None = None
) -> dict[str, Any]:
    """Overlay ``APGI_*`` environment variables onto a configuration.

    Only keys already present in ``config`` are overridden. An unrecognised
    ``APGI_*`` variable raises rather than being ignored: silently dropping a
    misspelled override would produce a run that differs from the one the
    operator believes they configured.

    Args:
        config: Base configuration.
        environ: Environment mapping; defaults to ``os.environ``.

    Returns:
        New configuration dict with overrides applied.

    Raises:
        ValueError: If an ``APGI_*`` variable does not name a known key.
    """
    env = os.environ if environ is None else environ
    resolved = dict(config)
    unknown: list[str] = []
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        key = name[len(ENV_PREFIX) :].lower()
        if key in resolved:
            resolved[key] = _coerce(raw)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"unrecognised APGI_* environment override(s): {sorted(unknown)}. "
            "These do not match any configuration key; fix the name or remove them "
            "(silently ignoring them would give you a run you did not configure)."
        )
    return resolved


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    use_env: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """Resolve a run configuration from defaults, file, environment and args.

    Precedence, lowest to highest: built-in defaults < file < environment <
    explicit ``overrides``.

    Args:
        path: Optional config file. When ``None``, the ``APGI_CONFIG``
            environment variable is consulted.
        overrides: Explicit last-word overrides (e.g. parsed CLI flags).
        use_env: Whether to apply ``APGI_*`` environment overrides.
        validate: Whether to validate the result against the Pydantic schema.

    Returns:
        Fully resolved configuration dictionary.

    Raises:
        ValueError: If validation fails or an unknown env override is present.
    """
    from config import CONFIG

    resolved: dict[str, Any] = dict(CONFIG)
    file_path = path if path is not None else os.environ.get(f"{ENV_PREFIX}CONFIG")
    if file_path:
        resolved.update(load_config_file(file_path))
    if use_env:
        resolved = apply_env_overrides(resolved)
    if overrides:
        resolved.update(overrides)
    if validate:
        from core.config_schema import APGIConfig

        # Round-tripping through the schema surfaces type errors at load time
        # rather than as a confusing failure deep in the pipeline.
        APGIConfig(**{k: v for k, v in resolved.items() if k in _schema_fields()})
    return resolved


def _schema_fields() -> frozenset[str]:
    """Return the field names declared by the Pydantic config schema."""
    from core.config_schema import APGIConfig

    return frozenset(APGIConfig.model_fields)
