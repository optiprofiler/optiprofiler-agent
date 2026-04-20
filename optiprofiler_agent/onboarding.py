"""First-run onboarding: provision ``~/.opagent/.env`` interactively.

Drives the ``opagent init`` subcommand and is also auto-invoked from
``cli.cli_main`` on the first launch when no API key is reachable through
any of the supported sources.

Design notes
------------
* **Schema-first.** All write operations route through ``_write_env_file``
  so we never accidentally drop a comment or change the file layout — the
  user can re-run ``opagent init`` and only the relevant ``KEY=value``
  lines change.
* **Secrets stay user-only.** After writing, we ``chmod 0600`` (POSIX
  only). The seed template ships with the same mode, so this is a
  no-op when the file was just bootstrap-copied.
* **Custom-provider opt-in.** Rather than ship a preset for every
  vendor (a maintenance treadmill), the wizard exposes a ``custom``
  option that maps to ``OPAGENT_CUSTOM_{BASE_URL,MODEL,API_KEY}``. The
  ``LLMConfig`` reads those at runtime (see ``config.py``).
* **Idempotent + scriptable.** ``--no-interactive`` makes the function
  bail out cleanly with a status code instead of blocking on stdin —
  required when the CLI auto-detects a missing key inside a CI job.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from optiprofiler_agent.config import PROVIDER_REGISTRY
from optiprofiler_agent.runtime import bootstrap, paths


# Ordered choices presented in the wizard. Built-in providers first, then
# the catch-all ``custom`` for any OpenAI-compatible endpoint we don't
# ship presets for.
_BUILTIN_ORDER: tuple[str, ...] = (
    "minimax",
    "kimi",
    "openai",
    "deepseek",
    "anthropic",
)


# Env vars whose presence we treat as "this user already has a working
# provider". If any of these are set (in real env, cwd .env, or
# ~/.opagent/.env) the auto-onboarding hook in cli.cli_main does NOT fire.
def known_provider_env_vars() -> tuple[str, ...]:
    keys: list[str] = []
    for name in PROVIDER_REGISTRY:
        env_key = PROVIDER_REGISTRY[name].get("env_key")
        if env_key:
            keys.append(env_key)
    return tuple(dict.fromkeys(keys))  # de-dup, preserve order


def has_any_provider_key() -> bool:
    """True iff any provider's env var resolves to a non-empty value."""
    return any(os.environ.get(k) for k in known_provider_env_vars())


def detect_configured_providers() -> list[str]:
    """Return provider names whose env_key currently resolves to a value.

    Used by the wizard to tell the user *which* provider is already set up
    rather than the generic "unknown". Order follows ``PROVIDER_REGISTRY``
    insertion order so the listing is deterministic.
    """
    seen: list[str] = []
    for name, info in PROVIDER_REGISTRY.items():
        env_key = info.get("env_key")
        if env_key and os.environ.get(env_key):
            seen.append(name)
    return seen


def active_default_provider() -> str | None:
    """The provider that *would* be used by ``LLMConfig()`` right now.

    Resolution order matches ``config._default_provider`` followed by the
    "any configured provider wins" fallback. Returns ``None`` only when
    truly nothing is configured.
    """
    explicit = os.environ.get("OPAGENT_DEFAULT_PROVIDER", "").strip().lower()
    if explicit and explicit in PROVIDER_REGISTRY:
        # Honor explicit setting only when the matching key is actually
        # present — otherwise fall through and report a usable default.
        info = PROVIDER_REGISTRY[explicit]
        if not info.get("env_key") or os.environ.get(info["env_key"]):
            return explicit
    configured = detect_configured_providers()
    return configured[0] if configured else None


# ---------------------------------------------------------------------------
# .env file mutation
# ---------------------------------------------------------------------------

# Match a ``KEY=value`` line, optionally preceded by whitespace and an
# inline ``#`` comment marker. We deliberately tolerate the seed template's
# commented-out hints (``# MINIMAX_API_KEY=...``) so the wizard re-uses the
# template line in place, keeping the file readable.
_ENV_LINE_RE = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=.*$")


def _read_env_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    """Merge ``updates`` into the dotenv file at ``path``.

    Preserves all comments and unrelated lines. For each key in ``updates``
    we replace the first matching ``KEY=...`` line (commented or not); any
    leftover keys are appended at the end under a clearly-marked section.
    """
    lines = _read_env_lines(path)
    remaining = dict(updates)
    out: list[str] = []
    for raw in lines:
        m = _ENV_LINE_RE.match(raw)
        if m and m.group(1) in remaining:
            key = m.group(1)
            value = remaining.pop(key)
            out.append(f"{key}={value}")
        else:
            out.append(raw)

    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.append("# --- written by `opagent init` ---")
        for key, value in remaining.items():
            out.append(f"{key}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    if os.name == "posix":
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

@dataclass
class OnboardResult:
    """What ``run_init`` ended up doing — used by tests + the CLI summary."""

    written_path: Path | None
    provider: str | None
    skipped: bool
    reason: str | None = None


def _print(line: str = "") -> None:
    """Indirection so tests can patch a single sink."""
    print(line)


def _prompt(question: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{question}{suffix}: ").strip()
    if not raw and default is not None:
        return default
    return raw


def _prompt_choice(prompt: str, choices: Iterable[str], *, default: str) -> str:
    options = list(choices)
    while True:
        _print(prompt)
        for idx, name in enumerate(options, 1):
            marker = " (default)" if name == default else ""
            _print(f"  {idx}. {name}{marker}")
        raw = input(f"Pick 1-{len(options)} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        _print(f"  '{raw}' is not a valid choice; try again.")


def _gather_builtin(provider: str) -> dict[str, str]:
    info = PROVIDER_REGISTRY[provider]
    env_key = info["env_key"]
    key = _prompt(f"Paste your {provider.upper()} API key (input is echoed)").strip()
    if not key:
        raise ValueError(f"Empty {env_key}; aborting.")
    return {
        env_key: key,
        "OPAGENT_DEFAULT_PROVIDER": provider,
    }


def _gather_custom() -> dict[str, str]:
    _print("\nCustom provider — any OpenAI-compatible endpoint.")
    _print("You'll need: API base URL, model name, and an API key.\n")
    base_url = _prompt("Base URL (e.g. https://api.example.com/v1)")
    if not base_url:
        raise ValueError("Empty base URL; aborting.")
    model = _prompt("Model identifier (e.g. my-model-1)")
    if not model:
        raise ValueError("Empty model name; aborting.")
    key = _prompt("API key")
    if not key:
        raise ValueError("Empty API key; aborting.")
    return {
        "OPAGENT_DEFAULT_PROVIDER": "custom",
        "OPAGENT_CUSTOM_BASE_URL": base_url,
        "OPAGENT_CUSTOM_MODEL": model,
        "OPAGENT_CUSTOM_API_KEY": key,
    }


def run_init(
    *,
    force: bool = False,
    no_interactive: bool = False,
) -> OnboardResult:
    """Run the interactive provider setup. Returns what was done.

    ``no_interactive=True`` short-circuits without touching stdin or the
    filesystem — used by the CLI's auto-trigger when stdin is not a tty
    (CI, piped input, IDE plugins) so we degrade gracefully instead of
    blocking forever on ``input()``.
    """
    bootstrap.ensure()  # make sure ~/.opagent/.env (template) exists

    env_path = paths.env_path()

    if no_interactive:
        return OnboardResult(
            written_path=None,
            provider=None,
            skipped=True,
            reason="no-interactive mode",
        )

    if not sys.stdin.isatty():
        return OnboardResult(
            written_path=None,
            provider=None,
            skipped=True,
            reason="stdin is not a tty",
        )

    configured = detect_configured_providers()
    active_default = active_default_provider()
    explicit_default = os.environ.get("OPAGENT_DEFAULT_PROVIDER", "").strip().lower()

    if configured and not force:
        # Be specific: list every key we found and what would actually be
        # used right now. The previous "provider 'unknown'" message left
        # users guessing what's already in their .env.
        _print(f"Detected existing API key(s) for: {', '.join(configured)}")
        if explicit_default and explicit_default in PROVIDER_REGISTRY:
            _print(f"Default provider (OPAGENT_DEFAULT_PROVIDER): {explicit_default}")
        elif active_default:
            _print(
                f"Default provider: {active_default}  "
                f"(inferred — set OPAGENT_DEFAULT_PROVIDER to make it explicit)"
            )
        _print(
            "\nNote: existing keys are kept. Picking a different provider "
            "below adds its key alongside; your old key is not erased."
        )
        ans = _prompt("Continue and (re)configure? [y/N]", default="n").lower()
        if ans not in ("y", "yes"):
            return OnboardResult(
                written_path=None,
                provider=active_default,
                skipped=True,
                reason="user declined overwrite",
            )

    _print("\nopagent init — pick the provider you want to use:")
    options = list(_BUILTIN_ORDER) + ["custom"]
    default_choice = active_default if active_default in options else "minimax"
    provider = _prompt_choice(
        "Available providers:",
        options,
        default=default_choice,
    )

    try:
        if provider == "custom":
            updates = _gather_custom()
        else:
            updates = _gather_builtin(provider)
    except ValueError as exc:
        _print(f"\n{exc}")
        return OnboardResult(
            written_path=None,
            provider=provider,
            skipped=True,
            reason=str(exc),
        )

    _write_env_file(env_path, updates)

    _print(f"\nWrote {len(updates)} keys to {env_path} (mode 0600).")
    _print("Run `opagent` to start chatting.")
    return OnboardResult(
        written_path=env_path,
        provider=provider,
        skipped=False,
    )


__all__ = [
    "OnboardResult",
    "has_any_provider_key",
    "known_provider_env_vars",
    "run_init",
]
