"""Environment-driven service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_ROOT = "/opt/arena"


def _flag(name: str, default: bool = False) -> bool:
    """Read a 0/1 environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be 0 or 1, got {value!r}")


def _int(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer environment value with a lower bound."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _float(name: str, default: float) -> float:
    """Read a float environment value."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """One coherent configuration shared by the API and its worker threads."""

    root: Path
    admin_token: str = ""
    public: bool = False
    jobs: int = 4
    target: float = 30.0
    max_games: int = 1000
    sims: int = 32
    pairs: int = 2
    anchor: str = "sq_g128"
    analysis_procs: int = 2
    bot: list[str] = field(default_factory=list)
    scheduler: bool = True
    name: str = "Intransitive arena"
    version: str = "1.0.0"

    @property
    def database_path(self) -> Path:
        """The sqlite file inside the runtime root."""
        return self.root / "arena.db"

    @property
    def players_dir(self) -> Path:
        """The directory holding one subdirectory per player."""
        return self.root / "players"

    def bot_command(self) -> list[str]:
        """The command line that runs the bot binary."""
        return list(self.bot) if self.bot else [str(self.root / "bot")]

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the ARENA_* environment variables."""
        root = Path(os.getenv("ARENA_ROOT", DEFAULT_ROOT)).expanduser()
        raw_bot = os.getenv("ARENA_BOT", "").strip()
        return cls(
            root=root,
            admin_token=os.getenv("ARENA_ADMIN_TOKEN", ""),
            public=_flag("ARENA_PUBLIC"),
            jobs=_int("ARENA_JOBS", 4),
            target=_float("ARENA_TARGET", 30.0),
            max_games=_int("ARENA_MAX_GAMES", 1000),
            sims=_int("ARENA_SIMS", 32),
            pairs=_int("ARENA_PAIRS", 2),
            anchor=os.getenv("ARENA_ANCHOR", "sq_g128"),
            analysis_procs=_int("ARENA_ANALYSIS_PROCS", 2),
            bot=raw_bot.split() if raw_bot else [],
            scheduler=_flag("ARENA_SCHEDULER", True),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the environment once per process; tests may call cache_clear."""
    return Settings.from_env()
