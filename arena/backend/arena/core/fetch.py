"""Rate-spaced source clients; all callers can inject an httpx client in tests."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

USER_AGENT = "IntransitiveArena/1.0 (+https://github.com/Cmiller132)"
MIN_SPACING_SECONDS = 0.250
_request_lock = asyncio.Lock()
_last_request = 0.0


def extract_henhen_game_id(value: str) -> str:
    """Extract a henhen game id from an id or URL."""
    text = value.strip()
    if "://" in text:
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        text = query.get("gameId", [parsed.path.rstrip("/").split("/")[-1]])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError("invalid henhen game id")
    return text


def extract_henhen_series_id(value: str) -> str:
    """Extract a henhen series id from an id or URL."""
    text = value.strip()
    if "://" in text:
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        text = query.get("series", [parsed.path.rstrip("/").split("/")[-1]])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError("invalid henhen series id")
    return text


def extract_meaf_game_id(value: str) -> str:
    """Extract a MEAF game id from an id or URL."""
    text = value.strip()
    if "://" in text:
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        text = (
            query.get("gameID") or query.get("gameId") or query.get("id") or [parsed.path.rstrip("/").split("/")[-1]]
        )[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError("invalid MEAF game id")
    return text


def extract_meaf_workshop_id(value: str) -> str:
    """Extract a MEAF workshop id from an id or URL."""
    text = value.strip()
    if "://" in text:
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        text = (query.get("id") or query.get("workshop") or [parsed.path.rstrip("/").split("/")[-1]])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError("invalid MEAF workshop id")
    return text


async def _get(url: str, client: httpx.AsyncClient | None = None) -> httpx.Response:
    """Fetch one URL, spacing requests across the process."""
    global _last_request
    async with _request_lock:
        delay = MIN_SPACING_SECONDS - (time.monotonic() - _last_request)
        if delay > 0:
            await asyncio.sleep(delay)
        if client is None:
            async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=20) as owned:
                response = await owned.get(url)
        else:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
        _last_request = time.monotonic()
    response.raise_for_status()
    return response


async def henhen_pgn(game_id: str, client: httpx.AsyncClient | None = None) -> str:
    """Fetch one henhen game's PGN."""
    gid = extract_henhen_game_id(game_id)
    return (await _get(f"https://api-rps.henhen1227.com/api/games/{gid}/pgn", client)).text


async def henhen_series(series_id: str, client: httpx.AsyncClient | None = None) -> list[str]:
    """Fetch the game ids of a henhen bot series."""
    sid = extract_henhen_series_id(series_id)
    data = (await _get(f"https://api-rps.henhen1227.com/api/bot-series/{sid}", client)).json()
    return [game["gameId"] for game in data.get("games", [])]


async def henhen_account_games(
    user_id: str, limit: int = 50, offset: int = 0, client: httpx.AsyncClient | None = None
) -> Any:
    """Fetch a page of a henhen account's games."""
    uid = extract_henhen_game_id(user_id)
    url = f"https://api-rps.henhen1227.com/api/accounts/{uid}/games?limit={limit}&offset={offset}"
    return (await _get(url, client)).json()


async def meaf_game(game_id: str, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Fetch one MEAF game history."""
    gid = extract_meaf_game_id(game_id)
    return (await _get(f"https://meaf.us/rps2/api/game_history?gameID={gid}", client)).json()


async def meaf_workshop(workshop_id: str, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Fetch one MEAF workshop payload."""
    wid = extract_meaf_workshop_id(workshop_id)
    return (await _get(f"https://meaf.us/rps2/api/workshop?id={wid}", client)).json()
