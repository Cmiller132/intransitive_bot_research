"""Conversions at the boundary keep every stored square in one absolute frame."""

from __future__ import annotations

from typing import Literal, TypeAlias

Frame: TypeAlias = Literal["meaf", "henhen"]
FILES = "abcdefghi"


def _check_square(sq: int) -> int:
    if isinstance(sq, bool) or not isinstance(sq, int) or not 0 <= sq < 81:
        raise ValueError(f"square index must be in 0..80, got {sq!r}")
    return sq


def mirror_file(sq: int) -> int:
    """Mirror a square left-to-right; this is the henhen display transform."""
    rank, file = divmod(_check_square(sq), 9)
    return rank * 9 + 8 - file


def to_frame(sq: int, frame: Frame) -> int:
    """Convert an absolute square index to a display-frame square index."""
    if frame == "meaf":
        return _check_square(sq)
    if frame == "henhen":
        return mirror_file(sq)
    raise ValueError(f"unknown frame {frame!r}")


def from_frame(sq: int, frame: Frame) -> int:
    """Convert a display-frame square index to an absolute square index."""
    return to_frame(sq, frame)  # both supported transforms are involutions


def square_name(sq: int, frame: Frame = "meaf") -> str:
    shown = to_frame(sq, frame)
    return f"{FILES[shown % 9]}{shown // 9 + 1}"


def parse_square(name: str, frame: Frame = "meaf") -> int:
    text = name.strip().lower()
    if len(text) != 2 or text[0] not in FILES or text[1] not in "123456789":
        raise ValueError(f"invalid square {name!r}")
    shown = (int(text[1]) - 1) * 9 + FILES.index(text[0])
    return from_frame(shown, frame)
