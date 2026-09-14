"""The rules, frames, notation and symmetry the bot binary must agree with."""

from __future__ import annotations

import pytest

from arena.core.frames import mirror_file, parse_square, square_name
from arena.core.game import (
    GameRecord,
    IllegalMoveError,
    MoveRecord,
    PositionState,
    Result,
    initial_board_list,
    legal_moves,
    replay,
    terminal_at,
)
from arena.core.notation import format_move, move_token, parse_move, parse_token
from arena.core.symmetry import images, symmetry_key


def test_initial_position_matches_the_rules() -> None:
    board = initial_board_list()
    assert len(board) == 81
    assert sum(1 <= cell <= 3 for cell in board) == 10
    assert sum(cell >= 4 for cell in board) == 10
    assert board[parse_square("b4")] == 1 and board[parse_square("e2")] == 2
    assert board[parse_square("c5")] == 3
    assert board[0] == 0 and board[80] == 0
    assert len(legal_moves(board, "blue")) == len(legal_moves(board, "red"))


def test_only_king_moves_and_winning_captures_are_legal() -> None:
    board = [0] * 81
    board[parse_square("e5")] = 1
    board[parse_square("e6")] = 6
    board[parse_square("f5")] = 5
    board[parse_square("d5")] = 4
    moves = dict.fromkeys(legal_moves(board, "blue"))
    origin = parse_square("e5")
    assert (origin, parse_square("e6")) in moves
    assert (origin, parse_square("f5")) not in moves
    assert (origin, parse_square("d5")) not in moves
    assert (origin, parse_square("e7")) not in moves
    assert len(moves) == 6


def test_game_ends_on_the_corner_and_on_elimination() -> None:
    board = [0] * 81
    board[parse_square("h8")] = 1
    board[parse_square("a2")] = 4
    record = GameRecord(
        setup=board, moves=[MoveRecord(parse_square("h8"), parse_square("i9"))], result=Result("blue", "corner", "1-0")
    )
    assert replay(record)[-1].result == Result("blue", "corner", "1-0")

    board = [0] * 81
    board[parse_square("e5")] = 1
    board[parse_square("e6")] = 6
    record = GameRecord(setup=board, moves=[MoveRecord(parse_square("e5"), parse_square("e6"))])
    end = replay(record)[-1].result
    assert end.winner == "blue" and end.reason == "no_pieces"


def test_stagnation_after_two_hundred_quiet_plies() -> None:
    board = [0] * 81
    board[parse_square("a5")] = 1
    board[parse_square("i5")] = 4
    moves = []
    for _ in range(50):
        moves.extend(
            [
                MoveRecord(parse_square("a5"), parse_square("a6")),
                MoveRecord(parse_square("i5"), parse_square("i6")),
                MoveRecord(parse_square("a6"), parse_square("a5")),
                MoveRecord(parse_square("i6"), parse_square("i5")),
            ]
        )
    record = GameRecord(setup=board, moves=moves)
    states = replay(record)
    assert states[199].result is None
    assert states[200].result == Result(None, "stagnation", "")
    assert states[200].psc == 200
    assert states[100].repetition > 2


def test_replay_rejects_an_illegal_move() -> None:
    record = GameRecord(moves=[MoveRecord(parse_square("e3"), parse_square("e5"))])
    with pytest.raises(IllegalMoveError) as error:
        replay(record)
    assert error.value.ply == 0


def test_standalone_terminal_detection() -> None:
    board = [0] * 81
    board[80] = 1
    board[0] = 4
    assert terminal_at(board, "red", 0) == Result("blue", "corner", "1-0")
    board = [0] * 81
    board[parse_square("e5")] = 4
    assert terminal_at(board, "blue", 0) == Result("red", "no_pieces", "0-1")
    assert terminal_at(initial_board_list(), "blue", 0) is None


def test_frames_and_tokens_round_trip() -> None:
    for square in range(81):
        assert mirror_file(mirror_file(square)) == square
        assert parse_token(move_token((square, square))) == (square, square)
    assert square_name(0, "henhen") == "i1"
    assert parse_square("i1", "henhen") == 0
    quiet = MoveRecord(parse_square("e3"), parse_square("f3"))
    assert format_move(quiet, initial_board_list(), "meaf", "meaf") == "E3-F3"
    assert parse_move("e3-f3", "meaf", "meaf").to_dict() == quiet.to_dict()
    assert move_token(quiet) == "e3-f3"


def test_symmetry_key_covers_all_six_images() -> None:
    state = PositionState(initial_board_list(), "blue", 0, 0, 1)
    key = symmetry_key(state)
    assert len(images(state)) == 6
    assert all(symmetry_key(image) == key for image in images(state))
