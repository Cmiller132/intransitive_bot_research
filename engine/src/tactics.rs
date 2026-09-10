//! Exact short tactics for the mover: which legal moves win at once, lose at
//! once to a reply, or force a win within three plies. `search` turns them
//! into exact values; model training uses them as labels. Every check is
//! tested against brute force in tests/tactics.rs.

use crate::board::{step, Action, Board, Cell, Piece, Square, GOAL, N_DIRS};
use crate::rules::{apply, Outcome, Rules, State};

/// Squares from which one king step enters the goal.
const GOAL_NEIGHBOURS: [Square; 3] = [70, 71, 79];

/// Squares beside the opponent's goal, seen in the opponent's frame.
const HOME_NEIGHBOURS: [Square; 3] = [1, 9, 10];

/// The most moves one move can take away from the other side: the captured
/// piece's own moves and the moves onto the square the moved piece now holds.
const MAX_REMOVED: u32 = 16;

fn may_enter(piece: Piece, target: Cell) -> bool {
    match target {
        Cell::Empty => true,
        Cell::Enemy(other) => piece.beats(other),
        Cell::Own(_) => false,
    }
}

/// Number of legal moves the enemy would have if it were to move.
fn enemy_mobility(board: &Board) -> u32 {
    let mut moves = 0;
    for from in 0..board.len() {
        let Cell::Enemy(piece) = board[from] else {
            continue;
        };
        for dir in 0..N_DIRS as u8 {
            if let Some(to) = step(from as Square, dir) {
                let ok = match board[to as usize] {
                    Cell::Empty => true,
                    Cell::Own(other) => piece.beats(other),
                    Cell::Enemy(_) => false,
                };
                moves += ok as u32;
            }
        }
    }
    moves
}

/// For each of `legal`, whether it wins at once.
pub fn wins_at_once(rules: &Rules, state: &State, legal: &[Action]) -> Vec<bool> {
    legal
        .iter()
        .map(|&action| apply(rules, state, action).1 == Outcome::Win)
        .collect()
}

/// True when the mover has a move that wins at once: goal entry, capturing
/// the last enemy piece, or leaving the enemy without a move. The stalemate
/// case is only searched when the enemy has at most `MAX_REMOVED` moves.
pub fn any_win_at_once(rules: &Rules, state: &State) -> bool {
    let board = &state.board;
    for &square in &GOAL_NEIGHBOURS {
        if let Cell::Own(piece) = board[square as usize] {
            if may_enter(piece, board[GOAL as usize]) {
                return true;
            }
        }
    }
    let mut enemies = board.iter().enumerate().filter(|(_, cell)| cell.is_enemy());
    if let (Some((square, &Cell::Enemy(prey))), None) = (enemies.next(), enemies.next()) {
        for dir in 0..N_DIRS as u8 {
            if let Some(from) = step(square as Square, dir) {
                if let Cell::Own(piece) = board[from as usize] {
                    if piece.beats(prey) {
                        return true;
                    }
                }
            }
        }
    }
    if enemy_mobility(board) > MAX_REMOVED {
        return false;
    }
    state
        .legal_actions()
        .into_iter()
        .any(|action| apply(rules, state, action).1 == Outcome::Win)
}

/// For each of `legal`, whether the reply to it can win at once.
pub fn loses_in_two(rules: &Rules, state: &State, legal: &[Action]) -> Vec<bool> {
    legal
        .iter()
        .map(|&action| {
            let (child, outcome) = apply(rules, state, action);
            outcome == Outcome::Ongoing && any_win_at_once(rules, &child)
        })
        .collect()
}

/// For each of `legal`, whether every reply to it leaves the mover a win at
/// once. A reply that wins or draws defeats the move.
pub fn wins_in_three(rules: &Rules, state: &State, legal: &[Action]) -> Vec<bool> {
    legal
        .iter()
        .map(|&action| forces_win(rules, state, action))
        .collect()
}

fn forces_win(rules: &Rules, state: &State, action: Action) -> bool {
    let (child, outcome) = apply(rules, state, action);
    if outcome != Outcome::Ongoing {
        return false;
    }
    // A win at once after any reply needs one of: a piece of ours beside our
    // goal, the enemy down to its last piece, or an enemy with few enough moves
    // that one reply and one move of ours can leave it none.
    let beside_goal = HOME_NEIGHBOURS
        .iter()
        .any(|&square| child.board[square as usize].is_enemy());
    let replies = child.legal_actions();
    if !beside_goal && child.own_count() > 1 && replies.len() as u32 > 2 * MAX_REMOVED {
        return false;
    }
    replies.iter().all(|&reply| {
        let (grandchild, outcome) = apply(rules, &child, reply);
        outcome == Outcome::Ongoing && any_win_at_once(rules, &grandchild)
    })
}

/// Whether the first occupied cell may capture the opposing second cell.
pub fn can_capture(attacker: Cell, target: Cell) -> bool {
    match (attacker, target) {
        (Cell::Own(a), Cell::Enemy(b)) | (Cell::Enemy(a), Cell::Own(b)) => a.beats(b),
        _ => false,
    }
}
/// A legal goal-entry action, if present.
pub fn goal_move(board: &Board) -> Option<Action> {
    for (from, dir) in [(70, 7), (71, 6), (79, 4)] {
        if let Cell::Own(piece) = board[from] {
            if may_enter(piece, board[GOAL as usize]) {
                return Some(crate::action(dir, from as u8));
            }
        }
    }
    None
}
/// Whether the opponent could enter home on its next move without a defense.
pub fn goal_threat(board: &Board) -> bool {
    HOME_NEIGHBOURS.iter().any(|&s| match board[s as usize] {
        Cell::Enemy(p) => may_enter(p, board[0].swap_side()),
        _ => false,
    })
}
/// Whether an occupied square is attacked by an adjacent opposing piece.
pub fn is_attacked(board: &Board, square: Square) -> bool {
    (0..N_DIRS as u8).any(|dir| {
        step(square, dir).is_some_and(|s| can_capture(board[s as usize], board[square as usize]))
    })
}
