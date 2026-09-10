//! The tactics module against brute force on random positions and on
//! constructed cases for each way of winning.

use engine::tactics::{any_win_at_once, loses_in_two, wins_at_once, wins_in_three};
use engine::{apply, from_to, Action, Cell, Outcome, Piece, Rules, State, N_SQUARES};

const RULES: Rules = Rules {
    capture_clock: None,
};

#[test]
fn attack_candidates_cover_every_changed_square() {
    use engine::tactics::{attack_candidates, is_attacked};
    let mut seed = 0x2026091016u64;
    for _ in 0..1000 {
        let mut state = State::initial();
        for cell in &mut state.board {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *cell = Cell::from_code((seed % 7) as u8).unwrap();
        }
        let legal = state.legal_actions();
        if legal.is_empty() {
            continue;
        }
        let action = legal[seed as usize % legal.len()];
        let affected = attack_candidates(&state.board, action);
        let child = engine::flip(&apply(&RULES, &state, action).0.board);
        for square in 0..81 {
            if state.board[square] != child[square]
                || is_attacked(&state.board, square as u8) != is_attacked(&child, square as u8)
            {
                assert_ne!(
                    affected & (1 << square),
                    0,
                    "action {action}, square {square}"
                );
            }
        }
    }
}

fn brute_win1(state: &State, action: Action) -> bool {
    apply(&RULES, state, action).1 == Outcome::Win
}

fn brute_any_win(state: &State) -> bool {
    state
        .legal_actions()
        .into_iter()
        .any(|action| brute_win1(state, action))
}

fn brute_loss2(state: &State, action: Action) -> bool {
    let (child, outcome) = apply(&RULES, state, action);
    outcome == Outcome::Ongoing && brute_any_win(&child)
}

fn brute_win3(state: &State, action: Action) -> bool {
    let (child, outcome) = apply(&RULES, state, action);
    if outcome != Outcome::Ongoing {
        return false;
    }
    child.legal_actions().into_iter().all(|reply| {
        let (grandchild, outcome) = apply(&RULES, &child, reply);
        outcome == Outcome::Ongoing && brute_any_win(&grandchild)
    })
}

fn check(state: &State) {
    let legal = state.legal_actions();
    let win1 = wins_at_once(&RULES, state, &legal);
    let loss2 = loses_in_two(&RULES, state, &legal);
    let win3 = wins_in_three(&RULES, state, &legal);
    assert_eq!(any_win_at_once(&RULES, state), brute_any_win(state));
    for (i, &action) in legal.iter().enumerate() {
        assert_eq!(win1[i], brute_win1(state, action), "win1 {action}");
        assert_eq!(loss2[i], brute_loss2(state, action), "loss2 {action}");
        assert_eq!(win3[i], brute_win3(state, action), "win3 {action}");
    }
}

fn place(cells: &[(usize, Cell)]) -> State {
    let mut state = State::initial();
    state.board = [Cell::Empty; N_SQUARES];
    for &(square, cell) in cells {
        state.board[square] = cell;
    }
    state
}

/// Positions reached by random play, including short endgames.
fn random_positions(count: usize, seed: u64) -> Vec<State> {
    let mut x = seed;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) as usize
    };
    let mut out = Vec::new();
    while out.len() < count {
        let mut state = State::initial();
        let plies = next() % 400;
        for _ in 0..plies {
            let legal = state.legal_actions();
            let (child, outcome) = apply(&RULES, &state, legal[next() % legal.len()]);
            if outcome != Outcome::Ongoing {
                break;
            }
            state = child;
        }
        out.push(state);
    }
    out
}

#[test]
fn matches_brute_force_on_random_positions() {
    for state in random_positions(300, 9) {
        check(&state);
    }
}

#[test]
fn matches_brute_force_on_thinned_positions() {
    // Random positions with most pieces removed reach the stalemate and
    // elimination paths that full boards rarely do.
    let mut x = 77u64;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) as usize
    };
    for mut state in random_positions(300, 5) {
        for cell in state.board.iter_mut() {
            if *cell != Cell::Empty && next() % 3 != 0 {
                *cell = Cell::Empty;
            }
        }
        if state.own_count() == 0 || state.enemy_count() == 0 || state.is_stalemated() {
            continue;
        }
        check(&state);
    }
}

#[test]
fn a_threat_the_opponent_cannot_parry_wins_in_three() {
    // Own scissors g8 next to the enemy paper holding i9; the enemy has no rock
    // to capture scissors and nothing to block with, so g8-h8 wins in three.
    let state = place(&[
        (69, Cell::Own(Piece::Scissors)),
        (80, Cell::Enemy(Piece::Paper)),
        (40, Cell::Enemy(Piece::Paper)),
    ]);
    let legal = state.legal_actions();
    let win3 = wins_in_three(&RULES, &state, &legal);
    let winning: Vec<u8> = legal
        .iter()
        .zip(&win3)
        .filter(|(_, &w)| w)
        .map(|(&a, _)| from_to(a).1)
        .collect();
    assert_eq!(winning, vec![70, 79], "h8 and h9 both threaten the goal");
    assert!(!wins_at_once(&RULES, &state, &legal).iter().any(|&w| w));
    check(&state);
}

#[test]
fn a_move_that_allows_goal_entry_loses_in_two() {
    // Enemy paper b2 reaches our home unless our scissors c3 captures it.
    let state = place(&[
        (10, Cell::Enemy(Piece::Paper)),
        (20, Cell::Own(Piece::Scissors)),
        (60, Cell::Own(Piece::Rock)),
        (79, Cell::Enemy(Piece::Rock)),
    ]);
    let legal = state.legal_actions();
    let loss2 = loses_in_two(&RULES, &state, &legal);
    for (&action, &loses) in legal.iter().zip(&loss2) {
        assert_eq!(loses, from_to(action) != (20, 10), "{action}");
    }
    check(&state);
}

#[test]
fn stalemate_and_elimination_are_wins_at_once() {
    // Enemy rock in the corner i1, hemmed in by own papers; a quiet move wins.
    let smother = place(&[
        (8, Cell::Enemy(Piece::Rock)),
        (7, Cell::Own(Piece::Paper)),
        (16, Cell::Own(Piece::Paper)),
        (17, Cell::Own(Piece::Paper)),
        (40, Cell::Own(Piece::Rock)),
    ]);
    assert!(any_win_at_once(&RULES, &smother));
    check(&smother);
    // Own paper e5 captures the last enemy rock f5.
    let last = place(&[
        (40, Cell::Own(Piece::Paper)),
        (41, Cell::Enemy(Piece::Rock)),
    ]);
    assert!(any_win_at_once(&RULES, &last));
    check(&last);
    // The same enemy rock with a second enemy piece far away: no win at once.
    let two = place(&[
        (40, Cell::Own(Piece::Paper)),
        (41, Cell::Enemy(Piece::Rock)),
        (4, Cell::Enemy(Piece::Scissors)),
    ]);
    assert!(!any_win_at_once(&RULES, &two));
    check(&two);
}
