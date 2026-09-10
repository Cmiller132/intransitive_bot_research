//! Rule invariants: initial position, legality, capture cycle, each end
//! condition in precedence order, frame flip round trip.

use engine::notation::{action_name, parse_action};
use engine::{
    action, apply, flip, from_to, Action, Cell, Outcome, Piece, Rules, State, N_ACTIONS, N_SQUARES,
};

fn place(cells: &[(usize, Cell)]) -> State {
    let mut state = State::initial();
    state.board = [Cell::Empty; N_SQUARES];
    for &(square, cell) in cells {
        state.board[square] = cell;
    }
    state
}

#[test]
fn staged_move_masks_match_the_rule_mask() {
    let mut seed = 0x2026091015u64;
    for _ in 0..1000 {
        let mut state = State::initial();
        for cell in &mut state.board {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *cell = Cell::from_code((seed % 7) as u8).unwrap();
        }
        let mask = state.legal_mask();
        let expected: Vec<_> = (0..N_ACTIONS as Action)
            .filter(|&a| mask[a as usize])
            .collect();
        let moves = state.legal_moves();
        assert_eq!(moves.len(), expected.len());
        assert_eq!(moves.is_empty(), expected.is_empty());
        let mut all = Vec::new();
        moves.all_into(&mut all);
        assert_eq!(all, expected);
        let mut captures = Vec::new();
        let mut quiets = Vec::new();
        moves.captures_into(&mut captures);
        moves.quiets_into(u128::MAX, &mut quiets);
        assert!(captures
            .iter()
            .all(|&a| state.board[from_to(a).1 as usize].is_enemy()));
        assert!(quiets
            .iter()
            .all(|&a| state.board[from_to(a).1 as usize] == Cell::Empty));
        captures.extend(quiets);
        captures.sort_unstable();
        assert_eq!(captures, expected);
        for action in 0..N_ACTIONS as Action {
            assert_eq!(moves.contains(action), mask[action as usize]);
        }
        assert!(!moves.contains(u16::MAX));
        let mut goals = Vec::new();
        moves.quiets_into(1 << 80, &mut goals);
        assert_eq!(
            goals,
            expected
                .iter()
                .copied()
                .filter(|&a| from_to(a).1 == 80 && state.board[80] == Cell::Empty)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn initial_position_has_twenty_pieces_and_blue_moves_first() {
    let state = State::initial();
    assert_eq!(state.own_count(), 10);
    assert_eq!(state.enemy_count(), 10);
    assert_eq!(state.ply, 0);
    assert_eq!(state.since_capture, 0);
    assert_eq!(state.board[28], Cell::Own(Piece::Rock)); // b4
    assert_eq!(state.board[13], Cell::Own(Piece::Paper)); // e2
    assert_eq!(state.board[22], Cell::Own(Piece::Scissors)); // e3
    assert_eq!(state.board[80 - 28], Cell::Enemy(Piece::Rock));
}

#[test]
fn legal_moves_are_king_steps_onto_empty_or_beaten_pieces() {
    // Own rock on e5 (40) surrounded by enemy rock (d5), enemy paper (f5), enemy scissors (e6),
    // own paper (e4); the rest empty.
    let state = place(&[
        (40, Cell::Own(Piece::Rock)),
        (39, Cell::Enemy(Piece::Rock)),
        (41, Cell::Enemy(Piece::Paper)),
        (49, Cell::Enemy(Piece::Scissors)),
        (31, Cell::Own(Piece::Paper)),
    ]);
    let mask = state.legal_mask();
    let rock_moves: Vec<Action> = (0..N_ACTIONS as Action)
        .filter(|&a| mask[a as usize] && from_to(a).0 == 40)
        .collect();
    let targets: Vec<u8> = rock_moves.iter().map(|&a| from_to(a).1).collect();
    assert!(targets.contains(&49), "rock captures scissors");
    assert!(!targets.contains(&39), "rock does not capture rock");
    assert!(!targets.contains(&41), "rock does not capture paper");
    assert!(!targets.contains(&31), "no move onto an own piece");
    assert_eq!(targets.len(), 5, "four empty neighbours plus the capture");
    assert_eq!(action_name(action(4, 40)), "e5-f5");
    assert_eq!(parse_action("Re5xf5"), Some(action(4, 40)));
}

#[test]
fn goal_entry_wins_before_any_other_end() {
    let state = place(&[(71, Cell::Own(Piece::Rock)), (0, Cell::Enemy(Piece::Paper))]);
    let rules = Rules {
        capture_clock: Some(1),
    };
    let goal = (0..N_ACTIONS as Action)
        .find(|&a| state.legal_mask()[a as usize] && from_to(a).1 == 80)
        .unwrap();
    let (child, outcome) = apply(&rules, &state, goal);
    assert_eq!(outcome, Outcome::Win);
    assert_eq!(child.ply, 1);
    assert_eq!(
        child.since_capture, 1,
        "the clock would have drawn, but the goal wins first"
    );
}

#[test]
fn capturing_the_last_enemy_piece_wins() {
    let state = place(&[
        (40, Cell::Own(Piece::Paper)),
        (41, Cell::Enemy(Piece::Rock)),
    ]);
    let capture = (0..N_ACTIONS as Action)
        .find(|&a| state.legal_mask()[a as usize] && from_to(a).1 == 41)
        .unwrap();
    let (child, outcome) = apply(&Rules::SITE, &state, capture);
    assert_eq!(outcome, Outcome::Win);
    assert_eq!(child.since_capture, 0);
    assert_eq!(
        child.enemy_count(),
        1,
        "the winner's piece is the enemy in the child's frame"
    );
}

#[test]
fn leaving_the_opponent_without_moves_wins() {
    // Enemy rock in the corner i1 (8), hemmed in by own papers on h1 (7), h2 (16), i2 (17);
    // an own rock elsewhere makes the quiet move.
    let state = place(&[
        (8, Cell::Enemy(Piece::Rock)),
        (7, Cell::Own(Piece::Paper)),
        (16, Cell::Own(Piece::Paper)),
        (17, Cell::Own(Piece::Paper)),
        (40, Cell::Own(Piece::Rock)),
    ]);
    let quiet = (0..N_ACTIONS as Action)
        .find(|&a| state.legal_mask()[a as usize] && from_to(a).0 == 40)
        .unwrap();
    let (child, outcome) = apply(&Rules::SITE, &state, quiet);
    assert_eq!(outcome, Outcome::Win);
    assert!(child.is_stalemated());
}

#[test]
fn capture_clock_draws_at_the_configured_ply_and_never_when_disabled() {
    let mut state = place(&[(40, Cell::Own(Piece::Rock)), (0, Cell::Enemy(Piece::Paper))]);
    state.since_capture = 199;
    let quiet = state.legal_actions()[0];
    let (child, outcome) = apply(&Rules::SITE, &state, quiet);
    assert_eq!(outcome, Outcome::Draw);
    assert_eq!(child.since_capture, 200);
    let (_, outcome) = apply(
        &Rules {
            capture_clock: None,
        },
        &state,
        quiet,
    );
    assert_eq!(outcome, Outcome::Ongoing);
    state.since_capture = 198;
    let (_, outcome) = apply(&Rules::SITE, &state, quiet);
    assert_eq!(outcome, Outcome::Ongoing);
}

#[test]
fn flipping_twice_is_the_identity() {
    let board = State::initial().board;
    assert_eq!(flip(&flip(&board)), board);
    assert_eq!(flip(&board), board, "the start position is symmetric");
    let state = place(&[
        (8, Cell::Own(Piece::Scissors)),
        (17, Cell::Enemy(Piece::Rock)),
    ]);
    let to_h1 = action(3, 8);
    assert!(state.legal_mask()[to_h1 as usize]);
    let (child, _) = apply(&Rules::SITE, &state, to_h1);
    assert_eq!(
        child.board[7],
        Cell::Own(Piece::Rock),
        "the enemy rock is now own, mirrored"
    );
    assert_eq!(child.board[17], Cell::Enemy(Piece::Scissors));
}
