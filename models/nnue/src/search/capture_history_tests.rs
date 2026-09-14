use super::*;
use engine::Piece::{Paper, Rock, Scissors};

fn position() -> State {
    let mut state = State {
        board: [Cell::Empty; 81],
        since_capture: 0,
        ply: 0,
    };
    state.board[40] = Cell::Own(Rock);
    state.board[41] = Cell::Enemy(Scissors);
    state.board[49] = Cell::Enemy(Scissors);
    state.board[60] = Cell::Enemy(Paper);
    state
}

fn action(from: u8, to: u8) -> u16 {
    (0..8)
        .find_map(|dir| (engine::step(from, dir) == Some(to)).then_some(engine::action(dir, from)))
        .unwrap()
}

#[test]
fn keys_separate_side_attacker_destination_and_victim_but_share_origins() {
    let state = position();
    let a = action(40, 41);
    let key = capture_history_index(&state, a).unwrap();
    let mut changed = state.clone();
    changed.ply = 1;
    assert_ne!(capture_history_index(&changed, a).unwrap(), key);
    changed = state.clone();
    changed.board[40] = Cell::Own(Paper);
    assert_ne!(capture_history_index(&changed, a).unwrap(), key);
    changed = state.clone();
    changed.board[41] = Cell::Enemy(Rock);
    assert_ne!(capture_history_index(&changed, a).unwrap(), key);
    assert_ne!(capture_history_index(&state, action(40, 49)).unwrap(), key);
    changed = state.clone();
    changed.board[32] = Cell::Own(Rock);
    assert_eq!(capture_history_index(&changed, action(32, 41)), Some(key));
    assert_eq!(capture_history_index(&state, action(40, 39)), None);
    for side in 0..2 {
        for attacker in [Rock, Paper, Scissors] {
            for victim in [Rock, Paper, Scissors] {
                changed.ply = side;
                changed.board[40] = Cell::Own(attacker);
                changed.board[41] = Cell::Enemy(victim);
                assert!(capture_history_index(&changed, a).unwrap() < CAPTURE_HISTORY_SIZE);
            }
        }
    }
}

#[test]
fn cutoff_updates_only_winner_and_searched_captures_with_bounded_gravity() {
    let state = position();
    let win = action(40, 41);
    let loss = action(40, 49);
    let quiet = action(40, 39);
    let w = capture_history_index(&state, win).unwrap();
    let l = capture_history_index(&state, loss).unwrap();
    let mut search = Searcher::new(1);
    search.update_captures(&state, win, &[quiet, loss], 3);
    assert_eq!(search.capture_history[w], 288);
    assert_eq!(search.capture_history[l], -288);
    assert_eq!(
        search.capture_history.iter().filter(|&&v| v != 0).count(),
        2
    );
    assert!(search.history.iter().flatten().all(|&v| v == 0));
    search.clear();
    search.update_captures(&state, quiet, &[loss], 3);
    assert_eq!(search.capture_history[w], 0);
    assert_eq!(search.capture_history[l], -288);
    for _ in 0..1000 {
        search.update_captures(&state, win, &[loss], 120);
        assert!(search.capture_history.iter().all(|v| v.abs() <= 16384));
    }
    search.clear();
    assert!(search.capture_history.iter().all(|&v| v == 0));
}

#[test]
fn ordering_uses_capture_history_and_keeps_tt_and_goal_priority() {
    let mut state = position();
    let a = action(40, 41);
    let b = action(40, 49);
    let mut search = Searcher::new(1);
    search.capture_history[capture_history_index(&state, b).unwrap()] = 16000;
    search.history[0][a as usize] = 16384;
    let mut actions = [a, b];
    search.order(&state, &mut actions, NO_MOVE, 0);
    assert_eq!(actions, [b, a]);
    search.order(&state, &mut actions, a, 0);
    assert_eq!(actions, [a, b]);
    state.board[70] = Cell::Own(Rock);
    let goal = action(70, 80);
    let mut actions = [b, goal];
    search.order(&state, &mut actions, NO_MOVE, 0);
    assert_eq!(actions, [goal, b]);
}

#[test]
fn main_search_trains_captures_but_interruption_and_quiescence_do_not() {
    let model = Model::from_bytes(include_bytes!("../../tests/fixtures/h768_dense.nnue")).unwrap();
    let state = position();
    let mut search = Searcher::new(1);
    let limits = Limits {
        nodes: 1,
        time: Duration::MAX,
        ..Limits::default()
    };
    search.search(&model, &state, None, limits);
    assert!(search.capture_history.iter().all(|&v| v == 0));
    search.nodes = 0;
    search.stopped = false;
    search.limits.nodes = 100_000;
    search.negamax(&model, &state, 1, -INF, -MATE, 0, false);
    assert!(!search.stopped);
    assert!(search.capture_history.iter().any(|&v| v > 0));
    let learned = search.capture_history;
    search.quiescence(&model, &state, -INF, INF, 0, 6);
    assert_eq!(search.capture_history, learned);
    search.search(&model, &state, None, limits);
    assert_eq!(search.capture_history, learned.map(|v| v / 2));
}
