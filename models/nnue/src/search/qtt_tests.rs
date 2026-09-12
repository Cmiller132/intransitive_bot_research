use super::*;

const DENSE: &[u8] = include_bytes!("../../tests/fixtures/h768_dense.nnue");
const RACE: &[u8] = include_bytes!("../../tests/fixtures/h256_race.nnue");

fn prepared(model: &Model, state: &State, ply: usize) -> Searcher {
    let mut search = Searcher::new(1);
    search.limits.time = Duration::MAX;
    search.acc = (0..MAX_PLY + 2)
        .map(|_| Accumulator::empty(model.hidden))
        .collect();
    search.acc[ply] = model.refresh(&state.board, state.since_capture, 200);
    search.hashes[ply] = [
        board_hash(&state.board),
        board_hash(&engine::flip(&state.board)),
    ];
    search
}

fn entry(state: &State, depth: i16, bound: u8, score: i32) -> Entry {
    Entry {
        key: position_key(board_hash(&state.board), state.since_capture),
        depth,
        bound,
        score,
        action: NO_MOVE,
        ..Entry::default()
    }
}

#[test]
fn horizon_tags_are_negative_and_distinct() {
    let mut tags = std::collections::HashSet::new();
    for ply in 0..MAX_PLY {
        for remaining in -(MAX_PLY as i32)..=6 {
            let tag = quiescence_depth(ply, remaining);
            assert!(tag < 0);
            assert!(tags.insert(tag));
        }
    }
}

#[test]
fn probes_require_matching_horizons_and_valid_bounds() {
    let model = Model::from_bytes(DENSE).unwrap();
    let state = State::initial();
    let stand = model.evaluate(&model.refresh(&state.board, 0, 200));
    let fake = stand + 100;
    for (bound, alpha, beta, hit) in [
        (1, -INF, INF, true),
        (2, fake - 2, fake - 1, true),
        (2, -INF, INF, false),
        (3, fake + 1, fake + 2, true),
        (3, -INF, INF, false),
        (0, -INF, INF, false),
    ] {
        let mut search = prepared(&model, &state, 0);
        search
            .tt
            .store(entry(&state, quiescence_depth(0, 0), bound, fake));
        assert_eq!(
            search.quiescence(&model, &state, alpha, beta, 0, 0),
            if hit { fake } else { stand }
        );
    }
    for depth in [1, 64, quiescence_depth(0, 1), quiescence_depth(1, 0)] {
        let mut search = prepared(&model, &state, 0);
        search.tt.store(entry(&state, depth, 1, fake));
        assert_eq!(search.quiescence(&model, &state, -INF, INF, 0, 0), stand);
    }
    let mut changed_clock = state.clone();
    changed_clock.since_capture = 1;
    let mut search = prepared(&model, &changed_clock, 0);
    search
        .tt
        .store(entry(&state, quiescence_depth(0, 0), 1, fake));
    assert_eq!(
        search.quiescence(&model, &changed_clock, -INF, INF, 0, 0),
        model.evaluate(&model.refresh(&state.board, 1, 200))
    );
}

#[test]
fn stored_stand_pat_bounds_use_the_original_window() {
    let model = Model::from_bytes(DENSE).unwrap();
    let state = State::initial();
    let stand = model.evaluate(&model.refresh(&state.board, 0, 200));
    for (alpha, beta, bound, remaining) in [
        (-INF, INF, 1, 0),
        (stand - 1, stand, 2, 6),
        (stand, stand + 1, 3, 0),
    ] {
        let mut search = prepared(&model, &state, 0);
        assert_eq!(
            search.quiescence(&model, &state, alpha, beta, 0, remaining),
            stand
        );
        let stored = search.tt.get(position_key(board_hash(&state.board), 0));
        assert_eq!(
            (stored.bound, stored.score, stored.depth),
            (bound, stand, quiescence_depth(0, remaining))
        );
        assert!(stored.static_valid);
        assert_eq!(stored.static_eval, stand);
        assert_eq!(stored.action, NO_MOVE);
    }
}

#[test]
fn full_search_bounds_survive_and_quiet_hash_moves_stay_outside_quiescence() {
    let model = Model::from_bytes(DENSE).unwrap();
    let state = State::initial();
    for table in [Table::local(1 << 20), Table::shared(1 << 20)] {
        let mut search = prepared(&model, &state, 0);
        search.tt = table;
        let mut full = entry(&state, 4, 1, 12345);
        full.action = state.legal_actions()[0];
        search.tt.store(full);
        let stand = model.evaluate(&search.acc[0]);
        assert_eq!(search.quiescence(&model, &state, -INF, INF, 0, 6), stand);
        assert_eq!(search.qnodes, 1);
        assert_eq!(search.tt.get(full.key).depth, 4);

        search.tt.clear();
        search
            .tt
            .store(entry(&state, quiescence_depth(0, 6), 1, 12345));
        search.nodes = 0;
        let score = search.negamax(&model, &state, 1, -INF, INF, 0, false);
        assert_ne!(score, 12345);
        assert!(search.nodes > 1);
        assert_eq!(search.tt.get(full.key).depth, 1);
    }
}

#[test]
fn negative_tags_have_equal_replacement_priority() {
    let state = State::initial();
    for mut table in [Table::local(1), Table::shared(1)] {
        let first = entry(&state, quiescence_depth(0, 6), 1, 10);
        table.store(first);
        let second = entry(&state, quiescence_depth(30, -20), 2, 20);
        table.store(second);
        assert_eq!(table.get(second.key).score, 20);
        let collision = Entry {
            key: second.key ^ 1,
            ..first
        };
        table.store(collision);
        assert_eq!(table.get(collision.key).key, collision.key);
    }
}

#[test]
fn goal_evasions_extend_zero_horizon_and_interruptions_do_not_store_bounds() {
    let model = Model::from_bytes(DENSE).unwrap();
    let mut state = State {
        board: [Cell::Empty; 81],
        since_capture: 199,
        ply: 0,
    };
    state.board[60] = Cell::Own(engine::Piece::Rock);
    state.board[10] = Cell::Enemy(engine::Piece::Paper);
    assert!(enemy_goal_threat(&state.board));
    // At clock 199 every quiet evasion ends by the engine's clock rule.
    let mut search = prepared(&model, &state, 0);
    assert_eq!(search.quiescence(&model, &state, -INF, INF, 0, 0), 0);
    state.since_capture = 0;
    let mut search = prepared(&model, &state, 0);
    assert_eq!(
        search.quiescence(&model, &state, -INF, INF, 0, 0),
        -MATE + 2
    );
    let saved = search.tt.get(position_key(board_hash(&state.board), 0));
    assert_eq!((saved.bound, from_tt(saved.score, 0)), (1, -MATE + 2));
    assert_ne!(saved.action, NO_MOVE);

    state.board[60] = Cell::Empty;
    state.board[79] = Cell::Own(engine::Piece::Rock);
    let mut search = prepared(&model, &state, 0);
    search
        .tt
        .store(entry(&state, quiescence_depth(0, 0), 1, -100));
    assert_eq!(search.quiescence(&model, &state, -INF, INF, 0, 0), MATE - 1);
    search.limits.nodes = search.nodes + 1;
    assert_eq!(search.quiescence(&model, &state, -INF, INF, 0, 0), 0);
    assert_eq!(
        search
            .tt
            .get(position_key(board_hash(&state.board), 0))
            .score,
        -100
    );

    // A real recursive capture is interrupted after its parent was evaluated.
    state.board.fill(Cell::Empty);
    state.board[40] = Cell::Own(engine::Piece::Rock);
    state.board[41] = Cell::Enemy(engine::Piece::Scissors);
    state.board[60] = Cell::Enemy(engine::Piece::Paper);
    let mut search = prepared(&model, &state, 0);
    search.limits.nodes = 2;
    search.quiescence(&model, &state, -INF, INF, 0, 6);
    assert!(search.stopped);
    assert_eq!(
        search
            .tt
            .get(position_key(board_hash(&state.board), 0))
            .bound,
        0
    );
}

#[test]
fn safety_frontier_does_not_reuse_or_store_quiescence_bounds() {
    let model = Model::from_bytes(DENSE).unwrap();
    let state = State::initial();
    let mut search = prepared(&model, &state, MAX_PLY);
    let original = entry(&state, quiescence_depth(MAX_PLY - 1, 0), 1, 12345);
    search.tt.store(original);
    let stand = model.evaluate(&search.acc[MAX_PLY]);
    assert_eq!(
        search.quiescence(&model, &state, -INF, INF, MAX_PLY, 0),
        stand
    );
    assert_eq!(search.tt.get(original.key).depth, original.depth);
    for ply in [0, 1, 60, 119] {
        for score in [-MATE + 121, -50, 0, 50, MATE - 121] {
            assert_eq!(from_tt(to_tt(score, ply), ply), score);
        }
    }
}

#[test]
fn fixture_fixed_node_signatures_are_repeatable() {
    let golden = [
        vec![
            (Some(604), 29, 2, 4000, 3635, Some((516, 36, 2))),
            (Some(576), -39, 3, 4000, 3543, Some((576, -39, 3))),
            (Some(348), -41, 2, 4000, 3838, Some((348, -41, 2))),
            (Some(588), 13, 2, 4000, 3658, Some((507, -42, 2))),
        ],
        vec![
            (Some(37), -215, 3, 4000, 2231, Some((37, -215, 3))),
            (Some(10), -140, 3, 4000, 3178, Some((10, -140, 3))),
            (Some(524), -66, 2, 4000, 1934, Some((37, 343, 2))),
            (Some(523), -202, 2, 4000, 1822, Some((37, 392, 2))),
        ],
    ];
    for (bytes, golden) in [DENSE, RACE].into_iter().zip(golden) {
        let model = Model::from_bytes(bytes).unwrap();
        let mut signatures = Vec::new();
        for line in include_str!("../../tests/fixtures/h768_positions.jsonl")
            .lines()
            .take(4)
        {
            let position: crate::diagnostic::Position = serde_json::from_str(line).unwrap();
            let state = position.state().unwrap();
            let mut search = Searcher::new(1);
            let limits = Limits {
                nodes: 4000,
                time: Duration::MAX,
                ..Limits::default()
            };
            let signature = |r: SearchResult| {
                (
                    r.action,
                    r.score,
                    r.depth,
                    r.nodes,
                    r.qnodes,
                    r.completed_root(),
                )
            };
            let expected = signature(search.search(&model, &state, None, limits));
            search.clear();
            assert_eq!(
                signature(search.search(&model, &state, None, limits)),
                expected
            );
            signatures.push(expected);
        }
        assert_eq!(signatures, golden);
    }
}

#[test]
fn fixture_quiescence_scores_survive_warm_tables_and_horizon_changes() {
    for bytes in [DENSE, RACE] {
        let model = Model::from_bytes(bytes).unwrap();
        for line in include_str!("../../tests/fixtures/h768_positions.jsonl")
            .lines()
            .take(12)
        {
            let position: crate::diagnostic::Position = serde_json::from_str(line).unwrap();
            let state = position.state().unwrap();
            let mut warm = prepared(&model, &state, 0);
            for remaining in [6, 4, 5, 0, -1] {
                let mut cold = prepared(&model, &state, 0);
                let expected = cold.quiescence(&model, &state, -INF, INF, 0, remaining);
                assert_eq!(
                    warm.quiescence(&model, &state, -INF, INF, 0, remaining),
                    expected
                );
                assert_eq!(
                    warm.quiescence(&model, &state, -INF, INF, 0, remaining),
                    expected
                );
                assert!(!warm.stopped && !cold.stopped);
            }
        }
    }
}
