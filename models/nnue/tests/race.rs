use engine::{apply, Cell, Outcome, Piece, Rules, State};
use nnue::net::{Accumulator, Model};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_file(hidden: usize, features: usize, buckets: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(2026091003);
    let mut bytes = b"RPSNNUE1".to_vec();
    for n in [
        if features == 1004 { 6 } else { 7 },
        features,
        hidden,
        255,
        64,
    ] {
        bytes.extend((n as u32).to_le_bytes());
    }
    bytes.extend(600f32.to_le_bytes());
    bytes.extend((buckets as u32).to_le_bytes());
    for _ in 0..hidden {
        bytes.extend(rng.random_range(32i16..128).to_le_bytes());
    }
    for _ in 0..features * hidden {
        bytes.extend(rng.random_range(-16i16..16).to_le_bytes());
    }
    for _ in 0..buckets * 2 * hidden {
        bytes.extend(rng.random_range(-64i16..64).to_le_bytes());
    }
    for _ in 0..buckets {
        bytes.extend(rng.random_range(-10000i32..10000).to_le_bytes());
    }
    bytes.extend(32u32.to_le_bytes());
    for _ in 0..32 * 2 * hidden {
        bytes.push(rng.random::<u8>());
    }
    for _ in 0..32 {
        bytes.extend(rng.random_range(-10000i32..10000).to_le_bytes());
    }
    for _ in 0..buckets * 32 {
        bytes.extend(rng.random_range(-64i16..64).to_le_bytes());
    }
    bytes
}

#[test]
fn format7_random_file_refresh_incremental_and_backends() {
    for (hidden, buckets) in [(512, 1), (768, 4)] {
        let bytes = random_file(hidden, 1022, buckets);
        let path =
            std::env::temp_dir().join(format!("nnue-race-{}-{hidden}.nnue", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let model = Model::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(model.features, 1022);
        assert_eq!(model.weights.len(), hidden * 1022);
        let mut avx2 = model.clone();
        avx2.disable_vnni();
        let mut rng = StdRng::seed_from_u64(2026091004);
        let mut state = State::initial();
        let mut captures = 0;
        let mut goals = 0;
        for turn in 0..2000 {
            if turn % 100 == 0 {
                state = State::initial();
                state.board = [Cell::Empty; 81];
                state.board[60] = Cell::Own(Piece::Rock);
                state.board[20] = Cell::Enemy(Piece::Scissors);
            }
            let clock = [50, 68, 200][turn / 100 % 3];
            let acc = model.refresh(&state.board, state.since_capture, clock);
            assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
            assert_eq!(model.raw(&acc), avx2.raw(&acc));
            let legal = state.legal_actions();
            let action = if turn % 3 == 0 {
                *legal
                    .iter()
                    .max_by_key(|&&a| {
                        let to = engine::from_to(a).1;
                        (to / 9).min(to % 9)
                    })
                    .unwrap()
            } else {
                legal[rng.random_range(0..legal.len())]
            };
            let (child, end) = apply(
                &Rules {
                    capture_clock: Some(clock),
                },
                &state,
                action,
            );
            captures += usize::from(child.since_capture == 0);
            goals += usize::from(engine::from_to(action).1 == 80);
            let mut incremental = Accumulator::empty(hidden);
            model.update(
                &acc,
                &state.board,
                action,
                child.since_capture,
                &mut incremental,
            );
            assert_eq!(
                incremental,
                model.refresh(&child.board, child.since_capture, clock)
            );
            state = if end == Outcome::Ongoing {
                child
            } else {
                State::initial()
            };
        }
        assert!(captures > 10 && goals > 5);
        for (offset, value) in [(8, 6u32), (8, 8), (12, 1004), (12, 1021), (12, 1023)] {
            let mut bad = bytes.clone();
            bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(Model::from_bytes(&bad).is_err());
        }
        assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }
}

#[test]
fn race_row_order_keeps_actual_tempo_in_both_perspectives() {
    let mut model = Model::from_bytes(&random_file(32, 1022, 1)).unwrap();
    model.bias.fill(0);
    model.weights.fill(0);
    for row in 1004..1022 {
        model.weights[row * 32] = (row - 1003) as i16;
    }
    let mut state = State::initial();
    state.board = [Cell::Empty; 81];
    state.board[70] = Cell::Own(Piece::Rock);
    state.board[71] = Cell::Enemy(Piece::Paper);
    assert_eq!(engine::race::race_buckets(&state.board).0, 0);
    assert_eq!(engine::race::race_buckets(&engine::flip(&state.board)).1, 8);
    for _ in 0..100 {
        let (m, o) = engine::race::race_buckets(&state.board);
        // Separate coordinates distinguish side groups, even when bucket sums match.
        for row in 1013..1022 {
            model.weights[row * 32] = 0;
            model.weights[row * 32 + 1] = (row - 1012) as i16;
        }
        let acc = model.refresh(&state.board, state.since_capture, 200);
        assert_eq!(&acc.own[..2], &[m as i32 + 1, o as i32 + 1]);
        assert_eq!(&acc.opponent[..2], &[o as i32 + 1, m as i32 + 1]);
        let action = state.legal_actions()[0];
        let (child, end) = apply(&Rules::SITE, &state, action);
        state = if end == Outcome::Ongoing {
            child
        } else {
            State::initial()
        };
    }
}

#[test]
fn zero_race_rows_preserve_format6_evaluations() {
    let original = random_file(512, 1004, 1);
    let mut widened = original.clone();
    widened[8..12].copy_from_slice(&7u32.to_le_bytes());
    widened[12..16].copy_from_slice(&1022u32.to_le_bytes());
    let at = 36 + 2 * 512 * 1005;
    widened.splice(at..at, vec![0; 18 * 512 * 2]);
    let six = Model::from_bytes(&original).unwrap();
    let seven = Model::from_bytes(&widened).unwrap();
    let mut state = State::initial();
    let mut rng = StdRng::seed_from_u64(2026091005);
    for _ in 0..1000 {
        let a = six.refresh(&state.board, state.since_capture, 200);
        let b = seven.refresh(&state.board, state.since_capture, 200);
        assert_eq!(a.own, b.own);
        assert_eq!(a.opponent, b.opponent);
        assert_eq!(six.raw(&a), seven.raw(&b));
        let legal = state.legal_actions();
        let (child, end) = apply(
            &Rules::SITE,
            &state,
            legal[rng.random_range(0..legal.len())],
        );
        state = if end == Outcome::Ongoing {
            child
        } else {
            State::initial()
        };
    }
}

#[test]
fn deferred_search_matches_eager_snapshot_at_twenty_thousand_nodes() {
    let mut bytes = random_file(512, 1004, 1);
    let random = random_file(512, 1022, 1);
    let at = 36 + 2 * 512 * 1005;
    bytes[8..12].copy_from_slice(&7u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&1022u32.to_le_bytes());
    bytes.splice(at..at, random[at..at + 18 * 512 * 2].iter().copied());
    let model = Model::from_bytes(&bytes).unwrap();
    // Positions and full search outputs from the eager format-7 binary.
    let fixture: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("fixtures/race_search.json")).unwrap();
    assert_eq!(fixture.len(), 100);
    let input = fixture
        .iter()
        .map(|r| r["position"].to_string() + "\n")
        .collect::<String>();
    let mut output = Vec::new();
    nnue::diagnostic::run(
        &model,
        std::io::Cursor::new(input),
        &mut output,
        Some(20_000),
        1,
    )
    .unwrap();
    let output = String::from_utf8(output).unwrap();
    assert_eq!(output.lines().count(), fixture.len());
    for (line, row) in output.lines().zip(fixture) {
        let mut actual: serde_json::Value = serde_json::from_str(line).unwrap();
        actual.as_object_mut().unwrap().remove("elapsed_ms");
        assert_eq!(actual, row["expected"]);
    }
}
