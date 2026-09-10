use std::time::Duration;

use engine::{apply, from_codes, Outcome, Rules, State};
use nnue::net::{convert_v3, Accumulator, Model, FILE_SIZE};
use nnue::search::{Limits, SearchPool, Searcher};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn prototype() -> Vec<u8> {
    let mut bytes = b"RPSNNUE1".to_vec();
    for n in [3u32, 972, 512, 255, 64] {
        bytes.extend(n.to_le_bytes());
    }
    bytes.extend(600f32.to_le_bytes());
    for _ in 0..512 {
        bytes.extend(96i16.to_le_bytes());
    }
    for i in 0..972 * 512 {
        bytes.extend((((i * 13 + i / 512 * 7) % 31) as i16 - 15).to_le_bytes());
    }
    for i in 0..1024 {
        bytes.extend((((i * 17) % 15) as i16 - 7).to_le_bytes());
    }
    bytes.extend(0i32.to_le_bytes());
    bytes.extend(32u32.to_le_bytes());
    for i in 0..32 * 1024 {
        bytes.push(((i * 13) % 255 - 128) as i8 as u8);
    }
    for i in 0..32 {
        bytes.extend(((i - 16) * 255 * 32i32).to_le_bytes());
    }
    for i in 0..32 {
        bytes.extend(((i - 16) as i16).to_le_bytes());
    }
    bytes
}

fn model() -> Model {
    Model::from_bytes(&convert_v3(&prototype()).unwrap()).unwrap()
}

#[test]
fn conversion_preserves_payload_and_rejects_other_schemas() {
    let source = prototype();
    let bytes = convert_v3(&source).unwrap();
    assert_eq!(bytes.len(), FILE_SIZE);
    assert_eq!(&bytes[36..1060], &source[32..1056]);
    assert_eq!(&bytes[1060..996388], &source[1056..996384]);
    assert!(bytes[996388..1029156].iter().all(|&b| b == 0));
    assert_eq!(&bytes[1029156..], &source[996384..]);
    for (offset, value) in [
        (8, 3u32),
        (12, 972),
        (16, 1024),
        (20, 256),
        (24, 65),
        (32, 4),
        (1031208, 16),
    ] {
        let mut bad = bytes.clone();
        bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(Model::from_bytes(&bad).is_err());
    }
    assert!(Model::from_bytes(&bytes[..100]).is_err());
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(Model::from_bytes(&bad).is_err());
    let mut bad = bytes;
    bad[996388] = 1;
    assert!(Model::from_bytes(&bad).is_ok());
    assert!(convert_v3(&bad).is_err());
}

#[test]
fn incremental_and_all_backends_match_thousand_move_trajectory() {
    let mut model = model();
    for (i, weight) in model.weights[972 * 512..].iter_mut().enumerate() {
        *weight = ((i * 7 + i / 512) % 31) as i16 - 15;
    }
    assert_thousand_move_trajectory(&model);
}

fn assert_thousand_move_trajectory(model: &Model) {
    let mut avx2 = (*model).clone();
    avx2.disable_vnni();
    let mut state = State::initial();
    let mut acc = model.refresh(&state.board, state.since_capture, 200);
    let mut rng = StdRng::seed_from_u64(9123);
    let mut captures = 0;
    for _ in 0..1000 {
        assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
        assert_eq!(model.raw(&acc), avx2.raw(&acc));
        let moves = state.legal_actions();
        let captures_available: Vec<_> = moves
            .iter()
            .copied()
            .filter(|&a| state.board[engine::from_to(a).1 as usize].is_enemy())
            .collect();
        let choices = if !captures_available.is_empty() {
            &captures_available
        } else {
            &moves
        };
        let action = choices[rng.random_range(0..choices.len())];
        let (child, outcome) = apply(&Rules::SITE, &state, action);
        captures += usize::from(child.since_capture == 0);
        let mut next = Accumulator::empty(model.hidden);
        model.update(&acc, &state.board, action, child.since_capture, &mut next);
        assert_eq!(next, model.refresh(&child.board, child.since_capture, 200));
        state = child;
        acc = next;
        if outcome != Outcome::Ongoing {
            state = State::initial();
            acc = model.refresh(&state.board, state.since_capture, 200);
        }
    }
    assert!(captures > 20);
}

#[test]
fn terminal_win_precedes_clock_and_limits_retain_a_legal_move() {
    let model = model();
    let mut codes = [0u8; 81];
    codes[70] = 1;
    codes[20] = 4;
    let state = State {
        board: from_codes(&codes).unwrap(),
        since_capture: 199,
        ply: 200,
    };
    let limits = Limits {
        time: Duration::from_secs(2),
        nodes: 100,
        depth: 2,
    };
    let won = Searcher::new(1).search(&model, &state, None, limits);
    assert_eq!(engine::from_to(won.action.unwrap()).1, 80);
    assert_eq!(won.score, 29_999);
    let state = State::initial();
    let allowed = [state.legal_actions()[3]];
    let r = Searcher::new(1).search(
        &model,
        &state,
        Some(&allowed),
        Limits {
            nodes: 100,
            ..limits
        },
    );
    assert_eq!(r.action, Some(allowed[0]));
    assert!(r.nodes <= 101);
    let r = Searcher::new(1).search(
        &model,
        &state,
        None,
        Limits {
            time: Duration::ZERO,
            ..limits
        },
    );
    assert!(state.legal_actions().contains(&r.action.unwrap()));
}

#[test]
fn shared_table_search_and_pv_stay_legal() {
    let model = model();
    let state = State::initial();
    let r = SearchPool::new(2, 4).search(
        &model,
        &state,
        None,
        Limits {
            time: Duration::from_secs(5),
            nodes: 1000,
            depth: 3,
        },
    );
    assert!(r.nodes <= 1004);
    let mut state = state;
    for a in r.pv {
        assert!(state.legal_mask()[a as usize]);
        let (next, end) = apply(&Rules::SITE, &state, a);
        state = next;
        if end != Outcome::Ongoing {
            break;
        }
    }
}

#[test]
fn leaf_paths_replay_to_the_recorded_evaluation() {
    for model in [
        model(),
        Model::from_bytes(include_bytes!("fixtures/h256_race.nnue")).unwrap(),
    ] {
        let path =
            std::env::temp_dir().join(format!("nnue-leaf-test-{}.jsonl", std::process::id()));
        let mut pool = SearchPool::new(1, 1);
        pool.set_leaves(&path).unwrap();
        pool.search(
            &model,
            &State::initial(),
            None,
            Limits {
                nodes: 1000,
                depth: 2,
                time: Duration::from_secs(5),
            },
        );
        assert!(pool.leaf_error().is_none());
        drop(pool);
        let records = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(!records.is_empty());
        for line in records.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let mut state = State::initial();
            for a in v["path"].as_array().unwrap() {
                let (child, end) = apply(&Rules::SITE, &state, a.as_u64().unwrap() as u16);
                assert_eq!(end, Outcome::Ongoing);
                state = child;
            }
            assert_eq!(
                serde_json::json!(engine::codes(&state.board).as_slice()),
                v["board"]
            );
            assert_eq!(
                state.since_capture as u64,
                v["since_capture"].as_u64().unwrap()
            );
            assert!(
                (model.raw(&model.refresh(&state.board, state.since_capture, 200))
                    - v["raw"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
            assert_eq!(
                model.evaluate(&model.refresh(&state.board, state.since_capture, 200)),
                v["score"].as_i64().unwrap() as i32
            );
        }
    }
}

#[test]
fn player_maps_simulations_to_nodes_on_one_worker() {
    use r#match::{Clock, History, Player};
    let path = std::env::temp_dir().join(format!("nnue-player-test-{}.nnue", std::process::id()));
    std::fs::write(&path, convert_v3(&prototype()).unwrap()).unwrap();
    let player = nnue::NnuePlayer::load(path.to_str().unwrap(), 4);
    std::fs::remove_file(path).unwrap();
    let mut player = player.unwrap();
    let state = State::initial();
    let action = player.choose(&state, &History::new(), Clock::Sims(1));
    assert!(state.legal_actions().contains(&action));
    assert!(!player.forfeited());
    assert_eq!(player.info().unwrap().sims, 1);
    let details = player.search_details().unwrap();
    assert!((2500..=2501).contains(&details["nodes"].as_u64().unwrap()));
    player.new_game();
    assert!(player.info().is_none());
}

#[test]
fn cancellation_keeps_a_legal_root_choice() {
    use std::sync::{atomic::AtomicU64, Arc};
    let mut search = Searcher::new(1);
    search.set_stop_signal(Arc::new(AtomicU64::new(1)), 0);
    let state = State::initial();
    let result = search.search(&model(), &state, None, Limits::default());
    assert!(result.aborted);
    assert!(state.legal_actions().contains(&result.action.unwrap()));
    assert!(result.nodes <= 1024);
}

#[test]
fn clock_rows_match_refresh_at_every_boundary_and_capture() {
    let mut model = model();
    for (i, weight) in model.weights[972 * 512..].iter_mut().enumerate() {
        *weight = (i / 512) as i16 - 16;
    }
    let mut codes = [0; 81];
    codes[40] = 1;
    codes[41] = 6;
    codes[20] = 4;
    for clock in [50, 68, 100, 200] {
        for since in 0..clock {
            let state = State {
                board: from_codes(&codes).unwrap(),
                since_capture: since,
                ply: since,
            };
            let acc = model.refresh(&state.board, since, clock);
            let mut next = Accumulator::empty(512);
            for a in state.legal_actions() {
                let (child, _) = apply(
                    &Rules {
                        capture_clock: Some(clock),
                    },
                    &state,
                    a,
                );
                model.update(&acc, &state.board, a, child.since_capture, &mut next);
                assert_eq!(
                    next,
                    model.refresh(&child.board, child.since_capture, clock)
                );
                assert_eq!(model.raw(&next), model.raw_scalar(&next));
            }
        }
    }
}

fn bucket_model() -> Model {
    let one = convert_v3(&prototype()).unwrap();
    let mut bytes = one[..1029156].to_vec();
    bytes[32..36].copy_from_slice(&4u32.to_le_bytes());
    for k in 1..=4i16 {
        for pair in one[1029156..1031204].chunks_exact(2) {
            bytes.extend((i16::from_le_bytes(pair.try_into().unwrap()) * k).to_le_bytes());
        }
    }
    for k in 0..4i32 {
        bytes.extend((64 * k).to_le_bytes());
    }
    bytes.extend(&one[1031208..1064108]);
    for k in 1..=4i16 {
        for pair in one[1064108..].chunks_exact(2) {
            bytes.extend((i16::from_le_bytes(pair.try_into().unwrap()) * k).to_le_bytes());
        }
    }
    assert_eq!(bytes.len(), 1_070_520);
    let model = Model::from_bytes(&bytes).unwrap();
    bytes.pop();
    assert!(Model::from_bytes(&bytes).is_err());
    model
}

#[test]
fn buckets_select_all_three_readouts_and_change_on_capture() {
    let model = bucket_model();
    let mut avx2 = model.clone();
    avx2.disable_vnni();
    for count in 2..=20usize {
        let mut codes = [0; 81];
        codes[40] = 1;
        codes[41] = 6;
        for (i, cell) in codes[10..10 + count - 2].iter_mut().enumerate() {
            *cell = if i % 2 == 0 { 1 } else { 4 };
        }
        let state = State {
            board: from_codes(&codes).unwrap(),
            since_capture: 159,
            ply: 159,
        };
        let acc = model.refresh(&state.board, state.since_capture, 200);
        let bucket = match count {
            2..=4 => 0,
            5..=8 => 1,
            9..=12 => 2,
            _ => 3,
        };
        let mut single = Model::from_bytes(&convert_v3(&prototype()).unwrap()).unwrap();
        single
            .output
            .copy_from_slice(&model.output[bucket * 1024..(bucket + 1) * 1024]);
        single.output_bias[0] = model.output_bias[bucket];
        single
            .dense
            .as_mut()
            .unwrap()
            .output
            .copy_from_slice(&model.dense.as_ref().unwrap().output[bucket * 32..(bucket + 1) * 32]);
        assert_eq!(model.raw(&acc), single.raw(&acc));
        assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
        assert_eq!(model.raw(&acc), avx2.raw(&acc));
        let capture = state
            .legal_actions()
            .into_iter()
            .find(|&a| engine::from_to(a) == (40, 41))
            .unwrap();
        let (child, _) = apply(&Rules::SITE, &state, capture);
        let mut updated = Accumulator::empty(512);
        model.update(
            &acc,
            &state.board,
            capture,
            child.since_capture,
            &mut updated,
        );
        assert_eq!(
            updated,
            model.refresh(&child.board, child.since_capture, 200)
        );
        assert_eq!(model.raw(&updated), model.raw_scalar(&updated));
    }
}

#[test]
fn unfinished_iterations_only_replace_a_completed_move_with_a_searched_move() {
    let model = model();
    let mut state = State::initial();
    let mut rng = StdRng::seed_from_u64(47);
    let mut observed = false;
    for _ in 0..32 {
        let result = Searcher::new(1).search(
            &model,
            &state,
            None,
            Limits {
                time: Duration::from_secs(5),
                nodes: 1500,
                depth: 8,
            },
        );
        if result.partial {
            assert!(result.aborted);
            assert!(state.legal_actions().contains(&result.action.unwrap()));
            if result.depth > 0 {
                let completed = Searcher::new(1).search(
                    &model,
                    &state,
                    None,
                    Limits {
                        time: Duration::from_secs(5),
                        nodes: u64::MAX,
                        depth: result.depth,
                    },
                );
                assert!(!completed.partial && !completed.aborted);
                assert_ne!(result.action, completed.action);
            }
            observed = true;
            break;
        }
        let moves = state.legal_actions();
        let (next, end) = apply(
            &Rules::SITE,
            &state,
            moves[rng.random_range(0..moves.len())],
        );
        state = if end == Outcome::Ongoing {
            next
        } else {
            State::initial()
        };
    }
    assert!(
        observed,
        "fixture must exercise an interrupted root improvement"
    );
}

#[test]
fn python_h768_export_matches_oracle_and_incremental_trajectory() {
    let model = Model::from_bytes(include_bytes!("fixtures/h768_dense.nnue")).unwrap();
    assert_eq!(model.hidden, 768);
    let mut avx2 = model.clone();
    avx2.disable_vnni();
    let fixture = include_str!("fixtures/h768_positions.jsonl");
    let mut output = Vec::new();
    nnue::diagnostic::run(&model, fixture.as_bytes(), &mut output, None, 1).unwrap();
    assert_eq!(String::from_utf8(output).unwrap().lines().count(), 62);
    for line in fixture.lines() {
        let row: nnue::diagnostic::Position = serde_json::from_str(line).unwrap();
        let state = row.state().unwrap();
        let acc = model.refresh(&state.board, state.since_capture, row.clock);
        assert_eq!(model.raw(&acc), avx2.raw(&acc));
    }
    assert_thousand_move_trajectory(&model);
}

#[test]
fn python_format7_export_matches_oracle_and_incremental_trajectory() {
    // Written by nnue.export from a random H256/B4 network with nonzero race
    // rows; the positions carry integer_eval's raw values, which the
    // diagnostic checks against the Rust evaluation.
    let model = Model::from_bytes(include_bytes!("fixtures/h256_race.nnue")).unwrap();
    assert_eq!(
        (model.features, model.hidden, model.buckets),
        (1022, 256, 4)
    );
    let fixture = include_str!("fixtures/h256_race_positions.jsonl");
    let mut output = Vec::new();
    nnue::diagnostic::run(&model, fixture.as_bytes(), &mut output, None, 1).unwrap();
    assert_eq!(String::from_utf8(output).unwrap().lines().count(), 60);
    assert_thousand_move_trajectory(&model);
}

fn width_fixture(hidden: usize, buckets: usize) -> Vec<u8> {
    let mut bytes = b"RPSNNUE1".to_vec();
    for value in [6, 1004, hidden as u32, 255, 64] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend(600f32.to_le_bytes());
    bytes.extend((buckets as u32).to_le_bytes());
    for _ in 0..hidden {
        bytes.extend(96i16.to_le_bytes());
    }
    for i in 0..1004 * hidden {
        bytes.extend((((i * 13 + i / hidden * 7) % 31) as i16 - 15).to_le_bytes());
    }
    for i in 0..buckets * 2 * hidden {
        bytes.extend((((i * 17) % 15) as i16 - 7).to_le_bytes());
    }
    for i in 0..buckets {
        bytes.extend((i as i32 - 2).to_le_bytes());
    }
    bytes.extend(32u32.to_le_bytes());
    for i in 0..32 * 2 * hidden {
        bytes.push(((i * 13) % 255) as u8);
    }
    for i in 0..32 {
        bytes.extend(((i - 16) * 255 * 32i32).to_le_bytes());
    }
    for i in 0..buckets * 32 {
        bytes.extend((if i % 7 == 0 { 0 } else { i as i16 - 16 }).to_le_bytes());
    }
    bytes
}

#[test]
fn width_and_bucket_dimensions_are_dynamic_and_lengths_are_strict() {
    for hidden in [32, 256, 384, 512, 768, 1024, 1056, 2080] {
        let bytes = width_fixture(hidden, 4);
        let model = Model::from_bytes(&bytes).unwrap();
        assert_eq!(model.hidden, hidden);
        let mut avx2 = model.clone();
        avx2.disable_vnni();
        let mut state = State::initial();
        let mut rng = StdRng::seed_from_u64(100);
        for _ in 0..32 {
            let acc = model.refresh(&state.board, state.since_capture, 200);
            assert_eq!(acc.own.len(), hidden);
            assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
            assert_eq!(model.raw(&acc), avx2.raw(&acc));
            let moves = state.legal_actions();
            let a = moves[rng.random_range(0..moves.len())];
            let (child, end) = apply(&Rules::SITE, &state, a);
            let mut updated = Accumulator::empty(hidden);
            model.update(&acc, &state.board, a, child.since_capture, &mut updated);
            assert_eq!(
                updated,
                model.refresh(&child.board, child.since_capture, 200)
            );
            state = if end == Outcome::Ongoing {
                child
            } else {
                State::initial()
            };
        }
        assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(Model::from_bytes(&bad).is_err());
        for width in [0u32, 31, 33, u32::MAX] {
            let mut bad = bytes.clone();
            bad[16..20].copy_from_slice(&width.to_le_bytes());
            assert!(Model::from_bytes(&bad).is_err());
        }
    }
}

#[test]
fn player_hash_option_is_reported_during_real_search() {
    use r#match::{Clock, History, Player};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h768_dense.nnue?hash=2"
    );
    let mut player = nnue::NnuePlayer::load(path, 1).unwrap();
    let state = State::initial();
    let action = player.choose(&state, &History::new(), Clock::Sims(1));
    assert!(state.legal_actions().contains(&action));
    assert_eq!(player.search_details().unwrap()["hash_mib"], 2);
    assert!(player.name().ends_with("?hash=2"));
}
