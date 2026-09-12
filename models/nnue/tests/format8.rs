use engine::{apply, Cell, Outcome, Rules, State};
use nnue::net::{Accumulator, Model};
use rand::{rngs::StdRng, Rng, SeedableRng};

const V8: &[u8] = include_bytes!("fixtures/format8_h32.nnue");
const V6: &[u8] = include_bytes!("fixtures/format6_h32.nnue");
const POSITIONS: &str = include_str!("fixtures/format8_positions.jsonl");

#[test]
fn shared_fixture_ids_raws_and_backends_match_both_formats() {
    for bytes in [V6, V8] {
        let model = Model::from_bytes(bytes).unwrap();
        let mut avx2 = model.clone();
        avx2.disable_vnni();
        let mut output = Vec::new();
        nnue::diagnostic::run(&model, POSITIONS.as_bytes(), &mut output, None, 1).unwrap();
        assert_eq!(String::from_utf8(output).unwrap().lines().count(), 512);
        for line in POSITIONS.lines() {
            let row: nnue::diagnostic::Position = serde_json::from_str(line).unwrap();
            let state = row.state().unwrap();
            let acc = model.refresh(&state.board, state.since_capture, row.clock);
            assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
            assert_eq!(model.raw(&acc), avx2.raw(&acc));
        }
    }
}

#[test]
fn both_readers_enforce_the_entire_header_and_exact_size() {
    for bytes in [V6, V8] {
        let h = 32;
        let f = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 236 + h * (2 * f + 70));
        for (at, values) in [
            (8, vec![3, 7, 9]),
            (12, vec![972, 1022, 0]),
            (16, vec![0, 31, 33, 1056, u32::MAX]),
            (20, vec![0, 256]),
            (24, vec![0, 65]),
            (28, vec![f32::NAN.to_bits(), 601f32.to_bits()]),
            (32, vec![0, 2, 4]),
            (36 + 2 * h + 2 * f * h + 4 * h + 4, vec![0, 31, 33]),
        ] {
            for value in values {
                let mut bad = bytes.to_vec();
                bad[at..at + 4].copy_from_slice(&value.to_le_bytes());
                assert!(
                    Model::from_bytes(&bad).is_err(),
                    "offset {at}, value {value}"
                );
            }
        }
        let mut bad = bytes.to_vec();
        bad[0] = 0;
        assert!(Model::from_bytes(&bad).is_err());
        for length in [0, 7, 35, bytes.len() - 1] {
            assert!(Model::from_bytes(&bytes[..length]).is_err());
        }
        let mut bad = bytes.to_vec();
        bad.push(0);
        assert!(Model::from_bytes(&bad).is_err());
    }
}

#[test]
fn malformed_oracle_fields_are_rejected() {
    let model = Model::from_bytes(V8).unwrap();
    let original: serde_json::Value =
        serde_json::from_str(POSITIONS.lines().next().unwrap()).unwrap();
    for field in ["ply", "context", "ids", "raw"] {
        let mut bad = original.clone();
        match field {
            "ply" => {
                bad.as_object_mut().unwrap().remove("ply");
            }
            "context" => bad["context"][0] = serde_json::json!(99),
            "ids" => bad["ids"][0][0] = bad["ids"][0][1].clone(),
            _ => bad["raw"] = serde_json::json!(99.),
        }
        assert!(
            nnue::diagnostic::run(&model, bad.to_string().as_bytes(), Vec::new(), None, 1).is_err(),
            "{field}"
        );
    }
}

#[test]
fn incremental_halves_match_refresh_across_games_and_context_boundaries() {
    let model = Model::from_bytes(V8).unwrap();
    let mut rng = StdRng::seed_from_u64(2026091223);
    let mut state = State::initial();
    let mut acc = model.refresh(&state.board, 0, 200);
    let mut captures = 0;
    let mut boundaries = 0;
    for _ in 0..2000 {
        let legal = state.legal_actions();
        let attacks: Vec<_> = legal
            .iter()
            .copied()
            .filter(|&a| state.board[engine::from_to(a).1 as usize] != Cell::Empty)
            .collect();
        let choices = if attacks.is_empty() { &legal } else { &attacks };
        let action = choices[rng.random_range(0..choices.len())];
        let (child, end) = apply(&Rules::SITE, &state, action);
        let mut next = Accumulator::empty(model.hidden);
        model.update(&acc, &state.board, action, child.since_capture, &mut next);
        assert_eq!(next, model.refresh(&child.board, child.since_capture, 200));
        assert_eq!(model.raw(&next), model.raw_scalar(&next));
        let before = acc.contexts();
        let after = next.contexts();
        let changes = usize::from(before[0] != after[1]) + usize::from(before[1] != after[0]);
        assert!(changes <= 1);
        boundaries += changes;
        captures += usize::from(state.board[engine::from_to(action).1 as usize] != Cell::Empty);
        if end == Outcome::Ongoing {
            state = child;
            acc = next;
        } else {
            state = State::initial();
            acc = model.refresh(&state.board, 0, 200);
        }
    }
    assert!(
        captures > 10 && boundaries > 10,
        "captures={captures}, boundaries={boundaries}"
    );
}
