//! Version 9 against bytes built here: the header and length contract, the
//! goal-corner rows in both frames, the output bucket and the incremental
//! updates around the two corners. The cross-language parity (the NumPy oracle
//! over the same file, the Rust kernels and the PyTorch forward) is in
//! `format9_fixtures.rs` with the Python-built fixtures.
use engine::{apply, from_codes, from_to, Rules, State};
use nnue::net::{Accumulator, Model};

const F9: usize = 13648;
const GOAL_BASE: usize = 13640;
/// The documented mapping of the pieces on the board (2..=20) to the head.
const BUCKETS: [usize; 19] = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7];

/// Readout 2H i16, readout bias i32, dense weights 32x2H i8, dense biases 32
/// i32 and residual readout 32 i16; every head differs from every other.
fn head_block(hidden: usize, head: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for k in 0..2 * hidden {
        bytes.extend((((17 * k + 29 * head) % 15) as i16 - 7).to_le_bytes());
    }
    bytes.extend(((head as i32 - 4) * 11).to_le_bytes());
    for i in 0..32 * 2 * hidden {
        bytes.push((((13 * i + 7 * head) % 255) as i32 - 128) as i8 as u8);
    }
    for i in 0..32i32 {
        bytes.extend(((i - 16) * 255 * 32).to_le_bytes());
    }
    for i in 0..32i32 {
        bytes.extend(((i - 16 + head as i32) as i16).to_le_bytes());
    }
    bytes
}

fn fixture9(hidden: usize) -> Vec<u8> {
    let mut bytes = b"RPSNNUE1".to_vec();
    for value in [9u32, F9 as u32, hidden as u32, 255, 64] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend(600f32.to_le_bytes());
    bytes.extend(8u32.to_le_bytes());
    for j in 0..hidden {
        bytes.extend((96 + (j % 7) as i16).to_le_bytes());
    }
    for row in 0..F9 {
        for j in 0..hidden {
            bytes.extend((((13 * row + 7 * j + 5 * (row / 486)) % 31) as i16 - 15).to_le_bytes());
        }
    }
    for head in 0..8 {
        bytes.extend(head_block(hidden, head));
    }
    bytes
}

fn board(codes: &[(usize, u8)]) -> engine::Board {
    let mut cells = [0u8; 81];
    for &(square, code) in codes {
        cells[square] = code;
    }
    from_codes(&cells).unwrap()
}

#[test]
fn the_reader_enforces_the_version_nine_header_and_exact_size() {
    for hidden in [32, 64] {
        let bytes = fixture9(hidden);
        // 36-byte header, bias, feature table and eight complete heads.
        assert_eq!(bytes.len(), 1604 + 2 * F9 * hidden + 546 * hidden);
        let model = Model::from_bytes(&bytes).unwrap();
        assert_eq!(
            (model.features, model.hidden, model.heads(), model.slots()),
            (F9, hidden, 8, 44)
        );
        for (at, values) in [
            (8u32, vec![6u32, 8, 10]),
            (12, vec![1004, 13640, 13647, 13649]),
            (16, vec![0, 31, 33, 1056]),
            (20, vec![0, 254, 256]),
            (24, vec![0, 63, 65]),
            (28, vec![f32::NAN.to_bits(), 601f32.to_bits()]),
            (32, vec![0, 1, 2, 4, 7, 9, 16]),
        ] {
            for value in values {
                let mut bad = bytes.clone();
                let at = at as usize;
                bad[at..at + 4].copy_from_slice(&value.to_le_bytes());
                assert!(
                    Model::from_bytes(&bad).is_err(),
                    "offset {at}, value {value}"
                );
            }
        }
        let mut bad = bytes.clone();
        bad[0] = 0;
        assert!(Model::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(Model::from_bytes(&bad).is_err());
        assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }
}

#[test]
fn goal_rows_report_both_corners_in_both_frames() {
    let model = Model::from_bytes(&fixture9(32)).unwrap();
    let empty = [GOAL_BASE, GOAL_BASE + 4];
    for (codes, own, opponent) in [
        // Nobody on a corner.
        (vec![(40, 1u8), (41, 4u8)], empty, empty),
        // An enemy paper blocks the mover's goal; the opponent's frame sees
        // its own paper standing in the mover's goal corner.
        (
            vec![(40, 1), (80, 5)],
            [GOAL_BASE + 2, GOAL_BASE + 4],
            [GOAL_BASE, GOAL_BASE + 4 + 2],
        ),
        // The mover's scissors stand in the opponent's goal corner.
        (
            vec![(0, 3), (40, 4)],
            [GOAL_BASE, GOAL_BASE + 4 + 3],
            [GOAL_BASE + 3, GOAL_BASE + 4],
        ),
        // Both corners occupied.
        (
            vec![(0, 1), (80, 6), (40, 2)],
            [GOAL_BASE + 3, GOAL_BASE + 4 + 1],
            [GOAL_BASE + 1, GOAL_BASE + 4 + 3],
        ),
    ] {
        let ids = model.feature_ids(&board(&codes), 4, 200);
        assert_eq!(ids[0][42..], own, "{codes:?}");
        assert_eq!(ids[1][42..], opponent, "{codes:?}");
        for side in ids {
            assert!(side[..42].iter().all(|&id| id < GOAL_BASE || id == F9));
        }
    }
}

#[test]
fn the_head_follows_the_piece_count_and_the_file_uses_it() {
    let model = Model::from_bytes(&fixture9(32)).unwrap();
    // The same network with head 0 in every bucket: a value that differs from
    // it can only come from the selected head.
    let mut flat = model.clone();
    flat.output = model.output[..2 * model.hidden].repeat(8);
    flat.output_bias = vec![model.output_bias[0]; 8];
    flat.dense = vec![model.dense[0].clone(); 8];
    let mut differed = 0;
    for total in 2..=20usize {
        let own = total / 2;
        let codes: Vec<_> = (0..own)
            .map(|i| (9 + i, 1u8))
            .chain((0..total - own).map(|i| (71 - i, 4u8)))
            .collect();
        let acc = model.refresh(&board(&codes), 7, 200);
        let bucket = model.bucket(&acc);
        assert_eq!(bucket, BUCKETS[total - 2], "{total} pieces");
        assert_eq!(bucket, ((total - 2) * 8 / 19).min(7));
        assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
        let mut avx2 = model.clone();
        avx2.disable_vnni();
        assert_eq!(model.raw(&acc), avx2.raw(&acc));
        assert_eq!(flat.bucket(&acc), bucket);
        if bucket == 0 {
            assert_eq!(model.raw(&acc), flat.raw(&acc));
        } else {
            assert_ne!(model.raw(&acc), flat.raw(&acc));
            differed += 1;
        }
    }
    assert_eq!(differed, 19 - 3);
    // One head per bucket, and each of the eight is reachable.
    assert_eq!(model.output.len(), 8 * 2 * model.hidden);
    assert_eq!(model.dense.len(), 8);
    let reached: std::collections::BTreeSet<_> = (2..=20).map(|t| BUCKETS[t - 2]).collect();
    assert_eq!(reached, (0..8).collect());
}

#[test]
fn incremental_updates_match_refresh_around_the_goal_corners() {
    let model = Model::from_bytes(&fixture9(32)).unwrap();
    let mut avx2 = model.clone();
    avx2.disable_vnni();
    let mut touched = 0;
    for codes in [
        // Moves onto the opponent's goal corner from all three neighbours.
        vec![(1, 1u8), (9, 2u8), (10, 3u8), (40, 4u8), (44, 6u8)],
        // A blocker steps off the opponent's goal corner.
        vec![(0, 2), (40, 5), (44, 4)],
        // An enemy rock blocks the mover's goal; paper may capture it there.
        vec![(79, 2), (80, 4), (30, 5), (31, 1)],
        // A move onto the mover's own goal wins; the row still has to follow.
        vec![(71, 3), (40, 1), (20, 5), (22, 6)],
        // Already-decided boards are never evaluated, but the rows stay total.
        vec![(0, 4), (80, 1), (40, 2), (41, 6)],
    ] {
        let state = State {
            board: board(&codes),
            since_capture: 3,
            ply: 3,
        };
        let acc = model.refresh(&state.board, 3, 200);
        assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
        assert_eq!(model.raw(&acc), avx2.raw(&acc));
        let legal = state.legal_actions();
        assert!(!legal.is_empty());
        for action in legal {
            let (child, _) = apply(&Rules::SITE, &state, action);
            let mut next = Accumulator::empty(model.hidden);
            model.update(&acc, &state.board, action, child.since_capture, &mut next);
            assert_eq!(
                next,
                model.refresh(&child.board, child.since_capture, 200),
                "action {action} from {codes:?}"
            );
            assert_eq!(model.raw(&next), model.raw_scalar(&next));
            assert_eq!(model.raw(&next), avx2.raw(&next));
            let (from, to) = from_to(action);
            touched += usize::from([from, to].iter().any(|&s| s == 0 || s == 80));
        }
    }
    assert!(touched >= 8, "corner moves exercised: {touched}");
}

#[test]
fn a_thousand_random_moves_keep_the_updates_equal_to_a_refresh() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let model = Model::from_bytes(&fixture9(32)).unwrap();
    let mut rng = StdRng::seed_from_u64(20260916);
    let mut state = State::initial();
    let mut acc = model.refresh(&state.board, 0, 200);
    let mut buckets = std::collections::BTreeSet::new();
    for _ in 0..1000 {
        buckets.insert(model.bucket(&acc));
        let legal = state.legal_actions();
        let captures: Vec<_> = legal
            .iter()
            .copied()
            .filter(|&a| state.board[from_to(a).1 as usize] != engine::Cell::Empty)
            .collect();
        let choices = if captures.is_empty() {
            &legal
        } else {
            &captures
        };
        let action = choices[rng.random_range(0..choices.len())];
        let (child, end) = apply(&Rules::SITE, &state, action);
        let mut next = Accumulator::empty(model.hidden);
        model.update(&acc, &state.board, action, child.since_capture, &mut next);
        assert_eq!(next, model.refresh(&child.board, child.since_capture, 200));
        assert_eq!(model.raw(&next), model.raw_scalar(&next));
        if end == engine::Outcome::Ongoing {
            state = child;
            acc = next;
        } else {
            state = State::initial();
            acc = model.refresh(&state.board, 0, 200);
        }
    }
    // Capture-biased play walks the piece count down through every bucket.
    assert_eq!(buckets, (0..8).collect());
}
