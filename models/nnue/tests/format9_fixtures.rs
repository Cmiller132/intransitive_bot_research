//! Version 9 against the Python-built fixtures: the NumPy oracle over the same
//! file bytes (ids, contexts, buckets and raw values), the scalar and SIMD
//! kernels, incremental updates against a refresh on every legal child, and the
//! migration file, which must evaluate every shared position exactly as the
//! format 8 network it was converted from.
use std::collections::BTreeSet;

use nnue::net::Model;

const NET: &[u8] = include_bytes!("fixtures/format9_h32.nnue");
const CONVERTED: &[u8] = include_bytes!("fixtures/format9_convert_h32.nnue");
const V8: &[u8] = include_bytes!("fixtures/format8_h32.nnue");
const POSITIONS: &str = include_str!("fixtures/format9_positions.jsonl");
const V8_POSITIONS: &str = include_str!("fixtures/format8_positions.jsonl");
const GOAL_BASE: usize = 13640;

fn rows(input: &str) -> Vec<nnue::diagnostic::Position> {
    input
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn the_oracle_the_kernels_and_the_incremental_path_agree_on_every_fixture_position() {
    let model = Model::from_bytes(NET).unwrap();
    assert_eq!(
        (model.features, model.hidden, model.heads()),
        (13648, 32, 8)
    );
    let mut avx2 = model.clone();
    avx2.disable_vnni();
    // The diagnostic checks every row's ids, context, bucket and raw value
    // against the oracle's, the scalar path against this backend, and every
    // legal child's incremental update against a refresh.
    let mut output = Vec::new();
    nnue::diagnostic::run(&model, POSITIONS.as_bytes(), &mut output, None, 1).unwrap();
    assert_eq!(String::from_utf8(output).unwrap().lines().count(), 380);
    let (mut buckets, mut corners) = (BTreeSet::new(), BTreeSet::new());
    for row in rows(POSITIONS) {
        let state = row.state().unwrap();
        let acc = model.refresh(&state.board, state.since_capture, row.clock);
        assert_eq!(model.raw(&acc), model.raw_scalar(&acc));
        assert_eq!(model.raw(&acc), avx2.raw(&acc));
        buckets.insert(model.bucket(&acc));
        let ids = model.feature_ids(&state.board, state.since_capture, row.clock);
        corners.extend(
            ids.iter()
                .flat_map(|side| side[42..].iter().map(|id| id - GOAL_BASE)),
        );
    }
    // Every head and every occupant of both corners, in both frames.
    assert_eq!(buckets, (0..8).collect());
    assert_eq!(corners, (0..8).collect());
}

#[test]
fn the_converted_format8_network_evaluates_every_shared_position_identically() {
    let eight = Model::from_bytes(V8).unwrap();
    let nine = Model::from_bytes(CONVERTED).unwrap();
    assert_eq!((nine.features, nine.heads(), nine.slots()), (13648, 8, 44));
    let mut buckets = BTreeSet::new();
    for row in rows(V8_POSITIONS) {
        let state = row.state().unwrap();
        let before = eight.refresh(&state.board, state.since_capture, row.clock);
        let after = nine.refresh(&state.board, state.since_capture, row.clock);
        // The goal rows are zero and every head is the one head, so the whole
        // integer evaluation is the incumbent's, bucket by bucket.
        assert_eq!(eight.raw(&before), nine.raw(&after));
        assert_eq!(eight.evaluate(&before), nine.evaluate(&after));
        assert_eq!(nine.raw(&after), nine.raw_scalar(&after));
        // The recorded value is the NumPy oracle's, which sums in its own
        // order: the f64 agreement is DESIGN's 1e-12, the integer one exact.
        assert!((nine.raw(&after) - row.raw.unwrap()).abs() <= 1e-12);
        buckets.insert(nine.bucket(&after));
    }
    assert_eq!(buckets, (0..8).collect());
}
