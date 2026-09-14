//! The Rust encoder against the Python reference (`conv.planes.reference`) on
//! the positions in `fixtures/planes.json`, written by that reference.

use conv::{encode, N_PLANES, PLANE_LEN};
use engine::{from_codes, State, N_SQUARES};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    board: Vec<u8>,
    since_capture: u32,
    ply: u32,
    clock: u32,
    planes: Vec<f32>,
}

#[test]
fn matches_the_python_reference() {
    let text = include_str!("fixtures/planes.json");
    let cases: Vec<Case> = serde_json::from_str(text).unwrap();
    assert!(cases.len() >= 20);
    let mut out = vec![0.0f32; PLANE_LEN];
    for (i, case) in cases.iter().enumerate() {
        let state = State {
            board: from_codes(&case.board).unwrap(),
            since_capture: case.since_capture,
            ply: case.ply,
        };
        encode(&state, case.clock, &mut out);
        assert_eq!(case.planes.len(), PLANE_LEN);
        for (j, (&got, &want)) in out.iter().zip(&case.planes).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "case {i} plane {} square {}: rust {got} python {want}",
                j / N_SQUARES,
                j % N_SQUARES
            );
        }
    }
    assert_eq!(N_PLANES, 46);
}
