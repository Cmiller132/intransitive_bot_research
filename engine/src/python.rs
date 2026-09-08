//! `engine` Python module: the rules on plain lists, so a model's GPU kernels
//! can be tested against the reference implementation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::board::{codes, from_codes, initial_board as start};
use crate::rules::{apply as apply_rules, Outcome, Rules, State};

fn state_from(board: Vec<u8>, since_capture: u32, ply: u32) -> PyResult<State> {
    let board = from_codes(&board)
        .ok_or_else(|| PyValueError::new_err("board must hold 81 cells coded 0..6"))?;
    Ok(State {
        board,
        since_capture,
        ply,
    })
}

/// Board cells as integers: 0 empty, 1-3 own rock/paper/scissors, 4-6 enemy.
#[pyfunction]
fn initial_board() -> Vec<u8> {
    codes(&start()).to_vec()
}

/// 648 flags for the mover's legal actions.
#[pyfunction]
fn legal_mask(board: Vec<u8>) -> PyResult<Vec<bool>> {
    Ok(state_from(board, 0, 0)?.legal_mask().to_vec())
}

/// `(child_board, since_capture, ply, outcome)` with outcome 0 ongoing, 1 win,
/// 2 draw; `capture_clock = 0` disables the clock.
#[pyfunction]
fn apply(
    board: Vec<u8>,
    since_capture: u32,
    ply: u32,
    action: u16,
    capture_clock: u32,
) -> PyResult<(Vec<u8>, u32, u32, u8)> {
    let state = state_from(board, since_capture, ply)?;
    if action as usize >= crate::board::N_ACTIONS || !state.legal_mask()[action as usize] {
        return Err(PyValueError::new_err(format!(
            "action {action} is not legal"
        )));
    }
    let rules = Rules {
        capture_clock: (capture_clock > 0).then_some(capture_clock),
    };
    let (child, outcome) = apply_rules(&rules, &state, action);
    let outcome = match outcome {
        Outcome::Ongoing => 0,
        Outcome::Win => 1,
        Outcome::Draw => 2,
    };
    Ok((
        codes(&child.board).to_vec(),
        child.since_capture,
        child.ply,
        outcome,
    ))
}

#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initial_board, m)?)?;
    m.add_function(wrap_pyfunction!(legal_mask, m)?)?;
    m.add_function(wrap_pyfunction!(apply, m)?)?;
    Ok(())
}
