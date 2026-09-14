//! The conv model at play time: planes, ONNX evaluator, player.

mod net;
mod planes;
mod player;

pub use net::{improved_policy, Meta, OrtNet};
pub use planes::{encode, CLOCK_SCALE, N_PLANES, PLANE_LEN};
pub use player::{ConvPlayer, PlaySettings};
