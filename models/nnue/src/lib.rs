//! CPU NNUE inference and alpha-beta play over the shared engine rules.
pub mod diagnostic;
pub mod net;
mod player;
#[cfg(feature = "profile")]
pub mod profile;
pub mod search;
pub use net::Model;
pub use player::NnuePlayer;
