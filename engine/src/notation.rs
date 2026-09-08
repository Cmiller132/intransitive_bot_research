//! Square names and move spelling in the canonical frame. Frame changes for the
//! site (which corner is Blue's home) belong to the site adapter, not here.

use crate::board::{action, from_to, Action, Square, DIRS, N_SQUARES};

/// `"a1"` for 0 through `"i9"` for 80.
pub fn square_name(square: Square) -> String {
    let file = (b'a' + square % 9) as char;
    let rank = (b'1' + square / 9) as char;
    format!("{file}{rank}")
}

/// Inverse of `square_name`; `None` for anything else.
pub fn parse_square(name: &str) -> Option<Square> {
    let bytes = name.as_bytes();
    if bytes.len() != 2 || !(b'a'..=b'i').contains(&bytes[0]) || !(b'1'..=b'9').contains(&bytes[1])
    {
        return None;
    }
    Some((bytes[1] - b'1') * 9 + (bytes[0] - b'a'))
}

/// `"c4-d5"`.
pub fn action_name(act: Action) -> String {
    let (from, to) = from_to(act);
    format!("{}-{}", square_name(from), square_name(to))
}

/// `"c4-d5"`, `"c4xd5"`, `"c4d5"` or `"Rc4xd5"` -> `(from, to)`.
pub fn parse_squares(text: &str) -> Option<(Square, Square)> {
    let text = text.trim_start_matches(|c: char| "RPSrps".contains(c));
    let (from, to) = match text.len() {
        4 => (&text[..2], &text[2..]),
        5 if text.as_bytes()[2] == b'-' || text.as_bytes()[2] == b'x' => (&text[..2], &text[3..]),
        _ => return None,
    };
    Some((parse_square(from)?, parse_square(to)?))
}

/// The action moving `from` to `to`, or `None` if that is not a one-square king move.
pub fn action_between(from: Square, to: Square) -> Option<Action> {
    let delta = (
        (to / 9) as i8 - (from / 9) as i8,
        (to % 9) as i8 - (from % 9) as i8,
    );
    DIRS.iter()
        .position(|&d| d == delta)
        .map(|dir| action(dir as u8, from))
}

/// `"c4-d5"`, `"c4xd5"` or `"Rc4xd5"` -> the action, or `None` if it is not a
/// one-square king move.
pub fn parse_action(text: &str) -> Option<Action> {
    let (from, to) = parse_squares(text)?;
    debug_assert!((to as usize) < N_SQUARES);
    action_between(from, to)
}
