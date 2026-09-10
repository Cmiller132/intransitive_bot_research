//! Geometric runner hint, shared by NNUE and Python. Buckets 0..7 mean 1..8
//! moves by that runner's side; 8 means no qualifying shortest path. The pair
//! is (actual mover, actual opponent), with goals 80 and 0 respectively.
//!
//! A path decreases king distance to goal on every step. Static friendly
//! occupants and enemies the runner cannot capture block it. Interception is
//! reachability by any opposing piece that beats the runner: king steps through
//! empty/capturable squares, with defending friendly pieces treated as transparent.
//! Runner-side pieces the interceptor cannot capture are static barriers (so an
//! escort can block interception routes). Reachability permits waiting.
//!
//! After runner step k, interceptors have k moves if the runner moves first,
//! k+1 otherwise. Goal entry wins before that reply, so use one fewer there.
//! A second-moving runner must also survive one interceptor move at its start.
//! Paths reachable by an interceptor within those counts are rejected.
//!
//! This is not a forced-win proof: barriers can move/be exchanged, non-predators
//! can become blockers, and the opponent can win elsewhere. Clocks are ignored.

use crate::{Board, Cell};

const ALL: u128 = (1u128 << 81) - 1;
const FILE_A: u128 = file_mask(0);
const FILE_I: u128 = file_mask(8);
const LAYERS: [[u128; 9]; 2] = layers();

const fn file_mask(file: usize) -> u128 {
    let mut mask = 0;
    let mut rank = 0;
    while rank < 9 {
        mask |= 1u128 << (rank * 9 + file);
        rank += 1;
    }
    mask
}

const fn layers() -> [[u128; 9]; 2] {
    let mut out = [[0; 9]; 2];
    let mut square = 0;
    while square < 81 {
        let r = square / 9;
        let f = square % 9;
        let lo = if r < f { r } else { f };
        let hi = if r > f { r } else { f };
        out[0][8 - lo] |= 1u128 << square;
        out[1][hi] |= 1u128 << square;
        square += 1;
    }
    out
}

#[inline]
fn expand(bits: u128) -> u128 {
    let horizontal = bits | ((bits & !FILE_I) << 1) | ((bits & !FILE_A) >> 1);
    (horizontal | (horizontal << 9) | (horizontal >> 9)) & ALL
}

struct Geometry {
    pieces: [[u128; 3]; 2],
    occupied: [u128; 2],
}

impl Geometry {
    fn new(board: &Board) -> Self {
        let mut pieces = [[0; 3]; 2];
        for (square, &cell) in board.iter().enumerate() {
            let (side, piece) = match cell {
                Cell::Empty => continue,
                Cell::Own(p) => (0, p),
                Cell::Enemy(p) => (1, p),
            };
            pieces[side][piece.code() as usize - 1] |= 1u128 << square;
        }
        Self {
            occupied: pieces.map(|p| p[0] | p[1] | p[2]),
            pieces,
        }
    }

    fn bucket(&self, side: usize) -> u8 {
        let mut best = 8;
        for kind in 0..3 {
            let runners = self.pieces[side][kind];
            if runners == 0 {
                continue;
            }
            let predator = (kind + 1) % 3;
            // The predator can cross the runner type, but not either other type.
            let barriers = self.occupied[side] & !runners;
            let mut reach = [0u128; 9];
            reach[0] = self.pieces[1 - side][predator];
            for k in 1..9 {
                reach[k] = expand(reach[k - 1]) & !barriers;
            }
            let prey = (kind + 2) % 3;
            let blocked =
                self.occupied[side] | (self.occupied[1 - side] & !self.pieces[1 - side][prey]);
            // All runners at a given distance share the same time-indexed filter.
            for distance in 1..=usize::from(best).min(8) {
                let mut paths = runners & LAYERS[side][distance];
                if side == 1 {
                    paths &= !reach[1];
                }
                for k in 1..=distance {
                    if paths == 0 {
                        break;
                    }
                    let defender_moves = k + side - usize::from(k == distance);
                    paths = expand(paths)
                        & LAYERS[side][distance - k]
                        & !blocked
                        & !reach[defender_moves];
                }
                if paths != 0 {
                    best = (distance - 1) as u8;
                    break;
                }
            }
        }
        best
    }
}

/// `(mover, opponent)` buckets, distances 1..8 encoded as 0..7; none as 8.
pub fn race_buckets(board: &Board) -> (u8, u8) {
    let geometry = Geometry::new(board);
    (geometry.bucket(0), geometry.bucket(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{apply, flip, from_to, mirror_anti, step, Outcome, Piece, Rules, State};

    // Independent square-by-square oracle: enumerate all shortest runner paths
    // against a scalar breadth-first reachability map for every interceptor.
    fn oracle(board: &Board, side: usize, only: Option<usize>) -> u8 {
        let board = if side == 0 { *board } else { flip(board) };
        fn distance(sq: usize) -> usize {
            8 - (sq / 9).min(sq % 9)
        }
        fn path(
            board: &Board,
            piece: Piece,
            sq: usize,
            left: usize,
            k: usize,
            second: usize,
            reach: &[usize; 81],
        ) -> bool {
            for dir in 0..8 {
                let Some(to) = step(sq as u8, dir) else {
                    continue;
                };
                let to = to as usize;
                if distance(to) != left - 1 {
                    continue;
                }
                if match board[to] {
                    Cell::Empty => false,
                    Cell::Enemy(p) => !piece.beats(p),
                    _ => true,
                } {
                    continue;
                }
                let turns = k + second - usize::from(left == 1);
                if reach[to] <= turns {
                    continue;
                }
                if left == 1 || path(board, piece, to, left - 1, k + 1, second, reach) {
                    return true;
                }
            }
            false
        }
        let mut best = 8;
        for (sq, &cell) in board.iter().enumerate() {
            if only.is_some_and(|s| s != sq) {
                continue;
            }
            let Cell::Own(piece) = cell else { continue };
            let d = distance(sq);
            if d == 0 || d > best {
                continue;
            }
            let mut reach = [usize::MAX; 81];
            let mut queue = std::collections::VecDeque::new();
            for (enemy, &cell) in board.iter().enumerate() {
                if matches!(cell, Cell::Enemy(p) if p.beats(piece)) {
                    reach[enemy] = 0;
                    queue.push_back(enemy);
                }
            }
            while let Some(from) = queue.pop_front() {
                for dir in 0..8 {
                    let Some(to) = step(from as u8, dir) else {
                        continue;
                    };
                    let to = to as usize;
                    // Only the runner type is capturable by its predator.
                    if matches!(board[to], Cell::Own(p) if p != piece) {
                        continue;
                    }
                    if reach[to] == usize::MAX {
                        reach[to] = reach[from] + 1;
                        queue.push_back(to);
                    }
                }
            }
            if side == 1 && reach[sq] <= 1 {
                continue;
            }
            if path(&board, piece, sq, d, 1, side, &reach) {
                best = d - 1;
            }
        }
        best as u8
    }

    // Actual rules, runner-only attacking turns and every legal defending reply.
    // The deadline is a goal entry, not an evaluation or the geometric filter.
    fn gets_through(state: &State, runner: u8, attacking: bool, plies: usize) -> bool {
        if plies == 0 {
            return false;
        }
        let moves: Vec<_> = state
            .legal_actions()
            .into_iter()
            .filter(|&a| !attacking || from_to(a).0 == runner)
            .collect();
        let continuation = |action| {
            let (_, to) = from_to(action);
            let captured = !attacking && to == runner;
            let (child, outcome) = apply(
                &Rules {
                    capture_clock: None,
                },
                state,
                action,
            );
            if captured {
                return false;
            }
            if outcome != Outcome::Ongoing {
                return attacking && to == crate::GOAL;
            }
            let next_runner = mirror_anti(if attacking { to } else { runner });
            gets_through(&child, next_runner, !attacking, plies - 1)
        };
        if attacking {
            moves.into_iter().any(continuation)
        } else {
            !moves.is_empty() && moves.into_iter().all(continuation)
        }
    }

    #[test]
    fn runners_tempo_escorts_and_blockers() {
        let mut board = [Cell::Empty; 81];
        board[60] = Cell::Own(Piece::Rock);
        board[20] = Cell::Enemy(Piece::Scissors);
        assert_eq!(race_buckets(&board).0, 1);
        board[20] = Cell::Empty;
        board[78] = Cell::Enemy(Piece::Paper);
        assert_eq!(race_buckets(&board).0, 8);

        board = [Cell::Empty; 81];
        board[70] = Cell::Own(Piece::Rock);
        board[71] = Cell::Enemy(Piece::Paper);
        assert_eq!(race_buckets(&board).0, 0); // goal before the reply
        assert_eq!(race_buckets(&flip(&board)).1, 8); // caught before moving

        board = [Cell::Empty; 81];
        board[51] = Cell::Own(Piece::Rock);
        board[50] = Cell::Enemy(Piece::Paper);
        board[71] = Cell::Own(Piece::Rock);
        assert_eq!(oracle(&board, 0, Some(51)), 8);
        board[60] = Cell::Own(Piece::Scissors);
        assert_eq!(oracle(&board, 0, Some(51)), 2);
        assert_eq!(race_buckets(&board).0, oracle(&board, 0, None));
        assert!(gets_through(
            &State {
                board,
                since_capture: 0,
                ply: 0
            },
            51,
            true,
            5
        ));

        board = [Cell::Empty; 81];
        board[60] = Cell::Own(Piece::Rock);
        board[70] = Cell::Enemy(Piece::Rock);
        assert_eq!(race_buckets(&board).0, 8); // all shortest paths blocked

        board[70] = Cell::Empty;
        board[71] = Cell::Enemy(Piece::Rock);
        assert_eq!(race_buckets(&board).0, 1);
        // A non-predator can move onto the goal: the hint is not a proof.
        assert!(!gets_through(
            &State {
                board,
                since_capture: 0,
                ply: 0
            },
            60,
            true,
            3
        ));
    }

    fn random(seed: &mut u64) -> usize {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed as usize
    }

    #[test]
    fn bitboards_match_exhaustive_geometric_paths() {
        let mut seed = 2026091001;
        for _ in 0..500 {
            let mut board = [Cell::Empty; 81];
            for _ in 0..20 {
                let sq = random(&mut seed) % 79 + 1;
                board[sq] = Cell::from_code((random(&mut seed) % 6 + 1) as u8).unwrap();
            }
            assert_eq!(
                race_buckets(&board),
                (oracle(&board, 0, None), oracle(&board, 1, None))
            );
            // Main-diagonal reflection preserves both goals and the mover's tempo.
            let mut reflected = board;
            for sq in 0..81 {
                reflected[sq / 9 + sq % 9 * 9] = board[sq];
            }
            assert_eq!(race_buckets(&board), race_buckets(&reflected));
        }
    }

    #[test]
    fn positive_duel_hints_survive_every_defence_within_distance() {
        let mut seed = 2026091002;
        let mut positives = 0;
        let mut stopped = 0;
        for _ in 0..120 {
            let runner = (5 + random(&mut seed) % 4) * 9 + 5 + random(&mut seed) % 4;
            let defender = (4 + random(&mut seed) % 5) * 9 + 4 + random(&mut seed) % 5;
            if runner == defender || runner == 80 || defender == 80 {
                continue;
            }
            let mut board = [Cell::Empty; 81];
            board[runner] = Cell::Own(Piece::Rock);
            board[defender] = Cell::Enemy(Piece::Paper);
            let distance = 8 - (runner / 9).min(runner % 9);
            for second in [false, true] {
                let (b, square) = if second {
                    (flip(&board), mirror_anti(runner as u8))
                } else {
                    (board, runner as u8)
                };
                let state = State {
                    board: b,
                    since_capture: 0,
                    ply: 0,
                };
                let bucket = if second {
                    race_buckets(&b).1
                } else {
                    race_buckets(&b).0
                };
                let actual =
                    gets_through(&state, square, !second, 2 * distance - usize::from(!second));
                if bucket != 8 {
                    positives += 1;
                    assert!(actual, "{runner} {defender} {second}");
                }
                if !actual {
                    stopped += 1;
                    assert_eq!(bucket, 8);
                }
            }
        }
        assert!(positives > 10 && stopped > 10);
    }
}
