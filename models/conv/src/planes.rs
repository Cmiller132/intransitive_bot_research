//! Observation planes (DESIGN.md item 1). The Python `conv.kernels.derive`
//! kernel and `conv.planes.reference` produce the same values.

use engine::{step, Board, Cell, Piece, State, GOAL, N_DIRS, N_SQUARES};

pub const N_PLANES: usize = 46;
pub const PLANE_LEN: usize = N_PLANES * N_SQUARES;

/// Normaliser of the two clock planes.
pub const CLOCK_SCALE: f32 = 200.0;

/// Cap and normaliser of the path-distance planes; a capped or unreachable
/// square holds 1.
const PATH_CAP: u16 = 16;

/// Distance of a square no search reached.
const UNREACHED: u16 = u16::MAX;

/// The three piece types in plane order inside every triple.
const TYPES: [Piece; 3] = [Piece::Rock, Piece::Paper, Piece::Scissors];

/// Plane-major `[N_PLANES][81]` floats for `state` under a capture clock of
/// `clock` plies (0 for no clock, which leaves plane 15 at zero).
pub fn encode(state: &State, clock: u32, out: &mut [f32]) {
    assert!(out.len() >= PLANE_LEN, "plane buffer is too short");
    let out = &mut out[..PLANE_LEN];
    out.fill(0.0);
    let board = &state.board;
    let goal_distance = |s: usize| ((8 - s / 9).max(8 - s % 9)) as f32;
    let home_distance = |s: usize| ((s / 9).max(s % 9)) as f32;

    // Planes 0-5 (piece one-hots) with the per-square move counts and the
    // per-type piece counts the global planes broadcast.
    let mut own_mobility = [0u8; N_SQUARES];
    let mut enemy_mobility = [0u8; N_SQUARES];
    let mut own_counts = [0u32; 3];
    let mut enemy_counts = [0u32; 3];
    for square in 0..N_SQUARES {
        let (piece, own) = match board[square] {
            Cell::Empty => continue,
            Cell::Own(piece) => (piece, true),
            Cell::Enemy(piece) => (piece, false),
        };
        let kind = piece.code() as usize - 1;
        let mut moves = 0u8;
        for dir in 0..N_DIRS as u8 {
            if let Some(to) = step(square as u8, dir) {
                moves += passable(board, piece, own, to as usize) as u8;
            }
        }
        if own {
            out[kind * N_SQUARES + square] = 1.0;
            own_counts[kind] += 1;
            own_mobility[square] = moves;
        } else {
            out[(3 + kind) * N_SQUARES + square] = 1.0;
            enemy_counts[kind] += 1;
            enemy_mobility[square] = moves;
        }
    }

    // Planes 6-8 and 9-11: a piece of that type beside the square could move
    // onto it, enemy first then own.
    for square in 0..N_SQUARES {
        for dir in 0..N_DIRS as u8 {
            let Some(from) = step(square as u8, dir) else {
                continue;
            };
            let (piece, own) = match board[from as usize] {
                Cell::Empty => continue,
                Cell::Own(piece) => (piece, true),
                Cell::Enemy(piece) => (piece, false),
            };
            if passable(board, piece, own, square) {
                let base = if own { 9 } else { 6 };
                out[(base + piece.code() as usize - 1) * N_SQUARES + square] = 1.0;
            }
        }
    }

    // Planes 18-29 and the race distances 40-45: one goal-path field and one
    // arrival field per type and side.
    for (kind, &piece) in TYPES.iter().enumerate() {
        for own in [true, false] {
            let target = if own { GOAL as usize } else { 0 };
            let goal_path = if passable(board, piece, own, target) {
                bfs(board, piece, own, |s| s == target, true)
            } else {
                [UNREACHED; N_SQUARES]
            };
            let held = if own {
                Cell::Own(piece)
            } else {
                Cell::Enemy(piece)
            };
            let arrival = bfs(board, piece, own, |s| board[s] == held, false);
            let (goal_plane, arrival_plane, race_plane) = if own {
                (18 + kind, 24 + kind, 40 + kind)
            } else {
                (21 + kind, 27 + kind, 43 + kind)
            };
            let mut race = 1.0f32;
            for square in 0..N_SQUARES {
                let distance = scale(goal_path[square]);
                out[goal_plane * N_SQUARES + square] = distance;
                out[arrival_plane * N_SQUARES + square] = scale(arrival[square]);
                if board[square] == held {
                    race = race.min(distance);
                }
            }
            out[race_plane * N_SQUARES..(race_plane + 1) * N_SQUARES].fill(race);
        }
    }

    // Plane 12: the two corners. The remaining planes are per-square values and
    // globals broadcast to every square.
    out[12 * N_SQUARES] = 1.0;
    out[12 * N_SQUARES + GOAL as usize] = 1.0;
    let since_capture = state.since_capture as f32 / CLOCK_SCALE;
    let clock_left = clock.saturating_sub(state.since_capture) as f32 / CLOCK_SCALE;
    let own_total: u32 = own_mobility.iter().map(|&m| m as u32).sum();
    let enemy_total: u32 = enemy_mobility.iter().map(|&m| m as u32).sum();
    for square in 0..N_SQUARES {
        out[13 * N_SQUARES + square] = 1.0;
        out[14 * N_SQUARES + square] = since_capture;
        out[15 * N_SQUARES + square] = clock_left;
        out[16 * N_SQUARES + square] = goal_distance(square) / 8.0;
        out[17 * N_SQUARES + square] = home_distance(square) / 8.0;
        out[30 * N_SQUARES + square] = own_mobility[square] as f32 / 8.0;
        out[31 * N_SQUARES + square] = enemy_mobility[square] as f32 / 8.0;
        for kind in 0..3 {
            out[(32 + kind) * N_SQUARES + square] = own_counts[kind] as f32 / 4.0;
            out[(35 + kind) * N_SQUARES + square] = enemy_counts[kind] as f32 / 4.0;
        }
        out[38 * N_SQUARES + square] = own_total as f32 / 64.0;
        out[39 * N_SQUARES + square] = enemy_total as f32 / 64.0;
    }
}

/// True when a `piece` of the given side may move onto `square`: it is empty
/// or holds a piece of the other side that `piece` beats.
fn passable(board: &Board, piece: Piece, own: bool, square: usize) -> bool {
    match board[square] {
        Cell::Empty => true,
        Cell::Own(other) => !own && piece.beats(other),
        Cell::Enemy(other) => own && piece.beats(other),
    }
}

/// Multi-source BFS over king steps for a `piece` of the given side: every
/// square `source` accepts starts at 0 and only passable squares are stepped
/// through. With `number_blocked` a square the piece may not enter still takes
/// the distance of its nearest passable neighbour plus one, which the goal-path
/// planes need (a piece standing there starts there); the arrival planes leave
/// such squares unreached.
fn bfs(
    board: &Board,
    piece: Piece,
    own: bool,
    source: impl Fn(usize) -> bool,
    number_blocked: bool,
) -> [u16; N_SQUARES] {
    let mut distance = [UNREACHED; N_SQUARES];
    let mut queue = [0u8; N_SQUARES];
    let mut tail = 0;
    for (square, entry) in distance.iter_mut().enumerate() {
        if source(square) {
            *entry = 0;
            queue[tail] = square as u8;
            tail += 1;
        }
    }
    let mut head = 0;
    while head < tail {
        let square = queue[head] as usize;
        head += 1;
        let next = distance[square] + 1;
        for dir in 0..N_DIRS as u8 {
            let Some(to) = step(square as u8, dir) else {
                continue;
            };
            let to = to as usize;
            if distance[to] != UNREACHED {
                continue;
            }
            if passable(board, piece, own, to) {
                distance[to] = next;
                queue[tail] = to as u8;
                tail += 1;
            } else if number_blocked {
                distance[to] = next;
            }
        }
    }
    distance
}

/// A path distance as a plane value: capped at `PATH_CAP` steps and scaled by
/// its reciprocal, so a capped or unreached square holds 1.
fn scale(distance: u16) -> f32 {
    distance.min(PATH_CAP) as f32 / PATH_CAP as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::Rules;

    const CLOCK: u32 = 200;

    fn planes_of(state: &State, clock: u32) -> Vec<f32> {
        let mut planes = vec![0f32; PLANE_LEN];
        encode(state, clock, &mut planes);
        planes
    }

    fn plane(planes: &[f32], index: usize) -> &[f32] {
        &planes[index * N_SQUARES..(index + 1) * N_SQUARES]
    }

    /// A position built from `(square, cell)` pairs, with a fresh clock.
    fn position(cells: &[(usize, Cell)]) -> State {
        let mut board = [Cell::Empty; N_SQUARES];
        for &(square, cell) in cells {
            board[square] = cell;
        }
        State {
            board,
            since_capture: 0,
            ply: 0,
        }
    }

    /// Squares reachable in one move by a piece of that type and side, counted
    /// straight from the board instead of through `passable`.
    fn attacked(board: &Board, piece: Piece, own: bool) -> f32 {
        let mut squares = [false; N_SQUARES];
        for from in 0..N_SQUARES {
            let holds = match board[from] {
                Cell::Own(other) => own && other == piece,
                Cell::Enemy(other) => !own && other == piece,
                Cell::Empty => false,
            };
            if !holds {
                continue;
            }
            for dir in 0..N_DIRS as u8 {
                let Some(to) = step(from as u8, dir) else {
                    continue;
                };
                squares[to as usize] |= match board[to as usize] {
                    Cell::Empty => true,
                    Cell::Own(target) => !own && piece.beats(target),
                    Cell::Enemy(target) => own && piece.beats(target),
                };
            }
        }
        squares.iter().filter(|&&hit| hit).count() as f32
    }

    /// Every attack plane holds exactly the squares of a direct one-move count.
    fn check_attack_maps(state: &State) {
        let planes = planes_of(state, CLOCK);
        for (kind, &piece) in TYPES.iter().enumerate() {
            assert_eq!(
                plane(&planes, 6 + kind).iter().sum::<f32>(),
                attacked(&state.board, piece, false),
                "enemy attack map of type {piece:?}"
            );
            assert_eq!(
                plane(&planes, 9 + kind).iter().sum::<f32>(),
                attacked(&state.board, piece, true),
                "own attack map of type {piece:?}"
            );
        }
    }

    #[test]
    fn initial_position_planes() {
        let state = State::initial();
        let planes = planes_of(&state, Rules::SITE.capture_clock.unwrap());
        let p = |index: usize| plane(&planes, index);
        // Pieces: own rock b4, and the anti-diagonal mirror for the enemy.
        assert_eq!(p(0)[28], 1.0);
        assert_eq!(p(3)[68], 1.0);
        assert_eq!(p(0).iter().sum::<f32>(), 3.0);
        assert_eq!(p(1).iter().sum::<f32>(), 4.0);
        assert_eq!(p(2).iter().sum::<f32>(), 3.0);
        assert_eq!(p(6)[44], 1.0, "the enemy rock on h6 threatens i5");
        assert_eq!(p(6)[0], 0.0);
        assert_eq!(p(12)[0], 1.0);
        assert_eq!(p(12)[80], 1.0);
        assert_eq!(p(12).iter().sum::<f32>(), 2.0);
        assert_eq!(p(13)[40], 1.0);
        assert_eq!(p(14)[0], 0.0, "no ply has passed");
        assert_eq!(p(15)[0], 1.0, "the whole site clock is left");
        assert_eq!(p(16)[28], 7.0 / 8.0, "b4 is 7 king steps from i9");
        assert_eq!(p(17)[28], 3.0 / 8.0);
        // Goal paths. Own scissors pass enemy paper only, so the d4-i9 diagonal
        // is walled by the enemy scissors on f6 and the enemy rock on g7 and the
        // shortest way round is 6 steps; own rock passes enemy scissors and own
        // paper enemy rock.
        assert_eq!(p(20)[80], 0.0);
        assert_eq!(p(20)[30], 6.0 / 16.0, "own scissors on d4");
        assert_eq!(p(20)[50], 4.0 / 16.0, "the enemy scissors on f6 is walled");
        assert_eq!(p(18)[12], 9.0 / 16.0, "own rock on d2");
        assert_eq!(p(19)[37], 8.0 / 16.0, "own paper on b5");
        assert_eq!(p(23)[50], 6.0 / 16.0, "the enemy mirror of own scissors");
        // Arrival maps of own scissors: c5 d4 e3 are sources, e4 is one step on
        // and the own rock on c3 is neither passable nor a source.
        assert_eq!(p(26)[30], 0.0);
        assert_eq!(p(26)[31], 1.0 / 16.0);
        assert_eq!(p(26)[20], 1.0);
        // Move counts: the own paper on d3 has 2 moves, the enemy none there.
        assert_eq!(p(30)[21], 2.0 / 8.0);
        assert_eq!(p(31)[21], 0.0);
        // Globals: 3 rock, 4 paper, 3 scissors and 36 moves a side.
        assert_eq!(p(32)[0], 3.0 / 4.0);
        assert_eq!(p(33)[0], 1.0);
        assert_eq!(p(34)[0], 3.0 / 4.0);
        assert_eq!(p(35)[0], 3.0 / 4.0);
        assert_eq!(p(36)[0], 1.0);
        assert_eq!(p(37)[0], 3.0 / 4.0);
        assert_eq!(p(38)[0], 36.0 / 64.0);
        assert_eq!(p(38)[0], p(30).iter().sum::<f32>() * 8.0 / 64.0);
        assert_eq!(p(39)[0], p(38)[0], "symmetric start");
        // Race distances: the leading rock, paper and scissors of each side.
        assert_eq!(p(40)[0], 9.0 / 16.0);
        assert_eq!(p(41)[0], 8.0 / 16.0);
        assert_eq!(p(42)[0], 6.0 / 16.0);
        assert_eq!(p(43)[0], 9.0 / 16.0);
        assert_eq!(p(44)[0], 8.0 / 16.0);
        assert_eq!(p(45)[0], 6.0 / 16.0);
        check_attack_maps(&state);
    }

    #[test]
    fn corridor_through_a_wall_of_rocks() {
        // Enemy rocks fill rank 5 up to h5; own scissors on a1 must reach i9
        // through the gap on i5, own paper walks over the rocks.
        let mut cells = vec![
            (0, Cell::Own(Piece::Scissors)),
            (1, Cell::Own(Piece::Paper)),
        ];
        cells.extend((36..44).map(|square| (square, Cell::Enemy(Piece::Rock))));
        let state = position(&cells);
        let planes = planes_of(&state, CLOCK);
        let p = |index: usize| plane(&planes, index);
        assert_eq!(p(20)[0], 12.0 / 16.0, "8 steps to i5, then 4 to i9");
        assert_eq!(p(42)[0], 12.0 / 16.0);
        assert_eq!(p(19)[1], 8.0 / 16.0, "paper beats rock and walks straight");
        assert_eq!(p(41)[0], 8.0 / 16.0);
        assert_eq!(p(20)[36], 8.0 / 16.0, "a5 is walled, its best exit is b6");
        assert_eq!(p(26)[44], 8.0 / 16.0, "own scissors arrive on i5 in 8");
        assert_eq!(p(26)[36], 1.0, "no own scissors may enter a5");
        check_attack_maps(&state);
    }

    #[test]
    fn a_wall_no_scissors_can_pass() {
        // The same wall closed: nothing own reaches i9 except paper.
        let mut cells = vec![(0, Cell::Own(Piece::Scissors))];
        cells.extend((36..45).map(|square| (square, Cell::Enemy(Piece::Rock))));
        let state = position(&cells);
        let planes = planes_of(&state, CLOCK);
        let p = |index: usize| plane(&planes, index);
        assert_eq!(p(20)[0], 1.0, "i9 is unreachable for own scissors");
        assert_eq!(p(20)[80], 0.0, "the goal itself is still passable");
        assert_eq!(p(20)[44], 4.0 / 16.0, "the wall square i5 is numbered");
        assert_eq!(p(20)[36], 8.0 / 16.0);
        assert_eq!(p(42)[0], 1.0, "the scissors race is lost");
        assert_eq!(p(26)[45], 1.0, "no own scissors crosses to rank 6");
        check_attack_maps(&state);
    }

    #[test]
    fn an_immortal_type_walks_everywhere() {
        // Only enemy rocks stand in the way, so own paper's goal path is the
        // plain Chebyshev distance on every square.
        let mut cells = vec![(0, Cell::Own(Piece::Paper))];
        cells.extend((36..54).map(|square| (square, Cell::Enemy(Piece::Rock))));
        let state = position(&cells);
        let planes = planes_of(&state, CLOCK);
        for square in 0..N_SQUARES {
            let chebyshev = (8 - square / 9).max(8 - square % 9) as f32;
            assert_eq!(
                plane(&planes, 19)[square],
                chebyshev / 16.0,
                "own paper goal path on {square}"
            );
        }
        assert_eq!(plane(&planes, 41)[0], 8.0 / 16.0);
        check_attack_maps(&state);
    }

    #[test]
    fn an_own_piece_on_the_goal_blocks_every_own_type() {
        let state = position(&[(80, Cell::Own(Piece::Rock)), (0, Cell::Enemy(Piece::Rock))]);
        let planes = planes_of(&state, CLOCK);
        for kind in 0..3 {
            assert!(plane(&planes, 18 + kind).iter().all(|&v| v == 1.0));
            assert!(plane(&planes, 21 + kind).iter().all(|&v| v == 1.0));
            assert_eq!(plane(&planes, 40 + kind)[0], 1.0);
            assert_eq!(plane(&planes, 43 + kind)[0], 1.0);
        }
    }

    #[test]
    fn clock_planes() {
        let mut state = State::initial();
        state.since_capture = 10;
        let planes = planes_of(&state, 50);
        assert_eq!(plane(&planes, 14)[0], 10.0 / 200.0);
        assert_eq!(plane(&planes, 15)[0], 40.0 / 200.0);
        let planes = planes_of(&state, 0);
        assert_eq!(plane(&planes, 14)[0], 10.0 / 200.0);
        assert_eq!(
            plane(&planes, 15)[0],
            0.0,
            "no clock leaves nothing to count"
        );
    }
}
