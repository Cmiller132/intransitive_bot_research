//! Observation planes (DESIGN.md item 1). The Python `sq.kernels.derive`
//! kernel and `sq.planes.reference` produce the same values.

use engine::{step, Cell, State, N_DIRS, N_SQUARES};

pub const N_PLANES: usize = 25;
pub const PLANE_LEN: usize = N_PLANES * N_SQUARES;

/// Normaliser of the two clock planes.
pub const CLOCK_SCALE: f32 = 200.0;

/// Plane-major `[N_PLANES][81]` floats for `state`.
pub fn encode(state: &State, out: &mut [f32]) {
    assert!(out.len() >= PLANE_LEN, "plane buffer is too short");
    let out = &mut out[..PLANE_LEN];
    out.fill(0.0);
    let board = &state.board;
    let goal_distance = |s: usize| ((8 - s / 9).max(8 - s % 9)) as f32;
    let home_distance = |s: usize| ((s / 9).max(s % 9)) as f32;

    let mut own_mobility = [0u8; N_SQUARES];
    let mut enemy_mobility = [0u8; N_SQUARES];
    let mut min_own = 9.0f32;
    let mut min_enemy = 9.0f32;
    let (mut own_count, mut enemy_count) = (0usize, 0usize);
    for s in 0..N_SQUARES {
        match board[s] {
            Cell::Empty => {}
            Cell::Own(piece) => {
                own_count += 1;
                min_own = min_own.min(goal_distance(s));
                out[(piece.code() as usize - 1) * N_SQUARES + s] = 1.0;
                for dir in 0..N_DIRS as u8 {
                    if let Some(to) = step(s as u8, dir) {
                        let ok = match board[to as usize] {
                            Cell::Empty => true,
                            Cell::Enemy(other) => piece.beats(other),
                            Cell::Own(_) => false,
                        };
                        own_mobility[s] += ok as u8;
                    }
                }
            }
            Cell::Enemy(piece) => {
                enemy_count += 1;
                min_enemy = min_enemy.min(home_distance(s));
                out[(piece.code() as usize + 2) * N_SQUARES + s] = 1.0;
                for dir in 0..N_DIRS as u8 {
                    if let Some(to) = step(s as u8, dir) {
                        let ok = match board[to as usize] {
                            Cell::Empty => true,
                            Cell::Own(other) => piece.beats(other),
                            Cell::Enemy(_) => false,
                        };
                        enemy_mobility[s] += ok as u8;
                    }
                }
            }
        }
    }
    // Threat planes 6-8: an adjacent enemy of that type could move onto the square.
    for s in 0..N_SQUARES {
        for dir in 0..N_DIRS as u8 {
            let Some(from) = step(s as u8, dir) else {
                continue;
            };
            let Cell::Enemy(attacker) = board[from as usize] else {
                continue;
            };
            let reachable = match board[s] {
                Cell::Empty => true,
                Cell::Own(target) => attacker.beats(target),
                Cell::Enemy(_) => false,
            };
            if reachable {
                out[(5 + attacker.code() as usize) * N_SQUARES + s] = 1.0;
            }
        }
    }
    out[9 * N_SQUARES] = 1.0;
    out[9 * N_SQUARES + 80] = 1.0;
    let own_total: u32 = own_mobility.iter().map(|&m| m as u32).sum();
    let enemy_total: u32 = enemy_mobility.iter().map(|&m| m as u32).sum();
    for s in 0..N_SQUARES {
        let cell = board[s];
        out[10 * N_SQUARES + s] = state.since_capture as f32 / CLOCK_SCALE;
        out[11 * N_SQUARES + s] = state.ply as f32 / CLOCK_SCALE;
        out[12 * N_SQUARES + s] = 1.0;
        out[13 * N_SQUARES + s] = if cell.is_own() {
            goal_distance(s) / 8.0
        } else {
            0.0
        };
        out[14 * N_SQUARES + s] = if cell.is_enemy() {
            home_distance(s) / 8.0
        } else {
            0.0
        };
        out[15 * N_SQUARES + s] = min_own / 8.0;
        out[16 * N_SQUARES + s] = min_enemy / 8.0;
        out[17 * N_SQUARES + s] = goal_distance(s) / 8.0;
        out[18 * N_SQUARES + s] = home_distance(s) / 8.0;
        out[19 * N_SQUARES + s] = own_mobility[s] as f32 / 8.0;
        out[20 * N_SQUARES + s] = enemy_mobility[s] as f32 / 8.0;
        out[21 * N_SQUARES + s] = own_total as f32 / 64.0;
        out[22 * N_SQUARES + s] = enemy_total as f32 / 64.0;
        out[23 * N_SQUARES + s] = own_count as f32 / 10.0;
        out[24 * N_SQUARES + s] = enemy_count as f32 / 10.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_planes() {
        let mut planes = [0f32; PLANE_LEN];
        encode(&State::initial(), &mut planes);
        let plane = |p: usize| &planes[p * N_SQUARES..(p + 1) * N_SQUARES];
        assert_eq!(plane(0)[28], 1.0, "own rock on b4");
        assert_eq!(plane(3)[80 - 28], 1.0, "enemy rock mirrored");
        assert_eq!(plane(0).iter().sum::<f32>(), 3.0);
        assert_eq!(plane(1).iter().sum::<f32>(), 4.0);
        assert_eq!(plane(6)[44], 1.0, "the enemy rock on h6 threatens i5");
        assert_eq!(plane(6)[0], 0.0);
        assert_eq!(plane(7)[52], 0.0, "an enemy square is never threatened");
        assert_eq!(plane(9)[0], 1.0);
        assert_eq!(plane(9)[80], 1.0);
        assert_eq!(plane(9).iter().sum::<f32>(), 2.0);
        assert_eq!(plane(10)[5], 0.0);
        assert_eq!(plane(12)[5], 1.0);
        assert_eq!(plane(13)[28], 7.0 / 8.0, "b4 is 7 king steps from i9");
        assert_eq!(plane(15)[0], 5.0 / 8.0, "d4 leads the race at distance 5");
        assert_eq!(plane(16)[0], 5.0 / 8.0);
        assert_eq!(plane(17)[0], 1.0);
        assert_eq!(plane(18)[80], 1.0);
        assert_eq!(plane(23)[0], 1.0);
        assert_eq!(plane(24)[0], 1.0);
        let total: f32 = plane(21)[0] * 64.0;
        assert_eq!(total, plane(19).iter().sum::<f32>() * 8.0);
        assert_eq!(plane(21)[0], plane(22)[0], "symmetric start");
    }
}
