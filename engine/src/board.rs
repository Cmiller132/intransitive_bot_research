//! Board representation in the canonical frame: the mover plays from a1 (index 0)
//! toward i9 (index 80). Used by `rules` and by every crate that reads a position.

pub const N_SQUARES: usize = 81;
pub const N_DIRS: usize = 8;
pub const N_ACTIONS: usize = N_DIRS * N_SQUARES;

/// King directions as (rank delta, file delta), in action-encoding order.
pub const DIRS: [(i8, i8); N_DIRS] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Square index `rank * 9 + file`, 0 = a1, 80 = i9.
pub type Square = u8;

/// Direction index into `DIRS`.
pub type Dir = u8;

/// `dir * 81 + from`.
pub type Action = u16;

/// The mover's goal (the opponent's home corner).
pub const GOAL: Square = 80;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Piece {
    Rock,
    Paper,
    Scissors,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty,
    Own(Piece),
    Enemy(Piece),
}

/// 81 cells, rank-major, canonical frame.
pub type Board = [Cell; N_SQUARES];

impl Piece {
    /// Rock beats scissors, scissors beats paper, paper beats rock.
    pub fn beats(self, other: Piece) -> bool {
        (self.code() as i8 - other.code() as i8).rem_euclid(3) == 1
    }

    /// 1 rock, 2 paper, 3 scissors: the integer coding used on the wire and in kernels.
    pub fn code(self) -> u8 {
        match self {
            Piece::Rock => 1,
            Piece::Paper => 2,
            Piece::Scissors => 3,
        }
    }

    pub fn from_code(code: u8) -> Option<Piece> {
        match code {
            1 => Some(Piece::Rock),
            2 => Some(Piece::Paper),
            3 => Some(Piece::Scissors),
            _ => None,
        }
    }
}

impl Cell {
    /// Own <-> enemy; empty stays empty.
    pub fn swap_side(self) -> Cell {
        match self {
            Cell::Empty => Cell::Empty,
            Cell::Own(piece) => Cell::Enemy(piece),
            Cell::Enemy(piece) => Cell::Own(piece),
        }
    }

    /// 0 empty, 1-3 own rock/paper/scissors, 4-6 enemy.
    pub fn code(self) -> u8 {
        match self {
            Cell::Empty => 0,
            Cell::Own(piece) => piece.code(),
            Cell::Enemy(piece) => piece.code() + 3,
        }
    }

    pub fn from_code(code: u8) -> Option<Cell> {
        match code {
            0 => Some(Cell::Empty),
            1..=3 => Piece::from_code(code).map(Cell::Own),
            4..=6 => Piece::from_code(code - 3).map(Cell::Enemy),
            _ => None,
        }
    }

    pub fn is_own(self) -> bool {
        matches!(self, Cell::Own(_))
    }

    pub fn is_enemy(self) -> bool {
        matches!(self, Cell::Enemy(_))
    }
}

/// Reflection across the anti-diagonal: (rank, file) -> (8 - file, 8 - rank).
pub fn mirror_anti(square: Square) -> Square {
    let (rank, file) = (square / 9, square % 9);
    (8 - file) * 9 + (8 - rank)
}

/// (from, to) of an action. The caller guarantees the target is on the board.
pub fn from_to(action: Action) -> (Square, Square) {
    let dir = (action as usize / N_SQUARES) as Dir;
    let from = (action as usize % N_SQUARES) as Square;
    let to = step(from, dir).expect("action target is on the board");
    (from, to)
}

/// The action moving `from` in direction `dir`.
pub fn action(dir: Dir, from: Square) -> Action {
    dir as Action * N_SQUARES as Action + from as Action
}

/// Target square of moving `from` in direction `dir`, or `None` off the board.
pub fn step(from: Square, dir: Dir) -> Option<Square> {
    let (dr, df) = DIRS[dir as usize];
    let rank = (from / 9) as i8 + dr;
    let file = (from % 9) as i8 + df;
    ((0..9).contains(&rank) && (0..9).contains(&file)).then(|| (rank * 9 + file) as Square)
}

/// The starting position with Blue (the first mover) as "own".
pub fn initial_board() -> Board {
    const SETUP: [(Piece, &[Square]); 3] = [
        (Piece::Rock, &[28, 20, 12]),      // b4 c3 d2
        (Piece::Paper, &[37, 29, 21, 13]), // b5 c4 d3 e2
        (Piece::Scissors, &[38, 30, 22]),  // c5 d4 e3
    ];
    let mut board = [Cell::Empty; N_SQUARES];
    for (piece, squares) in SETUP {
        for &square in squares {
            board[square as usize] = Cell::Own(piece);
            board[mirror_anti(square) as usize] = Cell::Enemy(piece);
        }
    }
    board
}

/// The same position seen by the other side: reflect and swap.
pub fn flip(board: &Board) -> Board {
    let mut out = [Cell::Empty; N_SQUARES];
    for (square, cell) in out.iter_mut().enumerate() {
        *cell = board[mirror_anti(square as Square) as usize].swap_side();
    }
    out
}

/// Board as the integer coding, 81 bytes.
pub fn codes(board: &Board) -> [u8; N_SQUARES] {
    let mut out = [0; N_SQUARES];
    for (code, cell) in out.iter_mut().zip(board) {
        *code = cell.code();
    }
    out
}

/// Board from the integer coding; `None` if any code is out of range.
pub fn from_codes(codes: &[u8]) -> Option<Board> {
    if codes.len() != N_SQUARES {
        return None;
    }
    let mut board = [Cell::Empty; N_SQUARES];
    for (cell, &code) in board.iter_mut().zip(codes) {
        *cell = Cell::from_code(code)?;
    }
    Some(board)
}
