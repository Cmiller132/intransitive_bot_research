//! RPSNNUE1: two canonical perspectives, sparse piece-square features and SCReLU.
//! All accumulator arithmetic is i32; output accumulation is i64. No saturation
//! or wraparound is permitted before the activation clamp.
use engine::tactics::is_attacked;
use engine::{Board, Cell};
const N_SQ: usize = 81;
fn action_from_to(a: u16) -> (usize, usize) {
    let (f, t) = engine::from_to(a);
    (f as usize, t as usize)
}
fn mirror_anti(s: usize) -> usize {
    engine::mirror_anti(s as u8) as usize
}
fn swap_side(c: u8) -> u8 {
    Cell::from_code(c).expect("valid cell").swap_side().code()
}

/// Size of the converted prototype (H512/B1).
pub const FILE_SIZE: usize = 1_064_172;
const CLOCK_BOUNDS: [u32; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160];
fn clock_rows(since: u32, clock: u32) -> [usize; 2] {
    [
        972 + CLOCK_BOUNDS.partition_point(|&b| b <= since) - 1,
        988 + CLOCK_BOUNDS.partition_point(|&b| b <= clock.saturating_sub(since)) - 1,
    ]
}
use std::{fs, path::Path};

pub const FEATURES: usize = 6 * N_SQ;
pub const TACTICAL_FEATURES: usize = 2 * FEATURES;
pub const HIDDEN_LANES: usize = 32;
#[cfg(target_arch = "x86_64")]
const DOT_CHUNK: usize = 4096;
pub const DENSE_WIDTH: usize = 32;

/// Reused by evaluations on one search thread, including scalar parity checks.
struct DenseScratch {
    input: Vec<u8>,
    #[cfg(target_arch = "x86_64")]
    widened: Vec<i16>,
}
thread_local! {
    static DENSE_SCRATCH: std::cell::RefCell<DenseScratch> = const {
        std::cell::RefCell::new(DenseScratch {
            input: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            widened: Vec::new(),
        })
    };
}

#[derive(Clone, Debug)]
pub struct DenseHead {
    pub weights: Vec<i8>,
    pub bias: Vec<i32>,
    pub output: Vec<i16>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub features: usize,
    pub hidden: usize,
    pub buckets: usize,
    pub qa: i32,
    pub qb: i32,
    pub eval_scale: f32,
    pub bias: Vec<i16>,
    pub weights: Vec<i16>,
    pub output: Vec<i16>,
    pub output_bias: Vec<i32>,
    pub dense: Option<DenseHead>,
    avx2: bool,
    vnni: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accumulator {
    pub own: Vec<i32>,
    pub opponent: Vec<i32>,
    pieces: usize,
    clock: u32,
    clock_rows: [usize; 2],
    race: [u8; 2],
    race_dirty: bool,
}

impl Accumulator {
    pub fn empty(hidden: usize) -> Self {
        Self {
            own: vec![0; hidden],
            opponent: vec![0; hidden],
            pieces: 0,
            clock: 0,
            clock_rows: [0; 2],
            race: [8; 2],
            race_dirty: false,
        }
    }
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 36 || &bytes[..8] != b"RPSNNUE1" {
            return Err("invalid NNUE header or magic".into());
        }
        let u = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let buckets = u(32) as usize;
        let hidden = u(16) as usize;
        let features = u(12) as usize;
        if !matches!((u(8), features), (6, 1004) | (7, 1022))
            || [u(20), u(24)] != [255, 64]
            || f32::from_le_bytes(bytes[28..32].try_into().unwrap()) != 600.
            || ![1, 4].contains(&buckets)
        {
            return Err("unsupported NNUE format, dimensions, scales or buckets".into());
        }
        if hidden == 0
            || !hidden.is_multiple_of(HIDDEN_LANES)
            || hidden as u64 > i64::MAX as u64 / (2 * 255 * 255 * 32768)
        {
            return Err(
                "hidden width must be a positive multiple of 32 with representable integer sums"
                    .into(),
            );
        }
        let expected = hidden
            .checked_mul(2 * features + 66 + 4 * buckets)
            .and_then(|v| v.checked_add(168 + 68 * buckets));
        if expected != Some(bytes.len()) {
            return Err("invalid NNUE payload length".into());
        }
        let features_start = 36 + 2 * hidden;
        let output_start = features_start + 2 * features * hidden;
        let bias_start = output_start + buckets * 2 * hidden * 2;
        let width_start = bias_start + buckets * 4;
        let dense_start = width_start + 4;
        let dense_bias_start = dense_start + 32 * 2 * hidden;
        let residual_start = dense_bias_start + 32 * 4;
        if u(width_start) != 32 {
            return Err("unsupported dense width".into());
        }
        let i16s = |a: usize, n: usize| {
            bytes[a..a + 2 * n]
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let i32s = |a: usize, n: usize| {
            bytes[a..a + 4 * n]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        Ok(Self {
            features,
            hidden,
            buckets,
            qa: 255,
            qb: 64,
            eval_scale: 600.,
            bias: i16s(36, hidden),
            weights: i16s(features_start, features * hidden),
            output: i16s(output_start, buckets * 2 * hidden),
            output_bias: i32s(bias_start, buckets),
            dense: Some(DenseHead {
                weights: bytes[dense_start..dense_bias_start]
                    .iter()
                    .map(|&b| b as i8)
                    .collect(),
                bias: i32s(dense_bias_start, 32),
                output: i16s(residual_start, buckets * 32),
            }),
            avx2: has_avx2(),
            vnni: has_vnni(),
        })
    }

    pub fn backend(&self) -> &'static str {
        if self.avx2 && self.vnni && self.dense.is_some() {
            "avx512vnni"
        } else if self.avx2 {
            "avx2"
        } else {
            "scalar"
        }
    }

    /// Benchmark/verification helper; AVX2 availability is still runtime checked.
    pub fn disable_vnni(&mut self) {
        self.vnni = false;
    }

    #[inline]
    fn feature(&self, piece: u8, square: usize) -> &[i16] {
        let start = ((piece as usize - 1) * N_SQ + square) * self.hidden;
        &self.weights[start..start + self.hidden]
    }

    #[inline]
    fn threat_feature(&self, piece: u8, square: usize) -> &[i16] {
        let start = (FEATURES + (piece as usize - 1) * N_SQ + square) * self.hidden;
        &self.weights[start..start + self.hidden]
    }

    pub fn refresh(&self, board: &Board, since_capture: u32, clock: u32) -> Accumulator {
        assert!(
            (50..=200).contains(&clock),
            "capture clock must be 50..=200"
        );
        let mut acc = Accumulator::empty(self.hidden);
        for j in 0..self.hidden {
            acc.own[j] = self.bias[j] as i32;
            acc.opponent[j] = self.bias[j] as i32;
        }
        for (square, cell) in board.iter().enumerate() {
            let piece = cell.code();
            if piece == 0 {
                continue;
            }
            assert!(piece <= 6, "invalid piece code");
            let own = self.feature(piece, square);
            let opp = self.feature(swap_side(piece), mirror_anti(square));
            for j in 0..self.hidden {
                acc.own[j] += own[j] as i32;
                acc.opponent[j] += opp[j] as i32;
            }
            if is_attacked(board, square as u8) {
                let own = self.threat_feature(piece, square);
                let opp = self.threat_feature(swap_side(piece), mirror_anti(square));
                for j in 0..self.hidden {
                    acc.own[j] += own[j] as i32;
                    acc.opponent[j] += opp[j] as i32;
                }
            }
        }
        acc.pieces = board.iter().filter(|&&cell| cell != Cell::Empty).count();
        acc.clock = clock;
        acc.clock_rows = clock_rows(since_capture, clock);
        for row in acc.clock_rows {
            let weights = &self.weights[row * self.hidden..(row + 1) * self.hidden];
            self.add_row(&mut acc.own, weights, true);
            self.add_row(&mut acc.opponent, weights, true);
        }
        if self.features == 1022 {
            let (mover, opponent) = engine::race::race_buckets(board);
            acc.race = [mover, opponent];
            for perspective in 0..2 {
                for side in 0..2 {
                    let row = 1004 + 9 * side + acc.race[side ^ perspective] as usize;
                    let values = if perspective == 0 {
                        &mut acc.own
                    } else {
                        &mut acc.opponent
                    };
                    self.add_row(
                        values,
                        &self.weights[row * self.hidden..(row + 1) * self.hidden],
                        true,
                    );
                }
            }
        }
        acc
    }

    /// Child canonicalization swaps the two parent perspectives; only moved and
    /// captured piece-square and changed attack/clock/race rows need updating.
    /// The caller supplies a legal move and the counter returned by engine::apply.
    pub fn update(
        &self,
        parent: &Accumulator,
        board: &Board,
        action: u16,
        child_since_capture: u32,
        child: &mut Accumulator,
    ) {
        self.update_deferred(parent, board, action, child_since_capture, child);
        if self.features == 1022 {
            let (from, to) = action_from_to(action);
            let mut moved = *board;
            moved[to] = moved[from];
            moved[from] = Cell::Empty;
            self.resolve_race(&engine::flip(&moved), child);
        }
    }

    /// Carry existing race rows through the frame swap until a value is needed.
    pub(crate) fn update_deferred(
        &self,
        parent: &Accumulator,
        board: &Board,
        action: u16,
        child_since_capture: u32,
        child: &mut Accumulator,
    ) {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Accumulator);
        let (from, to) = action_from_to(action);
        let piece = board[from].code();
        let captured = board[to].code();
        self.update_half(
            &parent.opponent,
            &mut child.own,
            self.feature(swap_side(piece), mirror_anti(from)),
            self.feature(swap_side(piece), mirror_anti(to)),
            (captured != 0).then(|| self.feature(swap_side(captured), mirror_anti(to))),
        );
        self.update_half(
            &parent.own,
            &mut child.opponent,
            self.feature(piece, from),
            self.feature(piece, to),
            (captured != 0).then(|| self.feature(captured, to)),
        );
        self.update_threats(board, from, to, child);
        child.pieces = parent.pieces - usize::from(captured != 0);
        child.clock = parent.clock;
        child.clock_rows = clock_rows(child_since_capture, parent.clock);
        for (old, new) in parent.clock_rows.into_iter().zip(child.clock_rows) {
            if old != new {
                let old_row = &self.weights[old * self.hidden..(old + 1) * self.hidden];
                let new_row = &self.weights[new * self.hidden..(new + 1) * self.hidden];
                for values in [&mut child.own, &mut child.opponent] {
                    self.add_row(values, old_row, false);
                    self.add_row(values, new_row, true);
                }
            }
        }
        child.race = [parent.race[1], parent.race[0]];
        child.race_dirty = self.features == 1022;
    }

    /// Resolve pending rows against this accumulator's canonical board.
    pub(crate) fn resolve_race(&self, board: &Board, acc: &mut Accumulator) {
        if !acc.race_dirty {
            return;
        }
        #[cfg(feature = "profile")]
        let query_probe = crate::profile::Probe::new(crate::profile::Zone::RaceQuery);
        let (mover, opponent) = engine::race::race_buckets(board);
        #[cfg(feature = "profile")]
        drop(query_probe);
        #[cfg(feature = "profile")]
        let _rows_probe = crate::profile::Probe::new(crate::profile::Zone::RaceRows);
        let next = [mover, opponent];
        for perspective in 0..2 {
            for side in 0..2 {
                let old = acc.race[side ^ perspective];
                let new = next[side ^ perspective];
                if old != new {
                    let values = if perspective == 0 {
                        &mut acc.own
                    } else {
                        &mut acc.opponent
                    };
                    for (bucket, add) in [(old, false), (new, true)] {
                        let row = 1004 + 9 * side + bucket as usize;
                        self.add_row(
                            values,
                            &self.weights[row * self.hidden..(row + 1) * self.hidden],
                            add,
                        );
                    }
                }
            }
        }
        acc.race = next;
        acc.race_dirty = false;
    }

    /// Only squares adjacent to the old/new location can change attack status.
    /// Their union contains at most 14 squares for a diagonal king move.
    fn update_threats(&self, board: &Board, from: usize, to: usize, child: &mut Accumulator) {
        let mut moved = *board;
        moved[to] = moved[from];
        moved[from] = Cell::Empty;
        let mut seen = [false; N_SQ];
        let mut affected = [0usize; 18];
        let mut count = 0;
        for center in [from, to] {
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let r = center as i32 / 9 + dr;
                    let c = center as i32 % 9 + dc;
                    if !(0..9).contains(&r) || !(0..9).contains(&c) {
                        continue;
                    }
                    let sq = (r * 9 + c) as usize;
                    if !seen[sq] {
                        seen[sq] = true;
                        affected[count] = sq;
                        count += 1;
                    }
                }
            }
        }
        for &sq in &affected[..count] {
            let old_piece = board[sq].code();
            let new_piece = moved[sq].code();
            let old_threat = old_piece != 0 && is_attacked(board, sq as u8);
            let new_threat = new_piece != 0 && is_attacked(&moved, sq as u8);
            if old_threat && (!new_threat || old_piece != new_piece) {
                self.add_row(
                    &mut child.opponent,
                    self.threat_feature(old_piece, sq),
                    false,
                );
                self.add_row(
                    &mut child.own,
                    self.threat_feature(swap_side(old_piece), mirror_anti(sq)),
                    false,
                );
            }
            if new_threat && (!old_threat || old_piece != new_piece) {
                self.add_row(
                    &mut child.opponent,
                    self.threat_feature(new_piece, sq),
                    true,
                );
                self.add_row(
                    &mut child.own,
                    self.threat_feature(swap_side(new_piece), mirror_anti(sq)),
                    true,
                );
            }
        }
    }

    #[inline]
    fn add_row(&self, acc: &mut [i32], row: &[i16], add: bool) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            unsafe {
                add_row_avx2(acc, row, add);
            }
            return;
        }
        let sign = if add { 1 } else { -1 };
        for j in 0..self.hidden {
            acc[j] += sign * row[j] as i32;
        }
    }

    #[inline]
    fn update_half(
        &self,
        src: &[i32],
        dst: &mut [i32],
        sub: &[i16],
        add: &[i16],
        cap: Option<&[i16]>,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // Feature rows and buffers have H entries; H is a multiple of 32.
            unsafe {
                update_avx2(src, dst, sub, add, cap);
            }
            return;
        }
        for j in 0..self.hidden {
            dst[j] = src[j] - sub[j] as i32 + add[j] as i32 - cap.map_or(0, |v| v[j] as i32);
        }
    }

    fn bucket(&self, acc: &Accumulator) -> usize {
        if self.buckets == 1 {
            0
        } else {
            match acc.pieces {
                0..=4 => 0,
                5..=8 => 1,
                9..=12 => 2,
                _ => 3,
            }
        }
    }

    pub fn sum_scalar(&self, acc: &Accumulator) -> i64 {
        let start = self.bucket(acc) * 2 * self.hidden;
        let mut sum = 0i64;
        for (perspective, values) in [&acc.own, &acc.opponent].iter().enumerate() {
            for (j, &value) in values.iter().enumerate() {
                let x = value.clamp(0, self.qa) as i64;
                sum += x * x * self.output[start + perspective * self.hidden + j] as i64;
            }
        }
        sum
    }

    pub fn sum(&self, acc: &Accumulator) -> i64 {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Readout);
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            let start = self.bucket(acc) * 2 * self.hidden;
            let output = &self.output[start..start + 2 * self.hidden];
            unsafe {
                return sum_avx2(&acc.own, &output[..self.hidden], self.qa)
                    + sum_avx2(&acc.opponent, &output[self.hidden..], self.qa);
            }
        }
        self.sum_scalar(acc)
    }

    pub fn raw(&self, acc: &Accumulator) -> f64 {
        debug_assert!(
            !acc.race_dirty,
            "race rows must be resolved before evaluation"
        );
        self.sum(acc) as f64 / (self.qa as f64 * self.qa as f64 * self.qb as f64)
            + self.output_bias[self.bucket(acc)] as f64 / self.qb as f64
            + self.dense_sum(acc, self.avx2) as f64 / (self.qa as f64 * self.qb as f64)
    }

    pub fn raw_scalar(&self, acc: &Accumulator) -> f64 {
        debug_assert!(
            !acc.race_dirty,
            "race rows must be resolved before evaluation"
        );
        self.sum_scalar(acc) as f64 / (self.qa as f64 * self.qa as f64 * self.qb as f64)
            + self.output_bias[self.bucket(acc)] as f64 / self.qb as f64
            + self.dense_sum(acc, false) as f64 / (self.qa as f64 * self.qb as f64)
    }

    fn dense_sum(&self, acc: &Accumulator, avx2: bool) -> i64 {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Dense);
        let Some(head) = &self.dense else {
            return 0;
        };
        let start = self.bucket(acc) * DENSE_WIDTH;
        let output = &head.output[start..start + DENSE_WIDTH];
        if output.iter().all(|&w| w == 0) {
            return 0;
        }
        DENSE_SCRATCH.with_borrow_mut(|scratch| {
            let input = &mut scratch.input;
            input.resize(2 * self.hidden, 0);
            for (offset, values) in [(0, &acc.own), (self.hidden, &acc.opponent)] {
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    unsafe {
                        if self.vnni {
                            quantize_screlu_avx512(
                                values,
                                &mut input[offset..offset + self.hidden],
                            );
                        } else {
                            quantize_screlu_avx2(values, &mut input[offset..offset + self.hidden]);
                        }
                    }
                    continue;
                }
                for (j, &value) in values.iter().enumerate() {
                    let x = value.clamp(0, 255);
                    // The denominator is odd, so an exact half tie is impossible.
                    input[offset + j] = ((x * x + 127) / 255) as u8;
                }
            }
            let input = &input[..2 * self.hidden];
            let mut active = [0usize; DENSE_WIDTH];
            let mut count = 0;
            for (row, &weight) in output.iter().enumerate() {
                if weight != 0 {
                    active[count] = row;
                    count += 1;
                }
            }
            let active = &active[..count];
            let mut dots = [0i64; DENSE_WIDTH];
            #[cfg(target_arch = "x86_64")]
            if avx2 && self.vnni {
                unsafe {
                    dense_sparse_vnni(input, &head.weights, active, &mut dots);
                }
            } else if avx2 {
                // Widen once, then reuse these input lanes for every output row.
                let widened = &mut scratch.widened;
                widened.resize(input.len(), 0);
                for (to, &from) in widened.iter_mut().zip(input) {
                    *to = from as i16;
                }
                unsafe {
                    dense_sparse_avx2(&widened[..input.len()], &head.weights, active, &mut dots);
                }
            } else {
                for (i, &row) in active.iter().enumerate() {
                    dots[i] = dense_dot_scalar_u8(
                        input,
                        &head.weights[row * input.len()..(row + 1) * input.len()],
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = avx2;
                for (i, &row) in active.iter().enumerate() {
                    dots[i] = dense_dot_scalar_u8(
                        input,
                        &head.weights[row * input.len()..(row + 1) * input.len()],
                    );
                }
            }
            let mut sum = 0i64;
            for (i, &row) in active.iter().enumerate() {
                let value = dots[i] + head.bias[row] as i64;
                let activation = round_div64_clipped(value);
                sum += activation as i64 * output[row] as i64;
            }
            sum
        })
    }

    pub fn evaluate(&self, acc: &Accumulator) -> i32 {
        (self.raw(acc) * self.eval_scale as f64)
            .round_ties_even()
            .clamp(-28_000., 28_000.) as i32
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn quantize_screlu_avx512(acc: &[i32], out: &mut [u8]) {
    use std::arch::x86_64::*;
    for i in (0..acc.len()).step_by(16) {
        let x = _mm512_min_epi32(
            _mm512_set1_epi32(255),
            _mm512_max_epi32(
                _mm512_setzero_si512(),
                _mm512_loadu_si512(acc.as_ptr().add(i).cast()),
            ),
        );
        let z = _mm512_add_epi32(_mm512_mullo_epi32(x, x), _mm512_set1_epi32(128));
        let q = _mm512_srli_epi32(_mm512_add_epi32(z, _mm512_srli_epi32(z, 8)), 8);
        _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm512_cvtepi32_epi8(q));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_screlu_avx2(acc: &[i32], out: &mut [u8]) {
    use std::arch::x86_64::*;
    for i in (0..acc.len()).step_by(8) {
        let x = _mm256_min_epi32(
            _mm256_set1_epi32(255),
            _mm256_max_epi32(
                _mm256_setzero_si256(),
                _mm256_loadu_si256(acc.as_ptr().add(i).cast()),
            ),
        );
        let z = _mm256_add_epi32(_mm256_mullo_epi32(x, x), _mm256_set1_epi32(128));
        let q = _mm256_srli_epi32(_mm256_add_epi32(z, _mm256_srli_epi32(z, 8)), 8);
        let packed16 = _mm_packus_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        _mm_storel_epi64(
            out.as_mut_ptr().add(i).cast(),
            _mm_packus_epi16(packed16, packed16),
        );
    }
}

#[inline]
fn round_div64_clipped(value: i64) -> i32 {
    if value <= 0 {
        return 0;
    }
    if value >= 255 * 64 {
        return 255;
    }
    let quotient = value >> 6;
    let remainder = value & 63;
    (quotient + i64::from(remainder > 32 || (remainder == 32 && quotient & 1 != 0))) as i32
}

fn dense_dot_scalar_u8(input: &[u8], weights: &[i8]) -> i64 {
    input
        .iter()
        .zip(weights)
        .map(|(&x, &w)| x as i64 * w as i64)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dense_sparse_vnni(input: &[u8], weights: &[i8], rows: &[usize], dots: &mut [i64]) {
    use std::arch::x86_64::*;
    let grouped = rows.len() / 4 * 4;
    for group in (0..grouped).step_by(4) {
        let w0 = weights.as_ptr().add(rows[group] * input.len());
        let w1 = weights.as_ptr().add(rows[group + 1] * input.len());
        let w2 = weights.as_ptr().add(rows[group + 2] * input.len());
        let w3 = weights.as_ptr().add(rows[group + 3] * input.len());
        for chunk in (0..input.len()).step_by(DOT_CHUNK) {
            let end = (chunk + DOT_CHUNK).min(input.len());
            let mut s0 = _mm512_setzero_si512();
            let mut s1 = s0;
            let mut s2 = s0;
            let mut s3 = s0;
            for i in (chunk..end).step_by(64) {
                let x = _mm512_loadu_si512(input.as_ptr().add(i).cast());
                s0 = _mm512_dpbusd_epi32(s0, x, _mm512_loadu_si512(w0.add(i).cast()));
                s1 = _mm512_dpbusd_epi32(s1, x, _mm512_loadu_si512(w1.add(i).cast()));
                s2 = _mm512_dpbusd_epi32(s2, x, _mm512_loadu_si512(w2.add(i).cast()));
                s3 = _mm512_dpbusd_epi32(s3, x, _mm512_loadu_si512(w3.add(i).cast()));
            }
            dots[group] += i64::from(_mm512_reduce_add_epi32(s0));
            dots[group + 1] += i64::from(_mm512_reduce_add_epi32(s1));
            dots[group + 2] += i64::from(_mm512_reduce_add_epi32(s2));
            dots[group + 3] += i64::from(_mm512_reduce_add_epi32(s3));
        }
    }
    for row in grouped..rows.len() {
        let w = weights.as_ptr().add(rows[row] * input.len());
        for chunk in (0..input.len()).step_by(DOT_CHUNK) {
            let end = (chunk + DOT_CHUNK).min(input.len());
            let mut sum = _mm512_setzero_si512();
            for i in (chunk..end).step_by(64) {
                sum = _mm512_dpbusd_epi32(
                    sum,
                    _mm512_loadu_si512(input.as_ptr().add(i).cast()),
                    _mm512_loadu_si512(w.add(i).cast()),
                );
            }
            dots[row] += i64::from(_mm512_reduce_add_epi32(sum));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dense_dot_avx2(input: &[i16], weights: &[i8]) -> i64 {
    use std::arch::x86_64::*;
    let mut total = 0i64;
    for chunk in (0..input.len()).step_by(DOT_CHUNK) {
        let end = (chunk + DOT_CHUNK).min(input.len());
        let mut sum = _mm256_setzero_si256();
        for i in (chunk..end).step_by(16) {
            let x = _mm256_loadu_si256(input.as_ptr().add(i).cast());
            let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(weights.as_ptr().add(i).cast()));
            // Widen before multiplication: maddubs would saturate signed-i16 pairs.
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x, w));
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum);
        total += lanes.iter().map(|&v| i64::from(v)).sum::<i64>();
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dense_sparse_avx2(input: &[i16], weights: &[i8], rows: &[usize], dots: &mut [i64]) {
    use std::arch::x86_64::*;
    let grouped = rows.len() / 4 * 4;
    for group in (0..grouped).step_by(4) {
        let w0 = weights.as_ptr().add(rows[group] * input.len());
        let w1 = weights.as_ptr().add(rows[group + 1] * input.len());
        let w2 = weights.as_ptr().add(rows[group + 2] * input.len());
        let w3 = weights.as_ptr().add(rows[group + 3] * input.len());
        for chunk in (0..input.len()).step_by(DOT_CHUNK) {
            let end = (chunk + DOT_CHUNK).min(input.len());
            let mut s0 = _mm256_setzero_si256();
            let mut s1 = s0;
            let mut s2 = s0;
            let mut s3 = s0;
            for i in (chunk..end).step_by(16) {
                let x = _mm256_loadu_si256(input.as_ptr().add(i).cast());
                s0 = _mm256_add_epi32(
                    s0,
                    _mm256_madd_epi16(x, _mm256_cvtepi8_epi16(_mm_loadu_si128(w0.add(i).cast()))),
                );
                s1 = _mm256_add_epi32(
                    s1,
                    _mm256_madd_epi16(x, _mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(i).cast()))),
                );
                s2 = _mm256_add_epi32(
                    s2,
                    _mm256_madd_epi16(x, _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(i).cast()))),
                );
                s3 = _mm256_add_epi32(
                    s3,
                    _mm256_madd_epi16(x, _mm256_cvtepi8_epi16(_mm_loadu_si128(w3.add(i).cast()))),
                );
            }
            for (index, vector) in [s0, s1, s2, s3].into_iter().enumerate() {
                let sum = _mm_add_epi32(
                    _mm256_castsi256_si128(vector),
                    _mm256_extracti128_si256(vector, 1),
                );
                let sum = _mm_hadd_epi32(sum, sum);
                dots[group + index] += i64::from(_mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum)));
            }
        }
    }
    for i in grouped..rows.len() {
        let start = rows[i] * input.len();
        dots[i] = dense_dot_avx2(input, &weights[start..start + input.len()]);
    }
}

fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn has_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_avx2(src: &[i32], dst: &mut [i32], sub: &[i16], add: &[i16], cap: Option<&[i16]>) {
    use std::arch::x86_64::*;
    for j in (0..src.len()).step_by(8) {
        let a = _mm256_loadu_si256(src.as_ptr().add(j).cast());
        let s = _mm256_cvtepi16_epi32(_mm_loadu_si128(sub.as_ptr().add(j).cast()));
        let p = _mm256_cvtepi16_epi32(_mm_loadu_si128(add.as_ptr().add(j).cast()));
        let mut value = _mm256_add_epi32(_mm256_sub_epi32(a, s), p);
        if let Some(cap) = cap {
            value = _mm256_sub_epi32(
                value,
                _mm256_cvtepi16_epi32(_mm_loadu_si128(cap.as_ptr().add(j).cast())),
            );
        }
        _mm256_storeu_si256(dst.as_mut_ptr().add(j).cast(), value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_row_avx2(acc: &mut [i32], row: &[i16], add: bool) {
    use std::arch::x86_64::*;
    for j in (0..acc.len()).step_by(8) {
        let old = _mm256_loadu_si256(acc.as_ptr().add(j).cast());
        let delta = _mm256_cvtepi16_epi32(_mm_loadu_si128(row.as_ptr().add(j).cast()));
        let value = if add {
            _mm256_add_epi32(old, delta)
        } else {
            _mm256_sub_epi32(old, delta)
        };
        _mm256_storeu_si256(acc.as_mut_ptr().add(j).cast(), value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(acc: &[i32], weights: &[i16], qa: i32) -> i64 {
    use std::arch::x86_64::*;
    let mut even = _mm256_setzero_si256();
    let mut odd = _mm256_setzero_si256();
    for j in (0..acc.len()).step_by(8) {
        let x = _mm256_min_epi32(
            _mm256_set1_epi32(qa),
            _mm256_max_epi32(
                _mm256_setzero_si256(),
                _mm256_loadu_si256(acc.as_ptr().add(j).cast()),
            ),
        );
        let squared = _mm256_mullo_epi32(x, x);
        let w = _mm256_cvtepi16_epi32(_mm_loadu_si128(weights.as_ptr().add(j).cast()));
        even = _mm256_add_epi64(even, _mm256_mul_epi32(squared, w));
        odd = _mm256_add_epi64(
            odd,
            _mm256_mul_epi32(_mm256_srli_epi64(squared, 32), _mm256_srli_epi64(w, 32)),
        );
    }
    let mut sums = [0i64; 4];
    _mm256_storeu_si256(sums.as_mut_ptr().cast(), _mm256_add_epi64(even, odd));
    sums.iter().sum()
}

/// Convert prototype v3 weights to the migration format without changing arithmetic.
pub fn convert_v3(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.len() != 1_031_400 || &bytes[..8] != b"RPSNNUE1" {
        return Err("expected prototype v3 H512/D32 file".into());
    }
    let u = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    if [u(8), u(12), u(16), u(20), u(24)] != [3, 972, 512, 255, 64] || u(998436) != 32 {
        return Err("expected prototype v3 F972/H512/D32".into());
    }
    let mut out = bytes[..32].to_vec();
    out[8..12].copy_from_slice(&6u32.to_le_bytes());
    out[12..16].copy_from_slice(&1004u32.to_le_bytes());
    out.extend(1u32.to_le_bytes());
    let end = 32 + 2 * 512 + 2 * 972 * 512;
    out.extend(&bytes[32..end]);
    out.resize(out.len() + 32 * 512 * 2, 0);
    out.extend(&bytes[end..]);
    Model::from_bytes(&out)?;
    Ok(out)
}

#[cfg(test)]
mod width_tests {
    use super::*;

    #[test]
    fn dense_dots_do_not_overflow_i32_at_large_widths() {
        let n = 131072 + 64;
        let input = vec![255u8; n];
        let mut weights = Vec::new();
        for weight in [-128i8, 127, -127, 126, 125] {
            weights.extend(std::iter::repeat_n(weight, n));
        }
        let rows = [0, 1, 2, 3, 4];
        let expected: Vec<_> = rows
            .iter()
            .map(|&r| dense_dot_scalar_u8(&input, &weights[r * n..(r + 1) * n]))
            .collect();
        assert!(expected[0] < i64::from(i32::MIN));
        assert!(expected[1] > i64::from(i32::MAX));
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            let mut actual = vec![0; 5];
            let widened: Vec<_> = input.iter().map(|&x| i16::from(x)).collect();
            unsafe {
                dense_sparse_avx2(&widened, &weights, &rows, &mut actual);
            }
            assert_eq!(actual, expected);
        }
        #[cfg(target_arch = "x86_64")]
        if has_vnni() {
            let mut actual = vec![0; 5];
            unsafe {
                dense_sparse_vnni(&input, &weights, &rows, &mut actual);
            }
            assert_eq!(actual, expected);
        }
    }
}

#[cfg(test)]
mod deferred_tests {
    use super::*;
    use engine::{apply, Outcome, Rules, State};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn pending_race_rows_survive_multiple_unresolved_ancestors() {
        let model = Model::from_bytes(include_bytes!("../tests/fixtures/h256_race.nnue")).unwrap();
        let mut rng = StdRng::seed_from_u64(2026091101);
        let mut state = State::initial();
        let mut acc = model.refresh(&state.board, 0, 200);
        let mut captures = 0;
        let mut pending_ancestors = 0;
        for turn in 0..2000 {
            pending_ancestors += usize::from(acc.race_dirty);
            let legal = state.legal_actions();
            let action = legal[rng.random_range(0..legal.len())];
            let (child, end) = apply(&Rules::SITE, &state, action);
            captures += usize::from(child.since_capture == 0);
            let mut next = Accumulator::empty(model.hidden);
            model.update_deferred(&acc, &state.board, action, child.since_capture, &mut next);
            let mut resolved = next.clone();
            model.resolve_race(&child.board, &mut resolved);
            assert_eq!(
                resolved,
                model.refresh(&child.board, child.since_capture, 200)
            );
            assert_eq!(model.raw(&resolved), model.raw_scalar(&resolved));
            let mut repeated = resolved.clone();
            model.resolve_race(&child.board, &mut repeated);
            assert_eq!(repeated, resolved);
            state = child;
            acc = if turn % 7 == 0 { resolved } else { next };
            if end != Outcome::Ongoing {
                state = State::initial();
                acc = model.refresh(&state.board, 0, 200);
            }
        }
        assert!(captures > 10 && pending_ancestors > 1000);
    }
}
