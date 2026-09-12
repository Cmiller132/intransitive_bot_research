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

const CLOCK_BOUNDS: [u32; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160];
fn clock_rows(features: usize, since: u32, clock: u32) -> [usize; 2] {
    [
        features - 32 + CLOCK_BOUNDS.partition_point(|&b| b <= since) - 1,
        features - 16 + CLOCK_BOUNDS.partition_point(|&b| b <= clock.saturating_sub(since)) - 1,
    ]
}
use std::{fs, path::Path};

pub const FEATURES: usize = 6 * N_SQ;
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
    pub qa: i32,
    pub qb: i32,
    pub eval_scale: f32,
    pub bias: Vec<i16>,
    pub weights: Vec<i16>,
    pub output: Vec<i16>,
    pub output_bias: i32,
    pub dense: DenseHead,
    avx2: bool,
    vnni: bool,
}

/// Exact material and clocks travel independently of deferred vector values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FeatureState {
    pub counts: [[u8; 3]; 2],
    pub since: u32,
    pub clock: u32,
}
impl FeatureState {
    pub fn new(board: &Board, since: u32, clock: u32) -> Self {
        let mut counts = [[0; 3]; 2];
        for cell in board {
            let code = cell.code() as usize;
            if code != 0 {
                counts[(code - 1) / 3][(code - 1) % 3] += 1;
            }
        }
        Self {
            counts,
            since,
            clock,
        }
    }
    pub fn contexts(self) -> [usize; 2] {
        let context =
            |c: [u8; 3]| c[0].min(2) as usize + 3 * c[1].min(2) as usize + 9 * c[2].min(2) as usize;
        [context(self.counts[1]), context(self.counts[0])]
    }
    pub fn after(self, board: &Board, action: u16, since: u32) -> Self {
        let mut counts = [self.counts[1], self.counts[0]];
        let captured = board[action_from_to(action).1].code();
        if captured != 0 {
            debug_assert!(captured >= 4);
            counts[0][(captured - 4) as usize] -= 1;
        }
        Self {
            counts,
            since,
            clock: self.clock,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accumulator {
    pub own: Vec<i32>,
    pub opponent: Vec<i32>,
    pub(crate) state: FeatureState,
}
impl Accumulator {
    pub fn empty(hidden: usize) -> Self {
        Self {
            own: vec![0; hidden],
            opponent: vec![0; hidden],
            state: FeatureState::default(),
        }
    }
    pub fn contexts(&self) -> [usize; 2] {
        self.state.contexts()
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
        if !matches!((u(8), features), (6, 1004) | (8, 13640))
            || [u(20), u(24)] != [255, 64]
            || f32::from_le_bytes(bytes[28..32].try_into().unwrap()) != 600.
            || buckets != 1
        {
            return Err("unsupported NNUE format, dimensions, scales or buckets".into());
        }
        if !(32..=1024).contains(&hidden) || !hidden.is_multiple_of(HIDDEN_LANES) {
            return Err("hidden width must be a multiple of 32 in 32..1024".into());
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
            qa: 255,
            qb: 64,
            eval_scale: 600.,
            bias: i16s(36, hidden),
            weights: i16s(features_start, features * hidden),
            output: i16s(output_start, buckets * 2 * hidden),
            output_bias: i32s(bias_start, 1)[0],
            dense: DenseHead {
                weights: bytes[dense_start..dense_bias_start]
                    .iter()
                    .map(|&b| b as i8)
                    .collect(),
                bias: i32s(dense_bias_start, 32),
                output: i16s(residual_start, 32),
            },
            avx2: has_avx2(),
            vnni: has_vnni(),
        })
    }

    pub fn backend(&self) -> &'static str {
        if self.avx2 && self.vnni {
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

    #[cfg(test)]
    pub(crate) fn scalar_clone(&self) -> Self {
        let mut scalar = self.clone();
        scalar.avx2 = false;
        scalar.vnni = false;
        scalar
    }

    #[inline]
    fn feature(&self, piece: u8, square: usize, context: usize) -> &[i16] {
        let row = context * FEATURES + (piece as usize - 1) * N_SQ + square;
        self.row(row)
    }
    #[inline]
    fn row(&self, row: usize) -> &[i16] {
        &self.weights[row * self.hidden..(row + 1) * self.hidden]
    }
    #[inline]
    fn threat_feature(&self, piece: u8, square: usize) -> &[i16] {
        self.row(self.features - 518 + (piece as usize - 1) * N_SQ + square)
    }
    pub(crate) fn requires_refresh(
        &self,
        before: FeatureState,
        after: FeatureState,
        side: usize,
    ) -> bool {
        self.features == 13640 && before.contexts()[side ^ 1] != after.contexts()[side]
    }

    /// Active rows in the file layout; padding occupies the unused slots.
    pub fn feature_ids(&self, board: &Board, since: u32, clock: u32) -> [[usize; 42]; 2] {
        self.ids(board, FeatureState::new(board, since, clock))
    }
    fn ids(&self, board: &Board, state: FeatureState) -> [[usize; 42]; 2] {
        let contexts = if self.features == 13640 {
            state.contexts()
        } else {
            [0; 2]
        };
        let mut ids = [[self.features; 42]; 2];
        let (mut pieces, mut threats) = (0, 20);
        for (square, cell) in board.iter().enumerate() {
            let piece = cell.code();
            if piece == 0 {
                continue;
            }
            assert!(pieces < 20, "at most twenty pieces fit the feature slots");
            ids[0][pieces] = contexts[0] * FEATURES + (piece as usize - 1) * N_SQ + square;
            ids[1][pieces] = contexts[1] * FEATURES
                + (swap_side(piece) as usize - 1) * N_SQ
                + mirror_anti(square);
            pieces += 1;
            if is_attacked(board, square as u8) {
                ids[0][threats] = self.features - 518 + (piece as usize - 1) * N_SQ + square;
                ids[1][threats] = self.features - 518
                    + (swap_side(piece) as usize - 1) * N_SQ
                    + mirror_anti(square);
                threats += 1;
            }
        }
        for side in &mut ids {
            side[40..].copy_from_slice(&clock_rows(self.features, state.since, state.clock));
        }
        ids
    }
    pub fn refresh(&self, board: &Board, since: u32, clock: u32) -> Accumulator {
        let mut acc = Accumulator::empty(self.hidden);
        self.refresh_into(board, since, clock, &mut acc);
        acc
    }
    pub(crate) fn refresh_into(
        &self,
        board: &Board,
        since: u32,
        clock: u32,
        acc: &mut Accumulator,
    ) -> u64 {
        assert!(
            (50..=200).contains(&clock),
            "capture clock must be 50..=200"
        );
        acc.state = FeatureState::new(board, since, clock);
        let ids = self.ids(board, acc.state);
        self.fill_half(&ids[0], &mut acc.own) + self.fill_half(&ids[1], &mut acc.opponent)
    }
    fn fill_half(&self, ids: &[usize; 42], dst: &mut [i32]) -> u64 {
        for (value, &bias) in dst.iter_mut().zip(&self.bias) {
            *value = bias as i32;
        }
        let mut rows = 0;
        for &id in ids {
            if id != self.features {
                self.add_row(dst, self.row(id), true);
                rows += 1;
            }
        }
        rows
    }

    /// Update both canonical halves, refreshing only a changed capped context.
    pub fn update(
        &self,
        parent: &Accumulator,
        board: &Board,
        action: u16,
        since: u32,
        child: &mut Accumulator,
    ) {
        let after = parent.state.after(board, action, since);
        self.update_perspective(
            &parent.opponent,
            board,
            action,
            parent.state,
            after,
            0,
            &mut child.own,
        );
        self.update_perspective(
            &parent.own,
            board,
            action,
            parent.state,
            after,
            1,
            &mut child.opponent,
        );
        child.state = after;
    }

    /// Materialise one half into its preallocated vector; returns added/removed rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_perspective(
        &self,
        src: &[i32],
        board: &Board,
        action: u16,
        before: FeatureState,
        after: FeatureState,
        side: usize,
        dst: &mut [i32],
    ) -> (u64, u64) {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Accumulator);
        let (from, to) = action_from_to(action);
        let piece = board[from].code();
        let captured = board[to].code();
        let mut moved = *board;
        moved[to] = moved[from];
        moved[from] = Cell::Empty;
        if self.requires_refresh(before, after, side) {
            let ids = self.ids(&engine::flip(&moved), after);
            return (self.fill_half(&ids[side], dst), 0);
        }
        let context = if self.features == 13640 {
            after.contexts()[side]
        } else {
            0
        };
        let transform = |piece: u8, square: usize| {
            if side == 0 {
                (swap_side(piece), mirror_anti(square))
            } else {
                (piece, square)
            }
        };
        let feature = |piece, square| {
            let (piece, square) = transform(piece, square);
            self.feature(piece, square, context)
        };
        self.update_half(
            src,
            dst,
            feature(piece, from),
            feature(piece, to),
            (captured != 0).then(|| feature(captured, to)),
        );
        let (mut added, mut removed) = (1, 1 + u64::from(captured != 0));
        let mut affected = engine::tactics::attack_candidates(board, action);
        while affected != 0 {
            let square = affected.trailing_zeros() as usize;
            affected &= affected - 1;
            let old = board[square].code();
            let new = moved[square].code();
            let old_threat = old != 0 && is_attacked(board, square as u8);
            let new_threat = new != 0 && is_attacked(&moved, square as u8);
            if old_threat && (!new_threat || old != new) {
                let (piece, square) = transform(old, square);
                self.add_row(dst, self.threat_feature(piece, square), false);
                removed += 1;
            }
            if new_threat && (!old_threat || old != new) {
                let (piece, square) = transform(new, square);
                self.add_row(dst, self.threat_feature(piece, square), true);
                added += 1;
            }
        }
        for (old, new) in clock_rows(self.features, before.since, before.clock)
            .into_iter()
            .zip(clock_rows(self.features, after.since, after.clock))
        {
            if old != new {
                self.add_row(dst, self.row(old), false);
                self.add_row(dst, self.row(new), true);
                added += 1;
                removed += 1;
            }
        }
        (added, removed)
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

    pub fn sum_scalar(&self, acc: &Accumulator) -> i64 {
        let start = 0;
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
        if self.vnni {
            let start = 0;
            let output = &self.output[start..start + 2 * self.hidden];
            unsafe {
                return sum_avx512(&acc.own, &output[..self.hidden], self.qa)
                    + sum_avx512(&acc.opponent, &output[self.hidden..], self.qa);
            }
        }
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            let start = 0;
            let output = &self.output[start..start + 2 * self.hidden];
            unsafe {
                return sum_avx2(&acc.own, &output[..self.hidden], self.qa)
                    + sum_avx2(&acc.opponent, &output[self.hidden..], self.qa);
            }
        }
        self.sum_scalar(acc)
    }

    pub fn raw(&self, acc: &Accumulator) -> f64 {
        self.sum(acc) as f64 / (self.qa as f64 * self.qa as f64 * self.qb as f64)
            + self.output_bias as f64 / self.qb as f64
            + self.dense_sum(acc, self.avx2) as f64 / (self.qa as f64 * self.qb as f64)
    }

    pub fn raw_scalar(&self, acc: &Accumulator) -> f64 {
        self.sum_scalar(acc) as f64 / (self.qa as f64 * self.qa as f64 * self.qb as f64)
            + self.output_bias as f64 / self.qb as f64
            + self.dense_sum(acc, false) as f64 / (self.qa as f64 * self.qb as f64)
    }

    fn dense_sum(&self, acc: &Accumulator, avx2: bool) -> i64 {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Dense);
        let head = &self.dense;
        let output = &head.output;
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
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn sum_avx512(acc: &[i32], weights: &[i16], qa: i32) -> i64 {
    use std::arch::x86_64::*;
    let mut even = _mm512_setzero_si512();
    let mut odd = _mm512_setzero_si512();
    for j in (0..acc.len()).step_by(16) {
        let x = _mm512_min_epi32(
            _mm512_set1_epi32(qa),
            _mm512_max_epi32(
                _mm512_setzero_si512(),
                _mm512_loadu_si512(acc.as_ptr().add(j).cast()),
            ),
        );
        let squared = _mm512_mullo_epi32(x, x);
        let w = _mm512_cvtepi16_epi32(_mm256_loadu_si256(weights.as_ptr().add(j).cast()));
        even = _mm512_add_epi64(even, _mm512_mul_epi32(squared, w));
        odd = _mm512_add_epi64(
            odd,
            _mm512_mul_epi32(_mm512_srli_epi64(squared, 32), _mm512_srli_epi64(w, 32)),
        );
    }
    let mut sums = [0i64; 8];
    _mm512_storeu_si512(sums.as_mut_ptr().cast(), _mm512_add_epi64(even, odd));
    sums.iter().sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_accumulation_matches_both_shared_position_oracles() {
        for bytes in [
            include_bytes!("../tests/fixtures/format6_h32.nnue").as_slice(),
            include_bytes!("../tests/fixtures/format8_h32.nnue").as_slice(),
        ] {
            let model = Model::from_bytes(bytes).unwrap().scalar_clone();
            assert_eq!(model.backend(), "scalar");
            crate::diagnostic::run(
                &model,
                include_bytes!("../tests/fixtures/format8_positions.jsonl").as_slice(),
                std::io::sink(),
                None,
                1,
            )
            .unwrap();
        }
    }

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
