//! Expectation-constrained pentanomial likelihood and the paired eval stopping rule.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const SCORES: [f64; 5] = [0., 0.25, 0.5, 0.75, 1.];
pub const S0: f64 = 0.50;
pub const S1: f64 = 0.52;
pub const BATCH: usize = 16;
pub const FIRST_CHECK: usize = 128;
pub const CAP: usize = 3008;
pub const BOUND: f64 = 2.944_438_979_166_440_3;
const ZERO_CELL: f64 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stop {
    Running,
    Accept,
    Reject,
    Inconclusive,
    Invalid,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub counts: [u64; 5],
    pub llr: Option<f64>,
    pub stop_reason: Stop,
    pub error: Option<String>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            counts: [0; 5],
            llr: None,
            stop_reason: Stop::Running,
            error: None,
        }
    }
}

impl State {
    pub fn invalidate(&mut self, reason: impl Into<String>) {
        self.stop_reason = Stop::Invalid;
        self.error = Some(reason.into());
    }

    /// Called only after retiring an entire batch in opening-id order.
    pub fn check(&mut self, pairs: usize) {
        if self.stop_reason != Stop::Running || pairs < FIRST_CHECK || !pairs.is_multiple_of(BATCH)
        {
            return;
        }
        if self.counts.iter().sum::<u64>() != pairs as u64 || pairs > CAP {
            self.invalidate("pentanomial count differs from complete pairs or exceeds cap");
            return;
        }
        match llr(self.counts, S0, S1) {
            Ok(value) => {
                self.llr = Some(value);
                self.stop_reason = if value >= BOUND {
                    Stop::Accept
                } else if value <= -BOUND {
                    Stop::Reject
                } else if pairs == CAP {
                    Stop::Inconclusive
                } else {
                    Stop::Running
                };
            }
            Err(error) => self.invalidate(format!("likelihood: {error:#}")),
        }
    }
}

/// Generalized LLR, with each empty cell replaced by 0.001 observations.
/// The regularized count multiplies the expectation, as in Fishtest LLRcalc.
pub fn llr(counts: [u64; 5], s0: f64, s1: f64) -> Result<f64> {
    ensure!(
        s0.is_finite() && s1.is_finite() && 0. < s0 && s0 < s1 && s1 < 1.,
        "hypotheses must satisfy 0 < s0 < s1 < 1"
    );
    ensure!(counts.iter().any(|&n| n != 0), "no observations");
    let weights = counts.map(|n| if n == 0 { ZERO_CELL } else { n as f64 });
    let total = weights.iter().sum::<f64>();
    let pdf = weights.map(|n| n / total);
    let lambda0 = multiplier(&pdf, s0)?;
    let lambda1 = multiplier(&pdf, s1)?;
    let value = (0..5)
        .map(|i| {
            weights[i]
                * ((lambda0 * (SCORES[i] - s0)).ln_1p() - (lambda1 * (SCORES[i] - s1)).ln_1p())
        })
        .sum::<f64>();
    ensure!(value.is_finite(), "nonfinite likelihood");
    Ok(value)
}

/// Solve sum(p[i]*(x[i]-s)/(1+lambda*(x[i]-s)))=0 inside its positive domain.
fn multiplier(pdf: &[f64; 5], s: f64) -> Result<f64> {
    let (mut low, mut high) = (-1. / (1. - s), 1. / s);
    for _ in 0..80 {
        let mid = low + (high - low) * 0.5;
        if mid == low || mid == high {
            break;
        }
        let residual = (0..5)
            .map(|i| {
                let d = SCORES[i] - s;
                pdf[i] * d / (1. + mid * d)
            })
            .sum::<f64>();
        if residual > 0. {
            low = mid;
        } else {
            high = mid;
        }
    }
    let lambda = low + (high - low) * 0.5;
    let fitted = std::array::from_fn::<_, 5, _>(|i| pdf[i] / (1. + lambda * (SCORES[i] - s)));
    let sum = fitted.iter().sum::<f64>();
    let mean = fitted.iter().zip(SCORES).map(|(p, x)| p * x).sum::<f64>();
    ensure!(
        fitted.iter().all(|p| p.is_finite() && *p > 0.)
            && (sum - 1.).abs() < 1e-7
            && (mean - s).abs() < 1e-7,
        "constrained likelihood did not converge"
    );
    Ok(lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_support_matches_binomial_likelihood() {
        let counts = [400, 0, 0, 0, 600];
        let expected = 600. * (S1 / S0).ln() + 400. * ((1. - S1) / (1. - S0)).ln();
        assert!((llr(counts, S0, S1).unwrap() - expected).abs() < 0.001);
    }

    #[test]
    fn sparse_and_degenerate_cells_have_finite_likelihood() {
        for counts in [[0, 0, 128, 0, 0], [128, 0, 0, 0, 0], [0, 0, 0, 0, 3008]] {
            assert!(llr(counts, S0, S1).unwrap().is_finite());
        }
        assert!(llr([0; 5], S0, S1).is_err());
        assert!(llr([1; 5], f64::NAN, S1).is_err());
    }

    #[test]
    fn colour_reflection_and_hypothesis_reflection_reverse_llr() {
        let counts = [15, 20, 72, 32, 19];
        let mut reflected = counts;
        reflected.reverse();
        assert!(
            (llr(counts, S0, S1).unwrap() + llr(reflected, 1. - S1, 1. - S0).unwrap()).abs() < 1e-9
        );
    }

    #[test]
    fn checks_wait_for_complete_batches_and_do_not_override_stops() {
        let mut state = State::default();
        state.counts[4] = 127;
        state.check(127);
        assert_eq!(state.stop_reason, Stop::Running);
        state.counts[4] = 128;
        state.check(128);
        assert_eq!(state.stop_reason, Stop::Accept);
        state.check(CAP);
        assert_eq!(state.stop_reason, Stop::Accept);
        state.invalidate("forfeit");
        state.check(CAP);
        assert_eq!(state.stop_reason, Stop::Invalid);
    }

    #[test]
    fn cap_without_boundary_is_inconclusive() {
        let mut state = State {
            counts: [1474, 0, 0, 0, 1534],
            ..State::default()
        };
        state.check(CAP);
        assert_eq!(state.stop_reason, Stop::Inconclusive);
        assert!(state.llr.unwrap().abs() < BOUND);
    }
}
