//! Sampled function timers for one-thread diagnostics; absent from normal builds.
use std::cell::RefCell;
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Zone {
    Accumulator,
    Dense,
    Readout,
    Movegen,
    Ordering,
    Table,
}
const NAMES: [&str; 6] = [
    "accumulator_update",
    "dense_head",
    "linear_readout",
    "move_generation",
    "move_ordering",
    "table",
];
#[derive(Clone, Copy, Default)]
struct Counter {
    calls: u64,
    samples: u64,
    nanos: u128,
}
thread_local! {
    static COUNTERS: RefCell<[Counter; 6]> = const { RefCell::new([Counter { calls: 0, samples: 0, nanos: 0 }; 6]) };
}

/// Times one call in 64, avoiding a clock read on the other calls.
pub struct Probe {
    zone: usize,
    start: Option<Instant>,
}
impl Probe {
    pub fn new(zone: Zone) -> Self {
        let zone = zone as usize;
        let sampled = COUNTERS.with_borrow_mut(|c| {
            c[zone].calls += 1;
            c[zone].calls.is_multiple_of(64)
        });
        Self {
            zone,
            start: sampled.then(Instant::now),
        }
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let nanos = start.elapsed().as_nanos();
            COUNTERS.with_borrow_mut(|c| {
                c[self.zone].samples += 1;
                c[self.zone].nanos += nanos;
            });
        }
    }
}
pub fn reset() {
    COUNTERS.with_borrow_mut(|c| c.fill(Counter::default()));
}

/// Empty sampled scopes estimate the timestamp/scope cost included in each sample.
pub fn calibrate() -> f64 {
    reset();
    for _ in 0..65_536 {
        let _probe = Probe::new(Zone::Table);
        std::hint::black_box(());
    }
    let overhead = COUNTERS.with_borrow(|c| {
        let counter = c[Zone::Table as usize];
        counter.nanos as f64 / counter.samples as f64
    });
    reset();
    overhead
}

/// Raw sample counts allow aggregation across roots without averaging percentages.
pub fn snapshot() -> serde_json::Value {
    COUNTERS.with_borrow(|c| {
        let mut out = serde_json::Map::new();
        for (name, counter) in NAMES.into_iter().zip(c) {
            out.insert(
                name.to_owned(),
                serde_json::json!({
                    "calls": counter.calls,
                    "samples": counter.samples,
                    "sample_ns": counter.nanos,
                }),
            );
        }
        out.into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_every_sixty_four_calls_and_resets() {
        reset();
        for _ in 0..130 {
            let _probe = Probe::new(Zone::Table);
        }
        let value = snapshot();
        assert_eq!(value["table"]["calls"], 130);
        assert_eq!(value["table"]["samples"], 2);
        assert_eq!(value["dense_head"]["calls"], 0);
        reset();
        assert_eq!(snapshot()["table"]["calls"], 0);
    }
}
