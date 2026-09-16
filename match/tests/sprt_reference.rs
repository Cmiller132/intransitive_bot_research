//! Independent Python reference vectors and complete opening-order stopping trajectories.
use r#match::sprt::{self, State, Stop};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    protocol: serde_json::Value,
    llr_vectors: Vec<Vector>,
    sequences: Vec<Sequence>,
}
#[derive(Deserialize)]
struct Vector {
    counts: [u64; 5],
    llr: f64,
    #[serde(rename = "llr_s0.50_s0.55")]
    alternative: f64,
}
#[derive(Deserialize)]
struct Sequence {
    scenario: String,
    pair_scores: Vec<f64>,
    decision: Stop,
    pairs: usize,
    llr: f64,
    counts: [u64; 5],
    trajectory: Vec<(usize, f64)>,
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-9,
        "{actual:.15} != {expected:.15}"
    );
}

#[test]
fn agrees_with_independent_reference_vectors_and_every_stopping_check() {
    let fixtures: Fixtures = serde_json::from_str(include_str!("fixtures/sprt.json")).unwrap();
    assert_eq!(fixtures.protocol["scores"], serde_json::json!(sprt::SCORES));
    assert_eq!(fixtures.protocol["s0"], sprt::S0);
    assert_eq!(fixtures.protocol["s1"], sprt::S1);
    assert_eq!(fixtures.protocol["batch"], sprt::BATCH);
    assert_eq!(fixtures.protocol["first_check"], sprt::FIRST_CHECK);
    assert_eq!(fixtures.protocol["cap"], sprt::CAP);
    close(
        fixtures.protocol["upper_bound"].as_f64().unwrap(),
        sprt::BOUND,
    );
    close(
        fixtures.protocol["lower_bound"].as_f64().unwrap(),
        -sprt::BOUND,
    );
    for vector in fixtures.llr_vectors {
        close(
            sprt::llr(vector.counts, sprt::S0, sprt::S1).unwrap(),
            vector.llr,
        );
        close(
            sprt::llr(vector.counts, 0.50, 0.55).unwrap(),
            vector.alternative,
        );
    }
    for sequence in fixtures.sequences {
        let mut state = State::default();
        let mut trajectory = Vec::new();
        let mut pairs = 0;
        for score in sequence.pair_scores {
            pairs += 1;
            let cell = sprt::SCORES.iter().position(|x| *x == score).unwrap();
            state.counts[cell] += 1;
            state.check(pairs, &sprt::Bounds::default());
            if pairs >= sprt::FIRST_CHECK && pairs.is_multiple_of(sprt::BATCH) {
                trajectory.push((pairs, state.llr.unwrap()));
            }
            if state.stop_reason != Stop::Running {
                break;
            }
        }
        assert_eq!(
            state.stop_reason, sequence.decision,
            "{}",
            sequence.scenario
        );
        assert_eq!(pairs, sequence.pairs, "{}", sequence.scenario);
        assert_eq!(state.counts, sequence.counts);
        close(state.llr.unwrap(), sequence.llr);
        assert_eq!(trajectory.len(), sequence.trajectory.len());
        for ((actual_pair, actual_llr), (expected_pair, expected_llr)) in
            trajectory.into_iter().zip(sequence.trajectory)
        {
            assert_eq!(actual_pair, expected_pair);
            close(actual_llr, expected_llr);
        }
    }
}
