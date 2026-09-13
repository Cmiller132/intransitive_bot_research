//! Seeded pair-distribution calibration of the production sequential stopping rule.
use r#match::sprt::{State, Stop, BATCH, CAP, FIRST_CHECK, SCORES};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
#[ignore = "run explicitly in release mode before live sequential tests"]
fn calibrated_pair_distributions() {
    let trials = 10_000;
    let cases = [
        ("broad_null", [0.15, 0.2, 0.3, 0.2, 0.15], 0),
        ("broad_alternative", [0.13, 0.2, 0.3, 0.2, 0.17], 1),
        ("binary_null", [0.5, 0., 0., 0., 0.5], 0),
        ("binary_alternative", [0.48, 0., 0., 0., 0.52], 1),
        ("high_draw_null", [0., 0.04, 0.92, 0.04, 0.], 0),
        ("high_draw_alternative", [0., 0., 0.92, 0.08, 0.], 1),
        ("asymmetric_null", [0.1, 0.1, 0.5, 0.3, 0.], 0),
        ("asymmetric_alternative", [0.08, 0.1, 0.5, 0.3, 0.02], 1),
        // Independent wins with probabilities (.8,.2) and (.82,.22) in the two colours.
        ("colour_bias_null", [0.16, 0., 0.68, 0., 0.16], 0),
        (
            "colour_bias_alternative",
            [0.1404, 0., 0.6792, 0., 0.1804],
            1,
        ),
        ("near_zero_null", [0., 0.0001, 0.9998, 0.0001, 0.], 0),
        ("all_draws_null", [0., 0., 1., 0., 0.], 0),
        ("midpoint", [0.49, 0., 0., 0., 0.51], 2),
    ];
    let mut failed = Vec::new();
    for (case, (name, pdf, hypothesis)) in cases.into_iter().enumerate() {
        let mean = pdf.iter().zip(SCORES).map(|(p, x)| p * x).sum::<f64>();
        let expected = [0.5, 0.52, 0.51][hypothesis];
        assert!((pdf.iter().sum::<f64>() - 1.).abs() < 1e-12 && (mean - expected).abs() < 1e-12);
        let mut rng = StdRng::seed_from_u64(2026091200 + case as u64);
        let mut counts = [0usize; 4];
        let mut total_pairs = 0usize;
        for _ in 0..trials {
            let mut state = State::default();
            for pair in 1..=CAP {
                let mut draw = rng.random::<f64>();
                let mut cell = 4;
                for (i, p) in pdf.iter().enumerate() {
                    draw -= p;
                    if draw < 0. {
                        cell = i;
                        break;
                    }
                }
                state.counts[cell] += 1;
                if pair >= FIRST_CHECK && pair.is_multiple_of(BATCH) {
                    state.check(pair);
                }
                if state.stop_reason != Stop::Running {
                    let index = match state.stop_reason {
                        Stop::Accept => 0,
                        Stop::Reject => 1,
                        Stop::Inconclusive => 2,
                        _ => 3,
                    };
                    counts[index] += 1;
                    total_pairs += pair;
                    break;
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({"case":name,"seed":2026091200+case as u64,"trials":trials,"pdf":pdf,"accept":counts[0],"reject":counts[1],"inconclusive":counts[2],"invalid":counts[3],"mean_pairs":total_pairs as f64/trials as f64})
        );
        // A prespecified four-standard-error allowance around the nominal 5% rate.
        let limit = 0.05 + 4. * (0.05_f64 * 0.95 / trials as f64).sqrt();
        if counts[3] != 0 || (hypothesis < 2 && counts[hypothesis] as f64 / trials as f64 > limit) {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "calibration failed: {failed:?}");
}
