//! Integration test for the ROUGE-L quality helper feeding report deltas.

#![cfg(feature = "eval")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use rig_tokudo::{Report, RunStats, Thresholds, mean_rouge_l_f1, rouge_l};

#[test]
fn rouge_l_quality_score_can_gate_report() {
    let score = mean_rouge_l_f1([
        ("alpha beta gamma", "alpha beta gamma"),
        ("cached answer with same facts", "answer with same facts"),
    ])
    .unwrap();
    assert!(score > 0.80);

    let report = Report::from_runs(
        RunStats {
            requests: 2,
            usd: 0.10,
            strong_calls: 2,
            ..RunStats::default()
        },
        RunStats {
            requests: 2,
            usd: 0.02,
            cache_hits: 1,
            strong_calls: 1,
            ..RunStats::default()
        },
        Some(score),
    );

    let outcome = report.evaluate(&Thresholds {
        min_usd_saved_pct: 0.50,
        min_cache_hit_rate: 0.40,
        min_cheap_model_share: 0.0,
        min_quality_score: Some(0.80),
    });
    assert!(outcome.all_passed());
}

#[test]
fn rouge_l_empty_pair_is_perfect_match() {
    let score = rouge_l("", "");
    assert_eq!(score.f1, 1.0);
}
