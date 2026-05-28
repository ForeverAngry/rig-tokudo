//! Replay bridge into `rig-retrieval-evals`.
//!
//! The `eval` feature lets hosts turn already-recorded Tokudo calls into
//! savings and quality metrics without invoking a provider again. Each
//! [`ReplayRow`] carries baseline call stats, Tokudo call stats, optional text
//! outputs, and the [`crate::Provenance`] produced by `OptimizedModel`.

use rig_retrieval_evals::{MetricReport, MultiReport};
use serde::{Deserialize, Serialize};

#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use crate::error::{Result, TokudoError};
use crate::provenance::{Provenance, RouterChoice};
use crate::quality::rouge_l;
use crate::report::{Report, RunStats};
#[cfg(not(target_family = "wasm"))]
use crate::report::{
    ReportArtifactMetadata, ReportArtifactPaths, report_serde_error, write_json, write_string,
};

/// Per-call counters captured during a replayable run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplayCallStats {
    /// Provider-reported input tokens for this call.
    pub prompt_tokens: u64,
    /// Provider-reported output tokens for this call.
    pub completion_tokens: u64,
    /// Wall-clock duration for this call, in milliseconds.
    pub wall_ms: u64,
    /// Estimated USD spent by this call.
    pub usd: f64,
    /// Provider-reported cached-input tokens.
    #[serde(default)]
    pub provider_cached_input_tokens: u64,
    /// Provider-reported cache-write tokens.
    #[serde(default)]
    pub provider_cache_write_tokens: u64,
    /// Provider-side cache price delta for this call.
    #[serde(default)]
    pub provider_cache_usd_delta: f64,
}

impl ReplayCallStats {
    /// Build call stats from Rig provider usage plus wall-clock and USD data.
    #[must_use]
    pub fn from_usage(usage: &rig::completion::Usage, wall_ms: u64, usd: f64) -> Self {
        Self {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            wall_ms,
            usd,
            provider_cached_input_tokens: usage.cached_input_tokens,
            provider_cache_write_tokens: usage.cache_creation_input_tokens,
            provider_cache_usd_delta: 0.0,
        }
    }

    fn accumulate_into(&self, run: &mut RunStats) {
        run.requests = run.requests.saturating_add(1);
        run.prompt_tokens = run.prompt_tokens.saturating_add(self.prompt_tokens);
        run.completion_tokens = run.completion_tokens.saturating_add(self.completion_tokens);
        run.wall_ms = run.wall_ms.saturating_add(self.wall_ms);
        run.usd += self.usd;
        run.provider_cached_input_tokens = run
            .provider_cached_input_tokens
            .saturating_add(self.provider_cached_input_tokens);
        run.provider_cache_write_tokens = run
            .provider_cache_write_tokens
            .saturating_add(self.provider_cache_write_tokens);
        run.provider_cache_usd_delta += self.provider_cache_usd_delta;
    }
}

/// One replay row for a labeled workload item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRow {
    /// Stable workload query/task identifier.
    pub query_id: String,
    /// Baseline call counters recorded during the original run.
    pub baseline: ReplayCallStats,
    /// Tokudo call counters recorded during the original run.
    pub tokudo: ReplayCallStats,
    /// Provenance returned by `OptimizedModel` for the Tokudo call.
    pub provenance: Provenance,
    /// Baseline output text, when quality replay should score ROUGE-L.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_output: Option<String>,
    /// Tokudo output text, when quality replay should score ROUGE-L.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokudo_output: Option<String>,
}

impl ReplayRow {
    /// Build a replay row with no text quality payloads.
    #[must_use]
    pub fn new(
        query_id: impl Into<String>,
        baseline: ReplayCallStats,
        tokudo: ReplayCallStats,
        provenance: Provenance,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            baseline,
            tokudo,
            provenance,
            baseline_output: None,
            tokudo_output: None,
        }
    }

    /// Attach baseline and Tokudo output text for ROUGE-L replay.
    #[must_use]
    pub fn with_outputs(
        mut self,
        baseline_output: impl Into<String>,
        tokudo_output: impl Into<String>,
    ) -> Self {
        self.baseline_output = Some(baseline_output.into());
        self.tokudo_output = Some(tokudo_output.into());
        self
    }
}

/// Replayed Tokudo savings and eval-native metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayReport {
    /// Tokudo-native baseline-vs-optimized savings report.
    pub savings: Report,
    /// `rig-retrieval-evals` metric bundle derived from the same rows.
    pub metrics: MultiReport,
}

#[cfg(not(target_family = "wasm"))]
impl ReplayReport {
    /// Write replay artifacts into `directory` using stable filenames.
    ///
    /// The layout is `report.json`, `report.md`, `metrics.json`,
    /// `metrics.md`, and `manifest.json`.
    pub fn write_artifacts(&self, directory: impl AsRef<Path>) -> Result<ReportArtifactPaths> {
        self.write_artifacts_with_metadata(directory, ReportArtifactMetadata::replay())
    }

    /// Write replay artifacts with caller-supplied manifest metadata.
    pub fn write_artifacts_with_metadata(
        &self,
        directory: impl AsRef<Path>,
        metadata: ReportArtifactMetadata,
    ) -> Result<ReportArtifactPaths> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)
            .map_err(|e| TokudoError::Report(format!("write report artifact: {e}")))?;
        let paths = ReportArtifactPaths::replay(directory);
        write_string(&paths.report_json, &self.savings.to_json()?)?;
        write_string(&paths.report_markdown, &self.savings.to_markdown())?;
        if let Some(metrics_json) = &paths.metrics_json {
            let json = serde_json::to_string_pretty(&self.metrics).map_err(report_serde_error)?;
            write_string(metrics_json, &json)?;
        }
        if let Some(metrics_markdown) = &paths.metrics_markdown {
            write_string(metrics_markdown, &self.metrics.to_markdown())?;
        }
        write_json(&paths.manifest_json, &metadata)?;
        Ok(paths)
    }
}

/// Build a [`ReplayReport`] from recorded rows without provider calls.
pub fn replay_report(rows: &[ReplayRow]) -> Result<ReplayReport> {
    if rows.is_empty() {
        return Err(TokudoError::Config(
            "replay requires at least one row".to_string(),
        ));
    }

    let mut baseline = RunStats::default();
    let mut tokudo = RunStats::default();
    let mut cache_hit = Vec::with_capacity(rows.len());
    let mut cheap_route = Vec::with_capacity(rows.len());
    let mut strong_route = Vec::with_capacity(rows.len());
    let mut usd_saved = Vec::with_capacity(rows.len());
    let mut usd_saved_pct = Vec::with_capacity(rows.len());
    let mut rouge = Vec::new();

    for row in rows {
        row.baseline.accumulate_into(&mut baseline);
        row.tokudo.accumulate_into(&mut tokudo);
        accumulate_route(&row.provenance, &mut tokudo);

        let cache_score = if row.provenance.cache_hit { 1.0 } else { 0.0 };
        cache_hit.push((row.query_id.clone(), cache_score));
        cheap_route.push((
            row.query_id.clone(),
            route_score(row.provenance.router_choice, RouterChoice::Cheap),
        ));
        strong_route.push((
            row.query_id.clone(),
            route_score(row.provenance.router_choice, RouterChoice::Strong),
        ));

        let provider_delta =
            row.tokudo.provider_cache_usd_delta - row.baseline.provider_cache_usd_delta;
        let saved = (row.baseline.usd - row.tokudo.usd) - provider_delta;
        usd_saved.push((row.query_id.clone(), saved));
        usd_saved_pct.push((row.query_id.clone(), saved_pct(saved, row.baseline.usd)));

        if let (Some(candidate), Some(reference)) = (&row.tokudo_output, &row.baseline_output) {
            rouge.push((row.query_id.clone(), rouge_l(candidate, reference).f1));
        }
    }

    let quality_score = if rouge.is_empty() {
        None
    } else {
        Some(rouge.iter().map(|(_, score)| *score).sum::<f64>() / rouge.len() as f64)
    };
    let savings = Report::from_runs(baseline, tokudo, quality_score);

    let mut reports = vec![
        MetricReport::from_per_query("tokudo.cache_hit".to_string(), cache_hit),
        MetricReport::from_per_query("tokudo.route.cheap".to_string(), cheap_route),
        MetricReport::from_per_query("tokudo.route.strong".to_string(), strong_route),
        MetricReport::from_per_query("tokudo.usd_saved".to_string(), usd_saved),
        MetricReport::from_per_query("tokudo.usd_saved_pct".to_string(), usd_saved_pct),
    ];
    if !rouge.is_empty() {
        reports.push(MetricReport::from_per_query(
            "quality.rouge_l_f1".to_string(),
            rouge,
        ));
    }

    Ok(ReplayReport {
        savings,
        metrics: MultiReport::new(reports),
    })
}

fn accumulate_route(provenance: &Provenance, run: &mut RunStats) {
    match provenance.router_choice {
        RouterChoice::CacheHit => run.cache_hits = run.cache_hits.saturating_add(1),
        RouterChoice::Cheap => run.cheap_calls = run.cheap_calls.saturating_add(1),
        RouterChoice::Strong => run.strong_calls = run.strong_calls.saturating_add(1),
        RouterChoice::PassThrough => run.strong_calls = run.strong_calls.saturating_add(1),
    }
}

fn route_score(actual: RouterChoice, expected: RouterChoice) -> f64 {
    if actual == expected { 1.0 } else { 0.0 }
}

fn saved_pct(saved: f64, baseline_usd: f64) -> f64 {
    if baseline_usd > 0.0 {
        (saved / baseline_usd).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn stats(usd: f64) -> ReplayCallStats {
        ReplayCallStats {
            prompt_tokens: 100,
            completion_tokens: 50,
            wall_ms: 25,
            usd,
            ..ReplayCallStats::default()
        }
    }

    fn provenance(choice: RouterChoice, hit: bool) -> Provenance {
        let mut provenance = Provenance::pass_through();
        provenance.router_choice = choice;
        provenance.cache_hit = hit;
        provenance
    }

    #[test]
    fn replay_builds_savings_and_metric_reports() {
        let rows = vec![
            ReplayRow::new(
                "q1",
                stats(0.10),
                stats(0.00),
                provenance(RouterChoice::CacheHit, true),
            )
            .with_outputs("alpha beta", "alpha beta"),
            ReplayRow::new(
                "q2",
                stats(0.10),
                stats(0.04),
                provenance(RouterChoice::Cheap, false),
            )
            .with_outputs("gamma delta", "gamma"),
        ];

        let report = replay_report(&rows).unwrap();

        assert_eq!(report.savings.baseline.requests, 2);
        assert_eq!(report.savings.tokudo.cache_hits, 1);
        assert_eq!(report.savings.tokudo.cheap_calls, 1);
        assert!(report.savings.deltas.usd_saved > 0.0);
        assert!(
            report
                .metrics
                .metrics
                .iter()
                .any(|metric| metric.metric == "quality.rouge_l_f1")
        );
    }

    #[test]
    fn replay_rejects_empty_rows() {
        let err = replay_report(&[]).unwrap_err();
        assert!(matches!(err, TokudoError::Config(_)));
    }

    #[test]
    fn replay_write_artifacts_creates_report_and_metrics_layout() {
        let rows = vec![
            ReplayRow::new(
                "q1",
                stats(0.10),
                stats(0.00),
                provenance(RouterChoice::CacheHit, true),
            )
            .with_outputs("alpha beta", "alpha beta"),
            ReplayRow::new(
                "q2",
                stats(0.10),
                stats(0.04),
                provenance(RouterChoice::Cheap, false),
            )
            .with_outputs("gamma delta", "gamma"),
        ];
        let report = replay_report(&rows).unwrap();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("rig-tokudo-replay-artifacts-{nanos}"));

        let paths = report.write_artifacts(&dir).unwrap();

        assert!(paths.report_json.exists());
        assert!(paths.report_markdown.exists());
        assert!(paths.manifest_json.exists());
        let (Some(metrics_json), Some(metrics_markdown)) =
            (paths.metrics_json, paths.metrics_markdown)
        else {
            panic!("replay artifacts should include metrics paths");
        };
        assert!(metrics_json.exists());
        assert!(metrics_markdown.exists());
        assert!(
            std::fs::read_to_string(metrics_markdown)
                .unwrap()
                .contains("tokudo.cache_hit")
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
