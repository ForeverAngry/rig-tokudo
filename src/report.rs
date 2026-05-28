//! Phase 6 measurement types: baseline-vs-tokudo run summaries, deltas, and
//! success-criteria evaluation.
//!
//! The headline savings benchmark in `examples/measure_savings.rs` populates
//! these structs from two runs over the same workload (one bare, one
//! tokudo-wrapped) and emits `report.json` + `report.md`. Real-provider
//! wiring (Ollama / OpenAI / Anthropic) and ROUGE-L quality scoring land
//! in Phase 6.1 once env-gated probes are in place.
//!
//! # Example
//!
//! ```
//! use rig_tokudo::report::{RunStats, Report, Thresholds};
//!
//! let baseline = RunStats {
//!     requests: 100,
//!     prompt_tokens: 50_000,
//!     completion_tokens: 25_000,
//!     wall_ms: 30_000,
//!     usd: 0.50,
//!     cache_hits: 0,
//!     cheap_calls: 0,
//!     strong_calls: 100,
//!     ..RunStats::default()
//! };
//! let tokudo = RunStats {
//!     requests: 100,
//!     prompt_tokens: 20_000,
//!     completion_tokens: 12_000,
//!     wall_ms: 18_000,
//!     usd: 0.20,
//!     cache_hits: 35,
//!     cheap_calls: 40,
//!     strong_calls: 25,
//!     ..RunStats::default()
//! };
//! let report = Report::from_runs(baseline, tokudo, None);
//! let outcome = report.evaluate(&Thresholds::default());
//! assert!(report.deltas.usd_saved_pct > 0.0);
//! let _ = outcome; // success-criteria evaluation result
//! ```

use serde::{Deserialize, Serialize};

#[cfg(not(target_family = "wasm"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_family = "wasm"))]
use crate::error::{Result, TokudoError};

/// Stable JSON filename for the Tokudo savings report.
#[cfg(not(target_family = "wasm"))]
pub const REPORT_JSON_FILE: &str = "report.json";
/// Stable Markdown filename for the Tokudo savings report.
#[cfg(not(target_family = "wasm"))]
pub const REPORT_MARKDOWN_FILE: &str = "report.md";
/// Stable JSON filename for replay/eval metric bundles.
#[cfg(not(target_family = "wasm"))]
pub const METRICS_JSON_FILE: &str = "metrics.json";
/// Stable Markdown filename for replay/eval metric bundles.
#[cfg(not(target_family = "wasm"))]
pub const METRICS_MARKDOWN_FILE: &str = "metrics.md";
/// Stable JSON filename for the artifact manifest.
#[cfg(not(target_family = "wasm"))]
pub const ARTIFACT_MANIFEST_FILE: &str = "manifest.json";

/// Aggregated counters for a single pass over a workload.
///
/// All token / USD figures are sums across requests; `wall_ms` is the
/// total elapsed wall-clock budget for the pass.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunStats {
    /// Total requests dispatched.
    pub requests: u64,
    /// Sum of provider-reported input ("prompt") tokens.
    pub prompt_tokens: u64,
    /// Sum of provider-reported output ("completion") tokens.
    pub completion_tokens: u64,
    /// Total wall time across all requests, in milliseconds.
    pub wall_ms: u64,
    /// Estimated USD spent. Computed from a pricing table; `0.0` when
    /// pricing is unavailable.
    pub usd: f64,
    /// Provider-reported cached-input tokens. These are tracked separately
    /// from Tokudo cache hits to avoid double-counting provider discounts.
    #[serde(default)]
    pub provider_cached_input_tokens: u64,
    /// Provider-reported cache-write tokens.
    #[serde(default)]
    pub provider_cache_write_tokens: u64,
    /// Provider-side cache price delta already reflected in `usd`.
    /// Positive values are provider discounts; negative values are write
    /// premiums. This is not counted as Tokudo savings.
    #[serde(default)]
    pub provider_cache_usd_delta: f64,
    /// Number of requests that were served by the cache.
    pub cache_hits: u64,
    /// Number of requests served by the cheap leg of a cascade.
    pub cheap_calls: u64,
    /// Number of requests served by the strong leg of a cascade (or
    /// every request, for the unwrapped baseline).
    pub strong_calls: u64,
}

impl RunStats {
    /// Total billed tokens (prompt + completion).
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Cache-hit rate as a fraction in `[0.0, 1.0]`. Returns `0.0` for
    /// empty runs.
    #[must_use]
    pub fn cache_hit_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.requests as f64
        }
    }

    /// Cheap-model share among non-cache requests (cascade utility).
    /// Returns `0.0` when every request hit the cache or the run was
    /// empty.
    #[must_use]
    pub fn cheap_share(&self) -> f64 {
        let dispatched = self.requests.saturating_sub(self.cache_hits);
        if dispatched == 0 {
            0.0
        } else {
            self.cheap_calls as f64 / dispatched as f64
        }
    }
}

/// Derived comparison between a baseline pass and a tokudo-wrapped pass
/// over the same workload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deltas {
    /// Gross absolute USD delta (`baseline.usd - tokudo.usd`) before
    /// separating provider-side cache discounts.
    pub gross_usd_saved: f64,
    /// Net provider-side cache delta in the Tokudo run after subtracting
    /// the baseline run's provider-cache delta.
    pub provider_cache_usd_delta: f64,
    /// Absolute USD savings (`baseline.usd - tokudo.usd`). May be
    /// negative if the tokudo pass somehow cost more. Provider-side cache
    /// discounts are subtracted so this reflects Tokudo-controlled savings.
    pub usd_saved: f64,
    /// USD savings as a fraction of baseline cost in `[0.0, 1.0]`.
    /// `0.0` when the baseline cost was `0.0`.
    pub usd_saved_pct: f64,
    /// Tokudo-pass cache hit rate; mirrors [`RunStats::cache_hit_rate`].
    pub cache_hit_rate: f64,
    /// Tokudo-pass cheap-leg share; mirrors [`RunStats::cheap_share`].
    pub cheap_model_share: f64,
    /// Wall-time speedup ratio (`baseline.wall_ms / tokudo.wall_ms`).
    /// `1.0` when both are zero or equal.
    pub speedup: f64,
    /// Quality preservation score against the baseline outputs. `None`
    /// when no judge / similarity scorer ran. Expected to be in
    /// `[0.0, 1.0]` for similarity metrics (e.g. ROUGE-L) or
    /// `[0.0, 5.0]` for LLM-as-judge.
    pub quality_score: Option<f64>,
}

impl Deltas {
    fn compute(baseline: &RunStats, tokudo: &RunStats, quality_score: Option<f64>) -> Self {
        let gross_usd_saved = baseline.usd - tokudo.usd;
        let provider_cache_usd_delta =
            tokudo.provider_cache_usd_delta - baseline.provider_cache_usd_delta;
        let usd_saved = gross_usd_saved - provider_cache_usd_delta;
        let usd_saved_pct = if baseline.usd > 0.0 {
            (usd_saved / baseline.usd).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let speedup = match (baseline.wall_ms, tokudo.wall_ms) {
            (0, 0) => 1.0,
            (_, 0) => f64::INFINITY,
            (b, t) => b as f64 / t as f64,
        };
        Self {
            gross_usd_saved,
            provider_cache_usd_delta,
            usd_saved,
            usd_saved_pct,
            cache_hit_rate: tokudo.cache_hit_rate(),
            cheap_model_share: tokudo.cheap_share(),
            speedup,
            quality_score,
        }
    }
}

/// Combined baseline + tokudo + delta report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Bare-provider pass over the workload.
    pub baseline: RunStats,
    /// Tokudo-wrapped pass over the same workload.
    pub tokudo: RunStats,
    /// Computed deltas.
    pub deltas: Deltas,
}

impl Report {
    /// Build a [`Report`] from two run summaries plus an optional
    /// quality score.
    #[must_use]
    pub fn from_runs(baseline: RunStats, tokudo: RunStats, quality_score: Option<f64>) -> Self {
        let deltas = Deltas::compute(&baseline, &tokudo, quality_score);
        Self {
            baseline,
            tokudo,
            deltas,
        }
    }

    /// Evaluate this report against `thresholds` and return per-criterion
    /// pass/fail results. The headline `examples/measure_savings.rs`
    /// binary exits non-zero when any required check fails.
    #[must_use]
    pub fn evaluate(&self, thresholds: &Thresholds) -> EvaluationOutcome {
        let usd_saved_pct = self.deltas.usd_saved_pct >= thresholds.min_usd_saved_pct;
        let cache_hit_rate = self.deltas.cache_hit_rate >= thresholds.min_cache_hit_rate;
        let cheap_share = self.deltas.cheap_model_share >= thresholds.min_cheap_model_share;
        let quality = match (self.deltas.quality_score, thresholds.min_quality_score) {
            (Some(actual), Some(min)) => actual >= min,
            (None, Some(_)) => false,
            _ => true,
        };
        EvaluationOutcome {
            usd_saved_pct,
            cache_hit_rate,
            cheap_share,
            quality,
        }
    }

    /// Render a human-readable Markdown summary table.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# tokudo savings report\n\n");
        out.push_str("| Metric | Baseline | Tokudo |\n");
        out.push_str("|---|---:|---:|\n");
        out.push_str(&format!(
            "| Requests | {} | {} |\n",
            self.baseline.requests, self.tokudo.requests
        ));
        out.push_str(&format!(
            "| Prompt tokens | {} | {} |\n",
            self.baseline.prompt_tokens, self.tokudo.prompt_tokens
        ));
        out.push_str(&format!(
            "| Completion tokens | {} | {} |\n",
            self.baseline.completion_tokens, self.tokudo.completion_tokens
        ));
        out.push_str(&format!(
            "| Wall ms | {} | {} |\n",
            self.baseline.wall_ms, self.tokudo.wall_ms
        ));
        out.push_str(&format!(
            "| USD | {:.4} | {:.4} |\n\n",
            self.baseline.usd, self.tokudo.usd
        ));
        out.push_str("## Deltas\n\n");
        out.push_str(&format!("- USD saved: {:.4}\n", self.deltas.usd_saved));
        out.push_str(&format!(
            "- Gross USD delta: {:.4}\n",
            self.deltas.gross_usd_saved
        ));
        out.push_str(&format!(
            "- Provider cache USD delta: {:.4}\n",
            self.deltas.provider_cache_usd_delta
        ));
        out.push_str(&format!(
            "- USD saved %: {:.2}\n",
            self.deltas.usd_saved_pct * 100.0
        ));
        out.push_str(&format!(
            "- Cache hit rate: {:.2}\n",
            self.deltas.cache_hit_rate * 100.0
        ));
        out.push_str(&format!(
            "- Cheap-model share: {:.2}\n",
            self.deltas.cheap_model_share * 100.0
        ));
        out.push_str(&format!("- Speedup: {:.2}x\n", self.deltas.speedup));
        if let Some(q) = self.deltas.quality_score {
            out.push_str(&format!("- Quality score: {q:.3}\n"));
        }
        out
    }

    /// Serialize this report as pretty-printed JSON.
    #[cfg(not(target_family = "wasm"))]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(report_serde_error)
    }

    /// Write `report.json`, `report.md`, and `manifest.json` into `directory`.
    #[cfg(not(target_family = "wasm"))]
    pub fn write_artifacts(&self, directory: impl AsRef<Path>) -> Result<ReportArtifactPaths> {
        self.write_artifacts_with_metadata(directory, ReportArtifactMetadata::savings())
    }

    /// Write report artifacts with caller-supplied manifest metadata.
    #[cfg(not(target_family = "wasm"))]
    pub fn write_artifacts_with_metadata(
        &self,
        directory: impl AsRef<Path>,
        metadata: ReportArtifactMetadata,
    ) -> Result<ReportArtifactPaths> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory).map_err(report_io_error)?;
        let paths = ReportArtifactPaths::savings(directory);
        write_string(&paths.report_json, &self.to_json()?)?;
        write_string(&paths.report_markdown, &self.to_markdown())?;
        write_json(&paths.manifest_json, &metadata)?;
        Ok(paths)
    }
}

/// Pass/fail thresholds mirroring the Phase 6.G success-criteria table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    /// Minimum required USD savings as a fraction (e.g. `0.40` → 40 %).
    pub min_usd_saved_pct: f64,
    /// Minimum required cache-hit rate as a fraction.
    pub min_cache_hit_rate: f64,
    /// Minimum required cheap-model share among non-cache requests.
    pub min_cheap_model_share: f64,
    /// Minimum required quality score, if a scorer ran.
    pub min_quality_score: Option<f64>,
}

impl Default for Thresholds {
    /// Default thresholds match the Phase 6.G success-criteria table
    /// (40 % USD saved, 30 % cache-hit, 50 % cheap-share, quality
    /// disabled by default).
    fn default() -> Self {
        Self {
            min_usd_saved_pct: 0.40,
            min_cache_hit_rate: 0.30,
            min_cheap_model_share: 0.50,
            min_quality_score: None,
        }
    }
}

/// Per-criterion pass/fail outcome produced by [`Report::evaluate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationOutcome {
    /// Whether USD savings cleared `min_usd_saved_pct`.
    pub usd_saved_pct: bool,
    /// Whether cache-hit rate cleared `min_cache_hit_rate`.
    pub cache_hit_rate: bool,
    /// Whether cheap-model share cleared `min_cheap_model_share`.
    pub cheap_share: bool,
    /// Whether quality score cleared `min_quality_score` (or no check
    /// was requested).
    pub quality: bool,
}

impl EvaluationOutcome {
    /// `true` iff every required criterion passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.usd_saved_pct && self.cache_hit_rate && self.cheap_share && self.quality
    }
}

/// Kind of report artifact set written to disk.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportArtifactKind {
    /// Savings-only [`Report`] artifacts.
    Savings,
    /// Replay artifacts containing both savings and eval metrics.
    Replay,
}

/// Manifest metadata written alongside report artifacts.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportArtifactMetadata {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Artifact set kind.
    pub kind: ReportArtifactKind,
    /// Optional host-supplied label such as a benchmark name or CI job id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Savings report JSON filename.
    pub report_json: String,
    /// Savings report Markdown filename.
    pub report_markdown: String,
    /// Eval metrics JSON filename, present for replay artifact sets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics_json: Option<String>,
    /// Eval metrics Markdown filename, present for replay artifact sets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics_markdown: Option<String>,
}

impl ReportArtifactMetadata {
    /// Metadata for a savings-only report artifact set.
    #[must_use]
    pub fn savings() -> Self {
        Self {
            schema_version: 1,
            kind: ReportArtifactKind::Savings,
            label: None,
            report_json: REPORT_JSON_FILE.to_string(),
            report_markdown: REPORT_MARKDOWN_FILE.to_string(),
            metrics_json: None,
            metrics_markdown: None,
        }
    }

    /// Metadata for a replay artifact set with eval metrics.
    #[must_use]
    pub fn replay() -> Self {
        Self {
            schema_version: 1,
            kind: ReportArtifactKind::Replay,
            label: None,
            report_json: REPORT_JSON_FILE.to_string(),
            report_markdown: REPORT_MARKDOWN_FILE.to_string(),
            metrics_json: Some(METRICS_JSON_FILE.to_string()),
            metrics_markdown: Some(METRICS_MARKDOWN_FILE.to_string()),
        }
    }

    /// Attach a host-supplied artifact label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Paths written by report artifact helpers.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportArtifactPaths {
    /// Artifact directory.
    pub directory: PathBuf,
    /// Savings report JSON path.
    pub report_json: PathBuf,
    /// Savings report Markdown path.
    pub report_markdown: PathBuf,
    /// Eval metrics JSON path, present for replay artifact sets.
    pub metrics_json: Option<PathBuf>,
    /// Eval metrics Markdown path, present for replay artifact sets.
    pub metrics_markdown: Option<PathBuf>,
    /// Manifest JSON path.
    pub manifest_json: PathBuf,
}

impl ReportArtifactPaths {
    /// Paths for the stable savings-only artifact layout.
    #[must_use]
    pub fn savings(directory: impl Into<PathBuf>) -> Self {
        let directory = directory.into();
        Self {
            report_json: directory.join(REPORT_JSON_FILE),
            report_markdown: directory.join(REPORT_MARKDOWN_FILE),
            metrics_json: None,
            metrics_markdown: None,
            manifest_json: directory.join(ARTIFACT_MANIFEST_FILE),
            directory,
        }
    }

    /// Paths for the stable replay artifact layout.
    #[must_use]
    pub fn replay(directory: impl Into<PathBuf>) -> Self {
        let directory = directory.into();
        Self {
            report_json: directory.join(REPORT_JSON_FILE),
            report_markdown: directory.join(REPORT_MARKDOWN_FILE),
            metrics_json: Some(directory.join(METRICS_JSON_FILE)),
            metrics_markdown: Some(directory.join(METRICS_MARKDOWN_FILE)),
            manifest_json: directory.join(ARTIFACT_MANIFEST_FILE),
            directory,
        }
    }
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn write_json<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    let json = serde_json::to_string_pretty(value).map_err(report_serde_error)?;
    write_string(path, &json)
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn write_string(path: &Path, value: &str) -> Result<()> {
    std::fs::write(path, value).map_err(report_io_error)
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn report_serde_error(err: serde_json::Error) -> TokudoError {
    TokudoError::Report(format!("serialize report artifact: {err}"))
}

#[cfg(not(target_family = "wasm"))]
fn report_io_error(err: std::io::Error) -> TokudoError {
    TokudoError::Report(format!("write report artifact: {err}"))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn run(usd: f64, hits: u64, cheap: u64, strong: u64) -> RunStats {
        RunStats {
            requests: hits + cheap + strong,
            prompt_tokens: 1_000,
            completion_tokens: 500,
            wall_ms: 1_000,
            usd,
            cache_hits: hits,
            cheap_calls: cheap,
            strong_calls: strong,
            ..RunStats::default()
        }
    }

    fn temp_report_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rig-tokudo-{name}-{nanos}"))
    }

    #[test]
    fn deltas_compute_usd_savings_pct() {
        let report = Report::from_runs(run(1.00, 0, 0, 100), run(0.50, 30, 50, 20), None);
        assert!((report.deltas.gross_usd_saved - 0.50).abs() < 1e-9);
        assert!((report.deltas.usd_saved - 0.50).abs() < 1e-9);
        assert!((report.deltas.usd_saved_pct - 0.50).abs() < 1e-9);
        assert!((report.deltas.cache_hit_rate - 0.30).abs() < 1e-9);
        // cheap_share = 50 / (100 - 30) = 50 / 70
        assert!((report.deltas.cheap_model_share - (50.0 / 70.0)).abs() < 1e-9);
    }

    #[test]
    fn deltas_subtract_provider_cache_delta_from_tokudo_savings() {
        let mut tokudo = run(0.40, 30, 50, 20);
        tokudo.provider_cache_usd_delta = 0.10;

        let report = Report::from_runs(run(1.00, 0, 0, 100), tokudo, None);

        assert!((report.deltas.gross_usd_saved - 0.60).abs() < 1e-9);
        assert!((report.deltas.provider_cache_usd_delta - 0.10).abs() < 1e-9);
        assert!((report.deltas.usd_saved - 0.50).abs() < 1e-9);
        assert!((report.deltas.usd_saved_pct - 0.50).abs() < 1e-9);
    }

    #[test]
    fn empty_baseline_does_not_divide_by_zero() {
        let report = Report::from_runs(run(0.0, 0, 0, 0), run(0.0, 0, 0, 0), None);
        assert_eq!(report.deltas.usd_saved_pct, 0.0);
        assert_eq!(report.deltas.speedup, 1.0);
    }

    #[test]
    fn evaluate_passes_when_all_criteria_met() {
        let report = Report::from_runs(run(1.00, 0, 0, 100), run(0.50, 40, 40, 20), None);
        let out = report.evaluate(&Thresholds::default());
        assert!(out.usd_saved_pct);
        assert!(out.cache_hit_rate);
        assert!(out.cheap_share);
        assert!(out.quality); // quality unrequested → vacuously true
        assert!(out.all_passed());
    }

    #[test]
    fn evaluate_fails_on_insufficient_savings() {
        let report = Report::from_runs(run(1.00, 0, 0, 100), run(0.80, 40, 40, 20), None);
        let out = report.evaluate(&Thresholds::default());
        assert!(!out.usd_saved_pct);
        assert!(!out.all_passed());
    }

    #[test]
    fn evaluate_fails_when_quality_required_but_absent() {
        let report = Report::from_runs(run(1.00, 0, 0, 100), run(0.50, 40, 40, 20), None);
        let thresholds = Thresholds {
            min_quality_score: Some(0.85),
            ..Thresholds::default()
        };
        let out = report.evaluate(&thresholds);
        assert!(!out.quality);
    }

    #[test]
    fn markdown_includes_quality_when_present() {
        let report = Report::from_runs(run(1.00, 0, 0, 100), run(0.50, 40, 40, 20), Some(0.91));
        let md = report.to_markdown();
        assert!(md.contains("Quality score"));
        assert!(md.contains("0.910"));
    }

    #[test]
    fn write_artifacts_creates_stable_report_layout() {
        let report = Report::from_runs(run(1.00, 0, 0, 100), run(0.50, 40, 40, 20), Some(0.91));
        let dir = temp_report_dir("report-artifacts");

        let paths = report
            .write_artifacts_with_metadata(
                &dir,
                ReportArtifactMetadata::savings().with_label("ci-smoke"),
            )
            .unwrap();

        assert_eq!(paths.report_json, dir.join(REPORT_JSON_FILE));
        assert_eq!(paths.report_markdown, dir.join(REPORT_MARKDOWN_FILE));
        assert_eq!(paths.metrics_json, None);
        assert!(paths.manifest_json.exists());
        let manifest = std::fs::read_to_string(&paths.manifest_json).unwrap();
        assert!(manifest.contains("ci-smoke"));
        assert!(
            std::fs::read_to_string(&paths.report_markdown)
                .unwrap()
                .contains("tokudo savings report")
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
