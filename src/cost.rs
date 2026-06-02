//! Pluggable cost estimation for optimized completions.
//!
//! `rig-tokudo` owns **no** pricing data. The decorator focuses on token
//! compression and management; turning provider-reported [`Usage`] into a USD
//! figure is a host concern. Supply a [`CostModel`] (a rate table, a billing
//! API, a `rig-model-catalog`-backed adapter) via
//! [`crate::OptimizedModelBuilder::with_cost_model`], or hand a per-call
//! literal through [`crate::TokudoOptions::with_cost_estimate`]. When neither
//! is set, USD fields stay `None` and only raw token counts are reported.

use rig::completion::Usage;

/// USD cost breakdown for a single completed turn.
///
/// `usd_actual` is the cost actually billed for the call. `provider_cache_usd_delta`
/// captures the provider-side cache price difference (positive when cache reads
/// reduced the bill, negative when cache writes cost more than ordinary input
/// tokens); it is tracked separately from Tokudo savings to avoid
/// double-counting provider discounts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CostBreakdown {
    /// USD actually billed for this call.
    pub usd_actual: f64,
    /// Provider-side cache price delta, separate from Tokudo savings.
    pub provider_cache_usd_delta: f64,
}

/// Host-supplied source of USD cost estimates.
///
/// Implement this to translate a request's model id plus provider-reported
/// [`Usage`] into a [`CostBreakdown`]. Return `None` when the model is unknown
/// or pricing is unavailable — the decorator then reports token counts only.
///
/// # Example
///
/// ```no_run
/// use rig::completion::Usage;
/// use rig_tokudo::cost::{CostBreakdown, CostModel};
///
/// struct FlatRate {
///     usd_per_million_input: f64,
///     usd_per_million_output: f64,
/// }
///
/// impl CostModel for FlatRate {
///     fn estimate(&self, _model: Option<&str>, usage: &Usage) -> Option<CostBreakdown> {
///         let usd = (usage.input_tokens as f64 * self.usd_per_million_input
///             + usage.output_tokens as f64 * self.usd_per_million_output)
///             / 1_000_000.0;
///         Some(CostBreakdown { usd_actual: usd, provider_cache_usd_delta: 0.0 })
///     }
/// }
/// ```
pub trait CostModel: Send + Sync {
    /// Estimate the USD cost for a completed turn.
    ///
    /// `model` is the request's model id (e.g. `openai:gpt-4o-mini`), when
    /// known. Return `None` to signal "unknown — report tokens only."
    fn estimate(&self, model: Option<&str>, usage: &Usage) -> Option<CostBreakdown>;
}
