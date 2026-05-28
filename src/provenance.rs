//! Per-call provenance metadata attached to optimized responses.

use serde::{Deserialize, Serialize};

#[cfg(feature = "lineage")]
use crate::lineage::LineageEdge;

/// Which leg of the decorator answered the call.
///
/// Used both by [`Provenance`] and by routers to advertise their decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterChoice {
    /// Cache returned a stored response — no provider call was made.
    CacheHit,
    /// The cheap leg of the cascade answered (and passed validation).
    Cheap,
    /// The strong leg of the cascade answered (cheap rejected or skipped).
    Strong,
    /// No router was configured; the inner model was called directly.
    #[default]
    PassThrough,
}

/// Provenance attached to every response produced by [`crate::OptimizedModel`].
///
/// `Provenance` is the load-bearing audit record. The headline benchmark in
/// Phase 6 reads `cache_hit`, `router_choice`, and `usd_saved_estimate` to
/// compute the savings report. Treat field additions as additive; renames or
/// removals are breaking changes for downstream evaluators.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Provenance {
    /// `true` when the cache returned the response instead of the provider.
    pub cache_hit: bool,
    /// Similarity score for semantic-cache hits, when available (0.0–1.0).
    pub similarity: Option<f32>,
    /// Stable identifier of the cache entry that served the request.
    pub source_id: Option<String>,
    /// Which leg answered.
    pub router_choice: RouterChoice,
    /// Ratio of compressed-prompt tokens to original-prompt tokens (0.0–1.0).
    pub compressed_ratio: Option<f32>,
    /// Estimated USD cost that would have been billed without optimization.
    pub usd_baseline_estimate: Option<f64>,
    /// Estimated USD cost actually billed (or 0.0 for a pure cache hit).
    pub usd_actual_estimate: Option<f64>,
    /// Provider-reported cached-input tokens for the actual provider call.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub provider_cached_input_tokens: u64,
    /// Provider-reported cache-write tokens for the actual provider call.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub provider_cache_write_tokens: u64,
    /// Provider-side cache price delta, separate from Tokudo savings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_cache_usd_delta: Option<f64>,
    /// Convenience: `usd_baseline_estimate - usd_actual_estimate` when both
    /// are present. `None` when either is unknown.
    pub usd_saved_estimate: Option<f64>,
    /// Per-call lineage edge linking the served response to its source.
    #[cfg(feature = "lineage")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<LineageEdge>,
}

impl Provenance {
    /// Construct a pass-through provenance record: no cache hit, no routing,
    /// no compression. Suitable for the Phase 1 decorator default.
    #[must_use]
    pub fn pass_through() -> Self {
        Self {
            cache_hit: false,
            similarity: None,
            source_id: None,
            router_choice: RouterChoice::PassThrough,
            compressed_ratio: None,
            usd_baseline_estimate: None,
            usd_actual_estimate: None,
            provider_cached_input_tokens: 0,
            provider_cache_write_tokens: 0,
            provider_cache_usd_delta: None,
            usd_saved_estimate: None,
            #[cfg(feature = "lineage")]
            lineage: None,
        }
    }
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

/// Wrapper attaching [`Provenance`] to a provider response.
///
/// `NormalizedResponse` is the public return type of [`crate::OptimizedModel`]
/// surfaces that opt out of impersonating `CompletionModel` directly. Callers
/// pattern-match `response` for the raw provider payload and read `provenance`
/// for telemetry.
#[derive(Debug, Clone)]
pub struct NormalizedResponse<R> {
    /// The underlying provider response (or the cached payload reshaped to
    /// match the provider type).
    pub response: R,
    /// Decorator metadata for this call.
    pub provenance: Provenance,
}

impl<R> NormalizedResponse<R> {
    /// Construct a `NormalizedResponse` with the supplied provenance.
    #[must_use]
    pub fn new(response: R, provenance: Provenance) -> Self {
        Self {
            response,
            provenance,
        }
    }

    /// Unwrap, discarding provenance metadata.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.response
    }
}
