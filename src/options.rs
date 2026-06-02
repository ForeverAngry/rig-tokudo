//! Per-request control sidecar for [`crate::OptimizedModel`] calls.

use serde::{Deserialize, Serialize};

/// Cache behavior selector for a single call.
///
/// Modeled after LiteLLM's cache-control vocabulary so hosts familiar with
/// that gateway can map intent 1:1.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CachePolicy {
    /// Default: read from cache, write on miss, honor configured TTL.
    #[default]
    Default,
    /// Skip the cache read but still write the response on completion.
    ///
    /// Equivalent to `Cache-Control: no-cache` on the HTTP side.
    NoCache,
    /// Skip both cache read and write.
    ///
    /// Equivalent to `Cache-Control: no-store`.
    NoStore,
    /// Skip cache read; pretend the entry doesn't exist; do write on success.
    ///
    /// Forces a fresh provider call while keeping the cache warm.
    ForceFresh,
}

/// Per-request controls. Pass alongside a `CompletionRequest` to override the
/// decorator defaults for a single call.
///
/// All fields are optional / default to the decorator's configured behavior;
/// use the builder helpers to set only what you need.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TokudoOptions {
    /// Cache policy override for this call.
    pub cache_policy: CachePolicy,
    /// Logical cache namespace (e.g. per-tenant or per-feature isolation).
    pub cache_namespace: Option<String>,
    /// When `true`, skip the router entirely and use the inner model directly.
    pub bypass_router: bool,
    /// When `true`, skip the compressor for this call.
    pub bypass_compress: bool,
    /// Host-supplied USD cost actually billed for this call.
    ///
    /// When set, it overrides any configured
    /// [`crate::cost::CostModel`] for this call; tokudo records the value
    /// verbatim on telemetry and [`crate::Provenance`].
    pub usd_actual_estimate: Option<f64>,
    /// Host-supplied provider-side cache USD delta for this call.
    ///
    /// Only consulted when [`Self::usd_actual_estimate`] is set; defaults to
    /// `0.0` otherwise.
    pub provider_cache_usd_delta: Option<f64>,
}

impl TokudoOptions {
    /// Construct an empty options sidecar (all defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cache policy.
    #[must_use]
    pub fn with_cache_policy(mut self, policy: CachePolicy) -> Self {
        self.cache_policy = policy;
        self
    }

    /// Set a cache namespace for this call.
    #[must_use]
    pub fn with_cache_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.cache_namespace = Some(namespace.into());
        self
    }

    /// Skip the configured router for this call.
    #[must_use]
    pub fn with_bypass_router(mut self) -> Self {
        self.bypass_router = true;
        self
    }

    /// Skip the configured compressor for this call.
    #[must_use]
    pub fn with_bypass_compress(mut self) -> Self {
        self.bypass_compress = true;
        self
    }

    /// Supply the USD cost billed for this call.
    ///
    /// Overrides any configured [`crate::cost::CostModel`] for this call only.
    #[must_use]
    pub fn with_cost_estimate(mut self, usd_actual: f64) -> Self {
        self.usd_actual_estimate = Some(usd_actual);
        self
    }

    /// Supply the provider-side cache USD delta for this call.
    ///
    /// Only takes effect alongside [`Self::with_cost_estimate`].
    #[must_use]
    pub fn with_provider_cache_usd_delta(mut self, delta: f64) -> Self {
        self.provider_cache_usd_delta = Some(delta);
        self
    }
}
