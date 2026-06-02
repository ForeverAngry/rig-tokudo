//! Pricing glue backed by `rig-model-catalog`.
//!
//! This module is available with the `model-catalog` feature. It owns no
//! rates and no pricing math — those live in `rig-model-catalog`. Tokudo
//! only maps a request's `rig_core::completion::Usage` onto the catalog's
//! scalar pricing API so the decorator can fill USD estimates on its
//! telemetry and replay rows.

use rig::completion::Usage;
use rig_model_catalog::PricingTable;

/// Re-exported resolved price quote from `rig-model-catalog`.
///
/// Tokudo does not define its own price type; resolution and the backing
/// catalog are owned by `rig-model-catalog`.
pub use rig_model_catalog::ResolvedPrice;

/// Estimate the actual USD cost for a model/usage pair.
///
/// The model id may be either explicit (`openai:gpt-4o-mini` or
/// `openai/gpt-4o-mini`) or provider-local (`gpt-4o-mini`). Resolution and
/// the cost arithmetic are delegated to `rig-model-catalog`.
#[must_use]
pub fn estimate_actual_usd(model: Option<&str>, usage: &Usage) -> Option<f64> {
    let table = PricingTable::builtin();
    let resolved = table.resolve(model?)?;
    Some(resolved.price.cost_for(
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_input_tokens,
        usage.cache_creation_input_tokens,
    ))
}

/// Estimate the provider-side cache USD delta for a model/usage pair.
///
/// Positive values mean provider cache reads reduced the bill; negative
/// values mean cache writes cost more than uncached input tokens. This is a
/// provider discount, **not** Tokudo savings — callers track it separately.
/// The arithmetic is owned by `rig_model_catalog::ModelPrice::cache_delta`.
#[must_use]
pub fn estimate_provider_cache_usd_delta(model: Option<&str>, usage: &Usage) -> Option<f64> {
    let table = PricingTable::builtin();
    let resolved = table.resolve(model?)?;
    Some(
        resolved
            .price
            .cache_delta(usage.cached_input_tokens, usage.cache_creation_input_tokens),
    )
}

/// Resolve a price from an arbitrary model id string.
///
/// Thin delegate to [`PricingTable::resolve`]; retained for callers that
/// already hold a table.
#[must_use]
pub fn resolve_price(table: &PricingTable, model_id: &str) -> Option<ResolvedPrice> {
    table.resolve(model_id)
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

    #[test]
    fn resolves_explicit_provider_prefix() {
        let table = PricingTable::builtin();
        let resolved = resolve_price(&table, "openai:gpt-4o-mini").unwrap();
        assert_eq!(resolved.provider.as_str(), "openai");
        assert_eq!(resolved.model, "gpt-4o-mini");
    }

    #[test]
    fn estimates_builtin_price() {
        let usage = Usage {
            input_tokens: 10_000,
            output_tokens: 2_000,
            total_tokens: 12_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            reasoning_tokens: 0,
        };
        let usd = estimate_actual_usd(Some("openai:gpt-4o-mini"), &usage).unwrap();
        assert!((usd - 0.0027).abs() < 1e-12);
    }

    #[test]
    fn estimates_provider_cache_delta_separately_from_actual_cost() {
        let usage = Usage {
            input_tokens: 10_000,
            output_tokens: 2_000,
            total_tokens: 22_000,
            cached_input_tokens: 10_000,
            cache_creation_input_tokens: 0,
            reasoning_tokens: 0,
        };

        let actual = estimate_actual_usd(Some("openai:gpt-4o-mini"), &usage).unwrap();
        let provider_delta =
            estimate_provider_cache_usd_delta(Some("openai:gpt-4o-mini"), &usage).unwrap();

        assert!((actual - 0.00345).abs() < 1e-12);
        assert!((provider_delta - 0.00075).abs() < 1e-12);
    }

    #[test]
    fn zero_token_usage_estimates_zero_cost_for_known_model() {
        let usage = Usage::new();

        let actual = estimate_actual_usd(Some("openai:gpt-4o-mini"), &usage).unwrap();
        let provider_delta =
            estimate_provider_cache_usd_delta(Some("openai:gpt-4o-mini"), &usage).unwrap();

        assert_eq!(actual, 0.0);
        assert_eq!(provider_delta, 0.0);
    }

    #[test]
    fn unknown_or_empty_model_returns_none() {
        let usage = Usage::new();
        let table = PricingTable::builtin();

        assert!(resolve_price(&table, "").is_none());
        assert!(resolve_price(&table, "unknown:nope").is_none());
        assert!(estimate_actual_usd(None, &usage).is_none());
        assert!(estimate_provider_cache_usd_delta(Some("unknown:nope"), &usage).is_none());
    }
}
