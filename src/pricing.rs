//! Pricing helpers backed by `rig-model-catalog`.
//!
//! This module is available with the `model-catalog` feature. It keeps
//! `rig-tokudo`'s hot path small: callers pass the request's model string
//! plus provider-reported usage, and the helper returns a USD estimate when
//! the built-in pricing catalog can resolve the model unambiguously.

use rig::completion::Usage;
use rig_model_catalog::{ModelPrice, PricingTable, ProviderId};

/// Resolved price quote for a request model.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPrice {
    /// Provider id used for the lookup.
    pub provider: ProviderId,
    /// Provider-specific model id used for the lookup.
    pub model: String,
    /// Price row returned by `rig-model-catalog`.
    pub price: ModelPrice,
}

/// Estimate the actual USD cost for a model/usage pair.
///
/// The model id may be either explicit (`openai:gpt-4o-mini` or
/// `openai/gpt-4o-mini`) or provider-local (`gpt-4o-mini`). Provider-local
/// ids resolve only when the built-in table contains exactly one provider
/// row for that model.
#[must_use]
pub fn estimate_actual_usd(model: Option<&str>, usage: &Usage) -> Option<f64> {
    let table = PricingTable::builtin();
    let resolved = resolve_price(&table, model?)?;
    Some(resolved.price.cost_for(
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_input_tokens,
        usage.cache_creation_input_tokens,
    ))
}

/// Estimate the provider-side cache price delta for a model/usage pair.
///
/// Positive values mean provider cache reads reduced the bill versus charging
/// those tokens at the uncached input rate. Negative values mean cache writes
/// cost more than ordinary input tokens. This is not Tokudo savings; callers
/// should track it separately to avoid double-counting provider discounts.
#[must_use]
pub fn estimate_provider_cache_usd_delta(model: Option<&str>, usage: &Usage) -> Option<f64> {
    let table = PricingTable::builtin();
    let resolved = resolve_price(&table, model?)?;
    let input_rate = resolved.price.input_per_million;
    let cached_rate = resolved
        .price
        .cached_input_per_million
        .unwrap_or(input_rate);
    let cache_write_rate = resolved.price.cache_write_per_million.unwrap_or(input_rate);
    let read_delta = usage.cached_input_tokens as f64 * (input_rate - cached_rate);
    let write_delta = usage.cache_creation_input_tokens as f64 * (input_rate - cache_write_rate);
    Some((read_delta + write_delta) / 1_000_000.0)
}

/// Resolve a price from an arbitrary model id string.
#[must_use]
pub fn resolve_price(table: &PricingTable, model_id: &str) -> Option<ResolvedPrice> {
    let trimmed = model_id.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(resolved) = resolve_explicit(table, trimmed, ':') {
        return Some(resolved);
    }
    if let Some(resolved) = resolve_explicit(table, trimmed, '/') {
        return Some(resolved);
    }

    let mut found: Option<ResolvedPrice> = None;
    for (provider, model, price) in table.iter() {
        if model != trimmed {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(ResolvedPrice {
            provider: provider.clone(),
            model: model.to_string(),
            price: price.clone(),
        });
    }
    found
}

fn resolve_explicit(
    table: &PricingTable,
    model_id: &str,
    delimiter: char,
) -> Option<ResolvedPrice> {
    let (provider, model) = model_id.split_once(delimiter)?;
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    let price = table.lookup(provider, model)?.clone();
    Some(ResolvedPrice {
        provider: ProviderId::new(provider),
        model: model.to_string(),
        price,
    })
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
}
