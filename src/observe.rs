//! Structured observability events emitted by the rig-tokudo decorator.
//!
//! Phase 5 emits five event kinds — `cache.hit`, `cache.miss`,
//! `route.decision`, `compress.applied`, and `cost.estimate` — through the
//! `tracing` dispatcher on the [`EVENT_TARGET`] target. The wire format is
//! a single flat JSON object carried in the `event` field, alongside scalar
//! `rig_tokudo.*` fields for collector routing without JSON parsing.
//!
//! The shape mirrors `rig-tap`'s `ObservabilityEvent` convention so
//! downstream subscribers can route both targets through the same
//! aggregator. With the `tap` feature enabled, the event also includes
//! `rig_tap.*` scalar fields on the `rig_tokudo` target; the payload stays
//! tokudo-owned because `rig-tap` intentionally has a closed event variant
//! set.
//!
//! # Example
//!
//! ```no_run
//! use rig_tokudo::observe::{emit, TokudoEvent};
//!
//! emit(&TokudoEvent::CacheMiss {
//!     cache_key: "abc123".into(),
//! });
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[cfg(feature = "lineage")]
use crate::lineage::LineageEdge;
use crate::provenance::RouterChoice;

/// Tracing target string used on every rig-tokudo observability event.
pub const EVENT_TARGET: &str = "rig_tokudo";

/// Current schema version for [`TokudoEvent`]. Bumped on breaking wire
/// changes.
pub const SCHEMA_VERSION: u32 = 1;

static TICK: AtomicU64 = AtomicU64::new(0);

/// Decorator-level event payloads. The wire format is a flat JSON object
/// tagged by `kind` (`cache.hit`, `cache.miss`, `route.decision`,
/// `compress.applied`, `cost.estimate`).
///
/// Variants are intentionally additive; renames or removals require a
/// [`SCHEMA_VERSION`] bump.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum TokudoEvent {
    /// The cache returned a stored response; the provider was skipped.
    #[serde(rename = "cache.hit")]
    CacheHit {
        /// Stable storage key for the entry.
        cache_key: String,
        /// Cache-supplied source identifier (typically equals `cache_key`).
        source_id: String,
        /// Similarity score for semantic caches; `None` for exact-match.
        #[serde(skip_serializing_if = "Option::is_none")]
        similarity: Option<f32>,
    },
    /// The cache was consulted but returned no entry; the provider call
    /// proceeded.
    #[serde(rename = "cache.miss")]
    CacheMiss {
        /// Stable storage key that was looked up.
        cache_key: String,
    },
    /// A router or cascade made a leg-selection decision.
    #[serde(rename = "route.decision")]
    RouteDecision {
        /// Which leg answered.
        choice: RouterChoice,
    },
    /// The compressor produced a measurable change to the request.
    #[serde(rename = "compress.applied")]
    CompressApplied {
        /// Approximate input tokens before compression.
        input_tokens: u32,
        /// Approximate output tokens after compression.
        output_tokens: u32,
        /// `output_tokens / input_tokens` when `input_tokens > 0`.
        #[serde(skip_serializing_if = "Option::is_none")]
        ratio: Option<f32>,
    },
    /// A token-usage snapshot for cost estimation. The decorator emits
    /// this event after every provider call (cache hits emit zero usage).
    /// USD fields are populated only when the host supplies a
    /// [`crate::cost::CostModel`] or a per-call
    /// [`crate::TokudoOptions::with_cost_estimate`] override; otherwise only
    /// raw token counts are carried.
    #[serde(rename = "cost.estimate")]
    CostEstimate {
        /// Whether the response came from the cache.
        cache_hit: bool,
        /// Provider-reported input tokens.
        input_tokens: u64,
        /// Provider-reported output tokens.
        output_tokens: u64,
        /// Provider-reported cached-input tokens.
        #[serde(default, skip_serializing_if = "is_zero")]
        cached_input_tokens: u64,
        /// Provider-reported cache-write tokens.
        #[serde(default, skip_serializing_if = "is_zero")]
        cache_write_tokens: u64,
        /// Provider-reported total tokens.
        total_tokens: u64,
        /// USD estimate of the actually-billed cost. `None` unless a host
        /// [`crate::cost::CostModel`] or per-call override supplied it.
        #[serde(skip_serializing_if = "Option::is_none")]
        usd_actual_estimate: Option<f64>,
        /// Provider-side cache price delta, separate from Tokudo savings.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_cache_usd_delta: Option<f64>,
    },
    /// Per-call source → response lineage edge for audit sinks.
    #[cfg(feature = "lineage")]
    #[serde(rename = "lineage.edge")]
    LineageEdge {
        /// Edge payload.
        #[serde(flatten)]
        edge: LineageEdge,
    },
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

impl TokudoEvent {
    /// Returns the wire `kind` discriminant for this event.
    #[must_use]
    pub fn discriminant(&self) -> &'static str {
        match self {
            TokudoEvent::CacheHit { .. } => "cache.hit",
            TokudoEvent::CacheMiss { .. } => "cache.miss",
            TokudoEvent::RouteDecision { .. } => "route.decision",
            TokudoEvent::CompressApplied { .. } => "compress.applied",
            TokudoEvent::CostEstimate { .. } => "cost.estimate",
            #[cfg(feature = "lineage")]
            TokudoEvent::LineageEdge { .. } => "lineage.edge",
        }
    }
}

/// Envelope wrapping a [`TokudoEvent`] with schema and monotonic ordering
/// metadata. Produced by [`build_event`]; serialized into the `event` field
/// of the emitted `tracing` event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokudoEnvelope {
    /// Schema version. See [`SCHEMA_VERSION`].
    pub version: u32,
    /// Wall-clock timestamp in milliseconds since the Unix epoch.
    pub occurred_at_millis: u64,
    /// Monotonic per-process counter.
    pub tick: u64,
    /// The event payload; flattened so the wire JSON stays a flat object.
    #[serde(flatten)]
    pub event: TokudoEvent,
}

/// Return the next monotonic per-process tick used to order tokudo events.
#[must_use]
pub fn next_tick() -> u64 {
    TICK.fetch_add(1, Ordering::Relaxed)
}

/// Return the current wall-clock time in milliseconds since the Unix epoch.
/// Returns `0` if the clock is set before the epoch.
#[must_use]
pub fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Build a [`TokudoEnvelope`] stamped with the next tick and the current
/// wall time.
#[must_use]
pub fn build_event(event: TokudoEvent) -> TokudoEnvelope {
    TokudoEnvelope {
        version: SCHEMA_VERSION,
        occurred_at_millis: now_millis(),
        tick: next_tick(),
        event,
    }
}

/// Emit `event` on the `rig_tokudo` tracing target. Serialization failures
/// are logged at `warn` and otherwise swallowed; telemetry never panics
/// the decorator hot path.
pub fn emit(event: &TokudoEvent) {
    let envelope = build_event(event.clone());
    match serde_json::to_string(&envelope) {
        Ok(json) => emit_json(event, &envelope, &json),
        Err(err) => {
            tracing::warn!(
                target: EVENT_TARGET,
                error = %err,
                "rig-tokudo: failed to serialize observability event",
            );
        }
    }
}

#[cfg(feature = "tap")]
fn emit_json(event: &TokudoEvent, envelope: &TokudoEnvelope, json: &str) {
    tracing::info!(
        target: EVENT_TARGET,
        event = %json,
        rig_tokudo.version = envelope.version,
        rig_tokudo.kind = event.discriminant(),
        rig_tokudo.tick = envelope.tick,
        rig_tokudo.occurred_at_millis = envelope.occurred_at_millis,
        rig_tap.version = rig_tap::SCHEMA_VERSION,
        rig_tap.kind = event.discriminant(),
        rig_tap.tick = envelope.tick,
        rig_tap.occurred_at_millis = envelope.occurred_at_millis,
        rig_tap.conversation_id = "tokudo",
    );
}

#[cfg(not(feature = "tap"))]
fn emit_json(event: &TokudoEvent, envelope: &TokudoEnvelope, json: &str) {
    tracing::info!(
        target: EVENT_TARGET,
        event = %json,
        rig_tokudo.version = envelope.version,
        rig_tokudo.kind = event.discriminant(),
        rig_tokudo.tick = envelope.tick,
        rig_tokudo.occurred_at_millis = envelope.occurred_at_millis,
    );
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
    fn discriminants_match_wire_strings() {
        assert_eq!(
            TokudoEvent::CacheMiss {
                cache_key: "k".into()
            }
            .discriminant(),
            "cache.miss"
        );
        assert_eq!(
            TokudoEvent::RouteDecision {
                choice: RouterChoice::Cheap
            }
            .discriminant(),
            "route.decision"
        );
        assert_eq!(
            TokudoEvent::CompressApplied {
                input_tokens: 100,
                output_tokens: 60,
                ratio: Some(0.6)
            }
            .discriminant(),
            "compress.applied"
        );
    }

    #[test]
    fn serialized_event_is_flat() {
        let env = build_event(TokudoEvent::CacheHit {
            cache_key: "abc".into(),
            source_id: "abc".into(),
            similarity: None,
        });
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["kind"], "cache.hit");
        assert_eq!(json["cache_key"], "abc");
        assert_eq!(json["version"], SCHEMA_VERSION);
        assert!(json["tick"].is_number());
    }

    #[test]
    fn tick_is_monotonic() {
        let a = next_tick();
        let b = next_tick();
        assert!(b > a);
    }
}
