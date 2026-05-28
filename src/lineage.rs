//! Feature-gated lineage records for Tokudo calls.
//!
//! A lineage edge links the response served by [`crate::OptimizedModel`] to
//! the source that produced it: a cache entry on cache hits, or the provider
//! call on misses / forced refreshes. The record is intentionally compact so
//! hosts can persist it in logs, reports, or audit stores without depending on
//! a runtime service.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::provenance::RouterChoice;

/// Relationship represented by a [`LineageEdge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineageRelation {
    /// The served response was reconstructed from a Tokudo cache entry.
    CacheHit,
    /// The served response came from the wrapped provider or dispatcher.
    ProviderCall,
}

/// A compact audit edge connecting a served response to its source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageEdge {
    /// Stable ID derived from the relation, source, target, and model name.
    pub edge_id: String,
    /// Source entity ID: cache entry ID for hits, request cache key for calls.
    pub source_id: String,
    /// Target entity ID: served response ID when available, otherwise the
    /// cache key for deterministic correlation.
    pub target_id: String,
    /// Relationship type.
    pub relation: LineageRelation,
    /// Route that answered the request.
    pub router_choice: RouterChoice,
    /// Model identifier from the request, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Tokudo cache key for the request.
    pub cache_key: String,
}

impl LineageEdge {
    /// Build a cache-hit lineage edge from the cache entry to the served
    /// reconstructed response.
    #[must_use]
    pub fn cache_hit(
        cache_key: impl Into<String>,
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        model: Option<String>,
    ) -> Self {
        let cache_key = cache_key.into();
        let source_id = source_id.into();
        let target_id = target_id.into();
        Self::new(
            LineageRelation::CacheHit,
            source_id,
            target_id,
            RouterChoice::CacheHit,
            model,
            cache_key,
        )
    }

    /// Build a provider-call lineage edge from the request cache key to the
    /// provider response.
    #[must_use]
    pub fn provider_call(
        cache_key: impl Into<String>,
        target_id: impl Into<String>,
        router_choice: RouterChoice,
        model: Option<String>,
    ) -> Self {
        let cache_key = cache_key.into();
        let target_id = target_id.into();
        Self::new(
            LineageRelation::ProviderCall,
            cache_key.clone(),
            target_id,
            router_choice,
            model,
            cache_key,
        )
    }

    fn new(
        relation: LineageRelation,
        source_id: String,
        target_id: String,
        router_choice: RouterChoice,
        model: Option<String>,
        cache_key: String,
    ) -> Self {
        let edge_id = edge_id(
            relation,
            &source_id,
            &target_id,
            router_choice,
            model.as_deref(),
        );
        Self {
            edge_id,
            source_id,
            target_id,
            relation,
            router_choice,
            model,
            cache_key,
        }
    }
}

fn edge_id(
    relation: LineageRelation,
    source_id: &str,
    target_id: &str,
    router_choice: RouterChoice,
    model: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{relation:?}").as_bytes());
    hasher.update(b"\0");
    hasher.update(source_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(target_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(format!("{router_choice:?}").as_bytes());
    hasher.update(b"\0");
    if let Some(model) = model {
        hasher.update(model.as_bytes());
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn edge_ids_are_stable_for_same_inputs() {
        let first = LineageEdge::provider_call(
            "cache-key",
            "message-id",
            RouterChoice::PassThrough,
            Some("mock".to_string()),
        );
        let second = LineageEdge::provider_call(
            "cache-key",
            "message-id",
            RouterChoice::PassThrough,
            Some("mock".to_string()),
        );

        assert_eq!(first.edge_id, second.edge_id);
    }

    #[test]
    fn cache_hit_edge_records_cache_relation() {
        let edge = LineageEdge::cache_hit("cache-key", "entry-1", "msg-1", None);

        assert_eq!(edge.relation, LineageRelation::CacheHit);
        assert_eq!(edge.router_choice, RouterChoice::CacheHit);
        assert_eq!(edge.source_id, "entry-1");
    }
}
