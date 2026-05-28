//! Semantic-cache adapter over Rig's dynamic vector-store retrieval surface.
//!
//! [`SemanticCache`] is intentionally read-through: [`rig::vector_store::VectorStoreIndexDyn`]
//! exposes search but not insertion, so `put` is a no-op and callers should
//! populate the underlying vector store through its native ingestion path.
//! Pair it with [`SemanticCacheKey`] so the cache lookup key is human text
//! rather than the exact-match SHA-256 key.

use std::sync::Arc;

use rig::completion::CompletionRequest;
use rig::vector_store::{VectorSearchRequest, VectorStoreIndexDyn, request::Filter};

use crate::error::{Result, TokudoError};
use crate::options::TokudoOptions;
use crate::traits::{Cache, CacheKey, CachedEntry};

/// Configuration for [`SemanticCache`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticCacheConfig {
    /// Number of vector hits to request. The first deserializable hit wins.
    pub samples: u64,
    /// Optional minimum similarity threshold delegated to the vector store.
    pub threshold: Option<f64>,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            samples: 1,
            threshold: Some(0.85),
        }
    }
}

/// Read-through cache backed by any `VectorStoreIndexDyn`.
///
/// The vector documents are expected to deserialize as [`CachedEntry`]. On
/// hit, the vector score is copied into `CachedEntry::similarity`, and an
/// empty `source_id` is replaced by the backend result id.
pub struct SemanticCache {
    index: Arc<dyn VectorStoreIndexDyn + Send + Sync>,
    config: SemanticCacheConfig,
}

impl SemanticCache {
    /// Construct with [`SemanticCacheConfig::default`].
    #[must_use]
    pub fn new(index: Arc<dyn VectorStoreIndexDyn + Send + Sync>) -> Self {
        Self::with_config(index, SemanticCacheConfig::default())
    }

    /// Construct with explicit configuration.
    #[must_use]
    pub fn with_config(
        index: Arc<dyn VectorStoreIndexDyn + Send + Sync>,
        config: SemanticCacheConfig,
    ) -> Self {
        Self { index, config }
    }
}

impl Cache for SemanticCache {
    async fn get(&self, key: &str) -> Result<Option<CachedEntry>> {
        let mut builder = VectorSearchRequest::<Filter<serde_json::Value>>::builder()
            .query(key.to_string())
            .samples(self.config.samples);
        if let Some(threshold) = self.config.threshold {
            builder = builder.threshold(threshold);
        }
        let req = builder.build();
        let hits = self
            .index
            .top_n(req)
            .await
            .map_err(|e| TokudoError::Cache(format!("semantic cache search: {e}")))?;

        if let Some((score, id, value)) = hits.into_iter().next() {
            let mut entry: CachedEntry = serde_json::from_value(value)
                .map_err(|e| TokudoError::Cache(format!("semantic cache decode: {e}")))?;
            entry.similarity = Some(score as f32);
            if entry.source_id.is_empty() {
                entry.source_id = id;
            }
            return Ok(Some(entry));
        }
        Ok(None)
    }

    async fn put(&self, _key: &str, _entry: CachedEntry) -> Result<()> {
        Ok(())
    }
}

/// Cache key strategy for semantic lookup.
///
/// Unlike [`crate::DefaultCacheKey`], this returns a normalized textual
/// projection of the request so vector stores can embed and search it.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemanticCacheKey;

impl CacheKey for SemanticCacheKey {
    fn key(&self, request: &CompletionRequest, options: &TokudoOptions) -> Result<String> {
        let projection = serde_json::json!({
            "preamble": request.preamble,
            "chat_history": request.chat_history,
            "namespace": options.cache_namespace,
        });
        serde_json::to_string(&projection)
            .map_err(|e| TokudoError::Cache(format!("serialize semantic key: {e}")))
    }
}
