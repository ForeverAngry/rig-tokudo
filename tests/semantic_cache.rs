//! Integration tests for the read-through semantic cache adapter.

#![cfg(feature = "cache-semantic")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::{Arc, Mutex};

use rig::vector_store::request::Filter;
use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use serde_json::Value;

use rig_tokudo::{Cache, CachedEntry, SemanticCache, SemanticCacheConfig};

#[derive(Clone, Default)]
struct MockIndex {
    queries: Arc<Mutex<Vec<String>>>,
    hits: Arc<Vec<(f64, String, Value)>>,
}

impl MockIndex {
    fn with_hits(hits: Vec<(f64, String, Value)>) -> Self {
        Self {
            queries: Arc::new(Mutex::new(Vec::new())),
            hits: Arc::new(hits),
        }
    }
}

impl VectorStoreIndex for MockIndex {
    type Filter = Filter<Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.queries.lock().unwrap().push(req.query().to_string());
        self.hits
            .iter()
            .map(|(score, id, value)| {
                let doc = serde_json::from_value::<T>(value.clone())?;
                Ok((*score, id.clone(), doc))
            })
            .collect()
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self
            .hits
            .iter()
            .map(|(score, id, _value)| (*score, id.clone()))
            .collect())
    }
}

fn cached_entry(id: &str) -> CachedEntry {
    CachedEntry {
        response: serde_json::json!({
            "choice": { "One": { "type": "text", "text": "cached" } },
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "reasoning_tokens": 0
            },
            "message_id": null,
            "raw_response": { "text": "cached" }
        }),
        expires_at_secs: None,
        similarity: None,
        source_id: id.into(),
    }
}

#[tokio::test]
async fn semantic_cache_returns_first_vector_hit_as_entry() {
    let entry = cached_entry("doc-1");
    let index = MockIndex::with_hits(vec![(
        0.91,
        "doc-1".into(),
        serde_json::to_value(entry).unwrap(),
    )]);
    let queries = index.queries.clone();
    let cache = SemanticCache::with_config(
        Arc::new(index),
        SemanticCacheConfig {
            samples: 1,
            threshold: Some(0.80),
        },
    );

    let hit = cache.get("semantic prompt").await.unwrap().unwrap();
    assert_eq!(hit.source_id, "doc-1");
    assert_eq!(hit.similarity, Some(0.91_f32));
    assert_eq!(queries.lock().unwrap().as_slice(), &["semantic prompt"]);
}

#[tokio::test]
async fn semantic_cache_put_is_noop_for_read_only_index() {
    let cache = SemanticCache::new(Arc::new(MockIndex::default()));
    cache.put("k", cached_entry("doc")).await.unwrap();
    assert!(cache.get("k").await.unwrap().is_none());
}
