//! Durable semantic cache backed by `rig-memvid`.

#[cfg(not(target_family = "wasm"))]
use rig_memvid::memvid_core::{PutOptions, SearchRequest};
#[cfg(not(target_family = "wasm"))]
use rig_memvid::{MemvidError, MemvidStore};

#[cfg(not(target_family = "wasm"))]
use crate::cache::SemanticCacheConfig;
#[cfg(not(target_family = "wasm"))]
use crate::error::{Result, TokudoError};
#[cfg(not(target_family = "wasm"))]
use crate::traits::{Cache, CachedEntry};

/// Configuration for [`MemvidSemanticCache`].
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone, PartialEq)]
pub struct MemvidSemanticCacheConfig {
    /// Read-side semantic search settings.
    pub search: SemanticCacheConfig,
    /// URI prefix used for cache frames.
    pub uri_prefix: String,
    /// Track name stored in memvid frame metadata.
    pub track: String,
    /// Kind stored in memvid frame metadata.
    pub kind: String,
}

#[cfg(not(target_family = "wasm"))]
impl Default for MemvidSemanticCacheConfig {
    fn default() -> Self {
        Self {
            search: SemanticCacheConfig {
                threshold: None,
                ..Default::default()
            },
            uri_prefix: "tokudo-cache".to_string(),
            track: "rig_tokudo_cache".to_string(),
            kind: "cached_completion".to_string(),
        }
    }
}

/// Durable semantic cache using a [`rig_memvid::MemvidStore`].
///
/// Cache entries are stored as text frames with a searchable key prefix plus
/// serialized [`CachedEntry`] JSON for reconstruction on hits.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone)]
pub struct MemvidSemanticCache {
    store: MemvidStore,
    config: MemvidSemanticCacheConfig,
}

#[cfg(not(target_family = "wasm"))]
impl MemvidSemanticCache {
    /// Construct with [`MemvidSemanticCacheConfig::default`].
    #[must_use]
    pub fn new(store: MemvidStore) -> Self {
        Self::with_config(store, MemvidSemanticCacheConfig::default())
    }

    /// Construct with explicit configuration.
    #[must_use]
    pub fn with_config(store: MemvidStore, config: MemvidSemanticCacheConfig) -> Self {
        Self { store, config }
    }

    /// Borrow the underlying store.
    #[must_use]
    pub fn store(&self) -> &MemvidStore {
        &self.store
    }
}

#[cfg(not(target_family = "wasm"))]
impl Cache for MemvidSemanticCache {
    async fn get(&self, key: &str) -> Result<Option<CachedEntry>> {
        let response = self
            .store
            .search(SearchRequest {
                query: key.to_string(),
                top_k: samples_to_top_k(self.config.search.samples),
                snippet_chars: MEMVID_CACHE_SNIPPET_CHARS,
                uri: None,
                scope: None,
                cursor: None,
                as_of_frame: None,
                as_of_ts: None,
                no_sketch: true,
                acl_context: None,
                acl_enforcement_mode: Default::default(),
            })
            .map_err(map_memvid_err)?;

        for hit in response.hits {
            let score = hit_score(&hit);
            if let Some(threshold) = self.config.search.threshold
                && score < threshold
            {
                continue;
            }
            let mut entry = match decode_cache_frame(&hit.text) {
                Ok(entry) => entry,
                Err(err) => {
                    tracing::debug!(
                        target: "rig_tokudo::cache::memvid",
                        frame_id = hit.frame_id,
                        error = %err,
                        "skipping non-cache memvid hit"
                    );
                    continue;
                }
            };
            entry.similarity = Some(score as f32);
            if entry.source_id.is_empty() {
                entry.source_id = hit.frame_id.to_string();
            }
            return Ok(Some(entry));
        }
        Ok(None)
    }

    async fn put(&self, key: &str, entry: CachedEntry) -> Result<()> {
        let text = encode_cache_frame(key, &entry)?;
        let options = PutOptions {
            uri: Some(format!(
                "{}/{}",
                self.config.uri_prefix,
                stable_component(key)
            )),
            title: Some("Tokudo cached completion".to_string()),
            track: Some(self.config.track.clone()),
            kind: Some(self.config.kind.clone()),
            auto_tag: false,
            extract_dates: false,
            extract_triplets: false,
            ..Default::default()
        };
        self.store
            .put_text(&text, options)
            .map(|_| ())
            .map_err(map_memvid_err)
    }
}

#[cfg(not(target_family = "wasm"))]
fn encode_cache_frame(key: &str, entry: &CachedEntry) -> Result<String> {
    let json = serde_json::to_string(&cache_frame_json(key, entry)?)
        .map_err(|e| TokudoError::Cache(format!("memvid cache encode: {e}")))?;
    Ok(format!("tokudo_cache_key: {key}\n{json}"))
}

#[cfg(not(target_family = "wasm"))]
fn decode_cache_frame(text: &str) -> std::result::Result<CachedEntry, serde_json::Error> {
    match text.find('{') {
        Some(start) => {
            let mut deserializer = serde_json::Deserializer::from_str(&text[start..]);
            serde::Deserialize::deserialize(&mut deserializer)
        }
        None => serde_json::from_str(text),
    }
}

#[cfg(not(target_family = "wasm"))]
fn cache_frame_json(key: &str, entry: &CachedEntry) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(entry)
        .map_err(|e| TokudoError::Cache(format!("memvid cache encode: {e}")))?;
    if let serde_json::Value::Object(object) = &mut value {
        object.insert(
            "tokudo_cache_key".to_string(),
            serde_json::Value::String(key.to_string()),
        );
    }
    Ok(value)
}

#[cfg(not(target_family = "wasm"))]
const MEMVID_CACHE_SNIPPET_CHARS: usize = 64 * 1024;

#[cfg(not(target_family = "wasm"))]
fn samples_to_top_k(samples: u64) -> usize {
    usize::try_from(samples).unwrap_or(1024).clamp(1, 1024)
}

#[cfg(not(target_family = "wasm"))]
fn hit_score(hit: &rig_memvid::memvid_core::SearchHit) -> f64 {
    match hit.score {
        Some(score) => f64::from(score),
        None => {
            let rank = u32::try_from(hit.rank).unwrap_or(u32::MAX);
            1.0 / (f64::from(rank) + 1.0)
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn stable_component(key: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(not(target_family = "wasm"))]
fn map_memvid_err(err: MemvidError) -> TokudoError {
    TokudoError::Cache(format!("memvid cache: {err}"))
}
