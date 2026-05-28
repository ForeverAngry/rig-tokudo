//! Public traits for the four decorator pillars: caching, compression,
//! routing, and validation.
//!
//! All four traits are async-agnostic in shape but use `async fn` where the
//! operation may legitimately await (cache I/O). Compression and routing are
//! synchronous because every implementation we anticipate is pure compute.
//!
//! Reference implementations live alongside the trait definitions for
//! discoverability and to ensure the type parameter defaults on
//! [`crate::OptimizedModel`] always resolve to something useful.

use rig::completion::CompletionRequest;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::options::TokudoOptions;
use crate::provenance::RouterChoice;

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// Strategy for converting a [`CompletionRequest`] + [`TokudoOptions`] into a
/// stable cache key.
///
/// Keys MUST be deterministic across runs and process restarts. They MUST
/// change when any semantically relevant field of the request changes
/// (preamble, tool registry signature, message slice, temperature bucket,
/// namespace). They MUST NOT change for fields that don't affect output
/// (e.g. `additional_params` ordering when irrelevant).
pub trait CacheKey: Send + Sync {
    /// Produce the key string for this request/options pair.
    fn key(&self, request: &CompletionRequest, options: &TokudoOptions) -> Result<String>;
}

/// Default cache key strategy: SHA-256 of a canonical JSON projection of the
/// request and the relevant subset of options.
///
/// Phase 1 hashes preamble, chat history, tool signatures (name + description),
/// temperature bucket (rounded to 0.1), max_tokens, output_schema presence,
/// and the cache namespace. The temperature bucket coarsens floating-point
/// noise so semantically-identical requests collide.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultCacheKey;

impl CacheKey for DefaultCacheKey {
    fn key(&self, request: &CompletionRequest, options: &TokudoOptions) -> Result<String> {
        use sha2::{Digest, Sha256};

        // Project to a stable, ordered serde value.
        let temp_bucket = request
            .temperature
            .map(|t| (t * 10.0).round() as i64)
            .unwrap_or(0);

        let tool_sig: Vec<_> = request
            .tools
            .iter()
            .map(|t| serde_json::json!({ "name": t.name, "desc": t.description }))
            .collect();

        let projection = serde_json::json!({
            "preamble": request.preamble,
            "model": request.model,
            "chat_history": request.chat_history,
            "tools": tool_sig,
            "temperature_bucket": temp_bucket,
            "max_tokens": request.max_tokens,
            "has_output_schema": request.output_schema.is_some(),
            "namespace": options.cache_namespace,
        });

        let canonical = serde_json::to_vec(&projection)
            .map_err(|e| crate::error::TokudoError::Cache(format!("serialize key: {e}")))?;

        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        Ok(hex::encode(hasher.finalize()))
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// A serialized cache entry. Stores the provider response as a JSON value so
/// the `Cache` trait stays object-safe and decoupled from `M::Response`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEntry {
    /// The serialized provider response (typically `serde_json::Value`).
    pub response: serde_json::Value,
    /// Unix-epoch seconds at which this entry should be considered expired.
    pub expires_at_secs: Option<u64>,
    /// Optional similarity score recorded by a semantic cache.
    pub similarity: Option<f32>,
    /// Stable identifier for this entry (e.g. the storage key).
    pub source_id: String,
}

/// Cache backend used by [`crate::OptimizedModel`].
///
/// Phase 1 ships [`NoCache`] as the default. Phase 2 introduces in-memory
/// and semantic (vector-store-backed) implementations.
#[allow(async_fn_in_trait)]
pub trait Cache: Send + Sync {
    /// Look up a key. Return `Ok(None)` on miss (not an error).
    async fn get(&self, key: &str) -> Result<Option<CachedEntry>>;

    /// Insert a key/value pair. TTL is encoded inside `entry.expires_at_secs`.
    async fn put(&self, key: &str, entry: CachedEntry) -> Result<()>;
}

/// A cache that never returns a hit and never stores anything.
///
/// The default backend for [`crate::OptimizedModel`] until a real cache is
/// wired in via [`crate::OptimizedModelBuilder::with_cache`].
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCache;

impl Cache for NoCache {
    async fn get(&self, _key: &str) -> Result<Option<CachedEntry>> {
        Ok(None)
    }

    async fn put(&self, _key: &str, _entry: CachedEntry) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------

/// Result of a compression pass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Approximate token count before compression.
    pub input_tokens: u32,
    /// Approximate token count after compression.
    pub output_tokens: u32,
}

impl CompressionStats {
    /// Compressed token ratio (output / input). `None` when input is zero.
    #[must_use]
    pub fn ratio(&self) -> Option<f32> {
        if self.input_tokens == 0 {
            None
        } else {
            Some(self.output_tokens as f32 / self.input_tokens as f32)
        }
    }
}

/// Pure-compute prompt compressor. Mutates the request in place and returns
/// stats describing the work.
///
/// Implementations MUST preserve any field they don't explicitly understand.
pub trait Compressor: Send + Sync {
    /// Compress the request. Return `Ok(None)` when no compression is applied
    /// (e.g. opted out for this call, or input below threshold).
    fn compress(
        &self,
        request: &mut CompletionRequest,
        options: &TokudoOptions,
    ) -> Result<Option<CompressionStats>>;
}

/// Compressor that never modifies the request.
///
/// The Phase 1 default; replaced by `JsonKeyPruner` in Phase 3.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCompressor;

impl Compressor for NoCompressor {
    fn compress(
        &self,
        _request: &mut CompletionRequest,
        _options: &TokudoOptions,
    ) -> Result<Option<CompressionStats>> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Router + Validator
// ---------------------------------------------------------------------------

/// Pre-call routing decision.
///
/// A `Router` advertises which leg of the cascade should answer; the
/// [`crate::OptimizedModel`] turns that hint into actual dispatch. Routers
/// that need to inspect provider responses (RouteLLM-style learned routers)
/// will arrive in Phase 4 alongside `route-predictive`.
pub trait Router: Send + Sync {
    /// Decide which leg should answer this request.
    fn route(&self, request: &CompletionRequest, options: &TokudoOptions) -> RouterChoice;
}

/// Pass-through router: always advertises [`RouterChoice::PassThrough`].
///
/// The Phase 1 default; replaced by `StaticCascade` in Phase 4.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassThroughRouter;

impl Router for PassThroughRouter {
    fn route(&self, _request: &CompletionRequest, _options: &TokudoOptions) -> RouterChoice {
        RouterChoice::PassThrough
    }
}

/// Post-call response validator. Returns `true` when the response is good
/// enough to keep; `false` triggers cascade fallback.
///
/// Validators are called with the raw provider response serialized as a JSON
/// value to keep the trait object-safe and decoupled from `M::Response`.
pub trait Validator: Send + Sync {
    /// Score the response. `true` accepts; `false` rejects.
    fn validate(&self, response: &serde_json::Value) -> Result<bool>;
}

/// Validator that accepts every response.
///
/// The Phase 1 default; replaced by length/regex/confidence validators in
/// Phase 4.
#[derive(Debug, Default, Clone, Copy)]
pub struct AcceptAllValidator;

impl Validator for AcceptAllValidator {
    fn validate(&self, _response: &serde_json::Value) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use rig::OneOrMany;
    use rig::completion::Message;

    fn request(prompt: &str) -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: Some("be brief".into()),
            chat_history: OneOrMany::one(Message::user(prompt)),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(256),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    #[test]
    fn default_key_is_deterministic() {
        let key = DefaultCacheKey;
        let req = request("hello");
        let opts = TokudoOptions::default();
        let a = key.key(&req, &opts).unwrap();
        let b = key.key(&req, &opts).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64, "sha256 hex should be 64 chars");
    }

    #[test]
    fn default_key_changes_on_prompt_change() {
        let key = DefaultCacheKey;
        let opts = TokudoOptions::default();
        let a = key.key(&request("hello"), &opts).unwrap();
        let b = key.key(&request("goodbye"), &opts).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn default_key_changes_on_namespace_change() {
        let key = DefaultCacheKey;
        let req = request("hello");
        let a = key
            .key(&req, &TokudoOptions::new().with_cache_namespace("alice"))
            .unwrap();
        let b = key
            .key(&req, &TokudoOptions::new().with_cache_namespace("bob"))
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn temperature_bucket_collapses_noise() {
        let key = DefaultCacheKey;
        let opts = TokudoOptions::default();
        let mut a = request("hello");
        let mut b = request("hello");
        a.temperature = Some(0.70);
        b.temperature = Some(0.71); // same bucket after *10 rounding (7).
        assert_eq!(key.key(&a, &opts).unwrap(), key.key(&b, &opts).unwrap());

        b.temperature = Some(0.80); // different bucket (8).
        assert_ne!(key.key(&a, &opts).unwrap(), key.key(&b, &opts).unwrap());
    }

    #[tokio::test]
    async fn no_cache_misses_then_no_ops() {
        let cache = NoCache;
        assert!(cache.get("any").await.unwrap().is_none());
        let entry = CachedEntry {
            response: serde_json::json!({}),
            expires_at_secs: None,
            similarity: None,
            source_id: "x".into(),
        };
        cache.put("any", entry).await.unwrap();
        assert!(cache.get("any").await.unwrap().is_none());
    }

    #[test]
    fn no_compressor_is_noop() {
        let c = NoCompressor;
        let mut req = request("hello");
        let before = req.preamble.clone();
        let stats = c.compress(&mut req, &TokudoOptions::default()).unwrap();
        assert!(stats.is_none());
        assert_eq!(req.preamble, before);
    }

    #[test]
    fn pass_through_router_always_passes() {
        let r = PassThroughRouter;
        assert_eq!(
            r.route(&request("x"), &TokudoOptions::default()),
            RouterChoice::PassThrough
        );
    }

    #[test]
    fn accept_all_validator_accepts() {
        let v = AcceptAllValidator;
        assert!(v.validate(&serde_json::json!("anything")).unwrap());
    }

    #[test]
    fn compression_stats_ratio() {
        let s = CompressionStats {
            input_tokens: 100,
            output_tokens: 25,
        };
        assert_eq!(s.ratio(), Some(0.25));

        let zero = CompressionStats {
            input_tokens: 0,
            output_tokens: 0,
        };
        assert_eq!(zero.ratio(), None);
    }
}
