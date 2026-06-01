//! Feature-gated coverage for the Foyer-backed exact cache.

#![cfg(feature = "cache-foyer")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::OneOrMany;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Message, Usage,
};
use rig::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use serde_json::json;

use rig_tokudo::{
    Cache, CachedEntry, FoyerCache, FoyerCacheConfig, OptimizedModel, RouterChoice, TokudoOptions,
};

fn entry(id: &str, expires_at_secs: Option<u64>) -> CachedEntry {
    CachedEntry {
        response: json!({ "id": id, "nested": { "ok": true } }),
        expires_at_secs,
        similarity: None,
        source_id: id.into(),
    }
}

#[tokio::test]
async fn foyer_cache_miss_then_hit_round_trips_json() {
    let cache = FoyerCache::new();
    assert!(cache.get("a").await.unwrap().is_none());

    cache.put("a", entry("a", None)).await.unwrap();

    let got = cache.get("a").await.unwrap().unwrap();
    assert_eq!(got.source_id, "a");
    assert_eq!(got.response["nested"]["ok"], json!(true));
}

#[tokio::test]
async fn foyer_cache_expired_entry_returns_miss_and_removes_entry() {
    let cache = FoyerCache::new();
    cache
        .put("expired", entry("expired", Some(0)))
        .await
        .unwrap();

    assert!(cache.get("expired").await.unwrap().is_none());
    assert!(cache.is_empty());
}

#[tokio::test]
async fn foyer_cache_capacity_zero_stores_nothing() {
    let cache = FoyerCache::with_config(FoyerCacheConfig {
        capacity: 0,
        shards: 1,
        name: "tokudo-test-zero".into(),
    });

    cache.put("a", entry("a", None)).await.unwrap();

    assert!(cache.get("a").await.unwrap().is_none());
    assert!(cache.is_empty());
}

#[tokio::test]
async fn foyer_cache_capacity_pressure_stays_bounded() {
    let cache = FoyerCache::with_config(FoyerCacheConfig {
        capacity: 1,
        shards: 1,
        name: "tokudo-test-bound".into(),
    });

    cache.put("a", entry("a", None)).await.unwrap();
    assert!(cache.get("a").await.unwrap().is_some());
    cache.put("b", entry("b", None)).await.unwrap();

    let a = cache.get("a").await.unwrap();
    let b = cache.get("b").await.unwrap();
    assert!(
        cache.len() <= 1,
        "foyer should keep entries within capacity"
    );
    assert!(
        a.is_none() || b.is_none(),
        "a capacity-1 cache cannot retain both entries"
    );
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct MockRaw {
    text: String,
    n: u32,
}

#[derive(Clone)]
struct MockModel {
    calls: Arc<AtomicUsize>,
    reply: String,
}

impl MockModel {
    fn new(reply: impl Into<String>) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            reply: reply.into(),
        }
    }
}

impl CompletionModel for MockModel {
    type Response = MockRaw;
    type StreamingResponse = ();
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self::new("mock")
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) as u32;
        let text = self.reply.clone();
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(text.clone())),
            usage: Usage::new(),
            raw_response: MockRaw { text, n },
            message_id: Some(format!("msg-{n}")),
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Err(CompletionError::ProviderError(
            "stream not supported in mock".into(),
        ))
    }
}

fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: Some("mock-model".into()),
        preamble: Some("system".into()),
        chat_history: OneOrMany::one(Message::user(prompt)),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: Some(0.7),
        max_tokens: Some(256),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    }
}

#[tokio::test]
async fn optimized_model_uses_foyer_cache_on_repeat_request() {
    let mock = MockModel::new("hello foyer");
    let calls = mock.calls.clone();
    let model = OptimizedModel::builder(mock)
        .with_cache(FoyerCache::new())
        .build();

    let first = model
        .complete(request("hi"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!first.provenance.cache_hit);

    let second = model
        .complete(request("hi"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(second.provenance.cache_hit);
    assert_eq!(second.provenance.router_choice, RouterChoice::CacheHit);
    assert_eq!(second.response.raw_response.text, "hello foyer");
    assert_eq!(second.response.raw_response.n, 0);
}
