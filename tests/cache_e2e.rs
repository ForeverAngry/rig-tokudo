//! End-to-end tests for the Phase 2 cache pillar: a mock `CompletionModel`
//! observes how many times the decorator forwards requests to the wrapped
//! provider, and we assert that repeated identical requests are served from
//! the in-memory cache.

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

use rig_tokudo::{CachePolicy, InMemoryCache, OptimizedModel, RouterChoice, TokudoOptions};

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
        MockModel::new("mock")
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
async fn second_identical_request_is_served_from_cache() {
    let mock = MockModel::new("hello world");
    let calls = mock.calls.clone();
    let model = OptimizedModel::builder(mock)
        .with_cache(InMemoryCache::new())
        .build();

    // First call: provider invoked, response written to cache.
    let r1 = model
        .complete(request("hi"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!r1.provenance.cache_hit);
    assert_eq!(r1.provenance.router_choice, RouterChoice::PassThrough);

    // Second identical call: hit reconstruction, no provider call.
    let r2 = model
        .complete(request("hi"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "provider must not be re-invoked on cache hit"
    );
    assert!(r2.provenance.cache_hit, "second call should report hit");
    assert_eq!(r2.provenance.router_choice, RouterChoice::CacheHit);
    assert!(r2.provenance.source_id.is_some());
    #[cfg(feature = "lineage")]
    {
        let Some(edge) = r2.provenance.lineage.as_ref() else {
            panic!("lineage edge should be present");
        };
        assert_eq!(edge.router_choice, RouterChoice::CacheHit);
        assert_eq!(edge.relation, rig_tokudo::LineageRelation::CacheHit);
        assert_eq!(edge.target_id, "msg-0");
    }

    // Reconstructed response payload matches what the provider returned.
    assert_eq!(r2.response.raw_response.text, "hello world");
    assert_eq!(r2.response.raw_response.n, 0);
    assert_eq!(r2.response.message_id.as_deref(), Some("msg-0"));
}

#[tokio::test]
async fn different_prompts_miss_the_cache() {
    let mock = MockModel::new("answer");
    let calls = mock.calls.clone();
    let model = OptimizedModel::builder(mock)
        .with_cache(InMemoryCache::new())
        .build();

    model
        .complete(request("alpha"), TokudoOptions::new())
        .await
        .unwrap();
    model
        .complete(request("beta"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn force_fresh_bypasses_read_but_still_writes() {
    let mock = MockModel::new("hi");
    let calls = mock.calls.clone();
    let model = OptimizedModel::builder(mock)
        .with_cache(InMemoryCache::new())
        .build();

    // Prime the cache.
    model
        .complete(request("hi"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // ForceFresh: bypass the read path → provider invoked again.
    let r = model
        .complete(
            request("hi"),
            TokudoOptions::new().with_cache_policy(CachePolicy::ForceFresh),
        )
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(!r.provenance.cache_hit);

    // Subsequent default call should now read the fresh entry from cache.
    let r3 = model
        .complete(request("hi"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2, "third call must hit cache");
    assert!(r3.provenance.cache_hit);
}

#[tokio::test]
async fn no_store_disables_writes() {
    let mock = MockModel::new("hi");
    let calls = mock.calls.clone();
    let model = OptimizedModel::builder(mock)
        .with_cache(InMemoryCache::new())
        .build();

    model
        .complete(
            request("hi"),
            TokudoOptions::new().with_cache_policy(CachePolicy::NoStore),
        )
        .await
        .unwrap();
    // Second call must miss because NoStore skipped the write.
    model
        .complete(request("hi"), TokudoOptions::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
