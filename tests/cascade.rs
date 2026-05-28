//! Integration tests for the Phase 4 [`StaticCascade`]: two mock completion
//! models (a cheap one returning a short reply, a strong one returning a long
//! reply) confirm the cascade falls back from cheap to strong when the
//! validator rejects.

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

use rig_tokudo::{
    InMemoryCache, LengthValidator, OptimizedModel, RouterChoice, StaticCascade, TokudoOptions,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct MockRaw {
    text: String,
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text = self.reply.clone();
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(text.clone())),
            usage: Usage::new(),
            raw_response: MockRaw { text },
            message_id: None,
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
async fn cheap_response_is_accepted_when_validator_passes() {
    let cheap = MockModel::new("a sufficiently long reply for the floor");
    let strong = MockModel::new("strong reply (should not be called)");
    let cheap_calls = cheap.calls.clone();
    let strong_calls = strong.calls.clone();

    let cascade = StaticCascade::new(cheap, strong, LengthValidator::new(10));
    let normalized = cascade.complete(request("hi")).await.unwrap();

    assert_eq!(cheap_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        strong_calls.load(Ordering::SeqCst),
        0,
        "strong leg must not be invoked when cheap is accepted",
    );
    assert_eq!(normalized.provenance.router_choice, RouterChoice::Cheap);
    assert_eq!(normalized.response.raw_response.text.len(), 39);
}

#[tokio::test]
async fn cheap_rejection_falls_back_to_strong() {
    let cheap = MockModel::new("nope"); // 4 chars — below the floor.
    let strong = MockModel::new("strong reply, plenty of characters here");
    let cheap_calls = cheap.calls.clone();
    let strong_calls = strong.calls.clone();

    let cascade = StaticCascade::new(cheap, strong, LengthValidator::new(20));
    let normalized = cascade.complete(request("hi")).await.unwrap();

    assert_eq!(cheap_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        strong_calls.load(Ordering::SeqCst),
        1,
        "strong leg must be invoked on cheap rejection",
    );
    assert_eq!(normalized.provenance.router_choice, RouterChoice::Strong);
    assert!(normalized.response.raw_response.text.starts_with("strong"));
}

#[tokio::test]
async fn cascade_plugs_into_optimized_model_and_caches_strong_reply() {
    // StaticCascade plugged in as the inner model of OptimizedModel.
    // Cheap leg returns a 4-char reply that LengthValidator{min:20} rejects;
    // cascade falls back to the strong leg. The strong reply is then cached
    // by the OptimizedModel layer, so a second call returns it as a hit
    // without invoking either leg.
    let cheap = MockModel::new("nope");
    let strong = MockModel::new("strong reply, plenty of characters here");
    let cheap_calls = cheap.calls.clone();
    let strong_calls = strong.calls.clone();

    let cascade = StaticCascade::new(cheap, strong, LengthValidator::new(20));
    let optimized = OptimizedModel::builder(cascade)
        .with_cache(InMemoryCache::new())
        .build();

    let first = optimized
        .complete(request("hi"), TokudoOptions::default())
        .await
        .unwrap();
    assert!(!first.provenance.cache_hit);
    assert_eq!(
        first.provenance.router_choice,
        RouterChoice::Strong,
        "cascade decision must propagate into OptimizedModel provenance",
    );
    assert_eq!(cheap_calls.load(Ordering::SeqCst), 1);
    assert_eq!(strong_calls.load(Ordering::SeqCst), 1);

    let second = optimized
        .complete(request("hi"), TokudoOptions::default())
        .await
        .unwrap();
    assert!(second.provenance.cache_hit);
    assert_eq!(second.provenance.router_choice, RouterChoice::CacheHit);
    assert_eq!(
        cheap_calls.load(Ordering::SeqCst),
        1,
        "cache hit must not invoke the cheap leg again",
    );
    assert_eq!(
        strong_calls.load(Ordering::SeqCst),
        1,
        "cache hit must not invoke the strong leg again",
    );
}
