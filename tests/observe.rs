//! Phase 5 integration test: capture rig-tokudo observability events with a
//! custom `tracing-subscriber` layer and assert that the decorator emits the
//! expected `cache.miss`, `cache.hit`, `cost.estimate`, and `route.decision`
//! events on the `rig_tokudo` target.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::sync::{Arc, Mutex};

use rig::OneOrMany;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Message, Usage,
};
use rig::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing::subscriber::with_default;
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

use rig_tokudo::{InMemoryCache, OptimizedModel, TokudoOptions};

// ---------------------------------------------------------------------------
// Mock completion model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct MockRaw {
    text: String,
}

#[derive(Clone)]
struct MockModel {
    reply: String,
}

impl CompletionModel for MockModel {
    type Response = MockRaw;
    type StreamingResponse = ();
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        MockModel {
            reply: "mock".into(),
        }
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let text = self.reply.clone();
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(text.clone())),
            usage: Usage {
                input_tokens: 7,
                output_tokens: 11,
                total_tokens: 18,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_tokens: 0,
            },
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
        model: Some("openai:gpt-4o-mini".into()),
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

// ---------------------------------------------------------------------------
// Capturing layer
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CapturedEvent {
    event_json: Option<String>,
    kind: Option<String>,
}

#[derive(Clone, Default)]
struct CapturingLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

struct CaptureVisitor<'a> {
    captured: &'a mut CapturedEvent,
}

impl Visit for CaptureVisitor<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "event" => self.captured.event_json = Some(value.to_string()),
            "rig_tokudo.kind" => self.captured.kind = Some(value.to_string()),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let rendered = format!("{value:?}");
        match field.name() {
            "event" => {
                // Display-formatted strings arrive here wrapped in quotes;
                // strip them to recover the raw JSON payload.
                let trimmed = rendered.trim_matches('"').replace("\\\"", "\"");
                self.captured.event_json = Some(trimmed);
            }
            "rig_tokudo.kind" => {
                self.captured.kind = Some(rendered.trim_matches('"').to_string());
            }
            _ => {}
        }
    }
}

impl<S: Subscriber> Layer<S> for CapturingLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let target = event.metadata().target().to_string();
        if !target.starts_with("rig_tokudo") {
            return;
        }
        let mut captured = CapturedEvent {
            event_json: None,
            kind: None,
        };
        let mut visitor = CaptureVisitor {
            captured: &mut captured,
        };
        event.record(&mut visitor);
        if let Ok(mut guard) = self.events.lock() {
            guard.push(captured);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn decorator_emits_miss_then_hit_with_cost_estimates() {
    let capture = CapturingLayer::default();
    let events = capture.events.clone();
    let subscriber = Registry::default().with(capture);

    let kinds: Vec<String> = with_default(subscriber, || {
        // The async block must stay inside `with_default` so that the
        // dispatcher is active for every `tracing::info!` emission.
        let model = OptimizedModel::builder(MockModel {
            reply: "hello".into(),
        })
        .with_cache(InMemoryCache::new())
        .build();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async move {
            // First call → cache.miss + route.decision + cost.estimate.
            let _ = model
                .complete(request("hi"), TokudoOptions::new())
                .await
                .unwrap();
            // Second identical call → cache.hit + cost.estimate(cache_hit=true).
            let _ = model
                .complete(request("hi"), TokudoOptions::new())
                .await
                .unwrap();
        });

        let guard = events.lock().unwrap();
        guard.iter().filter_map(|e| e.kind.clone()).collect()
    });

    assert!(
        kinds.iter().any(|k| k == "cache.miss"),
        "expected cache.miss in {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| k == "cache.hit"),
        "expected cache.hit in {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| k == "route.decision"),
        "expected route.decision in {kinds:?}"
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| k.as_str() == "cost.estimate")
            .count(),
        2,
        "expected exactly two cost.estimate events (miss + hit) in {kinds:?}"
    );
}

#[test]
fn cost_estimate_carries_provider_usage_on_miss() {
    let capture = CapturingLayer::default();
    let events = capture.events.clone();
    let subscriber = Registry::default().with(capture);

    let json_payloads: Vec<String> = with_default(subscriber, || {
        let model = OptimizedModel::builder(MockModel {
            reply: "hello".into(),
        })
        .with_cache(InMemoryCache::new())
        .build();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async move {
            let _ = model
                .complete(request("hi"), TokudoOptions::new())
                .await
                .unwrap();
        });

        let guard = events.lock().unwrap();
        guard
            .iter()
            .filter(|e| e.kind.as_deref() == Some("cost.estimate"))
            .filter_map(|e| e.event_json.clone())
            .collect()
    });

    assert_eq!(json_payloads.len(), 1, "expected one cost.estimate event");
    let parsed: serde_json::Value = serde_json::from_str(&json_payloads[0]).unwrap();
    assert_eq!(parsed["kind"], "cost.estimate");
    assert_eq!(parsed["cache_hit"], false);
    assert_eq!(parsed["input_tokens"], 7);
    assert_eq!(parsed["output_tokens"], 11);
    assert_eq!(parsed["total_tokens"], 18);

    #[cfg(feature = "model-catalog")]
    {
        let actual = parsed["usd_actual_estimate"].as_f64().unwrap();
        assert!((actual - 0.00000765).abs() < 1e-12);
    }
}

#[cfg(feature = "lineage")]
#[test]
fn lineage_event_links_provider_response_to_request_key() {
    let capture = CapturingLayer::default();
    let events = capture.events.clone();
    let subscriber = Registry::default().with(capture);

    let json_payloads: Vec<String> = with_default(subscriber, || {
        let model = OptimizedModel::builder(MockModel {
            reply: "hello".into(),
        })
        .with_cache(InMemoryCache::new())
        .build();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async move {
            let _ = model
                .complete(request("hi"), TokudoOptions::new())
                .await
                .unwrap();
        });

        let guard = events.lock().unwrap();
        guard
            .iter()
            .filter(|e| e.kind.as_deref() == Some("lineage.edge"))
            .filter_map(|e| e.event_json.clone())
            .collect()
    });

    assert_eq!(json_payloads.len(), 1, "expected one lineage.edge event");
    let parsed: serde_json::Value = serde_json::from_str(&json_payloads[0]).unwrap();
    assert_eq!(parsed["kind"], "lineage.edge");
    assert_eq!(parsed["relation"], "provider_call");
    assert_eq!(parsed["router_choice"], "pass_through");
    assert_eq!(parsed["model"], "openai:gpt-4o-mini");
    assert!(parsed["edge_id"].as_str().is_some());
    assert!(parsed["cache_key"].as_str().is_some());
}
