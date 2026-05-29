//! UAT-style agent chat-loop coverage for the public Tokudo value path.
//!
//! The lower-level integration tests cover cache, cascade, compression, and
//! observability independently. This test runs a conversation-shaped loop over
//! an `OptimizedModel` that combines those pillars so the documented use case
//! stays exercised end to end.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::indexing_slicing
)]

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::OneOrMany;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Message, Usage,
};
use rig::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use rig_tokudo::{
    CachePolicy, InMemoryCache, JsonKeyPruner, JsonKeyPrunerConfig, LengthValidator,
    OptimizedModel, Report, RouterChoice, RunStats, StaticCascade, TokudoOptions,
    estimate_actual_usd,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct UatRaw {
    text: String,
    observed_params_bytes: usize,
}

#[derive(Clone)]
struct CountingModel {
    calls: Arc<AtomicUsize>,
    reply: String,
    model_name: &'static str,
}

impl CountingModel {
    fn new(model_name: &'static str, reply: impl Into<String>) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            reply: reply.into(),
            model_name,
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl CompletionModel for CountingModel {
    type Response = UatRaw;
    type StreamingResponse = ();
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        CountingModel::new("openai:gpt-4o-mini", "mock")
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let observed_params_bytes = request
            .additional_params
            .as_ref()
            .and_then(|value| serde_json::to_string(value).ok())
            .map(|rendered| rendered.len())
            .unwrap_or(0);
        let text = self.reply.clone();
        let input_tokens = u64::try_from(observed_params_bytes / 4).unwrap_or(u64::MAX);
        let output_tokens = u64::try_from(text.len() / 4).unwrap_or(u64::MAX);

        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(text.clone())),
            usage: Usage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens.saturating_add(output_tokens),
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_tokens: 0,
            },
            raw_response: UatRaw {
                text,
                observed_params_bytes,
            },
            message_id: Some(format!("{}-{}", self.model_name, self.call_count())),
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Err(CompletionError::ProviderError(
            "stream not supported in UAT mock".into(),
        ))
    }
}

#[derive(Default)]
struct ChatSession {
    history: Vec<Message>,
}

impl ChatSession {
    fn request(&self, prompt: &str, params: Value) -> CompletionRequest {
        let mut messages = self.history.clone();
        messages.push(Message::user(prompt));
        CompletionRequest {
            model: Some("openai:gpt-4o-mini".into()),
            preamble: Some("You are a cost-aware support agent.".into()),
            chat_history: OneOrMany::many(messages).unwrap(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: Some(0.2),
            max_tokens: Some(256),
            tool_choice: None,
            additional_params: Some(params),
            output_schema: None,
        }
    }

    fn record_turn(&mut self, prompt: &str, answer: &str) {
        self.history.push(Message::user(prompt));
        self.history.push(Message::assistant(answer));
    }
}

fn noisy_context() -> Value {
    json!({
        "ticket_id": "TICK-42",
        "customer_tier": "enterprise",
        "events": [
            {"kind": "debug", "payload": "x".repeat(256)},
            {"kind": "debug", "payload": "y".repeat(256)},
            {"kind": "debug", "payload": "z".repeat(256)},
            {"kind": "debug", "payload": "w".repeat(256)}
        ],
        "nested": {"a": {"b": {"c": {"d": "too deep to keep"}}}}
    })
}

fn pruner() -> JsonKeyPruner {
    JsonKeyPruner::new(JsonKeyPrunerConfig {
        preserve_keys: HashSet::from(["ticket_id".to_string(), "customer_tier".to_string()]),
        max_depth: 2,
        max_array_len: 1,
        max_string_len: 32,
    })
}

fn add_provider_stats(
    stats: &mut RunStats,
    normalized: &rig_tokudo::NormalizedResponse<CompletionResponse<UatRaw>>,
) {
    stats.requests = stats.requests.saturating_add(1);
    if normalized.provenance.cache_hit {
        stats.cache_hits = stats.cache_hits.saturating_add(1);
        return;
    }

    let usage = normalized.response.usage;
    stats.prompt_tokens = stats.prompt_tokens.saturating_add(usage.input_tokens);
    stats.completion_tokens = stats.completion_tokens.saturating_add(usage.output_tokens);
    stats.usd += estimate_actual_usd(Some("openai:gpt-4o-mini"), &usage).unwrap_or(0.0);
    match normalized.provenance.router_choice {
        RouterChoice::Cheap => stats.cheap_calls = stats.cheap_calls.saturating_add(1),
        RouterChoice::Strong | RouterChoice::PassThrough => {
            stats.strong_calls = stats.strong_calls.saturating_add(1);
        }
        RouterChoice::CacheHit => {}
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn repeated_support_chat_uses_compression_cascade_cache_and_report_provenance() {
    let cheap = CountingModel::new("cheap", "no");
    let strong = CountingModel::new(
        "strong",
        "The maintenance window is approved; notify the enterprise customer.",
    );
    let cheap_probe = cheap.clone();
    let strong_probe = strong.clone();
    let cascade = StaticCascade::new(cheap, strong, LengthValidator::new(24));
    let agent_model = OptimizedModel::builder(cascade)
        .with_compressor(pruner())
        .with_cache(InMemoryCache::new())
        .build();

    let original_params_bytes = serde_json::to_string(&noisy_context()).unwrap().len();
    let mut tokudo_stats = RunStats::default();

    let mut first_session = ChatSession::default();
    let first = agent_model
        .complete(
            first_session.request(
                "Summarize ticket TICK-42 for the next agent.",
                noisy_context(),
            ),
            TokudoOptions::new(),
        )
        .await
        .unwrap();
    first_session.record_turn(
        "Summarize ticket TICK-42 for the next agent.",
        &first.response.raw_response.text,
    );
    add_provider_stats(&mut tokudo_stats, &first);

    assert!(!first.provenance.cache_hit);
    assert_eq!(first.provenance.router_choice, RouterChoice::Strong);
    assert!(first.provenance.compressed_ratio.is_some());
    assert!(first.response.raw_response.observed_params_bytes < original_params_bytes);
    assert_eq!(
        cheap_probe.call_count(),
        1,
        "cheap leg should be tried first"
    );
    assert_eq!(
        strong_probe.call_count(),
        1,
        "strong leg should answer after cheap rejection"
    );

    let mut second_session = ChatSession::default();
    let second = agent_model
        .complete(
            second_session.request(
                "Summarize ticket TICK-42 for the next agent.",
                noisy_context(),
            ),
            TokudoOptions::new(),
        )
        .await
        .unwrap();
    second_session.record_turn(
        "Summarize ticket TICK-42 for the next agent.",
        &second.response.raw_response.text,
    );
    add_provider_stats(&mut tokudo_stats, &second);

    assert!(second.provenance.cache_hit);
    assert_eq!(second.provenance.router_choice, RouterChoice::CacheHit);
    assert_eq!(second.provenance.usd_actual_estimate, Some(0.0));
    assert_eq!(
        cheap_probe.call_count(),
        1,
        "cache hit should skip cheap leg"
    );
    assert_eq!(
        strong_probe.call_count(),
        1,
        "cache hit should skip strong leg"
    );

    let mut tool_session = ChatSession::default();
    let tool_result = agent_model
        .complete(
            tool_session.request(
                "Run a one-off account mutation for TICK-42.",
                noisy_context(),
            ),
            TokudoOptions::new().with_cache_policy(CachePolicy::NoStore),
        )
        .await
        .unwrap();
    tool_session.record_turn(
        "Run a one-off account mutation for TICK-42.",
        &tool_result.response.raw_response.text,
    );
    add_provider_stats(&mut tokudo_stats, &tool_result);

    assert!(!tool_result.provenance.cache_hit);
    assert_eq!(tool_result.provenance.router_choice, RouterChoice::Strong);
    assert_eq!(cheap_probe.call_count(), 2);
    assert_eq!(strong_probe.call_count(), 2);

    let baseline = RunStats {
        requests: 3,
        strong_calls: 3,
        usd: tokudo_stats.usd * 3.0,
        ..RunStats::default()
    };
    let report = Report::from_runs(baseline, tokudo_stats, None);

    assert_eq!(report.tokudo.requests, 3);
    assert_eq!(report.tokudo.cache_hits, 1);
    assert!(report.deltas.cache_hit_rate > 0.30);
    assert!(report.deltas.usd_saved > 0.0);
    assert_eq!(first_session.history.len(), 2);
    assert_eq!(second_session.history.len(), 2);
    assert_eq!(tool_session.history.len(), 2);
}
