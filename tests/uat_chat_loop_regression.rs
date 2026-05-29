//! Comprehensive multi-turn UAT chat-loop coverage.
//!
//! Complements [`uat_agent_chat_loop`] (which focuses on the
//! compression + cascade-escalation + cache + report path) by exercising
//! a longer chat session against a single `OptimizedModel` and asserting
//! that:
//!
//! * the cheap leg is honored when the validator accepts (no needless
//!   escalation — regression for the cascade contract),
//! * exact-match cache hits skip the provider on repeated turns,
//! * `TokudoOptions` per-call overrides (`NoCache`, `NoStore`, `ForceFresh`,
//!   namespaces, `bypass_router`, `bypass_compress`) keep their documented
//!   semantics,
//! * cache namespaces isolate tenants,
//! * chat history growth invalidates the cache key for genuinely new turns,
//! * a fresh strong-only escalation path still works after the conversation
//!   has been driven through every override (regression).
//!
//! Kept feature-neutral so it compiles under every CI feature combo. Only
//! provenance flags and provider call counts are asserted — not USD
//! amounts, which depend on the optional `model-catalog` feature.

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
    OptimizedModel, RouterChoice, StaticCascade, TokudoOptions,
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
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
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
            message_id: Some(format!("{}-{call_index}", self.model_name)),
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
        "ticket_id": "TICK-99",
        "customer_tier": "enterprise",
        "events": [
            {"kind": "debug", "payload": "a".repeat(256)},
            {"kind": "debug", "payload": "b".repeat(256)},
            {"kind": "debug", "payload": "c".repeat(256)}
        ],
        "nested": {"x": {"y": {"z": {"deep": "too deep to keep"}}}}
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

/// Multi-turn loop exercising every documented `TokudoOptions` override and
/// the cheap-leg acceptance regression path.
#[tokio::test(flavor = "multi_thread")]
async fn multi_turn_chat_covers_per_call_overrides_and_namespaces() {
    // Cheap reply is long enough to clear LengthValidator(min=20), so the
    // cheap leg should answer on its own — the cascade must NOT escalate.
    let cheap = CountingModel::new("cheap", "Acknowledged: forwarding to the queue.");
    let strong = CountingModel::new("strong", "Detailed escalation answer for the team.");
    let cheap_probe = cheap.clone();
    let strong_probe = strong.clone();

    let cascade = StaticCascade::new(cheap, strong, LengthValidator::new(20));
    let agent = OptimizedModel::builder(cascade)
        .with_compressor(pruner())
        .with_cache(InMemoryCache::new())
        .build();

    let original_params_bytes = serde_json::to_string(&noisy_context()).unwrap().len();
    let mut chat = ChatSession::default();

    // Turn 1: cold cache. Cheap accepts → no escalation. Cache write happens.
    let prompt1 = "Status check on TICK-99?";
    let t1 = agent
        .complete(chat.request(prompt1, noisy_context()), TokudoOptions::new())
        .await
        .unwrap();
    assert!(!t1.provenance.cache_hit, "first turn should miss");
    assert_eq!(
        t1.provenance.router_choice,
        RouterChoice::Cheap,
        "cheap leg should answer when validator accepts"
    );
    assert!(
        t1.provenance.compressed_ratio.is_some(),
        "compressor should run by default"
    );
    assert!(
        t1.response.raw_response.observed_params_bytes < original_params_bytes,
        "compressed payload should be smaller than the raw context"
    );
    assert_eq!(cheap_probe.call_count(), 1);
    assert_eq!(
        strong_probe.call_count(),
        0,
        "strong leg must stay quiet when cheap is accepted"
    );
    chat.record_turn(prompt1, &t1.response.raw_response.text);

    // Turn 2: same surface prompt but history has grown → distinct cache
    // key, so we expect a miss. Cheap still accepts.
    let t2 = agent
        .complete(chat.request(prompt1, noisy_context()), TokudoOptions::new())
        .await
        .unwrap();
    assert!(
        !t2.provenance.cache_hit,
        "growing chat history must invalidate the cache key"
    );
    assert_eq!(t2.provenance.router_choice, RouterChoice::Cheap);
    assert_eq!(cheap_probe.call_count(), 2);
    chat.record_turn(prompt1, &t2.response.raw_response.text);

    // Snapshot history so subsequent "repeat" turns share the same key.
    let stable_history = chat.history.clone();
    let stable_request = |opts: TokudoOptions, prompt: &str, params: Value| -> CompletionRequest {
        let mut messages = stable_history.clone();
        messages.push(Message::user(prompt));
        let _ = opts; // options applied at agent.complete time
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
    };

    let repeat_prompt = "Confirm the ETA for TICK-99.";

    // Turn 3: prime the cache for `repeat_prompt`.
    let t3 = agent
        .complete(
            stable_request(TokudoOptions::new(), repeat_prompt, noisy_context()),
            TokudoOptions::new(),
        )
        .await
        .unwrap();
    assert!(!t3.provenance.cache_hit, "first occurrence should miss");
    assert_eq!(t3.provenance.router_choice, RouterChoice::Cheap);
    let cheap_after_prime = cheap_probe.call_count();
    assert_eq!(cheap_after_prime, 3);

    // Turn 4: identical prompt+history → cache hit. Provider must NOT be
    // called and provenance must reflect the cached path.
    let t4 = agent
        .complete(
            stable_request(TokudoOptions::new(), repeat_prompt, noisy_context()),
            TokudoOptions::new(),
        )
        .await
        .unwrap();
    assert!(t4.provenance.cache_hit, "second occurrence should hit");
    assert_eq!(t4.provenance.router_choice, RouterChoice::CacheHit);
    assert!(
        t4.provenance.source_id.is_some(),
        "cache hits must carry a source_id"
    );
    assert_eq!(t4.provenance.usd_actual_estimate, Some(0.0));
    assert_eq!(
        cheap_probe.call_count(),
        cheap_after_prime,
        "cache hit must skip the cheap leg"
    );

    // Turn 5: `NoCache` bypasses the read; provider is invoked again, then
    // the write still happens (so a subsequent default call hits).
    let t5 = agent
        .complete(
            stable_request(
                TokudoOptions::new().with_cache_policy(CachePolicy::NoCache),
                repeat_prompt,
                noisy_context(),
            ),
            TokudoOptions::new().with_cache_policy(CachePolicy::NoCache),
        )
        .await
        .unwrap();
    assert!(!t5.provenance.cache_hit, "NoCache must skip the cache read");
    assert_eq!(t5.provenance.router_choice, RouterChoice::Cheap);
    assert_eq!(cheap_probe.call_count(), cheap_after_prime + 1);
    let cheap_after_nocache = cheap_probe.call_count();

    // Turn 6: default options again → cache hit (NoCache still wrote to the
    // cache on turn 5).
    let t6 = agent
        .complete(
            stable_request(TokudoOptions::new(), repeat_prompt, noisy_context()),
            TokudoOptions::new(),
        )
        .await
        .unwrap();
    assert!(
        t6.provenance.cache_hit,
        "default read must hit after NoCache write"
    );
    assert_eq!(
        cheap_probe.call_count(),
        cheap_after_nocache,
        "hit after NoCache write must not call the provider"
    );

    // Turn 7: `NoStore` skips both read and write. Provider runs, but a
    // follow-up default turn must still see the *pre-NoStore* cached entry.
    let t7 = agent
        .complete(
            stable_request(
                TokudoOptions::new().with_cache_policy(CachePolicy::NoStore),
                repeat_prompt,
                noisy_context(),
            ),
            TokudoOptions::new().with_cache_policy(CachePolicy::NoStore),
        )
        .await
        .unwrap();
    assert!(!t7.provenance.cache_hit, "NoStore must skip the read");
    assert_eq!(cheap_probe.call_count(), cheap_after_nocache + 1);

    let t8 = agent
        .complete(
            stable_request(TokudoOptions::new(), repeat_prompt, noisy_context()),
            TokudoOptions::new(),
        )
        .await
        .unwrap();
    assert!(
        t8.provenance.cache_hit,
        "default read must still hit the prior write"
    );

    // Turn 9: `ForceFresh` skips the read but writes the new response. The
    // provider is invoked exactly once.
    let cheap_before_force = cheap_probe.call_count();
    let t9 = agent
        .complete(
            stable_request(
                TokudoOptions::new().with_cache_policy(CachePolicy::ForceFresh),
                repeat_prompt,
                noisy_context(),
            ),
            TokudoOptions::new().with_cache_policy(CachePolicy::ForceFresh),
        )
        .await
        .unwrap();
    assert!(!t9.provenance.cache_hit, "ForceFresh must skip the read");
    assert_eq!(cheap_probe.call_count(), cheap_before_force + 1);

    // Turn 10: namespace isolation — same prompt+history under a different
    // namespace must miss, then hit on a second namespace-scoped call.
    let cheap_before_ns = cheap_probe.call_count();
    let ns_opts = TokudoOptions::new().with_cache_namespace("tenant-b");
    let t10 = agent
        .complete(
            stable_request(ns_opts.clone(), repeat_prompt, noisy_context()),
            ns_opts.clone(),
        )
        .await
        .unwrap();
    assert!(
        !t10.provenance.cache_hit,
        "namespace switch must not see the default-namespace entry"
    );
    assert_eq!(cheap_probe.call_count(), cheap_before_ns + 1);

    let t11 = agent
        .complete(
            stable_request(ns_opts.clone(), repeat_prompt, noisy_context()),
            ns_opts,
        )
        .await
        .unwrap();
    assert!(
        t11.provenance.cache_hit,
        "second namespace-scoped call must hit"
    );
    assert_eq!(
        cheap_probe.call_count(),
        cheap_before_ns + 1,
        "namespace hit must skip the provider"
    );

    // Turn 12: `bypass_compress` keeps the raw payload intact end-to-end.
    let novel_prompt = "Draft a customer-facing note for TICK-99.";
    let bypass_opts = TokudoOptions::new().with_bypass_compress();
    let t12 = agent
        .complete(
            stable_request(bypass_opts.clone(), novel_prompt, noisy_context()),
            bypass_opts,
        )
        .await
        .unwrap();
    assert!(
        t12.provenance.compressed_ratio.is_none(),
        "bypass_compress disables compression"
    );
    assert_eq!(
        t12.response.raw_response.observed_params_bytes, original_params_bytes,
        "raw payload should reach the provider unchanged"
    );

    // Turn 13: `bypass_router` skips the router hint; the cascade still
    // runs and the cheap leg still answers (regression: bypass does not
    // suppress the inner dispatcher).
    let bypass_router_opts = TokudoOptions::new().with_bypass_router();
    let t13 = agent
        .complete(
            stable_request(
                bypass_router_opts.clone(),
                "Anything new on TICK-99?",
                noisy_context(),
            ),
            bypass_router_opts,
        )
        .await
        .unwrap();
    assert!(!t13.provenance.cache_hit);
    // Dispatcher reports its own RouterChoice; cascade reports Cheap here.
    assert_eq!(t13.provenance.router_choice, RouterChoice::Cheap);

    // Strong leg must have stayed silent for the entire chat.
    assert_eq!(
        strong_probe.call_count(),
        0,
        "strong leg should never be invoked when cheap clears the validator"
    );
}

/// Regression: when the cheap leg fails validation the cascade must still
/// escalate to the strong leg, and the existing cache + provenance contract
/// must continue to hold.
#[tokio::test(flavor = "multi_thread")]
async fn cascade_escalation_still_works_after_overrides() {
    // Cheap reply is too short to clear LengthValidator(min=20).
    let cheap = CountingModel::new("cheap", "no");
    let strong = CountingModel::new(
        "strong",
        "Escalated answer with sufficient length to satisfy validation.",
    );
    let cheap_probe = cheap.clone();
    let strong_probe = strong.clone();

    let cascade = StaticCascade::new(cheap, strong, LengthValidator::new(20));
    let agent = OptimizedModel::builder(cascade)
        .with_compressor(pruner())
        .with_cache(InMemoryCache::new())
        .build();

    let mut chat = ChatSession::default();
    let prompt = "Need detailed remediation steps for TICK-99.";

    // First turn escalates cheap → strong.
    let first = agent
        .complete(chat.request(prompt, noisy_context()), TokudoOptions::new())
        .await
        .unwrap();
    assert!(!first.provenance.cache_hit);
    assert_eq!(first.provenance.router_choice, RouterChoice::Strong);
    assert_eq!(cheap_probe.call_count(), 1);
    assert_eq!(strong_probe.call_count(), 1);
    chat.record_turn(prompt, &first.response.raw_response.text);

    // Snapshot history after turn 1 so the next two turns share an identical
    // cache key. The first repeat primes the cache (miss, escalates again);
    // the second repeat hits.
    let history_after_first = chat.history.clone();
    let repeat_request = || -> CompletionRequest {
        let mut messages = history_after_first.clone();
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
            additional_params: Some(noisy_context()),
            output_schema: None,
        }
    };

    let second = agent
        .complete(repeat_request(), TokudoOptions::new())
        .await
        .unwrap();
    assert!(
        !second.provenance.cache_hit,
        "history grew → new key, must miss"
    );
    assert_eq!(second.provenance.router_choice, RouterChoice::Strong);
    assert_eq!(cheap_probe.call_count(), 2);
    assert_eq!(strong_probe.call_count(), 2);

    let third = agent
        .complete(repeat_request(), TokudoOptions::new())
        .await
        .unwrap();
    assert!(third.provenance.cache_hit, "identical repeat must hit");
    assert_eq!(third.provenance.router_choice, RouterChoice::CacheHit);
    assert_eq!(cheap_probe.call_count(), 2);
    assert_eq!(strong_probe.call_count(), 2);

    // After a `NoStore` call neither leg's count budget for hits applies;
    // both must run again to satisfy the request, and the existing entry
    // is left in place.
    let fourth = agent
        .complete(
            repeat_request(),
            TokudoOptions::new().with_cache_policy(CachePolicy::NoStore),
        )
        .await
        .unwrap();
    assert!(!fourth.provenance.cache_hit);
    assert_eq!(fourth.provenance.router_choice, RouterChoice::Strong);
    assert_eq!(cheap_probe.call_count(), 3);
    assert_eq!(strong_probe.call_count(), 3);

    // Default options after NoStore must still find the prior cached entry.
    let fifth = agent
        .complete(repeat_request(), TokudoOptions::new())
        .await
        .unwrap();
    assert!(
        fifth.provenance.cache_hit,
        "NoStore must not evict prior entry"
    );
    assert_eq!(cheap_probe.call_count(), 3);
    assert_eq!(strong_probe.call_count(), 3);
}
