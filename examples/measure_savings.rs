//! Phase 6.F headline example: run a workload twice (bare baseline, then
//! tokudo-wrapped) and emit a `report.json` + `report.md` quantifying USD
//! savings, cache-hit rate, and cheap-model share.
//!
//! Phase 6 ships against a built-in `MockProvider` so the example runs
//! anywhere with no API keys, env vars, or local model server. USD estimates
//! come from a `CatalogCostModel` defined below, which prices calls using
//! `rig-model-catalog`'s bundled pricing table and is installed on the
//! optimizer via `with_cost_model`. The library owns no pricing data; swap in
//! any `CostModel` to price against your own rates or billing API. Real-provider
//! wiring (Ollama / OpenAI / Anthropic) is a follow-up: swap the `MockProvider`
//! for any type implementing `rig::completion::CompletionModel` and the rest of
//! the pipeline keeps working.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example measure_savings
//! ```
//!
//! Output artifacts are written under `./report/`. The process exits with a
//! non-zero status code if any default Phase 6.G threshold is missed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rig::OneOrMany;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Message, Usage,
};
use rig::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};

use rig_tokudo::cost::{CostBreakdown, CostModel};
use rig_tokudo::{
    CachePolicy, InMemoryCache, OptimizedModel, Report, RunStats, Thresholds, TokudoOptions,
};

use rig_model_catalog::{ModelPrice, PricingTable, ProviderId};

// ---------------------------------------------------------------------------
// Mock provider
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MockRaw {
    text: String,
}

/// A deterministic mock completion model. Token usage is derived from the
/// last user message; latency is simulated only at the report level.
#[derive(Clone)]
struct MockProvider {
    model: &'static str,
    /// Per-call simulated latency in milliseconds.
    latency_ms: u64,
}

impl CompletionModel for MockProvider {
    type Response = MockRaw;
    type StreamingResponse = ();
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        MockProvider {
            model: "openai:gpt-4o-mini",
            latency_ms: 50,
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        // Token counts are a deterministic char-count proxy.
        let prompt_chars: usize = serde_json::to_string(&request.chat_history)
            .map(|s| s.len())
            .unwrap_or(0);
        let prompt_tokens = (prompt_chars / 4) as u64;
        let reply = format!(
            "[{}] simulated answer ({} prompt chars)",
            self.model, prompt_chars
        );
        let output_tokens = (reply.len() / 4) as u64;

        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(reply.clone())),
            usage: Usage {
                input_tokens: prompt_tokens,
                output_tokens,
                total_tokens: prompt_tokens.saturating_add(output_tokens),
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_tokens: 0,
            },
            raw_response: MockRaw { text: reply },
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

// ---------------------------------------------------------------------------
// Catalog-backed CostModel
// ---------------------------------------------------------------------------

/// A `CostModel` that prices calls from `rig-model-catalog`'s bundled pricing
/// table. This lives in the example — the library itself owns no pricing data;
/// hosts plug in whatever pricing source they like via `with_cost_model`.
#[derive(Clone)]
struct CatalogCostModel {
    table: PricingTable,
}

impl CatalogCostModel {
    fn builtin() -> Self {
        Self {
            table: PricingTable::builtin(),
        }
    }

    /// Resolve a price row from an arbitrary model id (`openai:gpt-4o-mini`,
    /// `openai/gpt-4o-mini`, or provider-local `gpt-4o-mini`).
    fn resolve(&self, model_id: &str) -> Option<ModelPrice> {
        let trimmed = model_id.trim();
        if trimmed.is_empty() {
            return None;
        }
        for delimiter in [':', '/'] {
            if let Some((provider, model)) = trimmed.split_once(delimiter)
                && !provider.is_empty()
                && !model.is_empty()
                && let Some(price) = self.table.lookup(provider, model)
            {
                let _ = ProviderId::new(provider);
                return Some(price.clone());
            }
        }
        // Provider-local id: resolve only when exactly one provider prices it.
        let mut found: Option<ModelPrice> = None;
        for (_provider, model, price) in self.table.iter() {
            if model != trimmed {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(price.clone());
        }
        found
    }
}

impl CostModel for CatalogCostModel {
    fn estimate(&self, model: Option<&str>, usage: &Usage) -> Option<CostBreakdown> {
        let price = self.resolve(model?)?;
        let usd_actual = price.cost_for(
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_input_tokens,
            usage.cache_creation_input_tokens,
        );
        let input_rate = price.input_per_million;
        let cached_rate = price.cached_input_per_million.unwrap_or(input_rate);
        let cache_write_rate = price.cache_write_per_million.unwrap_or(input_rate);
        let read_delta = usage.cached_input_tokens as f64 * (input_rate - cached_rate);
        let write_delta =
            usage.cache_creation_input_tokens as f64 * (input_rate - cache_write_rate);
        Some(CostBreakdown {
            usd_actual,
            provider_cache_usd_delta: (read_delta + write_delta) / 1_000_000.0,
        })
    }
}

fn estimate_usd(cost: &CatalogCostModel, provider: &MockProvider, usage: &Usage) -> f64 {
    cost.estimate(Some(provider.model), usage)
        .map(|b| b.usd_actual)
        .unwrap_or(0.0)
}

fn estimate_provider_cache_delta(
    cost: &CatalogCostModel,
    provider: &MockProvider,
    usage: &Usage,
) -> f64 {
    cost.estimate(Some(provider.model), usage)
        .map(|b| b.provider_cache_usd_delta)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Workload
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WorkloadItem {
    #[allow(dead_code)]
    id: String,
    tags: Vec<String>,
    prompt: String,
}

impl WorkloadItem {
    fn is_tool_call(&self) -> bool {
        self.tags.iter().any(|t| t == "tool_call")
    }
}

fn workload_path() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("data").join("workload.jsonl")
}

fn load_workload(path: &Path) -> Vec<WorkloadItem> {
    let raw = fs::read_to_string(path).expect("read workload.jsonl");
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<WorkloadItem>(l).expect("parse workload line"))
        .collect()
}

fn build_request(model: &str, prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: Some(model.into()),
        preamble: Some("You are a helpful assistant.".into()),
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
// Runs
// ---------------------------------------------------------------------------

async fn run_baseline(
    cost: &CatalogCostModel,
    provider: &MockProvider,
    items: &[WorkloadItem],
) -> RunStats {
    let mut stats = RunStats::default();
    let start = Instant::now();
    for item in items {
        let resp = provider
            .completion(build_request(provider.model, &item.prompt))
            .await
            .expect("baseline completion");
        stats.requests += 1;
        stats.prompt_tokens += resp.usage.input_tokens;
        stats.completion_tokens += resp.usage.output_tokens;
        stats.usd += estimate_usd(cost, provider, &resp.usage);
        stats.provider_cached_input_tokens += resp.usage.cached_input_tokens;
        stats.provider_cache_write_tokens += resp.usage.cache_creation_input_tokens;
        stats.provider_cache_usd_delta +=
            estimate_provider_cache_delta(cost, provider, &resp.usage);
        stats.strong_calls += 1;
        // Simulated provider latency.
        stats.wall_ms = stats.wall_ms.saturating_add(provider.latency_ms);
    }
    let _elapsed = start.elapsed();
    stats
}

async fn run_tokudo(
    cost: CatalogCostModel,
    provider: MockProvider,
    items: &[WorkloadItem],
) -> RunStats {
    let model = OptimizedModel::builder(provider.clone())
        .with_cache(InMemoryCache::new())
        .with_cost_model(cost.clone())
        .build();
    let mut stats = RunStats::default();
    for item in items {
        let options = if item.is_tool_call() {
            TokudoOptions::new().with_cache_policy(CachePolicy::NoStore)
        } else {
            TokudoOptions::new()
        };
        let normalized = model
            .complete(build_request(provider.model, &item.prompt), options)
            .await
            .expect("tokudo completion");
        stats.requests += 1;
        if normalized.provenance.cache_hit {
            stats.cache_hits += 1;
            // Cache hits skip the provider; treat latency as ~free.
            stats.wall_ms = stats.wall_ms.saturating_add(1);
        } else {
            let usage = normalized.response.usage;
            stats.prompt_tokens += usage.input_tokens;
            stats.completion_tokens += usage.output_tokens;
            stats.usd += estimate_usd(&cost, &provider, &usage);
            stats.provider_cached_input_tokens += usage.cached_input_tokens;
            stats.provider_cache_write_tokens += usage.cache_creation_input_tokens;
            stats.provider_cache_usd_delta +=
                estimate_provider_cache_delta(&cost, &provider, &usage);
            stats.strong_calls += 1;
            stats.wall_ms = stats.wall_ms.saturating_add(provider.latency_ms);
        }
    }
    stats
}

// Tool-call rows must never have hit the cache.
fn assert_tool_call_bypass(items: &[WorkloadItem], tokudo: &RunStats) {
    // The mock provider's cache uses the prompt text as part of the key;
    // since `NoStore` skips reads, tool-call rows cannot contribute to
    // `cache_hits`. Sanity check: total cache_hits cannot exceed the number
    // of non-tool-call repeats.
    let non_tool: u64 = items.iter().filter(|i| !i.is_tool_call()).count() as u64;
    assert!(
        tokudo.cache_hits <= non_tool,
        "tool-call bypass violated: {} hits across {} non-tool rows",
        tokudo.cache_hits,
        non_tool,
    );
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::try_init().ok();

    // Same priced provider for both passes — the report quantifies decorator
    // behavior, not provider differences.
    let provider = MockProvider {
        model: "openai:gpt-4o-mini",
        latency_ms: 50,
    };

    let items = load_workload(&workload_path());
    println!("workload: {} items", items.len());

    let cost = CatalogCostModel::builtin();
    let baseline = run_baseline(&cost, &provider, &items).await;
    let tokudo = run_tokudo(cost.clone(), provider.clone(), &items).await;
    assert_tool_call_bypass(&items, &tokudo);

    let report = Report::from_runs(baseline, tokudo, None);

    let report_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("report");
    report.write_artifacts(&report_dir)?;

    println!("\n{}", report.to_markdown());

    // Phase 6.G success-criteria gate. The mock workload is engineered to
    // surface the cache pillar; cascade thresholds stay at zero until this
    // example grows a cheap/strong pair.
    let thresholds = Thresholds {
        min_usd_saved_pct: 0.20,
        min_cache_hit_rate: 0.25,
        min_cheap_model_share: 0.0, // cascade not wired into OptimizedModel yet
        min_quality_score: None,
    };
    let outcome = report.evaluate(&thresholds);
    println!("\nevaluation: {outcome:?}");
    if !outcome.all_passed() {
        std::process::exit(1);
    }
    Ok(())
}
