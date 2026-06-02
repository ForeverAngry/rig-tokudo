# rig-tokudo

[![Crates.io](https://img.shields.io/crates/v/rig-tokudo.svg)](https://crates.io/crates/rig-tokudo)
[![Docs.rs](https://docs.rs/rig-tokudo/badge.svg)](https://docs.rs/rig-tokudo)

Cost-optimization decorator for [Rig](https://crates.io/crates/rig-core)
`CompletionModel`s. Combines an approximate cache (with quality controls),
prompt compression, cascade routing (cheap → strong), and provenance/observability.

> **Status: v0.2.4.** Exact-match cache, semantic cache
> adapter, JSON compression, static cascade routing inside `OptimizedModel`,
> tokudo observability, pluggable USD cost estimation, and ROUGE-L quality
> scoring are wired. A multi-turn UAT regression test
> ([`tests/uat_chat_loop_regression.rs`](tests/uat_chat_loop_regression.rs))
> exercises the cache / cascade / compression matrix end-to-end against a
> recording fake `CompletionModel`. See [Roadmap](#roadmap).
>
> **Pricing note:** tokudo owns no pricing data. Supply USD estimates by
> implementing the [`CostModel`](src/cost.rs) trait and installing it with
> `OptimizedModel::builder(model).with_cost_model(..)`, or hand a per-call
> literal through `TokudoOptions::with_cost_estimate(..)`. Without either,
> events and provenance carry token counts only and leave USD fields `None`.
> The [`measure_savings`](examples/measure_savings.rs) example shows a
> `CostModel` backed by `rig-model-catalog`'s bundled `PricingTable`.

## Why

Decorating any `CompletionModel` with `OptimizedModel` lets a host:

1. **Cache** repeat / near-duplicate prompts to skip the provider call.
2. **Cascade**: try a cheap model first, validate, fall back to a strong one
   only when needed (FrugalGPT-style).
3. **Compress** verbose prompts (JSON pruning by default; LLMLingua-style pruning behind a feature).
4. **Observe** every decision: hit/miss, router choice, and USD saved from a
   host-supplied `CostModel`.

Tokudo is valuable when the expensive part of a workflow is repeated or
overpowered completion calls. It sits directly around the model call and makes
per-request tradeoffs: reuse a previous answer, shrink the request, try a
cheaper model first, or record enough provenance to prove what happened.

It is not an agent orchestrator, memory store, model catalog, observability
backend, or retrieval evaluator. Those jobs stay in the companion crates that
own them. Tokudo uses those crates at the edges: a host `CostModel` supplies
pricing (e.g. backed by `rig-model-catalog`), `rig-tap` receives telemetry,
`rig-memvid` can back a durable semantic cache, and `rig-retrieval-evals` can
consume replay rows for measurement.

## Use Cases

| Use case | Tokudo value | Not the right fit when |
| --- | --- | --- |
| High-volume support, extraction, enrichment, or routing calls | Cache repeat prompts and route easy requests to a cheaper model while keeping a strong fallback. | Every request is unique, high-stakes, and already requires the strongest model. |
| JSON-heavy tool or API prompts | Prune structural noise before dispatch without changing the host's `CompletionModel` integration. | You need semantic summarization or task planning before the prompt is built. |
| Cost and latency experiments | Compare baseline vs wrapped runs with savings, cache-hit rate, cheap-model share, quality score, and stable report artifacts. | You need a full benchmark harness for retriever quality or dataset management. |
| Auditable optimization rollouts | Attach provenance, lineage edges, and tap-compatible telemetry to explain cache hits, provider calls, and router decisions. | You need long-term trace storage, dashboards, or alerting; Tokudo only emits the events. |
| Durable semantic reuse | Store semantic cache entries in `.mv2` through the optional `cache-memvid` feature. | You need a general-purpose agent memory system; use `rig-memvid` directly for that. |

## Quick start

```rust,no_run
use rig_tokudo::{OptimizedModel, TokudoOptions};

// Wrap any rig CompletionModel.
// Defaults emit tokudo/tap-compatible telemetry. USD estimates stay `None`
// until you install a `CostModel` via `.with_cost_model(..)` or pass a
// per-call literal through `TokudoOptions::with_cost_estimate(..)`.
# fn demo<M: rig::completion::CompletionModel>(model: M) {
let _wrapped = OptimizedModel::builder(model).build();
# }
```

Use the optional Foyer backend when exact-match cache traffic needs a bounded,
sharded in-memory cache rather than the simple default test cache:

```rust,no_run
use rig_tokudo::{FoyerCache, FoyerCacheConfig, OptimizedModel};

# fn demo<M: rig::completion::CompletionModel>(model: M) {
let cache = FoyerCache::with_config(FoyerCacheConfig {
   capacity: 10_000,
   shards: 16,
   name: "support-cache".into(),
});
let _wrapped = OptimizedModel::builder(model).with_cache(cache).build();
# }
```

Enable it with `--features cache-foyer`.

## Feature flags

| Flag                  | Status   | Purpose                                                               |
| --------------------- | -------- | --------------------------------------------------------------------- |
| `tap`                 | Default  | Emit `rig_tap.*`-aligned scalar fields on tokudo events.              |
| `cache-semantic`      | Shipped  | Read-through cache over any `VectorStoreIndexDyn`.                    |
| `cache-memvid`        | Shipped  | Durable semantic-cache backend backed by `rig-memvid`.                |
| `cache-foyer`         | Optional | Exact-match cache backed by Foyer's in-memory cache.                  |
| `compress-llmlingua`  | Optional | Dependency-free LLMLingua-style prompt token pruning.                 |
| `route-predictive`    | Optional | Train-free calibration-set router for cheap-vs-strong hints.          |
| `lineage`             | Optional | Per-call response lineage edges for cache/provider audit trails.      |
| `eval`                | Shipped  | ROUGE-L helpers and replay bridge into `rig-retrieval-evals`.         |

## Roadmap

- **Phase 0–1**: bootstrap + core traits / `OptimizedModel` shell — **shipped**.
- **Phase 2**: cache pillar (`Cache`, `CacheKey`, `InMemoryCache`) — **shipped**.
- **Phase 3**: compression (`JsonKeyPruner`) — **shipped**.
- **Phase 4**: routing (`StaticCascade`, validators) and `OptimizedModel` wiring — **shipped**.
- **Phase 5**: observability + default-on `tap`, with a pluggable `CostModel` for USD estimates and provider-side cache-token deltas kept separate from Tokudo savings — **shipped**.
- **Phase 6**: measurement — savings reports, semantic cache, ROUGE-L quality scoring, and `measure_savings` example — **shipped**.
- **Phase 6.1**: optional `lineage` telemetry edges link served responses to cache entries or provider calls — **shipped**.
- **Phase 6.2**: `eval` replay bridge emits `rig-retrieval-evals` metric reports from recorded Tokudo provenance — **shipped**.
- **Phase 6.4**: report artifact persistence writes stable JSON/Markdown/manifest layouts for savings and replay reports — **shipped**.

Predictive routing is train-free today: hosts provide calibration examples
through `PredictiveRouter`, and the router chooses the closest cheap/strong
route by lexical overlap. Learned routers remain out of the default build.
Lineage telemetry is opt-in: enable `lineage` to attach a compact edge to
`Provenance` and emit a `lineage.edge` event for each served response.
The `eval` replay bridge consumes recorded baseline/Tokudo rows and produces
native `rig-retrieval-evals::MetricReport` rows without re-running providers.
`cache-memvid` stores serialized cached completions in `.mv2` files while
indexing the semantic request projection as a searchable frame prefix.
`cache-foyer` provides a higher-concurrency exact-match backend for repeated
request keys; it is opt-in because Foyer's published memory crate brings its
runtime adapter only when the feature is enabled.
`Report::write_artifacts` and `ReplayReport::write_artifacts` persist
CI-friendly `report.json`, `report.md`, `metrics.*`, and `manifest.json`
outputs with stable filenames.

## License

MIT OR Apache-2.0.
