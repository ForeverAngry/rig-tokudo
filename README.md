# rig-tokudo

[![Crates.io](https://img.shields.io/crates/v/rig-tokudo.svg)](https://crates.io/crates/rig-tokudo)
[![Docs.rs](https://docs.rs/rig-tokudo/badge.svg)](https://docs.rs/rig-tokudo)

Cost-optimization decorator for [Rig](https://crates.io/crates/rig-core)
`CompletionModel`s. Combines an approximate cache (with quality controls),
prompt compression, cascade routing (cheap → strong), and provenance/observability.

> **Status: v0.1.0.** Exact-match cache, semantic cache
> adapter, JSON compression, static cascade routing inside `OptimizedModel`,
> tokudo observability, `rig-model-catalog` pricing, and ROUGE-L quality scoring
> are wired. See [Roadmap](#roadmap).

## Why

Decorating any `CompletionModel` with `OptimizedModel` lets a host:

1. **Cache** repeat / near-duplicate prompts to skip the provider call.
2. **Cascade**: try a cheap model first, validate, fall back to a strong one
   only when needed (FrugalGPT-style).
3. **Compress** verbose prompts (JSON pruning by default; LLMLingua-style pruning behind a feature).
4. **Observe** every decision: hit/miss, router choice, USD saved via
   `rig-model-catalog` pricing.

## Quick start

```rust,no_run
use rig_tokudo::{OptimizedModel, TokudoOptions};

// Wrap any rig CompletionModel.
// Defaults emit tokudo/tap-compatible telemetry and fill USD estimates when
// the request model resolves through rig-model-catalog's built-in pricing table.
# fn demo<M: rig::completion::CompletionModel>(model: M) {
let _wrapped = OptimizedModel::builder(model).build();
# }
```

## Feature flags

| Flag                  | Status   | Purpose                                                               |
| --------------------- | -------- | --------------------------------------------------------------------- |
| `tap`                 | Default  | Emit `rig_tap.*`-aligned scalar fields on tokudo events.              |
| `model-catalog`       | Default  | USD pricing from `rig-model-catalog`, including provider cache-token deltas. |
| `cache-semantic`      | Shipped  | Read-through cache over any `VectorStoreIndexDyn`.                    |
| `cache-memvid`        | Shipped  | Durable semantic-cache backend backed by `rig-memvid`.                |
| `compress-llmlingua`  | Optional | Dependency-free LLMLingua-style prompt token pruning.                 |
| `route-predictive`    | Optional | Train-free calibration-set router for cheap-vs-strong hints.          |
| `lineage`             | Optional | Per-call response lineage edges for cache/provider audit trails.      |
| `eval`                | Shipped  | ROUGE-L helpers and replay bridge into `rig-retrieval-evals`.         |

## Roadmap

- **Phase 0–1**: bootstrap + core traits / `OptimizedModel` shell — **shipped**.
- **Phase 2**: cache pillar (`Cache`, `CacheKey`, `InMemoryCache`) — **shipped**.
- **Phase 3**: compression (`JsonKeyPruner`) — **shipped**.
- **Phase 4**: routing (`StaticCascade`, validators) and `OptimizedModel` wiring — **shipped**.
- **Phase 5**: observability + default-on `tap` / `model-catalog` pricing, with provider-side cache-token deltas kept separate from Tokudo savings — **shipped**.
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
`Report::write_artifacts` and `ReplayReport::write_artifacts` persist
CI-friendly `report.json`, `report.md`, `metrics.*`, and `manifest.json`
outputs with stable filenames.

## License

MIT OR Apache-2.0.
