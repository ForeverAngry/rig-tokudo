# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Gate the `measure_savings` example on `model-catalog` and qualify strict
  rustdoc links so no-default CI builds only compatible targets.

## [0.2.1](https://github.com/ForeverAngry/rig-tokudo/compare/v0.2.0...v0.2.1) - 2026-05-28

### Documentation

- Sync README, ROADMAP, and AGENTS.md to v0.2.0 ([#2](https://github.com/ForeverAngry/rig-tokudo/pull/2))

## [0.2.0](https://github.com/ForeverAngry/rig-tokudo/compare/v0.1.0...v0.2.0) - 2026-05-28

### CI

- Make tokudo release workflow publish-safe
- Add tokudo release workflows

### Added

- Phase 0: repository bootstrap — `Cargo.toml`, AGENTS / Copilot conventions,
  `justfile`, README skeleton, license placeholders, lint configuration
  (clippy deny/forbid on `unwrap`, `panic`, indexing, etc.).
- Phase 1: core traits and types:
  - `OptimizedModel<M>` decorator shell with a builder.
  - `TokudoError` typed error enum.
  - `Provenance` and `NormalizedResponse` wrapper.
  - `TokudoOptions` per-request control sidecar (`CachePolicy`, namespace,
    router bypass).
  - `Cache`, `CacheKey`, `Compressor`, `Router`, `Validator` traits.
  - `NoCache`, `IdentityCacheKey`, `NoCompressor`, `PassThroughRouter`,
    and `AcceptAllValidator` reference implementations.
- Feature-flag scaffolding for `tap`, `model-catalog`, `cache-semantic`,
  `cache-memvid`, `compress-llmlingua`, `route-predictive`, `eval`.
- Phase 2: exact-match cache pillar:
  - `CachedCompletionResponse<T>` carrier — serializable mirror of
    `rig::completion::CompletionResponse` (which is `Debug`-only) covering
    `choice`, `usage`, `message_id`, and the provider's `raw_response`.
  - `CachedRef<'a, T>` zero-clone borrowing serializer for cache writes.
  - `InMemoryCache` + `InMemoryCacheConfig`: `std::sync::Mutex<HashMap>`
    backend with capacity-bounded FIFO eviction and optional default TTL.
    Guards are dropped before any await (`await_holding_lock` clean).
  - `OptimizedModel::complete` now performs full cache hit reconstruction:
    on hit, the wrapped provider is **not** invoked and `Provenance` is
    populated with `cache_hit=true`, `router_choice=CacheHit`, `source_id`,
    and `similarity` (when carried by the entry).
  - Cache writes respect `CachePolicy::NoStore`; reads respect `NoCache` /
    `NoStore` / `ForceFresh`.
  - `tests/cache_e2e.rs`: mock `CompletionModel` proves identical requests
    are served from cache (provider call counter stays at 1), prompt
    changes miss, `ForceFresh` bypasses reads but still writes, and
    `NoStore` disables writes.
- Phase 3: structural compression pillar:
  - `JsonKeyPruner` + `JsonKeyPrunerConfig` — deterministic JSON pruner
    with preserve-key set, `max_depth` collapse, `max_array_len`
    truncation with `"..."` marker, and `max_string_len` ellipsis.
  - Compressor pass applies to `CompletionRequest::additional_params` and
    to documents whose `text` parses as JSON; plain-text documents are
    left untouched.
  - `CompressionStats` populated with char-count proxy (real tokenization
    arrives with the `model-catalog` integration in Phase 5).
- `compress-llmlingua` feature: `LlmlinguaCompressor` and
  `LlmlinguaConfig` provide a dependency-free, LLMLingua-inspired token
  pruner. The adapter scores word tokens by position, structural keywords,
  configurable preserve terms, and configurable salient terms, then keeps the
  highest-scoring tokens in original order with `[...]` elision markers. It
  plugs into the existing `Compressor` trait and mutates preamble, text chat
  blocks, and request documents without pulling a tokenizer or transformer
  stack into the default build.
- Phase 4: cheap → strong cascade routing:
  - `StaticCascade<Cheap, Strong, V>` — dispatches the cheap leg first,
    runs a `Validator` against the JSON-serialized raw response, and falls
    back to the strong leg on rejection. Returns a `NormalizedResponse`
    with `Provenance::router_choice` set to `Cheap` or `Strong`. Constrains
    `Strong::Response == Cheap::Response` so both legs return the same
    provider payload type.
  - `LengthValidator` — minimum cumulative character count across all
    string leaves in the response JSON. Schema-agnostic (no hard-coded
    OpenAI / Anthropic shapes).
  - `RegexValidator` — compiled regex pattern matched against the
    concatenated string leaves. Construction surfaces `TokudoError::Config`
    on invalid patterns.
  - `ConfidenceValidator` — JSON Pointer (RFC 6901) lookup against a
    numeric field with a `min` threshold. Missing or non-numeric fields
    count as rejection.
  - `tests/cascade.rs`: two mock completion models confirm (a) the strong
    leg is **not** called when the cheap leg passes validation and (b) the
    strong leg is dispatched exactly once when the cheap leg is rejected.
  - 10 unit tests for validators cover length floors, nested counting,
    regex matching, bad-pattern errors, threshold pass/fail, missing
    pointer, and nested pointer lookups.
  - Note: `OptimizedModel` does not yet wrap `StaticCascade` directly; the
    cascade is a standalone primitive in Phase 4. Threading it through
    `OptimizedModel`'s `M` parameter (so cache + compression + cascade
    stack together) is a Phase 4.1 follow-up.
- `route-predictive` feature: `PredictiveRouter`,
  `PredictiveRouteExample`, and `PredictiveRouterConfig` provide a
  dependency-free calibration-set router. It extracts normalized request
  tokens, compares them against cheap/strong examples with Jaccard overlap,
  and conservatively routes unknown prompts to `Strong`.
- `lineage` feature: `LineageEdge` and `LineageRelation` provide compact
  source → response audit records. When enabled, `OptimizedModel` attaches a
  lineage edge to `Provenance` and emits a `lineage.edge` event for cache hits
  and provider calls, linking served responses to cache entry IDs or request
  cache keys.
- Phase 4.1: cascade wiring through `OptimizedModel`:
  - New `DispatchModel` trait + `DispatchOutcome<R>` (`src/dispatch.rs`)
    abstract the inner call site. A blanket `impl<M: CompletionModel>
    DispatchModel for M` keeps existing user code source-compatible — no
    user-visible bound on `OptimizedModel<M, ...>` changed shape.
  - `StaticCascade` now implements `DispatchModel` directly, so a cascade
    can be passed straight to `OptimizedModel::builder(cascade)` and the
    leg that answered (`Cheap` / `Strong`) propagates into
    `Provenance.router_choice` on the returned `NormalizedResponse`.
    Cache + compression + cascade now stack in one decorator.
  - Integration test `cascade_plugs_into_optimized_model_and_caches_strong_reply`
    verifies the strong reply gets cached on first miss and that a second
    call returns it as a hit without invoking either leg.
- Phase 5: observability pillar:
  - `TokudoEvent` enum and `TokudoEnvelope` carrier emitted via
    `tracing::info!` on the `rig_tokudo` target. Wire shape is a flat JSON
    object tagged by `kind` (`cache.hit`, `cache.miss`, `route.decision`,
    `compress.applied`, `cost.estimate`) wrapped in `version`,
    `occurred_at_millis`, and a monotonic per-process `tick`. Scalar
    `rig_tokudo.kind` / `rig_tokudo.tick` / `rig_tokudo.version` fields
    sit alongside the JSON payload so OpenTelemetry collectors can route
    without parsing.
  - `OptimizedModel::complete` now emits `cache.miss` (or `cache.hit`),
    `compress.applied` (when the compressor reports stats),
    `route.decision` (when the router runs), and `cost.estimate` (after
    every provider call, or with zero usage on cache hits).
  - `StaticCascade::complete` emits `route.decision` with `Cheap` or
    `Strong` after each leg decision.
  - `observe::emit` and `observe::build_event` helpers keep telemetry
    failure-tolerant: serialization errors are logged at `warn` on the
    same target and never propagate into the hot path.
  - `tests/observe.rs` integration test uses a custom
    `tracing-subscriber` layer to capture events and asserts the
    decorator emits `cache.miss`, `cache.hit`, `route.decision`, and two
    `cost.estimate` events across a miss-then-hit pair, plus that the
    miss-side `cost.estimate` carries the provider's reported token
    counts.
- Phase 5.1: default-on tap + pricing integration:
  - `tap` and `model-catalog` are now default features.
  - With `tap`, `observe::emit` keeps the tokudo-owned JSON payload on the
    `rig_tokudo` target and also emits aligned `rig_tap.*` scalar fields
    (`version`, `kind`, `tick`, `occurred_at_millis`, `conversation_id`) so
    collectors can route tokudo events next to native `rig-tap` events
    without requiring a new `rig_tap::EventKind` variant.
  - With `model-catalog`, new pricing helpers wrap
    `rig_model_catalog::PricingTable` and fill miss-side
    `TokudoEvent::CostEstimate::usd_actual_estimate` plus
    `Provenance.usd_actual_estimate`; cache hits record
    `usd_actual_estimate = Some(0.0)` and `usd_saved_estimate` from the
    cached response's original usage when the request model resolves.
    Provider-reported `cached_input_tokens` and
    `cache_creation_input_tokens` now flow into `Provenance`,
    `TokudoEvent::CostEstimate`, and `RunStats`; the new
    `estimate_provider_cache_usd_delta` helper computes the provider-side
    cache price delta separately so `Report::from_runs` subtracts provider
    cache discounts from Tokudo-controlled USD savings.
  - Fixed the local `rig-model-catalog` path dependency to point at
    `../rig-model-catalog` for in-tree validation.
  - `examples/measure_savings.rs` now derives mock-provider USD from the
    real `rig-model-catalog` pricing table instead of local hard-coded mock
    rates.
  - `tests/tap_envelope.rs` asserts `tap` scalar alignment, and
    `tests/observe.rs` now asserts model-catalog USD estimation on cost
    events.
- Phase 6.1: semantic cache + pure quality scoring:
  - `cache-semantic` now exposes `SemanticCache`, `SemanticCacheConfig`,
    and `SemanticCacheKey`. `SemanticCache` is a read-through adapter over
    `rig::vector_store::VectorStoreIndexDyn`; because Rig's dynamic vector
    surface is search-only, `put` is intentionally a no-op and callers
    populate the vector index through its native ingestion path.
  - `cache-memvid` exposes `MemvidSemanticCache` and
    `MemvidSemanticCacheConfig` as a durable `.mv2` backend over
    `rig_memvid::MemvidStore`. It stores serialized `CachedEntry` payloads
    as frame text with a searchable cache-key prefix and reads hits back via
    memvid-native search.
  - `eval` now exposes pure Rust ROUGE-L helpers: `RougeLScore`,
    `rouge_l`, and `mean_rouge_l_f1`, suitable for feeding
    `Report::from_runs(..., Some(score))`.
  - `eval` also exposes the replay bridge: `ReplayCallStats`, `ReplayRow`,
    `ReplayReport`, and `replay_report`. Hosts can feed recorded baseline
    stats, Tokudo stats, optional outputs, and `Provenance` into the bridge
    to produce a Tokudo `Report` plus native `rig-retrieval-evals::MultiReport`
    metrics (`tokudo.cache_hit`, route shares, USD savings, and optional
    `quality.rouge_l_f1`) without re-running provider calls.
  - Report artifact persistence helpers: `Report::write_artifacts` writes
    stable `report.json`, `report.md`, and `manifest.json` files;
    `ReplayReport::write_artifacts` adds `metrics.json` and `metrics.md` for
    the paired `rig-retrieval-evals::MultiReport`. The artifact manifest can
    carry a caller-supplied label for CI/benchmark runs.
  - Added `tests/semantic_cache.rs` and `tests/quality_regression.rs`.
- Phase 6: measurement & savings benchmark:
  - `Report`, `RunStats`, `Deltas`, `Thresholds`, and `EvaluationOutcome`
    in `src/report.rs` — pure aggregation types covering request count,
    prompt/completion tokens, provider cache-token counts, wall-clock ms,
    USD, cache hits, and cheap/strong call counts; `Report::from_runs`
    computes deterministic deltas (gross USD delta, provider-cache delta,
    Tokudo USD saved, USD saved %, cache-hit rate, cheap-model share,
    speedup) with div-by-zero guarded paths, and `Report::to_markdown`
    renders a pipe-separated savings table. `Thresholds::default()`
    encodes the Phase 6.G success criteria
    (`usd_saved_pct ≥ 0.40`, `cache_hit_rate ≥ 0.30`,
    `cheap_model_share ≥ 0.50`).
  - `examples/measure_savings.rs` — headline mock-provider workload
    runner. Loads `data/workload.jsonl`, runs a baseline pass against a
    deterministic `MockProvider`, then a tokudo-wrapped pass with
    `InMemoryCache`. Writes `report/report.json` and `report/report.md`,
    evaluates against an example-tuned `Thresholds`
    (`min_usd_saved_pct=0.20`, `min_cache_hit_rate=0.25`,
    `min_cheap_model_share=0.0`), and exits non-zero on miss. Asserts
    `tool_call`-tagged rows never populate the cache via
    `CachePolicy::NoStore`.
  - `data/workload.jsonl` — curated 20-row workload covering novel,
    repeat, and tool-call tags so the cache pillar surfaces a measurable
    hit rate even without a real model server.
  - `tests/proptest_cache.rs` — four `proptest` cases: deterministic
    `DefaultCacheKey` for identical requests, temperature-bucket
    collision within `±0.04` of an aligned bucket center, preamble
    change → distinct key, namespace change → distinct key.
  - Re-exports `Deltas`, `EvaluationOutcome`, `Report`, `RunStats`, and
    `Thresholds` from `lib.rs`.
  - Deferred beyond v0.1: real-provider wiring (Ollama / OpenAI /
    Anthropic) and `criterion` benches.
