# Roadmap

`rig-tokudo` is the cost-optimization decorator for Rig
`CompletionModel`s. This roadmap supersedes the inline list in
[README.md](README.md#roadmap) for planning purposes; the README stays
the marketing-grade summary. For day-to-day conventions see
[AGENTS.md](AGENTS.md).

## Positioning

Tokudo owns per-completion cost optimization. Its public surface should stay
focused on cache, compression, routing, provenance, and measurement around a
single `CompletionModel` call. It may integrate with sibling crates for pricing,
telemetry, durable cache storage, or replay metrics, but it should not absorb
their responsibilities: orchestration belongs in `rig-compose`, reusable memory
stores in `rig-memvid` / memory-policy crates, telemetry transport and schema in
`rig-tap`, model metadata in `rig-model-catalog`, and retrieval-quality harnesses
in `rig-retrieval-evals`.

## Landed (v0.2.0)

- **Phase 0–1**: bootstrap, lint policy, `OptimizedModel<M, R, C, K>`
  decorator shell, `TokudoError`, `TokudoOptions`, `Provenance`,
  `NormalizedResponse`, and the `Cache` / `CacheKey` / `Compressor` /
  `Router` / `Validator` traits.
- **Phase 2 — exact-match cache pillar**: `CachedCompletionResponse<T>`
  carrier, `CachedRef<'a, T>` borrowing serializer, `InMemoryCache` with
  capacity-bounded FIFO eviction and optional default TTL, and full hit
  reconstruction inside `OptimizedModel::complete` (provider call is
  skipped on hit; `Provenance::cache_hit = true`).
- **Phase 3 — compression**: `JsonKeyPruner` deterministic key-pruning
  compressor; `NoCompressor` reference. `compress-llmlingua` adds an
  optional dependency-free, LLMLingua-inspired prompt token pruner with
  configurable preserve/salience terms.
- **Phase 4 — cascade routing**: `StaticCascade` router + validator
  pipeline wired through `OptimizedModel`; provider calls fall back from
  cheap → strong only when the validator rejects the cheap response.
  `route-predictive` adds a train-free calibration-set router that can hint
  cheap vs strong before dispatch.
- **Phase 5 — observability**: default-on `tap` and `model-catalog` features
  emit `Provenance` decisions (cache hit/miss, router choice, USD saved)
  through the `rig-tap` schema. Provider-side cache tokens and cache price
  deltas are surfaced separately so reports do not double-count provider
  prompt-cache discounts as Tokudo savings.
- **Phase 6 — measurement**: savings reports, semantic cache adapter,
  ROUGE-L quality scoring, and the `measure_savings` example.
- **Phase 6.1 — lineage telemetry**: optional `lineage` feature emits
  compact source → response edges and attaches them to `Provenance`, linking
  cache hits to cache entry IDs and provider calls to response IDs.
- **Phase 6.2 — eval replay bridge**: `eval` exposes replay rows that turn
  recorded baseline/Tokudo call stats and `Provenance` into Tokudo savings
  reports plus native `rig-retrieval-evals::MetricReport` bundles without
  re-running provider calls.
- **Phase 6.3 — durable memvid cache writes**: `cache-memvid` now provides a
  real `MemvidSemanticCache` backend. It writes serialized `CachedEntry`
  payloads into `.mv2` frames, indexes the semantic request projection as a
  searchable frame prefix, and reads hits back through memvid-native search.
- **Phase 6.4 — report artifact persistence**: `Report::write_artifacts`
  writes stable `report.json`, `report.md`, and `manifest.json` outputs;
  `ReplayReport::write_artifacts` adds `metrics.json` and `metrics.md` for
  `rig-retrieval-evals` bundles from replay runs.
- `CachePolicy::NoStore` / `NoCache` / `ForceFresh` honored on both read
  and write paths.
- `cache-memvid` is shipped as a feature-gated durable semantic cache over
  `rig_memvid::MemvidStore`. The generic [`SemanticCache`](src/cache/semantic.rs)
  remains read-through for arbitrary `VectorStoreIndexDyn` backends.

## Next Work

- No planned v0.1 roadmap items remain open. Future work should be driven by
  real-provider benchmark feedback and the reopen triggers below.

## Prototype Grade

- `InMemoryCache` is single-process and `Mutex`-backed. It is correct
  but is not a substitute for a shared cache across replicas — use
  `cache-memvid` or a Redis-style backend behind the trait before
  production rollouts.
- `JsonKeyPruner` is conservative by design: it prunes keys, not values.
  Hosts that need value-level prompt pruning can enable `compress-llmlingua`
  or wrap the `Compressor` trait with their own tokenizer/scorer.
- ROUGE-L scoring in the savings example is informational. It is not a
  gate; do not branch CI on it without an explicit baseline.
- The default `model-catalog` pricing snapshot is whatever ships in
  `rig-model-catalog`'s `data/pricing.json`. Override at runtime via
  `PricingTable::with(...)` when rates drift.

## Out of Scope

- Forking or vendoring `rig-core`. `OptimizedModel` decorates whatever
  `CompletionModel` the host supplies.
- Depending on `rig-compose`, `rig-resources`, or `rig-mcp`. Tokudo must
  remain consumable by all of them.
- Building a UI. Telemetry rides the `rig-tap` schema; dashboards belong
  in the host.
- Cross-call planning (multi-turn budget allocation). Tokudo decides per
  call; planning across turns belongs in `rig-compose`.

## Reopen Triggers

- `rig-core::completion::Usage` gains new cache-related fields beyond
  `cached_input` / `cache_write`. `Provenance` and `ModelPrice::cost_for`
  follow additively.
- A provider ships a stable, cheap structured-output validator we can
  call instead of round-tripping the strong model — `StaticCascade`
  gets a new validator backend.
- `rig-memvid` changes `.mv2` append/search-text semantics. Revalidate
  `MemvidSemanticCache` write/read behavior against the new release.
