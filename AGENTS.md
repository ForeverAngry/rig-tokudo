# AGENTS.md

Guidance for AI coding agents working in `rig-tokudo`. Mirrors
[.github/copilot-instructions.md](.github/copilot-instructions.md).

## Project

`rig-tokudo` is a cost-optimization decorator around Rig's `CompletionModel`.
v0.1 surface:

- `OptimizedModel<M, R, C, K>` — the decorator wrapping any `CompletionModel`
  ([src/model.rs](src/model.rs)).
- `Cache` / `CacheKey` traits + `DefaultCacheKey` + `InMemoryCache`
  ([src/cache/](src/cache/), Phase 2).
- `Router` / `Validator` traits + `StaticCascade` ([src/route/](src/route/), Phase 4).
- `Compressor` trait + `NoCompressor` + `JsonKeyPruner` ([src/compress/](src/compress/), Phase 3).
- `Provenance` + `NormalizedResponse` ([src/provenance.rs](src/provenance.rs)).
- `TokudoOptions` per-request controls ([src/options.rs](src/options.rs)).

## Rules

- Rust 2024, MSRV 1.89. Library is runtime-agnostic; do not add `tokio` to
  `[dependencies]`.
- Errors: typed `thiserror` enum in [src/error.rs](src/error.rs); return
  `Result<_, TokudoError>`. No ad-hoc `Box<dyn Error>` or `String` error types.
- Never `.await` while holding a `Mutex`/`RwLock` guard
  (`clippy::await_holding_lock = deny`).
- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, `dbg!`,
  indexing/slicing, or `unreachable!` in library code — clippy `deny`/`forbid`.
  Use `?`, `ok_or(TokudoError::...)`, `get(..)`, `match`. Allowed in `tests/`,
  `examples/`, `#[cfg(test)]` blocks (gate with
  `#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]`).
- Use `tracing` for logs; no `println!` in library code.
- Document new `pub` items with `///` rustdoc and a `no_run` example.
- Re-export new public items from [src/lib.rs](src/lib.rs).

## Feature flags

Default = `["tap", "model-catalog"]` since Phase 5 shipped. Other optional
features: `cache-semantic`,
`cache-memvid`, `compress-llmlingua`, `route-predictive`, `lineage`, `eval`.
Gate optional code with `#[cfg(feature = "...")]`. Once Phases 5 and 6 land,
`tap` and `model-catalog` move into the default set.

## Validation

```sh
just check
# fmt --check + clippy --all-targets -- -D warnings (× feature combos) + tests
```

Integration tests live in [tests/](tests/). Examples must keep building:
`cargo build --examples`.

## Scope

Do not vendor `rig-core`. The crate must not depend on `rig-compose`,
`rig-resources`, or `rig-mcp` — it has to be consumable by all of them.
Update [README.md](README.md) and [CHANGELOG.md](CHANGELOG.md) for
user-visible changes.
