//! `rig-tokudo` — cost-optimization decorator for Rig completion models.
//!
//! See the [README](https://github.com/ForeverAngry/rig-tokudo) and the
//! `AGENTS.md` companion file for project conventions. The crate ships
//! the [`OptimizedModel`] decorator and its supporting cache, compression,
//! routing, observability, pricing, and measurement surfaces.
//!
//! # Decorator pillars
//!
//! - [`Cache`] / [`CacheKey`] — cache hits skip the provider call.
//! - [`Compressor`] — shrinks the prompt before dispatch.
//! - [`Router`] / [`Validator`] — cascade cheap → strong models.
//! - [`Provenance`] — per-call audit record consumed by observability and
//!   savings reports.

#![warn(missing_docs)]

pub mod cache;
pub mod compress;
pub mod dispatch;
pub mod error;
#[cfg(feature = "lineage")]
pub mod lineage;
pub mod model;
pub mod observe;
pub mod options;
#[cfg(feature = "model-catalog")]
pub mod pricing;
pub mod provenance;
#[cfg(feature = "eval")]
pub mod quality;
#[cfg(feature = "eval")]
pub mod replay;
pub mod report;
pub mod route;
pub mod traits;

pub use cache::{CachedCompletionResponse, InMemoryCache, InMemoryCacheConfig};
#[cfg(feature = "cache-memvid")]
pub use cache::{MemvidSemanticCache, MemvidSemanticCacheConfig};
#[cfg(feature = "cache-semantic")]
pub use cache::{SemanticCache, SemanticCacheConfig, SemanticCacheKey};
pub use compress::{JsonKeyPruner, JsonKeyPrunerConfig};
#[cfg(feature = "compress-llmlingua")]
pub use compress::{LlmlinguaCompressor, LlmlinguaConfig};
pub use dispatch::{DispatchModel, DispatchOutcome};
pub use error::{Result, TokudoError};
#[cfg(feature = "lineage")]
pub use lineage::{LineageEdge, LineageRelation};
pub use model::{OptimizedModel, OptimizedModelBuilder};
pub use observe::{TokudoEnvelope, TokudoEvent};
pub use options::{CachePolicy, TokudoOptions};
#[cfg(feature = "model-catalog")]
pub use pricing::{
    ResolvedPrice, estimate_actual_usd, estimate_provider_cache_usd_delta, resolve_price,
};
#[cfg(feature = "lineage")]
pub use provenance::{NormalizedResponse, Provenance, RouterChoice};
#[cfg(not(feature = "lineage"))]
pub use provenance::{NormalizedResponse, Provenance, RouterChoice};
#[cfg(feature = "eval")]
pub use quality::{RougeLScore, mean_rouge_l_f1, rouge_l};
#[cfg(feature = "eval")]
pub use replay::{ReplayCallStats, ReplayReport, ReplayRow, replay_report};
pub use report::{Deltas, EvaluationOutcome, Report, RunStats, Thresholds};
#[cfg(not(target_family = "wasm"))]
pub use report::{ReportArtifactKind, ReportArtifactMetadata, ReportArtifactPaths};
pub use route::{ConfidenceValidator, LengthValidator, RegexValidator, StaticCascade};
#[cfg(feature = "route-predictive")]
pub use route::{PredictiveRouteExample, PredictiveRouter, PredictiveRouterConfig};
pub use traits::{
    AcceptAllValidator, Cache, CacheKey, CachedEntry, CompressionStats, Compressor,
    DefaultCacheKey, NoCache, NoCompressor, PassThroughRouter, Router, Validator,
};
