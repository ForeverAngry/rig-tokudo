//! Typed errors for `rig-tokudo`.

use thiserror::Error;

/// Errors produced by the rig-tokudo decorator stack.
///
/// Variants are intentionally coarse-grained per pillar (cache / router /
/// compressor / provider / model-catalog / tap). Add a new variant rather than
/// reusing an existing one when surfacing a new fault domain.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TokudoError {
    /// Cache backend failure (lookup, insert, eviction, serialization).
    #[error("cache error: {0}")]
    Cache(String),

    /// Router or validator failure (cost lookup, cheap/strong dispatch).
    #[error("router error: {0}")]
    Router(String),

    /// Compressor failure (selector parse, oversize input, model adapter).
    #[error("compressor error: {0}")]
    Compress(String),

    /// Failure raised by the wrapped provider model.
    #[error("provider error: {0}")]
    Provider(#[from] rig::completion::CompletionError),

    /// Failure pulling tokenizer or pricing data from `rig-model-catalog`.
    #[error("model-catalog error: {0}")]
    ModelMeta(String),

    /// Failure emitting an observability event.
    #[error("tap error: {0}")]
    Tap(String),

    /// Failure rendering or writing savings/eval report artifacts.
    #[error("report error: {0}")]
    Report(String),

    /// Configuration was invalid at build or call time.
    #[error("configuration error: {0}")]
    Config(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, TokudoError>;
