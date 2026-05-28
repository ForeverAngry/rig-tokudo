//! Prompt-compression primitives.
//!
//! Phase 3 ships [`JsonKeyPruner`], a deterministic structural compressor that
//! shrinks `additional_params` and any JSON-valued documents in a
//! [`rig::completion::CompletionRequest`]. The `compress-llmlingua` feature
//! adds a dependency-free LLMLingua-inspired prompt token pruner.

mod json_pruner;
#[cfg(feature = "compress-llmlingua")]
mod llmlingua;

pub use json_pruner::{JsonKeyPruner, JsonKeyPrunerConfig};
#[cfg(feature = "compress-llmlingua")]
pub use llmlingua::{LlmlinguaCompressor, LlmlinguaConfig};
