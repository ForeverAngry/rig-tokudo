//! Cache backends and the serialized response carrier used to round-trip a
//! [`rig::completion::CompletionResponse`] through any [`crate::Cache`].
//!
//! Phase 2 ships [`InMemoryCache`] as the default in-process backend. The
//! semantic / vector-store-backed cache lands behind the `cache-semantic`
//! feature in a follow-up step.

#[cfg(feature = "cache-foyer")]
mod foyer;
mod memory;
#[cfg(feature = "cache-memvid")]
mod memvid;
#[cfg(feature = "cache-semantic")]
mod semantic;

#[cfg(feature = "cache-foyer")]
pub use foyer::{FoyerCache, FoyerCacheConfig};
pub use memory::{InMemoryCache, InMemoryCacheConfig};
#[cfg(feature = "cache-memvid")]
pub use memvid::{MemvidSemanticCache, MemvidSemanticCacheConfig};
#[cfg(feature = "cache-semantic")]
pub use semantic::{SemanticCache, SemanticCacheConfig, SemanticCacheKey};

use rig::OneOrMany;
use rig::completion::{AssistantContent, CompletionResponse, Usage};
use serde::{Deserialize, Serialize};

/// Owned, serializable mirror of [`CompletionResponse`] suitable for cache
/// storage and reconstruction on hit.
///
/// `CompletionResponse<T>` itself is not `Serialize` (it only derives `Debug`),
/// so `rig-tokudo` stores its load-bearing fields here. The wrapped
/// `raw_response: T` round-trips via the provider's own `Serialize` /
/// `DeserializeOwned` bounds, which are required by the
/// [`rig::completion::CompletionModel`] trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedCompletionResponse<T> {
    /// One or more assistant content blocks chosen by the provider.
    pub choice: OneOrMany<AssistantContent>,
    /// Token usage reported by the provider.
    pub usage: Usage,
    /// Provider-assigned message identifier, when present.
    pub message_id: Option<String>,
    /// Raw provider response payload.
    pub raw_response: T,
}

impl<T> CachedCompletionResponse<T> {
    /// Build the owned cache record from a borrowed [`CompletionResponse`].
    ///
    /// Allocates one clone of each field; no clone of `T` is performed by
    /// the caller (it is borrowed in the [`Serialize`] impl that downstream
    /// cache backends invoke).
    pub fn from_borrowed(response: &CompletionResponse<T>) -> CachedRef<'_, T> {
        CachedRef {
            choice: &response.choice,
            usage: &response.usage,
            message_id: &response.message_id,
            raw_response: &response.raw_response,
        }
    }

    /// Reconstruct a [`CompletionResponse`] from this cached record.
    #[must_use]
    pub fn into_response(self) -> CompletionResponse<T> {
        CompletionResponse {
            choice: self.choice,
            usage: self.usage,
            message_id: self.message_id,
            raw_response: self.raw_response,
        }
    }
}

/// Borrowing serializer for [`CompletionResponse`] used when writing to a cache.
///
/// Returned by [`CachedCompletionResponse::from_borrowed`]; opaque to callers
/// other than as a [`Serialize`] target.
#[derive(Debug, Serialize)]
pub struct CachedRef<'a, T> {
    choice: &'a OneOrMany<AssistantContent>,
    usage: &'a Usage,
    message_id: &'a Option<String>,
    raw_response: &'a T,
}
