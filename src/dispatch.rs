//! [`DispatchModel`] — the internal call abstraction that
//! [`crate::OptimizedModel`] uses on the cache-miss path.
//!
//! Any [`rig::completion::CompletionModel`] satisfies this trait via a
//! blanket implementation, so existing user code continues to type-check
//! unchanged. [`crate::StaticCascade`] also implements it directly, which
//! lets a cascade plug straight into `OptimizedModel` as the inner model
//! without users having to wrap each leg separately.
//!
//! The outcome carries an optional [`RouterChoice`] override so a cascade
//! can report which leg actually answered; `OptimizedModel` records that
//! into the final [`crate::Provenance`] when present.

use rig::completion::{CompletionModel, CompletionRequest, CompletionResponse};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::provenance::RouterChoice;

/// Outcome of a single [`DispatchModel::dispatch`] call.
#[derive(Debug)]
pub struct DispatchOutcome<R> {
    /// The raw completion response from whatever leg answered.
    pub response: CompletionResponse<R>,
    /// Override for [`crate::Provenance::router_choice`]. `None` when the
    /// dispatcher has no opinion (e.g. a plain `CompletionModel`).
    pub route: Option<RouterChoice>,
}

/// Internal call abstraction used by [`crate::OptimizedModel`].
///
/// Implemented automatically for every [`CompletionModel`] (with the
/// outcome's `route` field set to `None`), and directly by
/// [`crate::StaticCascade`] so cascades plug in without re-wrapping.
#[allow(async_fn_in_trait)]
pub trait DispatchModel: Send + Sync {
    /// Raw response payload — matches `CompletionModel::Response`.
    type Response: Send + Sync + Serialize + DeserializeOwned;

    /// Dispatch a request and return the raw response plus any optional
    /// routing override.
    async fn dispatch(&self, request: CompletionRequest)
    -> Result<DispatchOutcome<Self::Response>>;
}

impl<M> DispatchModel for M
where
    M: CompletionModel + Send + Sync,
    M::Response: Send + Sync + Serialize + DeserializeOwned,
{
    type Response = M::Response;

    async fn dispatch(
        &self,
        request: CompletionRequest,
    ) -> Result<DispatchOutcome<Self::Response>> {
        let response = self.completion(request).await?;
        Ok(DispatchOutcome {
            response,
            route: None,
        })
    }
}
