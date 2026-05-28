//! Static two-leg cascade: dispatch a cheap model first, fall back to a
//! strong model when the configured [`Validator`] rejects the cheap reply.

use rig::completion::{CompletionModel, CompletionRequest, CompletionResponse};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::dispatch::{DispatchModel, DispatchOutcome};
use crate::error::{Result, TokudoError};
use crate::observe::{self, TokudoEvent};
use crate::provenance::{NormalizedResponse, Provenance, RouterChoice};
use crate::traits::Validator;

/// Static cheap → strong cascade.
///
/// `StaticCascade` is a standalone primitive that owns two
/// [`CompletionModel`]s with the same `Response` type and a [`Validator`]
/// that scores the cheap reply. The cascade always tries `cheap` first; on
/// validation rejection it forwards the same [`CompletionRequest`] to
/// `strong` and returns the strong reply unconditionally.
///
/// The cascade returns a [`NormalizedResponse`] whose
/// [`Provenance::router_choice`] reflects the leg that actually answered
/// (`Cheap` or `Strong`).
///
/// `StaticCascade` also implements [`DispatchModel`], so it can be plugged
/// directly into [`crate::OptimizedModel`] as the inner model to compose
/// cache + compression + cascade in one decorator. The cascade's route
/// decision propagates into [`Provenance::router_choice`] on the resulting
/// [`NormalizedResponse`].
///
/// # Example
///
/// ```no_run
/// # async fn demo<C, S>(cheap: C, strong: S, req: rig::completion::CompletionRequest)
/// # where
/// #     C: rig::completion::CompletionModel,
/// #     S: rig::completion::CompletionModel<Response = C::Response>,
/// #     C::Response: serde::Serialize,
/// # {
/// use rig_tokudo::{LengthValidator, StaticCascade};
///
/// let cascade = StaticCascade::new(cheap, strong, LengthValidator::new(64));
/// let normalized = cascade.complete(req).await.unwrap();
/// # let _ = normalized;
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct StaticCascade<Cheap, Strong, V> {
    cheap: Cheap,
    strong: Strong,
    validator: V,
}

impl<Cheap, Strong, V> StaticCascade<Cheap, Strong, V> {
    /// Construct a cascade with the given legs and validator.
    pub fn new(cheap: Cheap, strong: Strong, validator: V) -> Self {
        Self {
            cheap,
            strong,
            validator,
        }
    }

    /// Borrow the cheap leg.
    pub fn cheap(&self) -> &Cheap {
        &self.cheap
    }

    /// Borrow the strong leg.
    pub fn strong(&self) -> &Strong {
        &self.strong
    }
}

impl<Cheap, Strong, V> StaticCascade<Cheap, Strong, V>
where
    Cheap: CompletionModel,
    Strong: CompletionModel<Response = Cheap::Response>,
    Cheap::Response: Serialize,
    V: Validator,
{
    /// Dispatch the cascade.
    ///
    /// Calls `cheap` first, validates the raw response, and falls back to
    /// `strong` on rejection. The returned [`NormalizedResponse::provenance`]
    /// records which leg actually answered.
    pub async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<NormalizedResponse<CompletionResponse<Cheap::Response>>> {
        // The cheap leg consumes the request; clone it up-front so we can
        // re-dispatch the same query to the strong leg on rejection.
        let fallback_request = request.clone();

        let cheap_resp = self.cheap.completion(request).await?;
        let raw_json = serde_json::to_value(&cheap_resp.raw_response)
            .map_err(|e| TokudoError::Cache(format!("serialize cheap response: {e}")))?;

        if self.validator.validate(&raw_json)? {
            let mut prov = Provenance::pass_through();
            prov.router_choice = RouterChoice::Cheap;
            observe::emit(&TokudoEvent::RouteDecision {
                choice: RouterChoice::Cheap,
            });
            tracing::debug!(
                target: "rig_tokudo::route",
                leg = "cheap",
                "cascade accepted cheap response",
            );
            return Ok(NormalizedResponse::new(cheap_resp, prov));
        }

        tracing::debug!(
            target: "rig_tokudo::route",
            leg = "cheap",
            "cascade rejected cheap response; falling back to strong",
        );
        let strong_resp = self.strong.completion(fallback_request).await?;
        let mut prov = Provenance::pass_through();
        prov.router_choice = RouterChoice::Strong;
        observe::emit(&TokudoEvent::RouteDecision {
            choice: RouterChoice::Strong,
        });
        Ok(NormalizedResponse::new(strong_resp, prov))
    }
}

impl<Cheap, Strong, V> DispatchModel for StaticCascade<Cheap, Strong, V>
where
    Cheap: CompletionModel + Send + Sync,
    Strong: CompletionModel<Response = Cheap::Response> + Send + Sync,
    Cheap::Response: Send + Sync + Serialize + DeserializeOwned,
    V: Validator,
{
    type Response = Cheap::Response;

    async fn dispatch(
        &self,
        request: CompletionRequest,
    ) -> Result<DispatchOutcome<Self::Response>> {
        let normalized = self.complete(request).await?;
        Ok(DispatchOutcome {
            response: normalized.response,
            route: Some(normalized.provenance.router_choice),
        })
    }
}
