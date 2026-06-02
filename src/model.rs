//! The [`OptimizedModel`] decorator and its builder.
//!
//! Call flow:
//!
//! 1. Compute the cache key.
//! 2. Probe the cache (skipped under [`CachePolicy::NoCache`] /
//!    [`CachePolicy::NoStore`] / [`CachePolicy::ForceFresh`]). On hit,
//!    reconstruct the [`CompletionResponse`] from
//!    [`CachedCompletionResponse`] and return without calling the provider.
//! 3. Apply the compressor.
//! 4. Record the router's pre-call hint into [`Provenance`].
//! 5. Forward to the wrapped [`rig::completion::CompletionModel`].
//! 6. Run the validator on the raw response.
//! 7. Write the response into the cache (skipped under
//!    [`CachePolicy::NoStore`]).
//! 8. Return a [`NormalizedResponse`] carrying the full
//!    [`CompletionResponse`] and decorator [`Provenance`].

use rig::completion::{CompletionRequest, CompletionResponse};

use std::sync::Arc;

use crate::cache::CachedCompletionResponse;
use crate::cost::{CostBreakdown, CostModel};
use crate::dispatch::DispatchModel;
use crate::error::{Result, TokudoError};
#[cfg(feature = "lineage")]
use crate::lineage::LineageEdge;
use crate::observe::{self, TokudoEvent};
use crate::options::{CachePolicy, TokudoOptions};
use crate::provenance::{NormalizedResponse, Provenance, RouterChoice};
use crate::traits::{
    AcceptAllValidator, Cache, CacheKey, CachedEntry, Compressor, DefaultCacheKey, NoCache,
    NoCompressor, PassThroughRouter, Router, Validator,
};

/// Cost-optimization decorator around any [`rig::completion::CompletionModel`].
///
/// Type parameters select the strategy for each pillar. Phase 1 defaults
/// (`PassThroughRouter`, `NoCompressor`, `DefaultCacheKey`, `NoCache`,
/// `AcceptAllValidator`) make `OptimizedModel<M>` behave as a logging
/// pass-through; swap them in via the `with_*` builder methods.
///
/// # Example
///
/// ```no_run
/// # async fn demo<M: rig::completion::CompletionModel>(model: M, req: rig::completion::CompletionRequest) {
/// use rig_tokudo::{OptimizedModel, TokudoOptions};
///
/// let optimizer = OptimizedModel::builder(model).build();
/// let response = optimizer.complete(req, TokudoOptions::default()).await.unwrap();
/// assert!(!response.provenance.cache_hit);
/// # }
/// ```
pub struct OptimizedModel<
    M,
    R = PassThroughRouter,
    C = NoCompressor,
    K = DefaultCacheKey,
    Ca = NoCache,
    V = AcceptAllValidator,
> {
    model: M,
    router: R,
    compressor: C,
    key: K,
    cache: Ca,
    validator: V,
    cost_model: Option<Arc<dyn CostModel>>,
}

impl<M: DispatchModel> OptimizedModel<M> {
    /// Begin building an `OptimizedModel` around `model` using Phase 1 defaults.
    #[must_use]
    pub fn builder(model: M) -> OptimizedModelBuilder<M> {
        OptimizedModelBuilder {
            model,
            router: PassThroughRouter,
            compressor: NoCompressor,
            key: DefaultCacheKey,
            cache: NoCache,
            validator: AcceptAllValidator,
            cost_model: None,
        }
    }
}

impl<M, R, C, K, Ca, V> OptimizedModel<M, R, C, K, Ca, V>
where
    M: DispatchModel,
    R: Router,
    C: Compressor,
    K: CacheKey,
    Ca: Cache,
    V: Validator,
{
    /// Borrow the wrapped completion model.
    #[must_use]
    pub fn inner(&self) -> &M {
        &self.model
    }

    /// Resolve the USD cost for a call: a per-call override on `options` wins,
    /// otherwise the configured [`CostModel`], otherwise `None`.
    fn estimate_cost(
        &self,
        model: Option<&str>,
        usage: &rig::completion::Usage,
        options: &TokudoOptions,
    ) -> Option<CostBreakdown> {
        if let Some(usd_actual) = options.usd_actual_estimate {
            return Some(CostBreakdown {
                usd_actual,
                provider_cache_usd_delta: options.provider_cache_usd_delta.unwrap_or(0.0),
            });
        }
        self.cost_model
            .as_ref()
            .and_then(|cm| cm.estimate(model, usage))
    }

    /// Run the optimized completion flow.
    pub async fn complete(
        &self,
        mut request: CompletionRequest,
        options: TokudoOptions,
    ) -> Result<NormalizedResponse<CompletionResponse<M::Response>>> {
        let mut provenance = Provenance::pass_through();
        let model_name = request.model.clone();

        // Step 1: cache key.
        let cache_key = self.key.key(&request, &options)?;

        // Step 2: cache read (and hit reconstruction).
        let allow_read = !matches!(
            options.cache_policy,
            CachePolicy::NoCache | CachePolicy::NoStore | CachePolicy::ForceFresh
        );
        if allow_read && let Some(entry) = self.cache.get(&cache_key).await? {
            let cached: CachedCompletionResponse<M::Response> =
                serde_json::from_value(entry.response.clone())
                    .map_err(|e| TokudoError::Cache(format!("deserialize cached response: {e}")))?;
            let baseline = self.estimate_cost(model_name.as_deref(), &cached.usage, &options);
            let usd_baseline_estimate = baseline.map(|b| b.usd_actual);
            let provider_cache_usd_delta = baseline.map(|b| b.provider_cache_usd_delta);
            observe::emit(&TokudoEvent::CacheHit {
                cache_key: cache_key.clone(),
                source_id: entry.source_id.clone(),
                similarity: entry.similarity,
            });
            observe::emit(&TokudoEvent::CostEstimate {
                cache_hit: true,
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
                cache_write_tokens: 0,
                total_tokens: 0,
                usd_actual_estimate: Some(0.0),
                provider_cache_usd_delta: Some(0.0),
            });
            provenance.cache_hit = true;
            provenance.router_choice = RouterChoice::CacheHit;
            provenance.similarity = entry.similarity;
            provenance.source_id = Some(entry.source_id);
            provenance.usd_actual_estimate = Some(0.0);
            provenance.usd_baseline_estimate = usd_baseline_estimate;
            provenance.provider_cached_input_tokens = cached.usage.cached_input_tokens;
            provenance.provider_cache_write_tokens = cached.usage.cache_creation_input_tokens;
            provenance.provider_cache_usd_delta = provider_cache_usd_delta;
            provenance.usd_saved_estimate = usd_baseline_estimate;
            #[cfg(feature = "lineage")]
            {
                let target_id = cached
                    .message_id
                    .clone()
                    .unwrap_or_else(|| cache_key.clone());
                let edge = LineageEdge::cache_hit(
                    cache_key.clone(),
                    provenance
                        .source_id
                        .clone()
                        .unwrap_or_else(|| cache_key.clone()),
                    target_id,
                    model_name.clone(),
                );
                observe::emit(&TokudoEvent::LineageEdge { edge: edge.clone() });
                provenance.lineage = Some(edge);
            }
            return Ok(NormalizedResponse::new(cached.into_response(), provenance));
        }
        if allow_read {
            observe::emit(&TokudoEvent::CacheMiss {
                cache_key: cache_key.clone(),
            });
        }

        // Step 3: compression.
        if !options.bypass_compress
            && let Some(stats) = self.compressor.compress(&mut request, &options)?
        {
            provenance.compressed_ratio = stats.ratio();
            observe::emit(&TokudoEvent::CompressApplied {
                input_tokens: stats.input_tokens,
                output_tokens: stats.output_tokens,
                ratio: stats.ratio(),
            });
        }

        // Step 4: router hint.
        if !options.bypass_router {
            let choice = self.router.route(&request, &options);
            provenance.router_choice = choice;
            observe::emit(&TokudoEvent::RouteDecision { choice });
        }

        // Step 5: forward to inner dispatcher (plain model or cascade).
        let outcome = self.model.dispatch(request).await?;
        let resp = outcome.response;
        if let Some(route) = outcome.route {
            provenance.router_choice = route;
        }

        let breakdown = self.estimate_cost(model_name.as_deref(), &resp.usage, &options);
        let usd_actual_estimate = breakdown.map(|b| b.usd_actual);
        let provider_cache_usd_delta = breakdown.map(|b| b.provider_cache_usd_delta);
        provenance.usd_actual_estimate = usd_actual_estimate;
        provenance.usd_baseline_estimate = usd_actual_estimate;
        provenance.provider_cached_input_tokens = resp.usage.cached_input_tokens;
        provenance.provider_cache_write_tokens = resp.usage.cache_creation_input_tokens;
        provenance.provider_cache_usd_delta = provider_cache_usd_delta;
        provenance.usd_saved_estimate = usd_actual_estimate.map(|_| 0.0);
        #[cfg(feature = "lineage")]
        {
            let target_id = resp.message_id.clone().unwrap_or_else(|| cache_key.clone());
            let edge = LineageEdge::provider_call(
                cache_key.clone(),
                target_id,
                provenance.router_choice,
                model_name.clone(),
            );
            observe::emit(&TokudoEvent::LineageEdge { edge: edge.clone() });
            provenance.lineage = Some(edge);
        }

        observe::emit(&TokudoEvent::CostEstimate {
            cache_hit: false,
            input_tokens: resp.usage.input_tokens,
            output_tokens: resp.usage.output_tokens,
            cached_input_tokens: resp.usage.cached_input_tokens,
            cache_write_tokens: resp.usage.cache_creation_input_tokens,
            total_tokens: resp.usage.total_tokens,
            usd_actual_estimate,
            provider_cache_usd_delta,
        });

        // Step 6: validator on raw response.
        let raw_json = serde_json::to_value(&resp.raw_response)
            .map_err(|e| TokudoError::Cache(format!("serialize for validate: {e}")))?;
        let _accepted = self.validator.validate(&raw_json)?;

        // Step 7: cache write (unless NoStore).
        if !matches!(options.cache_policy, CachePolicy::NoStore) {
            let value = serde_json::to_value(CachedCompletionResponse::from_borrowed(&resp))
                .map_err(|e| TokudoError::Cache(format!("serialize for cache write: {e}")))?;
            let entry = CachedEntry {
                response: value,
                expires_at_secs: None,
                similarity: None,
                source_id: cache_key.clone(),
            };
            self.cache.put(&cache_key, entry).await?;
        }

        Ok(NormalizedResponse::new(resp, provenance))
    }
}

/// Builder for [`OptimizedModel`].
///
/// Each `with_*` method takes ownership and returns a new builder with the
/// corresponding type parameter updated. Call [`Self::build`] to finalize.
pub struct OptimizedModelBuilder<
    M,
    R = PassThroughRouter,
    C = NoCompressor,
    K = DefaultCacheKey,
    Ca = NoCache,
    V = AcceptAllValidator,
> {
    model: M,
    router: R,
    compressor: C,
    key: K,
    cache: Ca,
    validator: V,
    cost_model: Option<Arc<dyn CostModel>>,
}

impl<M, R, C, K, Ca, V> OptimizedModelBuilder<M, R, C, K, Ca, V> {
    /// Plug in a [`Router`].
    pub fn with_router<R2: Router>(self, router: R2) -> OptimizedModelBuilder<M, R2, C, K, Ca, V> {
        OptimizedModelBuilder {
            model: self.model,
            router,
            compressor: self.compressor,
            key: self.key,
            cache: self.cache,
            validator: self.validator,
            cost_model: self.cost_model,
        }
    }

    /// Plug in a [`Compressor`].
    pub fn with_compressor<C2: Compressor>(
        self,
        compressor: C2,
    ) -> OptimizedModelBuilder<M, R, C2, K, Ca, V> {
        OptimizedModelBuilder {
            model: self.model,
            router: self.router,
            compressor,
            key: self.key,
            cache: self.cache,
            validator: self.validator,
            cost_model: self.cost_model,
        }
    }

    /// Plug in a [`CacheKey`] strategy.
    pub fn with_cache_key<K2: CacheKey>(
        self,
        key: K2,
    ) -> OptimizedModelBuilder<M, R, C, K2, Ca, V> {
        OptimizedModelBuilder {
            model: self.model,
            router: self.router,
            compressor: self.compressor,
            key,
            cache: self.cache,
            validator: self.validator,
            cost_model: self.cost_model,
        }
    }

    /// Plug in a [`Cache`] backend.
    pub fn with_cache<Ca2: Cache>(self, cache: Ca2) -> OptimizedModelBuilder<M, R, C, K, Ca2, V> {
        OptimizedModelBuilder {
            model: self.model,
            router: self.router,
            compressor: self.compressor,
            key: self.key,
            cache,
            validator: self.validator,
            cost_model: self.cost_model,
        }
    }

    /// Plug in a [`Validator`].
    pub fn with_validator<V2: Validator>(
        self,
        validator: V2,
    ) -> OptimizedModelBuilder<M, R, C, K, Ca, V2> {
        OptimizedModelBuilder {
            model: self.model,
            router: self.router,
            compressor: self.compressor,
            key: self.key,
            cache: self.cache,
            validator,
            cost_model: self.cost_model,
        }
    }

    /// Plug in a host-supplied [`CostModel`] for USD cost estimates.
    ///
    /// Without one (and without a per-call
    /// [`TokudoOptions::with_cost_estimate`](crate::TokudoOptions::with_cost_estimate)
    /// override) tokudo reports token counts only and leaves USD fields `None`.
    #[must_use]
    pub fn with_cost_model(mut self, cost_model: impl CostModel + 'static) -> Self {
        self.cost_model = Some(Arc::new(cost_model));
        self
    }

    /// Finalize into an [`OptimizedModel`].
    #[must_use]
    pub fn build(self) -> OptimizedModel<M, R, C, K, Ca, V> {
        OptimizedModel {
            model: self.model,
            router: self.router,
            compressor: self.compressor,
            key: self.key,
            cache: self.cache,
            validator: self.validator,
            cost_model: self.cost_model,
        }
    }
}
