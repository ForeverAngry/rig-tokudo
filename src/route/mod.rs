//! Phase 4 routing primitives: static cheap → strong cascade plus a small
//! library of post-call validators.
//!
//! The [`StaticCascade`] dispatches the cheap leg first, runs a [`crate::Validator`]
//! against the raw provider response, and on rejection re-dispatches to the
//! strong leg. The validators in [`validator`] cover the three patterns
//! called out in the project plan: minimum-length, regex-match, and a
//! JSON-confidence threshold.

pub mod cascade;
#[cfg(feature = "route-predictive")]
pub mod predictive;
pub mod validator;

pub use cascade::StaticCascade;
#[cfg(feature = "route-predictive")]
pub use predictive::{PredictiveRouteExample, PredictiveRouter, PredictiveRouterConfig};
pub use validator::{ConfidenceValidator, LengthValidator, RegexValidator};
