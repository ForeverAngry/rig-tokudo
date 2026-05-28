//! Phase 6.B property tests: `DefaultCacheKey` must be deterministic and
//! must depend only on the semantically-relevant fields it projects.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use proptest::prelude::*;
use rig::OneOrMany;
use rig::completion::{CompletionRequest, Message};
use rig_tokudo::{CacheKey, DefaultCacheKey, TokudoOptions};

fn request(
    preamble: &str,
    user_msg: &str,
    model: &str,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
) -> CompletionRequest {
    CompletionRequest {
        model: Some(model.to_string()),
        preamble: Some(preamble.to_string()),
        chat_history: OneOrMany::one(Message::user(user_msg)),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature,
        max_tokens,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    }
}

proptest! {
    /// Identical inputs always produce identical keys (referential transparency).
    #[test]
    fn key_is_deterministic(
        preamble in ".{0,64}",
        msg in ".{0,128}",
        model in "[a-z0-9\\-]{1,32}",
        temp in proptest::option::of(0.0f64..2.0),
        max_tokens in proptest::option::of(1u64..4096),
    ) {
        let req = request(&preamble, &msg, &model, temp, max_tokens);
        let opts = TokudoOptions::new();
        let k1 = DefaultCacheKey.key(&req, &opts).unwrap();
        let k2 = DefaultCacheKey.key(&req, &opts).unwrap();
        prop_assert_eq!(k1, k2);
    }

    /// Temperature variations within the same 0.1 bucket must collide.
    ///
    /// The bucket boundaries fall at `n / 10 + 0.05` (where `n` is an
    /// integer), so we generate two offsets within `±0.04` of an aligned
    /// bucket center to guarantee both inputs round to the same integer.
    #[test]
    fn temperature_bucket_collides_within_bucket(
        bucket in 1u32..18,
        offset_a in -0.04f64..0.04,
        offset_b in -0.04f64..0.04,
    ) {
        let center = bucket as f64 / 10.0;
        let req_a = request("p", "m", "model", Some(center + offset_a), None);
        let req_b = request("p", "m", "model", Some(center + offset_b), None);
        let opts = TokudoOptions::new();
        let k_a = DefaultCacheKey.key(&req_a, &opts).unwrap();
        let k_b = DefaultCacheKey.key(&req_b, &opts).unwrap();
        prop_assert_eq!(k_a, k_b);
    }

    /// Different preambles must produce different keys.
    #[test]
    fn preamble_changes_key(
        p1 in "[a-z]{1,32}",
        p2 in "[a-z]{1,32}",
    ) {
        prop_assume!(p1 != p2);
        let req_a = request(&p1, "msg", "model", None, None);
        let req_b = request(&p2, "msg", "model", None, None);
        let opts = TokudoOptions::new();
        let k_a = DefaultCacheKey.key(&req_a, &opts).unwrap();
        let k_b = DefaultCacheKey.key(&req_b, &opts).unwrap();
        prop_assert_ne!(k_a, k_b);
    }

    /// Different cache namespaces must produce different keys for the
    /// same request — namespaces are isolation boundaries.
    #[test]
    fn namespace_changes_key(
        ns1 in "[a-z]{1,16}",
        ns2 in "[a-z]{1,16}",
    ) {
        prop_assume!(ns1 != ns2);
        let req = request("p", "m", "model", None, None);
        let opts_a = TokudoOptions::new().with_cache_namespace(ns1);
        let opts_b = TokudoOptions::new().with_cache_namespace(ns2);
        let k_a = DefaultCacheKey.key(&req, &opts_a).unwrap();
        let k_b = DefaultCacheKey.key(&req, &opts_b).unwrap();
        prop_assert_ne!(k_a, k_b);
    }
}
