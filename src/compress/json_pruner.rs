//! [`JsonKeyPruner`] — deterministic JSON structural compressor.

use std::collections::HashSet;

use rig::completion::CompletionRequest;
use serde_json::Value;

use crate::error::Result;
use crate::options::TokudoOptions;
use crate::traits::{CompressionStats, Compressor};

/// Replacement placeholder for elided structures.
const ELIDED: &str = "...";

/// Configuration for [`JsonKeyPruner`].
///
/// Limits are applied to *unpreserved* sub-values. Whenever a key matches the
/// [`preserve_keys`](JsonKeyPrunerConfig::preserve_keys) set, the associated
/// value is copied through verbatim — depth, array length, and string length
/// limits are ignored for that subtree.
#[derive(Debug, Clone)]
pub struct JsonKeyPrunerConfig {
    /// Key names (at any depth) whose values are preserved verbatim.
    pub preserve_keys: HashSet<String>,
    /// Maximum nesting depth. Containers beyond this depth collapse to the
    /// string `"..."`. The top level is depth 0.
    pub max_depth: usize,
    /// Maximum number of elements retained in any array. Excess elements are
    /// replaced by a single trailing `"..."`.
    pub max_array_len: usize,
    /// Maximum number of characters retained in any string. Strings longer
    /// than this are truncated and suffixed with `"..."`.
    pub max_string_len: usize,
}

impl Default for JsonKeyPrunerConfig {
    fn default() -> Self {
        Self {
            preserve_keys: HashSet::new(),
            max_depth: 4,
            max_array_len: 8,
            max_string_len: 512,
        }
    }
}

/// Deterministic JSON structural compressor.
///
/// Pruning rules (applied recursively to [`Value::Object`] / [`Value::Array`]):
///
/// 1. If the current key is in [`JsonKeyPrunerConfig::preserve_keys`], the
///    value is left untouched.
/// 2. If the current depth exceeds [`max_depth`](JsonKeyPrunerConfig::max_depth),
///    the value is replaced by the string `"..."`.
/// 3. Arrays longer than
///    [`max_array_len`](JsonKeyPrunerConfig::max_array_len) are truncated, and
///    a final `"..."` element is appended to mark elision.
/// 4. Strings longer than
///    [`max_string_len`](JsonKeyPrunerConfig::max_string_len) are truncated
///    with a `"..."` suffix.
///
/// The compressor is applied to:
///
/// - [`CompletionRequest::additional_params`] (when set).
/// - The `text` field of each [`rig::completion::Document`] *if it parses as
///   JSON*. Plain-text documents are left untouched.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rig_tokudo::{JsonKeyPruner, JsonKeyPrunerConfig};
///
/// let pruner = JsonKeyPruner::new(JsonKeyPrunerConfig {
///     preserve_keys: HashSet::from(["id".to_string()]),
///     max_depth: 2,
///     max_array_len: 3,
///     max_string_len: 64,
/// });
/// let mut v = serde_json::json!({
///     "id": "keep-me",
///     "blob": { "a": { "b": { "c": 1 } } },
/// });
/// pruner.prune_value(&mut v);
/// assert_eq!(v["id"], "keep-me");
/// ```
#[derive(Debug, Clone)]
pub struct JsonKeyPruner {
    config: JsonKeyPrunerConfig,
}

impl JsonKeyPruner {
    /// Construct with the given configuration.
    #[must_use]
    pub fn new(config: JsonKeyPrunerConfig) -> Self {
        Self { config }
    }

    /// Construct with default limits and no preserved keys.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(JsonKeyPrunerConfig::default())
    }

    /// Apply pruning to an in-place [`Value`].
    pub fn prune_value(&self, value: &mut Value) {
        self.walk(value, 0);
    }

    fn walk(&self, value: &mut Value, depth: usize) {
        if depth > self.config.max_depth {
            *value = Value::String(ELIDED.to_string());
            return;
        }
        match value {
            Value::Object(map) => {
                for (k, v) in map.iter_mut() {
                    if self.config.preserve_keys.contains(k) {
                        continue;
                    }
                    self.walk(v, depth + 1);
                }
            }
            Value::Array(items) => {
                let truncated = items.len() > self.config.max_array_len;
                if truncated {
                    items.truncate(self.config.max_array_len);
                }
                for item in items.iter_mut() {
                    self.walk(item, depth + 1);
                }
                if truncated {
                    items.push(Value::String(ELIDED.to_string()));
                }
            }
            Value::String(s) if s.chars().count() > self.config.max_string_len => {
                let mut truncated: String = s.chars().take(self.config.max_string_len).collect();
                truncated.push_str(ELIDED);
                *s = truncated;
            }
            _ => {}
        }
    }

    /// Estimate the character footprint of a JSON value (used as a token proxy).
    fn approx_chars(value: &Value) -> u32 {
        serde_json::to_string(value)
            .map(|s| u32::try_from(s.len()).unwrap_or(u32::MAX))
            .unwrap_or(0)
    }
}

impl Compressor for JsonKeyPruner {
    fn compress(
        &self,
        request: &mut CompletionRequest,
        _options: &TokudoOptions,
    ) -> Result<Option<CompressionStats>> {
        let mut input_chars: u64 = 0;
        let mut output_chars: u64 = 0;
        let mut touched = false;

        // 1. additional_params
        if let Some(params) = request.additional_params.as_mut() {
            let before = u64::from(Self::approx_chars(params));
            self.prune_value(params);
            let after = u64::from(Self::approx_chars(params));
            input_chars = input_chars.saturating_add(before);
            output_chars = output_chars.saturating_add(after);
            touched = true;
        }

        // 2. Document texts that parse as JSON.
        for doc in request.documents.iter_mut() {
            let Ok(mut parsed) = serde_json::from_str::<Value>(&doc.text) else {
                continue;
            };
            let before = u64::try_from(doc.text.len()).unwrap_or(u64::MAX);
            self.prune_value(&mut parsed);
            let Ok(new_text) = serde_json::to_string(&parsed) else {
                continue;
            };
            let after = u64::try_from(new_text.len()).unwrap_or(u64::MAX);
            doc.text = new_text;
            input_chars = input_chars.saturating_add(before);
            output_chars = output_chars.saturating_add(after);
            touched = true;
        }

        if !touched {
            return Ok(None);
        }

        // Char count is used as a coarse token proxy; real tokenization is a
        // follow-up enhancement.
        Ok(Some(CompressionStats {
            input_tokens: u32::try_from(input_chars).unwrap_or(u32::MAX),
            output_tokens: u32::try_from(output_chars).unwrap_or(u32::MAX),
        }))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use rig::OneOrMany;
    use rig::completion::{Document, Message};

    fn pruner(max_depth: usize, max_array_len: usize, max_string_len: usize) -> JsonKeyPruner {
        JsonKeyPruner::new(JsonKeyPrunerConfig {
            preserve_keys: HashSet::new(),
            max_depth,
            max_array_len,
            max_string_len,
        })
    }

    #[test]
    fn collapses_beyond_max_depth() {
        let p = pruner(1, 8, 64);
        let mut v = serde_json::json!({ "a": { "b": { "c": 1 } } });
        p.prune_value(&mut v);
        // depth 0 = root object, depth 1 = "a" subtree (preserved as object),
        // depth 2 = "b" subtree which exceeds max_depth=1 → elided.
        assert_eq!(v["a"]["b"], Value::String(ELIDED.into()));
    }

    #[test]
    fn truncates_arrays_with_marker() {
        let p = pruner(8, 2, 64);
        let mut v = serde_json::json!([1, 2, 3, 4]);
        p.prune_value(&mut v);
        assert_eq!(v, serde_json::json!([1, 2, "..."]));
    }

    #[test]
    fn truncates_long_strings() {
        let p = pruner(8, 8, 5);
        let mut v = serde_json::json!("abcdefghij");
        p.prune_value(&mut v);
        assert_eq!(v, Value::String("abcde...".into()));
    }

    #[test]
    fn preserved_keys_are_immune() {
        let p = JsonKeyPruner::new(JsonKeyPrunerConfig {
            preserve_keys: HashSet::from(["raw".to_string()]),
            max_depth: 1,
            max_array_len: 1,
            max_string_len: 3,
        });
        let mut v = serde_json::json!({
            "raw": { "deep": { "nested": "untouched-and-long" } },
            "other": "abcdef",
        });
        p.prune_value(&mut v);
        assert_eq!(v["raw"]["deep"]["nested"], "untouched-and-long");
        assert_eq!(v["other"], "abc...");
    }

    #[test]
    fn compress_no_op_when_no_targets() {
        let p = JsonKeyPruner::with_defaults();
        let mut req = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::user("hi")),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };
        assert!(
            p.compress(&mut req, &TokudoOptions::new())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn compress_prunes_additional_params() {
        let p = pruner(1, 2, 8);
        let mut req = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::user("hi")),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: Some(serde_json::json!({
                "deep": { "x": { "y": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" } },
                "list": [
                    "first-element-with-a-long-string",
                    "second-element-also-long-enough",
                    "third-element-with-extra-padding",
                    "fourth-element-also-substantial",
                    "fifth-and-final-padding-element",
                ],
            })),
            output_schema: None,
        };
        let stats = p
            .compress(&mut req, &TokudoOptions::new())
            .unwrap()
            .expect("touched");
        assert!(
            stats.output_tokens < stats.input_tokens,
            "expected shrink: {} -> {}",
            stats.input_tokens,
            stats.output_tokens
        );
        let params = req.additional_params.unwrap();
        assert_eq!(params["deep"]["x"], Value::String("...".into()));
        // The array is truncated to two short-string items + ellipsis.
        let list = params["list"].as_array().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[2], Value::String("...".into()));
    }

    #[test]
    fn compress_prunes_json_documents_only() {
        let p = pruner(8, 2, 64);
        let big_array = serde_json::json!([
            "alpha-with-some-padding-bytes",
            "beta-with-some-padding-bytes",
            "gamma-with-some-padding-bytes",
            "delta-with-some-padding-bytes",
            "epsilon-with-some-padding-bytes",
        ])
        .to_string();
        let plain = "this is not json".to_string();
        let mut req = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::user("hi")),
            documents: vec![
                Document {
                    id: "json".into(),
                    text: big_array,
                    additional_props: Default::default(),
                },
                Document {
                    id: "plain".into(),
                    text: plain.clone(),
                    additional_props: Default::default(),
                },
            ],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };
        let stats = p
            .compress(&mut req, &TokudoOptions::new())
            .unwrap()
            .expect("touched");
        assert!(
            stats.output_tokens < stats.input_tokens,
            "expected shrink: {} -> {}",
            stats.input_tokens,
            stats.output_tokens
        );
        let parsed: Value = serde_json::from_str(&req.documents[0].text).unwrap();
        let items = parsed.as_array().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[2], Value::String("...".into()));
        assert_eq!(req.documents[1].text, plain);
    }
}
