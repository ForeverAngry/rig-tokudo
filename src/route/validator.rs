//! Concrete [`Validator`] implementations used by cascade routers.
//!
//! All validators operate on the JSON-serialized provider response. The
//! cascade serializes `M::Response` via `serde_json::to_value` before
//! invoking `validate`, so these implementations only see a `Value` and
//! never the raw provider type.
//!
//! Three patterns are covered:
//!
//! - [`LengthValidator`] — reject responses whose textual payload is below
//!   a configurable character floor.
//! - [`RegexValidator`] — accept only when the textual payload matches a
//!   compiled regex.
//! - [`ConfidenceValidator`] — accept only when a numeric JSON field
//!   (configurable JSON Pointer) meets a minimum threshold.

use regex::Regex;
use serde_json::Value;

use crate::error::{Result, TokudoError};
use crate::traits::Validator;

/// Reject responses whose extracted text falls below a character minimum.
///
/// Text extraction walks the JSON tree and concatenates every string leaf;
/// this is intentionally schema-agnostic so it works across providers
/// without hard-coding the OpenAI/Anthropic shapes.
#[derive(Debug, Clone, Copy)]
pub struct LengthValidator {
    min_chars: usize,
}

impl LengthValidator {
    /// Construct a validator that requires at least `min_chars` characters
    /// of cumulative text in the response.
    #[must_use]
    pub fn new(min_chars: usize) -> Self {
        Self { min_chars }
    }
}

impl Validator for LengthValidator {
    fn validate(&self, response: &Value) -> Result<bool> {
        let mut total = 0usize;
        collect_chars(response, &mut total, self.min_chars);
        Ok(total >= self.min_chars)
    }
}

/// Accept only responses whose extracted text matches a compiled regex.
///
/// Useful for enforcing structured outputs from cheap leg replies (e.g.
/// JSON-only, ISO-8601 dates, citation markers).
#[derive(Debug, Clone)]
pub struct RegexValidator {
    pattern: Regex,
}

impl RegexValidator {
    /// Compile `pattern` and return a validator. Errors surface as
    /// [`TokudoError::Config`].
    pub fn new(pattern: &str) -> Result<Self> {
        let compiled = Regex::new(pattern)
            .map_err(|e| TokudoError::Config(format!("invalid regex {pattern:?}: {e}")))?;
        Ok(Self { pattern: compiled })
    }
}

impl Validator for RegexValidator {
    fn validate(&self, response: &Value) -> Result<bool> {
        let mut buf = String::new();
        collect_text(response, &mut buf);
        Ok(self.pattern.is_match(&buf))
    }
}

/// Accept only responses whose numeric confidence field meets a threshold.
///
/// The validator reads a JSON Pointer (RFC 6901) — for example
/// `/confidence` or `/usage/confidence` — and accepts when the pointed-at
/// value is a number at least equal to `min`. A missing or non-numeric
/// pointer counts as a rejection.
#[derive(Debug, Clone)]
pub struct ConfidenceValidator {
    pointer: String,
    min: f64,
}

impl ConfidenceValidator {
    /// Construct a validator pointing at `pointer` (a JSON Pointer string)
    /// requiring at least `min` confidence.
    #[must_use]
    pub fn new(pointer: impl Into<String>, min: f64) -> Self {
        Self {
            pointer: pointer.into(),
            min,
        }
    }
}

impl Validator for ConfidenceValidator {
    fn validate(&self, response: &Value) -> Result<bool> {
        let observed = response
            .pointer(&self.pointer)
            .and_then(Value::as_f64)
            .unwrap_or(f64::NEG_INFINITY);
        Ok(observed >= self.min)
    }
}

/// Walk the JSON tree and accumulate string-leaf characters into `total`,
/// stopping early once `cap` is reached.
fn collect_chars(value: &Value, total: &mut usize, cap: usize) {
    if *total >= cap {
        return;
    }
    match value {
        Value::String(s) => {
            *total = total.saturating_add(s.chars().count());
        }
        Value::Array(items) => {
            for item in items {
                if *total >= cap {
                    return;
                }
                collect_chars(item, total, cap);
            }
        }
        Value::Object(map) => {
            for (_, v) in map {
                if *total >= cap {
                    return;
                }
                collect_chars(v, total, cap);
            }
        }
        _ => {}
    }
}

/// Walk the JSON tree and append every string leaf to `buf`, separated by
/// single spaces. Used by [`RegexValidator`] for pattern matching.
fn collect_text(value: &Value, buf: &mut String) {
    match value {
        Value::String(s) => {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(s);
        }
        Value::Array(items) => {
            for item in items {
                collect_text(item, buf);
            }
        }
        Value::Object(map) => {
            for (_, v) in map {
                collect_text(v, buf);
            }
        }
        _ => {}
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
    use serde_json::json;

    #[test]
    fn length_validator_accepts_when_text_meets_floor() {
        let v = LengthValidator::new(10);
        let resp = json!({"text": "hello world!"});
        assert!(v.validate(&resp).unwrap());
    }

    #[test]
    fn length_validator_rejects_short_text() {
        let v = LengthValidator::new(20);
        let resp = json!({"text": "tiny"});
        assert!(!v.validate(&resp).unwrap());
    }

    #[test]
    fn length_validator_counts_across_nested_fields() {
        let v = LengthValidator::new(15);
        let resp = json!({
            "choices": [
                {"text": "hello "},
                {"text": "world "},
                {"text": "today"},
            ]
        });
        assert!(v.validate(&resp).unwrap());
    }

    #[test]
    fn regex_validator_matches_pattern() {
        let v = RegexValidator::new(r"\d{4}-\d{2}-\d{2}").unwrap();
        let resp = json!({"answer": "the date is 2026-05-22"});
        assert!(v.validate(&resp).unwrap());
    }

    #[test]
    fn regex_validator_rejects_when_no_match() {
        let v = RegexValidator::new(r"^ONLY$").unwrap();
        let resp = json!({"answer": "something else"});
        assert!(!v.validate(&resp).unwrap());
    }

    #[test]
    fn regex_validator_returns_config_error_on_bad_pattern() {
        let err = RegexValidator::new("(unclosed").unwrap_err();
        assert!(matches!(err, TokudoError::Config(_)));
    }

    #[test]
    fn confidence_validator_accepts_when_above_threshold() {
        let v = ConfidenceValidator::new("/confidence", 0.7);
        let resp = json!({"confidence": 0.92, "text": "ok"});
        assert!(v.validate(&resp).unwrap());
    }

    #[test]
    fn confidence_validator_rejects_when_below_threshold() {
        let v = ConfidenceValidator::new("/confidence", 0.7);
        let resp = json!({"confidence": 0.42});
        assert!(!v.validate(&resp).unwrap());
    }

    #[test]
    fn confidence_validator_rejects_when_pointer_missing() {
        let v = ConfidenceValidator::new("/usage/confidence", 0.0);
        let resp = json!({"text": "no usage block"});
        assert!(!v.validate(&resp).unwrap());
    }

    #[test]
    fn confidence_validator_walks_nested_pointer() {
        let v = ConfidenceValidator::new("/usage/score", 0.5);
        let resp = json!({"usage": {"score": 0.51}});
        assert!(v.validate(&resp).unwrap());
    }
}
