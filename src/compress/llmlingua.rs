//! LLMLingua-inspired deterministic prompt pruning.
//!
//! This module intentionally does not depend on a tokenizer or transformer
//! stack. Hosts can tune the pruning policy with salient/preserved terms and
//! use it anywhere a [`crate::Compressor`] is accepted.

use std::cmp::Ordering;
use std::collections::HashSet;

use rig::completion::message::{Text, UserContent};
use rig::completion::{AssistantContent, CompletionRequest, Message};

use crate::error::Result;
use crate::options::TokudoOptions;
use crate::traits::{CompressionStats, Compressor};

const OMITTED: &str = "[...]";

/// Configuration for [`LlmlinguaCompressor`].
#[derive(Debug, Clone)]
pub struct LlmlinguaConfig {
    /// Desired output/input token ratio. Values are clamped to `0.05..=1.0`.
    pub target_ratio: f32,
    /// Minimum word-token count before pruning is attempted.
    pub min_tokens: usize,
    /// Normalized terms that must always be retained.
    pub preserve_terms: HashSet<String>,
    /// Normalized terms that receive a strong salience boost.
    pub salient_terms: HashSet<String>,
}

impl Default for LlmlinguaConfig {
    fn default() -> Self {
        Self {
            target_ratio: 0.60,
            min_tokens: 64,
            preserve_terms: HashSet::new(),
            salient_terms: HashSet::new(),
        }
    }
}

impl LlmlinguaConfig {
    /// Add a normalized or raw term that should always be retained.
    #[must_use]
    pub fn with_preserve_term(mut self, term: impl AsRef<str>) -> Self {
        if let Some(normalized) = normalize(term.as_ref()) {
            self.preserve_terms.insert(normalized);
        }
        self
    }

    /// Add a normalized or raw term that should receive a salience boost.
    #[must_use]
    pub fn with_salient_term(mut self, term: impl AsRef<str>) -> Self {
        if let Some(normalized) = normalize(term.as_ref()) {
            self.salient_terms.insert(normalized);
        }
        self
    }
}

/// Dependency-free, LLMLingua-inspired token pruner.
///
/// The compressor scores word tokens by configurable salience, preserve terms,
/// position, and simple lexical signals, then keeps the highest-scoring tokens
/// in original order. Removed spans collapse to `[...]` so downstream models can
/// see where context was elided.
#[derive(Debug, Clone)]
pub struct LlmlinguaCompressor {
    config: LlmlinguaConfig,
}

impl LlmlinguaCompressor {
    /// Construct a compressor from explicit config.
    #[must_use]
    pub fn new(config: LlmlinguaConfig) -> Self {
        Self { config }
    }

    /// Construct with default pruning knobs.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(LlmlinguaConfig::default())
    }

    /// Prune a standalone string and return `(input_tokens, output_tokens)`.
    pub fn compress_text(&self, text: &mut String) -> Option<(usize, usize)> {
        let tokens = tokenize(text);
        let total_words = tokens
            .iter()
            .filter(|token| token.ordinal.is_some())
            .count();
        if total_words < self.config.min_tokens {
            return None;
        }

        let target_ratio = self.config.target_ratio.clamp(0.05, 1.0);
        let target_words = ((total_words as f32) * target_ratio).ceil() as usize;
        if target_words >= total_words {
            return None;
        }

        let keep_ordinals = self.keep_ordinals(&tokens, total_words, target_words);
        let compressed = render_compressed(&tokens, &keep_ordinals);
        let output_words = compressed.split_whitespace().count();
        if output_words >= total_words || compressed == *text {
            return None;
        }

        *text = compressed;
        Some((total_words, output_words))
    }

    fn keep_ordinals(
        &self,
        tokens: &[Token],
        total_words: usize,
        target_words: usize,
    ) -> HashSet<usize> {
        let mut ranked = Vec::new();
        for token in tokens {
            if let (Some(ordinal), Some(normalized)) = (token.ordinal, token.normalized.as_ref()) {
                ranked.push(RankedToken {
                    ordinal,
                    score: self.score_token(&token.text, normalized, ordinal, total_words),
                });
            }
        }
        ranked.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.ordinal.cmp(&right.ordinal))
        });

        ranked
            .into_iter()
            .take(target_words)
            .map(|ranked_token| ranked_token.ordinal)
            .collect()
    }

    fn score_token(
        &self,
        original: &str,
        normalized: &str,
        ordinal: usize,
        total_words: usize,
    ) -> f32 {
        if self.config.preserve_terms.contains(normalized) {
            return 10_000.0;
        }

        let mut score = 1.0;
        if self.config.salient_terms.contains(normalized) {
            score += 100.0;
        }
        if is_structural_word(normalized) {
            score += 12.0;
        }
        if normalized.chars().count() >= 7 {
            score += 3.0;
        }
        if original.chars().any(char::is_numeric) {
            score += 2.0;
        }
        if original.chars().next().is_some_and(char::is_uppercase) {
            score += 1.0;
        }
        if ordinal < 12 || total_words.saturating_sub(ordinal) <= 12 {
            score += 4.0;
        }
        if is_stopword(normalized) {
            score -= 2.0;
        }
        score
    }
}

impl Compressor for LlmlinguaCompressor {
    fn compress(
        &self,
        request: &mut CompletionRequest,
        _options: &TokudoOptions,
    ) -> Result<Option<CompressionStats>> {
        let mut input_tokens = 0usize;
        let mut output_tokens = 0usize;
        let mut touched = false;

        if let Some(preamble) = request.preamble.as_mut()
            && let Some((input, output)) = self.compress_text(preamble)
        {
            input_tokens = input_tokens.saturating_add(input);
            output_tokens = output_tokens.saturating_add(output);
            touched = true;
        }

        for message in request.chat_history.iter_mut() {
            if let Some((input, output)) = self.compress_message(message) {
                input_tokens = input_tokens.saturating_add(input);
                output_tokens = output_tokens.saturating_add(output);
                touched = true;
            }
        }

        for document in &mut request.documents {
            if let Some((input, output)) = self.compress_text(&mut document.text) {
                input_tokens = input_tokens.saturating_add(input);
                output_tokens = output_tokens.saturating_add(output);
                touched = true;
            }
        }

        if !touched {
            return Ok(None);
        }

        Ok(Some(CompressionStats {
            input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
            output_tokens: u32::try_from(output_tokens).unwrap_or(u32::MAX),
        }))
    }
}

impl LlmlinguaCompressor {
    fn compress_message(&self, message: &mut Message) -> Option<(usize, usize)> {
        match message {
            Message::System { content } => self.compress_text(content),
            Message::User { content } => compress_user_content(self, content.iter_mut()),
            Message::Assistant { content, .. } => {
                compress_assistant_content(self, content.iter_mut())
            }
        }
    }
}

fn compress_user_content<'a>(
    compressor: &LlmlinguaCompressor,
    content: impl Iterator<Item = &'a mut UserContent>,
) -> Option<(usize, usize)> {
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;
    let mut touched = false;
    for item in content {
        if let UserContent::Text(Text { text, .. }) = item
            && let Some((input, output)) = compressor.compress_text(text)
        {
            input_tokens = input_tokens.saturating_add(input);
            output_tokens = output_tokens.saturating_add(output);
            touched = true;
        }
    }
    touched.then_some((input_tokens, output_tokens))
}

fn compress_assistant_content<'a>(
    compressor: &LlmlinguaCompressor,
    content: impl Iterator<Item = &'a mut AssistantContent>,
) -> Option<(usize, usize)> {
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;
    let mut touched = false;
    for item in content {
        if let AssistantContent::Text(Text { text, .. }) = item
            && let Some((input, output)) = compressor.compress_text(text)
        {
            input_tokens = input_tokens.saturating_add(input);
            output_tokens = output_tokens.saturating_add(output);
            touched = true;
        }
    }
    touched.then_some((input_tokens, output_tokens))
}

#[derive(Debug)]
struct Token {
    text: String,
    normalized: Option<String>,
    ordinal: Option<usize>,
}

#[derive(Debug)]
struct RankedToken {
    ordinal: usize,
    score: f32,
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut ordinal = 0usize;
    text.split_whitespace()
        .map(|raw| {
            let normalized = normalize(raw);
            let token_ordinal = normalized.as_ref().map(|_| {
                let current = ordinal;
                ordinal = ordinal.saturating_add(1);
                current
            });
            Token {
                text: raw.to_string(),
                normalized,
                ordinal: token_ordinal,
            }
        })
        .collect()
}

fn render_compressed(tokens: &[Token], keep_ordinals: &HashSet<usize>) -> String {
    let mut output = Vec::new();
    let mut eliding = false;
    for token in tokens {
        match token.ordinal {
            Some(ordinal) if keep_ordinals.contains(&ordinal) => {
                output.push(token.text.clone());
                eliding = false;
            }
            Some(_) if !eliding => {
                output.push(OMITTED.to_string());
                eliding = true;
            }
            Some(_) => {}
            None => {
                output.push(token.text.clone());
                eliding = false;
            }
        }
    }
    output.join(" ")
}

fn normalize(raw: &str) -> Option<String> {
    let normalized = raw
        .chars()
        .filter(|character| character.is_alphanumeric() || *character == '_' || *character == '-')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn is_structural_word(normalized: &str) -> bool {
    matches!(
        normalized,
        "must"
            | "never"
            | "always"
            | "only"
            | "except"
            | "because"
            | "therefore"
            | "error"
            | "warning"
            | "security"
            | "policy"
            | "schema"
            | "json"
            | "sql"
            | "api"
    )
}

fn is_stopword(normalized: &str) -> bool {
    matches!(
        normalized,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "to"
            | "with"
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use rig::OneOrMany;
    use rig::completion::{Document, Message};

    fn long_text() -> String {
        "The system must preserve security policy schema details while trimming ordinary filler words from this lengthy prompt. The request includes account identifiers, JSON constraints, deadline warnings, audit requirements, and explicit never expose secret values instructions. Extra background prose repeats lower-value phrasing about the same project context, the same project context, and the same project context so pruning has safe room to work."
            .to_string()
    }

    #[test]
    fn compress_text_keeps_preserved_and_salient_terms() {
        let compressor = LlmlinguaCompressor::new(LlmlinguaConfig {
            target_ratio: 0.35,
            min_tokens: 12,
            preserve_terms: HashSet::from(["secret".to_string()]),
            salient_terms: HashSet::from(["json".to_string(), "audit".to_string()]),
        });
        let mut text = long_text();
        let before = text.split_whitespace().count();
        let stats = compressor.compress_text(&mut text).unwrap();

        assert_eq!(stats.0, before);
        assert!(stats.1 < stats.0);
        assert!(text.contains("secret"));
        assert!(text.contains("JSON"));
        assert!(text.contains("audit"));
        assert!(text.contains(OMITTED));
    }

    #[test]
    fn short_text_is_left_unchanged() {
        let compressor = LlmlinguaCompressor::new(LlmlinguaConfig {
            min_tokens: 100,
            ..LlmlinguaConfig::default()
        });
        let mut text = "short prompt".to_string();

        assert!(compressor.compress_text(&mut text).is_none());
        assert_eq!(text, "short prompt");
    }

    #[test]
    fn compressor_mutates_request_text_surfaces() {
        let compressor = LlmlinguaCompressor::new(LlmlinguaConfig {
            target_ratio: 0.40,
            min_tokens: 12,
            preserve_terms: HashSet::from(["secret".to_string()]),
            salient_terms: HashSet::new(),
        });
        let mut request = CompletionRequest {
            model: Some("openai:gpt-4o-mini".to_string()),
            preamble: Some(long_text()),
            chat_history: OneOrMany::one(Message::user(long_text())),
            documents: vec![Document {
                id: "doc".to_string(),
                text: long_text(),
                additional_props: std::collections::HashMap::new(),
            }],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let stats = compressor
            .compress(&mut request, &TokudoOptions::default())
            .unwrap()
            .unwrap();

        assert!(stats.output_tokens < stats.input_tokens);
        assert!(
            request
                .preamble
                .as_deref()
                .is_some_and(|text| text.contains(OMITTED))
        );
        assert!(
            request
                .documents
                .iter()
                .any(|doc| doc.text.contains(OMITTED))
        );
    }
}
