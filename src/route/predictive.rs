//! Train-free predictive router.
//!
//! The `route-predictive` feature provides a tiny calibration-set router that
//! predicts whether a request should try the cheap leg or skip directly to the
//! strong leg. It is deliberately pure-data and dependency-free: hosts provide
//! representative prompt examples labeled with the route that passed their
//! validator, and the router uses lexical overlap to choose a leg.

use std::collections::HashSet;

use rig::completion::message::{Text, UserContent};
use rig::completion::{AssistantContent, CompletionRequest, Message};

use crate::options::TokudoOptions;
use crate::provenance::RouterChoice;
use crate::traits::Router;

/// One calibration prompt and the route that should answer similar prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictiveRouteExample {
    /// Representative prompt text.
    pub text: String,
    /// Route that historically passed validation for this kind of prompt.
    pub choice: RouterChoice,
}

impl PredictiveRouteExample {
    /// Build an example where the cheap leg is expected to pass validation.
    #[must_use]
    pub fn cheap(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            choice: RouterChoice::Cheap,
        }
    }

    /// Build an example where the strong leg should be used directly.
    #[must_use]
    pub fn strong(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            choice: RouterChoice::Strong,
        }
    }
}

/// Configuration for [`PredictiveRouter`].
#[derive(Debug, Clone, PartialEq)]
pub struct PredictiveRouterConfig {
    /// Minimum lexical Jaccard similarity required to trust a calibration
    /// example. Below this threshold the router chooses `Strong`.
    pub min_similarity: f32,
}

impl Default for PredictiveRouterConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.20,
        }
    }
}

/// Train-free router backed by a small calibration set.
///
/// The router extracts normalized word tokens from the incoming request,
/// compares them with each calibration example using Jaccard similarity, and
/// returns the closest example's route when it clears
/// [`PredictiveRouterConfig::min_similarity`]. Unknown prompts route to
/// [`RouterChoice::Strong`] as the conservative default.
#[derive(Debug, Clone)]
pub struct PredictiveRouter {
    examples: Vec<PredictiveRouteExample>,
    config: PredictiveRouterConfig,
}

impl PredictiveRouter {
    /// Construct with default config.
    #[must_use]
    pub fn new(examples: impl IntoIterator<Item = PredictiveRouteExample>) -> Self {
        Self {
            examples: examples.into_iter().collect(),
            config: PredictiveRouterConfig::default(),
        }
    }

    /// Replace the router config.
    #[must_use]
    pub fn with_config(mut self, config: PredictiveRouterConfig) -> Self {
        self.config = config;
        self
    }

    /// Borrow the calibration examples.
    #[must_use]
    pub fn examples(&self) -> &[PredictiveRouteExample] {
        &self.examples
    }

    fn predict(&self, request: &CompletionRequest) -> RouterChoice {
        if self.examples.is_empty() {
            return RouterChoice::PassThrough;
        }

        let request_tokens = request_tokens(request);
        if request_tokens.is_empty() {
            return RouterChoice::Strong;
        }

        let mut best_choice = RouterChoice::Strong;
        let mut best_similarity = 0.0f32;
        for example in &self.examples {
            let example_tokens = normalize_tokens(&example.text);
            let similarity = jaccard(&request_tokens, &example_tokens);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_choice = match example.choice {
                    RouterChoice::Cheap => RouterChoice::Cheap,
                    RouterChoice::Strong => RouterChoice::Strong,
                    RouterChoice::CacheHit | RouterChoice::PassThrough => RouterChoice::Strong,
                };
            }
        }

        if best_similarity >= self.config.min_similarity {
            best_choice
        } else {
            RouterChoice::Strong
        }
    }
}

impl Router for PredictiveRouter {
    fn route(&self, request: &CompletionRequest, _options: &TokudoOptions) -> RouterChoice {
        self.predict(request)
    }
}

fn request_tokens(request: &CompletionRequest) -> HashSet<String> {
    let mut text = String::new();
    if let Some(preamble) = &request.preamble {
        text.push_str(preamble);
        text.push(' ');
    }
    for message in request.chat_history.iter() {
        collect_message_text(message, &mut text);
        text.push(' ');
    }
    for document in &request.documents {
        text.push_str(&document.text);
        text.push(' ');
    }
    normalize_tokens(&text)
}

fn collect_message_text(message: &Message, output: &mut String) {
    match message {
        Message::System { content } => output.push_str(content),
        Message::User { content } => {
            for item in content.iter() {
                if let UserContent::Text(Text { text, .. }) = item {
                    output.push_str(text);
                    output.push(' ');
                }
            }
        }
        Message::Assistant { content, .. } => {
            for item in content.iter() {
                if let AssistantContent::Text(Text { text, .. }) = item {
                    output.push_str(text);
                    output.push(' ');
                }
            }
        }
    }
}

fn normalize_tokens(text: &str) -> HashSet<String> {
    text.split_whitespace().filter_map(normalize).collect()
}

fn normalize(raw: &str) -> Option<String> {
    let normalized = raw
        .chars()
        .filter(|character| character.is_alphanumeric() || *character == '_' || *character == '-')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if normalized.is_empty() || is_stopword(&normalized) {
        None
    } else {
        Some(normalized)
    }
}

fn jaccard(left: &HashSet<String>, right: &HashSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
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

    fn request(prompt: &str) -> CompletionRequest {
        CompletionRequest {
            model: Some("openai:gpt-4o-mini".to_string()),
            preamble: Some("route requests conservatively".to_string()),
            chat_history: OneOrMany::one(Message::user(prompt.to_string())),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    #[test]
    fn similar_cheap_example_routes_to_cheap() {
        let router = PredictiveRouter::new([
            PredictiveRouteExample::cheap("summarize changelog release notes briefly"),
            PredictiveRouteExample::strong("solve multi step algebra proof with citations"),
        ]);

        let choice = router.route(
            &request("please summarize these changelog notes briefly"),
            &TokudoOptions::default(),
        );

        assert_eq!(choice, RouterChoice::Cheap);
    }

    #[test]
    fn similar_strong_example_routes_to_strong() {
        let router = PredictiveRouter::new([
            PredictiveRouteExample::cheap("summarize changelog release notes briefly"),
            PredictiveRouteExample::strong("solve multi step algebra proof with citations"),
        ]);

        let choice = router.route(
            &request("solve this algebra proof and include citations"),
            &TokudoOptions::default(),
        );

        assert_eq!(choice, RouterChoice::Strong);
    }

    #[test]
    fn low_similarity_routes_to_strong() {
        let router = PredictiveRouter::new([PredictiveRouteExample::cheap(
            "summarize changelog release notes briefly",
        )])
        .with_config(PredictiveRouterConfig {
            min_similarity: 0.90,
        });

        let choice = router.route(
            &request("classify a security incident timeline"),
            &TokudoOptions::default(),
        );

        assert_eq!(choice, RouterChoice::Strong);
    }

    #[test]
    fn request_tokens_include_documents() {
        let router = PredictiveRouter::new([PredictiveRouteExample::cheap(
            "retrieval freshness conflict report",
        )]);
        let mut req = request("briefly summarize");
        req.documents.push(Document {
            id: "doc".to_string(),
            text: "freshness conflict report".to_string(),
            additional_props: std::collections::HashMap::new(),
        });

        let choice = router.route(&req, &TokudoOptions::default());

        assert_eq!(choice, RouterChoice::Cheap);
    }
}
