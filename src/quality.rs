//! Pure Rust quality helpers for savings reports.
//!
//! The first metric is ROUGE-L F1, a longest-common-subsequence similarity
//! score useful for cheap regression checks between baseline and optimized
//! text outputs.

/// ROUGE-L precision/recall/F1 result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RougeLScore {
    /// Longest-common-subsequence length divided by candidate length.
    pub precision: f64,
    /// Longest-common-subsequence length divided by reference length.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
}

/// Compute ROUGE-L over whitespace-tokenized text.
#[must_use]
pub fn rouge_l(candidate: &str, reference: &str) -> RougeLScore {
    let candidate_tokens = tokenize(candidate);
    let reference_tokens = tokenize(reference);
    if candidate_tokens.is_empty() && reference_tokens.is_empty() {
        return RougeLScore {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        };
    }
    if candidate_tokens.is_empty() || reference_tokens.is_empty() {
        return RougeLScore {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        };
    }

    let lcs = lcs_len(&candidate_tokens, &reference_tokens) as f64;
    let precision = lcs / candidate_tokens.len() as f64;
    let recall = lcs / reference_tokens.len() as f64;
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    RougeLScore {
        precision,
        recall,
        f1,
    }
}

/// Compute mean ROUGE-L F1 across `(candidate, reference)` pairs.
#[must_use]
pub fn mean_rouge_l_f1<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Option<f64> {
    let mut count = 0_u64;
    let mut sum = 0.0;
    for (candidate, reference) in pairs {
        count = count.saturating_add(1);
        sum += rouge_l(candidate, reference).f1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect()
}

fn lcs_len(candidate: &[String], reference: &[String]) -> usize {
    let mut previous = vec![0_usize; reference.len().saturating_add(1)];
    for candidate_token in candidate {
        let mut current = vec![0_usize; reference.len().saturating_add(1)];
        let mut diagonal = 0_usize;
        let mut left = 0_usize;
        for (idx, reference_token) in reference.iter().enumerate() {
            let up = previous.get(idx.saturating_add(1)).copied().unwrap_or(0);
            let value = if candidate_token == reference_token {
                diagonal.saturating_add(1)
            } else {
                up.max(left)
            };
            if let Some(slot) = current.get_mut(idx.saturating_add(1)) {
                *slot = value;
            }
            diagonal = up;
            left = value;
        }
        previous = current;
    }
    previous.last().copied().unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scores_one() {
        let score = rouge_l("alpha beta", "alpha beta");
        assert!((score.f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn partial_match_scores_between_zero_and_one() {
        let score = rouge_l("alpha gamma", "alpha beta gamma");
        assert!(score.f1 > 0.0);
        assert!(score.f1 < 1.0);
    }

    #[test]
    fn mean_returns_none_for_empty_input() {
        let pairs: Vec<(&str, &str)> = Vec::new();
        assert!(mean_rouge_l_f1(pairs).is_none());
    }
}
