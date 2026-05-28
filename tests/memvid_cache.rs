//! Integration tests for the durable Memvid semantic cache.

#![cfg(feature = "cache-memvid")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::time::{SystemTime, UNIX_EPOCH};

use rig_memvid::MemvidStore;
use rig_tokudo::{Cache, CachedEntry, MemvidSemanticCache, MemvidSemanticCacheConfig};

fn temp_mv2_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("rig-tokudo-memvid-cache-{nanos}.mv2"))
}

fn cached_entry(source_id: &str) -> CachedEntry {
    CachedEntry {
        response: serde_json::json!({
            "choice": { "One": { "type": "text", "text": "cached" } },
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "reasoning_tokens": 0
            },
            "message_id": "cached-message",
            "raw_response": { "text": "cached" }
        }),
        expires_at_secs: None,
        similarity: None,
        source_id: source_id.into(),
    }
}

#[tokio::test]
async fn memvid_cache_writes_and_reads_cached_entry() {
    let path = temp_mv2_path();
    let store = MemvidStore::builder()
        .path(&path)
        .enable_lex()
        .create()
        .unwrap();
    let cache = MemvidSemanticCache::with_config(store, MemvidSemanticCacheConfig::default());

    cache
        .put("tokudocacheunique", cached_entry("entry-1"))
        .await
        .unwrap();
    assert_eq!(cache.store().frame_count().unwrap(), 1);
    let hit = cache
        .get("tokudocacheunique")
        .await
        .unwrap()
        .expect("cache hit");

    assert_eq!(hit.source_id, "entry-1");
    assert!(hit.similarity.is_some());
    assert_eq!(hit.response["message_id"], "cached-message");

    let _ = std::fs::remove_file(path);
}
