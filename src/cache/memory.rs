//! Bounded in-process cache backend.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Result, TokudoError};
use crate::traits::{Cache, CachedEntry};

/// Configuration for [`InMemoryCache`].
#[derive(Debug, Clone, Copy)]
pub struct InMemoryCacheConfig {
    /// Maximum number of live (non-expired) entries before FIFO eviction.
    ///
    /// `None` disables the bound. A bound of `0` means the cache never
    /// stores anything.
    pub max_entries: Option<usize>,
    /// Default TTL applied when [`CachedEntry::expires_at_secs`] is `None`.
    ///
    /// `None` means entries live until evicted by the capacity bound.
    pub default_ttl_secs: Option<u64>,
}

impl Default for InMemoryCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: Some(1024),
            default_ttl_secs: None,
        }
    }
}

/// In-process cache backed by a `HashMap` plus a FIFO insertion queue for
/// capacity-bounded eviction.
///
/// `InMemoryCache` is intentionally minimal: O(1) `get` and amortized O(1)
/// `put`, no background tasks, no LRU promotion. Expired entries are evicted
/// lazily on `get` and during `put` when the capacity is exceeded.
///
/// # Example
///
/// ```
/// use rig_tokudo::{InMemoryCache, Cache, CachedEntry};
///
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let cache = InMemoryCache::new();
/// let entry = CachedEntry {
///     response: serde_json::json!({ "ok": true }),
///     expires_at_secs: None,
///     similarity: None,
///     source_id: "abc".into(),
/// };
/// cache.put("k", entry).await?;
/// assert!(cache.get("k").await?.is_some());
/// # Ok(()) }
/// ```
pub struct InMemoryCache {
    config: InMemoryCacheConfig,
    inner: Mutex<Inner>,
}

struct Inner {
    map: HashMap<String, CachedEntry>,
    order: VecDeque<String>,
}

impl InMemoryCache {
    /// Construct with default configuration: 1024-entry FIFO bound, no TTL.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(InMemoryCacheConfig::default())
    }

    /// Construct with an explicit [`InMemoryCacheConfig`].
    #[must_use]
    pub fn with_config(config: InMemoryCacheConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    /// Number of live (not yet evicted) entries. Note: may include expired
    /// entries that haven't been reaped yet.
    pub fn len(&self) -> Result<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| TokudoError::Cache(format!("lock poisoned: {e}")))?;
        Ok(guard.map.len())
    }

    /// Returns `true` when there are no live entries.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn is_expired(entry: &CachedEntry, now: u64) -> bool {
    matches!(entry.expires_at_secs, Some(deadline) if now >= deadline)
}

impl Cache for InMemoryCache {
    async fn get(&self, key: &str) -> Result<Option<CachedEntry>> {
        let now = now_secs();
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| TokudoError::Cache(format!("lock poisoned: {e}")))?;

        let Some(entry) = guard.map.get(key).cloned() else {
            return Ok(None);
        };
        if is_expired(&entry, now) {
            guard.map.remove(key);
            guard.order.retain(|k| k != key);
            return Ok(None);
        }
        Ok(Some(entry))
    }

    async fn put(&self, key: &str, mut entry: CachedEntry) -> Result<()> {
        // Apply default TTL when caller didn't supply one.
        if entry.expires_at_secs.is_none()
            && let Some(default_ttl) = self.config.default_ttl_secs
        {
            entry.expires_at_secs = Some(now_secs().saturating_add(default_ttl));
        }

        let mut guard = self
            .inner
            .lock()
            .map_err(|e| TokudoError::Cache(format!("lock poisoned: {e}")))?;

        // Capacity 0 means "store nothing".
        if matches!(self.config.max_entries, Some(0)) {
            return Ok(());
        }

        let is_new = !guard.map.contains_key(key);
        guard.map.insert(key.to_string(), entry);
        if is_new {
            guard.order.push_back(key.to_string());
        }

        // Evict oldest entries until under the bound.
        if let Some(max) = self.config.max_entries {
            while guard.map.len() > max
                && let Some(oldest) = guard.order.pop_front()
            {
                guard.map.remove(&oldest);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn entry(id: &str, expires_at_secs: Option<u64>) -> CachedEntry {
        CachedEntry {
            response: serde_json::json!({ "id": id }),
            expires_at_secs,
            similarity: None,
            source_id: id.into(),
        }
    }

    #[tokio::test]
    async fn miss_then_hit() {
        let cache = InMemoryCache::new();
        assert!(cache.get("a").await.unwrap().is_none());
        cache.put("a", entry("a", None)).await.unwrap();
        let got = cache.get("a").await.unwrap().unwrap();
        assert_eq!(got.source_id, "a");
    }

    #[tokio::test]
    async fn ttl_expiry_evicts_on_get() {
        let cache = InMemoryCache::new();
        // Already-expired entry.
        cache.put("e", entry("e", Some(0))).await.unwrap();
        assert!(cache.get("e").await.unwrap().is_none());
        assert!(cache.is_empty().unwrap());
    }

    #[tokio::test]
    async fn capacity_evicts_oldest() {
        let cache = InMemoryCache::with_config(InMemoryCacheConfig {
            max_entries: Some(2),
            default_ttl_secs: None,
        });
        cache.put("a", entry("a", None)).await.unwrap();
        cache.put("b", entry("b", None)).await.unwrap();
        cache.put("c", entry("c", None)).await.unwrap();
        assert!(
            cache.get("a").await.unwrap().is_none(),
            "a should be evicted"
        );
        assert!(cache.get("b").await.unwrap().is_some());
        assert!(cache.get("c").await.unwrap().is_some());
        assert_eq!(cache.len().unwrap(), 2);
    }

    #[tokio::test]
    async fn capacity_zero_stores_nothing() {
        let cache = InMemoryCache::with_config(InMemoryCacheConfig {
            max_entries: Some(0),
            default_ttl_secs: None,
        });
        cache.put("a", entry("a", None)).await.unwrap();
        assert!(cache.get("a").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn default_ttl_applies_when_unset() {
        let cache = InMemoryCache::with_config(InMemoryCacheConfig {
            max_entries: Some(8),
            default_ttl_secs: Some(3600),
        });
        cache.put("a", entry("a", None)).await.unwrap();
        let got = cache.get("a").await.unwrap().unwrap();
        let deadline = got.expires_at_secs.unwrap();
        assert!(deadline >= now_secs());
    }

    #[tokio::test]
    async fn overwrite_does_not_grow_order() {
        let cache = InMemoryCache::with_config(InMemoryCacheConfig {
            max_entries: Some(2),
            default_ttl_secs: None,
        });
        cache.put("a", entry("a", None)).await.unwrap();
        cache.put("a", entry("a-updated", None)).await.unwrap();
        cache.put("b", entry("b", None)).await.unwrap();
        // Both should still be present; overwrite shouldn't have pushed `a` out of order.
        assert!(cache.get("a").await.unwrap().is_some());
        assert!(cache.get("b").await.unwrap().is_some());
    }
}
