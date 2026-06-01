//! Foyer-backed exact-match cache backend.

use crate::error::Result;
use crate::traits::{Cache, CachedEntry};

/// Configuration for [`FoyerCache`].
#[derive(Debug, Clone)]
pub struct FoyerCacheConfig {
    /// Maximum weighted capacity for the Foyer in-memory cache.
    ///
    /// The default Foyer weighter counts each entry as `1`, so this is an
    /// entry count unless a future adapter exposes custom weighting.
    pub capacity: usize,
    /// Number of cache shards used by Foyer.
    pub shards: usize,
    /// Metrics/name prefix used by Foyer.
    pub name: String,
}

impl Default for FoyerCacheConfig {
    fn default() -> Self {
        Self {
            capacity: 1024,
            shards: 8,
            name: "rig-tokudo".into(),
        }
    }
}

/// Exact-match cache backend backed by Foyer's in-memory cache.
///
/// `FoyerCache` implements Tokudo's [`Cache`] trait without changing the
/// request/response cache contract. It stores [`CachedEntry`] values under
/// the cache key produced by the configured [`crate::CacheKey`]. Entry expiry
/// still follows [`CachedEntry::expires_at_secs`]; Foyer's own eviction policy
/// is treated as a normal cache miss.
///
/// # Example
///
/// ```no_run
/// use rig_tokudo::{Cache, CachedEntry, FoyerCache};
///
/// # async fn demo() -> rig_tokudo::Result<()> {
/// let cache = FoyerCache::new();
/// cache
///     .put(
///         "request-key",
///         CachedEntry {
///             response: serde_json::json!({"ok": true}),
///             expires_at_secs: None,
///             similarity: None,
///             source_id: "request-key".into(),
///         },
///     )
///     .await?;
/// assert!(cache.get("request-key").await?.is_some());
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct FoyerCache {
    config: FoyerCacheConfig,
    inner: foyer_memory::Cache<String, CachedEntry>,
}

impl FoyerCache {
    /// Construct a cache with [`FoyerCacheConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(FoyerCacheConfig::default())
    }

    /// Construct a cache with explicit Foyer settings.
    #[must_use]
    pub fn with_config(config: FoyerCacheConfig) -> Self {
        let mut config = config;
        config.shards = config.shards.max(1);
        let inner = foyer_memory::Cache::<String, CachedEntry>::builder(config.capacity)
            .with_name(config.name.clone())
            .with_shards(config.shards)
            .build();
        Self { config, inner }
    }

    /// Number of entries currently tracked by Foyer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.entries()
    }

    /// Returns `true` when the cache has no tracked entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Configuration used to build this cache.
    #[must_use]
    pub fn config(&self) -> &FoyerCacheConfig {
        &self.config
    }
}

impl Default for FoyerCache {
    fn default() -> Self {
        Self::new()
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn is_expired(entry: &CachedEntry, now: u64) -> bool {
    matches!(entry.expires_at_secs, Some(deadline) if now >= deadline)
}

impl Cache for FoyerCache {
    async fn get(&self, key: &str) -> Result<Option<CachedEntry>> {
        let Some(entry) = self.inner.get(key) else {
            return Ok(None);
        };
        let value = (*entry).clone();
        if is_expired(&value, now_secs()) {
            self.inner.remove(key);
            return Ok(None);
        }
        Ok(Some(value))
    }

    async fn put(&self, key: &str, entry: CachedEntry) -> Result<()> {
        if self.config.capacity == 0 {
            return Ok(());
        }
        self.inner.insert(key.to_string(), entry);
        Ok(())
    }
}
