"""
Glass Cache - Structure Caching for Performance
================================================

Caches grammatical structures to avoid recomputation.
Useful when evaluating similar prompts multiple times.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta


class StructureCache:
    """
    LRU cache for grammatical structures.

    Stores StructurePattern objects keyed by text hash.
    Automatically expires old entries and limits size.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_hours: int = 24,
        persistent: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            max_size: Maximum number of cached items
            ttl_hours: Time-to-live in hours (0 = no expiration)
            persistent: Save cache to disk
            cache_dir: Directory for persistent cache
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours) if ttl_hours > 0 else None
        self.persistent = persistent
        self.cache_dir = cache_dir or Path.home() / ".cache" / "glass"

        # In-memory cache: {text_hash: (structure, timestamp)}
        self._cache: Dict[str, tuple] = {}

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }

        # Load persistent cache if enabled
        if self.persistent:
            self._load_from_disk()

    def _hash(self, text: str) -> str:
        """Compute cache key for text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def get(self, text: str):
        """
        Get cached structure for text.

        Returns:
            StructurePattern or None if not cached
        """
        key = self._hash(text)

        if key not in self._cache:
            self.stats["misses"] += 1
            return None

        structure, timestamp = self._cache[key]

        # Check expiration
        if self.ttl and datetime.now() - timestamp > self.ttl:
            del self._cache[key]
            self.stats["expired"] += 1
            self.stats["misses"] += 1
            return None

        # Update timestamp (LRU)
        self._cache[key] = (structure, datetime.now())
        self.stats["hits"] += 1

        return structure

    def put(self, text: str, structure):
        """
        Cache structure for text.

        Args:
            text: Input text
            structure: StructurePattern to cache
        """
        key = self._hash(text)

        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = (structure, datetime.now())

        # Persist if enabled
        if self.persistent:
            self._save_to_disk()

    def _evict_lru(self):
        """Evict least recently used item"""
        if not self._cache:
            return

        # Find oldest timestamp
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k][1]
        )

        del self._cache[oldest_key]
        self.stats["evictions"] += 1

    def clear(self):
        """Clear all cached entries"""
        self._cache.clear()
        if self.persistent:
            cache_file = self.cache_dir / "structure_cache.pkl"
            if cache_file.exists():
                cache_file.unlink()

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            **self.stats,
            "size": len(self._cache),
            "hit_rate": hit_rate,
        }

    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()

        print("Cache Statistics:")
        print(f"  Size: {stats['size']}/{self.max_size}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
        print(f"  Evictions: {stats['evictions']}")
        print(f"  Expired: {stats['expired']}")

    def _save_to_disk(self):
        """Save cache to disk"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "structure_cache.pkl"

            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)

        except Exception as e:
            # Fail silently for cache errors
            pass

    def _load_from_disk(self):
        """Load cache from disk"""
        try:
            cache_file = self.cache_dir / "structure_cache.pkl"

            if not cache_file.exists():
                return

            with open(cache_file, 'rb') as f:
                self._cache = pickle.load(f)

            # Remove expired entries
            now = datetime.now()
            expired_keys = [
                k for k, (_, ts) in self._cache.items()
                if self.ttl and now - ts > self.ttl
            ]

            for k in expired_keys:
                del self._cache[k]

        except Exception as e:
            # Fail silently for cache errors
            self._cache = {}


class CachedGrammaticalMapper:
    """
    GrammaticalMapper with built-in caching.

    Drop-in replacement for GrammaticalMapper that caches results.
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_size: int = 1000,
        cache_ttl_hours: int = 24,
    ):
        """
        Args:
            cache_enabled: Enable caching
            cache_size: Maximum cache size
            cache_ttl_hours: Cache expiration time
        """
        try:
            from .grammatical_mapper import GrammaticalMapper
        except ImportError:
            from grammatical_mapper import GrammaticalMapper

        self.mapper = GrammaticalMapper()
        self.cache_enabled = cache_enabled

        if self.cache_enabled:
            self.cache = StructureCache(
                max_size=cache_size,
                ttl_hours=cache_ttl_hours,
                persistent=False,  # In-memory only by default
            )

    def extract_structure(self, text: str):
        """
        Extract structure with caching.

        Args:
            text: Input text

        Returns:
            StructurePattern
        """
        if not self.cache_enabled:
            return self.mapper.extract_structure(text)

        # Check cache
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        # Compute and cache
        structure = self.mapper.extract_structure(text)
        self.cache.put(text, structure)

        return structure

    def check_consistency(self, prompt_structure, response_structure, threshold: float = 0.6):
        """Forward to underlying mapper"""
        return self.mapper.check_consistency(
            prompt_structure,
            response_structure,
            threshold=threshold
        )

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.cache_enabled:
            return {"enabled": False}

        return {"enabled": True, **self.cache.get_stats()}

    def print_cache_stats(self):
        """Print cache statistics"""
        if not self.cache_enabled:
            print("Cache: Disabled")
            return

        self.cache.print_stats()


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Demo caching
    mapper = CachedGrammaticalMapper(cache_enabled=True)

    texts = [
        "Who won the 2019 Nobel Prize in Physics?",
        "What is the capital of France?",
        "Who won the 2019 Nobel Prize in Physics?",  # Repeat - should hit cache
        "What is the capital of France?",            # Repeat - should hit cache
        "How many planets are in the solar system?",
    ]

    print("Extracting structures (with caching):\n")

    for i, text in enumerate(texts, 1):
        structure = mapper.extract_structure(text)
        print(f"[{i}] {text}")
        print(f"    Entities: {structure.entities}")

    print("\n" + "="*60)
    mapper.print_cache_stats()
    print("="*60)

    # Expected output:
    # - 5 requests
    # - 3 unique texts
    # - 2 cache hits (40% hit rate)
