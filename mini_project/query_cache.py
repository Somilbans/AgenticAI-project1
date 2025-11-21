"""
Memoization cache for LLM query intent results and full responses.

This module provides caching for query intent analysis and full responses to avoid redundant LLM calls.
The cache is automatically invalidated when data ingestion changes are detected.
"""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from mini_project.vector_store import _get_collection


class QueryIntentCache:
    """
    Thread-safe cache for query intent results and full responses.
    
    The cache automatically invalidates when collection data changes are detected
    by comparing collection counts before and after operations.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}  # Query intent cache
        self._response_cache: Dict[str, str] = {}  # Full response cache for search-only queries
        self._justification_cache: Dict[str, str] = {}  # Justification cache for matches
        self._lock = threading.Lock()
        self._data_version: Dict[str, int] = {}  # collection_name -> count
        self._initialized = False
        # Cache for data version check results (to avoid redundant collection access)
        # Keyed by db_path string representation
        self._data_version_cache: Dict[str, Dict[str, int]] = {}  # db_path -> version
        self._data_version_cache_time: Dict[str, float] = {}  # db_path -> timestamp
        self._data_version_cache_ttl: float = 1.0  # Cache for 1 second
        # Cache hit/miss metrics for monitoring
        self._intent_hits: int = 0
        self._intent_misses: int = 0
        self._response_hits: int = 0
        self._response_misses: int = 0
        self._justification_hits: int = 0
        self._justification_misses: int = 0
    
    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for the query string."""
        # Normalize query: lowercase and strip whitespace
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _get_data_version(self, db_path: Optional[Path] = None) -> Dict[str, int]:
        """
        Get current data version by checking collection counts.
        
        Returns a dict mapping collection names to their document counts.
        """
        version = {}
        try:
            # Check both main collections
            for collection_name in ["onbench", "availablepositions"]:
                try:
                    collection = _get_collection(collection_name, db_path=db_path)
                    version[collection_name] = collection.count()
                except Exception:
                    # Collection might not exist yet
                    version[collection_name] = 0
        except Exception:
            # If we can't check, return empty version
            pass
        return version
    
    def _check_data_changed(self, db_path: Optional[Path] = None) -> bool:
        """
        Check if data has changed since last cache check.
        
        Uses a short-lived cache (1 second) to avoid redundant collection access
        when multiple cache methods are called in quick succession.
        
        Returns True if data has changed (cache should be invalidated).
        """
        current_time = time.time()
        db_path_key = str(db_path) if db_path else "default"
        
        # Check if we have a cached version check result that's still valid
        with self._lock:
            cached_time = self._data_version_cache_time.get(db_path_key, 0.0)
            cached_version = self._data_version_cache.get(db_path_key)
            
            if (cached_version is not None and 
                (current_time - cached_time) < self._data_version_cache_ttl):
                # Use cached result
                current_version = cached_version
            else:
                # Cache miss or expired - get fresh version
                current_version = self._get_data_version(db_path)
                # Update cache
                self._data_version_cache[db_path_key] = current_version
                self._data_version_cache_time[db_path_key] = current_time
        
        with self._lock:
            if not self._initialized:
                # First time - initialize and don't invalidate
                self._data_version = current_version
                self._initialized = True
                return False
            
            # Compare with stored version
            if current_version != self._data_version:
                # Data has changed - update version and invalidate cache
                self._data_version = current_version
                # Clear the version cache since data changed
                self._data_version_cache.clear()
                self._data_version_cache_time.clear()
                return True
        
        return False
    
    def get(self, query: str, db_path: Optional[Path] = None) -> Optional[Dict]:
        """
        Get cached query intent result if available and data hasn't changed.
        
        Returns None if cache miss or data has changed.
        """
        # Check if data has changed (this will update version if changed)
        if self._check_data_changed(db_path):
            # Data changed - clear all caches
            with self._lock:
                cache_size = len(self._cache)
                response_size = len(self._response_cache)
                justification_size = len(self._justification_cache)
                self._cache.clear()
                self._response_cache.clear()
                self._justification_cache.clear()
                total_size = cache_size + response_size + justification_size
                if total_size > 0:
                    print(f"[CACHE INVALIDATE] Data changed - Cleared {cache_size} intent, {response_size} response, {justification_size} justification cached entries", flush=True)
                return None
        
        query_hash = self._get_query_hash(query)
        
        with self._lock:
            result = self._cache.get(query_hash)
            if result is not None:
                self._intent_hits += 1
            else:
                self._intent_misses += 1
            return result
    
    def _get_response_cache_key(self, query: str, top_k: int, db_path: Optional[Path] = None) -> str:
        """
        Generate cache key for response cache.
        
        The cache key is based on query, top_k, and db_path.
        Direction and search_json are not included as they're derived from the query.
        
        Args:
            query: User's query string
            top_k: Number of results requested
            db_path: Optional database path
        
        Returns:
            MD5 hash of the normalized cache key string
        """
        # Normalize query: lowercase and strip whitespace
        normalized_query = query.lower().strip()
        # Create cache key string
        db_path_str = str(db_path) if db_path else 'default'
        cache_key_str = f"{normalized_query}|{top_k}|{db_path_str}"
        # Generate MD5 hash
        return hashlib.md5(cache_key_str.encode('utf-8')).hexdigest()
    
    def get_response(self, query: str, direction: str, search_json: Dict, top_k: int, db_path: Optional[Path] = None) -> Optional[str]:
        """
        Get cached full response for search-only queries.
        
        Returns None if cache miss or data has changed.
        Only caches responses for candidate_only, job_only, and both (search) directions.
        """
        # Check if data has changed
        if self._check_data_changed(db_path):
            return None
        
        # Only cache search-only queries (not matching queries)
        if direction not in ["candidate_only", "job_only", "both"]:
            return None
        
        cache_key = self._get_response_cache_key(query, top_k, db_path)
        
        with self._lock:
            result = self._response_cache.get(cache_key)
            if result is not None:
                self._response_hits += 1
            else:
                self._response_misses += 1
            return result
    
    def set_response(self, query: str, direction: str, search_json: Dict, top_k: int, response: str, db_path: Optional[Path] = None) -> None:
        """
        Store full response in cache for search-only queries.
        """
        # Only cache search-only queries
        if direction not in ["candidate_only", "job_only", "both"]:
            return
        
        cache_key = self._get_response_cache_key(query, top_k, db_path)
        
        with self._lock:
            self._response_cache[cache_key] = response
    
    def _get_match_key(self, match_data: Dict) -> Dict:
        """
        Extract normalized match key from match data.
        
        Creates a consistent dictionary representation of match data
        for use in cache key generation.
        
        Args:
            match_data: Dictionary containing candidate, position, and score data
        
        Returns:
            Normalized dictionary with match key fields
        """
        return {
            "candidate_name": match_data.get("candidate", {}).get("employee_name", ""),
            "candidate_skills": match_data.get("candidate", {}).get("custom_skills", ""),
            "candidate_exp": match_data.get("candidate", {}).get("estimated_experience_years"),
            "position_title": match_data.get("position", {}).get("required_designation", ""),
            "position_skills": match_data.get("position", {}).get("custom_skills", ""),
            "pos_min_exp": match_data.get("position", {}).get("min_experience_years"),
            "pos_max_exp": match_data.get("position", {}).get("max_experience_years"),
            "skill_score": match_data.get("skill_score", 0),
            "exp_score": match_data.get("experience_score", 0),
            "combined_score": match_data.get("combined_score", 0)
        }
    
    def _get_justification_cache_key(self, match_data: Dict) -> str:
        """
        Generate cache key for justification based on match data.
        
        Args:
            match_data: Dictionary containing candidate, position, and scores
        
        Returns:
            MD5 hash of the normalized match key
        """
        match_key = self._get_match_key(match_data)
        return hashlib.md5(json.dumps(match_key, sort_keys=True).encode('utf-8')).hexdigest()
    
    def get_justification(self, match_data: Dict) -> Optional[str]:
        """
        Get cached justification for a match based on match content.
        
        Returns None if cache miss.
        """
        cache_key = self._get_justification_cache_key(match_data)
        
        with self._lock:
            result = self._justification_cache.get(cache_key)
            if result is not None:
                self._justification_hits += 1
            else:
                self._justification_misses += 1
            return result
    
    def set_justification(self, match_data: Dict, justification: str) -> None:
        """
        Store justification in cache.
        """
        cache_key = self._get_justification_cache_key(match_data)
        
        with self._lock:
            self._justification_cache[cache_key] = justification
    
    def set(self, query: str, result: Dict, db_path: Optional[Path] = None) -> None:
        """
        Store query intent result in cache.
        
        Also updates data version to current state.
        """
        query_hash = self._get_query_hash(query)
        
        with self._lock:
            # Update data version to current state
            self._data_version = self._get_data_version(db_path)
            self._initialized = True
            
            # Store result
            self._cache[query_hash] = result
    
    def clear(self) -> None:
        """Manually clear all caches and reset metrics."""
        with self._lock:
            self._cache.clear()
            self._response_cache.clear()
            self._justification_cache.clear()
            self._data_version = {}
            self._initialized = False
            # Clear data version cache
            self._data_version_cache.clear()
            self._data_version_cache_time.clear()
            # Reset metrics
            self._intent_hits = 0
            self._intent_misses = 0
            self._response_hits = 0
            self._response_misses = 0
            self._justification_hits = 0
            self._justification_misses = 0
    
    def invalidate(self, db_path: Optional[Path] = None) -> None:
        """
        Invalidate cache by marking data as changed.
        
        This is useful when data ingestion is known to have occurred.
        """
        with self._lock:
            cache_size = len(self._cache)
            response_size = len(self._response_cache)
            justification_size = len(self._justification_cache)
            # Update to current version, which will trigger cache clear on next get
            self._data_version = self._get_data_version(db_path)
            self._cache.clear()
            self._response_cache.clear()
            self._justification_cache.clear()
            self._initialized = True
            # Clear data version cache
            self._data_version_cache.clear()
            self._data_version_cache_time.clear()
            total_size = cache_size + response_size + justification_size
            if total_size > 0:
                print(f"[CACHE INVALIDATE] Manual invalidation - Cleared {cache_size} intent, {response_size} response, {justification_size} justification cached entries", flush=True)
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics including hit/miss metrics.
        
        Returns:
            Dictionary with cache sizes, hit/miss counts, hit rates, and other metrics
        """
        with self._lock:
            # Calculate hit rates
            intent_total = self._intent_hits + self._intent_misses
            response_total = self._response_hits + self._response_misses
            justification_total = self._justification_hits + self._justification_misses
            
            intent_hit_rate = (self._intent_hits / intent_total * 100) if intent_total > 0 else 0.0
            response_hit_rate = (self._response_hits / response_total * 100) if response_total > 0 else 0.0
            justification_hit_rate = (self._justification_hits / justification_total * 100) if justification_total > 0 else 0.0
            
            return {
                "intent_cache_size": len(self._cache),
                "intent_hits": self._intent_hits,
                "intent_misses": self._intent_misses,
                "intent_hit_rate": round(intent_hit_rate, 2),
                "response_cache_size": len(self._response_cache),
                "response_hits": self._response_hits,
                "response_misses": self._response_misses,
                "response_hit_rate": round(response_hit_rate, 2),
                "justification_cache_size": len(self._justification_cache),
                "justification_hits": self._justification_hits,
                "justification_misses": self._justification_misses,
                "justification_hit_rate": round(justification_hit_rate, 2),
                "data_version": self._data_version.copy(),
                "initialized": self._initialized
            }


# Global cache instance
_global_cache: Optional[QueryIntentCache] = None
_cache_lock = threading.Lock()


def get_cache() -> QueryIntentCache:
    """Get or create the global query intent cache instance."""
    global _global_cache
    with _cache_lock:
        if _global_cache is None:
            _global_cache = QueryIntentCache()
        return _global_cache


def clear_cache() -> None:
    """Clear the global cache."""
    cache = get_cache()
    cache.clear()


def invalidate_cache(db_path: Optional[Path] = None) -> None:
    """
    Invalidate the global cache (e.g., after data ingestion).
    
    Args:
        db_path: Optional database path to check
    """
    cache = get_cache()
    cache.invalidate(db_path)

