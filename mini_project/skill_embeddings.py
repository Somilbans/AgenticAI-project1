"""
Skill embedding pre-computation and caching system.

This module extracts unique skills from ChromaDB collections after ingestion,
pre-computes embeddings for all skills, and stores them for reuse during matching.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from openai import OpenAI

from mini_project.matching import _get_openai_client, _normalize_skill, _split_skill_string
from mini_project.vector_store import get_client


class SkillEmbeddingCache:
    """
    Manages pre-computed skill embeddings with persistent storage.
    """
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize the skill embedding cache.
        
        Args:
            cache_file: Path to JSON file for persistent storage.
                       Defaults to 'skill_embeddings_cache.json' in the project root.
        """
        if cache_file is None:
            # Default to project root
            cache_file = Path(__file__).parent.parent / "skill_embeddings_cache.json"
        self.cache_file = cache_file
        self._cache: Dict[str, List[float]] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load embeddings from persistent storage."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache = {k: v for k, v in data.items()}
                print(f"[SKILL CACHE] Loaded {len(self._cache)} skill embeddings from cache", flush=True)
            except Exception as e:
                print(f"[SKILL CACHE] Error loading cache: {e}", flush=True)
                self._cache = {}
        else:
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Save embeddings to persistent storage."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f)
            print(f"[SKILL CACHE] Saved {len(self._cache)} skill embeddings to {self.cache_file}", flush=True)
        except Exception as e:
            print(f"[SKILL CACHE] Error saving cache: {e}", flush=True)
    
    def get(self, skill: str) -> Optional[List[float]]:
        """
        Get embedding for a skill if it exists in cache.
        
        Args:
            skill: Normalized skill string
        
        Returns:
            Embedding vector or None if not found
        """
        normalized = _normalize_skill(skill)
        if normalized:
            return self._cache.get(normalized)
        return None
    
    def get_batch(self, skills: List[str]) -> Dict[str, List[float]]:
        """
        Get embeddings for multiple skills.
        
        Args:
            skills: List of skill strings
        
        Returns:
            Dictionary mapping normalized skills to their embeddings
        """
        result = {}
        for skill in skills:
            normalized = _normalize_skill(skill)
            if normalized and normalized in self._cache:
                result[normalized] = self._cache[normalized]
        return result
    
    def set(self, skill: str, embedding: List[float]) -> None:
        """
        Store an embedding for a skill.
        
        Args:
            skill: Skill string
            embedding: Embedding vector
        """
        normalized = _normalize_skill(skill)
        if normalized:
            self._cache[normalized] = embedding
    
    def set_batch(self, skills: List[str], embeddings: List[List[float]]) -> None:
        """
        Store embeddings for multiple skills.
        
        Args:
            skills: List of skill strings
            embeddings: List of embedding vectors
        """
        for skill, embedding in zip(skills, embeddings):
            self.set(skill, embedding)
        self._save_cache()
    
    def clear(self) -> None:
        """Clear the cache."""
        count = len(self._cache)
        self._cache.clear()
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
            except Exception:
                pass
        print(f"[SKILL CACHE] Cleared {count} skill embeddings", flush=True)
    
    def size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._cache)


# Global cache instance
_global_cache: Optional[SkillEmbeddingCache] = None


def get_skill_cache(cache_file: Optional[Path] = None) -> SkillEmbeddingCache:
    """
    Get the global skill embedding cache instance.
    
    Args:
        cache_file: Optional path to cache file (only used on first call)
    
    Returns:
        SkillEmbeddingCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = SkillEmbeddingCache(cache_file)
    return _global_cache


def extract_unique_skills_from_collection(
    collection_name: str,
    db_path: Optional[Path] = None
) -> Set[str]:
    """
    Extract all unique skills from a ChromaDB collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
        db_path: Optional database path
    
    Returns:
        Set of unique normalized skills
    """
    client = get_client(db_path=db_path)
    
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        print(f"[SKILL EXTRACT] Collection '{collection_name}' not found", flush=True)
        return set()
    
    # Get all documents with metadata
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", [])
    
    unique_skills = set()
    
    for metadata in metadatas:
        # Extract custom_skills from metadata
        skills_str = metadata.get("custom_skills", "") or ""
        if skills_str:
            skills_list = _split_skill_string(skills_str)
            unique_skills.update(skills_list)
    
    print(f"[SKILL EXTRACT] Found {len(unique_skills)} unique skills in '{collection_name}'", flush=True)
    return unique_skills


def extract_all_unique_skills(db_path: Optional[Path] = None) -> Set[str]:
    """
    Extract all unique skills from both 'onbench' and 'availablepositions' collections.
    
    Args:
        db_path: Optional database path
    
    Returns:
        Set of all unique normalized skills
    """
    all_skills = set()
    
    # Extract from bench collection
    bench_skills = extract_unique_skills_from_collection("onbench", db_path=db_path)
    all_skills.update(bench_skills)
    
    # Extract from positions collection
    position_skills = extract_unique_skills_from_collection("availablepositions", db_path=db_path)
    all_skills.update(position_skills)
    
    print(f"[SKILL EXTRACT] Total unique skills across all collections: {len(all_skills)}", flush=True)
    return all_skills


def precompute_skill_embeddings(
    skills: Optional[Set[str]] = None,
    db_path: Optional[Path] = None,
    model: str = "text-embedding-3-small",
    cache_file: Optional[Path] = None,
    force_recompute: bool = False
) -> None:
    """
    Pre-compute embeddings for all unique skills and store them in cache.
    
    This should be called after data ingestion to prepare embeddings for matching.
    
    Args:
        skills: Optional set of skills to embed. If None, extracts from ChromaDB.
        db_path: Optional database path for extracting skills
        model: OpenAI embedding model to use
        cache_file: Optional path to cache file
        force_recompute: If True, recompute embeddings even if they exist in cache
    """
    cache = get_skill_cache(cache_file)
    
    # Extract skills from ChromaDB if not provided
    if skills is None:
        skills = extract_all_unique_skills(db_path=db_path)
    
    if not skills:
        print("[SKILL EMBED] No skills found to embed", flush=True)
        return
    
    # Filter out skills that are already cached (unless force_recompute)
    if not force_recompute:
        skills_to_embed = [s for s in skills if cache.get(s) is None]
    else:
        skills_to_embed = list(skills)
    
    if not skills_to_embed:
        print(f"[SKILL EMBED] All {len(skills)} skills already cached", flush=True)
        return
    
    print(f"[SKILL EMBED] Computing embeddings for {len(skills_to_embed)} skills...", flush=True)
    start_time = time.time()
    
    # Batch embed all skills
    client = _get_openai_client()
    
    # OpenAI allows up to 2048 inputs per request, but we'll batch in chunks of 100 for safety
    batch_size = 100
    all_embeddings = []
    total_tokens = 0
    
    for i in range(0, len(skills_to_embed), batch_size):
        batch = skills_to_embed[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(skills_to_embed) + batch_size - 1)//batch_size
        print(f"[SKILL EMBED] Processing batch {batch_num}/{total_batches} ({len(batch)} skills)...", flush=True)
        
        batch_start = time.time()
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_time = time.time() - batch_start
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
        # Extract token usage
        prompt_tokens = getattr(response.usage, 'prompt_tokens', len(batch) * 100)
        batch_tokens = getattr(response.usage, 'total_tokens', prompt_tokens)
        total_tokens += batch_tokens
        
        print(f"  Batch {batch_num}: {batch_time:.3f}s, {batch_tokens} tokens", flush=True)
    
    # Store in cache
    cache.set_batch(skills_to_embed, all_embeddings)
    
    elapsed = time.time() - start_time
    
    print(f"[SKILL EMBED] Completed in {elapsed:.3f}s", flush=True)
    print(f"[SKILL EMBED] Total tokens used: {total_tokens}", flush=True)
    print(f"[SKILL EMBED] Cache now contains {cache.size()} skill embeddings", flush=True)


def get_embeddings_from_cache(skills: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Get embeddings for skills from cache, computing missing ones if needed.
    
    This is the main function used during matching to get embeddings efficiently.
    
    Args:
        skills: List of skill strings
        model: OpenAI embedding model (for fallback if cache miss)
    
    Returns:
        List of embedding vectors in the same order as input skills
    """
    cache = get_skill_cache()
    
    # Get cached embeddings
    cached = cache.get_batch(skills)
    
    # Find missing skills
    missing_skills = []
    missing_indices = []
    for idx, skill in enumerate(skills):
        normalized = _normalize_skill(skill)
        if normalized and normalized not in cached:
            missing_skills.append(skill)
            missing_indices.append(idx)
    
    # Compute missing embeddings if any
    if missing_skills:
        print(f"[SKILL CACHE] Cache miss: computing {len(missing_skills)} embeddings on-the-fly", flush=True)
        cache_start = time.time()
        client = _get_openai_client()
        response = client.embeddings.create(
            model=model,
            input=missing_skills
        )
        cache_time = time.time() - cache_start
        
        missing_embeddings = [item.embedding for item in response.data]
        
        # Store in cache
        cache.set_batch(missing_skills, missing_embeddings)
        
        # Extract token usage
        prompt_tokens = getattr(response.usage, 'prompt_tokens', len(missing_skills) * 100)
        total_tokens = getattr(response.usage, 'total_tokens', prompt_tokens)
        
        print(f"  Time: {cache_time:.3f}s, Tokens: {total_tokens}", flush=True)
        
        # Add to cached dict
        for skill, emb in zip(missing_skills, missing_embeddings):
            normalized = _normalize_skill(skill)
            if normalized:
                cached[normalized] = emb
    
    # Reconstruct embeddings in original order
    result = []
    for skill in skills:
        normalized = _normalize_skill(skill)
        if normalized and normalized in cached:
            result.append(cached[normalized])
        else:
            # Fallback: return zero vector if somehow still missing
            print(f"[SKILL CACHE] Warning: embedding not found for '{skill}'", flush=True)
            result.append([0.0] * 1536)  # text-embedding-3-small dimension
    
    return result

