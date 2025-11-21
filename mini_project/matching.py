"""
Matching functions for scoring skill and experience compatibility between bench employees and positions.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
from openai import OpenAI


def score_skill_match(
    bench_skills: str | None,
    position_skills: str | None,
    *,
    use_llm: bool = True,
    similarity_threshold: float = 0.7,
    exact_match_fallback: bool = True,
) -> float:
    """
    Score how well a bench employee's skills match a position's required skills.
    
    Uses LLM-based semantic similarity with fuzzy matching, falling back to exact matching.
    Uses F1-score approach: balances precision (how relevant the employee's skills are)
    and recall (how completely the employee covers the position's requirements).
    
    Args:
        bench_skills: Comma-separated string of employee skills (e.g., "Python, React, SQL")
        position_skills: Comma-separated string of required skills (e.g., "Python, JavaScript")
        use_llm: Whether to use LLM embeddings for semantic matching (default: True)
        similarity_threshold: Minimum cosine similarity for fuzzy match (default: 0.7)
        exact_match_fallback: Fall back to exact matching if LLM fails (default: True)
    
    Returns:
        Score between 0.0 (no match) and 1.0 (perfect match)
    """
    if not bench_skills or not position_skills:
        return 0.0
    
    # Normalize and split skills
    bench_list = _split_skill_string(bench_skills)
    position_list = _split_skill_string(position_skills)
    
    if not bench_list or not position_list:
        return 0.0
    
    # Try LLM-based semantic matching if enabled
    if use_llm:
        try:
            return _score_with_embeddings(
                bench_list, 
                position_list, 
                similarity_threshold=similarity_threshold
            )
        except Exception:
            # If LLM fails, fall back to exact matching if enabled
            if exact_match_fallback:
                return _score_exact_match(bench_list, position_list)
            return 0.0
    
    # Use exact matching
    return _score_exact_match(bench_list, position_list)


def score_experience_match(
    bench_experience: float | None,
    position_min_experience: float | None,
    position_max_experience: float | None,
    tolerance: float = 2.0,
) -> float:
    """
    Score how well a bench employee's experience matches a position's experience requirements.
    
    Args:
        bench_experience: Employee's estimated years of experience
        position_min_experience: Minimum years required for the position
        position_max_experience: Maximum years required for the position (None means no max)
        tolerance: Allowed deviation in years (default: 2.0)
    
    Returns:
        Score between 0.0 (poor match) and 1.0 (perfect match)
    """
    if bench_experience is None:
        return 0.0
    
    if position_min_experience is None and position_max_experience is None:
        # No experience requirement specified
        return 1.0
    
    # Normalize min/max
    min_exp = position_min_experience or 0.0
    max_exp = position_max_experience or float('inf')
    
    # Perfect match: experience is within the required range
    if min_exp <= bench_experience <= max_exp:
        return 1.0
    
    # Calculate distance from the range
    if bench_experience < min_exp:
        # Below minimum
        distance = min_exp - bench_experience
        if distance <= tolerance:
            # Within tolerance, score decreases linearly
            return max(0.0, 1.0 - (distance / tolerance) * 0.5)
        else:
            # Too far below, score decreases more sharply
            return max(0.0, 0.5 - (distance - tolerance) / (tolerance * 2))
    
    else:  # bench_experience > max_exp
        # Above maximum
        if max_exp == float('inf'):
            # No max limit, but employee has more experience (usually okay)
            return 1.0
        
        distance = bench_experience - max_exp
        if distance <= tolerance:
            # Within tolerance, slight penalty
            return max(0.0, 1.0 - (distance / tolerance) * 0.3)
        else:
            # Too far above, more penalty
            return max(0.0, 0.7 - (distance - tolerance) / (tolerance * 3))


def _get_openai_client() -> OpenAI:
    """Get OpenAI client instance."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def _get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    Get embeddings for a list of texts using OpenAI.
    Uses pre-computed skill embeddings cache when available.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use
    
    Returns:
        List of embedding vectors
    """
    import time
    embed_start_time = time.time()
    
    # Try to get from skill cache first (for skills)
    # Skills are typically short, single words or phrases without complex structure
    try:
        # Check if these look like skills (simple heuristic: short, no complex punctuation)
        are_skills = all(
            len(text.strip()) < 100 and 
            text.count(',') == 0 and
            text.count('|') == 0
            for text in texts
        )
        
        if are_skills and len(texts) > 0:
            # Use skill cache
            from mini_project.skill_embeddings import get_embeddings_from_cache
            embeddings = get_embeddings_from_cache(texts, model=model)
            embed_end_time = time.time()
            embed_time = embed_end_time - embed_start_time
            
            print(f"[LLM QUERY] Get Embeddings (from cache)", flush=True)
            print(f"  Model: {model}", flush=True)
            print(f"  Input Texts: {len(texts)}", flush=True)
            print(f"  Time Taken: {embed_time:.3f} seconds", flush=True)
            
            return embeddings
    except Exception as e:
        # Fall back to direct API call if cache fails
        print(f"[LLM QUERY] Cache lookup failed, using API: {e}", flush=True)
    
    # Fallback: direct API call (for non-skill texts or if cache fails)
    client = _get_openai_client()
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    embed_end_time = time.time()
    embed_time = embed_end_time - embed_start_time
    
    # Extract token usage from embeddings response
    prompt_tokens = getattr(response.usage, 'prompt_tokens', len(texts) * 100)
    total_tokens = getattr(response.usage, 'total_tokens', prompt_tokens)
    
    print(f"[LLM QUERY] Get Embeddings (API call)", flush=True)
    print(f"  Model: {model}", flush=True)
    print(f"  Time Taken: {embed_time:.3f} seconds", flush=True)
    print(f"  Input Texts: {len(texts)}", flush=True)
    print(f"  Prompt Tokens: {prompt_tokens}", flush=True)
    print(f"  Total Tokens: {total_tokens}", flush=True)
    
    return [item.embedding for item in response.data]


def _calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score between -1 and 1 (typically 0 to 1 for embeddings)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def _score_with_embeddings(
    bench_skills: list[str],
    position_skills: list[str],
    similarity_threshold: float = 0.7,
) -> float:
    """
    Score skill match using LLM embeddings for semantic similarity.
    
    Args:
        bench_skills: List of employee skills
        position_skills: List of required skills
        similarity_threshold: Minimum similarity to consider a match
    
    Returns:
        F1 score based on semantic matches
    """
    # Get embeddings for all skills
    all_skills = bench_skills + position_skills
    embeddings = _get_embeddings(all_skills)
    
    bench_embeddings = embeddings[:len(bench_skills)]
    position_embeddings = embeddings[len(bench_skills):]
    
    # Find matches using similarity threshold
    matches = []
    matched_bench_indices = set()
    matched_position_indices = set()
    
    # For each position skill, find the best matching bench skill
    for pos_idx, pos_emb in enumerate(position_embeddings):
        best_match_idx = None
        best_similarity = 0.0
        
        for bench_idx, bench_emb in enumerate(bench_embeddings):
            similarity = _calculate_cosine_similarity(pos_emb, bench_emb)
            
            # Check for exact match first (similarity should be very high)
            bench_skill_norm = _normalize_skill(bench_skills[bench_idx])
            pos_skill_norm = _normalize_skill(position_skills[pos_idx])
            if bench_skill_norm == pos_skill_norm:
                similarity = 1.0
            
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = bench_idx
        
        if best_match_idx is not None:
            matches.append((best_match_idx, pos_idx, best_similarity))
            matched_bench_indices.add(best_match_idx)
            matched_position_indices.add(pos_idx)
    
    if not matches:
        return 0.0
    
    # Calculate precision and recall with weighted similarity
    # Weight matches by their similarity score
    weighted_matches = sum(sim for _, _, sim in matches)
    
    # Precision: weighted relevant skills / total employee skills
    precision = weighted_matches / len(bench_skills) if bench_skills else 0.0
    
    # Recall: weighted matched skills / total required skills
    recall = weighted_matches / len(position_skills) if position_skills else 0.0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return round(f1_score, 3)


def _score_exact_match(bench_skills: list[str], position_skills: list[str]) -> float:
    """
    Score skill match using exact string matching (fallback method).
    
    Args:
        bench_skills: List of employee skills
        position_skills: List of required skills
    
    Returns:
        F1 score based on exact matches
    """
    bench_set = {_normalize_skill(s) for s in bench_skills if _normalize_skill(s)}
    position_set = {_normalize_skill(s) for s in position_skills if _normalize_skill(s)}
    
    if not bench_set or not position_set:
        return 0.0
    
    intersection = bench_set & position_set
    
    if not intersection:
        return 0.0
    
    precision = len(intersection) / len(bench_set) if bench_set else 0.0
    recall = len(intersection) / len(position_set) if position_set else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return round(f1_score, 3)


def _split_skill_string(skills_str: str) -> list[str]:
    """
    Split a comma-separated skills string into a list of normalized skills.
    
    Args:
        skills_str: Comma-separated string of skills
    
    Returns:
        List of normalized skill strings
    """
    if not skills_str:
        return []
    
    skills = []
    for skill in skills_str.split(','):
        normalized = _normalize_skill(skill)
        if normalized:
            skills.append(normalized)
    
    return skills


def _normalize_skill(skill: str) -> str | None:
    """
    Normalize a single skill string.
    
    Args:
        skill: Raw skill string
    
    Returns:
        Normalized skill string or None if empty
    """
    if not skill:
        return None
    
    # Trim whitespace and convert to lowercase
    normalized = skill.strip().lower()
    
    # Remove empty strings
    if not normalized:
        return None
    
    return normalized

