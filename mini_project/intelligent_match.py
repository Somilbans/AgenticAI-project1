"""
Intelligent matching system that uses LLM to understand query intent and match
candidates to positions or vice versa using skill and experience scores.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mini_project.llm import answer_question, _get_client
from mini_project.llm_metrics import LLMMetrics, extract_usage_from_response
from mini_project.matching import score_experience_match, score_skill_match
from mini_project.query_cache import get_cache
from mini_project.vector_store import query_collection


def _row_to_dict(row, df_columns) -> Dict:
    """
    Convert a named tuple from itertuples() to a dictionary.
    
    Handles column names that may have been normalized and ensures all 
    DataFrame columns are included in the result.
    
    Args:
        row: Named tuple from DataFrame.itertuples()
        df_columns: List of column names from the DataFrame
    
    Returns:
        Dictionary with all column values
    """
    result = {}
    for col in df_columns:
        # Get the value using the column name directly
        # Column names are normalized (lowercase, underscores), so they should match
        value = getattr(row, col, None)
        result[col] = value
    return result


def _normalize_skill_fields(intent_result: Dict) -> Dict:
    """
    Normalize any skill-related fields to use only custom_skills.
    
    This ensures that even if the LLM returns other skill fields,
    we consolidate them into custom_skills for consistent searching.
    """
    if "search_json" not in intent_result:
        return intent_result
    
    search_json = intent_result["search_json"]
    
    # Skill field names that should be normalized to custom_skills
    skill_fields_bench = ["employee_skills", "custom_skill_groups", "pi_skills", "piskill_group"]
    skill_fields_position = ["skill1", "skill2", "skill3", "skill_description"]
    
    # Normalize bench_fields
    if "bench_fields" in search_json and isinstance(search_json["bench_fields"], dict):
        bench_fields = search_json["bench_fields"]
        skills_value = None
        
        # Collect skills from custom_skills first (if present)
        if "custom_skills" in bench_fields and bench_fields["custom_skills"]:
            skills_value = bench_fields["custom_skills"]
        
        # Collect skills from other skill fields and merge
        for field in skill_fields_bench:
            if field in bench_fields and bench_fields[field]:
                if skills_value:
                    skills_value = f"{skills_value} {bench_fields[field]}"
                else:
                    skills_value = bench_fields[field]
                # Remove the non-standard field
                del bench_fields[field]
        
        # Set custom_skills with merged value
        if skills_value:
            bench_fields["custom_skills"] = skills_value
        elif any(field in bench_fields for field in skill_fields_bench):
            # If we had skill fields but no value, ensure custom_skills exists
            bench_fields["custom_skills"] = None
    
    # Normalize position_fields
    if "position_fields" in search_json and isinstance(search_json["position_fields"], dict):
        position_fields = search_json["position_fields"]
        skills_value = None
        
        # Collect skills from custom_skills first (if present)
        if "custom_skills" in position_fields and position_fields["custom_skills"]:
            skills_value = position_fields["custom_skills"]
        
        # Collect skills from other skill fields and merge
        for field in skill_fields_position:
            if field in position_fields and position_fields[field]:
                if skills_value:
                    skills_value = f"{skills_value} {position_fields[field]}"
                else:
                    skills_value = position_fields[field]
                # Remove the non-standard field
                del position_fields[field]
        
        # Set custom_skills with merged value
        if skills_value:
            position_fields["custom_skills"] = skills_value
        elif any(field in position_fields for field in skill_fields_position):
            # If we had skill fields but no value, ensure custom_skills exists
            position_fields["custom_skills"] = None
    
    return intent_result


def understand_query_intent(
    question: str, 
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> Dict[str, str]:
    """
    Use LLM to understand the query intent and determine matching direction.
    
    Results are memoized to avoid redundant LLM calls for similar queries.
    Cache is automatically invalidated when data ingestion changes are detected.
    
    Args:
        question: The user's query
        db_path: Optional database path for cache version checking
        metrics_collector: Optional list to collect LLM metrics
    
    Returns:
        Dict with keys:
        - "direction": One of the supported directions
        - "search_json": JSON object with collection parameters to search
        - "reasoning": Explanation of the decision
    """
    # Check cache first
    cache = get_cache()
    cached_result = cache.get(question, db_path=db_path)
    if cached_result is not None:
        print(f"[CACHE HIT] Query: '{question}' - Using cached intent result", flush=True)
        return cached_result
    
    # Cache miss - call LLM
    print(f"[CACHE MISS] Query: '{question}' - Calling LLM for intent analysis", flush=True)
    client = _get_client()
    
    # Track LLM call timing
    llm_start_time = time.time()
    
    prompt = f"""Analyze this query and determine the matching intent. Think carefully about what the user is asking for.

Query: "{question}"

Available Collection Fields:

BENCH Collection (onbench) - Fields available for searching:
- employee_name, employee_code, designation, grade_subgrade, employee_track
- custom_skills (USE THIS FOR ALL SKILLS - do not use employee_skills, custom_skill_groups, pi_skills, or piskill_group)
- estimated_experience_years, employee_joining_date, joining_month

POSITIONS Collection (availablepositions) - Fields available for searching:
- required_designation, project_name, project_id, grade
- custom_skills (USE THIS FOR ALL SKILLS - do not use skill1, skill2, skill3, or skill_description)
- location, industry, updated_industry
- min_experience_years, max_experience_years, experience_required
- responsibilities

Respond with JSON only, no other text. Use this format:
{{
    "direction": "candidate_to_job" | "job_to_candidate" | "candidate_only" | "job_only" | "both" | "irrelevant_query",
    "search_json": {{
        "keywords": "main search keywords from the query",
        "bench_fields": {{"field_name": "value or null", ...}},
        "position_fields": {{"field_name": "value or null", ...}}
    }},
    "reasoning": "brief explanation of why you chose this direction"
}}

The search_json should include:
- "keywords": Extract the main search terms (e.g., "react", "Python", "Pune", "John")
- "bench_fields": Object with relevant bench collection field names and their values (if mentioned in query), or null if not mentioned
  Example: {{"custom_skills": "react", "employee_name": "John", "location": null, "estimated_experience_years": null}}
- "position_fields": Object with relevant position collection field names and their values (if mentioned in query), or null if not mentioned
  Example: {{"custom_skills": "react", "required_designation": "developer", "location": "Pune", "min_experience_years": null}}

Directions (choose the most appropriate one):
- "candidate_to_job": User wants to find jobs/positions for a specific candidate/employee on bench. The focus is on finding suitable positions for a candidate. Example: "Find jobs for John", "What positions suit this employee?"
- "job_to_candidate": User wants to find candidates/employees for a specific job/position. The focus is on finding suitable candidates for a position. Example: "Who can fill the Python developer role?", "Find candidates for this position"
- "candidate_only": User wants to search or query information about candidates/employees only, without matching to jobs. Example: "Show me employees on bench", "List all candidates", "Who is on the bench?"
- "job_only": User wants to search or query information about jobs/positions only, without matching to candidates. Example: "List available positions", "Show open roles", "What positions are available?"
- "both": User wants to see information from both candidates and jobs, or wants to see matches between them. Example: "Show best matches", "Match employees to positions", "What candidates and positions do we have?"
- "irrelevant_query": Query is completely out of scope or not related to bench employees, positions, or job matching at all. Example: "What's the weather?", "Tell me a joke", "How to cook pasta?"

IMPORTANT for search_json:
- Extract keywords from the query (skills, technologies, names, locations, etc.)
- Populate bench_fields with relevant field-value pairs if the query mentions bench-related information
- Populate position_fields with relevant field-value pairs if the query mentions position-related information
- Use null for fields not mentioned in the query
- Only include fields that are relevant to the query
- CRITICAL: For ANY skills mentioned in the query, ALWAYS use "custom_skills" field ONLY in both bench_fields and position_fields. Do NOT use employee_skills, skill1, skill2, skill3, skill_description, or any other skill-related fields.

Examples:
- Query: "find employees for react" → 
  search_json: {{
    "keywords": "react",
    "bench_fields": {{"custom_skills": "react"}},
    "position_fields": {{}}
  }}

- Query: "find job for react in Pune" → 
  search_json: {{
    "keywords": "react Pune",
    "bench_fields": {{}},
    "position_fields": {{"custom_skills": "react", "location": "Pune"}}
  }}

- Query: "employees in Pune with 5 years experience" → 
  search_json: {{
    "keywords": "Pune 5 years experience",
    "bench_fields": {{"location": "Pune", "estimated_experience_years": 5}},
    "position_fields": {{}}
  }}

- Query: "Python developer role" → 
  search_json: {{
    "keywords": "Python developer",
    "bench_fields": {{"custom_skills": "Python"}},
    "position_fields": {{"required_designation": "developer", "custom_skills": "Python"}}
  }}

- Query: "find candidates with Java and Spring skills" → 
  search_json: {{
    "keywords": "Java Spring",
    "bench_fields": {{"custom_skills": "Java Spring"}},
    "position_fields": {{}}
  }}

Important: 
- Analyze the intent carefully, not just keywords
- Consider the context and what the user is really asking for
- Include relevant collection field names in search_json to improve search accuracy
- If the query asks about matching/finding suitable options, use candidate_to_job or job_to_candidate
- If the query just asks to list/show/search, use candidate_only or job_only
- If the query asks about both or matching in general, use both
- Only use irrelevant_query if it's completely unrelated to the domain
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at analyzing queries and understanding user intent. You analyze queries carefully and return only valid JSON with the direction, search_json (containing keywords, bench_fields, and position_fields), and reasoning."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        # Extract metrics
        llm_end_time = time.time()
        llm_time = llm_end_time - llm_start_time
        metrics = extract_usage_from_response(response, "gpt-4o-mini")
        metrics.time_taken = llm_time
        
        # Collect metrics if collector provided
        if metrics_collector is not None:
            metrics_collector.append(metrics)
        
        print(f"[LLM QUERY] Query Intent Analysis", flush=True)
        print(f"  Model: gpt-4o-mini", flush=True)
        print(f"  Time Taken: {llm_time:.3f} seconds", flush=True)
        print(f"  Prompt Tokens: {metrics.prompt_tokens}", flush=True)
        print(f"  Completion Tokens: {metrics.completion_tokens}", flush=True)
        print(f"  Total Tokens: {metrics.total_tokens}", flush=True)
        
        result = json.loads(response.choices[0].message.content)
        # Validate the direction
        valid_directions = ["candidate_to_job", "job_to_candidate", "candidate_only", "job_only", "both", "irrelevant_query"]
        if result.get("direction") not in valid_directions:
            # If LLM returns invalid direction, default to both
            result["direction"] = "both"
            result["reasoning"] = f"Invalid direction returned, defaulting to both. Original: {result.get('direction')}"
        
        # Normalize skill fields to custom_skills only
        result = _normalize_skill_fields(result)
        
        # Store in cache
        cache.set(question, result, db_path=db_path)
        print(f"[CACHE STORE] Query: '{question}' - Stored intent result in cache", flush=True)
        
        return result
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try one more time with a simpler prompt
        try:
            retry_start_time = time.time()
            simple_prompt = f"""Analyze this query: "{question}"

Return JSON with direction (one of: candidate_to_job, job_to_candidate, candidate_only, job_only, both, irrelevant_query), search_json (with keywords, bench_fields, position_fields), and reasoning."""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": simple_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract metrics for retry
            retry_end_time = time.time()
            retry_time = retry_end_time - retry_start_time
            retry_metrics = extract_usage_from_response(response, "gpt-4o-mini")
            retry_metrics.time_taken = retry_time
            
            # Collect metrics if collector provided
            if metrics_collector is not None:
                metrics_collector.append(retry_metrics)
            
            print(f"[LLM QUERY] Query Intent Analysis (Retry)", flush=True)
            print(f"  Model: gpt-4o-mini", flush=True)
            print(f"  Time Taken: {retry_time:.3f} seconds", flush=True)
            print(f"  Prompt Tokens: {retry_metrics.prompt_tokens}", flush=True)
            print(f"  Completion Tokens: {retry_metrics.completion_tokens}", flush=True)
            print(f"  Total Tokens: {retry_metrics.total_tokens}", flush=True)
            
            result = json.loads(response.choices[0].message.content)
            if result.get("direction") not in valid_directions:
                result["direction"] = "both"
            
            # Normalize skill fields to custom_skills only
            result = _normalize_skill_fields(result)
            
            # Store in cache
            cache.set(question, result, db_path=db_path)
            
            return result
        except Exception as retry_error:
            # Last resort: return both as safe default
            return {
                "direction": "both",
                "search_json": {"keywords": question, "bench_fields": {}, "position_fields": {}},
                "reasoning": "LLM analysis failed, defaulting to search both collections"
            }
    except Exception as e:
        # If LLM completely fails, return a safe default
        return {
            "direction": "both",
            "search_json": {"keywords": question, "bench_fields": {}, "position_fields": {}},
            "reasoning": f"Error analyzing query with LLM: {str(e)}, defaulting to search both collections"
        }


def get_semantic_search_results(
    search_json: Dict,
    collection_name: str,
    top_k: int = 10,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Get search results from ChromaDB using structured search JSON.
    
    Args:
        search_json: Dict with keys:
            - "keywords": Main search keywords
            - "bench_fields": Dict of bench field-value pairs (if searching bench)
            - "position_fields": Dict of position field-value pairs (if searching positions)
        collection_name: Name of the collection to search
        top_k: Number of results to return
        db_path: Optional database path
    
    Returns:
        DataFrame with search results
    """
    from mini_project.vector_store import _get_collection
    
    collection = _get_collection(collection_name, db_path=db_path)
    
    # Cache collection count to avoid calling it twice
    collection_count = collection.count()
    if collection_count == 0:
        return pd.DataFrame()
    
    # Build search query from JSON
    keywords = search_json.get("keywords", "")
    
    # Get relevant fields based on collection
    if collection_name == "onbench":
        relevant_fields = search_json.get("bench_fields", {})
    elif collection_name == "availablepositions":
        relevant_fields = search_json.get("position_fields", {})
    else:
        relevant_fields = {}
    
    # Build enhanced query text combining keywords and field values
    query_parts = [keywords] if keywords else []
    
    # Add field values to search query
    for field, value in relevant_fields.items():
        if value is not None and value != "":
            if isinstance(value, (int, float)):
                query_parts.append(f"{field}: {value}")
            else:
                query_parts.append(f"{field}: {value}")
    
    query_text = " ".join(query_parts).strip()
    
    # If no query text, use empty string (will return all results)
    if not query_text:
        query_text = ""
    
    # Query and get results with metadata (use cached count)
    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection_count)
    )
    
    # Get metadatas which contain the actual row data
    metadatas = results.get("metadatas", [[]])
    if not metadatas or not metadatas[0]:
        return pd.DataFrame()
    
    # Convert list of dicts to DataFrame
    rows = metadatas[0]
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Apply field-based filtering if specified
    if relevant_fields:
        for field, value in relevant_fields.items():
            if value is not None and value != "" and field in df.columns:
                if isinstance(value, (int, float)):
                    # For numeric fields, allow approximate matching
                    df = df[df[field].apply(lambda x: _matches_numeric(x, value))]
                else:
                    # For string fields, use case-insensitive partial matching
                    df = df[df[field].astype(str).str.contains(str(value), case=False, na=False)]
    
    return df.head(top_k)


def _matches_numeric(value, target: float, tolerance: float = 0.5) -> bool:
    """Check if numeric value matches target within tolerance."""
    try:
        if value is None:
            return False
        num_value = float(value)
        return abs(num_value - target) <= tolerance
    except (ValueError, TypeError):
        return False


def match_candidate_to_jobs(
    search_json: Dict,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> List[Dict]:
    """
    Find jobs for a candidate on the bench.
    
    Args:
        search_json: Search parameters with keywords and field filters
        top_k: Number of matches to return
        db_path: Optional database path
    
    Returns list of matches with scores and justifications.
    """
    # Get candidate from bench
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=5, db_path=db_path
    )
    
    if bench_df.empty:
        return []
    
    # Get all positions (use empty search_json to get all)
    all_positions_search = {"keywords": "", "bench_fields": {}, "position_fields": {}}
    positions_df = get_semantic_search_results(
        all_positions_search, "availablepositions", top_k=20, db_path=db_path
    )
    
    if positions_df.empty:
        return []
    
    # Match candidate to positions
    matches = []
    for candidate_row in bench_df.itertuples():
        candidate_skills = getattr(candidate_row, "custom_skills", "") or ""
        candidate_exp = getattr(candidate_row, "estimated_experience_years", None)
        
        for position_row in positions_df.itertuples():
            pos_skills = getattr(position_row, "custom_skills", "") or ""
            pos_min_exp = getattr(position_row, "min_experience_years", None)
            pos_max_exp = getattr(position_row, "max_experience_years", None)
            
            # Calculate scores
            skill_score = score_skill_match(candidate_skills, pos_skills)
            exp_score = score_experience_match(
                candidate_exp, pos_min_exp, pos_max_exp
            )
            
            # Combined score
            combined_score = (skill_score * 0.6) + (exp_score * 0.4)
            
            if combined_score > 0.3:  # Minimum threshold
                # Convert named tuples to dicts
                candidate_dict = _row_to_dict(candidate_row, bench_df.columns)
                position_dict = _row_to_dict(position_row, positions_df.columns)
                matches.append({
                    "candidate": candidate_dict,
                    "position": position_dict,
                    "skill_score": skill_score,
                    "experience_score": exp_score,
                    "combined_score": combined_score,
                })
    
    # Sort by combined score
    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Generate justifications
    for match in matches[:top_k]:
        match["justification"] = generate_justification(match, metrics_collector=metrics_collector)
    
    return matches[:top_k]


def match_job_to_candidates(
    search_json: Dict,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> List[Dict]:
    """
    Find candidates for a specific job/position.
    
    Args:
        search_json: Search parameters with keywords and field filters
        top_k: Number of matches to return
        db_path: Optional database path
    
    Returns list of matches with scores and justifications.
    """
    # Get position
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=5, db_path=db_path
    )
    
    if positions_df.empty:
        return []
    
    # Get all candidates from bench (use empty search_json to get all)
    all_candidates_search = {"keywords": "", "bench_fields": {}, "position_fields": {}}
    bench_df = get_semantic_search_results(
        all_candidates_search, "onbench", top_k=20, db_path=db_path
    )
    
    if bench_df.empty:
        return []
    
    # Match position to candidates
    matches = []
    for position_row in positions_df.itertuples():
        pos_skills = getattr(position_row, "custom_skills", "") or ""
        pos_min_exp = getattr(position_row, "min_experience_years", None)
        pos_max_exp = getattr(position_row, "max_experience_years", None)
        
        for candidate_row in bench_df.itertuples():
            candidate_skills = getattr(candidate_row, "custom_skills", "") or ""
            candidate_exp = getattr(candidate_row, "estimated_experience_years", None)
            
            # Calculate scores
            skill_score = score_skill_match(candidate_skills, pos_skills)
            exp_score = score_experience_match(
                candidate_exp, pos_min_exp, pos_max_exp
            )
            
            # Combined score
            combined_score = (skill_score * 0.6) + (exp_score * 0.4)
            
            if combined_score > 0.3:  # Minimum threshold
                # Convert named tuples to dicts
                position_dict = _row_to_dict(position_row, positions_df.columns)
                candidate_dict = _row_to_dict(candidate_row, bench_df.columns)
                matches.append({
                    "position": position_dict,
                    "candidate": candidate_dict,
                    "skill_score": skill_score,
                    "experience_score": exp_score,
                    "combined_score": combined_score,
                })
    
    # Sort by combined score
    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Generate justifications
    for match in matches[:top_k]:
        match["justification"] = generate_justification(match, metrics_collector=metrics_collector)
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mini_project.llm import answer_question, _get_client
from mini_project.llm_metrics import LLMMetrics, extract_usage_from_response
from mini_project.matching import score_experience_match, score_skill_match
from mini_project.query_cache import get_cache
from mini_project.vector_store import query_collection


def _row_to_dict(row, df_columns) -> Dict:
    """
    Convert a named tuple from itertuples() to a dictionary.
    
    Handles column names that may have been normalized and ensures all 
    DataFrame columns are included in the result.
    
    Args:
        row: Named tuple from DataFrame.itertuples()
        df_columns: List of column names from the DataFrame
    
    Returns:
        Dictionary with all column values
    """
    result = {}
    for col in df_columns:
        # Get the value using the column name directly
        # Column names are normalized (lowercase, underscores), so they should match
        value = getattr(row, col, None)
        result[col] = value
    return result


def _normalize_skill_fields(intent_result: Dict) -> Dict:
    """
    Normalize any skill-related fields to use only custom_skills.
    
    This ensures that even if the LLM returns other skill fields,
    we consolidate them into custom_skills for consistent searching.
    """
    if "search_json" not in intent_result:
        return intent_result
    
    search_json = intent_result["search_json"]
    
    # Skill field names that should be normalized to custom_skills
    skill_fields_bench = ["employee_skills", "custom_skill_groups", "pi_skills", "piskill_group"]
    skill_fields_position = ["skill1", "skill2", "skill3", "skill_description"]
    
    # Normalize bench_fields
    if "bench_fields" in search_json and isinstance(search_json["bench_fields"], dict):
        bench_fields = search_json["bench_fields"]
        skills_value = None
        
        # Collect skills from custom_skills first (if present)
        if "custom_skills" in bench_fields and bench_fields["custom_skills"]:
            skills_value = bench_fields["custom_skills"]
        
        # Collect skills from other skill fields and merge
        for field in skill_fields_bench:
            if field in bench_fields and bench_fields[field]:
                if skills_value:
                    skills_value = f"{skills_value} {bench_fields[field]}"
                else:
                    skills_value = bench_fields[field]
                # Remove the non-standard field
                del bench_fields[field]
        
        # Set custom_skills with merged value
        if skills_value:
            bench_fields["custom_skills"] = skills_value
        elif any(field in bench_fields for field in skill_fields_bench):
            # If we had skill fields but no value, ensure custom_skills exists
            bench_fields["custom_skills"] = None
    
    # Normalize position_fields
    if "position_fields" in search_json and isinstance(search_json["position_fields"], dict):
        position_fields = search_json["position_fields"]
        skills_value = None
        
        # Collect skills from custom_skills first (if present)
        if "custom_skills" in position_fields and position_fields["custom_skills"]:
            skills_value = position_fields["custom_skills"]
        
        # Collect skills from other skill fields and merge
        for field in skill_fields_position:
            if field in position_fields and position_fields[field]:
                if skills_value:
                    skills_value = f"{skills_value} {position_fields[field]}"
                else:
                    skills_value = position_fields[field]
                # Remove the non-standard field
                del position_fields[field]
        
        # Set custom_skills with merged value
        if skills_value:
            position_fields["custom_skills"] = skills_value
        elif any(field in position_fields for field in skill_fields_position):
            # If we had skill fields but no value, ensure custom_skills exists
            position_fields["custom_skills"] = None
    
    return intent_result


def understand_query_intent(
    question: str, 
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> Dict[str, str]:
    """
    Use LLM to understand the query intent and determine matching direction.
    
    Results are memoized to avoid redundant LLM calls for similar queries.
    Cache is automatically invalidated when data ingestion changes are detected.
    
    Args:
        question: The user's query
        db_path: Optional database path for cache version checking
        metrics_collector: Optional list to collect LLM metrics
    
    Returns:
        Dict with keys:
        - "direction": One of the supported directions
        - "search_json": JSON object with collection parameters to search
        - "reasoning": Explanation of the decision
    """
    # Check cache first
    cache = get_cache()
    cached_result = cache.get(question, db_path=db_path)
    if cached_result is not None:
        print(f"[CACHE HIT] Query: '{question}' - Using cached intent result", flush=True)
        return cached_result
    
    # Cache miss - call LLM
    print(f"[CACHE MISS] Query: '{question}' - Calling LLM for intent analysis", flush=True)
    client = _get_client()
    
    # Track LLM call timing
    llm_start_time = time.time()
    
    prompt = f"""Analyze this query and determine the matching intent. Think carefully about what the user is asking for.

Query: "{question}"

Available Collection Fields:

BENCH Collection (onbench) - Fields available for searching:
- employee_name, employee_code, designation, grade_subgrade, employee_track
- custom_skills (USE THIS FOR ALL SKILLS - do not use employee_skills, custom_skill_groups, pi_skills, or piskill_group)
- estimated_experience_years, employee_joining_date, joining_month

POSITIONS Collection (availablepositions) - Fields available for searching:
- required_designation, project_name, project_id, grade
- custom_skills (USE THIS FOR ALL SKILLS - do not use skill1, skill2, skill3, or skill_description)
- location, industry, updated_industry
- min_experience_years, max_experience_years, experience_required
- responsibilities

Respond with JSON only, no other text. Use this format:
{{
    "direction": "candidate_to_job" | "job_to_candidate" | "candidate_only" | "job_only" | "both" | "irrelevant_query",
    "search_json": {{
        "keywords": "main search keywords from the query",
        "bench_fields": {{"field_name": "value or null", ...}},
        "position_fields": {{"field_name": "value or null", ...}}
    }},
    "reasoning": "brief explanation of why you chose this direction"
}}

The search_json should include:
- "keywords": Extract the main search terms (e.g., "react", "Python", "Pune", "John")
- "bench_fields": Object with relevant bench collection field names and their values (if mentioned in query), or null if not mentioned
  Example: {{"custom_skills": "react", "employee_name": "John", "location": null, "estimated_experience_years": null}}
- "position_fields": Object with relevant position collection field names and their values (if mentioned in query), or null if not mentioned
  Example: {{"custom_skills": "react", "required_designation": "developer", "location": "Pune", "min_experience_years": null}}

Directions (choose the most appropriate one):
- "candidate_to_job": User wants to find jobs/positions for a specific candidate/employee on bench. The focus is on finding suitable positions for a candidate. Example: "Find jobs for John", "What positions suit this employee?"
- "job_to_candidate": User wants to find candidates/employees for a specific job/position. The focus is on finding suitable candidates for a position. Example: "Who can fill the Python developer role?", "Find candidates for this position"
- "candidate_only": User wants to search or query information about candidates/employees only, without matching to jobs. Example: "Show me employees on bench", "List all candidates", "Who is on the bench?"
- "job_only": User wants to search or query information about jobs/positions only, without matching to candidates. Example: "List available positions", "Show open roles", "What positions are available?"
- "both": User wants to see information from both candidates and jobs, or wants to see matches between them. Example: "Show best matches", "Match employees to positions", "What candidates and positions do we have?"
- "irrelevant_query": Query is completely out of scope or not related to bench employees, positions, or job matching at all. Example: "What's the weather?", "Tell me a joke", "How to cook pasta?"

IMPORTANT for search_json:
- Extract keywords from the query (skills, technologies, names, locations, etc.)
- Populate bench_fields with relevant field-value pairs if the query mentions bench-related information
- Populate position_fields with relevant field-value pairs if the query mentions position-related information
- Use null for fields not mentioned in the query
- Only include fields that are relevant to the query
- CRITICAL: For ANY skills mentioned in the query, ALWAYS use "custom_skills" field ONLY in both bench_fields and position_fields. Do NOT use employee_skills, skill1, skill2, skill3, skill_description, or any other skill-related fields.

Examples:
- Query: "find employees for react" → 
  search_json: {{
    "keywords": "react",
    "bench_fields": {{"custom_skills": "react"}},
    "position_fields": {{}}
  }}

- Query: "find job for react in Pune" → 
  search_json: {{
    "keywords": "react Pune",
    "bench_fields": {{}},
    "position_fields": {{"custom_skills": "react", "location": "Pune"}}
  }}

- Query: "employees in Pune with 5 years experience" → 
  search_json: {{
    "keywords": "Pune 5 years experience",
    "bench_fields": {{"location": "Pune", "estimated_experience_years": 5}},
    "position_fields": {{}}
  }}

- Query: "Python developer role" → 
  search_json: {{
    "keywords": "Python developer",
    "bench_fields": {{"custom_skills": "Python"}},
    "position_fields": {{"required_designation": "developer", "custom_skills": "Python"}}
  }}

- Query: "find candidates with Java and Spring skills" → 
  search_json: {{
    "keywords": "Java Spring",
    "bench_fields": {{"custom_skills": "Java Spring"}},
    "position_fields": {{}}
  }}

Important: 
- Analyze the intent carefully, not just keywords
- Consider the context and what the user is really asking for
- Include relevant collection field names in search_json to improve search accuracy
- If the query asks about matching/finding suitable options, use candidate_to_job or job_to_candidate
- If the query just asks to list/show/search, use candidate_only or job_only
- If the query asks about both or matching in general, use both
- Only use irrelevant_query if it's completely unrelated to the domain
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at analyzing queries and understanding user intent. You analyze queries carefully and return only valid JSON with the direction, search_json (containing keywords, bench_fields, and position_fields), and reasoning."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        # Extract metrics
        llm_end_time = time.time()
        llm_time = llm_end_time - llm_start_time
        metrics = extract_usage_from_response(response, "gpt-4o-mini")
        metrics.time_taken = llm_time
        
        # Collect metrics if collector provided
        if metrics_collector is not None:
            metrics_collector.append(metrics)
        
        print(f"[LLM QUERY] Query Intent Analysis", flush=True)
        print(f"  Model: gpt-4o-mini", flush=True)
        print(f"  Time Taken: {llm_time:.3f} seconds", flush=True)
        print(f"  Prompt Tokens: {metrics.prompt_tokens}", flush=True)
        print(f"  Completion Tokens: {metrics.completion_tokens}", flush=True)
        print(f"  Total Tokens: {metrics.total_tokens}", flush=True)
        
        result = json.loads(response.choices[0].message.content)
        # Validate the direction
        valid_directions = ["candidate_to_job", "job_to_candidate", "candidate_only", "job_only", "both", "irrelevant_query"]
        if result.get("direction") not in valid_directions:
            # If LLM returns invalid direction, default to both
            result["direction"] = "both"
            result["reasoning"] = f"Invalid direction returned, defaulting to both. Original: {result.get('direction')}"
        
        # Normalize skill fields to custom_skills only
        result = _normalize_skill_fields(result)
        
        # Store in cache
        cache.set(question, result, db_path=db_path)
        print(f"[CACHE STORE] Query: '{question}' - Stored intent result in cache", flush=True)
        
        return result
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try one more time with a simpler prompt
        try:
            retry_start_time = time.time()
            simple_prompt = f"""Analyze this query: "{question}"

Return JSON with direction (one of: candidate_to_job, job_to_candidate, candidate_only, job_only, both, irrelevant_query), search_json (with keywords, bench_fields, position_fields), and reasoning."""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": simple_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract metrics for retry
            retry_end_time = time.time()
            retry_time = retry_end_time - retry_start_time
            retry_metrics = extract_usage_from_response(response, "gpt-4o-mini")
            retry_metrics.time_taken = retry_time
            
            # Collect metrics if collector provided
            if metrics_collector is not None:
                metrics_collector.append(retry_metrics)
            
            print(f"[LLM QUERY] Query Intent Analysis (Retry)", flush=True)
            print(f"  Model: gpt-4o-mini", flush=True)
            print(f"  Time Taken: {retry_time:.3f} seconds", flush=True)
            print(f"  Prompt Tokens: {retry_metrics.prompt_tokens}", flush=True)
            print(f"  Completion Tokens: {retry_metrics.completion_tokens}", flush=True)
            print(f"  Total Tokens: {retry_metrics.total_tokens}", flush=True)
            
            result = json.loads(response.choices[0].message.content)
            if result.get("direction") not in valid_directions:
                result["direction"] = "both"
            
            # Normalize skill fields to custom_skills only
            result = _normalize_skill_fields(result)
            
            # Store in cache
            cache.set(question, result, db_path=db_path)
            
            return result
        except Exception as retry_error:
            # Last resort: return both as safe default
            return {
                "direction": "both",
                "search_json": {"keywords": question, "bench_fields": {}, "position_fields": {}},
                "reasoning": "LLM analysis failed, defaulting to search both collections"
            }
    except Exception as e:
        # If LLM completely fails, return a safe default
        return {
            "direction": "both",
            "search_json": {"keywords": question, "bench_fields": {}, "position_fields": {}},
            "reasoning": f"Error analyzing query with LLM: {str(e)}, defaulting to search both collections"
        }


def match_candidate_to_jobs(
    search_json: Dict,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> List[Dict]:
    """
    Find jobs for a candidate on the bench.
    
    Args:
        search_json: Search parameters with keywords and field filters
        top_k: Number of matches to return
        db_path: Optional database path
    
    Returns list of matches with scores and justifications.
    """
    # Get candidate from bench
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=5, db_path=db_path
    )
    
    if bench_df.empty:
        return []
    
    # Get all positions (use empty search_json to get all)
    all_positions_search = {"keywords": "", "bench_fields": {}, "position_fields": {}}
    positions_df = get_semantic_search_results(
        all_positions_search, "availablepositions", top_k=20, db_path=db_path
    )
    
    if positions_df.empty:
        return []
    
    # Match candidate to positions
    matches = []
    for candidate_row in bench_df.itertuples():
        candidate_skills = getattr(candidate_row, "custom_skills", "") or ""
        candidate_exp = getattr(candidate_row, "estimated_experience_years", None)
        
        for position_row in positions_df.itertuples():
            pos_skills = getattr(position_row, "custom_skills", "") or ""
            pos_min_exp = getattr(position_row, "min_experience_years", None)
            pos_max_exp = getattr(position_row, "max_experience_years", None)
            
            # Calculate scores
            skill_score = score_skill_match(candidate_skills, pos_skills)
            exp_score = score_experience_match(
                candidate_exp, pos_min_exp, pos_max_exp
            )
            
            # Combined score
            combined_score = (skill_score * 0.6) + (exp_score * 0.4)
            
            if combined_score > 0.3:  # Minimum threshold
                # Convert named tuples to dicts
                candidate_dict = _row_to_dict(candidate_row, bench_df.columns)
                position_dict = _row_to_dict(position_row, positions_df.columns)
                matches.append({
                    "candidate": candidate_dict,
                    "position": position_dict,
                    "skill_score": skill_score,
                    "experience_score": exp_score,
                    "combined_score": combined_score,
                })
    
    # Sort by combined score
    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Generate justifications
    for match in matches[:top_k]:
        match["justification"] = generate_justification(match, metrics_collector=metrics_collector)
    
    return matches[:top_k]


def match_job_to_candidates(
    search_json: Dict,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> List[Dict]:
    """
    Find best matches between candidates and positions.
    
    Args:
        search_json: Search parameters with keywords and field filters
        top_k: Number of matches to return
        db_path: Optional database path
    
    Returns list of matches with scores and justifications.
    """
    # Get candidates and positions
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=10, db_path=db_path
    )
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=10, db_path=db_path
    )
    
    if bench_df.empty or positions_df.empty:
        return []
    
    # Match all candidates to all positions
    matches = []
    for candidate_row in bench_df.itertuples():
        candidate_skills = getattr(candidate_row, "custom_skills", "") or ""
        candidate_exp = getattr(candidate_row, "estimated_experience_years", None)
        
        for position_row in positions_df.itertuples():
            pos_skills = getattr(position_row, "custom_skills", "") or ""
            pos_min_exp = getattr(position_row, "min_experience_years", None)
            pos_max_exp = getattr(position_row, "max_experience_years", None)
            
            # Calculate scores
            skill_score = score_skill_match(candidate_skills, pos_skills)
            exp_score = score_experience_match(
                candidate_exp, pos_min_exp, pos_max_exp
            )
            
            # Combined score
            combined_score = (skill_score * 0.6) + (exp_score * 0.4)
            
            if combined_score > 0.3:  # Minimum threshold
                # Convert named tuples to dicts
                candidate_dict = _row_to_dict(candidate_row, bench_df.columns)
                position_dict = _row_to_dict(position_row, positions_df.columns)
                matches.append({
                    "candidate": candidate_dict,
                    "position": position_dict,
                    "skill_score": skill_score,
                    "experience_score": exp_score,
                    "combined_score": combined_score,
                })
    
    # Sort by combined score
    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Generate justifications
    for match in matches[:top_k]:
        match["justification"] = generate_justification(match, metrics_collector=metrics_collector)
    
    return matches[:top_k]


def match_job_to_candidates(
    search_json: Dict,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> List[Dict]:
    """
    Find candidates for a specific job/position.
    
    Args:
        search_json: Search parameters with keywords and field filters
        top_k: Number of matches to return
        db_path: Optional database path
    
    Returns list of matches with scores and justifications.
    """
    # Get position
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=5, db_path=db_path
    )
    
    if positions_df.empty:
        return []
    
    # Get all candidates from bench (use empty search_json to get all)
    all_candidates_search = {"keywords": "", "bench_fields": {}, "position_fields": {}}
    bench_df = get_semantic_search_results(
        all_candidates_search, "onbench", top_k=20, db_path=db_path
    )
    
    if bench_df.empty:
        return []
    
    # Match position to candidates
    matches = []
    for position_row in positions_df.itertuples():
        pos_skills = getattr(position_row, "custom_skills", "") or ""
        pos_min_exp = getattr(position_row, "min_experience_years", None)
        pos_max_exp = getattr(position_row, "max_experience_years", None)
        
        for candidate_row in bench_df.itertuples():
            candidate_skills = getattr(candidate_row, "custom_skills", "") or ""
            candidate_exp = getattr(candidate_row, "estimated_experience_years", None)
            
            # Calculate scores
            skill_score = score_skill_match(candidate_skills, pos_skills)
            exp_score = score_experience_match(
                candidate_exp, pos_min_exp, pos_max_exp
            )
            
            # Combined score
            combined_score = (skill_score * 0.6) + (exp_score * 0.4)
            
            if combined_score > 0.3:  # Minimum threshold
                # Convert named tuples to dicts
                position_dict = _row_to_dict(position_row, positions_df.columns)
                candidate_dict = _row_to_dict(candidate_row, bench_df.columns)
                matches.append({
                    "position": position_dict,
                    "candidate": candidate_dict,
                    "skill_score": skill_score,
                    "experience_score": exp_score,
                    "combined_score": combined_score,
                })
    
    # Sort by combined score
    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Generate justifications
    for match in matches[:top_k]:
        match["justification"] = generate_justification(match, metrics_collector=metrics_collector)
    
    return matches[:top_k]
    
    return matches[:top_k]


def generate_justification(
    match: Dict,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Generate a descriptive justification for a match using LLM.
    Uses cache to avoid redundant LLM calls for identical matches.
    """
    # Check cache first
    cache = get_cache()
    cached_justification = cache.get_justification(match)
    if cached_justification is not None:
        print(f"[CACHE HIT] Justification - Using cached justification for match", flush=True)
        return cached_justification
    
    # Cache miss - generate justification
    print(f"[CACHE MISS] Justification - Generating new justification", flush=True)
    
    candidate = match.get("candidate", {})
    position = match.get("position", {})
    skill_score = match.get("skill_score", 0)
    exp_score = match.get("experience_score", 0)
    combined_score = match.get("combined_score", 0)
    
    candidate_name = candidate.get("employee_name", "Candidate")
    candidate_skills = candidate.get("custom_skills", "N/A")
    candidate_exp = candidate.get("estimated_experience_years", "N/A")
    
    position_title = position.get("required_designation", "Position")
    position_skills = position.get("custom_skills", "N/A")
    pos_min_exp = position.get("min_experience_years", "N/A")
    pos_max_exp = position.get("max_experience_years", "N/A")
    
    context = f"""Candidate: {candidate_name}
Candidate Skills: {candidate_skills}
Candidate Experience: {candidate_exp} years

Position: {position_title}
Required Skills: {position_skills}
Required Experience: {pos_min_exp} - {pos_max_exp} years

Matching Scores:
- Skill Match Score: {skill_score:.3f} (out of 1.0)
- Experience Match Score: {exp_score:.3f} (out of 1.0)
- Combined Score: {combined_score:.3f} (out of 1.0)
"""
    
    prompt = """Generate a descriptive justification for why this candidate is a good match for this position. 
Include specific details about:
1. How well the candidate's skills align with the position requirements
2. How well the candidate's experience matches the position requirements
3. Overall assessment of the match quality

Be specific and reference the scores provided. Keep it concise (2-3 sentences)."""
    
    try:
        # Use LLM to generate justification
        justify_start_time = time.time()
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful recruiting analyst. Generate concise, descriptive justifications for candidate-position matches."},
                {"role": "user", "content": f"{context}\n\n{prompt}"}
            ],
            max_tokens=200
        )
        
        # Extract metrics
        justify_end_time = time.time()
        justify_time = justify_end_time - justify_start_time
        justify_metrics = extract_usage_from_response(response, "gpt-4o-mini")
        justify_metrics.time_taken = justify_time
        
        # Collect metrics if collector provided
        if metrics_collector is not None:
            metrics_collector.append(justify_metrics)
        
        print(f"[LLM QUERY] Generate Justification", flush=True)
        print(f"  Model: gpt-4o-mini", flush=True)
        print(f"  Time Taken: {justify_time:.3f} seconds", flush=True)
        print(f"  Prompt Tokens: {justify_metrics.prompt_tokens}", flush=True)
        print(f"  Completion Tokens: {justify_metrics.completion_tokens}", flush=True)
        print(f"  Total Tokens: {justify_metrics.total_tokens}", flush=True)
        
        justification = response.choices[0].message.content.strip()
        
        # Store in cache
        cache.set_justification(match, justification)
        print(f"[CACHE STORE] Justification - Stored in cache", flush=True)
        
        return justification
    except Exception:
        # Fallback justification
        justification = f"This candidate has a skill match score of {skill_score:.3f} and experience match score of {exp_score:.3f}, resulting in a combined score of {combined_score:.3f}. The candidate's skills align with the position requirements, and their experience level matches the position's expectations."
        # Cache fallback too
        cache.set_justification(match, justification)
        return justification


def search_candidates_only(
    search_json: Dict,
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Search only in the bench/candidates collection.
    
    Args:
        search_json: Search parameters with keywords and field filters
        query: Original query text for context
        top_k: Number of results to return
        db_path: Optional database path
    """
    # Check cache first
    cache = get_cache()
    cached_response = cache.get_response(query, "candidate_only", search_json, top_k, db_path=db_path)
    if cached_response is not None:
        print(f"[CACHE HIT] Response - Using cached response for candidate_only query", flush=True)
        return cached_response
    
    print(f"[CACHE MISS] Response - Generating new response for candidate_only query", flush=True)
    
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=top_k, db_path=db_path
    )
    
    if bench_df.empty:
        response = f"No candidates found for: {query}"
        cache.set_response(query, "candidate_only", search_json, top_k, response, db_path=db_path)
        return response
    
    # Use LLM to format the response
    context_chunks = []
    for row in bench_df.itertuples():
        employee_name = getattr(row, "employee_name", "N/A")
        skills = getattr(row, "custom_skills", "N/A")
        experience = getattr(row, "estimated_experience_years", "N/A")
        context = f"Employee: {employee_name}, Skills: {skills}, Experience: {experience} years"
        context_chunks.append(context)
    
    prompt = f"Based on the following candidate information, provide a clear summary answering: {query}"
    answer_text, answer_metrics = answer_question(prompt, context_chunks)


def find_best_matches(
    search_json: Dict,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> List[Dict]:
    """
    Find best matches between candidates and positions.
    
    Args:
        search_json: Search parameters with keywords and field filters
        top_k: Number of matches to return
        db_path: Optional database path
    
    Returns list of matches with scores and justifications.
    """
    # Get candidates and positions
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=10, db_path=db_path
    )
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=10, db_path=db_path
    )
    
    if bench_df.empty or positions_df.empty:
        return []
    
    # Match all candidates to all positions
    matches = []
    for candidate_row in bench_df.itertuples():
        candidate_skills = getattr(candidate_row, "custom_skills", "") or ""
        candidate_exp = getattr(candidate_row, "estimated_experience_years", None)
        
        for position_row in positions_df.itertuples():
            pos_skills = getattr(position_row, "custom_skills", "") or ""
            pos_min_exp = getattr(position_row, "min_experience_years", None)
            pos_max_exp = getattr(position_row, "max_experience_years", None)
            
            # Calculate scores
            skill_score = score_skill_match(candidate_skills, pos_skills)
            exp_score = score_experience_match(
                candidate_exp, pos_min_exp, pos_max_exp
            )
            
            # Combined score
            combined_score = (skill_score * 0.6) + (exp_score * 0.4)
            
            if combined_score > 0.3:  # Minimum threshold
                # Convert named tuples to dicts
                candidate_dict = _row_to_dict(candidate_row, bench_df.columns)
                position_dict = _row_to_dict(position_row, positions_df.columns)
                matches.append({
                    "candidate": candidate_dict,
                    "position": position_dict,
                    "skill_score": skill_score,
                    "experience_score": exp_score,
                    "combined_score": combined_score,
                })
    
    # Sort by combined score
    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Generate justifications
    for match in matches[:top_k]:
        match["justification"] = generate_justification(match, metrics_collector=metrics_collector)
    
    return matches[:top_k]


def generate_justification(
    match: Dict,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Generate a descriptive justification for a match using LLM.
    Uses cache to avoid redundant LLM calls for identical matches.
    """
    # Check cache first
    cache = get_cache()
    cached_justification = cache.get_justification(match)
    if cached_justification is not None:
        print(f"[CACHE HIT] Justification - Using cached justification for match", flush=True)
        return cached_justification
    
    # Cache miss - generate justification
    print(f"[CACHE MISS] Justification - Generating new justification", flush=True)
    
    candidate = match.get("candidate", {})
    position = match.get("position", {})
    skill_score = match.get("skill_score", 0)
    exp_score = match.get("experience_score", 0)
    combined_score = match.get("combined_score", 0)
    
    candidate_name = candidate.get("employee_name", "Candidate")
    candidate_skills = candidate.get("custom_skills", "N/A")
    candidate_exp = candidate.get("estimated_experience_years", "N/A")
    
    position_title = position.get("required_designation", "Position")
    position_skills = position.get("custom_skills", "N/A")
    pos_min_exp = position.get("min_experience_years", "N/A")
    pos_max_exp = position.get("max_experience_years", "N/A")
    
    context = f"""Candidate: {candidate_name}
Candidate Skills: {candidate_skills}
Candidate Experience: {candidate_exp} years

Position: {position_title}
Required Skills: {position_skills}
Required Experience: {pos_min_exp} - {pos_max_exp} years

Matching Scores:
- Skill Match Score: {skill_score:.3f} (out of 1.0)
- Experience Match Score: {exp_score:.3f} (out of 1.0)
- Combined Score: {combined_score:.3f} (out of 1.0)
"""
    
    prompt = """Generate a descriptive justification for why this candidate is a good match for this position. 
Include specific details about:
1. How well the candidate's skills align with the position requirements
2. How well the candidate's experience matches the position requirements
3. Overall assessment of the match quality

Be specific and reference the scores provided. Keep it concise (2-3 sentences)."""
    
    try:
        # Use LLM to generate justification
        justify_start_time = time.time()
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful recruiting analyst. Generate concise, descriptive justifications for candidate-position matches."},
                {"role": "user", "content": f"{context}\n\n{prompt}"}
            ],
            max_tokens=200
        )
        
        # Extract metrics
        justify_end_time = time.time()
        justify_time = justify_end_time - justify_start_time
        justify_metrics = extract_usage_from_response(response, "gpt-4o-mini")
        justify_metrics.time_taken = justify_time
        
        # Collect metrics if collector provided
        if metrics_collector is not None:
            metrics_collector.append(justify_metrics)
        
        print(f"[LLM QUERY] Generate Justification", flush=True)
        print(f"  Model: gpt-4o-mini", flush=True)
        print(f"  Time Taken: {justify_time:.3f} seconds", flush=True)
        print(f"  Prompt Tokens: {justify_metrics.prompt_tokens}", flush=True)
        print(f"  Completion Tokens: {justify_metrics.completion_tokens}", flush=True)
        print(f"  Total Tokens: {justify_metrics.total_tokens}", flush=True)
        
        justification = response.choices[0].message.content.strip()
        
        # Store in cache
        cache.set_justification(match, justification)
        print(f"[CACHE STORE] Justification - Stored in cache", flush=True)
        
        return justification
    except Exception:
        # Fallback justification
        justification = f"This candidate has a skill match score of {skill_score:.3f} and experience match score of {exp_score:.3f}, resulting in a combined score of {combined_score:.3f}. The candidate's skills align with the position requirements, and their experience level matches the position's expectations."
        # Cache fallback too
        cache.set_justification(match, justification)
        return justification


def search_jobs_only(
    search_json: Dict,
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Search only in the positions/jobs collection.
    
    Args:
        search_json: Search parameters with keywords and field filters
        query: Original query text for context
        top_k: Number of results to return
        db_path: Optional database path
    """
    # Check cache first
    cache = get_cache()
    cached_response = cache.get_response(query, "job_only", search_json, top_k, db_path=db_path)
    if cached_response is not None:
        print(f"[CACHE HIT] Response - Using cached response for job_only query", flush=True)
        return cached_response
    
    print(f"[CACHE MISS] Response - Generating new response for job_only query", flush=True)
    
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=top_k, db_path=db_path
    )
    
    if positions_df.empty:
        response = f"No positions found for: {query}"
        cache.set_response(query, "job_only", search_json, top_k, response, db_path=db_path)
        return response
    
    # Use LLM to format the response
    context_chunks = []
    for row in positions_df.itertuples():
        designation = getattr(row, "required_designation", "N/A")
        skills = getattr(row, "custom_skills", "N/A")
        min_exp = getattr(row, "min_experience_years", "N/A")
        max_exp = getattr(row, "max_experience_years", "N/A")
        context = f"Position: {designation}, Skills: {skills}, Experience: {min_exp} - {max_exp} years"
        context_chunks.append(context)
    
    prompt = f"Based on the following position information, provide a clear summary answering: {query}"
    answer_text, answer_metrics = answer_question(prompt, context_chunks)
    
    # Collect metrics if collector provided
    if metrics_collector is not None:
        metrics_collector.append(answer_metrics)
    
    # Store in cache
    cache.set_response(query, "job_only", search_json, top_k, answer_text, db_path=db_path)
    print(f"[CACHE STORE] Response - Stored job_only response in cache", flush=True)
    
    return answer_text


def search_candidates_only(
    search_json: Dict,
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Search only in the bench/candidates collection.
    
    Args:
        search_json: Search parameters with keywords and field filters
        query: Original query text for context
        top_k: Number of results to return
        db_path: Optional database path
    """
    # Check cache first
    cache = get_cache()
    cached_response = cache.get_response(query, "candidate_only", search_json, top_k, db_path=db_path)
    if cached_response is not None:
        print(f"[CACHE HIT] Response - Using cached response for candidate_only query", flush=True)
        return cached_response
    
    print(f"[CACHE MISS] Response - Generating new response for candidate_only query", flush=True)
    
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=top_k, db_path=db_path
    )
    
    if bench_df.empty:
        response = f"No candidates found for: {query}"
        cache.set_response(query, "candidate_only", search_json, top_k, response, db_path=db_path)
        return response
    
    # Use LLM to format the response
    context_chunks = []
    for row in bench_df.itertuples():
        employee_name = getattr(row, "employee_name", "N/A")
        skills = getattr(row, "custom_skills", "N/A")
        experience = getattr(row, "estimated_experience_years", "N/A")
        context = f"Employee: {employee_name}, Skills: {skills}, Experience: {experience} years"
        context_chunks.append(context)
    
    prompt = f"Based on the following candidate information, provide a clear summary answering: {query}"
    answer_text, answer_metrics = answer_question(prompt, context_chunks)
    
    # Collect metrics if collector provided
    if metrics_collector is not None:
        metrics_collector.append(answer_metrics)
    
    # Store in cache
    cache.set_response(query, "candidate_only", search_json, top_k, answer_text, db_path=db_path)
    print(f"[CACHE STORE] Response - Stored candidate_only response in cache", flush=True)
    
    return answer_text


def search_both_collections(
    search_json: Dict,
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Search both collections and provide context-aware results.
    
    Args:
        search_json: Search parameters with keywords and field filters
        query: Original query text for context
        top_k: Number of results to return
        db_path: Optional database path
    """
    # Check cache first
    cache = get_cache()
    cached_response = cache.get_response(query, "both", search_json, top_k, db_path=db_path)
    if cached_response is not None:
        print(f"[CACHE HIT] Response - Using cached response for both query", flush=True)
        return cached_response
    
    print(f"[CACHE MISS] Response - Generating new response for both query", flush=True)
    
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=top_k, db_path=db_path
    )
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=top_k, db_path=db_path
    )
    
    if bench_df.empty and positions_df.empty:
        response = f"No results found for: {query}"
        cache.set_response(query, "both", search_json, top_k, response, db_path=db_path)
        return response
    
    # Build context from both collections
    context_chunks = []
    for row in bench_df.itertuples():
        employee_name = getattr(row, "employee_name", "N/A")
        skills = getattr(row, "custom_skills", "N/A")
        experience = getattr(row, "estimated_experience_years", "N/A")
        context = f"Candidate: {employee_name}, Skills: {skills}, Experience: {experience} years"
        context_chunks.append(context)
    
    for row in positions_df.itertuples():
        designation = getattr(row, "required_designation", "N/A")
        skills = getattr(row, "custom_skills", "N/A")
        min_exp = getattr(row, "min_experience_years", "N/A")
        max_exp = getattr(row, "max_experience_years", "N/A")
        context = f"Position: {designation}, Skills: {skills}, Experience: {min_exp} - {max_exp} years"
        context_chunks.append(context)
    
    # Use LLM to provide context-aware answer
    prompt = f"Based on the following information from both candidates and positions, provide a comprehensive answer to: {query}"
    answer_text, answer_metrics = answer_question(prompt, context_chunks)
    
    # Collect metrics if collector provided
    if metrics_collector is not None:
        metrics_collector.append(answer_metrics)
    
    # Store in cache
    cache.set_response(query, "both", search_json, top_k, answer_text, db_path=db_path)
    print(f"[CACHE STORE] Response - Stored both response in cache", flush=True)
    
    return answer_text


def intelligent_match(
    search_json: Dict,
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Search only in the bench/candidates collection.
    
    Args:
        search_json: Search parameters with keywords and field filters
        query: Original query text for context
        top_k: Number of results to return
        db_path: Optional database path
    """
    # Check cache first
    cache = get_cache()
    cached_response = cache.get_response(query, "candidate_only", search_json, top_k, db_path=db_path)
    if cached_response is not None:
        print(f"[CACHE HIT] Response - Using cached response for candidate_only query", flush=True)
        return cached_response
    
    print(f"[CACHE MISS] Response - Generating new response for candidate_only query", flush=True)
    
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=top_k, db_path=db_path
    )
    
    if bench_df.empty:
        response = f"No candidates found for: {query}"
        cache.set_response(query, "candidate_only", search_json, top_k, response, db_path=db_path)
        return response
    
    # Use LLM to format the response
    context_chunks = []
    for row in bench_df.itertuples():
        employee_name = getattr(row, "employee_name", "N/A")
        skills = getattr(row, "custom_skills", "N/A")
        experience = getattr(row, "estimated_experience_years", "N/A")
        context = f"Employee: {employee_name}, Skills: {skills}, Experience: {experience} years"
        context_chunks.append(context)
    
    prompt = f"Based on the following candidate information, provide a clear summary answering: {query}"
    answer_text, answer_metrics = answer_question(prompt, context_chunks)
    
    # Collect metrics if collector provided
    if metrics_collector is not None:
        metrics_collector.append(answer_metrics)
    
    # Store in cache
    cache.set_response(query, "candidate_only", search_json, top_k, answer_text, db_path=db_path)
    print(f"[CACHE STORE] Response - Stored candidate_only response in cache", flush=True)
    
    return answer_text


def search_jobs_only(
    search_json: Dict,
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Search only in the positions/jobs collection.
    
    Args:
        search_json: Search parameters with keywords and field filters
        query: Original query text for context
        top_k: Number of results to return
        db_path: Optional database path
    """
    # Check cache first
    cache = get_cache()
    cached_response = cache.get_response(query, "job_only", search_json, top_k, db_path=db_path)
    if cached_response is not None:
        print(f"[CACHE HIT] Response - Using cached response for job_only query", flush=True)
        return cached_response
    
    print(f"[CACHE MISS] Response - Generating new response for job_only query", flush=True)
    
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=top_k, db_path=db_path
    )
    
    if positions_df.empty:
        response = f"No positions found for: {query}"
        cache.set_response(query, "job_only", search_json, top_k, response, db_path=db_path)
        return response
    
    # Use LLM to format the response
    context_chunks = []
    for row in positions_df.itertuples():
        designation = getattr(row, "required_designation", "N/A")
        skills = getattr(row, "custom_skills", "N/A")
        min_exp = getattr(row, "min_experience_years", "N/A")
        max_exp = getattr(row, "max_experience_years", "N/A")
        context = f"Position: {designation}, Skills: {skills}, Experience: {min_exp} - {max_exp} years"
        context_chunks.append(context)
    
    prompt = f"Based on the following position information, provide a clear summary answering: {query}"
    answer_text, answer_metrics = answer_question(prompt, context_chunks)
    
    # Collect metrics if collector provided
    if metrics_collector is not None:
        metrics_collector.append(answer_metrics)
    
    # Store in cache
    cache.set_response(query, "job_only", search_json, top_k, answer_text, db_path=db_path)
    print(f"[CACHE STORE] Response - Stored job_only response in cache", flush=True)
    
    return answer_text


def search_both_collections(
    search_json: Dict,
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    metrics_collector: Optional[List[LLMMetrics]] = None
) -> str:
    """
    Search both collections and provide context-aware results.
    
    Args:
        search_json: Search parameters with keywords and field filters
        query: Original query text for context
        top_k: Number of results to return
        db_path: Optional database path
    """
    # Check cache first (only for search, not matching)
    cache = get_cache()
    cached_response = cache.get_response(query, "both", search_json, top_k, db_path=db_path)
    if cached_response is not None:
        print(f"[CACHE HIT] Response - Using cached response for both query", flush=True)
        return cached_response
    
    print(f"[CACHE MISS] Response - Generating new response for both query", flush=True)
    
    bench_df = get_semantic_search_results(
        search_json, "onbench", top_k=top_k, db_path=db_path
    )
    positions_df = get_semantic_search_results(
        search_json, "availablepositions", top_k=top_k, db_path=db_path
    )
    
    if bench_df.empty and positions_df.empty:
        response = f"No results found for: {query}"
        cache.set_response(query, "both", search_json, top_k, response, db_path=db_path)
        return response
    
    # Build context from both collections
    context_chunks = []
    
    if not bench_df.empty:
        context_chunks.append("=== Candidates on Bench ===")
        for row in bench_df.itertuples():
            employee_name = getattr(row, "employee_name", "N/A")
            skills = getattr(row, "custom_skills", "N/A")
            experience = getattr(row, "estimated_experience_years", "N/A")
            context = f"Employee: {employee_name}, Skills: {skills}, Experience: {experience} years"
            context_chunks.append(context)
    
    if not positions_df.empty:
        context_chunks.append("=== Available Positions ===")
        for row in positions_df.itertuples():
            designation = getattr(row, "required_designation", "N/A")
            skills = getattr(row, "custom_skills", "N/A")
            min_exp = getattr(row, "min_experience_years", "N/A")
            max_exp = getattr(row, "max_experience_years", "N/A")
            context = f"Position: {designation}, Skills: {skills}, Experience: {min_exp} - {max_exp} years"
            context_chunks.append(context)
    
    # Use LLM to provide context-aware answer
    prompt = f"Based on the following information from both candidates and positions, provide a comprehensive answer to: {query}"
    answer_text, answer_metrics = answer_question(prompt, context_chunks)
    
    # Collect metrics if collector provided
    if metrics_collector is not None:
        metrics_collector.append(answer_metrics)
    
    # Store in cache
    cache.set_response(query, "both", search_json, top_k, answer_text, db_path=db_path)
    print(f"[CACHE STORE] Response - Stored both response in cache", flush=True)
    
    return answer_text


def intelligent_match(
    question: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """
    Main function for intelligent matching.
    
    Uses LLM to understand query intent, then performs appropriate matching or searching.
    
    Returns:
        Tuple of (answer_text, metrics_dict) where metrics_dict contains:
        - total_tokens: Total tokens used across all LLM calls
        - total_time: Total time taken for all LLM calls
        - llm_calls: Number of LLM calls made
        - breakdown: List of individual LLM call metrics
    """
    query_start_time = time.time()
    total_metrics = LLMMetrics()
    llm_calls = []
    
    # Understand query intent (with memoization)
    intent = understand_query_intent(question, db_path=db_path, metrics_collector=llm_calls)
    print(f"[DEBUG] Intent: {intent}")
    direction = intent.get("direction", "both")
    search_json = intent.get("search_json", {"keywords": question, "bench_fields": {}, "position_fields": {}})
    print(f"[DEBUG] Search JSON: {search_json}")
    
    # Handle irrelevant queries
    if direction == "irrelevant_query":
        query_end_time = time.time()
        # Aggregate all collected metrics
        for metrics in llm_calls:
            total_metrics += metrics
        metrics_dict = {
            "total_tokens": total_metrics.total_tokens,
            "total_time_seconds": round(query_end_time - query_start_time, 3),
            "llm_calls": len(llm_calls),
            "breakdown": [m.to_dict() for m in llm_calls]
        }
        return "I'm sorry, but this query is out of scope. I can only help with questions related to bench employees, available positions, and matching candidates to jobs. Please ask about employees, positions, or job matching.", metrics_dict
    
    # Perform matching/searching based on direction
    answer = ""
    if direction == "candidate_to_job":
        matches = match_candidate_to_jobs(search_json, top_k=top_k, db_path=db_path, metrics_collector=llm_calls)
        answer = format_candidate_to_job_response(matches, question)
    
    elif direction == "job_to_candidate":
        matches = match_job_to_candidates(search_json, top_k=top_k, db_path=db_path, metrics_collector=llm_calls)
        answer = format_job_to_candidate_response(matches, question)
    
    elif direction == "candidate_only":
        answer = search_candidates_only(search_json, question, top_k=top_k, db_path=db_path, metrics_collector=llm_calls)
    
    elif direction == "job_only":
        answer = search_jobs_only(search_json, question, top_k=top_k, db_path=db_path, metrics_collector=llm_calls)
    
    elif direction == "both":
        # Check if query suggests matching, otherwise just search both
        question_lower = question.lower()
        if any(word in question_lower for word in ["match", "suitable", "fit", "best", "recommend"]):
            matches = find_best_matches(search_json, top_k=top_k, db_path=db_path, metrics_collector=llm_calls)
            answer = format_best_match_response(matches, question)
        else:
            answer = search_both_collections(search_json, question, top_k=top_k, db_path=db_path, metrics_collector=llm_calls)
    
    else:
        # Fallback to both
        answer = search_both_collections(search_json, question, top_k=top_k, db_path=db_path, metrics_collector=llm_calls)
    
    # Aggregate all collected metrics
    for metrics in llm_calls:
        total_metrics += metrics
    
    # Add metrics summary to answer
    query_end_time = time.time()
    total_time = query_end_time - query_start_time
    
    metrics_summary = f"\n\n---\n[Query Metrics] Total Tokens: {total_metrics.total_tokens}, Total Time: {total_time:.3f}s, LLM Calls: {len(llm_calls)}"
    if llm_calls:
        metrics_summary += "\nBreakdown:"
        for i, call_metrics in enumerate(llm_calls, 1):
            metrics_summary += f"\n  {i}. {call_metrics.model}: {call_metrics.total_tokens} tokens, {call_metrics.time_taken:.3f}s"
    
    metrics_dict = {
        "total_tokens": total_metrics.total_tokens,
        "total_time_seconds": round(total_time, 3),
        "llm_calls": len(llm_calls),
        "breakdown": [m.to_dict() for m in llm_calls]
    }
    
    return answer + metrics_summary, metrics_dict


def format_candidate_to_job_response(matches: List[Dict], question: str) -> str:
    """Format response for candidate-to-job matching."""
    if not matches:
        return f"No suitable positions found for: {question}"
    
    response = f"Found {len(matches)} suitable position(s) for the candidate:\n\n"
    
    for idx, match in enumerate(matches, 1):
        position = match["position"]
        response += f"--- Match #{idx} (Score: {match['combined_score']:.3f}) ---\n"
        response += f"Position: {position.get('required_designation', 'N/A')}\n"
        response += f"Skills Required: {position.get('custom_skills', 'N/A')}\n"
        response += f"Experience: {position.get('min_experience_years', 'N/A')} - {position.get('max_experience_years', 'N/A')} years\n"
        response += f"Skill Score: {match['skill_score']:.3f} | Experience Score: {match['experience_score']:.3f}\n"
        response += f"Justification: {match['justification']}\n\n"
    
    return response


def format_job_to_candidate_response(matches: List[Dict], question: str) -> str:
    """Format response for job-to-candidate matching."""
    if not matches:
        return f"No suitable candidates found for: {question}"
    
    response = f"Found {len(matches)} suitable candidate(s) for the position:\n\n"
    
    for idx, match in enumerate(matches, 1):
        candidate = match["candidate"]
        response += f"--- Match #{idx} (Score: {match['combined_score']:.3f}) ---\n"
        response += f"Candidate: {candidate.get('employee_name', 'N/A')}\n"
        response += f"Skills: {candidate.get('custom_skills', 'N/A')}\n"
        response += f"Experience: {candidate.get('estimated_experience_years', 'N/A')} years\n"
        response += f"Skill Score: {match['skill_score']:.3f} | Experience Score: {match['experience_score']:.3f}\n"
        response += f"Justification: {match['justification']}\n\n"
    
    return response


def format_best_match_response(matches: List[Dict], question: str) -> str:
    """Format response for best match."""
    if not matches:
        return f"No matches found for: {question}"
    
    response = f"Found {len(matches)} best match(es):\n\n"
    
    for idx, match in enumerate(matches, 1):
        candidate = match["candidate"]
        position = match["position"]
        response += f"--- Match #{idx} (Score: {match['combined_score']:.3f}) ---\n"
        response += f"Candidate: {candidate.get('employee_name', 'N/A')}\n"
        response += f"  Skills: {candidate.get('custom_skills', 'N/A')}\n"
        response += f"  Experience: {candidate.get('estimated_experience_years', 'N/A')} years\n"
        response += f"Position: {position.get('required_designation', 'N/A')}\n"
        response += f"  Required Skills: {position.get('custom_skills', 'N/A')}\n"
        response += f"  Experience Range: {position.get('min_experience_years', 'N/A')} - {position.get('max_experience_years', 'N/A')} years\n"
        response += f"Skill Score: {match['skill_score']:.3f} | Experience Score: {match['experience_score']:.3f}\n"
        response += f"Justification: {match['justification']}\n\n"
    
    return response

