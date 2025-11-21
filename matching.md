# Matching Functions Documentation

This document explains the skill and experience matching functions used to match bench employees with available positions.

## Overview

The matching system uses two scoring functions to evaluate compatibility:
1. **Skill Matching** - Evaluates how well an employee's skills align with position requirements
2. **Experience Matching** - Evaluates how well an employee's experience matches position requirements

Both functions return scores between 0.0 (no match) and 1.0 (perfect match).

---

## 1. Skill Matching: `score_skill_match()`

### Purpose
Determines how well a bench employee's skills match the skills required for a position using **LLM-based semantic similarity** with fuzzy matching capabilities.

### How It Works

The function uses **LLM embeddings** (OpenAI's text-embedding-3-small model) to understand the semantic meaning of skills, allowing it to match similar skills even when they're written differently. It then applies an **F1-score approach**, which balances two important metrics:

#### Precision
- **Definition**: How many of the employee's skills are relevant to the position
- **Formula**: `(Matching Skills) / (Total Employee Skills)`
- **Example**: If an employee has 10 skills and 6 match the position, precision = 6/10 = 0.6

#### Recall
- **Definition**: How completely the employee covers the position's required skills
- **Formula**: `(Matching Skills) / (Total Required Skills)`
- **Example**: If a position requires 8 skills and the employee has 6 of them, recall = 6/8 = 0.75

#### LLM-Based Semantic Matching
- **Embeddings**: Each skill is converted to a vector using OpenAI embeddings
- **Cosine Similarity**: Measures semantic similarity between skill vectors (0.0 to 1.0)
- **Fuzzy Matching**: Skills with similarity ≥ threshold (default 0.7) are considered matches
- **Exact Match Priority**: Exact string matches get similarity = 1.0 automatically
- **Weighted Scoring**: Matches are weighted by their similarity score

#### F1 Score
- **Definition**: Harmonic mean of precision and recall (weighted by similarity)
- **Formula**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Weighted Precision**: `Sum of similarity scores / Total employee skills`
- **Weighted Recall**: `Sum of similarity scores / Total required skills`
- **Why F1 Score?**: It balances both metrics. A high F1 score means:
  - The employee's skills are relevant (high precision)
  - The employee covers most required skills (high recall)

### Fuzzy Matching Examples

**Example 1: Exact Match**
- Employee: "Python, React, SQL"
- Position: "Python, React, SQL"
- Result: All skills match exactly → F1 = 1.0

**Example 2: Semantic Similarity Match**
- Employee: "JavaScript, React.js, Node.js"
- Position: "JS, React, Node"
- Result: LLM recognizes semantic similarity:
  - "JavaScript" ≈ "JS" (similarity ~0.95)
  - "React.js" ≈ "React" (similarity ~0.92)
  - "Node.js" ≈ "Node" (similarity ~0.94)
  - F1 score based on weighted similarities

**Example 3: Related Skills Match**
- Employee: "Machine Learning, Deep Learning, Neural Networks"
- Position: "ML, AI, Data Science"
- Result: LLM recognizes related concepts:
  - "Machine Learning" ≈ "ML" (similarity ~0.88)
  - "Deep Learning" ≈ "AI" (similarity ~0.75)
  - "Neural Networks" ≈ "Data Science" (similarity ~0.72)
  - F1 score reflects semantic relationships

**Example 4: No Match (Below Threshold)**
- Employee: "Java, C++, C#"
- Position: "Python, React, SQL"
- Result: Similarities all < 0.7 threshold → F1 = 0.0

### Fallback to Exact Matching

If LLM-based matching fails (API error, no API key, etc.), the function automatically falls back to exact string matching, ensuring the system always works even without LLM access.

### Function Signature
```python
def score_skill_match(
    bench_skills: str | None,
    position_skills: str | None,
    *,
    use_llm: bool = True,
    similarity_threshold: float = 0.7,
    exact_match_fallback: bool = True,
) -> float
```

### Parameters
- `bench_skills`: Comma-separated string of employee skills (e.g., "Python, React, SQL")
- `position_skills`: Comma-separated string of required skills (e.g., "Python, JavaScript")
- `use_llm`: Whether to use LLM embeddings for semantic matching (default: True)
- `similarity_threshold`: Minimum cosine similarity for fuzzy match (default: 0.7)
  - Lower threshold (e.g., 0.6) = more lenient matching
  - Higher threshold (e.g., 0.8) = stricter matching
- `exact_match_fallback`: Fall back to exact matching if LLM fails (default: True)

### Returns
- Score between 0.0 (no match) and 1.0 (perfect match)

### Notes
- **LLM-Based Matching**: Uses OpenAI's `text-embedding-3-small` model for semantic understanding
- **Fuzzy Matching**: Matches similar skills (e.g., "JS" = "JavaScript", "ML" = "Machine Learning")
- **Exact Match Priority**: Exact string matches automatically get similarity = 1.0
- **Weighted Scoring**: Matches are weighted by their similarity score for more accurate F1 calculation
- **Fallback**: Automatically falls back to exact matching if LLM is unavailable
- **Normalization**: Skills are normalized (lowercased and trimmed) before comparison
- **Empty Handling**: Empty or None values return 0.0
- **Case-Insensitive**: Matching is case-insensitive

### Configuration

**Similarity Threshold Guidelines:**
- **0.9-1.0**: Very strict (almost exact matches only)
- **0.7-0.9**: Balanced (recommended default)
- **0.5-0.7**: Lenient (matches related but different skills)
- **< 0.5**: Very lenient (may match unrelated skills)

**When to Adjust Threshold:**
- **Lower threshold**: Use when skills are written in very different formats/styles
- **Higher threshold**: Use when you want stricter matching and exact skill alignment

---

## 2. Experience Matching: `score_experience_match()`

### Purpose
Determines how well a bench employee's years of experience match the experience requirements for a position.

### How It Works

The function uses a **tolerance-based scoring system**:

#### Perfect Match (Score = 1.0)
- Employee's experience falls within the required range: `min_experience ≤ employee_experience ≤ max_experience`

#### Near Match (Score = 0.5 to 1.0)
- Employee is slightly below minimum: Experience is within tolerance (default 2 years)
  - Score decreases linearly: `1.0 - (distance / tolerance) × 0.5`
- Employee is slightly above maximum: Experience is within tolerance
  - Score decreases slightly: `1.0 - (distance / tolerance) × 0.3`

#### Poor Match (Score < 0.5)
- Employee is far below minimum: Beyond tolerance
  - Score decreases sharply
- Employee is far above maximum: Beyond tolerance
  - Score decreases with penalty

#### Special Cases
- **No experience requirement**: If both min and max are None, returns 1.0
- **No maximum limit**: If max is None/Infinity and employee exceeds min, returns 1.0 (more experience is usually acceptable)

### Example Scenarios

**Scenario 1: Perfect Match**
- Employee experience: 5 years
- Position: min=4, max=6 years
- Result: 1.0 (within range)

**Scenario 2: Near Match (Below Minimum)**
- Employee experience: 3 years
- Position: min=4, max=6 years
- Tolerance: 2 years
- Distance: 1 year below
- Result: 1.0 - (1/2) × 0.5 = 0.75

**Scenario 3: Near Match (Above Maximum)**
- Employee experience: 7 years
- Position: min=4, max=6 years
- Tolerance: 2 years
- Distance: 1 year above
- Result: 1.0 - (1/2) × 0.3 = 0.85

**Scenario 4: Poor Match (Too Far Below)**
- Employee experience: 1 year
- Position: min=4, max=6 years
- Tolerance: 2 years
- Distance: 3 years below (1 year beyond tolerance)
- Result: 0.5 - (3-2)/(2×2) = 0.25

**Scenario 5: No Maximum Limit**
- Employee experience: 10 years
- Position: min=5, max=None
- Result: 1.0 (more experience is acceptable)

### Function Signature
```python
def score_experience_match(
    bench_experience: float | None,
    position_min_experience: float | None,
    position_max_experience: float | None,
    tolerance: float = 2.0,
) -> float
```

### Parameters
- `bench_experience`: Employee's estimated years of experience
- `position_min_experience`: Minimum years required for the position
- `position_max_experience`: Maximum years required (None means no max)
- `tolerance`: Allowed deviation in years (default: 2.0)

### Returns
- Score between 0.0 (poor match) and 1.0 (perfect match)

### Notes
- Tolerance allows flexibility for near-matches
- Default tolerance is 2 years
- Handles None values gracefully
- More experience is generally acceptable (less penalty than insufficient experience)

---

## Usage Example

```python
from mini_project.matching import score_skill_match, score_experience_match

# Example employee and position data
employee_skills = "Python, React.js, SQL, JavaScript"
position_skills = "Python, React, SQL, JS"
employee_experience = 5.0
position_min_exp = 4.0
position_max_exp = 6.0

# Calculate scores with LLM-based fuzzy matching (default)
skill_score = score_skill_match(employee_skills, position_skills)
# This will match "React.js" ≈ "React" and "JavaScript" ≈ "JS" using semantic similarity

# Or customize the matching behavior
skill_score_custom = score_skill_match(
    employee_skills, 
    position_skills,
    use_llm=True,              # Use LLM embeddings
    similarity_threshold=0.75, # Stricter matching
    exact_match_fallback=True # Fallback if LLM fails
)

# Use exact matching only (no LLM)
skill_score_exact = score_skill_match(
    employee_skills,
    position_skills,
    use_llm=False
)

experience_score = score_experience_match(
    employee_experience, 
    position_min_exp, 
    position_max_exp
)

# Combined score (weighted)
combined_score = (skill_score * 0.6) + (experience_score * 0.4)

print(f"Skill Match (LLM): {skill_score:.3f}")
print(f"Skill Match (Exact): {skill_score_exact:.3f}")
print(f"Experience Match: {experience_score:.3f}")
print(f"Combined Score: {combined_score:.3f}")
```

### Environment Setup

To use LLM-based matching, set the OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

---

## Design Decisions

### Why F1 Score for Skills?
- **Balanced Evaluation**: Considers both relevance (precision) and completeness (recall)
- **Prevents Over-Skilling Bias**: An employee with 20 skills but only 2 matching gets low precision
- **Prevents Under-Skilling Bias**: An employee with 2 skills matching 2 required gets perfect recall but low precision if they have many irrelevant skills

### Why Tolerance for Experience?
- **Real-World Flexibility**: Exact matches are rare; tolerance allows for reasonable near-matches
- **Asymmetric Penalties**: Being slightly over-qualified is less penalized than being under-qualified
- **Practical Hiring**: Employers often accept candidates slightly outside the stated range

### Score Interpretation
- **0.8 - 1.0**: Excellent match, highly recommended
- **0.6 - 0.8**: Good match, recommended
- **0.4 - 0.6**: Moderate match, consider with caution
- **0.2 - 0.4**: Poor match, not recommended
- **0.0 - 0.2**: Very poor match, avoid

---

## Implementation Details

### LLM Embedding Process

1. **Skill Normalization**: All skills are normalized (lowercase, trimmed)
2. **Embedding Generation**: Skills are converted to vectors using OpenAI's embedding model
3. **Similarity Calculation**: Cosine similarity is computed between all skill pairs
4. **Match Detection**: Skills with similarity ≥ threshold are considered matches
5. **Weighted Scoring**: Matches contribute to F1 score weighted by their similarity

### Performance Considerations

- **API Calls**: Each matching operation requires one API call to get embeddings
- **Batch Processing**: All skills are embedded in a single API call for efficiency
- **Caching**: Consider caching embeddings for frequently used skills (not implemented yet)
- **Fallback**: Exact matching is fast and requires no API calls

### Error Handling

- **API Failures**: Automatically falls back to exact matching if LLM API fails
- **Missing API Key**: Falls back to exact matching with a warning
- **Invalid Input**: Returns 0.0 for empty or None inputs
- **Network Issues**: Gracefully handles timeouts and connection errors

## Future Enhancements

Potential improvements to consider:
1. **Embedding Caching**: Cache skill embeddings to reduce API calls
2. **Skill Weighting**: Assign importance weights to required skills
3. **Experience Context**: Consider domain-specific experience vs. general experience
4. **Multi-Model Support**: Support for different embedding models
5. **Batch Optimization**: Optimize embedding calls for large skill sets
6. **Custom Thresholds**: Per-skill similarity thresholds based on importance

