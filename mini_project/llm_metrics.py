"""
Helper module for tracking LLM usage metrics (tokens and time).
"""

import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LLMMetrics:
    """Container for LLM call metrics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_taken: float = 0.0  # in seconds
    model: str = ""
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "time_taken_seconds": round(self.time_taken, 3),
            "model": self.model
        }
    
    def __add__(self, other: "LLMMetrics") -> "LLMMetrics":
        """Add two metrics together."""
        return LLMMetrics(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            time_taken=self.time_taken + other.time_taken,
            model=self.model or other.model
        )


def extract_usage_from_response(response, model: str = "") -> LLMMetrics:
    """
    Extract token usage from OpenAI API response.
    
    Handles both chat.completions and responses API formats.
    """
    metrics = LLMMetrics(model=model)
    
    try:
        # Try chat.completions format first
        if hasattr(response, 'usage'):
            usage = response.usage
            metrics.prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            metrics.completion_tokens = getattr(usage, 'completion_tokens', 0)
            metrics.total_tokens = getattr(usage, 'total_tokens', 0)
        # Try responses API format
        elif hasattr(response, 'output') and hasattr(response, 'usage'):
            usage = response.usage
            metrics.prompt_tokens = getattr(usage, 'input_tokens', 0)
            metrics.completion_tokens = getattr(usage, 'output_tokens', 0)
            metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
    except Exception:
        # If we can't extract, return empty metrics
        pass
    
    return metrics


def track_llm_call(func_name: str, model: str = ""):
    """
    Decorator to track LLM call metrics.
    
    Usage:
        @track_llm_call("query_intent", "gpt-4o-mini")
        def my_llm_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            time_taken = end_time - start_time
            
            # Try to extract metrics from result
            metrics = None
            if hasattr(result, '__iter__') and not isinstance(result, str):
                # If result is a tuple (response, metrics), extract both
                if isinstance(result, tuple) and len(result) == 2:
                    response_obj, metrics = result
                    result = response_obj
                else:
                    # Try to extract from response object
                    metrics = extract_usage_from_response(result, model)
            
            if metrics is None:
                metrics = LLMMetrics(model=model)
            
            metrics.time_taken = time_taken
            
            print(f"[LLM METRICS] {func_name} - Model: {model}, "
                  f"Tokens: {metrics.total_tokens} (prompt: {metrics.prompt_tokens}, "
                  f"completion: {metrics.completion_tokens}), "
                  f"Time: {time_taken:.3f}s", flush=True)
            
            return result, metrics
        return wrapper
    return decorator


