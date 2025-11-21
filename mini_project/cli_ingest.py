"""
CLI helpers for ingesting spreadsheets and storing them in Chroma.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import typer

from mini_project import config
from mini_project.llm import answer_question
from mini_project.reader import load_workbook
from mini_project.report import print_summary
from mini_project.vector_store import upsert_dataframe
from mini_project.writer import save_as_csv


def _process_file(file_path: Path, output_dir: Path) -> Tuple[Path, int, int]:
    """
    Load a single spreadsheet, save the DataFrame, and return stats.
    """
    df = load_workbook(file_path)
    stem = file_path.stem.lower()
    if stem == "onbench":
        df = _enrich_bench_dataframe(df)
    elif stem == "availablepositions":
        df = _enrich_available_positions(df)
    destination = save_as_csv(df, file_path, output_dir)
    upsert_dataframe(df, collection_name=file_path.stem.lower())
    return destination, len(df), len(df.columns)


def _process_all(files: List[Path], output_dir: Path) -> List[Tuple[Path, int, int]]:
    """
    Iterate through every configured file and collect their summaries.
    """
    summaries = []
    for path in files:
        if not path.exists():
            typer.echo(f"Skipping missing file: {path}")
            continue
        summaries.append(_process_file(path, output_dir))
    return summaries


def ingest_command(
    files: List[Path] = typer.Argument(
        None,
        help="Spreadsheet files to ingest (defaults to config.FILES_TO_READ).",
    ),
    output_dir: Path = typer.Option(
        config.OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory where CSV copies will be written.",
    ),
) -> None:
    """
    Ingest spreadsheets, serialize them, and upsert rows into Chroma.
    """
    targets = files or config.FILES_TO_READ
    results = _process_all(targets, output_dir)
    if not results:
        typer.echo("No files processed. Check the inputs and try again.")
        raise typer.Exit(code=1)
    print_summary(results)


def _enrich_bench_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    bench_df = df.copy()

    def _combine_skills(row: pd.Series) -> str | None:
        combined = _parse_list(row.get("employee_skills")) + _parse_list(
            row.get("pi_skills")
        )
        unique_skills = _unique_preserve(combined)
        return ", ".join(unique_skills) if unique_skills else None

    def _dedupe_skill_groups(cell: object) -> str | None:
        groups = _unique_preserve(_parse_list(cell))
        return ", ".join(groups) if groups else None

    bench_df["custom_skills"] = bench_df.apply(_combine_skills, axis=1)
    bench_df["custom_skill_groups"] = bench_df["piskill_group"].apply(
        _dedupe_skill_groups
    )
    bench_df["estimated_experience_years"] = bench_df.apply(
        _estimate_bench_experience, axis=1
    )

    return bench_df


def _enrich_available_positions(df: pd.DataFrame) -> pd.DataFrame:
    positions_df = df.copy()

    def _extract(row: pd.Series) -> Dict[str, Optional[float]]:
        context = "\n".join(
            filter(
                None,
                [
                    f"Grade: {row.get('grade')}",
                    f"Designation: {row.get('required_designation')}",
                    f"Experience field: {row.get('experience_required')}",
                    f"Skills: {row.get('skill_description')}",
                    f"Responsibilities: {row.get('responsibilities')}",
                ],
            )
        )
        prompt = (
            "Read the job requirement and extract the minimum and maximum total years "
            "of experience mentioned or implied. Respond strictly as JSON with keys "
            '\"min\" and \"max\" using numeric values (decimals allowed) or null when unknown.'
        )
        result = _query_llm_for_json(prompt, context)
        return {
            "min": result.get("min") if isinstance(result, dict) else None,
            "max": result.get("max") if isinstance(result, dict) else None,
        }

    extracted = positions_df.apply(_extract, axis=1)

    def _apply_experience(row: pd.Series, llm_data: Dict[str, Optional[float]]) -> Tuple[float, float]:
        text_min, text_max = _parse_experience_text(str(row.get("experience_required") or ""))
        min_val = llm_data.get("min")
        max_val = llm_data.get("max")
        if min_val is None:
            min_val = text_min
        if max_val is None:
            max_val = text_max
        if min_val is None and max_val is not None:
            min_val = max_val
        if max_val is None and min_val is not None:
            max_val = min_val
        if min_val is None:
            min_val = 0.0
        if max_val is None:
            max_val = min_val
        return float(min_val), float(max_val)

    experience_pairs = [
        _apply_experience(row, llm_data)
        for (_, row), llm_data in zip(positions_df.iterrows(), extracted)
    ]
    positions_df["min_experience_years"] = [pair[0] for pair in experience_pairs]
    positions_df["max_experience_years"] = [pair[1] for pair in experience_pairs]

    def _collect_skills(row: pd.Series) -> str | None:
        raw_values = [
            row.get("skill1"),
            row.get("skill2"),
            row.get("skill3"),
            row.get("skill_description"),
        ]
        tokens = []
        for raw in raw_values:
            tokens.extend(_parse_list(raw))
        unique_tokens = _unique_preserve(tokens)
        return ", ".join(unique_tokens) if unique_tokens else None

    positions_df["custom_skills"] = positions_df.apply(_collect_skills, axis=1)
    return positions_df


def _estimate_bench_experience(row: pd.Series) -> Optional[float]:
    context = "\n".join(
        filter(
            None,
            [
                f"Grade/Subgrade: {row.get('grade_subgrade')}",
                f"Employee track: {row.get('employee_track')}",
                f"Custom skills: {row.get('custom_skills') or row.get('employee_skills')}",
            ],
        )
    )
    prompt = (
        "Estimate the total years of professional experience for this bench employee "
        "based on grade, track, and skills. Respond strictly as JSON "
        'like {\"years\": 7.5}. Use null if you cannot estimate.'
    )
    data = _query_llm_for_json(prompt, context)
    value = data.get("years") if isinstance(data, dict) else None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        fallback = _extract_float(value)
        if fallback is not None:
            return fallback
    # heuristics using grade/track
    heuristic = _grade_to_experience(str(row.get("grade_subgrade") or ""))
    if heuristic is not None:
        return heuristic
    return 3.0


def _query_llm_for_json(question: str, context: str) -> Dict[str, Optional[float]]:
    try:
        response_text = answer_question(question, [context])
    except Exception as exc:
        typer.echo(f"[LLM] Falling back due to error: {exc}")
        return {}
    response_text = response_text.strip()
    # attempt to parse json
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # fallback parse e.g. "min: 5, max: 8"
        matches = re.findall(r"(min|max|years)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", response_text, flags=re.IGNORECASE)
        data: Dict[str, Optional[float]] = {}
        for key, value in matches:
            data[key.lower()] = float(value)
        if not data:
            number = _extract_float(response_text)
            if number is not None:
                data["years"] = number
        return data


def _extract_float(text: str) -> Optional[float]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(text))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _parse_experience_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract min/max from phrases such as "Between 12 to 15 Years", "8+ years".
    """
    normalized = text.lower()
    between = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:to|-\s*|–)\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if between:
        return float(between.group(1)), float(between.group(2))

    at_least = re.search(r"(?:at\s+least|\bmin(?:imum)?)\s*(\d+(?:\.\d+)?)", normalized)
    if at_least:
        value = float(at_least.group(1))
        return value, value

    plus = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years|yrs)?", normalized)
    if plus:
        value = float(plus.group(1))
        return value, value

    single = re.search(r"(\d+(?:\.\d+)?)\s*(?:years|yrs)", normalized)
    if single:
        value = float(single.group(1))
        return value, value

    return None, None


def _grade_to_experience(grade: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)", grade)
    if not match:
        return None
    value = float(match.group(1))
    # simple mapping: grade number * 2 for years, minimum 1
    return max(1.0, round(value * 2, 1))


def _parse_list(cell: object) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    parts = [item.strip() for item in str(cell).split(",")]
    return [item for item in parts if item]


def _unique_preserve(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

