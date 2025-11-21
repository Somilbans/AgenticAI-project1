"""
Generate analytical insights and charts from ingested CSV data.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

INSIGHTS_DIR = Path("insights")


def generate_insights(
    bench_csv: Path = Path("mini_output/OnBench.csv"),
    positions_csv: Path = Path("mini_output/AvailablePositions.csv"),
    output_dir: Path = INSIGHTS_DIR,
) -> Dict[str, object]:
    """
    Produce summary statistics and charts for bench and positions data.
    Returns the in-memory summary so callers can re-use it without re-reading disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_df = pd.read_csv(bench_csv)
    positions_df = pd.read_csv(positions_csv)

    bench_summary = {
        "label": "Bench Insights",
        "experience_distribution": _experience_distribution(
            bench_df.get("estimated_experience_years")
        ),
        "skill_distribution": _skill_distribution(bench_df.get("custom_skills")),
    }

    positions_summary = {
        "label": "Positions Insights",
        "experience_distribution": _experience_distribution(
            _positions_experience_series(positions_df)
        ),
        "location_experience": _location_experience(positions_df),
        "grade_experience": _grade_experience(positions_df),
        "skill_distribution": _skill_distribution(positions_df.get("custom_skills")),
    }

    summary = {
        "bench": bench_summary,
        "positions": positions_summary,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # charts
    _plot_experience_distribution(
        bench_summary["experience_distribution"],
        output_dir,
        filename="bench_experience.png",
        title="Bench Experience Distribution",
        color="#2563eb",
    )
    _plot_skill_distribution(
        bench_summary["skill_distribution"],
        output_dir,
        filename="bench_skill_distribution.png",
        title="Bench Skill Distribution (Top)",
        color="#10b981",
    )
    _plot_experience_distribution(
        positions_summary["experience_distribution"],
        output_dir,
        filename="positions_experience.png",
        title="Positions Experience Demand",
        color="#7c3aed",
    )
    _plot_location_experience(positions_summary["location_experience"], output_dir)
    _plot_grade_experience(positions_summary["grade_experience"], output_dir)
    _plot_skill_distribution(
        positions_summary["skill_distribution"],
        output_dir,
        filename="positions_skill_distribution.png",
        title="Position Skill Distribution (Top)",
        color="#f97316",
    )
    return summary


def read_summary(output_dir: Path = INSIGHTS_DIR) -> Dict[str, object]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return generate_insights(output_dir=output_dir)
    summary = json.loads(summary_path.read_text())
    bench = summary.get("bench", {})
    positions = summary.get("positions", {})
    if (
        "skill_distribution" not in bench
        or "skill_distribution" not in positions
    ):
        return generate_insights(output_dir=output_dir)
    return summary


def _experience_distribution(series: pd.Series | None) -> Dict[str, int]:
    bins = [0, 5, 7, 10, 20]
    labels = ["0-5 yrs", "5-7 yrs", "7-10 yrs", "10-20 yrs"]
    if series is None:
        return {label: 0 for label in labels}
    buckets = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    counts = buckets.value_counts().sort_index()
    return {label: int(counts.get(label, 0)) for label in labels}


def _positions_experience_series(df: pd.DataFrame) -> pd.Series:
    if "min_experience_years" not in df or "max_experience_years" not in df:
        return pd.Series(dtype=float)
    avg = df[["min_experience_years", "max_experience_years"]].mean(axis=1, skipna=True)
    avg = avg.fillna(df["min_experience_years"]).fillna(df["max_experience_years"])
    return avg.fillna(0.0)


def _skill_distribution(
    series: pd.Series | None, top_n: int | None = None
) -> List[Dict[str, object]]:
    counter = _skill_counter(series)
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is None:
        top_items = items
    else:
        top_items = items[:top_n]
    return [{"skill": skill, "count": count} for skill, count in top_items]


def _location_experience(df: pd.DataFrame) -> List[Dict[str, float]]:
    if "location" not in df:
        return []
    grouped = df.groupby("location")[
        ["min_experience_years", "max_experience_years"]
    ].mean(numeric_only=True)
    summary = (
        grouped.dropna(how="all")
        .reset_index()
        .rename(
            columns={
                "location": "label",
                "min_experience_years": "avg_min",
                "max_experience_years": "avg_max",
            }
        )
    )
    summary = summary.fillna(0.0)
    return summary.to_dict(orient="records")


def _grade_experience(df: pd.DataFrame) -> List[Dict[str, float]]:
    if "grade" not in df:
        return []
    grouped = df.groupby("grade")[
        ["min_experience_years", "max_experience_years"]
    ].mean(numeric_only=True)
    records = []
    for grade, row in grouped.dropna(how="all").iterrows():
        try:
            grade_numeric = float(str(grade).split()[0].replace("+", ""))
        except ValueError:
            grade_numeric = None
        records.append(
            {
                "grade": grade,
                "grade_numeric": grade_numeric,
                "avg_min": float(row.get("min_experience_years", 0.0) or 0.0),
                "avg_max": float(row.get("max_experience_years", 0.0) or 0.0),
            }
        )
    records.sort(
        key=lambda item: item["grade_numeric"]
        if item["grade_numeric"] is not None
        else float("inf")
    )
    return records


def _collect_skills(series: pd.Series | None) -> List[str]:
    tokens: List[str] = []
    if series is None:
        return tokens
    for cell in series.dropna():
        tokens.extend(
            [skill.strip() for skill in str(cell).split(",") if skill.strip()]
        )
    return tokens


def _skill_counter(series: pd.Series | None) -> Counter:
    return Counter(map(str.lower, _collect_skills(series)))


def _plot_experience_distribution(
    data: Dict[str, int],
    output_dir: Path,
    filename: str,
    title: str,
    color: str,
) -> None:
    labels = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(value),
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def _plot_location_experience(data: List[Dict[str, float]], output_dir: Path) -> None:
    if not data:
        return
    labels = [item["label"] for item in data]
    min_vals = [item["avg_min"] for item in data]
    max_vals = [item["avg_max"] for item in data]
    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(
        [i - width / 2 for i in x],
        min_vals,
        width=width,
        label="Avg Min",
        color="#06b6d4",
    )
    plt.bar(
        [i + width / 2 for i in x],
        max_vals,
        width=width,
        label="Avg Max",
        color="#0ea5e9",
    )
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylabel("Years")
    plt.title("Positions: Avg Experience by Location")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "location_experience.png")
    plt.close()


def _plot_grade_experience(data: List[Dict[str, float]], output_dir: Path) -> None:
    if not data:
        return
    labels = [item["grade"] for item in data]
    min_vals = [item["avg_min"] for item in data]
    max_vals = [item["avg_max"] for item in data]
    plt.figure(figsize=(7, 4))
    plt.plot(labels, min_vals, marker="o", label="Avg Min", color="#a855f7")
    plt.plot(labels, max_vals, marker="o", label="Avg Max", color="#3b82f6")
    plt.fill_between(labels, min_vals, max_vals, color="#c4b5fd", alpha=0.3)
    plt.ylabel("Years")
    plt.xlabel("Grade")
    plt.title("Positions: Grade vs Experience")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "grade_experience.png")
    plt.close()


def _plot_skill_distribution(
    data: List[Dict[str, object]],
    output_dir: Path,
    filename: str,
    title: str,
    color: str,
) -> None:
    if not data:
        return
    labels = [item["skill"] for item in data]
    values = [item["count"] for item in data]
    plt.figure(figsize=(7, 5))
    bars = plt.barh(labels[::-1], values[::-1], color=color)
    plt.title(title)
    plt.xlabel("Mentions")
    for bar, value in zip(bars, values[::-1]):
        plt.text(
            value + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
        )
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()

