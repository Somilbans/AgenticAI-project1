"""
CLI helpers for querying Chroma and summarizing results with an LLM.
"""

from pathlib import Path
from typing import Optional

import typer

from mini_project.llm import answer_question
from mini_project.vector_store import list_collections, query_collection


def answer_from_cli(
    question: str,
    collection: str,
    top_k: int,
    db_path: Optional[Path],
) -> str:
    docs = query_collection(
        question,
        collection_name=collection,
        top_k=top_k,
        db_path=db_path,
    )
    if not docs:
        raise ValueError("No results found in Chroma.")
    answer_text, _ = answer_question(question, docs)
    return answer_text


def resolve_top_k(value: Optional[int], interactive: bool = True) -> int:
    """
    Determine the top_k value, optionally prompting the user if not provided.
    """
    if value is not None:
        if value < 1:
            raise ValueError("top-k must be >= 1.")
        return value

    if not interactive:
        return 3

    while True:
        entry = typer.prompt(
            "Enter top-k (number of rows to retrieve)", default="3"
        ).strip()
        if not entry.isdigit():
            typer.echo("Please enter a positive integer.")
            continue
        top_k = int(entry)
        if top_k < 1:
            typer.echo("top-k must be >= 1.")
            continue
        return top_k


def resolve_collection(
    requested: Optional[str],
    db_path: Optional[Path],
    interactive: bool = True,
) -> str:
    """
    Return the requested collection, or prompt the user if not provided.
    """
    collections = list(list_collections(db_path=db_path))
    if not collections:
        raise ValueError("No collections found. Run the ingest command first.")

    if requested:
        if requested not in collections:
            raise ValueError(
                f"Collection '{requested}' not found. Available: {', '.join(collections)}"
            )
        return requested

    if not interactive:
        return collections[0]

    typer.echo("Available collections:")
    for idx, name in enumerate(collections, start=1):
        typer.echo(f" [{idx}] {name}")
    selection = typer.prompt("Select collection number").strip()
    if not selection.isdigit():
        raise ValueError("Collection selection must be a number.")
    index = int(selection) - 1
    if not 0 <= index < len(collections):
        raise ValueError("Selected number is out of range.")
    return collections[index]


def ask_command(
    question: str = typer.Argument(..., help="Natural language question."),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Chroma collection to search (leave empty to pick interactively).",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="How many rows to retrieve (prompted if omitted).",
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db-path",
        "-d",
        help="Directory containing the Chroma database (defaults to config value).",
    ),
) -> None:
    """
    Ask a single question against the Chroma database.
    """
    try:
        chosen_collection = resolve_collection(collection, db_path)
        resolved_top_k = resolve_top_k(top_k)
        answer = answer_from_cli(question, chosen_collection, resolved_top_k, db_path)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    typer.echo("Answer:\n")
    typer.echo(answer)


def chat_command(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection to search during the session (leave empty to pick at start).",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="How many rows to retrieve (prompted if omitted).",
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db-path",
        "-d",
        help="Directory containing the Chroma database (defaults to config value).",
    ),
) -> None:
    """
    Start an interactive Q&A loop backed by the Chroma database.
    """
    typer.echo("Interactive mode. Choose an option each round.\n")
    try:
        chosen_collection = resolve_collection(collection, db_path)
        current_top_k = resolve_top_k(top_k)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    while True:
        typer.echo(
            f"Options: [1] Ask question  [2] Change top-k (current={current_top_k})  [q] Quit"
        )
        choice = typer.prompt("Select option").strip().lower()
        if choice in {"q", "quit"}:
            typer.echo("Goodbye!")
            break
        if choice == "2":
            current_top_k = resolve_top_k(None)
            continue
        if choice not in {"", "1"}:
            typer.echo("Unrecognized option. Please choose 1, 2, or q.")
            continue
        question = typer.prompt("Question").strip()
        if not question:
            typer.echo("Please enter a question or choose Quit.")
            continue
        try:
            answer = answer_from_cli(question, chosen_collection, current_top_k, db_path)
        except ValueError as exc:
            typer.echo(str(exc))
            continue
        typer.echo(f"\nAnswer:\n{answer}\n")

