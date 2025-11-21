"""
Unified CLI entry point for ingesting spreadsheets and querying Chroma.
"""

import json
from pathlib import Path

import typer

from mini_project.cli_ingest import ingest_command
from mini_project.cli_query import ask_command, chat_command
from mini_project.insights import generate_insights
from mini_project.server import run_server

app = typer.Typer(help="Mini project CLI for data ingestion, Q&A, and server.")
app.command("ingest")(ingest_command)
app.command("ask")(ask_command)
app.command("chat")(chat_command)


@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host interface for the FastAPI server.",
    ),
    port: int = typer.Option(
        5000,
        "--port",
        "-p",
        help="Port for the FastAPI server.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Run FastAPI with debug mode (better error messages, detailed logging).",
    ),
    reload: bool = typer.Option(
        True,
        "--reload/--no-reload",
        help="Enable auto-reload on file changes (works with --debug).",
    ),
) -> None:
    """
    Run the FastAPI server that powers the React UI.
    """
    run_server(host=host, port=port, debug=debug, auto_open=True, use_reloader=reload)


@app.command()
def insight(
    bench_csv: Path = typer.Option(
        Path("mini_output/OnBench.csv"),
        "--bench",
        help="Path to bench CSV file.",
    ),
    positions_csv: Path = typer.Option(
        Path("mini_output/AvailablePositions.csv"),
        "--positions",
        help="Path to positions CSV file.",
    ),
) -> None:
    """
    Generate insight charts and summary JSON.
    """
    summary = generate_insights(bench_csv=bench_csv, positions_csv=positions_csv)
    typer.echo("Generated insights:")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()

