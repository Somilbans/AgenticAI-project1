"""
Flask server and helpers that power the React UI + API endpoints.
"""

from __future__ import annotations

import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory

from mini_project.cli_query import (
    answer_from_cli,
    resolve_collection,
    resolve_top_k,
)
from mini_project.insights import INSIGHTS_DIR, generate_insights, read_summary
from mini_project.vector_store import list_collections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "ui"


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(UI_DIR),
        static_url_path="",
    )

    @app.post("/api/query")
    def query_endpoint():
        payload = request.get_json(force=True)
        question = payload.get("question", "").strip()
        collection = payload.get("collection")
        top_k = payload.get("top_k")
        db_path = _resolve_db_path(payload.get("db_path"))

        if not question:
            return jsonify({"error": "Question is required."}), 400

        try:
            chosen_collection = resolve_collection(
                collection, db_path, interactive=False
            )
            resolved_top_k = resolve_top_k(top_k, interactive=False)
            answer = answer_from_cli(
                question,
                chosen_collection,
                resolved_top_k,
                db_path,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify(
            {
                "answer": answer,
                "meta": {
                    "collection": chosen_collection,
                    "top_k": resolved_top_k,
                    "db_path": str(db_path) if db_path else None,
                },
            }
        )

    @app.route("/api/collections", methods=["GET"])
    def collections_endpoint():
        db_path = _resolve_db_path(request.args.get("db_path"))
        try:
            names = list(list_collections(db_path=db_path))
        except Exception as exc:
            return jsonify({"collections": [], "error": str(exc)}), 400
        return jsonify({"collections": names})

    @app.get("/api/insights")
    def insights_endpoint():
        summary = read_summary()
        return jsonify(summary)

    @app.route("/")
    def serve_index():
        return send_from_directory(UI_DIR, "index.html")

    @app.route("/<path:asset_path>")
    def serve_static(asset_path: str):
        return send_from_directory(UI_DIR, asset_path)

    @app.route("/insights/<path:asset_path>")
    def serve_insight_assets(asset_path: str):
        return send_from_directory(INSIGHTS_DIR, asset_path)

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    auto_open: bool = True,
) -> None:
    """
    Start the Flask server, optionally auto-opening the UI in a browser.
    """
    app = create_app()
    if auto_open:
        _open_browser_async(_build_ui_url(host, port))
    app.run(host=host, port=port, debug=debug)


def _build_ui_url(host: str, port: int) -> str:
    visible_host = "127.0.0.1" if host == "0.0.0.0" else host
    return f"http://{visible_host}:{port}/"


def _open_browser_async(url: str) -> None:
    def _runner():
        time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_runner, daemon=True).start()


def _resolve_db_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    return Path(raw_path)


if __name__ == "__main__":
    run_server()

