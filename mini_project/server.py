"""
FastAPI server and helpers that power the React UI + API endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mini_project.cli_query import (
    answer_from_cli,
    resolve_collection,
    resolve_top_k,
)
from mini_project.insights import INSIGHTS_DIR, generate_insights, read_summary
from mini_project.vector_store import list_collections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "ui"


# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    mode: str = "collection"  # "collection" or "intelligent"
    collection: Optional[str] = None
    top_k: Optional[int] = None
    db_path: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    meta: dict


def create_app() -> FastAPI:
    app = FastAPI(
        title="Bench Intelligence Hub API",
        description="API for intelligent matching between bench employees and positions",
        version="1.0.0",
    )
    
    # Enable CORS for all routes
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for UI
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")
    
    # Mount insights assets
    if INSIGHTS_DIR.exists():
        app.mount("/insights", StaticFiles(directory=str(INSIGHTS_DIR)), name="insights")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_endpoint(payload: QueryRequest):
        question = payload.question.strip()
        mode = payload.mode
        collection = payload.collection
        top_k = payload.top_k
        db_path = _resolve_db_path(payload.db_path)

        if not question:
            raise HTTPException(status_code=400, detail="Question is required.")

        try:
            if mode == "intelligent":
                # Use intelligent matching system
                from mini_project.intelligent_match import intelligent_match
                
                resolved_top_k = resolve_top_k(top_k, interactive=False)
                answer, metrics = intelligent_match(
                    question,
                    top_k=resolved_top_k,
                    db_path=db_path,
                )
                chosen_collection = "intelligent"
            else:
                # Collection-based query
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
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        meta = {
            "collection": chosen_collection,
            "top_k": resolved_top_k,
            "db_path": str(db_path) if db_path else None,
        }
        
        # Add metrics if available (for intelligent mode)
        if mode == "intelligent" and isinstance(answer, tuple):
            answer, metrics = answer
            meta["metrics"] = metrics
        
        return QueryResponse(
            answer=answer,
            meta=meta,
        )

    @app.get("/api/collections")
    async def collections_endpoint(db_path: Optional[str] = None):
        resolved_db_path = _resolve_db_path(db_path)
        try:
            names = list(list_collections(db_path=resolved_db_path))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"collections": names}

    @app.get("/api/insights")
    async def insights_endpoint():
        summary = read_summary()
        return summary

    @app.get("/")
    async def serve_index():
        index_path = UI_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="index.html not found")

    @app.get("/{asset_path:path}")
    async def serve_static(asset_path: str):
        """Serve static files from UI directory."""
        file_path = UI_DIR / asset_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Fallback to index.html for SPA routing
        index_path = UI_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="File not found")

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    auto_open: bool = False,
    use_reloader: bool = True,
) -> None:
    """
    Start the FastAPI server.
    
    Args:
        host: Host interface for the FastAPI server
        port: Port for the FastAPI server
        debug: Enable debug mode (better error messages)
        auto_open: Automatically open browser when server starts (disabled by default)
        use_reloader: Enable auto-reload on file changes
    """
    import uvicorn
    
    app = create_app()
    
    # Print the URL so user can open it manually
    visible_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{visible_host}:{port}/"
    print(f"\n🚀 Server starting at {url}")
    print(f"📚 API docs available at {url}docs\n")
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=use_reloader and debug,
        log_level="debug" if debug else "info",
    )


def _resolve_db_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    return Path(raw_path)


if __name__ == "__main__":
    run_server()
