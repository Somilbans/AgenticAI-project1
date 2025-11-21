"""
Simple helpers for saving DataFrame rows into a ChromaDB collection.
"""

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import chromadb
import pandas as pd
from chromadb.api import Collection
from chromadb.utils import embedding_functions

from . import config

def _resolve_db_path(db_path: Optional[Path]) -> Path:
    """
    Determine which directory should hold the Chroma DB.
    """
    if db_path is not None:
        return db_path
    return config.VECTOR_DB_DIR


def get_client(db_path: Optional[Path] = None) -> chromadb.PersistentClient:
    """
    Return a persistent Chroma client rooted at the configured directory.
    """
    resolved_path = _resolve_db_path(db_path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(resolved_path))


def _get_collection(name: str, db_path: Optional[Path] = None) -> Collection:
    """
    Return a Chroma collection wired with a default embedding function so we
    can query without supplying embeddings manually.
    """
    client = get_client(db_path=db_path)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(name=name, embedding_function=embed_fn)


def upsert_dataframe(df: pd.DataFrame, collection_name: str) -> None:
    """
    Store each row of the DataFrame as a document in the named collection.

    We keep it simple:
    - Document text is a pipe-delimited string of "column=value".
    - Metadata holds the original row as a dict.
    """
    collection = _get_collection(collection_name)

    normalized_df = _sanitize_dataframe(df)

    documents: List[str] = []
    metadatas: List[Mapping[str, Any]] = []
    ids: List[str] = []

    for idx, row in normalized_df.iterrows():
        documents.append(_row_to_document(row))
        metadatas.append(row.to_dict())
        ids.append(f"{collection_name}-{idx}")

    if not ids:
        return

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def _row_to_document(row: Mapping[str, Any]) -> str:
    """
    Convert a row into a compact string representation.
    """
    return " | ".join(f"{key}={value}" for key, value in row.items())


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up data before inserting into Chroma:
    - Convert datetime-like values to ISO strings.
    - Fill NaNs with empty strings to avoid serialization surprises.
    """
    sanitized = df.copy()

    for column in sanitized.columns:
        if pd.api.types.is_datetime64_any_dtype(sanitized[column]):
            sanitized[column] = sanitized[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            sanitized[column] = sanitized[column].fillna("")
    return sanitized


def query_collection(
    question: str,
    collection_name: str,
    top_k: int = 3,
    db_path: Optional[Path] = None,
) -> List[str]:
    """
    Fetch the top-k documents relevant to the question from the collection.
    """
    collection = _get_collection(collection_name, db_path=db_path)
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[question], n_results=top_k)
    return results.get("documents", [[]])[0]


def list_collections(db_path: Optional[Path] = None) -> Sequence[str]:
    """
    Return the names of collections stored in the specified database path.
    """
    client = get_client(db_path=db_path)
    return [collection.name for collection in client.list_collections()]

