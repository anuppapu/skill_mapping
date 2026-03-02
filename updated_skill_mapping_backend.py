"""
skill_mapping_backend.py
========================
Reusable backend module for the Skill Mapping pipeline.

This module is framework-agnostic — it contains NO Streamlit or FastAPI
imports. Any frontend (React, Streamlit, CLI) can call these functions.

Dependent Modules (already in workspace):
    llm_client.py      — LLM metadata extraction (uses prompt.py + schema.py)
    prompt.py          — Extraction prompt template
    schema.py          — Pydantic validation models for extracted metadata

Pipeline Steps:
    1. Load employee data (Excel / CSV / pickle)
    2. Flatten employee profiles for embedding
    3. Generate embeddings via API
    4. Build & manage FAISS indexes
    5. Extract requirement metadata via LLM  (delegates to llm_client.py)
    6. Flatten requirement for embedding (aligned with employee format)
    7. FAISS semantic search
    8. Rule-based hybrid scoring
    9. Final ranking (embedding similarity + rule score)
   10. LLM-powered explanation for top matches
"""

import os
import json
import asyncio
import logging
import sqlite3
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import faiss
import requests
from dotenv import load_dotenv

# ── Existing workspace modules ──────────────────────────────────────────────
from llm_client import extract_project_metadata    # LLM metadata extraction (prompt.py + schema.py)

# ── Load environment variables (.env) ───────────────────────────────────────
load_dotenv()

# Embedding API config (from .env)
EMBED_API_URL = os.getenv("EMBED_API_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))
EMBED_APPLICATION_TYPE = os.getenv("EMBED_APPLICATION_TYPE", "Developer")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")

# Local embedding model config (from .env)
LOCAL_EMBED_MODEL_PATH = os.getenv("LOCAL_EMBED_MODEL_PATH", "all-MiniLM-L6-v2")
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "384"))

# Lazy-loaded local model instance (loaded once on first use)
_local_model = None
_local_model_lock = threading.Lock()

# LLM model params (from .env) — endpoint
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-5.2")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "20000"))


# Data paths (from .env)
DB_PATH = os.getenv("DB_PATH", "enterprise_employee.db")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_indexes")
EMPLOYEE_DATA_PATH = os.getenv("EMPLOYEE_DATA_PATH", "employee_data.pkl")

os.makedirs(INDEX_DIR, exist_ok=True)


def get_embed_dim(embedding_model: str = "text-embedding-3-large") -> int:
    """Return the embedding dimension for the given model."""
    if embedding_model == "all-MiniLM-L6-v2":
        return LOCAL_EMBED_DIM
    return EMBED_DIM


def _load_local_model():
    """
    Lazy-load the local sentence-transformers model (all-MiniLM-L6-v2).
    Thread-safe — the model is loaded exactly once.
    """
    global _local_model
    with _local_model_lock:
        if _local_model is None:
            from sentence_transformers import SentenceTransformer
            logging.info(f"[EMBED-LOCAL] Loading local model from '{LOCAL_EMBED_MODEL_PATH}'...")
            _local_model = SentenceTransformer(LOCAL_EMBED_MODEL_PATH)
            logging.info(f"[EMBED-LOCAL] Model loaded. Dimension: {_local_model.get_sentence_embedding_dimension()}")
        return _local_model

logging.basicConfig(
    filename="enterprise_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── Columns from the uploaded Excel / CSV used to build employee embedding text ──
# These are the exact columns read from the employee data file and consumed by
# flatten_employee() to produce the text string sent to the SAIS Embedding API.
#
#   Column Name                        Used For
#   ──────────────────────────────────  ────────────────────────────────────────
#   domain                              Primary domain of the employee
#   domain_experience_years              Years of experience in that domain
#   total_experience_years               Total career experience in years
#   past_project_execution_summary       Free-text summary of past project work
#   skill                                Individual skill name (one per row)
#   skill_category                       Category / grouping of the skill
#   skill_experience_years               Years of experience with that skill
#
# NOTE: Each employee may span multiple rows (one row per skill).  The rows are
#       grouped by employee_id, then flatten_employee() merges them into a single
#       text block that is embedded as one vector.
EMBED_COLUMNS = [
    "domain",
    "domain_experience_years",
    "skill",
    "skill_category",
    "skill_experience_years",
    "total_experience_years",
    "past_project_execution_summary",
]

# ── Global embedding progress tracker ──────────────────────────────────────
# Polled by GET /api/embeddings/progress from the frontend.
_embedding_progress_lock = threading.Lock()
embedding_progress: dict = {
    "running": False,
    "phase": "idle",           # idle | flattening | embedding | normalizing | saving | done | error
    "current": 0,              # employees embedded so far
    "total": 0,                # total employees to embed
    "current_employee_id": "", # ID of the employee currently being embedded
    "message": "",             # human-readable status line
}

def _update_progress(**kwargs):
    """Thread-safe update of the global embedding_progress dict."""
    with _embedding_progress_lock:
        embedding_progress.update(kwargs)

def get_embedding_progress() -> dict:
    """Return a snapshot of the current embedding progress."""
    with _embedding_progress_lock:
        return dict(embedding_progress)


# ============================================================================
# DATABASE INITIALISATION
# ============================================================================

def init_db() -> None:
    """
    Create the SQLite tables if they don't exist.

    Tables:
        index_registry       — FAISS index metadata (type, params, vector count)
        audit_logs           — Operational audit trail
        embedding_registry   — **NEW** Maps employee_id → faiss_id (int64).
                               Replaces the old employee_ids_order.json file.
                               Supports delete-detection via the `is_active` flag.

    The embedding_registry table stores:
        employee_id  TEXT PK  — Business employee identifier (e.g. "EMP001")
        faiss_id     INTEGER  — Unique int64 ID used inside FAISS IndexIDMap
        embedded_at  TEXT     — ISO timestamp when embedding was generated
        is_active    INTEGER  — 1 = present in latest employee file,
                                0 = was deleted (stale embedding)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS index_registry (
            index_name        TEXT PRIMARY KEY,
            index_type        TEXT,
            nlist             INTEGER,
            nprobe            INTEGER,
            m                 INTEGER,
            ef_search         INTEGER,
            vector_count      INTEGER,
            created_at        TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id    TEXT,
            action    TEXT,
            details   TEXT,
            timestamp TEXT
        )
    """)
    # ── NEW: Embedding registry (replaces employee_ids_order.json) ──────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS embedding_registry (
            employee_id   TEXT    PRIMARY KEY,
            faiss_id      INTEGER UNIQUE NOT NULL,
            embedded_at   TEXT    NOT NULL,
            is_active     INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ============================================================================
# EMBEDDING REGISTRY HELPERS  (SQLite-backed, replaces JSON mapping)
# ============================================================================

def _emp_id_to_faiss_id(employee_id: str) -> int:
    """
    Convert a string employee_id to a deterministic int64 for FAISS IndexIDMap.

    Uses a CRC-based hash trimmed to 63 bits (positive int64) so the same
    employee_id always maps to the same faiss_id across runs.
    """
    import hashlib
    h = hashlib.sha256(str(employee_id).encode()).hexdigest()
    return int(h[:15], 16)  # 60-bit positive integer — safe for int64


def _get_embedded_ids() -> dict:
    """
    Return {employee_id: faiss_id} for all ACTIVE embeddings in the registry.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT employee_id, faiss_id FROM embedding_registry WHERE is_active = 1"
    ).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def _get_all_embedded_ids() -> dict:
    """
    Return {employee_id: faiss_id} for ALL embeddings (active + inactive).
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT employee_id, faiss_id FROM embedding_registry"
    ).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def _faiss_id_to_emp_id(faiss_id: int) -> Optional[str]:
    """
    Reverse-lookup: given a FAISS int64 ID, return the employee_id string.
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT employee_id FROM embedding_registry WHERE faiss_id = ? AND is_active = 1",
        (int(faiss_id),)
    ).fetchone()
    conn.close()
    return row[0] if row else None


# ============================================================================
# EMPLOYEE DATA LOADING
# ============================================================================

def load_employee_data(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load employee data from an Excel/CSV file (path or raw bytes).

    Returns a cleaned DataFrame and persists it to EMPLOYEE_DATA_PATH.
    """
    if file_bytes is not None:
        import io
        if file_name and file_name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = sheet_name or xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
    elif file_path is not None:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            xls = pd.ExcelFile(file_path)
            sheet = sheet_name or xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
    elif os.path.exists(EMPLOYEE_DATA_PATH):
        return pd.read_pickle(EMPLOYEE_DATA_PATH)
    else:
        raise FileNotFoundError("No employee data source provided and no persisted data found.")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    df = df.fillna("")
    df = df.reset_index(drop=True)
    df.to_pickle(EMPLOYEE_DATA_PATH)
    return df


def get_employee_dataframe() -> pd.DataFrame:
    """Return persisted employee DataFrame (raises if not available)."""
    if os.path.exists(EMPLOYEE_DATA_PATH):
        return pd.read_pickle(EMPLOYEE_DATA_PATH)
    raise FileNotFoundError("Employee data not loaded yet. Upload data first.")


# ============================================================================
# FLATTEN HELPERS
# ============================================================================

def flatten_employee(emp_group: pd.DataFrame) -> str:
    """
    Flatten all skill rows for one employee into a single text block for embedding.

    Columns consumed from the uploaded employee data (Excel / CSV):
        • domain                          → "Primary Domain" line
        • domain_experience_years          → "Domain Experience" line
        • total_experience_years           → "Total Experience" line
        • past_project_execution_summary   → "Past Project Summary" line
        • skill                            → each unique skill bullet
        • skill_category                   → category label per skill bullet
        • skill_experience_years           → experience years per skill bullet

    The first four columns are taken from the first row of the group (they are
    identical across all rows for the same employee).  The last three columns
    are iterated over all rows to collect every distinct skill entry.

    Output format example:
        Employee Profile
        Primary Domain: Capital Markets
        Domain Experience: 8 years
        Total Experience: 12 years
        Past Project Summary: Led migration of …
        Skills:
        - Skill: Python | Category: Programming | Experience: 6 years
        - Skill: SQL    | Category: Database    | Experience: 5 years
    """
    first = emp_group.iloc[0]

    seen_skills: set = set()
    skill_lines: list = []
    for _, row in emp_group.iterrows():
        skill_name = str(row.get("skill", "")).strip()
        if skill_name and skill_name.lower() not in seen_skills:
            seen_skills.add(skill_name.lower())
            skill_lines.append(
                f"- Skill: {skill_name} | Category: {row.get('skill_category', '')} "
                f"| Experience: {row['skill_experience_years']} years"
            )
    skill_text = "\n".join(skill_lines)

    return (
        f"Employee Profile\n"
        f"Primary Domain: {first['domain']}\n"
        f"Domain Experience: {first['domain_experience_years']} years\n"
        f"Total Experience: {first['total_experience_years']} years\n"
        f"Past Project Summary: {first.get('past_project_execution_summary', 'N/A')}\n"
        f"Skills:\n{skill_text}"
    )


def flatten_requirement_for_embedding(extracted: dict) -> str:
    """
    Flatten LLM-extracted requirement JSON into employee-aligned text
    so both vectors live in the same semantic space.

    ── FORMAT ALIGNMENT (CRITICAL) ──────────────────────────────────────
    Skill lines mirror flatten_employee() but adapt to what the user
    actually specified:

      • Mandatory skill WITH per-skill experience stated:
            - Skill: Python | Category: Programming | Experience: 5 years

      • Mandatory skill WITHOUT per-skill experience:
            - Skill: Python | Category: Programming

      • Optional skill:
            - Skill: E-commerce | Category: DOMAIN | Optional

    The overall minimum_overall_experience_years is used ONLY in the
    "Domain Experience" and "Total Experience" header lines — it is
    NOT injected into individual skill lines when the user did not
    specify per-skill experience.
    ─────────────────────────────────────────────────────────────────────
    """
    domain = ", ".join(extracted.get("domain", [])) or "N/A"
    min_exp = extracted.get("minimum_overall_experience_years") or 0
    description = extracted.get("project_description", "N/A")

    # ── Build skill lines ──────────────────────────────────────────────────
    skill_lines: list = []

    for s in extracted.get("mandatory_skills", []):
        skill_name = s.get("skill", "").strip()
        category = s.get("category", "").strip() or "General"
        per_skill_exp = s.get("min_experience_years")
        # Only include "| Experience: X years" when the user explicitly
        # mentioned per-skill experience.  If null / 0, omit it entirely.
        if per_skill_exp:
            skill_lines.append(
                f"- Skill: {skill_name} | Category: {category} "
                f"| Experience: {per_skill_exp} years"
            )
        else:
            skill_lines.append(
                f"- Skill: {skill_name} | Category: {category}"
            )

    for s in extracted.get("optional_skills", []):
        skill_name = s.get("skill", "").strip()
        category = s.get("category", "").strip() or "General"
        # Optional skills are tagged with "| Optional" so the embedding
        # model understands these are nice-to-have, not hard requirements.
        skill_lines.append(
            f"- Skill: {skill_name} | Category: {category} | Optional"
        )

    skill_text = "\n".join(skill_lines) if skill_lines else "None"

    # ── Final text block — mirrors flatten_employee() structure ─────────────
    return (
        f"Employee Profile\n"
        f"Primary Domain: {domain}\n"
        f"Domain Experience: {min_exp} years\n"
        f"Total Experience: {min_exp} years\n"
        f"Past Project Summary: {description}\n"
        f"Skills:\n{skill_text}"
    )


def flatten_requirement_display(extracted: dict) -> str:
    """Human-readable display text for a requirement (not for embedding)."""

    def format_skills(skill_list):
        if not skill_list:
            return "None"
        parts = []
        for s in skill_list:
            txt = s.get("skill", "")
            exp = s.get("min_experience_years", "")
            cat = s.get("category", "")
            if exp:
                txt += f" ({exp} years"
                if cat:
                    txt += f", {cat}"
                txt += ")"
            elif cat:
                txt += f" ({cat})"
            parts.append(txt)
        return ", ".join(parts)

    return (
        f"Project Description:\n{extracted.get('project_description', '')}\n\n"
        f"Domain: {', '.join(extracted.get('domain', [])) or 'N/A'}\n"
        f"Criticality: {extracted.get('project_criticality', 'N/A')}\n"
        f"Duration: {extracted.get('project_duration_months', 'N/A')} months\n"
        f"Required Bandwidth: {extracted.get('required_bandwidth_percentage', 'N/A')}%\n"
        f"Minimum Overall Experience: {extracted.get('minimum_overall_experience_years', 'N/A')} years\n\n"
        f"Mandatory Skills:\n{format_skills(extracted.get('mandatory_skills', []))}\n\n"
        f"Optional Skills:\n{format_skills(extracted.get('optional_skills', []))}"
    ).strip()


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

# Last embedding API error — surfaced in progress / result messages
_last_embed_api_error: str = ""


def get_embedding(text: str, max_retries: int = 3) -> Optional[List[float]]:
    """
    Generate an embedding vector for a single text string via SAIS API.

    Includes retry logic with exponential back-off:
        Attempt 1 → immediate
        Attempt 2 → wait 2 s
        Attempt 3 → wait 4 s

    On final failure, stores the error detail in _last_embed_api_error
    so that upstream callers can surface it to the user.
    """
    global _last_embed_api_error
    payload = json.dumps({"model": EMBED_MODEL_NAME, "input": text})
    headers = {
        "applicationType": EMBED_APPLICATION_TYPE,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBED_API_KEY}",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(EMBED_API_URL, headers=headers, data=payload, timeout=300)

            if response.status_code == 200:
                result = response.json().get("result")
                if not result:
                    _last_embed_api_error = "API returned 200 but 'result' key is missing from response."
                    logging.error(f"[EMBED-API] {_last_embed_api_error} | Body: {response.text[:300]}")
                    return None
                data = result.get("data")
                if not data:
                    _last_embed_api_error = "API returned 200 but 'result.data' is empty."
                    logging.error(f"[EMBED-API] {_last_embed_api_error} | Body: {response.text[:300]}")
                    return None
                return data[0].get("embedding")

            # ── Non-200 status code ──────────────────────────────────────
            err_body = response.text[:500]
            _last_embed_api_error = f"HTTP {response.status_code}: {err_body}"
            logging.error(
                f"[EMBED-API] Attempt {attempt}/{max_retries} failed — "
                f"HTTP {response.status_code}: {err_body}"
            )

            # 401/403 = auth error — token may be expired or malformed
            if response.status_code in (401, 403):
                logging.error(
                    f"[EMBED-API] Auth error (HTTP {response.status_code}). "
                    f"Check EMBED_API_KEY in .env — it may be expired or malformed."
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None

            # 429 = rate-limited — wait longer
            if response.status_code == 429:
                wait = 2 ** (attempt + 1)
                logging.warning(f"[EMBED-API] Rate-limited. Waiting {wait}s before retry…")
                time.sleep(wait)
                continue

            # 5xx = server error — retry with backoff
            if response.status_code >= 500:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None

            # Other 4xx — don't retry
            return None

        except requests.exceptions.Timeout:
            _last_embed_api_error = f"Request timed out (attempt {attempt}/{max_retries})"
            logging.error(f"[EMBED-API] {_last_embed_api_error}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return None

        except requests.exceptions.ConnectionError as e:
            _last_embed_api_error = f"Connection error: {e}"
            logging.error(f"[EMBED-API] Attempt {attempt}/{max_retries} — {_last_embed_api_error}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return None

        except Exception as e:
            _last_embed_api_error = f"Unexpected error: {e}"
            logging.error(f"[EMBED-API] {_last_embed_api_error}")
            return None

    return None


def get_embeddings_batch(
    texts: List[str],
    batch_size: int = 16,
    progress_callback=None,
    employee_ids: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Generate embeddings for a list of texts. Returns (N, EMBED_DIM) array.

    Args:
        texts:             List of flattened text strings.
        batch_size:        (reserved for future batch API support)
        progress_callback: Optional callable(current_index, total, employee_id)
                           called after each successful embedding.
        employee_ids:      Optional parallel list of employee IDs (for logging).

    On failure, sets _last_embed_api_error with the root-cause detail
    so the caller / UI can display the actual problem.
    """
    global _last_embed_api_error
    all_embeddings = []
    total = len(texts)
    for i, text in enumerate(texts):
        eid = employee_ids[i] if employee_ids else f"record_{i}"
        emb = get_embedding(text)
        if emb is None:
            err = _last_embed_api_error or "Unknown error"
            logging.error(
                f"[EMBED-BATCH] Embedding failed for {eid} ({i+1}/{total}) "
                f"after retries. Last API error: {err}"
            )
            _last_embed_api_error = (
                f"Failed on employee {eid} ({i+1}/{total}). API error: {err}"
            )
            return None
        all_embeddings.append(emb)
        if progress_callback:
            progress_callback(i + 1, total, str(eid))
    return np.array(all_embeddings, dtype="float32")


def get_embedding_local(text: str) -> Optional[List[float]]:
    """
    Generate an embedding vector for a single text string using the local
    all-MiniLM-L6-v2 model (sentence-transformers).
    """
    try:
        model = _load_local_model()
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    except Exception as e:
        logging.error(f"[EMBED-LOCAL] Error encoding text: {e}")
        return None


def get_embeddings_batch_local(
    texts: List[str],
    batch_size: int = 32,
    progress_callback=None,
    employee_ids: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Generate embeddings for a list of texts using the local all-MiniLM-L6-v2 model.
    Returns (N, LOCAL_EMBED_DIM) array.

    Uses sentence-transformers batch encoding for efficiency.
    Progress callback is called per-batch.
    """
    try:
        model = _load_local_model()
        total = len(texts)
        all_embeddings = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            batch_embs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(batch_embs)

            if progress_callback:
                eid = employee_ids[end - 1] if employee_ids else f"record_{end - 1}"
                progress_callback(end, total, str(eid))

        result = np.vstack(all_embeddings).astype("float32")
        logging.info(f"[EMBED-LOCAL] Encoded {total} texts → shape {result.shape}")
        return result
    except Exception as e:
        logging.error(f"[EMBED-LOCAL] Batch encoding failed: {e}")
        return None


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise in-place (required for cosine similarity via inner product)."""
    faiss.normalize_L2(vectors)
    return vectors


def generate_embeddings_incremental(
    df: pd.DataFrame,
    embedding_model: str = "text-embedding-3-large",
    embeddings_npy_filename: str = "employee_embeddings.npy",
) -> dict:
    """
    Generate / synchronise employee embeddings with the current employee DataFrame.

    This function handles THREE scenarios:
    ───────────────────────────────────────────────────────────────────────────
    SCENARIO 1 — FRESH RUN (no embeddings exist yet)
        • Generates embeddings for ALL employees in the DataFrame.
        • Saves to the specified .npy file.
        • Registers every employee_id → faiss_id mapping in SQLite
          (embedding_registry table).

    SCENARIO 2 — INCREMENTAL APPEND (new employees added to the file)
        • Compares employee IDs in the DataFrame against the embedding_registry.
        • Generates embeddings ONLY for employees NOT yet in the registry.
        • Appends new vectors to the existing .npy file.
        • Inserts new rows into embedding_registry.

    SCENARIO 3 — DELETE DETECTION (employees removed from the file)
        • Detects employee IDs that exist in embedding_registry but are
          MISSING from the current DataFrame.
        • Marks those rows as is_active = 0 in embedding_registry.
        • Rebuilds the .npy file containing only ACTIVE embeddings.
        • ⚠️  After deletions, ALL FAISS indexes must be recreated because
          the .npy file positions have changed. The function returns
          `rebuild_indexes_required = True` so the caller can handle this.
    ───────────────────────────────────────────────────────────────────────────

    Args:
        df:                     Employee DataFrame (multi-row per employee, grouped by employee_id)
        embedding_model:        "text-embedding-3-large" (API) or "all-MiniLM-L6-v2" (local)
        embeddings_npy_filename: Name of the .npy file to save embeddings to

    Returns:
        dict with keys:
            status               — "ok" | "error"
            new_count            — Number of newly embedded employees
            deleted_count        — Number of employees marked as deleted
            total_active         — Total active embeddings after this run
            rebuild_indexes_required — True if indexes must be recreated
            vectors              — numpy array of all active embeddings (or None)
    """
    # Ensure .npy extension
    if not embeddings_npy_filename.endswith(".npy"):
        embeddings_npy_filename += ".npy"
    embeddings_path = embeddings_npy_filename
    use_local = (embedding_model == "all-MiniLM-L6-v2")
    logging.info(f"[EMBED] Using model: {embedding_model}, output: {embeddings_path}")

    # ── Step 1: Identify all unique employee IDs in the current DataFrame ────
    #    The DataFrame has multiple rows per employee (one per skill).
    #    We group by employee_id to get one profile per employee.
    #
    #    IMPORTANT: Use a SORTED list for iteration order — NOT a set.
    #    pandas groupby().groups.keys() returns sorted keys.  skill_map.py
    #    iterates them directly (sorted), so the .npy row order is sorted.
    #    We must do the same here so that employee_ids_order.json matches
    #    the .npy row order.  Using set() would randomise the iteration
    #    order and cause create_index() to assign wrong faiss_ids to vectors.
    grouped = df.groupby("employee_id")
    current_emp_ids_ordered = sorted(str(eid) for eid in grouped.groups.keys())
    current_emp_ids = set(current_emp_ids_ordered)   # O(1) membership checks
    logging.info(f"[EMBED] Current DataFrame has {len(current_emp_ids)} unique employees.")

    # ── Step 2: Load the existing embedding registry from SQLite ─────────────
    #    Returns {employee_id_str: faiss_id_int} for ALL rows (active + inactive).
    #    Previously this was stored in employee_ids_order.json — now in SQLite.
    all_registered = _get_all_embedded_ids()       # all rows (active & inactive)
    active_registered = _get_embedded_ids()         # only is_active = 1
    registered_emp_ids = set(all_registered.keys())
    active_emp_ids = set(active_registered.keys())
    logging.info(f"[EMBED] Registry has {len(registered_emp_ids)} total, {len(active_emp_ids)} active embeddings.")

    # ── Step 3: Detect DELETED employees ─────────────────────────────────────
    #    These are employee IDs that were previously embedded (is_active=1) but
    #    are NO LONGER present in the current DataFrame.
    #
    #    Example:
    #      Registry (active): {EMP001, EMP002, EMP003, EMP004}
    #      Current DataFrame: {EMP001, EMP002, EMP004, EMP005}
    #      → Deleted: {EMP003}   (was active, now missing from file)
    #      → New:     {EMP005}   (in file, not in registry at all)
    deleted_ids = active_emp_ids - current_emp_ids
    logging.info(f"[EMBED] Deleted employees detected: {len(deleted_ids)} -> {deleted_ids if len(deleted_ids) <= 10 else '(too many to list)'}")
    # ── Step 4: Detect NEW employees ─────────────────────────────────────────
    #    These are employee IDs in the DataFrame that have NEVER been embedded.
    #    Note: we check against ALL registered (not just active) to avoid
    #    re-embedding a previously-deleted-then-re-added employee.
    #    CRITICAL: iterate the SORTED list, NOT the set, so .npy row order
    #    is deterministic and matches employee_ids_order.json.
    new_ids = [eid for eid in current_emp_ids_ordered if eid not in registered_emp_ids]
    logging.info(f"[EMBED] New employees to embed: {len(new_ids)}")

    # ── Step 5: Also detect RE-ACTIVATED employees ───────────────────────────
    #    These were previously marked is_active=0 (deleted) but have now
    #    reappeared in the file. We re-activate them without re-embedding.
    reactivated_ids = set()
    for eid in current_emp_ids:
        if eid in registered_emp_ids and eid not in active_emp_ids:
            reactivated_ids.add(eid)
    if reactivated_ids:
        logging.info(f"[EMBED] Re-activated employees (already embedded): {len(reactivated_ids)}")

    # ── Step 6: Check if there's anything to do ──────────────────────────────
    #    IMPORTANT: If the target .npy file does NOT exist, this is a fresh run
    #    for this embedding model — we must embed ALL employees even if the
    #    registry already knows them (from a different model's run).
    npy_file_missing = not os.path.exists(embeddings_path)
    if npy_file_missing and not new_ids:
        # The .npy doesn't exist but registry says everyone is embedded
        # → this is a different model run. Force full re-generation.
        logging.info(
            f"[EMBED] Target file '{embeddings_path}' does not exist but registry "
            f"has {len(active_emp_ids)} entries. Forcing full generation for this model."
        )
        new_ids = list(current_emp_ids_ordered)  # embed ALL employees
        deleted_ids = set()
        reactivated_ids = set()

    if not new_ids and not deleted_ids and not reactivated_ids:
        logging.info("[EMBED] All employee embeddings are up to date. No changes needed.")
        _update_progress(
            running=False, phase="done", current=0, total=0,
            message=f"All {len(active_emp_ids)} embeddings already up to date. No changes needed.",
        )
        vectors = np.load(embeddings_path) if os.path.exists(embeddings_path) else None
        return {
            "status": "ok",
            "new_count": 0,
            "deleted_count": 0,
            "reactivated_count": 0,
            "total_active": len(active_emp_ids),
            "rebuild_indexes_required": False,
            "vectors": vectors,
        }

    # ── Step 7: Generate embeddings for NEW employees ────────────────────────
    #    For each new employee, flatten their multi-row profile into a single
    #    text block, then call the SAIS embedding API.
    new_vectors = None
    new_faiss_ids = []
    if new_ids:
        _update_progress(phase="flattening", current=0, total=len(new_ids),
                         message=f"Flattening {len(new_ids)} employee profiles…")
        logging.info(f"[EMBED] Flattening {len(new_ids)} employee profiles for embedding...")
        texts = []
        for eid in new_ids:
            emp_group = grouped.get_group(eid)
            flattened = flatten_employee(emp_group)
            texts.append(flattened)
            logging.debug(f"[EMBED]   Employee {eid}: flattened {len(flattened)} chars")

        model_label = "local model" if use_local else "API"
        logging.info(f"[EMBED] Calling {model_label} for {len(texts)} texts...")
        _update_progress(phase="embedding", current=0, total=len(texts),
                         message=f"Embedding 0 / {len(texts)} employees via {model_label}…")

        def _on_embed_progress(current, total, emp_id):
            _update_progress(
                current=current, total=total,
                current_employee_id=emp_id,
                message=f"Embedding {current} / {total} employees… (current: {emp_id})",
            )

        if use_local:
            new_vectors = get_embeddings_batch_local(
                texts,
                progress_callback=_on_embed_progress,
                employee_ids=new_ids,
            )
        else:
            new_vectors = get_embeddings_batch(
                texts,
                progress_callback=_on_embed_progress,
                employee_ids=new_ids,
            )

        if new_vectors is None:
            err_detail = _last_embed_api_error or "Unknown embedding error"
            _update_progress(running=False, phase="error",
                             message=f"Embedding failed: {err_detail}")
            logging.error(f"[EMBED] Embedding call failed. Detail: {err_detail}")
            return {"status": "error", "message": f"Embedding generation failed: {err_detail}"}

        # L2-normalise so FAISS inner-product = cosine similarity
        # (local model already normalises, but faiss.normalize_L2 is idempotent)
        logging.info(f"[EMBED] Normalising {new_vectors.shape[0]} new vectors (L2 → unit length)...")
        new_vectors = normalize_embeddings(new_vectors)

        # Compute deterministic faiss_id for each new employee
        new_faiss_ids = [_emp_id_to_faiss_id(eid) for eid in new_ids]
        logging.info(f"[EMBED] Generated {len(new_faiss_ids)} faiss_ids for new employees.")

    # ── Step 8: Update SQLite embedding_registry ─────────────────────────────
    _update_progress(phase="saving", message="Updating registry & saving embeddings…")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()

    #  8a. Mark DELETED employees as inactive
    if deleted_ids:
        logging.info(f"[EMBED] Marking {len(deleted_ids)} employees as is_active=0 in registry...")
        for eid in deleted_ids:
            cursor.execute(
                "UPDATE embedding_registry SET is_active = 0 WHERE employee_id = ?",
                (eid,)
            )

    #  8b. Re-activate previously-deleted employees that reappeared
    if reactivated_ids:
        logging.info(f"[EMBED] Re-activating {len(reactivated_ids)} employees in registry...")
        for eid in reactivated_ids:
            cursor.execute(
                "UPDATE embedding_registry SET is_active = 1 WHERE employee_id = ?",
                (eid,)
            )

    #  8c. Insert NEW employee → faiss_id mappings
    if new_ids:
        logging.info(f"[EMBED] Inserting {len(new_ids)} new rows into embedding_registry...")
        for eid, fid in zip(new_ids, new_faiss_ids):
            cursor.execute(
                "INSERT OR REPLACE INTO embedding_registry (employee_id, faiss_id, embedded_at, is_active) "
                "VALUES (?, ?, ?, 1)",
                (eid, fid, now_iso),
            )

    conn.commit()
    conn.close()
    logging.info("[EMBED] SQLite embedding_registry updated.")

    # ── Step 9: Rebuild / Append the .npy embeddings file ────────────────────
    #
    #    CASE A — Deletions occurred:
    #       We must REBUILD the .npy file from scratch using only ACTIVE IDs
    #       because positional alignment with FAISS indexes is broken.
    #       → All FAISS indexes must be recreated afterwards.
    #
    #    CASE B — Only new employees (no deletions):
    #       We simply APPEND the new vectors to the existing .npy file.
    #       → Existing FAISS indexes can use append_to_index().
    #
    rebuild_required = bool(deleted_ids) or bool(reactivated_ids)

    # npy_id_order tracks the ACTUAL employee_id order in the .npy file.
    # This MUST be saved to employee_ids_order.json so that create_index()
    # can correctly assign faiss_ids to .npy rows.
    npy_id_order: list = []

    if rebuild_required:
        # ── REBUILD: Reload all active embeddings ────────────────────────────
        logging.info("[EMBED] Deletions/reactivations detected → rebuilding .npy from active set...")
        active_registry = _get_embedded_ids()  # fresh read after updates

        # Collect vectors: existing active + new
        all_active_vectors = []
        all_active_faiss_ids = []

        # Load existing .npy if it exists
        if os.path.exists(embeddings_path):
            old_vectors = np.load(embeddings_path)
            # We need to know which positions in old_vectors map to which employee.
            # Use the old JSON to get position → employee_id mapping.
            old_ids_path = "employee_ids_order.json"
            if os.path.exists(old_ids_path):
                old_id_list = json.load(open(old_ids_path))
                for pos, eid in enumerate(old_id_list):
                    if eid in active_registry and pos < old_vectors.shape[0]:
                        all_active_vectors.append(old_vectors[pos])
                        all_active_faiss_ids.append(active_registry[eid])
                        npy_id_order.append(eid)
            else:
                # No old JSON — this shouldn't happen in normal flow but handle gracefully
                logging.warning("[EMBED] No old employee_ids_order.json found during rebuild.")

        # Append new vectors
        if new_vectors is not None:
            for i, eid in enumerate(new_ids):
                all_active_vectors.append(new_vectors[i])
                all_active_faiss_ids.append(_emp_id_to_faiss_id(eid))
                npy_id_order.append(eid)

        if all_active_vectors:
            all_vectors = np.array(all_active_vectors, dtype="float32")
            np.save(embeddings_path, all_vectors)
            logging.info(f"[EMBED] Rebuilt .npy with {all_vectors.shape[0]} active vectors.")
        else:
            all_vectors = None
            logging.warning("[EMBED] No active vectors after rebuild.")

    else:
        # ── APPEND ONLY: No deletions, just add new vectors ─────────────────
        # Load existing .npy row order from JSON, then append new_ids.
        if os.path.exists("employee_ids_order.json"):
            with open("employee_ids_order.json", "r") as f:
                npy_id_order = json.load(f)
        else:
            npy_id_order = []

        if os.path.exists(embeddings_path) and new_vectors is not None:
            logging.info(f"[EMBED] Appending {new_vectors.shape[0]} new vectors to existing .npy...")
            existing_vectors = np.load(embeddings_path)
            all_vectors = np.vstack([existing_vectors, new_vectors])
            np.save(embeddings_path, all_vectors)
            npy_id_order.extend(new_ids)  # new rows appended at the end
            logging.info(f"[EMBED] .npy now has {all_vectors.shape[0]} vectors.")
        elif new_vectors is not None:
            logging.info(f"[EMBED] Creating new .npy with {new_vectors.shape[0]} vectors...")
            all_vectors = new_vectors
            np.save(embeddings_path, all_vectors)
            npy_id_order = list(new_ids)  # fresh .npy, order = new_ids order
        else:
            all_vectors = np.load(embeddings_path) if os.path.exists(embeddings_path) else None

    # ── Step 10: Save employee_ids_order.json in ACTUAL .npy row order ───────
    #    CRITICAL: This JSON records which employee_id is at each .npy row
    #    position. create_index() relies on this to assign the correct
    #    faiss_id to each vector.  Previously this was saved as sorted()
    #    which did NOT match the .npy row order — that bug caused wrong
    #    employee IDs to be returned from FAISS searches.
    with open("employee_ids_order.json", "w") as f:
        json.dump(npy_id_order, f)
    logging.info(f"[EMBED] employee_ids_order.json updated ({len(npy_id_order)} IDs, in .npy row order).")

    # ── Step 11: Summary ─────────────────────────────────────────────────────
    active_registry = _get_embedded_ids()  # fresh count after all updates
    total_active = len(active_registry)
    logging.info(
        f"[EMBED] DONE — new: {len(new_ids)}, deleted: {len(deleted_ids)}, "
        f"reactivated: {len(reactivated_ids)}, total_active: {total_active}, "
        f"rebuild_indexes_required: {rebuild_required}"
    )

    _update_progress(
        running=False, phase="done", current=len(new_ids), total=len(new_ids),
        message=f"Done — {len(new_ids)} new, {len(deleted_ids)} deleted, {total_active} total active.",
    )

    return {
        "status": "ok",
        "new_count": len(new_ids),
        "deleted_count": len(deleted_ids),
        "reactivated_count": len(reactivated_ids),
        "total_active": total_active,
        "rebuild_indexes_required": rebuild_required,
        "vectors": all_vectors,
    }


# ============================================================================
# FAISS INDEX MANAGEMENT
# ============================================================================

def create_index(
    index_name: str,
    index_type: str = "flat",
    nlist: int = 20,
    nprobe: int = 10,
    m: int = 16,
    ef_search: int = 32,
    embeddings_npy_filename: str = "employee_embeddings.npy",
) -> dict:
    """
    Create a new FAISS IndexIDMap2 index from the specified embeddings .npy file.

    The base index (flat / ivf / hnsw / ivf_hnsw) is wrapped in
    faiss.IndexIDMap2 so that each vector is stored with the employee's
    deterministic faiss_id (from the SQLite embedding_registry).

    Args:
        embeddings_npy_filename: The .npy file to load vectors from (default: employee_embeddings.npy)
    """
    # Ensure .npy extension
    if not embeddings_npy_filename.endswith(".npy"):
        embeddings_npy_filename += ".npy"

    # ── Step 1: Validation ───────────────────────────────────────────────
    index_path = f"{INDEX_DIR}/{index_name}.index"
    if os.path.exists(index_path):
        return {"error": "Index already exists. Use append to add new vectors."}

    if not os.path.exists(embeddings_npy_filename):
        return {"error": f"No embeddings found at '{embeddings_npy_filename}'. Generate embeddings first."}

    # ── Step 2: Load vectors ─────────────────────────────────────────────
    vectors = np.load(embeddings_npy_filename)
    embed_dim = vectors.shape[1]
    logging.info(f"[INDEX] Loaded {vectors.shape[0]} vectors (dim={embed_dim}) from {embeddings_npy_filename}")

    # ── Step 3: Load active faiss_ids from SQLite registry ───────────────
    active_registry = _get_embedded_ids()          # {employee_id: faiss_id}
    if len(active_registry) != vectors.shape[0]:
        logging.warning(
            f"[INDEX] Vector count ({vectors.shape[0]}) != active registry count "
            f"({len(active_registry)}). The embeddings file may be out of sync. "
            f"Using the first {min(vectors.shape[0], len(active_registry))} entries."
        )

    # Build the faiss_ids array preserving the same order as the legacy
    # employee_ids_order.json (which matches the .npy row order).
    if os.path.exists("employee_ids_order.json"):
        with open("employee_ids_order.json", "r") as f:
            ordered_emp_ids = json.load(f)
    else:
        # Fallback: alphabetical from registry
        ordered_emp_ids = sorted(active_registry.keys())

    faiss_ids = np.array(
        [active_registry[eid] for eid in ordered_emp_ids if eid in active_registry],
        dtype="int64",
    )
    # Trim vectors if registry has fewer entries
    n = min(vectors.shape[0], len(faiss_ids))
    vectors = vectors[:n]
    faiss_ids = faiss_ids[:n]
    logging.info(f"[INDEX] Using {n} vector–ID pairs for index '{index_name}'")

    # ── Step 4: Build the base index ─────────────────────────────────────
    if index_type == "flat":
        base_index = faiss.IndexFlatIP(embed_dim)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(embed_dim)
        base_index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist)
        base_index.train(vectors)                  # IVF needs training
        base_index.nprobe = nprobe
    elif index_type == "hnsw":
        base_index = faiss.IndexHNSWFlat(embed_dim, m)
        base_index.hnsw.efSearch = ef_search
    elif index_type == "ivf_hnsw":
        quantizer = faiss.IndexHNSWFlat(embed_dim, m)
        base_index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist)
        base_index.train(vectors)                  # IVF needs training
        base_index.nprobe = nprobe
        base_index.quantizer.hnsw.efSearch = ef_search
    else:
        return {"error": f"Invalid index type: {index_type}"}

    # ── Step 5: Wrap in IndexIDMap2 ──────────────────────────────────────
    # IndexIDMap2 stores the reverse mapping internally so that
    # faiss.read_index() will reconstruct it fully from disk.
    index = faiss.IndexIDMap2(base_index)
    logging.info(f"[INDEX] Base index wrapped in IndexIDMap2 (type={index_type})")

    # ── Step 6: Add vectors with their deterministic faiss_ids ───────────
    index.add_with_ids(vectors, faiss_ids)
    logging.info(f"[INDEX] Added {n} vectors with IDs to IndexIDMap2")

    # ── Step 7: Persist & register ───────────────────────────────────────
    faiss.write_index(index, index_path)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO index_registry VALUES (?,?,?,?,?,?,?,?)",
        (index_name, index_type, nlist, nprobe, m, ef_search, int(n), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    logging.info(f"[INDEX] Index '{index_name}' created — {n} vectors, type={index_type}")
    return {"status": "ok", "index_name": index_name, "vector_count": int(n)}


def append_to_index(index_name: str, rebuild: bool = False, embeddings_npy_filename: str = "employee_embeddings.npy") -> dict:
    """
    Append new embeddings to an existing FAISS IndexIDMap2 index.

    If `rebuild=True` (set automatically when employees were deleted),
    the index is deleted and recreated from scratch so that stale
    vectors are removed.
    """
    # Ensure .npy extension
    if not embeddings_npy_filename.endswith(".npy"):
        embeddings_npy_filename += ".npy"

    index_path = f"{INDEX_DIR}/{index_name}.index"
    if not os.path.exists(index_path):
        return {"error": f"Index '{index_name}' does not exist."}
    if not os.path.exists(embeddings_npy_filename):
        return {"error": f"No embeddings found at '{embeddings_npy_filename}'."}

    # ── Rebuild path (employees were deleted → full recreation) ──────────
    if rebuild:
        logging.info(f"[INDEX] Rebuild requested for '{index_name}' — recreating from scratch.")
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT index_type, nlist, nprobe, m, ef_search FROM index_registry WHERE index_name=?",
            (index_name,),
        ).fetchone()
        conn.close()
        if row is None:
            return {"error": f"Index '{index_name}' not in registry."}
        idx_type, nlist, nprobe, m, ef_search = row
        # Remove old files so create_index does not complain
        os.remove(index_path)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM index_registry WHERE index_name=?", (index_name,))
        conn.commit()
        conn.close()
        return create_index(index_name, idx_type, nlist, nprobe, m, ef_search, embeddings_npy_filename=embeddings_npy_filename)

    # ── Normal append path ───────────────────────────────────────────────
    index = faiss.read_index(index_path)
    all_vectors = np.load(embeddings_npy_filename)
    active_registry = _get_embedded_ids()          # {employee_id: faiss_id}

    # Determine which faiss_ids are already in the index.
    # IndexIDMap2 stores an id_map we can inspect.
    existing_ids_in_index = set()
    if hasattr(index, 'id_map') and index.id_map.size() > 0:
        existing_ids_in_index = set(
            int(index.id_map.at(i)) for i in range(index.id_map.size())
        )
    else:
        # Fallback: treat all vectors in the index as existing
        existing_ids_in_index = set()
    logging.info(f"[INDEX] Existing IDs in index: {len(existing_ids_in_index)}")

    # Get ordered employee list (matches .npy row order)
    if os.path.exists("employee_ids_order.json"):
        with open("employee_ids_order.json", "r") as f:
            ordered_emp_ids = json.load(f)
    else:
        ordered_emp_ids = sorted(active_registry.keys())

    # Find NEW IDs not yet in the index
    new_indices = []    # row positions in the .npy file
    new_faiss_ids = []  # corresponding faiss_ids
    for i, eid in enumerate(ordered_emp_ids):
        if eid not in active_registry:
            continue
        fid = active_registry[eid]
        if fid not in existing_ids_in_index and i < all_vectors.shape[0]:
            new_indices.append(i)
            new_faiss_ids.append(fid)

    if not new_indices:
        logging.info(f"[INDEX] No new vectors to append to '{index_name}'.")
        return {"status": "ok", "message": "No new vectors to append.", "total": int(index.ntotal)}

    new_vectors = all_vectors[np.array(new_indices)]
    new_faiss_ids_arr = np.array(new_faiss_ids, dtype="int64")

    index.add_with_ids(new_vectors, new_faiss_ids_arr)
    faiss.write_index(index, index_path)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE index_registry SET vector_count=? WHERE index_name=?",
        (int(index.ntotal), index_name),
    )
    conn.commit()
    conn.close()
    logging.info(
        f"[INDEX] Appended {len(new_indices)} vectors to '{index_name}'. "
        f"Total now: {index.ntotal}"
    )
    return {
        "status": "ok",
        "appended": len(new_indices),
        "total": int(index.ntotal),
    }


def delete_index(index_name: str) -> dict:
    """Delete a FAISS index from disk and registry."""
    path = f"{INDEX_DIR}/{index_name}.index"
    if not os.path.exists(path):
        return {"error": f"Index '{index_name}' not found."}
    os.remove(path)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM index_registry WHERE index_name=?", (index_name,))
    conn.commit()
    conn.close()
    logging.info(f"Index {index_name} deleted.")
    return {"status": "ok"}


def list_indexes() -> List[str]:
    """Return names of all registered FAISS indexes."""
    conn = sqlite3.connect(DB_PATH)
    names = pd.read_sql("SELECT index_name FROM index_registry", conn)["index_name"].tolist()
    conn.close()
    return names


def list_npy_files() -> List[str]:
    """Return names of all .npy embedding files in the workspace directory."""
    npy_files = [f for f in os.listdir(".") if f.endswith(".npy")]
    npy_files.sort()
    return npy_files


# ============================================================================
# LLM CALLS
# ============================================================================
# NOTE: extract_project_metadata() is imported from llm_client.py
#       (which uses prompt.py for the extraction prompt and schema.py
#       for Pydantic validation). No duplicate here.
#
# _llm_chat() below is used ONLY for the reasoning/explanation step.
# ============================================================================

def _llm_chat(prompt: str) -> str:
    """
    Synchronous LLM chat/completions call using OpenAI API.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-5.1")
    max_tokens = int(os.getenv("OPENAI_LLM_MAX_TOKENS", "4096"))

    if not openai_api_key:
        return "LLM call failed, check the OpenAI API key"

    endpoint_url = f"{openai_api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }
    data = {
        "model": openai_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    try:
        # Reduced timeout to 120s; 300s is often too long for user-facing apps
        resp = requests.post(endpoint_url, json=data, headers=headers, timeout=120)
        resp.raise_for_status() # Automatically handles 4xx/5xx errors
        
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Basic cleaning of markdown if prompt asks for JSON
        return content.replace("```json", "").replace("```", "").strip()

    except requests.exceptions.HTTPError as e:
        return f"API Error (HTTP {resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Connection Error: {str(e)}"


# ============================================================================
# HYBRID SCORING
# ============================================================================

def calculate_score(
    emp_group: pd.DataFrame,
    requirement: dict,
    weights: dict,
) -> Tuple[float, float]:
    """
    Calculate a hybrid match score for an employee against a requirement.

    Returns (hybrid_score 0-100, calibrated 0.0-1.0).
    """
    first = emp_group.iloc[0]

    emp_skills = {
        row["skill"].strip().lower(): row["skill_experience_years"]
        for _, row in emp_group.iterrows()
        if row["skill"]
    }

    mandatory = [s.strip().lower() for s in requirement.get("mandatory_skills", [])]

    # 1. Mandatory skill coverage
    if mandatory:
        matched = len(set(emp_skills.keys()) & set(mandatory))
        mandatory_score = matched / len(mandatory)
    else:
        mandatory_score = 1.0

    # 2. Skill experience score
    skill_scores = []
    for skill in mandatory:
        if skill in emp_skills:
            if "skill_requirements" in requirement and skill in requirement["skill_requirements"]:
                req_years = requirement["skill_requirements"][skill]
                ratio = min(emp_skills[skill] / req_years, 1.0) if req_years > 0 else 1.0
            else:
                ratio = 1.0
        else:
            ratio = 0.0
        skill_scores.append(ratio)
    skill_experience_score = sum(skill_scores) / len(skill_scores) if skill_scores else 0

    # 3. Domain score
    domain_req = requirement.get("domain_experience", 0)
    domain_score = min(first["domain_experience_years"] / domain_req, 1.0) if domain_req > 0 else 1.0

    # 4. Total experience score
    exp_score = min(first["total_experience_years"] / 10, 1.0)

    # 5. Weighted technical score
    technical_score = (
        weights.get("mandatory", 0.4) * mandatory_score
        + weights.get("skill_exp", 0.3) * skill_experience_score
        + weights.get("domain", 0.2) * domain_score
        + weights.get("experience", 0.1) * exp_score
    )

    # 6. Availability factor
    bandwidth = first.get("current_bandwidth_percent", 100)
    availability_factor = 0.5 + (bandwidth / 200)

    # 7. Final hybrid score
    final = technical_score * availability_factor * 100
    calibrated = round(final / 100, 3)
    return round(final, 2), calibrated


def compute_final_rank(embedding_similarity: float, calibrated_rule_score: float) -> float:
    """Final Rank = 0.6 × Embedding Similarity + 0.4 × Rule Score."""
    return round(0.6 * embedding_similarity + 0.4 * calibrated_rule_score, 4)


# ============================================================================
# LLM EXPLANATION
# ============================================================================

async def llm_reason(emp_text: str, req_text: str, score: float) -> str:
    """Generate an executive-ready LLM explanation for a match."""
    prompt = f"""You are an Enterprise Workforce Intelligence Advisor.

Evaluate whether the employee should be allocated to the given role.
Provide a structured, executive-ready analysis.

Employee Profile:
{emp_text}

Role Requirement:
{req_text}

Final Match Score (0–1): {score}

Provide the response in the following structured format:

1. Executive Fit Summary (Overall Fit, Deployment Readiness, Risk Level, Confidence)
2. Competency Alignment Analysis (Technical, Domain, Seniority, Capacity)
3. Gap & Risk Assessment (Critical Gaps, Development Gaps, Operational Risks)
4. Deployment Recommendation (Action, Ramp-up Plan, Mitigation, Checkpoints)
5. Upskilling & Development Plan (Priority Skills, Certifications, 30-60-90 Plan)
6. Strategic Workforce Value (Growth Potential, Leadership Pipeline, Versatility)
7. Resource Optimization Insight (Overqualified?, Opportunity Cost, Career Impact)
"""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _llm_chat(prompt))
        return result
    except Exception as e:
        return f"LLM reasoning error: {e}"


# ============================================================================
# FULL SKILL-MAPPING SEARCH PIPELINE
# ============================================================================

def skill_mapping_search(
    requirement_text: str,
    index_name: str,
    weights: Optional[dict] = None,
    top_n: int = 20,
    explain_top: int = 10,
    embedding_model: str = "text-embedding-3-large",
) -> dict:
    """
    End-to-end skill-mapping search.

    Args:
        requirement_text: Raw free-text requirement (project description, JD, etc.)
        index_name: Name of the FAISS index to search.
        weights: Scoring weight dict {mandatory, skill_exp, domain, experience}.
        top_n: Number of top matches to return.
        explain_top: How many top matches get LLM explanations.
        embedding_model: "text-embedding-3-large" (API) or "all-MiniLM-L6-v2" (local)

    Returns:
        dict with keys: extracted_metadata, display_text, embedding_text,
                        search_latency_sec, results, explanations
    """
    if weights is None:
        weights = {"mandatory": 0.4, "skill_exp": 0.3, "domain": 0.2, "experience": 0.1}

    # 1. Extract metadata via LLM
    extracted = extract_project_metadata(requirement_text)
    if "error" in extracted:
        return {"error": "Requirement extraction failed", "details": extracted}

    # 2. Flatten for display & embedding
    display_text = flatten_requirement_display(extracted)
    embedding_text = flatten_requirement_for_embedding(extracted)

    # 3. Generate requirement embedding using the selected model
    use_local = (embedding_model == "all-MiniLM-L6-v2")
    if use_local:
        req_emb = get_embedding_local(embedding_text)
    else:
        req_emb = get_embedding(embedding_text)
    if req_emb is None:
        return {"error": "Failed to generate embedding for the requirement."}
    req_vector = normalize_embeddings(np.array([req_emb], dtype="float32"))

    # 4. Build requirement dict for rule-based scoring
    requirement = {
        "domain": ", ".join(extracted.get("domain", [])),
        "domain_experience": extracted.get("minimum_overall_experience_years") or 0,
        "mandatory_skills": [s.get("skill", "") for s in extracted.get("mandatory_skills", [])],
    }

    # 5. Load FAISS IndexIDMap2 index
    # ── The index was built with faiss.IndexIDMap2, so search results
    #    in I[0] return the deterministic faiss_ids (not positional offsets).
    #    We reverse-lookup each faiss_id → employee_id via SQLite.
    index_path = f"{INDEX_DIR}/{index_name}.index"
    if not os.path.exists(index_path):
        return {"error": f"Index '{index_name}' not found."}
    index = faiss.read_index(index_path)

    # Apply search params from registry
    conn = sqlite3.connect(DB_PATH)
    registry = pd.read_sql(
        f"SELECT * FROM index_registry WHERE index_name='{index_name}'", conn
    )
    conn.close()
    if registry.empty:
        return {"error": f"Index '{index_name}' not in registry."}
    reg = registry.iloc[0]

    # For IndexIDMap2-wrapped indexes, nprobe / efSearch must be set on
    # the *underlying* index, not the wrapper.  We access it via index.index
    # (the base index inside the IDMap wrapper).
    base = index.index if hasattr(index, "index") else index
    if reg["index_type"] in ("ivf", "ivf_hnsw"):
        base.nprobe = reg["nprobe"]
    if reg["index_type"] == "hnsw":
        base.hnsw.efSearch = reg["ef_search"]
    if reg["index_type"] == "ivf_hnsw":
        base.quantizer.hnsw.efSearch = reg["ef_search"]

    # 6. FAISS search — I[0] now contains faiss_ids (not positional indexes)
    start = time.time()
    D, I = index.search(req_vector, min(50, index.ntotal))
    latency = round(time.time() - start, 3)

    # 7. Load employee data
    df = get_employee_dataframe()
    grouped = df.groupby("employee_id")

    # 8. Build results — reverse-lookup faiss_id → employee_id via SQLite
    results = []
    seen: set = set()
    for rank, faiss_id in enumerate(I[0]):
        if faiss_id < 0:
            continue  # -1 means "no more results"
        emp_id = _faiss_id_to_emp_id(int(faiss_id))
        if emp_id is None:
            logging.warning(f"[SEARCH] faiss_id {faiss_id} not found in embedding_registry — skipping.")
            continue
        if emp_id in seen:
            continue
        seen.add(emp_id)

        emp_group = grouped.get_group(emp_id)
        similarity = float(D[0][rank])
        rule_score, calibrated_rule = calculate_score(emp_group, requirement, weights)
        final_rank = compute_final_rank(similarity, calibrated_rule)
        confidence = "High" if final_rank >= 0.82 else "Medium" if final_rank >= 0.65 else "Low"

        first = emp_group.iloc[0]
        unique_skills = list(dict.fromkeys(emp_group["skill"].tolist()))

        results.append({
            "employee_id": str(first["employee_id"]),
            "employee_name": str(first["employee_name"]),
            "domain": str(first["domain"]),
            "total_experience_years": float(first["total_experience_years"]),
            "current_bandwidth_percent": float(first["current_bandwidth_percent"]),
            "skills": unique_skills,
            "similarity": round(similarity, 4),
            "rule_score": rule_score,
            "calibrated_rule_score": calibrated_rule,
            "final_rank": final_rank,
            "confidence": confidence,
        })

    results.sort(key=lambda x: x["final_rank"], reverse=True)
    top_results = results[:top_n]

    # 9. LLM explanations for top N
    explanations: List[str] = []
    if explain_top > 0:
        async def _run_explanations():
            tasks = []
            for r in top_results[:explain_top]:
                emp_group = grouped.get_group(r["employee_id"])
                emp_text = flatten_employee(emp_group)
                tasks.append(llm_reason(emp_text, embedding_text, r["final_rank"]))
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If called from an already-running async context (like FastAPI)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    explanations = list(pool.submit(asyncio.run, _run_explanations()).result())
            else:
                explanations = asyncio.run(_run_explanations())
        except RuntimeError:
            explanations = asyncio.run(_run_explanations())

    # Attach explanations to results
    for i, expl in enumerate(explanations):
        if i < len(top_results):
            top_results[i]["explanation"] = expl

    return {
        "extracted_metadata": extracted,
        "display_text": display_text,
        "embedding_text": embedding_text,
        "search_latency_sec": latency,
        "total_matches": len(results),
        "results": top_results,
    }
