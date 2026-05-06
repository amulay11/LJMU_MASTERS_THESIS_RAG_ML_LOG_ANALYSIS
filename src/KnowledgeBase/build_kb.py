"""
build_kb.py — One-time builder for the LogSense Qdrant vector knowledge base.

Run from the Experiments root:
    python src/KnowledgeBase/build_kb.py

Collections created
-------------------
Dataset-specific (separated so retrieval never crosses datasets):
    bgl_logs          BGL few-shot log samples  (label: Normal / Anomalous)
    bgl_architecture  BGL system architecture reference
    bgl_severity      BGL RAS severity / error taxonomy
    bgl_rca           BGL root-cause analysis examples

    hdfs_logs         HDFS block-trace samples  (label: Normal / Anomalous)
    hdfs_architecture HDFS system architecture reference
    hdfs_severity     HDFS severity / error taxonomy
    hdfs_rca          HDFS root-cause analysis examples

Role-based (shared across datasets — retrieval scoped by role at query time):
    role_sre          SRE on-call reference guide
    role_devops       DevOps engineer reference guide

Each collection stores rich payloads for filtering:
    text, dataset, category, source, label (logs), rca_id, rca_section,
    log_component, log_level, anomaly_type

Requires
--------
    pip install qdrant-client sentence-transformers python-docx pandas tqdm
"""

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from docx import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR   = Path(__file__).resolve().parent          # src/KnowledgeBase/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # Experiments/
KB_DATA_DIR  = PROJECT_ROOT / "Documents"           # Experiments/KnowledgeBase/
DATASETS_DIR = PROJECT_ROOT / "Datasets"
QDRANT_PATH  = SCRIPT_DIR / "qdrant_store"              # persistent local DB

BGL_STRUCTURED_LOG = DATASETS_DIR / "BGL" / "Sample" / "BGL_2k.log_structured.csv"
BGL_TEMPLATES      = DATASETS_DIR / "BGL" / "Sample" / "BGL_2k.log_templates.csv"
HDFS_TRACES        = DATASETS_DIR / "HDFS" / "Full_HDFS_v1" / "preprocessed" / "Event_traces.csv"
HDFS_TEMPLATES     = DATASETS_DIR / "HDFS" / "Full_HDFS_v1" / "preprocessed" / "HDFS.log_templates.csv"

# =============================================================================
# CONSTANTS
# =============================================================================

EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
EMBED_DIM        = 768
EMBED_BATCH_SIZE = 64
RANDOM_SEED      = 42

# Test-set caps (must match notebook values to exclude the correct logs)
TEST_NORMAL_CAP    = 200
TEST_ANOMALY_CAP   = 15
N_FEW_SHOT_NORMAL  = 5
N_FEW_SHOT_ANOM    = 5

# KB log optimisation: deduplicate by template, then cap normals at this
# multiple of anomalous entries to reduce redundancy and class imbalance.
BGL_KB_NORMAL_RATIO  = 2

# HDFS traces have no shared template structure — cap each class directly.
HDFS_KB_ANOMALY_CAP = 100
HDFS_KB_NORMAL_CAP  = 200

MAX_CHUNK_CHARS = 600   # target chunk size for docx paragraphs

COLLECTIONS = [
    "bgl_logs", "bgl_architecture", "bgl_severity", "bgl_rca",
    "hdfs_logs", "hdfs_architecture", "hdfs_severity", "hdfs_rca",
    "role_sre", "role_devops",
]

# =============================================================================
# HELPERS — JSON
# =============================================================================

def load_rca_json(path: Path) -> List[Dict]:
    """Load an RCA JSON file, tolerating a missing closing bracket."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw.endswith("]"):
        raw += "\n]"
    return json.loads(raw)


# =============================================================================
# HELPERS — DOCX
# =============================================================================

def _is_heading(para) -> bool:
    style_name = para.style.name if para.style else ""
    return style_name.startswith("Heading") or style_name == "Title"


def extract_docx_chunks(path: Path, max_chars: int = MAX_CHUNK_CHARS) -> List[Dict]:
    """
    Extract text chunks from a Word document.

    Strategy:
    - Track section titles from heading paragraphs.
    - Accumulate body paragraphs into chunks ≤ max_chars.
    - Each chunk carries its section title as metadata.
    - Very long single paragraphs are split on sentence boundaries.
    """
    if not path.exists():
        print(f"  SKIP (not found): {path.name}")
        return []

    doc     = Document(str(path))
    chunks  = []
    section = "Introduction"
    buf     = []
    buf_len = 0

    def flush(section: str, buf: List[str]) -> None:
        text = " ".join(buf).strip()
        if text:
            chunks.append({"text": f"[{section}] {text}", "section": section})

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if _is_heading(para):
            flush(section, buf)
            buf, buf_len = [], 0
            section = text
            continue

        # Split very long paragraphs on sentence boundary
        if len(text) > max_chars * 1.5:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                if buf_len + len(sent) > max_chars and buf:
                    flush(section, buf)
                    buf, buf_len = [], 0
                buf.append(sent)
                buf_len += len(sent)
        else:
            if buf_len + len(text) > max_chars and buf:
                flush(section, buf)
                buf, buf_len = [], 0
            buf.append(text)
            buf_len += len(text)

    flush(section, buf)
    return chunks


# =============================================================================
# HELPERS — RCA JSON → DOCUMENTS
# =============================================================================

def rca_to_documents(entries: List[Dict]) -> List[Dict]:
    """
    Convert RCA JSON entries into three document types per entry:
        overview  — anomaly type + summary + confidence
        analysis  — detailed description + causal chain
        evidence  — supporting evidence (logs, templates, context, indicators)
    """
    docs = []
    for e in entries:
        rid = e.get("rca_id", "?")
        at  = e.get("anomaly_type", "")
        ds  = e.get("dataset", "")

        # 1. Overview
        docs.append({
            "text": (
                f"[RCA Overview | {rid}] {at}\n"
                f"Summary: {e.get('root_cause_summary', '')}\n"
                f"Confidence: {e.get('confidence_level', '')} — "
                f"{e.get('confidence_reasoning', '')}"
            ),
            "rca_id":      rid,
            "rca_section": "overview",
            "anomaly_type": at,
            "dataset":     ds,
        })

        # 2. Analysis + causal chain
        chain_text = ""
        for step in e.get("causal_chain", []):
            chain_text += (
                f"\n  [{step.get('timestamp', '')}] {step.get('event', '')}: "
                f"{step.get('description', '')}"
            )
        docs.append({
            "text": (
                f"[RCA Analysis | {rid}] {at}\n"
                f"{e.get('root_cause_description', '')}\n"
                f"Causal Chain:{chain_text}"
            ),
            "rca_id":       rid,
            "rca_section":  "analysis",
            "anomaly_type": at,
            "dataset":      ds,
        })

        # 3. Evidence
        se = e.get("supporting_evidence", {})
        ev_parts = []
        for log in se.get("relevant_logs", []):
            ev_parts.append(f"Log: {log}")
        for tmpl in se.get("log_templates", []):
            ev_parts.append(f"Template: {tmpl}")
        if se.get("system_context"):
            ev_parts.append(f"Context: {se['system_context']}")
        if se.get("historical_similarity"):
            ev_parts.append(f"Historical: {se['historical_similarity']}")
        for ind in se.get("system_state_indicators", []):
            ev_parts.append(f"Indicator: {ind}")

        docs.append({
            "text": (
                f"[RCA Evidence | {rid}] {at}\n"
                + "\n".join(ev_parts)
            ),
            "rca_id":       rid,
            "rca_section":  "evidence",
            "anomaly_type": at,
            "dataset":      ds,
        })

    return docs


# =============================================================================
# HELPERS — LOG SAMPLES
# =============================================================================

def _build_bgl_log_text(row: pd.Series) -> str:
    comp    = str(row.get("Component", "")).strip()
    level   = str(row.get("Level", "")).strip()
    content = str(row.get("Content", "")).strip()
    tmpl    = str(row.get("EventTemplate", "")).strip()
    return f"[{comp}] [{level}] {content} | Template: {tmpl}"


def load_bgl_logs() -> List[Dict]:
    """
    Load BGL log samples, excluding the test set (same split as E02 notebook).
    Returns all non-test logs as document dicts for the KB.
    """
    if not BGL_STRUCTURED_LOG.exists():
        print(f"  SKIP BGL logs (not found): {BGL_STRUCTURED_LOG}")
        return []

    df_logs      = pd.read_csv(BGL_STRUCTURED_LOG)
    df_templates = pd.read_csv(BGL_TEMPLATES)
    df = df_logs.merge(df_templates, on="EventId", how="left", suffixes=("", "_tmpl"))
    df["is_normal"]    = df["Label"] == "-"
    df["binary_label"] = (~df["is_normal"]).astype(int)
    df["log_text"]     = df.apply(_build_bgl_log_text, axis=1)

    rng      = np.random.default_rng(RANDOM_SEED)
    df_norm  = df[df["is_normal"]].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_anom  = df[~df["is_normal"]].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Mirror the exact split used in E02 notebooks
    # Few-shot pool (first N rows after shuffle)
    fs_norm_end  = N_FEW_SHOT_NORMAL
    fs_anom_end  = N_FEW_SHOT_ANOM
    # Test set (next slice after few-shot)
    test_norm_end = fs_norm_end + TEST_NORMAL_CAP
    test_anom_end = fs_anom_end + TEST_ANOMALY_CAP

    # KB = everything OUTSIDE the test slice (few-shot pool + remainder)
    kb_norm = pd.concat([
        df_norm.iloc[:fs_norm_end],          # few-shot normals (already in prompt)
        df_norm.iloc[test_norm_end:],        # normals beyond test cap
    ], ignore_index=True)
    kb_anom = pd.concat([
        df_anom.iloc[:fs_anom_end],          # few-shot anomalies
        df_anom.iloc[test_anom_end:],        # anomalies beyond test cap
    ], ignore_index=True)

    raw_norm, raw_anom = len(kb_norm), len(kb_anom)

    # Step 1 — Template-level deduplication.
    # Rows are already shuffled, so .first() picks a random representative
    # per unique EventTemplate, eliminating near-duplicate vectors.
    kb_norm = kb_norm.groupby("EventTemplate", sort=False).first().reset_index()
    kb_anom = kb_anom.groupby("EventTemplate", sort=False).first().reset_index()

    # Step 2 — Balance classes.
    # Cap normals at BGL_KB_NORMAL_RATIO × anomalies so anomaly signal
    # is not drowned out during retrieval.
    max_norm = min(len(kb_norm), BGL_KB_NORMAL_RATIO * len(kb_anom))
    kb_norm = kb_norm.sample(n=max_norm, random_state=RANDOM_SEED)

    print(f"  BGL KB (before dedup/balance): Normal={raw_norm}, Anomalous={raw_anom}")
    print(f"  BGL KB (after  dedup/balance): Normal={len(kb_norm)}, Anomalous={len(kb_anom)}  "
          f"(ratio {BGL_KB_NORMAL_RATIO}:1, 1 rep/template)")

    docs = []
    for _, row in kb_norm.iterrows():
        docs.append({
            "text":          row["log_text"],
            "label":         "Normal",
            "log_component": str(row.get("Component", "")).strip(),
            "log_level":     str(row.get("Level", "")).strip(),
            "dataset":       "BGL",
        })
    for _, row in kb_anom.iterrows():
        docs.append({
            "text":          row["log_text"],
            "label":         "Anomalous",
            "log_component": str(row.get("Component", "")).strip(),
            "log_level":     str(row.get("Level", "")).strip(),
            "dataset":       "BGL",
        })

    return docs


def _clean_template(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[\*\]", "", text)).strip()


def load_hdfs_logs() -> List[Dict]:
    """
    Load HDFS block-trace samples, excluding the test set.
    Skips gracefully if preprocessed data is not available.
    """
    if not HDFS_TRACES.exists() or not HDFS_TEMPLATES.exists():
        print("  SKIP HDFS logs (preprocessed data not found — gitignored large file)")
        return []

    df_traces    = pd.read_csv(HDFS_TRACES)
    df_templates = pd.read_csv(HDFS_TEMPLATES)
    tmpl_lookup  = dict(zip(df_templates["EventId"], df_templates["EventTemplate"]))

    df_traces["is_normal"]    = df_traces["Label"] == "Success"
    df_traces["binary_label"] = (~df_traces["is_normal"]).astype(int)

    def build_hdfs_text(row: pd.Series) -> str:
        feat = str(row.get("Features", "[]"))
        eids = re.findall(r"E\d+", feat)
        seen, tmpls = set(), []
        for eid in eids:
            if eid not in seen:
                seen.add(eid)
                tmpls.append(_clean_template(tmpl_lookup.get(eid, eid)))
        return "HDFS Block Trace | " + " -> ".join(tmpls)

    df_traces["log_text"] = df_traces.apply(build_hdfs_text, axis=1)

    df_norm = df_traces[df_traces["is_normal"]].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_anom = df_traces[~df_traces["is_normal"]].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    test_norm_end = N_FEW_SHOT_NORMAL  + TEST_NORMAL_CAP
    test_anom_end = N_FEW_SHOT_ANOM    + TEST_ANOMALY_CAP

    kb_norm = pd.concat([df_norm.iloc[:N_FEW_SHOT_NORMAL],  df_norm.iloc[test_norm_end:]], ignore_index=True)
    kb_anom = pd.concat([df_anom.iloc[:N_FEW_SHOT_ANOM],    df_anom.iloc[test_anom_end:]], ignore_index=True)

    raw_norm, raw_anom = len(kb_norm), len(kb_anom)

    # Cap HDFS classes — traces are unique sequences so template-dedup doesn't
    # apply; a random sample keeps the KB manageable without losing diversity.
    kb_norm = kb_norm.sample(n=min(HDFS_KB_NORMAL_CAP,  len(kb_norm)), random_state=RANDOM_SEED)
    kb_anom = kb_anom.sample(n=min(HDFS_KB_ANOMALY_CAP, len(kb_anom)), random_state=RANDOM_SEED)

    print(f"  HDFS KB (before cap): Normal={raw_norm}, Anomalous={raw_anom}")
    print(f"  HDFS KB (after  cap): Normal={len(kb_norm)}, Anomalous={len(kb_anom)}")

    docs = []
    for _, row in kb_norm.iterrows():
        docs.append({"text": row["log_text"], "label": "Normal",    "dataset": "HDFS", "log_component": "", "log_level": ""})
    for _, row in kb_anom.iterrows():
        docs.append({"text": row["log_text"], "label": "Anomalous", "dataset": "HDFS", "log_component": "", "log_level": ""})

    return docs


# =============================================================================
# EMBEDDING
# =============================================================================

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding", leave=False):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


# =============================================================================
# QDRANT — COLLECTION MANAGEMENT
# =============================================================================

def create_collection(client: QdrantClient, name: str, overwrite: bool = True) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        if overwrite:
            client.delete_collection(name)
        else:
            print(f"  EXISTS (skipping): {name}")
            return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def upsert_collection(
    client:     QdrantClient,
    model:      SentenceTransformer,
    collection: str,
    docs:       List[Dict],
    category:   str,
    dataset:    str,
    source:     str,
) -> None:
    """Embed docs and upsert into Qdrant with standardised payload."""
    if not docs:
        print(f"  EMPTY — skipping {collection}")
        return

    texts   = [d["text"] for d in docs]
    vectors = embed_texts(model, texts)

    points = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        payload = {
            "text":          doc["text"],
            "dataset":       dataset,
            "category":      category,
            "source":        source,
            "label":         doc.get("label", ""),
            "rca_id":        doc.get("rca_id", ""),
            "rca_section":   doc.get("rca_section", ""),
            "anomaly_type":  doc.get("anomaly_type", ""),
            "section":       doc.get("section", ""),
            "log_component": doc.get("log_component", ""),
            "log_level":     doc.get("log_level", ""),
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload=payload,
        ))

    # Upsert in batches of 256
    batch_size = 256
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[i : i + batch_size])

    print(f"  Upserted {len(points):>5} docs -> {collection}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 64)
    print("  LogSense Knowledge Base Builder")
    print(f"  Qdrant store : {QDRANT_PATH}")
    print(f"  Embed model  : {EMBED_MODEL}")
    print("=" * 64)

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    print("\nLoading embedding model ...")
    model = SentenceTransformer(EMBED_MODEL)
    print("  Model loaded")

    # Create all collections (overwrite existing)
    print("\nCreating collections ...")
    for name in COLLECTIONS:
        create_collection(client, name, overwrite=True)
        print(f"  Created: {name}")

    # ── BGL LOGS ──────────────────────────────────────────────────────────────
    print("\n[1/10] BGL log samples ...")
    bgl_log_docs = load_bgl_logs()
    upsert_collection(client, model, "bgl_logs", bgl_log_docs,
                      category="logs", dataset="BGL", source="BGL_2k.log_structured.csv")

    # ── BGL ARCHITECTURE ──────────────────────────────────────────────────────
    print("\n[2/10] BGL system architecture ...")
    arch_path = KB_DATA_DIR / "BGL" / "BlueGeneL_SystemArchitecture.docx"
    bgl_arch_docs = extract_docx_chunks(arch_path)
    upsert_collection(client, model, "bgl_architecture", bgl_arch_docs,
                      category="architecture", dataset="BGL", source=arch_path.name)

    # ── BGL SEVERITY ──────────────────────────────────────────────────────────
    print("\n[3/10] BGL RAS severity taxonomy ...")
    sev_path = KB_DATA_DIR / "BGL" / "BGL_RAS_Severity_Taxonomy.docx"
    bgl_sev_docs = extract_docx_chunks(sev_path)
    upsert_collection(client, model, "bgl_severity", bgl_sev_docs,
                      category="severity", dataset="BGL", source=sev_path.name)

    # ── BGL RCA ───────────────────────────────────────────────────────────────
    print("\n[4/10] BGL RCA examples ...")
    bgl_rca_raw  = load_rca_json(KB_DATA_DIR / "BGL" / "bgl_rca.json")
    bgl_rca_docs = rca_to_documents(bgl_rca_raw)
    upsert_collection(client, model, "bgl_rca", bgl_rca_docs,
                      category="rca", dataset="BGL", source="bgl_rca.json")

    # ── HDFS LOGS ─────────────────────────────────────────────────────────────
    print("\n[5/10] HDFS log samples ...")
    hdfs_log_docs = load_hdfs_logs()
    upsert_collection(client, model, "hdfs_logs", hdfs_log_docs,
                      category="logs", dataset="HDFS", source="Event_traces.csv")

    # ── HDFS ARCHITECTURE ─────────────────────────────────────────────────────
    print("\n[6/10] HDFS system architecture ...")
    hdfs_arch_path = KB_DATA_DIR / "HDFS" / "HDFS_System_Architecture_Reference.docx"
    hdfs_arch_docs = extract_docx_chunks(hdfs_arch_path)
    upsert_collection(client, model, "hdfs_architecture", hdfs_arch_docs,
                      category="architecture", dataset="HDFS", source=hdfs_arch_path.name)

    # ── HDFS SEVERITY ─────────────────────────────────────────────────────────
    print("\n[7/10] HDFS severity taxonomy ...")
    hdfs_sev_path = KB_DATA_DIR / "HDFS" / "HDFS_Severity_Error_Taxonomy.docx"
    hdfs_sev_docs = extract_docx_chunks(hdfs_sev_path)
    upsert_collection(client, model, "hdfs_severity", hdfs_sev_docs,
                      category="severity", dataset="HDFS", source=hdfs_sev_path.name)

    # ── HDFS RCA ──────────────────────────────────────────────────────────────
    print("\n[8/10] HDFS RCA examples ...")
    hdfs_rca_raw  = load_rca_json(KB_DATA_DIR / "HDFS" / "hdfs_rca.json")
    hdfs_rca_docs = rca_to_documents(hdfs_rca_raw)
    upsert_collection(client, model, "hdfs_rca", hdfs_rca_docs,
                      category="rca", dataset="HDFS", source="hdfs_rca.json")

    # ── ROLE: SRE ─────────────────────────────────────────────────────────────
    print("\n[9/10] SRE reference guide ...")
    sre_path  = KB_DATA_DIR / "RoleBased Knowledge" / "SRE_Reference_Guide_RAG.docx"
    sre_docs  = extract_docx_chunks(sre_path)
    upsert_collection(client, model, "role_sre", sre_docs,
                      category="role_sre", dataset="SHARED", source=sre_path.name)

    # ── ROLE: DEVOPS ──────────────────────────────────────────────────────────
    print("\n[10/10] DevOps reference guide ...")
    devops_path = KB_DATA_DIR / "RoleBased Knowledge" / "DevOps_Reference_Guide_RAG.docx"
    devops_docs = extract_docx_chunks(devops_path)
    upsert_collection(client, model, "role_devops", devops_docs,
                      category="role_devops", dataset="SHARED", source=devops_path.name)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  BUILD COMPLETE — Collection summary:")
    print("=" * 64)
    for name in COLLECTIONS:
        info = client.get_collection(name)
        print(f"  {name:<22} {info.points_count:>6} points")
    print("=" * 64)


if __name__ == "__main__":
    main()
