"""
kb_utils.py — Query interface for the LogSense Qdrant vector knowledge base.

Usage in RAG notebooks
----------------------
    import sys, os
    sys.path.insert(0, os.path.abspath("../KnowledgeBase"))
    from kb_utils import KBClient

    kb = KBClient()

    # Retrieve similar BGL logs (both classes)
    results = kb.query_bgl_logs("machine check interrupt FATAL", top_k=5)

    # Retrieve only anomalous BGL logs
    results = kb.query_bgl_logs("data TLB error", top_k=5, label="Anomalous")

    # Architecture context for a BGL query
    results = kb.query_bgl_architecture("torus interconnect link failure", top_k=3)

    # Severity / risk context
    results = kb.query_bgl_severity("FATAL RAS hardware event", top_k=3)

    # RCA examples matching a fault pattern
    results = kb.query_bgl_rca("DDR memory test failure node card", top_k=3)

    # Role-specific remediation guidance
    results = kb.query_role("sre", "node isolation procedure hardware fault", top_k=3)
    results = kb.query_role("devops", "drain node scheduler re-queue job", top_k=3)

    # Multi-collection query (returns merged ranked list)
    results = kb.query_multi(
        ["bgl_architecture", "bgl_severity", "bgl_rca"],
        "KERNDTLB data TLB interrupt FATAL",
        top_k_per_collection=2,
    )

Each result dict contains:
    text, dataset, category, source, label, rca_id, rca_section,
    anomaly_type, section, log_component, log_level, score
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONSTANTS (keep in sync with build_kb.py)
# =============================================================================

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QDRANT_PATH = Path(__file__).resolve().parent / "qdrant_store"

# Module-level singleton — prevents "already accessed by another instance"
# errors when a notebook cell is re-run inside the same kernel.
_QDRANT_SINGLETONS: dict = {}

COLLECTION_NAMES = {
    "bgl_logs":          "bgl_logs",
    "bgl_architecture":  "bgl_architecture",
    "bgl_severity":      "bgl_severity",
    "bgl_rca":           "bgl_rca",
    "hdfs_logs":         "hdfs_logs",
    "hdfs_architecture": "hdfs_architecture",
    "hdfs_severity":     "hdfs_severity",
    "hdfs_rca":          "hdfs_rca",
    "role_sre":          "role_sre",
    "role_devops":       "role_devops",
}


# =============================================================================
# CLIENT
# =============================================================================

class KBClient:
    """
    Thin wrapper around QdrantClient for LogSense KB queries.

    Parameters
    ----------
    qdrant_path : path to the local Qdrant store directory.
                  Defaults to the qdrant_store/ folder next to this file.
    """

    def __init__(self, qdrant_path: Optional[Path] = None) -> None:
        path = qdrant_path or QDRANT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Qdrant store not found at {path}. "
                "Run src/KnowledgeBase/build_kb.py first."
            )
        key = str(path)
        if key not in _QDRANT_SINGLETONS:
            try:
                _QDRANT_SINGLETONS[key] = QdrantClient(path=key)
            except Exception as exc:
                _msg = str(exc).lower()
                if (
                    "AlreadyLocked" in type(exc).__name__
                    or "Permission" in str(exc)
                    or "lock" in _msg
                    or "already accessed" in _msg
                    or "concurrent access" in _msg
                ):
                    raise RuntimeError(
                        "Qdrant storage is locked by another kernel or a stale client.\n"
                        "Fix: in the kernel that holds the lock call KBClient.close_all(), "
                        "then re-run this cell.  If unsure which kernel, shut down all other "
                        "notebook kernels (Kernel → Shut Down All Kernels) and retry."
                    ) from None
                raise
        self._client = _QDRANT_SINGLETONS[key]
        self._model  = None   # lazy-loaded on first query

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def _embed(self, text: str) -> List[float]:
        """Embed a query string with BGE query prefix."""
        query = f"Represent this sentence for searching relevant passages: {text}"
        vec   = self._get_model().encode(query, normalize_embeddings=True)
        return vec.tolist()

    # ── GENERIC QUERY ─────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        text: str,
        top_k: int = 5,
        label: Optional[str] = None,
        rca_section: Optional[str] = None,
        extra_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search a single collection.

        Parameters
        ----------
        collection  : one of COLLECTION_NAMES values
        text        : natural-language query
        top_k       : number of results to return
        label       : filter to "Normal" or "Anomalous" (logs only)
        rca_section : filter to "overview", "analysis", or "evidence" (RCA only)
        extra_filter: additional qdrant FieldCondition kwargs (advanced)
        """
        must_conditions = []
        if label:
            must_conditions.append(
                FieldCondition(key="label", match=MatchValue(value=label))
            )
        if rca_section:
            must_conditions.append(
                FieldCondition(key="rca_section", match=MatchValue(value=rca_section))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        result = self._client.query_points(
            collection_name=collection,
            query=self._embed(text),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {**hit.payload, "score": round(hit.score, 4)}
            for hit in result.points
        ]

    # ── BGL CONVENIENCE METHODS ───────────────────────────────────────────────

    def query_bgl_logs(
        self, text: str, top_k: int = 5, label: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve BGL log samples. Filter by label='Normal'/'Anomalous' if needed."""
        return self.query("bgl_logs", text, top_k=top_k, label=label)

    def query_bgl_architecture(self, text: str, top_k: int = 3) -> List[Dict]:
        """Retrieve BGL system architecture context."""
        return self.query("bgl_architecture", text, top_k=top_k)

    def query_bgl_severity(self, text: str, top_k: int = 3) -> List[Dict]:
        """Retrieve BGL RAS severity / error taxonomy context."""
        return self.query("bgl_severity", text, top_k=top_k)

    def query_bgl_rca(
        self, text: str, top_k: int = 3, section: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve BGL RCA examples.
        section: 'overview' | 'analysis' | 'evidence' | None (all sections)
        """
        return self.query("bgl_rca", text, top_k=top_k, rca_section=section)

    # ── HDFS CONVENIENCE METHODS ──────────────────────────────────────────────

    def query_hdfs_logs(
        self, text: str, top_k: int = 5, label: Optional[str] = None
    ) -> List[Dict]:
        return self.query("hdfs_logs", text, top_k=top_k, label=label)

    def query_hdfs_architecture(self, text: str, top_k: int = 3) -> List[Dict]:
        return self.query("hdfs_architecture", text, top_k=top_k)

    def query_hdfs_severity(self, text: str, top_k: int = 3) -> List[Dict]:
        return self.query("hdfs_severity", text, top_k=top_k)

    def query_hdfs_rca(
        self, text: str, top_k: int = 3, section: Optional[str] = None
    ) -> List[Dict]:
        return self.query("hdfs_rca", text, top_k=top_k, rca_section=section)

    # ── ROLE-BASED METHODS ────────────────────────────────────────────────────

    def query_role(self, role: str, text: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve role-based guidance.
        role: 'sre' | 'devops'
        """
        role = role.lower().strip()
        if role not in ("sre", "devops"):
            raise ValueError("role must be 'sre' or 'devops'")
        collection = f"role_{role}"
        return self.query(collection, text, top_k=top_k)

    # ── MULTI-COLLECTION QUERY ────────────────────────────────────────────────

    def query_multi(
        self,
        collections: List[str],
        text: str,
        top_k_per_collection: int = 3,
    ) -> List[Dict]:
        """
        Query multiple collections and return a merged list ranked by score.

        Useful for RAG pipelines that need context from several categories
        (e.g. architecture + severity + RCA) in a single call.
        """
        results = []
        for col in collections:
            try:
                hits = self.query(col, text, top_k=top_k_per_collection)
                for h in hits:
                    h["_collection"] = col
                results.extend(hits)
            except Exception as exc:
                print(f"  Warning: query failed for collection '{col}': {exc}")
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ── UTILITY ───────────────────────────────────────────────────────────────

    @staticmethod
    def close_all() -> None:
        """Close every singleton QdrantClient and release their file locks.
        Call this before running build_kb.py from inside the same kernel."""
        for key, client in list(_QDRANT_SINGLETONS.items()):
            try:
                client.close()
            except Exception:
                pass
        _QDRANT_SINGLETONS.clear()

    def collection_stats(self) -> None:
        """Print document counts for all collections."""
        print(f"{'Collection':<22}  {'Points':>8}")
        print("-" * 34)
        for name in COLLECTION_NAMES.values():
            try:
                info  = self._client.get_collection(name)
                count = info.points_count
            except Exception:
                count = "N/A"
            print(f"{name:<22}  {count!s:>8}")

    def format_results(self, results: List[Dict], max_text: int = 120) -> str:
        """Return a formatted string of results suitable for injecting into a prompt."""
        lines = []
        for i, r in enumerate(results, 1):
            label   = f" | {r['label']}"      if r.get("label")   else ""
            rca_id  = f" | {r['rca_id']}"     if r.get("rca_id")  else ""
            section = f" | {r['rca_section']}" if r.get("rca_section") else ""
            header  = f"[{i}] {r.get('category','').upper()}{label}{rca_id}{section} (score={r['score']:.3f})"
            text    = r["text"][:max_text] + ("..." if len(r["text"]) > max_text else "")
            lines.append(f"{header}\n{text}")
        return "\n\n".join(lines)
