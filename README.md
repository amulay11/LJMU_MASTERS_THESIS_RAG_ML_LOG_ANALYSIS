# LogSense — Log Anomaly Detection with ML and RAG

LJMU Masters Thesis: Evaluating machine learning and retrieval-augmented generation approaches for system log anomaly detection, root cause analysis, and role-based remediation guidance.

---

## Datasets Overview

Two publicly available system log datasets are used. Both are parsed using the **Drain** log template miner, which extracts structured event templates from raw log text.

### BGL (Blue Gene/L Supercomputer Logs)

| Property | Value |
|---|---|
| Source | IBM Blue Gene/L supercomputer (Argonne National Laboratory) |
| Log entries | 2,000 (sample) |
| Normal | 1,857 (92.9%) |
| Anomalous | 143 (7.1%) |
| Event templates (Drain) | 120 |
| Log format | `[COMPONENT] [LEVEL] <content> \| Template: <drain_pattern>` |
| Label column | `Label` — normal entries are marked `-`; anomalous entries carry a fault category |

BGL logs originate from the RAS (Reliability, Availability, Serviceability) subsystem of a 65,536-node supercomputer. Log levels include `INFO`, `WARNING`, `ERROR`, and `FATAL`. Anomalies represent hardware faults (machine check interrupts, DDR memory failures, torus link errors), software failures (MPI SIGKILL, ciod I/O daemon loss), and kernel panics.

### HDFS (Hadoop Distributed File System Logs)

| Property | Value |
|---|---|
| Source | HDFS cluster at University of Illinois Urbana-Champaign |
| Block traces (full dataset) | 575,061 |
| Normal | 558,223 (97.1%) |
| Anomalous | 16,838 (2.9%) |
| Event templates (Drain) | 29 |
| Log format | `HDFS Block Trace \| <event_sequence>` (sequence of Drain event IDs, e.g. `E5 -> E22 -> E11`) |
| Label column | `Label` — normal entries are `Success`; anomalous entries are `Failure` |

HDFS logs are grouped into per-block traces — each trace is a sequence of event IDs representing the full lifecycle of one HDFS block (allocation, replication, deletion). Anomalies correspond to blocks that failed replication or triggered DataNode errors.

---

## Knowledge Base Overview and Strategy

The knowledge base (KB) is a persistent **Qdrant** vector store (`src/KnowledgeBase/qdrant_store/`) built by `src/KnowledgeBase/build_kb.py`. It holds ten collections, keeping datasets and knowledge types strictly separated so retrieval never crosses dataset or role boundaries.

### Collections

| Collection | Content | Dataset |
|---|---|---|
| `bgl_logs` | BGL log samples (labeled Normal / Anomalous) | BGL |
| `bgl_architecture` | BlueGene/L system architecture reference | BGL |
| `bgl_severity` | BGL RAS severity and error taxonomy | BGL |
| `bgl_rca` | BGL root cause analysis examples | BGL |
| `hdfs_logs` | HDFS block trace samples (labeled Normal / Anomalous) | HDFS |
| `hdfs_architecture` | HDFS system architecture reference | HDFS |
| `hdfs_severity` | HDFS severity and error taxonomy | HDFS |
| `hdfs_rca` | HDFS root cause analysis examples | HDFS |
| `role_sre` | SRE on-call reference guide | Shared |
| `role_devops` | DevOps engineer reference guide | Shared |

### KB Population Strategy

**Test-set isolation:** The KB excludes all logs in the test set (200 normal + 15 anomalous for BGL) to prevent retrieval leakage. The same random seed and split logic is applied in both the KB builder and the experiment notebooks, ensuring strict disjointness.

**BGL log optimisation (two steps):**
1. *Template-level deduplication* — rows are grouped by Drain `EventTemplate` and one representative is kept per unique template. This eliminates near-duplicate embedding vectors and reduces retrieval noise.
2. *Class balancing* — normals are capped at 2× the anomalous count so anomalous signal is not diluted during nearest-neighbour retrieval.

**HDFS log optimisation:** HDFS traces are unique event sequences with no shared template structure, so template deduplication does not apply. A random cap is used instead: 200 normal + 100 anomalous traces, keeping the KB manageable without sacrificing diversity.

**Source documents per collection:**

| Content type | Source format | Example source files |
|---|---|---|
| Log samples | CSV (parsed by Drain) | `BGL_2k.log_structured.csv`, `Event_traces.csv` |
| Architecture | Word document (.docx) | `BlueGeneL_SystemArchitecture.docx`, `HDFS_System_Architecture_Reference.docx` |
| Severity taxonomy | Word document (.docx) | `BGL_RAS_Severity_Taxonomy.docx`, `HDFS_Severity_Error_Taxonomy.docx` |
| RCA examples | JSON | `bgl_rca.json`, `hdfs_rca.json` |
| Role guides | Word document (.docx) | `SRE_Reference_Guide_RAG.docx`, `DevOps_Reference_Guide_RAG.docx` |

---

## Data Chunking, Embedding and Storage

### Chunking Strategy

**Log samples** are stored one entry per vector. Each entry is a formatted string:
```
[COMPONENT] [LEVEL] <content> | Template: <drain_pattern>         (BGL)
HDFS Block Trace | <event_id_1> -> <event_id_2> -> ...             (HDFS)
```

**Word documents (.docx)** are chunked using a section-aware strategy (`extract_docx_chunks` in `build_kb.py`):
- Heading paragraphs mark section boundaries; the section title is prepended to each chunk as `[Section Title] <text>`.
- Body paragraphs are accumulated into chunks up to **600 characters**.
- Paragraphs exceeding 900 characters are split on sentence boundaries before accumulation.

**RCA JSON entries** are expanded into three documents per entry (`rca_to_documents`):
| Sub-document | Content |
|---|---|
| `overview` | Anomaly type, root cause summary, confidence level and reasoning |
| `analysis` | Detailed description and full causal chain (timestamped steps) |
| `evidence` | Supporting log samples, matched templates, system context, historical similarity, state indicators |

### Embedding

| Property | Value |
|---|---|
| Model | `BAAI/bge-base-en-v1.5` |
| Embedding dimension | 768 |
| Normalisation | L2-normalised (cosine similarity via dot product) |
| Query prefix | `"Represent this sentence for searching relevant passages: <query>"` (BGE instruction prefix) |
| Batch size | 64 |

All vectors are L2-normalised at index time. Queries use the BGE instruction prefix to improve retrieval quality over plain embedding.

### Storage in Vector Store

| Property | Value |
|---|---|
| Vector store | Qdrant (local persistent mode) |
| Storage path | `src/KnowledgeBase/qdrant_store/` |
| Distance metric | Cosine |
| Collections | 10 (separated by dataset and knowledge type) |
| Access | `KBClient` singleton (`src/KnowledgeBase/kb_utils.py`) |

Each point in the vector store carries a rich payload for filtered retrieval:

```
text, dataset, category, source, label, rca_id, rca_section,
anomaly_type, section, log_component, log_level
```

The `KBClient` class (`kb_utils.py`) provides typed query methods — `query_bgl_logs()`, `query_bgl_architecture()`, `query_bgl_rca()`, `query_role()`, `query_multi()` — and a module-level singleton to prevent Qdrant file lock conflicts when notebook cells are re-run.

---

## Experiments for Log Analysis using ML and RAG

All experiments operate on the **BGL dataset** using the same held-out test set (200 normal + 15 anomalous logs) to ensure comparability across approaches. ML models use a one-class unsupervised setting; LLM and RAG experiments use `llama-3.1-8b-instant` for generation and `qwen/qwen3-32b` as the LLM-as-judge evaluator.

| # | Title (Model / Approach) | Short Description | Notebook |
|---|---|---|---|
| E01-A | Deep SVDD — BGL | One-class neural network that learns a compact hypersphere around normal log embeddings; anomalies are points that fall outside the boundary | [E01_A_BGL_DeepSVDD_Pipeline.ipynb](src/ML%20Based/E01_A_BGL_DeepSVDD_Pipeline.ipynb) |
| E01-B | Isolation Forest — BGL | Unsupervised ensemble that isolates anomalies by randomly partitioning feature space; anomalies require fewer splits to isolate than normal points | [E01_B_BGL_IsolationForest_Pipeline.ipynb](src/ML%20Based/E01_B_BGL_IsolationForest_Pipeline.ipynb) |
| E01-C | One-Class SVM — BGL | Kernel-based decision boundary fitted to the normal log distribution in TF-IDF feature space; points outside the boundary are flagged as anomalous | [E01_C_BGL_OneClassSVM_Pipeline.ipynb](src/ML%20Based/E01_C_BGL_OneClassSVM_Pipeline.ipynb) |
| E02 | LLM Baseline (Few-Shot CoT) | Few-shot Chain-of-Thought prompting with no retrieval; 5 real normal + 5 handcrafted anomalous in-context examples guide the model's reasoning and structured JSON output | [E02_BGL_LLM_Baseline.ipynb](src/LLM/E02_BGL_LLM_Baseline.ipynb) |
| E03 | Vanilla RAG | Standard dense retrieval from Qdrant across five collections (log examples, architecture, severity, RCA, role guides) injected as a structured multi-section prompt; serves as the RAG baseline | [E03_BGL_RAG_Baseline.ipynb](src/RAG/Vanilla%20RAG/E03_BGL_RAG_Baseline.ipynb) |
| E04 | DeepSVDD-Guided RAG | Two-stage pipeline: DeepSVDD assigns an anomaly distance score per log; the score is used to route inference between a standard RAG path and an anomaly-focused retrieval path | [E04_DeepSVDD_RAG_Pipeline.ipynb](src/RAG/Vanilla%20RAG/E04_DeepSVDD_RAG_Pipeline.ipynb) |
| E05 | Hybrid RAG (BM25 + Dense) | Combines BM25 sparse keyword retrieval with Qdrant dense retrieval; both ranked lists are merged via Reciprocal Rank Fusion (RRF, k=60) before the top-k candidates are passed to the LLM | [E05_BGL_DIRECT_HYBRID_RAG.ipynb](src/RAG/Hybrid%20RAG/E05_BGL_DIRECT_HYBRID_RAG.ipynb) |
| E06 | Self-Reflective RAG | Extends Vanilla RAG with an iterative self-critique loop: if model confidence falls below 0.75 the model receives its own initial response and is prompted to revise it (up to 2 reflection rounds) | [E06_BGL_DIRECT_SELF_REFLECTIVE_RAG.ipynb](src/RAG/Self%20Reflective%20RAG/E06_BGL_DIRECT_SELF_REFLECTIVE_RAG.ipynb) |
| E07 | Graph RAG | Builds a NetworkX knowledge graph over the KB corpus using component/level co-occurrence; 1-hop graph traversal expands the query with related templates before merging with dense retrieval results | [E07_BGL_DIRECT_GRAPH_RAG.ipynb](src/RAG/Graph%20RAG/E07_BGL_DIRECT_GRAPH_RAG.ipynb) |
| E08 | Temporal RAG | Adds time as a retrieval dimension: dense candidates are reranked by a combined score (`0.7 × semantic + 0.3 × exp(−\|Δt\|/86400)`); retrieved logs are presented to the LLM chronologically to surface evolving error patterns | [E08_BGL_DIRECT_TEMPORAL_RAG.ipynb](src/RAG/Temporal%20RAG/E08_BGL_DIRECT_TEMPORAL_RAG.ipynb) |
| E09 | DeepSVDD + Hybrid RAG | Combines Deep SVDD routing (Stage A/B/C) with BM25+Dense hybrid retrieval (RRF fusion, k=60); uncertain and anomalous logs are enriched with context retrieved from both sparse and dense indexes | [E09_DeepSVDD_Hybrid_RAG_Pipeline.ipynb](src/RAG/Hybrid%20RAG/E09_DeepSVDD_Hybrid_RAG_Pipeline.ipynb) |

---

## LLMs and Prompting Strategy

### Models

| Role | Model | Provider |
|---|---|---|
| Detection / generation | `llama-3.3-70b-versatile` | Groq |
| LLM-as-judge (evaluation) | `qwen/qwen3-32b` | Groq |

Multiple Groq API keys are used in a parallel worker pool (up to 5 primary + 5 fallback) with per-thread key assignment and automatic rotation on rate-limit errors.

### Prompting — LLM Baseline (E02)

The LLM baseline uses **few-shot Chain-of-Thought (CoT)** prompting. The system prompt embeds:

- A senior SRE persona with domain expertise in the target system.
- A four-step CoT reasoning instruction: (1) identify component and level, (2) interpret the event, (3) assess severity, (4) decide and output JSON.
- **5 real Normal examples** drawn from the actual log dataset (excluded from the test set), each with a log-specific CoT analysis.
- **5 handcrafted Anomalous examples** covering diverse fault types (machine check interrupt, ciod I/O loss, DDR memory failure, MPI SIGKILL, torus link error) — deliberately different from the test anomaly types to prevent template anchoring.

Output schema for anomalies:
```json
{
  "label": "Anomalous", "confidence": 0.95,
  "root_cause": "<failure hypothesis + system risk>",
  "sre_action": "<SRE on-call immediate action>",
  "devops_action": "<DevOps infrastructure action>"
}
```

### Prompting — Vanilla RAG (E03)

The RAG approach replaces the in-prompt few-shot examples with **retrieved context from the persistent Qdrant KB**. For each test log, three separate retrieval queries are issued and injected into the user-turn prompt in clearly labelled sections.

#### Retrieval strategy

Three independent Qdrant queries are issued per log entry, each targeting a different knowledge type:

| Section | Collections queried | K per collection | Purpose |
|---|---|---|---|
| `[A]` Log examples | `bgl_logs` | 5 | Nearest-neighbour log hits with ground-truth labels — used for label-based retrieval metrics (MRR, Hit Rate, Context Precision) |
| `[B]` System knowledge | `bgl_architecture`, `bgl_severity`, `bgl_rca` | 2 | Architecture reference, severity taxonomy, and RCA examples — provides domain-specific evidence for RCA and risk scoring |
| `[C]` Role scope reference | `role_sre`, `role_devops` | 2 | SRE and DevOps reference guides — used **only** to understand each role's scope of action; model is explicitly instructed not to copy verbatim |

The user-turn prompt is structured as three labelled blocks (A / B / C) so the model can distinguish log evidence from role process knowledge. Section `[C]` carries the explicit label: *"do not copy verbatim — derive specific actions from the log evidence above"*.

#### Anti-parroting constraint

Role guide content (e.g. "Review and prune alerts quarterly", "Deploy a canary to 1% of nodes") is generic process knowledge, not log-specific guidance. When role chunks are mixed into a single undifferentiated context block, the model tends to reproduce guide phrases verbatim rather than grounding remediation steps in the observed log evidence. Two complementary mitigations are applied:

1. **Structural separation** — role guide hits are placed in a distinct `[C]` section with an explicit "do not copy verbatim" label, keeping them visually and semantically separate from log evidence (`[A]`) and system knowledge (`[B]`).
2. **System prompt constraint** — `build_rag_system_prompt()` in `src/Prompts/detection_prompts.py` includes an explicit instruction: *"Use the Role scope reference only to understand the appropriate level and type of action for each role. Generate steps that are specific to this log entry — grounded in the log evidence and system knowledge. Do not reproduce generic process phrases from the role guide verbatim."*

The system prompt instructs the model to use retrieved context to inform classification and produce a richer structured output:

```json
{
  "label": "Anomalous", "confidence": 0.95,
  "anomaly_explanation": "...",
  "rca": {
    "summary": "...", "detailed_description": "...",
    "confidence_level": "High|Medium|Low", "confidence_reasoning": "...",
    "causal_chain": ["trigger", "effect", "impact"],
    "supporting_evidence": ["evidence 1", "evidence 2"]
  },
  "risk_score": {"system_impact": 3, "error_type": 4, "cascade_potential": 3},
  "remediation": {"sre_action": "...", "devops_action": "..."}
}
```

**Risk score** (`system_impact × 0.4 + error_type × 0.3 + cascade_potential × 0.2`) is computed programmatically from the LLM-returned factors — the model is not trusted to compute the total.

All prompt templates are shared across experiments via `src/Prompts/detection_prompts.py` and `src/Prompts/eval_prompts.py` to ensure identical prompt wording for overlapping metrics.

---

## Evaluation Strategy

All experiments evaluate on the **same held-out test set**: 200 normal + 15 anomalous BGL logs, mirroring the real-world ~7% anomaly rate. The test set is strictly disjoint from both the few-shot examples in the LLM prompt and the persistent KB used for RAG retrieval.

### Standard Anomaly Detection Metrics

Applied to every model and approach on the full test set:

| Metric | Description |
|---|---|
| Accuracy | Overall correct classifications |
| Precision | Of predicted anomalies, fraction that are true anomalies |
| Recall | Of true anomalies, fraction correctly detected |
| F1-Score | Harmonic mean of precision and recall |
| FPR | False positive rate (normal logs misclassified as anomalous) |
| FNR | False negative rate (anomalies missed) |
| AUC-ROC | Area under the ROC curve using model confidence scores |

### RAG-Specific Retrieval Metrics

Computed from retrieval results on the true-anomaly subset (no LLM judge required):

| Metric | Description |
|---|---|
| MRR | Mean Reciprocal Rank of the first correctly-labeled retrieved document |
| Hit Rate @K | Fraction of queries where at least one correctly-labeled document appears in top-K |
| Context Precision | Fraction of retrieved documents that match the true label |

### LLM-as-Judge Metrics (RAGAs and Extended)

**Judge model:** `qwen/qwen3-32b`. Applied to all true-anomaly predictions. Each metric is scored 0–1 via a structured JSON response from the judge.

**RAGAs metrics** (shared across LLM and RAG approaches):

| Metric | What is scored |
|---|---|
| Faithfulness | Is the root cause / RCA grounded only in observable log evidence (no hallucination)? |
| Answer Relevance | Is the explanation specific to this anomaly, not a generic response? |
| Context Precision | Do the retrieved documents match the log's true label? |
| Context Recall | Does the retrieved context contain the information needed to derive the correct RCA? |

**Additional RAG metrics:**

| Metric | What is scored |
|---|---|
| Root Cause Score | Technical quality and specificity of the RCA (summary + causal chain) |
| Evidence Support Score | Are the supporting evidence items traceable to the log content? |
| Severity Agreement | Does the predicted risk level (High/Medium/Low) match the log severity level? |
| Consistency Score | Variance of risk scores across logs sharing the same Drain template |
| Role Appropriateness (SRE) | Is the SRE action actionable and appropriate for on-call response? |
| Role Appropriateness (DevOps) | Is the DevOps action actionable and appropriate for infrastructure response? |
| Completeness Score | Are all 10 required output fields present and non-empty? |

### Efficiency Metrics

Reported for LLM and RAG approaches on the full test set:

| Metric | Description |
|---|---|
| Retrieval latency | Time to embed query and fetch top-K from Qdrant |
| LLM prefill latency | Time to process the prompt (non-generation) |
| Generation latency | Time to generate the response tokens |
| Total latency | Sum of all three; reported as mean, median, p95, p99 |

---

## Repository Structure

```
Experiments/
├── Datasets/
│   ├── BGL/Sample/              # 2k BGL structured logs + templates
│   └── HDFS/                    # HDFS sample + full preprocessed traces
├── KnowledgeBase/
│   ├── BGL/                     # Architecture .docx, severity .docx, RCA .json
│   ├── HDFS/                    # Architecture .docx, severity .docx, RCA .json
│   └── RoleBased Knowledge/     # SRE and DevOps reference guides (.docx)
└── src/
    ├── KnowledgeBase/
    │   ├── build_kb.py          # One-time KB builder (run once before experiments)
    │   └── kb_utils.py          # KBClient — typed query interface for RAG notebooks
    ├── ML Based/                # E01 A/B/C — unsupervised ML anomaly detection
    ├── LLM/                     # E02 — few-shot CoT LLM baseline
    ├── RAG/
    │   ├── Vanilla RAG/         # E03 — RAG baseline; E04 — DeepSVDD-guided RAG
    │   ├── Hybrid RAG/          # E05 — BM25 + Dense with RRF fusion
    │   ├── Self Reflective RAG/ # E06 — iterative self-critique RAG
    │   ├── Graph RAG/           # E07 — knowledge-graph-expanded retrieval
    │   └── Temporal RAG/        # E08 — time-decay reranked retrieval
    └── Prompts/
        ├── detection_prompts.py # Shared LLM and RAG prompt builders
        └── eval_prompts.py      # Shared LLM-as-judge evaluation prompts
```
