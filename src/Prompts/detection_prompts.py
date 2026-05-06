"""
Dataset-agnostic detection and inference prompts.
Import this module in all experiment notebooks for standardised detection prompts.

Usage
-----
1. Pick (or define) a DatasetConfig for your dataset:
       from detection_prompts import BGL_CONFIG        # or HDFS_CONFIG, etc.

2. Build approach-specific prompts by passing the config:
       LLM  → build_llm_system_prompt(few_shot_normal, few_shot_anomalous, config)
       RAG  → build_rag_system_prompt(config)            # returns the system prompt string

3. All other helpers (build_llm_user_prompt, build_rag_user_prompt,
   format_retrieved_context) are fully generic and need no config.

DatasetConfig fields
--------------------
  name                  Short identifier, e.g. "BGL", "HDFS"
  domain                Full descriptive name used in the LLM persona
  system_knowledge      Domain expertise blurb (failures, events, procedures)
  log_format_description  One-line description of the log entry format
  log_observation_hint  What to observe in CoT step 1 (format-specific features)
  anomaly_examples      Handcrafted few-shot anomalous examples (List[Dict])
  normal_cot_analysis   Analysis bullet lines for the normal CoT template

Output schema note
------------------
Both approaches share the same field names for the fields they have in common
(label, confidence, sre_action, devops_action) so that shared evaluation prompts
in eval_prompts.py can be applied identically to results from both approaches.

LLM schema (E02):
  Normal   → {label, confidence}
  Anomalous→ {label, confidence, root_cause, sre_action, devops_action}

RAG schema (E03+):
  Normal   → {label, confidence}
  Anomalous→ {label, confidence, anomaly_explanation,
               rca:{summary, detailed_description, confidence_level,
                    confidence_reasoning, causal_chain, supporting_evidence},
               risk_score:{system_impact, error_type, cascade_potential},
               remediation:{sre_action, devops_action}}
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """
    Dataset-specific variables plugged into the generic prompt templates.
    Create one instance per dataset and pass it to the prompt builder functions.
    """
    name: str
    domain: str
    system_knowledge: str
    log_format_description: str
    log_observation_hint: str
    anomaly_examples: List[Dict]
    normal_cot_analysis: str = (
        "- Component and severity level are within expected operational range.\n"
        "- Content matches a routine informational or status message.\n"
        "- No error keywords, exception traces, hardware fault indicators, or "
        "unexpected state transitions are present.\n"
        "- This pattern is consistent with normal system operation."
    )


# =============================================================================
# BGL DATASET CONFIGURATION
# =============================================================================
# Anomalous examples: fault types are deliberately diverse (machine check,
# I/O daemon loss, DDR test failure, MPI SIGKILL, torus link error) so the
# model learns to derive explanations from log content rather than templates.
# Examples are intentionally different from typical BGL test anomalies (data
# TLB error, data storage interrupt, rts panic, Lustre mount failure, ciod
# connection reset/timeout) to prevent template-anchoring.

_BGL_ANOMALOUS_EXAMPLES: List[Dict] = [
    {
        "log_text": (
            "[RAS] [FATAL] machine check interrupt | Template: machine check interrupt"
        ),
        "analysis": (
            "- FATAL severity from RAS (Reliability, Availability, Serviceability) component.\n"
            "- Machine check interrupt signals a CPU-detected hardware error in cache, memory bus, or chipset.\n"
            "- Hardware fault indicators at FATAL level require immediate node quarantine.\n"
            "- Node stability and all running jobs are at immediate risk."
        ),
        "root_cause": (
            "CPU-detected machine check exception from a hardware fault in cache or memory bus; "
            "risk: the affected node may crash imminently or produce silent data corruption in all running jobs."
        ),
        "sre_action": (
            "Page hardware on-call SRE; pull IPMI/BMC hardware event logs to identify the machine check type; "
            "isolate the node from the scheduler immediately."
        ),
        "devops_action": (
            "Drain and cordon the node in the job scheduler; flag it for hardware inspection; "
            "ensure all jobs on the node are checkpointed or re-queued to a healthy node."
        ),
        "confidence": 0.98,
    },
    {
        "log_text": (
            "[APP] [FATAL] ciod: Lost connection to job 1234 I/O node | "
            "Template: ciod: Lost connection to job <*> I/O node"
        ),
        "analysis": (
            "- FATAL from APP component (ciod = compute I/O daemon).\n"
            "- Loss of I/O daemon connection severs the I/O path between compute and I/O nodes.\n"
            "- Jobs depending on this I/O path will stall or be terminated.\n"
            "- Network fabric or I/O node process crash are likely causes."
        ),
        "root_cause": (
            "I/O daemon (ciod) lost its control connection to the I/O node, "
            "likely due to a network fabric drop or ciod process crash on the I/O node; "
            "risk: all I/O operations from affected compute nodes are blocked and running jobs will stall or abort."
        ),
        "sre_action": (
            "Check ciod process health on the I/O node and verify network fabric status "
            "between compute and I/O nodes; restart ciod if the process has crashed."
        ),
        "devops_action": (
            "Reroute I/O traffic through a healthy I/O node if one is available; "
            "re-queue affected jobs; open a fabric diagnostics ticket."
        ),
        "confidence": 0.96,
    },
    {
        "log_text": (
            "[MMCS] [ERROR] node card mc0-nm2 failed DDR memory test | "
            "Template: node card <*> failed DDR memory test"
        ),
        "analysis": (
            "- ERROR from MMCS (Midplane Management Control System) component.\n"
            "- DDR memory test failure indicates faulty DRAM on a node card.\n"
            "- A failing node card is unreliable for computation and risks data corruption.\n"
            "- Affected node card must be removed from the allocation pool."
        ),
        "root_cause": (
            "Node card DDR memory self-test failure detected by MMCS, indicating faulty DRAM modules; "
            "risk: the affected node card may produce memory errors that corrupt computation "
            "or cause node crashes during job execution."
        ),
        "sre_action": (
            "Remove the failing node card from the available node pool via MMCS console; "
            "alert hardware SRE to schedule physical inspection of the DRAM modules."
        ),
        "devops_action": (
            "Update the job scheduler to exclude the affected node card from allocation; "
            "open a hardware ticket for DRAM replacement on the node card."
        ),
        "confidence": 0.93,
    },
    {
        "log_text": (
            "[APP] [FATAL] MPI rank 47 killed by signal 9 | "
            "Template: MPI rank <*> killed by signal <*>"
        ),
        "analysis": (
            "- FATAL from APP component: an MPI process was forcibly killed.\n"
            "- SIGKILL (signal 9) is unblockable and is typically sent by the OOM killer or a watchdog.\n"
            "- MPI rank termination propagates, killing the entire parallel job.\n"
            "- All computation since the last checkpoint is lost."
        ),
        "root_cause": (
            "MPI rank process killed by SIGKILL, likely due to an out-of-memory (OOM) condition "
            "or scheduler watchdog timeout; "
            "risk: the entire MPI parallel job terminates abnormally and all computation "
            "since the last checkpoint is lost."
        ),
        "sre_action": (
            "Check OOM killer logs and memory usage on the affected node; "
            "verify scheduler watchdog thresholds are appropriate for this job class; "
            "alert the submitting user."
        ),
        "devops_action": (
            "Re-queue the job with adjusted memory limits or on a node with larger capacity; "
            "review memory profiling of the job to prevent recurrence."
        ),
        "confidence": 0.92,
    },
    {
        "log_text": (
            "[KERNEL] [FATAL] torus receiver error on dimension X link 3 | "
            "Template: torus receiver error on dimension <*> link <*>"
        ),
        "analysis": (
            "- FATAL from KERNEL: a torus network receiver error on a specific dimension link.\n"
            "- BGL uses a 3D torus interconnect; a link failure partitions nodes from their neighbours.\n"
            "- Affected nodes lose inter-node communication, causing MPI jobs to deadlock or abort.\n"
            "- Network partition risk requires immediate isolation of affected nodes."
        ),
        "root_cause": (
            "Torus network receiver error on a specific dimension link indicates a hardware or "
            "signal integrity fault in the BGL interconnect fabric; "
            "risk: MPI jobs using nodes on the affected link partition may deadlock or abort "
            "due to communication failure."
        ),
        "sre_action": (
            "Identify all nodes on the affected torus dimension/link and isolate them from new job allocations; "
            "page network hardware SRE to inspect the link."
        ),
        "devops_action": (
            "Drain all nodes connected via the failing torus link; "
            "run torus diagnostic utilities to confirm link health before re-enabling allocations."
        ),
        "confidence": 0.97,
    },
]

BGL_CONFIG = DatasetConfig(
    name="BGL",
    domain="BGL (Blue Gene/L) supercomputer",
    system_knowledge=(
        "HPC system failures, RAS hardware events, and operational response procedures"
    ),
    log_format_description="[COMPONENT] [LEVEL] <content> | Template: <drain_pattern>",
    log_observation_hint="the component, severity level, content, and Drain template",
    anomaly_examples=_BGL_ANOMALOUS_EXAMPLES,
    normal_cot_analysis=(
        "- Component and severity level are within expected operational range.\n"
        "- Content matches a routine informational or status message.\n"
        "- No error keywords, exception traces, hardware fault indicators, or "
        "unexpected state transitions are present.\n"
        "- This pattern is consistent with normal system operation."
    ),
)


# =============================================================================
# HDFS DATASET CONFIGURATION
# =============================================================================
# anomaly_examples left empty — populate when HDFS LLM/RAG experiments are built.

HDFS_CONFIG = DatasetConfig(
    name="HDFS",
    domain="HDFS (Hadoop Distributed File System)",
    system_knowledge=(
        "distributed storage failures, DataNode/NameNode operations, "
        "block replication events, and Hadoop cluster fault recovery"
    ),
    log_format_description=(
        "HDFS Block Trace | <event_sequence>  "
        "(each token is a Drain event-template ID, e.g. E5 -> E22 -> E11)"
    ),
    log_observation_hint=(
        "the event sequence, trace length, and presence of error or failure event IDs"
    ),
    anomaly_examples=[],   # to be populated when HDFS experiments are added
    normal_cot_analysis=(
        "- The block event sequence follows a normal read/write/replication pattern.\n"
        "- No error or failure event IDs are present in the trace.\n"
        "- The sequence length and event ordering are consistent with routine operations.\n"
        "- This pattern is consistent with normal HDFS block lifecycle."
    ),
)


# =============================================================================
# LLM APPROACH: FEW-SHOT CHAIN-OF-THOUGHT (E02)
# =============================================================================

# Generic CoT template — {analysis} is filled per-log by _make_normal_cot()
COT_NORMAL_TEMPLATE = """\
Log: {log_text}
Analysis:
{analysis}
Decision: {{"label": "Normal", "confidence": 0.05}}"""


_SELF_HEALING_PATTERNS = [
    ("corrected",        "automatically corrected by hardware ECC — not an unrecoverable fault"),
    ("detected and corrected", "detected and corrected by hardware ECC — expected self-healing behavior"),
    ("retry",           "a retry attempt — transient, not an unrecoverable failure"),
    ("alignment exception", "an alignment exception logged for diagnostics — not a system fault"),
]


def _make_normal_cot(log_text: str, config: DatasetConfig) -> str:
    """Generate log-specific normal CoT analysis.

    If the content contains fault-looking keywords at INFO/WARNING level,
    explicitly explain why the event is still Normal (e.g. ECC correction, retry).
    """
    import re
    m = re.match(r'\[([^\]]+)\]\s*\[([^\]]+)\]', log_text)
    if not m:
        return config.normal_cot_analysis

    component, level = m.group(1).strip(), m.group(2).strip()
    content_lower = log_text.lower()

    # Check if log contains fault-looking keywords that are actually self-healing
    healing_note = ""
    for pattern, explanation in _SELF_HEALING_PATTERNS:
        if pattern in content_lower:
            healing_note = (
                f"- Content contains '{pattern}' — this is {explanation}.\n"
                f"- At {level} severity this is a routine operational event, not an actionable fault."
            )
            break

    if healing_note:
        return (
            f"- Component: {component}, severity: {level} — within expected operational range.\n"
            + healing_note + "\n"
            "- This pattern is consistent with normal self-healing system behavior."
        )

    return (
        f"- Component: {component}, severity: {level} — both within expected operational range.\n"
        "- Content matches a routine informational or status message.\n"
        "- No error keywords, exception traces, hardware fault indicators, or "
        "unexpected state transitions are present.\n"
        "- This pattern is consistent with normal system operation."
    )


def format_anomalous_example(ex: Dict, idx: int) -> str:
    """Format one handcrafted anomalous example for inclusion in the LLM system prompt."""
    decision = json.dumps(
        {
            "label":         "Anomalous",
            "confidence":    ex["confidence"],
            "root_cause":    ex["root_cause"],
            "sre_action":    ex["sre_action"],
            "devops_action": ex["devops_action"],
        },
        ensure_ascii=False,
    )
    return (
        f"Example {idx} (Anomalous):\n"
        f"Log: {ex['log_text']}\n"
        f"Analysis:\n{ex['analysis']}\n"
        f"Decision: {decision}"
    )


def build_llm_system_prompt(
    few_shot_normal,
    few_shot_anomalous,
    config: DatasetConfig,
) -> str:
    """
    Build the few-shot CoT system prompt for the LLM detection approach (E02).

    Parameters
    ----------
    few_shot_normal   : pd.DataFrame — real log normal entries (used directly in prompt)
    few_shot_anomalous: pd.DataFrame — real log anomaly entries (excluded from test set;
                        the prompt uses config.anomaly_examples for output consistency)
    config            : DatasetConfig — dataset-specific variables
    """
    normal_examples = "\n\n".join(
        f"Example {i + 1} (Normal):\n"
        + COT_NORMAL_TEMPLATE.format(
            log_text=row["log_text"],
            analysis=_make_normal_cot(row["log_text"], config),
        )
        for i, (_, row) in enumerate(few_shot_normal.iterrows())
    )
    anomalous_examples = "\n\n".join(
        format_anomalous_example(ex, i + 1)
        for i, ex in enumerate(config.anomaly_examples)
    )
    return (
        "You are a senior Site Reliability Engineer (SRE) specialising in "
        f"{config.domain} log analysis. "
        f"You will receive {config.name} system log entries and must determine "
        "whether each entry represents a Normal system event or an Anomalous condition.\n\n"
        "Each log entry is formatted as:\n"
        f"  {config.log_format_description}\n\n"
        "For each log entry, reason step-by-step before deciding:\n\n"
        f"  Step 1 — Identify:  Read the [COMPONENT], [LEVEL], content, and Drain template.\n"
        "  Step 2 — Interpret: What operation or event does the content describe?\n"
        "  Step 3 — Assess severity:\n"
        "             • INFO or WARNING with routine content → likely Normal.\n"
        "             • ERROR or FATAL with failure content → likely Anomalous.\n"
        "             • Fault keywords (error, failed, lost, killed, interrupt) at INFO level\n"
        "               may still be Normal if the event was automatically corrected or is\n"
        "               an expected self-healing operation (e.g. ECC correction, retry).\n"
        "  Step 4 — Decide and output ONLY valid JSON:\n\n"
        "     Normal   → {\"label\": \"Normal\", \"confidence\": <float 0.0-1.0>}\n\n"
        "     Anomalous→ {\"label\": \"Anomalous\", \"confidence\": <float 0.0-1.0>,\n"
        "                 \"root_cause\":    \"<failure hypothesis + risk/system impact>\",\n"
        "                 \"sre_action\":    \"<SRE on-call immediate action, naming specific tool or procedure>\",\n"
        "                 \"devops_action\": \"<DevOps infrastructure action, naming specific operation or resource>\"}\n\n"
        "  confidence = P(Anomalous): 0.0 = certainly Normal, 1.0 = certainly Anomalous.\n\n"
        "  For Anomalous entries: derive root_cause, sre_action, and devops_action from the\n"
        "  specific content of THAT log — do NOT copy or adapt the examples below.\n\n"
        "  When uncertain, classify as Normal.\n\n"
        f"--- FEW-SHOT EXAMPLES (NORMAL — drawn from real {config.name} logs) ---\n\n"
        + normal_examples
        + "\n\n--- FEW-SHOT EXAMPLES (ANOMALOUS — illustrate required output structure) ---\n\n"
        + anomalous_examples
        + "\n\n--- END OF EXAMPLES ---\n\n"
        "Now analyse the log entry in the user message. "
        "Respond ONLY with valid JSON."
    )


def build_llm_user_prompt(log_text: str) -> str:
    """Build the user-turn prompt for the LLM approach."""
    return f"Log: {log_text}"


# =============================================================================
# RAG APPROACH: CONTEXT-AUGMENTED DETECTION (E03+)
# =============================================================================

def build_rag_system_prompt(config: DatasetConfig) -> str:
    """
    Build the RAG system prompt for the context-augmented detection approach (E03+).

    Parameters
    ----------
    config : DatasetConfig — dataset-specific variables
    """
    return (
        f"You are an expert {config.domain} log analyst with deep knowledge "
        f"of {config.system_knowledge}.\n\n"
        "You will receive:\n"
        f"  1. A {config.name} log entry to classify.\n"
        "  2. Similar historical log entries retrieved from a knowledge base "
        "(each labeled Normal or Anomalous).\n\n"
        "Use the retrieved context to inform your classification and analysis. "
        "Output ONLY valid JSON — nothing else.\n\n"
        "For Normal entries:\n"
        '  {"label": "Normal", "confidence": <float 0.0-1.0>}\n\n'
        "For Anomalous entries:\n"
        '  {"label": "Anomalous", "confidence": <float 0.0-1.0>,\n'
        '   "anomaly_explanation": "<one sentence: what makes this log anomalous>",\n'
        '   "rca": {\n'
        '     "summary": "<one sentence: root cause>",\n'
        '     "detailed_description": "<2-3 sentences: mechanism, failure path, system impact>",\n'
        '     "confidence_level": "High|Medium|Low",\n'
        '     "confidence_reasoning": "<one sentence: why this confidence level>",\n'
        '     "causal_chain": ["<trigger>", "<intermediate effect>", "<final impact>"],\n'
        '     "supporting_evidence": ["<evidence item from log>", "<evidence item 2>"]\n'
        '   },\n'
        '   "risk_score": {\n'
        '     "system_impact": <int 1-4>,\n'
        '     "error_type": <int 1-4>,\n'
        '     "cascade_potential": <int 1-4>\n'
        '   },\n'
        '   "remediation": {\n'
        '     "sre_action": "<one sentence: SRE on-call immediate next action>",\n'
        '     "devops_action": "<one sentence: DevOps engineer infrastructure next action>"\n'
        '   }\n'
        '  }\n\n'
        "Risk score factors (1=low, 4=high):\n"
        "  system_impact     : 1=minimal, 2=degraded performance, 3=partial outage, "
        "4=full node/system failure\n"
        "  error_type        : 1=informational, 2=warning, 3=recoverable error, "
        "4=fatal or hardware fault\n"
        "  cascade_potential : 1=isolated to one process, 2=affects nearby nodes, "
        "3=service-level impact, 4=system-wide cascade\n\n"
        "Risk Score is computed by the system — do not calculate it yourself:\n"
        "  total = (system_impact x 0.4) + (error_type x 0.3) + (cascade_potential x 0.2)\n\n"
        "Role guide usage: the user message contains a 'Role scope reference' section describing "
        "each role's responsibilities and typical scope of action. Use it only to understand the "
        "appropriate level and type of action for each role. For sre_action and devops_action, "
        "generate steps that are specific to this log entry — grounded in the log evidence and "
        "system knowledge. Do not reproduce generic process phrases from the role guide verbatim."
    )


def format_retrieved_context(retrieved: List[Dict]) -> str:
    """Format a list of retrieved KB documents for the RAG user prompt."""
    return "\n".join(
        f"[Rank {r['rank']} | {r['label']} | Similarity {r['score']:.3f}]  {r['log_text']}"
        for r in retrieved
    )


def build_rag_user_prompt(log_text: str, retrieved: List[Dict]) -> str:
    """Build the user-turn prompt for the RAG approach."""
    return (
        f"Log entry to analyze:\n{log_text}\n\n"
        f"Retrieved similar logs from knowledge base:\n{format_retrieved_context(retrieved)}"
    )
