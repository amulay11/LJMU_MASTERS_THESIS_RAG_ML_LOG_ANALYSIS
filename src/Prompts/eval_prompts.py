"""
Shared LLM-as-judge evaluation prompts for BGL log anomaly experiments.
Import this module in all experiment notebooks to ensure standardised,
directly comparable evaluation scores across approaches.

Shared across ALL approaches (E02, E03, future):
    JUDGE_SYSTEM_PROMPT
    build_faithfulness_prompt
    build_answer_relevance_prompt
    build_sre_appropriateness_prompt
    build_devops_appropriateness_prompt

LLM approach only (E02 output schema: root_cause, sre_action, devops_action):
    build_completeness_prompt_llm

RAG approach only (E03 output schema: full 10-field nested output):
    build_context_recall_prompt
    build_root_cause_score_prompt
    build_evidence_support_prompt
    build_completeness_prompt_rag
"""

from typing import List, Optional

# =============================================================================
# SHARED: JUDGE SYSTEM PROMPT
# Used as the system prompt for every single judge/evaluation API call.
# =============================================================================

JUDGE_SYSTEM_PROMPT = (
    "You are an objective evaluator of AI-generated log analysis responses. "
    "Score the response on the specified dimension. "
    'Respond ONLY with valid JSON: {"score": <float 0.0-1.0>, "reason": "<one sentence>"}'
)

# =============================================================================
# SHARED: FAITHFULNESS
# Does the explanation accurately reflect only what is observable in the log?
# Used by: E02 (root_cause field), E03 (rca_summary + supporting_evidence)
# =============================================================================

def build_faithfulness_prompt(
    log_text: str,
    rca_or_root_cause: str,
    supporting_evidence: Optional[List[str]] = None,
) -> str:
    """
    Score whether the root cause / RCA is grounded in the log entry (the only context).

    Parameters
    ----------
    log_text           : the BGL log entry being evaluated
    rca_or_root_cause  : E02 → root_cause field;  E03 → rca_summary field
    supporting_evidence: E03 → rca_supporting_evidence list;  E02 → omit (None)
    """
    ev_line = ""
    if supporting_evidence:
        ev_line = f"\nAI supporting evidence: {'; '.join(supporting_evidence)}"
    return (
        "The log entry below is the ONLY available context — there is no additional "
        "documentation, runbook, or external knowledge to draw from.\n\n"
        f"Log entry:\n{log_text}\n\n"
        f"AI root cause / RCA summary: {rca_or_root_cause or '(empty)'}"
        + ev_line + "\n\n"
        "Score FAITHFULNESS (0-1): Based solely on what is directly observable in "
        "the log entry, does the root cause accurately reflect the evidence present? "
        "1.0 = fully grounded in log evidence, "
        "0.0 = hallucinated or contradicts the log."
    )

# =============================================================================
# SHARED: ANSWER RELEVANCE
# Is the explanation specifically relevant to THIS anomaly, not generic?
# Used by: E02 (root_cause), E03 (anomaly_explanation + rca_summary)
# =============================================================================

def build_answer_relevance_prompt(
    log_text: str,
    explanation: str,
    rca_summary: str = "",
) -> str:
    """
    Score whether the explanation is specific to this anomaly vs generic.

    Parameters
    ----------
    log_text   : the BGL log entry being evaluated
    explanation: E02 → root_cause field;  E03 → anomaly_explanation field
    rca_summary: E03 → rca_summary field;  E02 → omit (defaults to empty)
    """
    rca_line = f"\nAI RCA summary: {rca_summary}" if rca_summary else ""
    return (
        f"Log entry:\n{log_text}\n\n"
        f"AI anomaly explanation: {explanation or '(empty)'}"
        + rca_line + "\n\n"
        "Score ANSWER RELEVANCE (0-1): Is the explanation specifically and meaningfully "
        "relevant to the anomaly indicated by this particular log entry? "
        "Penalise generic or template-like explanations that could apply to any fault — "
        "reward explanations that address the specific component, error type, and context "
        "visible in this log. "
        "1.0 = directly and specifically addresses this anomaly, "
        "0.0 = generic, off-topic, or unrelated to the observed fault."
    )

# =============================================================================
# SHARED: ROLE APPROPRIATENESS — SRE
# Is sre_action technically precise and actionable for an SRE on-call?
# Used by: E02, E03 (identical sre_action field in both schemas)
# =============================================================================

def build_sre_appropriateness_prompt(sre_action: str, log_text: str = "") -> str:
    log_line = f"Log entry:\n{log_text}\n\n" if log_text else ""
    return (
        log_line
        + f"SRE next action: {sre_action or '(empty)'}\n\n"
        "Score ROLE APPROPRIATENESS for a Site Reliability Engineer (SRE) (0-1).\n\n"
        "A HIGH-QUALITY SRE action MUST:\n"
        "  • Reference the specific component or failure type visible in the log.\n"
        "  • Name a concrete tool, command, or procedure (e.g. 'check IPMI/BMC logs', "
        "'drain node via scheduler', 'page on-call', 'isolate the affected node').\n"
        "  • Be immediately executable by an on-call SRE — no further interpretation needed.\n\n"
        "PENALISE heavily:\n"
        "  • Generic phrases: 'monitor the system', 'check logs', 'investigate further', "
        "'contact support', 'review the situation', 'notify the team'.\n"
        "  • Actions that do not reference the specific fault or component visible in the log.\n"
        "  • DevOps/infra tasks misassigned to SRE (scheduler reconfiguration, hardware replacement).\n\n"
        "Scoring:\n"
        "  1.0 = component-specific + concrete tool/procedure + immediately actionable\n"
        "  0.7 = correct domain but missing one of: specificity, tool name, or component reference\n"
        "  0.4 = partially relevant but generic or missing key action detail\n"
        "  0.0 = generic template, wrong role, or empty"
    )

# =============================================================================
# SHARED: ROLE APPROPRIATENESS — DevOps
# Is devops_action operationally clear and actionable for a DevOps engineer?
# Used by: E02, E03 (identical devops_action field in both schemas)
# =============================================================================

def build_devops_appropriateness_prompt(devops_action: str, log_text: str = "") -> str:
    log_line = f"Log entry:\n{log_text}\n\n" if log_text else ""
    return (
        log_line
        + f"DevOps next action: {devops_action or '(empty)'}\n\n"
        "Score ROLE APPROPRIATENESS for a DevOps Engineer (0-1).\n\n"
        "A HIGH-QUALITY DevOps action MUST:\n"
        "  • Address infrastructure, scheduler, or deployment-level remediation — NOT monitoring or alerting.\n"
        "  • Reference the specific resource, node, or service impacted (visible in the log).\n"
        "  • Name a concrete operation (e.g. 'drain and cordon node', 'open hardware replacement ticket', "
        "'update scheduler exclusion list', 're-queue affected jobs with adjusted memory limits').\n\n"
        "PENALISE heavily:\n"
        "  • Generic phrases: 'review the logs', 'notify the team', 'monitor infrastructure', "
        "'check the system', 'update documentation', 'investigate further'.\n"
        "  • SRE-level actions (paging on-call, setting alerts, incident management) misassigned to DevOps.\n"
        "  • Actions that do not address the specific failure type visible in the log.\n\n"
        "Scoring:\n"
        "  1.0 = infrastructure-specific + concrete operation + references the impacted resource\n"
        "  0.7 = correct domain but missing one of: specificity, operation name, or resource reference\n"
        "  0.4 = partially relevant but generic or missing key detail\n"
        "  0.0 = generic template, SRE-level action, wrong role, or empty"
    )

# =============================================================================
# LLM APPROACH (E02): COMPLETENESS — 3-field schema
# Checks: root_cause, sre_action, devops_action
# =============================================================================

def build_completeness_prompt_llm(
    root_cause: str,
    sre_action: str,
    devops_action: str,
) -> str:
    """
    Completeness for the LLM (E02) output schema.
    Expected fields: root_cause (with impact), sre_action, devops_action.
    """
    return (
        "Evaluate the COMPLETENESS of the following anomaly explanation.\n\n"
        "Expected components:\n"
        "  1. Root cause hypothesis with risk / system impact\n"
        "  2. SRE-oriented next action for resolution\n"
        "  3. DevOps-oriented next action for resolution\n\n"
        f"Root cause & impact : {root_cause or '(empty)'}\n"
        f"SRE action          : {sre_action or '(empty)'}\n"
        f"DevOps action       : {devops_action or '(empty)'}\n\n"
        "Score COMPLETENESS (0-1): "
        "1.0 = all three components present and substantive, "
        "0.5 = partially complete (some components thin or missing), "
        "0.0 = all components empty or absent."
    )

# =============================================================================
# RAG APPROACH (E03): CONTEXT RECALL
# Does the retrieved KB context contain enough to derive the RCA?
# =============================================================================

def build_context_recall_prompt(
    log_text: str,
    retrieved_context: str,
    rca_summary: str,
) -> str:
    return (
        f"Log entry:\n{log_text}\n\n"
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"AI RCA summary: {rca_summary or '(empty)'}\n\n"
        "Score CONTEXT RECALL (0-1): Does the retrieved context contain sufficient "
        "information to derive the RCA? "
        "1.0 = context fully supports and explains the RCA, "
        "0.0 = context is irrelevant or insufficient."
    )

# =============================================================================
# RAG APPROACH (E03): ROOT CAUSE QUALITY SCORE
# Is the RCA technically sound with a plausible causal chain?
# =============================================================================

def build_root_cause_score_prompt(
    log_text: str,
    rca_summary: str,
    rca_detailed: str,
    causal_chain: List[str],
) -> str:
    chain = " -> ".join(causal_chain) if causal_chain else "(none)"
    return (
        f"Log entry:\n{log_text}\n\n"
        f"RCA summary: {rca_summary or '(empty)'}\n"
        f"RCA detailed: {rca_detailed or '(empty)'}\n"
        f"Causal chain: {chain}\n\n"
        "Score ROOT CAUSE QUALITY (0-1): Is the RCA technically sound, specific to "
        "this log's fault type, and does the causal chain represent a plausible failure path? "
        "1.0 = excellent RCA with sound causal chain, "
        "0.0 = vague, incorrect, or implausible."
    )

# =============================================================================
# RAG APPROACH (E03): EVIDENCE SUPPORT SCORE
# Are supporting_evidence items grounded in the actual log?
# =============================================================================

def build_evidence_support_prompt(
    log_text: str,
    supporting_evidence: List[str],
) -> str:
    ev = "\n".join(f"  - {e}" for e in supporting_evidence) if supporting_evidence else "  (none)"
    return (
        f"Log entry:\n{log_text}\n\n"
        f"AI supporting evidence items:\n{ev}\n\n"
        "Score EVIDENCE SUPPORT (0-1): Are all cited evidence items directly observable "
        "in or inferable from this specific log entry? "
        "1.0 = all items grounded in the log, "
        "0.0 = fabricated or not related to this log."
    )

# =============================================================================
# RAG APPROACH (E03): COMPLETENESS — 10-field schema
# Checks all fields in the full RAG output schema.
# =============================================================================

def build_completeness_prompt_rag(
    anomaly_explanation: str,
    rca_summary: str,
    rca_detailed: str,
    confidence_level: str,
    confidence_reasoning: str,
    causal_chain: List[str],
    supporting_evidence: List[str],
    risk_total_score: float,
    sre_action: str,
    devops_action: str,
) -> str:
    """
    Completeness for the RAG (E03) output schema.
    Expected fields: all 10 components of the full structured output.
    """
    return (
        "Evaluate COMPLETENESS of the anomaly analysis.\n"
        "Expected fields:\n"
        "  (1) anomaly_explanation  (2) rca.summary  (3) rca.detailed_description\n"
        "  (4) rca.confidence_level  (5) rca.confidence_reasoning\n"
        "  (6) rca.causal_chain (3+ steps)  (7) rca.supporting_evidence (2+ items)\n"
        "  (8) risk_score (system_impact, error_type, cascade_potential)  "
        "(9) remediation.sre_action  (10) remediation.devops_action\n\n"
        f"anomaly_explanation  : {anomaly_explanation or '(empty)'}\n"
        f"rca_summary          : {rca_summary or '(empty)'}\n"
        f"rca_detailed         : {rca_detailed or '(empty)'}\n"
        f"confidence_level     : {confidence_level or '(empty)'}\n"
        f"confidence_reasoning : {confidence_reasoning or '(empty)'}\n"
        f"causal_chain         : {causal_chain}\n"
        f"supporting_evidence  : {supporting_evidence}\n"
        f"risk_total_score     : {risk_total_score}\n"
        f"sre_action           : {sre_action or '(empty)'}\n"
        f"devops_action        : {devops_action or '(empty)'}\n\n"
        "Score COMPLETENESS (0-1): 1.0 = all fields present and substantive."
    )
