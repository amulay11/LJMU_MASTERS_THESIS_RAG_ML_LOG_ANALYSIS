# BGL Dataset Specification

## 1. Document Metadata

| Field | Value |
|-------|--------|
| **Dataset name** | BGL (Blue Gene/L) RAS Log Dataset |
| **Version** | 1.0 |
| **Last updated** | 2025-03-14 |
| **Description** | Specification for the BGL log dataset—a sample of Blue Gene/L Reliability, Availability, and Serviceability (RAS) event logs used for log analysis and failure-prediction research. |

---

## 2. Dataset Overview

### 2.1 Source & Domain

- **Source:** IBM Blue Gene/L (BG/L) supercomputer RAS event logs.
- **Domain:** High-performance computing (HPC); system reliability, availability, and serviceability; kernel and application-level failure events from a massively parallel system.
- **Log type:** RAS (Reliability, Availability, and Serviceability) events produced by the BG/L firmware and software stack, including kernel (KERNEL) and application (APP) components.

### 2.2 Scope

- **This dataset:** A **2,000-line sample** of BGL RAS logs (`BGL_2k`) plus derived structured and template files.
- **Temporal coverage (in sample):** Log timestamps in the sample span **2005-06-03** to **2005-06-11** (9 days).
- **Scale:** The sample is a subset of logs that would have been produced by a full Blue Gene/L system (see Operational Context for system scale).

### 2.3 Research Relevance

- **Failure prediction:** RAS logs have been used to predict memory, network, and I/O failures (e.g. ~80% prediction for memory/network, ~47% for application I/O in published studies).
- **Anomaly detection:** Log lines are labeled (e.g. normal vs. APPREAD, KERNDTLB) for supervised or semi-supervised anomaly detection.
- **Log parsing & template mining:** Fixed-format messages and provided templates support log parsing, event-type identification, and template extraction benchmarks.
- **Benchmark:** Widely used in log analysis and failure-prediction literature (e.g. ICDM, IBM Research publications).

### 2.4 Operational Context

*This subsection describes the operational context of the system that produced the BGL logs. It is intended for use as a knowledge-base component in RAG or other retrieval-aided systems.*

#### 2.4.1 System Identity and Role

- **System:** IBM Blue Gene/L (BG/L), a massively parallel supercomputer developed by IBM in partnership with the U.S. Department of Energy’s National Nuclear Security Administration (NNSA).
- **Primary deployment (relevant to this dataset):** Lawrence Livermore National Laboratory (LLNL), Terascale Simulation Facility. The system was deployed in 2005 and became the world’s fastest supercomputer (TOP500 #1 from November 2004 to June 2008).
- **Peak performance:** 360 teraflops (design); Linpack record ~280.6 teraflops at 131,072 processors (128K).
- **Workload:** Large-scale scientific applications—e.g. hydrodynamics, quantum chemistry, molecular dynamics, climate modeling, and Stockpile Stewardship Program workloads—often spanning thousands of processors. At this scale, failures are common and are treated as a normal operational concern.

#### 2.4.2 Hardware and Node Organization

- **Scale:** Up to **65,536 dual-processor nodes** (131,072 processors); 128K-processor configurations were used in production.
- **Rack and node hierarchy:** Compute nodes are organized in **racks** and **midplanes**. Each rack contains 1,024 nodes (2,048 cores). Node naming in the logs reflects this hierarchy (see *Node naming* below).
- **Compute node:** Each node has two **PowerPC 440** cores at 700 MHz, with shared network injection/reception FIFOs. The node runs a lightweight **Compute Node Kernel (CNK)**—no virtual memory, no multi-tasking, single-application-per-node for efficiency. CNK delegates file I/O to dedicated **I/O nodes**.
- **I/O nodes:** Run **INK (I/O Node Kernel)**, a modified Linux kernel. They handle filesystem and control traffic for compute nodes. Log messages referring to **ciod** (see below) relate to the control I/O daemon running in this environment.
- **Networks:** Blue Gene/L has five networks; two are central for messaging:
  - **3-D Torus:** General point-to-point communication (~175 MB/s per link). Log events mention “torus” (e.g. torus receiver/sender, retransmission, pipe errors).
  - **Tree network:** Collective operations (broadcast, reduction), ~350 MB/s per link. Log events mention “tree” (e.g. tree receiver re-synch).
- **Link chip:** Key component in the system-on-a-chip design; interconnect and RAS logic are tied to link and network hardware, which appears in many RAS messages (e.g. torus/tree errors, DCR values).

#### 2.4.3 RAS (Reliability, Availability, and Serviceability)

- **RAS** is IBM’s framework for logging and managing reliability, availability, and serviceability. On Blue Gene/L, RAS encompasses:
  - **Correctable errors:** e.g. instruction-cache parity corrected, DDR errors detected and corrected, L3 EDRAM errors, torus/tree link errors. Many appear as **INFO**.
  - **Uncorrectable or fatal events:** e.g. **FATAL** kernel or application failures (TLB errors, alignment faults, ciod control-stream failures, core dumps).
- **Management model:** Production systems used a **database-centric** approach: configuration and operational data were stored in a relational database on the **service node**, which acted as a hub for management processes and RAS querying.
- **Log collection:** RAS event logs were collected over long periods (e.g. 100+ days in research) and processed for failure analysis and prediction. Filtering and coalescing could reduce raw log volume significantly (e.g. >99% reduction in some studies) while retaining failure-relevant events.

#### 2.4.4 Key Software and Log Sources

- **CNK (Compute Node Kernel):** Minimal OS on compute nodes; logs from the kernel side appear as **RAS KERNEL** with events such as instruction cache parity, TLB errors, alignment exceptions, “generating core.*”, and torus/tree/link errors.
- **ciod (Control I/O daemon):** Software responsible for control and I/O coordination (e.g. loading programs, node map, CioStream sockets). Many **RAS APP** messages refer to **ciod**—e.g. “failed to read message prefix on control stream”, “LOGIN chdir(…) failed”, “Error creating node map from file”, “Error loading … program image”. These often indicate **FATAL** application-level failures.
- **BGLMaster / mmcs_server:** System management and monitoring; referenced in the full template set (e.g. BGLMaster start, mmcs_server exit). Not prominent in the 2k sample but part of the broader BGL operational context.
- **Node map:** Configuration file describing canonical-to-logical node mapping; errors in this file or in reading it show up as ciod node-map and coordinate errors.

#### 2.4.5 Node Naming and Location (Log Field “Node”)

- **Format in logs:** `Rxx-Mx-Nx-C:Jxx-Uxx` or `Rxx-Mx-Nx-I:Jxx-Uxx`.
  - **R** = Rack.
  - **M** = Midplane within rack.
  - **N** = Node position (hex digit: 0–9, A–F).
  - **C vs I:** **C** = Compute node, **I** = I/O node.
  - **J** = Slot/card identifier.
  - **U** = Unit (e.g. U01, U11).
- **Example:** `R30-M0-N9-C:J16-U01` = Rack 30, Midplane 0, Node 9, Compute node, slot J16, unit U01. **I** in the same position indicates an I/O node (e.g. for ciod-related APP FATALs).

#### 2.4.6 Failure Modes Reflected in the Logs

- **Transient hardware:** Bit flips, cache parity (corrected), DDR/torus/tree correctable errors.
- **Permanent or severe hardware:** TLB errors, alignment faults, repeated correctable errors leading to core dumps or node failure.
- **Software/config:** ciod failures (control stream, node map, program load, LOGIN chdir), BGLMaster/mmcs issues.
- **Network:** Torus and tree link/pipe errors, retransmissions, re-synch events.
- **Cooling/power:** Referenced in the full template set (e.g. PGOOD, power module, fan modules); less frequent in the 2k sample.

Understanding this operational context helps interpret event types, severity (INFO vs FATAL), and labels (e.g. APPREAD vs KERNDTLB) when building log analysis or RAG-based assistants.

---

## 3. Folder Contents & File Inventory

### 3.1 File List

| File | Format | Approx. size (lines) | Purpose |
|------|--------|------------------------|---------|
| `BGL_2k.log` | Plain text (.log) | 2,000 | Raw RAS log lines (2k sample). |
| `BGL_2k.log_structured.csv` | CSV | 2,001 (1 header + 2,000 rows) | Parsed log records with columns for timestamp, node, type, component, level, content, EventId, EventTemplate, and Label. |
| `BGL_2k.log_templates.csv` | CSV | 121 (1 header + 120 templates) | Event templates observed in the 2k sample (EventId, EventTemplate). |
| `BGL_templates.csv` | CSV | 378 (1 header + 377 templates) | Full template catalog for BGL RAS (all known event types; superset of 2k templates). |

### 3.2 Relationship Between Files

- **BGL_2k.log** is the **source** raw log.
- **BGL_2k.log_structured.csv** is a **parsed** version of `BGL_2k.log`: each line in the log corresponds to one row; columns are parsed fields plus **EventId** and **EventTemplate** from template matching.
- **BGL_2k.log_templates.csv** contains exactly the **templates observed in the 2k sample** (120 event types). Used to interpret EventId/EventTemplate in the structured file.
- **BGL_templates.csv** is the **full BGL RAS template dictionary** (377 templates). Use it for parsing larger BGL logs or for reference; the 2k sample uses only a subset.

---

## 4. File Specifications

### 4.1 BGL_2k.log

- **Purpose:** Raw, human-readable RAS log sample (2,000 lines) as produced or exported from the Blue Gene/L RAS logging pipeline.
- **Format:** Plain text, one log entry per line. No explicit header. Fields are space-separated; the message content may contain spaces.
- **Schema (log line structure):**
  - **Optional label (first column):** Either `-` (normal) or an anomaly/failure label (e.g. `APPREAD`, `KERNDTLB`). If present, it is the first token; otherwise the line can start with `-`.
  - **Timestamp (Unix-like):** Integer seconds since epoch (e.g. 1117838570).
  - **Date:** `YYYY.MM.DD` (e.g. 2005.06.03).
  - **Node:** Location string (e.g. R02-M1-N0-C:J12-U11).
  - **Time:** Full timestamp with microseconds (e.g. 2005-06-03-15.42.50.675872).
  - **Node (repeat):** Same as Node.
  - **Type:** `RAS`.
  - **Component:** `KERNEL` or `APP`.
  - **Level:** `INFO` or `FATAL` (and possibly others in full BGL).
  - **Content:** Free-form message (rest of line); may contain spaces, colons, IPs, hex values.
- **Sample lines:**

```text
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
APPREAD 1117869872 2005.06.04 R04-M1-N4-I:J18-U11 2005-06-04-00.24.32.432192 R04-M1-N4-I:J18-U11 RAS APP FATAL ciod: failed to read message prefix on control stream (CioStream socket to 172.16.96.116:33569
KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt
```

- **Notes:** Content is unstructured; parsing requires the fixed field order above. Labels are only present for some anomaly types; “-” denotes normal or unlabeled.

### 4.2 BGL_2k.log_structured.csv

- **Purpose:** Machine-readable, parsed version of `BGL_2k.log` with one row per log line and columns for every logical field plus EventId, EventTemplate, and Label.
- **Format:** CSV, comma-separated. Content field may be quoted if it contains commas.
- **Schema:**

| Column | Description |
|--------|-------------|
| LineId | Sequential line number (1–2000). |
| Label | Anomaly label: `-` (normal), `APPREAD`, `KERNDTLB`, or other (see Section 6). |
| Timestamp | Unix timestamp (integer seconds). |
| Date | Date only, `YYYY.MM.DD`. |
| Node | Node location (e.g. R02-M1-N0-C:J12-U11). |
| Time | Full timestamp with microseconds. |
| NodeRepeat | Same as Node. |
| Type | Always `RAS` in this sample. |
| Component | `KERNEL` or `APP`. |
| Level | `INFO` or `FATAL`. |
| Content | Raw message text. |
| EventId | Template identifier (e.g. E77, E33, E55). |
| EventTemplate | Template string with `<*>` for variable parts. |

- **Sample row (header + 2 rows):**

```csv
LineId,Label,Timestamp,Date,Node,Time,NodeRepeat,Type,Component,Level,Content,EventId,EventTemplate
1,-,1117838570,2005.06.03,R02-M1-N0-C:J12-U11,2005-06-03-15.42.50.675872,R02-M1-N0-C:J12-U11,RAS,KERNEL,INFO,instruction cache parity error corrected,E77,instruction cache parity error corrected
2,-,1117838573,2005.06.03,R02-M1-N0-C:J12-U11,2005-06-03-15.42.53.276129,R02-M1-N0-C:J12-U11,RAS,KERNEL,INFO,instruction cache parity error corrected,E77,instruction cache parity error corrected
```

- **Notes:** EventId/EventTemplate come from matching Content to the template set. Use `BGL_2k.log_templates.csv` to interpret EventIds in this file.

### 4.3 BGL_2k.log_templates.csv

- **Purpose:** Catalog of event templates that appear in the 2k sample. Used for log parsing, event-type aggregation, and linking EventIds to semantic descriptions.
- **Format:** CSV with header: `EventId,EventTemplate`.
- **Schema:**
  - **EventId:** Unique template id (e.g. E1, E77).
  - **EventTemplate:** Template string; `<*>` denotes a variable token (number, address, node, path, etc.).
- **Sample rows:**

```csv
EventId,EventTemplate
E3,<*> double-hummer alignment exceptions
E18,"CE sym <*>, at <*>, mask <*>"
E33,ciod: failed to read message prefix on control stream (CioStream socket to <*>:<*>
E55,data TLB error interrupt
E67,generating core.<*>
E77,instruction cache parity error corrected
```

- **Notes:** 120 templates in this file. Variable parts are normalized to `<*>`; actual values are in the structured CSV Content column.

### 4.4 BGL_templates.csv

- **Purpose:** Full set of BGL RAS event templates (377 templates). Use for parsing larger BGL logs or as a reference for all known BGL RAS event types.
- **Format:** Same as `BGL_2k.log_templates.csv`: `EventId,EventTemplate`.
- **Notes:** Includes templates not present in the 2k sample (e.g. BGLMaster, Lustre, fan/power, more ciod variants). EventIds range from E1 to E377 (approximately).

---

## 5. Log Format & Semantics

### 5.1 Raw Log Line Structure

A typical line has the form:

`[Label] Timestamp Date Node Time NodeRepeat Type Component Level Content`

- **Label** is optional; when present it is an anomaly/failure type (`-`, `APPREAD`, `KERNDTLB`, etc.).
- **Timestamp** is Unix seconds; **Date** and **Time** give human-readable date and time with microseconds.
- **Node** (and NodeRepeat) identify the source location (rack-midplane-node and C/I, slot, unit).
- **Type** is `RAS`; **Component** is `KERNEL` or `APP`; **Level** is `INFO` or `FATAL`.
- **Content** is the remainder of the line and may contain IPs, paths, hex values, and punctuation.

### 5.2 Field Semantics

- **Label:** Ground-truth anomaly/failure category for the line (for research labels). `-` = normal or non-anomalous.
- **Level:** **INFO** = informational (often correctable events); **FATAL** = fatal or uncorrectable (e.g. TLB error, ciod failure, alignment fault).
- **Component:** **KERNEL** = CNK/kernel RAS; **APP** = application/ciod RAS.
- **EventId / EventTemplate:** Identify the *type* of event (e.g. “instruction cache parity error corrected”, “generating core.<*>”, “data TLB error interrupt”). Useful for aggregation and failure-prediction features.

### 5.3 Template Convention

- **`<*>`** in a template means “one or more variable tokens” (numbers, addresses, paths, etc.). For example, `generating core.<*>` matches “generating core.2275” or “generating core.862”.
- Templates are matched to the **Content** field; the same EventId can appear many times with different variable values. EventIds are stable across the 2k and full template files for the same template text.

---

## 6. Data Analysis Insights

### 6.1 Basic Statistics

- **Total log lines (2k sample):** 2,000.
- **Structured rows:** 2,000 (1:1 with raw log).
- **Unique event types (in 2k):** 120 (from `BGL_2k.log_templates.csv`).
- **Full template set:** 377 templates (`BGL_templates.csv`).
- **Date range:** 2005-06-03 to 2005-06-11 (9 days).
- **Unique nodes (from sample):** Many distinct Rxx-Mx-Nx-C/I:Jxx-Uxx values; spread across racks (e.g. R00–R37), compute (C) and I/O (I) nodes.

### 6.2 Event Type Distribution

- **Frequent event types (in 2k sample):** Examples include:
  - **E77** – instruction cache parity error corrected (INFO).
  - **E67** – generating core.\* (core dumps; INFO).
  - **E18** – CE sym \*, at \*, mask \* (correctable error; INFO).
  - **E3** – \* double-hummer alignment exceptions (INFO).
  - **E55** – data TLB error interrupt (FATAL; associated with KERNDTLB).
  - **E33** – ciod: failed to read message prefix on control stream (FATAL; associated with APPREAD).
- **Component mix:** Both **KERNEL** and **APP** appear; APP events are often ciod-related and FATAL.

### 6.3 Label / Severity Distribution

- **Normal (Label = `-`):** Majority of lines (~93% in typical 2k samples).
- **Anomaly labels (examples in 2k):**
  - **APPREAD:** Application read/control-stream failures (ciod; FATAL); small count (e.g. 3 in one sample).
  - **KERNDTLB:** Kernel data TLB error interrupt (FATAL); multiple occurrences (e.g. 60) from the same node/time window.
- **Level:** Most lines **INFO**; **FATAL** lines correlate with anomaly labels and with events such as E33, E55, E64 (force load/store alignment).

### 6.4 Temporal Patterns

- Logs are not uniformly distributed in time; there are bursts (e.g. many “generating core” or “instruction cache parity” in short intervals) and quieter periods.
- **KERNDTLB** lines often appear in tight clusters (same node, minutes apart), consistent with repeated TLB errors on a single node.
- Date range (2005-06-03 to 2005-06-11) supports studies over a multi-day window.

### 6.5 Data Quality Notes

- **Completeness:** All 2,000 raw lines have a corresponding structured row with EventId and EventTemplate.
- **Labels:** Only a subset of lines have non-“-” labels; the rest are treated as normal. Suitable for binary (normal vs anomaly) or multi-class (by label type) tasks.
- **Content:** Some Content strings may be truncated in raw form (e.g. long socket messages); EventTemplate still identifies the event type.
- **Node format:** Consistent Rxx-Mx-Nx-C:Jxx-Uxx / I:Jxx-Uxx; parsing Node gives rack, midplane, node, compute vs I/O, and slot/unit for location-based analysis.

---

## 7. Recommendations for Research Use

- **Log parsing / template mining:** Use `BGL_2k.log` or `BGL_2k.log_structured.csv` with `BGL_2k.log_templates.csv` (or `BGL_templates.csv` for full dictionary) to evaluate parsers or template miners. EventId in the structured file can serve as ground truth for event type.
- **Anomaly detection:** Use **Label** (e.g. `-` vs APPREAD/KERNDTLB) for supervised or semi-supervised methods; use **Level** (FATAL) and **EventId** (E33, E55, etc.) as additional signals.
- **Failure prediction:** Use sequences of EventIds, nodes, and timestamps; incorporate operational context (Section 2.4) to interpret event semantics (e.g. ciod vs kernel, torus/tree vs TLB).
- **Train/test:** For temporal validity, split by time (e.g. first 70% dates for train, rest for test) or by node to avoid leakage.
- **Caveats:** The 2k sample is small; for scaling studies or rare events, use or generate larger BGL RAS logs. Labels are provided only for a few anomaly types; other failure modes may be unlabeled.

---

## 8. Appendix

### A. Column Reference (BGL_2k.log_structured.csv)

| Column | Type | Example |
|--------|------|---------|
| LineId | Integer | 1 |
| Label | Categorical | -, APPREAD, KERNDTLB |
| Timestamp | Integer | 1117838570 |
| Date | String | 2005.06.03 |
| Node | String | R02-M1-N0-C:J12-U11 |
| Time | String | 2005-06-03-15.42.50.675872 |
| NodeRepeat | String | R02-M1-N0-C:J12-U11 |
| Type | String | RAS |
| Component | Categorical | KERNEL, APP |
| Level | Categorical | INFO, FATAL |
| Content | String | (variable) |
| EventId | String | E77, E33, E55, … |
| EventTemplate | String | (template with \<*\>) |

### B. Template Counts

- **BGL_2k.log_templates.csv:** 120 templates (E1–E120 in the 2k subset; Ids may not be consecutive in the file).
- **BGL_templates.csv:** 377 templates (E1–E377; full BGL RAS catalog).

---

*End of BGL Dataset Specification.*
