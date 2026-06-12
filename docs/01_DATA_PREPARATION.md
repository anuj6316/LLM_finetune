# 🗃️ Part 1: Raw Dataset Ingestion, Merging & Schema Normalization

This document outlines the pipeline used to collect, merge, and normalize the raw generated records into a unified **Bronze Dataset**.

---

## 1. Raw Data Sources & Versions
The dataset consists of a multi-generation collection of interview question sets. The records are generated from different scripts and configurations over time:

| Version | Source Tag | Records | Key Characteristics |
| :--- | :--- | :---: | :--- |
| **3.1** | `raw_v3` | 98 | Raw generation, tone is unspecified. |
| **3.2** | `raw_v3` | 949 | Raw generation, tone is unspecified. |
| **3.3** | `raw_v3` | 57 | Raw generation, tone is unspecified. |
| **3.4** | `raw_v3_new` | 1,582 | Upgraded generation script, specific tone directives. |
| **4.0** | `raw_v4_type2` | 1,327 | Upgraded formatting script, specific tone directives. |
| **Total**| — | **4,013** | Unified starting pool. |

---

## 2. The Bronze Merge Pipeline
The merge is executed in `ds_merge.ipynb`. It ingests these multi-source inputs and normalizes them into a single file at:
`dataset/processed/bronze/processed_bronze_dataset.jsonl`

### 🔧 Normalization Protocol:
1. **Schema Integrity:** Standardizes the conversation list into exactly three turns: `system`, `user` (job description), and `assistant` (generated questions).
2. **Metadata Uniformity:** Extracts metadata coordinates (`domain`, `industry`, `level`, `company_size`, `quality_score`, `version`, `q_counts`) and maps them to a consistent schema layout.
3. **Traceability:** Adds a `_bronze` metadata block to track:
   * Unique `raw_id`
   * Original `source_file`
   * Original `source_line_num`
   * Verification `content_hash`

---

## 3. The Round-Robin Tone Balance Strategy
When generating a subset of combinations (e.g. creating a target subset of 1,000 samples out of 39,200 possibilities across 5 tones), a standard random shuffle yields no statistical guarantee of class balance. You could easily end up with 300 samples of one tone and 100 of another.

To resolve this, the generator script applies a **Round-Robin Interleaving Strategy**:

### 🛠️ Interleaving Algorithm:
1. All generated combinations are grouped by their specific `tone`.
2. Each per-tone group is shuffled independently using a random seed.
3. The groups are interleaved sequentially in strict rotation:
   $$\text{Combo List} = [\text{Tone}_1[0], \text{Tone}_2[0], \dots, \text{Tone}_n[0], \text{Tone}_1[1], \dots]$$
4. The generator draws sequentially from this list until the target sample count is reached.

### 📈 Outcome:
This guarantees that for a target of $N$ samples across $T$ tones, each tone gets exactly $N/T$ samples (e.g. exactly 200 samples per tone for a 1,000 target), ensuring a clean, stratified base distribution for fine-tuning.
