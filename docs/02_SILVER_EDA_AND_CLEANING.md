# 🔬 Part 2: Silver EDA & Quality Cleaning Pipeline

This document details how the unified Bronze dataset is analyzed, cleaned, and split into train, validation, and test datasets in `dataset_prep.ipynb` (Silver phase).

---

## 1. The 11 Data Quality Filters
To prevent model degradation, formatting collapse, or parroting behavior during training on a small 4B parameter model, we apply 11 strict data filters:

| Step | Check Name | Detection Heuristic / Threshold | Mitigation Action |
| :--- | :--- | :--- | :--- |
| **5.1** | **Null / Empty Check** | String length of system, user, or assistant $= 0$ after strip. | Drop record |
| **5.2** | **Exact Duplicate Pairs** | Exact match on `user` + `assistant` fields. | Drop duplicate |
| **5.3** | **Near-Duplicate Prompts** | Matches first 200 characters of JDs (normalized whitespace). | Drop duplicate |
| **5.4** | **Assistant Format Check** | Verifies assistant response begins with a numbered list (`^\d+\.`).| Flag and drop |
| **5.5** | **Outlier Length Detection**| IQR outlier checks on character lengths. | Flag only |
| **5.6** | **System Prompt Variance** | Compares unique system prompts. Override all with standard template. | Standardize in Silver |
| **5.7** | **Question Count Integrity**| Regex `re.findall(r"^\d+\.", assistant)` must count exactly $= 20$. | Drop if count $\neq 20$ |
| **5.8** | **Min Token Length** | Assistant response token count must be $\ge 350$ tokens. | Drop if too short |
| **5.9** | **Input Leakage Check** | Sliding window similarity: overlap window score $> 15\%$. | Drop high overlap |
| **5.10**| **Within-Response Repetition**| Counts unique 8-grams. Repetitive ratio must be $< 8\%$. | Drop if repetition $> 8\%$ |
| **5.11**| **Encoding Artifacts** | Searches for HTML entities (`&amp;`), replacement char (`\ufffd`), control chars. | Flag & Clean |

---

## 2. Combined Cleaning Drop Decision
The notebook merges these filters into a single boolean mask. Records violating any of these constraints are dropped before training:

```python
drop_mask = (
    (df["_q_count"] != 20)                          # Malformed Q count
    | (df["assistant_tokens"] < MIN_ASSISTANT_TOKENS) # Too short / truncated
    | (df["_leakage"] > LEAKAGE_THRESHOLD)           # Copied input text
    | (df["_repetition"] > REPETITION_THRESHOLD)     # Generation loop collapse
    | (df["total_tokens_correct"] > recommended * 1.05)  # Over sequence budget
    | df["assistant"].str.strip().eq("")             # Empty response
)
clean_df = df[~drop_mask].copy()
```

### 📊 Dataset Health Profile:
Our verification audit over the 4,013 Bronze records yielded the following metrics:
* **Malformed Question Count (!= 20):** 0 records
* **Too short (< 350 assistant tokens):** 0 records
* **Input Leakage (> 15% overlap):** 0 records
* **Repetition loops (> 8% ratio):** 0 records
* **Total dropped records:** 0 (100.0% clean pass)

---

## 3. Stratified Train/Val/Test Split
To prevent data leakage and ensure that minority job categories (domains) are represented in all splits, we perform a **stratified split** on the `domain` column:

* **Split Ratios:** 80% Train, 10% Validation, 10% Test.
* **Stratification Mapping:** Runs `train_test_split(stratify=df['domain'])`.
* **Output Export:** Saves the subsets back into their raw JSONL formats:
  * `dataset/processed/train.jsonl`
  * `dataset/processed/val.jsonl`
  * `dataset/processed/test.jsonl`
