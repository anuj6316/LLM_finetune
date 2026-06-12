# 🏗️ Bronze Dataset Notebook — Engineering Guide
### Production-level approach: Raw → Bronze (Merge + Normalize)

---

## 📊 What You're Working With (Audit Results)

| File | Records | Version(s) | Has `tone`? | Quality Score Range |
|---|---|---|---|---|
| `raw_dataset.jsonl` | 1,104 | 3.1, 3.2, 3.3 | ❌ Missing | 90–100 |
| `raw_dataset_type2.jsonl` | 481 | 4.0 | ✅ (5 distinct tones) | 95–99 |
| `new_raw_dataset.jsonl` | 334 | 3.4 | ✅ (2 tones) | 98–100 |
| **Total** | **1,919** | — | — | — |

**Key findings before you write a single line of code:**
- All 3 files share the **same top-level schema**: `{"conversations": [...], "metadata": {...}}`
- All conversations have the **same 3-turn structure**: `[system, user, assistant]`
- **Zero cross-file duplicates** on `user` content (confirmed via MD5 hash)
- `raw_dataset.jsonl` lacks the `tone` field entirely — you need to normalize this
- Quality scores differ meaningfully: type2 is noisier (95–99), raw/new are mostly 100
- `version` is a reliable proxy for *generation batch/prompt template* — keep it

---

## 🧠 The Senior ML Engineer Mental Model

In production data pipelines, data tiers follow a **medallion architecture**:

```
RAW (immutable) → BRONZE (merged, schema-normalized) → SILVER (cleaned, deduped) → GOLD (model-ready)
```

Your notebook is specifically the **RAW → BRONZE** step. The rules here are:

> [!IMPORTANT]
> **Bronze is NOT cleaning.** It is **faithful ingestion + schema normalization**.
> Never drop records, never fix content, never filter. Just normalize the envelope.
> Drops and filters happen in Silver.

This matters because you want a full audit trail. If you filter in Bronze and later realize you were wrong, you've lost data you can never get back.

---

## 🗂️ Notebook Structure (Cell-by-Cell Blueprint)

### Cell 0: Notebook Header (Markdown)

```markdown
# Bronze Dataset Builder
## Raw → Bronze: Multi-source Merge + Schema Normalization
**Goal**: Ingest all raw JSONL sources into a single, schema-normalized bronze file.
**Output**: `dataset/processed/bronze/processed_bronze_dataset.jsonl`
**DO NOT**: filter, clean content, or drop records here. That is Silver's job.
```

---

### Cell 1: Imports

```python
import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
```

> [!TIP]
> In production, engineers add `logging` instead of `print`. For a notebook that you run interactively, structured prints with emoji prefixes are fine — but use a consistent pattern.

---

### Cell 2: CONFIG Block (Single Source of Truth)

```python
# ════════════════════════════════════════════════════════
#  CONFIG  —  change only this cell between runs
# ════════════════════════════════════════════════════════

PROJECT_ROOT = Path("..").resolve()

RAW_SOURCES = [
    {
        "path": PROJECT_ROOT / "dataset/raw/raw_dataset.jsonl",
        "source_tag": "raw_v3",        # human-readable batch ID
        "default_tone": "not_specified", # tone is missing in this file
    },
    {
        "path": PROJECT_ROOT / "dataset/raw/raw_dataset_type2.jsonl",
        "source_tag": "raw_v4_type2",
        "default_tone": None,            # tone IS present, don't override
    },
    {
        "path": PROJECT_ROOT / "dataset/raw/new_raw_dataset.jsonl",
        "source_tag": "raw_v3_new",
        "default_tone": None,
    },
]

OUT_DIR  = PROJECT_ROOT / "dataset/processed/bronze"
OUT_FILE = OUT_DIR / "processed_bronze_dataset.jsonl"

PIPELINE_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

print(f"✅ Config loaded | Run ID: {PIPELINE_RUN_ID}")
print(f"📂 Output → {OUT_FILE}")
```

> [!NOTE]
> `PIPELINE_RUN_ID` and `source_tag` are critical in production. When you later find a bad batch of records, you filter by `source_tag`, not by guessing. This is called **data lineage**.

---

### Cell 3: Ingestion Layer — `load_jsonl()`

```python
def load_jsonl(path: Path, source_tag: str) -> tuple[list[dict], list[dict]]:
    """
    Loads a JSONL file. Returns (valid_records, error_log).
    Each record gets a raw_id and source_tag injected at load time.
    
    Senior pattern: never silently swallow errors. 
    Collect them and report at the end.
    """
    records, errors = [], []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Inject lineage fields at read time — before ANY processing
                rec["_meta"] = {
                    "raw_id":     str(uuid.uuid4()),
                    "source_tag": source_tag,
                    "source_file": path.name,
                    "line_num":   line_num,
                }
                records.append(rec)
            except json.JSONDecodeError as e:
                errors.append({
                    "file": path.name,
                    "line": line_num,
                    "error": str(e),
                })
    
    return records, errors
```

> [!IMPORTANT]
> Notice `_meta` uses a leading underscore — this is a convention to signal "this field was injected by the pipeline, not from the source". It will never collide with your data schema.

---

### Cell 4: Schema Normalizer — `normalize_record()`

This is the heart of the Bronze step. The only job is to **make every record look identical** regardless of which source file it came from.

```python
BRONZE_SCHEMA_VERSION = "bronze_1.0"

def normalize_record(rec: dict, default_tone: Optional[str]) -> dict:
    """
    Normalize a raw record into the canonical Bronze schema.
    
    Bronze schema contract:
    - conversations: list of {role, content} — untouched
    - metadata: fully normalized with all fields present
    - _bronze: pipeline provenance block
    
    Rule: ADD missing fields with sentinel values. NEVER remove or alter content.
    """
    convs = rec.get("conversations", [])
    meta  = rec.get("metadata", {})
    _meta = rec.get("_meta", {})
    
    # ── 1. Normalize tone ──────────────────────────────────────────────────────
    # If tone is missing in the original AND we have no default, set to sentinel
    tone = meta.get("tone")
    if tone is None:
        tone = default_tone if default_tone is not None else "not_specified"
    
    # ── 2. Normalize q_counts — ensure all 5 keys always exist ────────────────
    raw_qc = meta.get("q_counts", {})
    q_counts = {
        "[Technical]":   raw_qc.get("[Technical]",   0),
        "[Behavioral]":  raw_qc.get("[Behavioral]",  0),
        "[Situational]": raw_qc.get("[Situational]", 0),
        "[Culture]":     raw_qc.get("[Culture]",     0),
        "[Career]":      raw_qc.get("[Career]",      0),
    }
    
    # ── 3. Compute a content fingerprint for later dedup in Silver ─────────────
    user_content   = next((c["content"] for c in convs if c["role"] == "user"), "")
    content_hash   = hashlib.sha256(user_content.encode("utf-8")).hexdigest()[:16]
    
    return {
        "conversations": convs,  # ← NEVER modified in Bronze
        "metadata": {
            "domain":        meta.get("domain",        "Unknown"),
            "industry":      meta.get("industry",      "Unknown"),
            "level":         meta.get("level",         "Unknown"),
            "company_size":  meta.get("company_size",  "Unknown"),
            "tone":          tone,                      # ← normalized
            "quality_score": meta.get("quality_score", None),
            "q_counts":      q_counts,                  # ← all 5 keys guaranteed
            "version":       str(meta.get("version",   "Unknown")),
        },
        "_bronze": {
            "raw_id":          _meta.get("raw_id"),
            "source_tag":      _meta.get("source_tag"),
            "source_file":     _meta.get("source_file"),
            "source_line_num": _meta.get("line_num"),
            "content_hash":    content_hash,
            "pipeline_run_id": PIPELINE_RUN_ID,
            "schema_version":  BRONZE_SCHEMA_VERSION,
        }
    }
```

> [!NOTE]
> `content_hash` is SHA-256 of the user prompt. In Silver, deduplication is just `df.drop_duplicates("content_hash")`. Doing the hashing here costs nothing but saves a full re-read later.

---

### Cell 5: Pipeline Orchestrator — `run_bronze_pipeline()`

```python
def run_bronze_pipeline(sources: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    """
    Orchestrates: Load → Validate → Normalize → Merge.
    Returns (bronze_records, ingestion_report_df)
    """
    all_bronze  = []
    report_rows = []
    all_errors  = []
    
    for source in sources:
        path        = source["path"]
        source_tag  = source["source_tag"]
        default_tone = source["default_tone"]
        
        print(f"\n{'─'*60}")
        print(f"📥 Loading: {path.name}  [{source_tag}]")
        
        # Step 1: Load
        raw_records, errors = load_jsonl(path, source_tag)
        all_errors.extend(errors)
        print(f"   Loaded     : {len(raw_records):,} records")
        if errors:
            print(f"   ⚠️  Parse errors: {len(errors)}")
        
        # Step 2: Normalize
        normalized = [normalize_record(r, default_tone) for r in raw_records]
        all_bronze.extend(normalized)
        
        # Step 3: Accumulate ingestion report
        report_rows.append({
            "source_file":    path.name,
            "source_tag":     source_tag,
            "records_loaded": len(raw_records),
            "parse_errors":   len(errors),
            "tone_filled_in": sum(
                1 for r in normalized
                if r["_bronze"]["source_tag"] == source_tag
                and default_tone is not None
            ),
        })
        print(f"   Normalized : ✅")
    
    print(f"\n{'═'*60}")
    print(f"✅ MERGE COMPLETE")
    print(f"   Total records: {len(all_bronze):,}")
    print(f"   Parse errors : {len(all_errors)}")
    
    report_df = pd.DataFrame(report_rows)
    return all_bronze, report_df, all_errors
```

---

### Cell 6: Run the Pipeline

```python
bronze_records, ingestion_report, parse_errors = run_bronze_pipeline(RAW_SOURCES)
display(ingestion_report)
```

Expected output:

```
────────────────────────────────────────────────────
📥 Loading: raw_dataset.jsonl  [raw_v3]
   Loaded     : 1,104 records
   Normalized : ✅
...
════════════════════════════════════════════════════
✅ MERGE COMPLETE
   Total records: 1,919
   Parse errors : 0
```

---

### Cell 7: Duplicate Detection (Bronze Gate)

```python
# Bronze does NOT drop duplicates — it only REPORTS them.
# Dropping is Silver's job.

hashes = [r["_bronze"]["content_hash"] for r in bronze_records]
hash_counts = pd.Series(hashes).value_counts()
dups = hash_counts[hash_counts > 1]

print(f"Unique content hashes : {hash_counts[hash_counts == 1].sum():,}")
print(f"Duplicate content hashes: {len(dups)}")

if not dups.empty:
    print("\n⚠️  Duplicates found (Bronze will still write them, Silver will drop):")
    print(dups.head(10))
```

> [!WARNING]
> If you drop here, you lose the ability to audit WHY duplicates existed. Was it a pipeline bug? Two datasets that overlapped? Write them all, tag them, filter in Silver.

---

### Cell 8: Quick Sanity Checks Before Writing

```python
# Quick structural assertion before you write anything to disk
assert len(bronze_records) > 0, "Pipeline produced 0 records!"

# Every record must have the canonical keys
required_keys = {"conversations", "metadata", "_bronze"}
schema_fails = [
    i for i, r in enumerate(bronze_records) 
    if not required_keys.issubset(r.keys())
]
assert not schema_fails, f"Schema failures at indices: {schema_fails[:10]}"

# Every _bronze block must have a run_id
missing_run_id = [
    i for i, r in enumerate(bronze_records) 
    if not r["_bronze"].get("pipeline_run_id")
]
assert not missing_run_id, "Some records missing pipeline_run_id!"

print(f"✅ All {len(bronze_records):,} records passed pre-write assertions")
print(f"   Run ID : {PIPELINE_RUN_ID}")
```

> [!TIP]
> These `assert` statements ARE your unit tests. In a production Airflow/Prefect DAG this exact pattern becomes a `DataQualityOperator`. Here in notebook form, they serve the same purpose.

---

### Cell 9: Write Bronze Output

```python
def write_jsonl(records: list[dict], out_path: Path) -> None:
    """Write records to JSONL. Creates parent dirs automatically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

write_jsonl(bronze_records, OUT_FILE)

# Verify the write
written_count = sum(1 for _ in open(OUT_FILE))
assert written_count == len(bronze_records), \
    f"Write verification failed: wrote {written_count}, expected {len(bronze_records)}"

print(f"✅ Written  : {written_count:,} records → {OUT_FILE}")
print(f"   File size: {OUT_FILE.stat().st_size / 1024:.1f} KB")
```

---

### Cell 10: Bronze Ingestion Report (EDA Preview)

This is the bridge between Bronze and your Silver EDA notebook.

```python
# Flatten _bronze metadata into a DataFrame for quick analysis
report_data = []
for r in bronze_records:
    b = r["_bronze"]
    m = r["metadata"]
    report_data.append({
        "source_tag":     b["source_tag"],
        "source_file":    b["source_file"],
        "content_hash":   b["content_hash"],
        "domain":         m["domain"],
        "industry":       m["industry"],
        "level":          m["level"],
        "tone":           m["tone"],
        "quality_score":  m["quality_score"],
        "version":        m["version"],
    })

bronze_df = pd.DataFrame(report_data)

print("=== Source Distribution ===")
print(bronze_df["source_tag"].value_counts().to_string())

print("\n=== Tone Distribution ===")
print(bronze_df["tone"].value_counts().to_string())

print("\n=== Quality Score Stats ===")
print(bronze_df["quality_score"].describe())

print("\n=== Version Distribution ===")
print(bronze_df["version"].value_counts().to_string())
```

---

## 📁 Output Schema Contract

Every record in `processed_bronze_dataset.jsonl` will have this exact shape:

```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "domain":        "string",
    "industry":      "string",
    "level":         "string",
    "company_size":  "string",
    "tone":          "string (never null)",
    "quality_score": "number | null",
    "q_counts": {
      "[Technical]": 0, "[Behavioral]": 0,
      "[Situational]": 0, "[Culture]": 0, "[Career]": 0
    },
    "version": "string"
  },
  "_bronze": {
    "raw_id":          "uuid4",
    "source_tag":      "raw_v3 | raw_v4_type2 | raw_v3_new",
    "source_file":     "filename.jsonl",
    "source_line_num": 42,
    "content_hash":    "sha256[:16]",
    "pipeline_run_id": "20260611T150000Z",
    "schema_version":  "bronze_1.0"
  }
}
```

---

## 🔜 What Goes in Your Silver / EDA Notebook

After this Bronze notebook runs successfully, your **Silver EDA notebook** (`dataset_prep.ipynb` refactored) should:

1. **Load** `processed_bronze_dataset.jsonl` (single source, guaranteed schema)
2. **Dedup** on `_bronze.content_hash` — drop and log
3. **Quality filter** — e.g., `quality_score >= 95`
4. **Token length analysis** — using `transformers` tokenizer for Gemma 4
5. **Outlier detection** — flag conversations where assistant response is too short/long
6. **Distribution plots** — domain/industry/tone/level/quality breakdown
7. **Train/Val/Test split** — stratified on `domain` or `level`
8. **Write** `dataset/processed/silver/` splits

> [!TIP]
> The `_bronze.source_tag` field makes stratification by data source trivial. You can see if one source dominates your train set and balance accordingly.

---

## ⚡ Checklist Before Running

- [ ] Run `pip install pandas` if not already installed
- [ ] Confirm `dataset/raw/` paths exist relative to `scripts/`
- [ ] `OUT_DIR` will be auto-created by `mkdir(parents=True, exist_ok=True)`
- [ ] Bronze notebook does **not** require a GPU or tokenizer — runs in seconds
