## Synthetic Dataset Generation Pipeline

**Status**: Active | **Version**: 3.2 | **Progress**: 1,157 / 2,500 samples (46%)

### Overview
Automated pipeline to generate synthetic "Job Description → 20 Interview Questions" pairs for fine-tuning Gemma 4 E4B via Unsloth (QLoRA). Uses LLM API calls with async concurrency, 3-level deduplication, and quality scoring.

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total samples generated | 1,157 |
| Target samples | 2,500 |
| Completion | 46% |
| Combinations covered | 7,840 (14 industries × 20 domains × 7 levels × 4 company sizes) |
| Current combo index | 474 |

### Diversity Matrix
- **Industries**: 14 (Software/SaaS, FinTech, Healthcare, EdTech, BFSI, Gaming/Metaverse, AgriTech, etc.)
- **Domains**: 20 (Backend, Frontend, Full Stack, Data Science, ML/AI, DevOps/SRE, Product Management, HR/People Ops, etc.)
- **Levels**: 7 (Junior through VP/C-suite)
- **Company Sizes**: 4 (Early-stage Startup through Large Enterprise)

### Question Distribution (per sample)
- 7 × `[Technical]` — tests tools/skills from the JD
- 5 × `[Behavioral]` — "Tell me about a time..."
- 4 × `[Situational]` — "Imagine..."/"How would you approach..."
- 2 × `[Culture]` — values, collaboration
- 2 × `[Career]` — motivation, growth

### Key Features
- **3-Level Deduplication**: JD fingerprints, cross-sample stem reuse, within-sample Jaccard similarity
- **Quality Scoring**: 0-100 scale (distribution compliance, no repetition, specificity, length variety)
- **Resumability**: Checkpoint system saves progress every 25 samples
- **Async Concurrency**: Configurable parallel API calls (default: 4)

### Output Format
**ShareGPT JSONL** (Unsloth/SFTTrainer native) with `conversations` array and `metadata` object.

### Files
| File | Description |
|------|-------------|
| `dataset_generator.py` | Main generation script (754 lines) |
| `config.yml` | All tuneable parameters |
| `raw_dataset.jsonl` | Generated dataset (1,157 samples) |
| `checkpoint.json` | Current run state |
| `README.md` | HuggingFace-style dataset card |

### Usage
```bash
python dataset_generator.py
```
Pipeline stages: Generate JD → Generate Questions → Split (train/val/test) → Write README

### Notes
- Train/val/test splits (141 samples) are from an earlier run and need regeneration
- Project is mid-flight; checkpoint mechanism ready for resumption
