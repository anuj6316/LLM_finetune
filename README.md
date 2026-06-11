---
language: en
license: apache-2.0
task_categories:
  - text-generation
tags:
  - interview-questions
  - job-description
  - recruiting
  - fine-tuning
  - gemma
---

# JD → Interview Questions Dataset

Generated with `dataset_generator.py` v3.2.

## Stats

| Split | Samples |
|-------|---------|
| Train | 1487 |
| Val   | 185 |
| Test  | 187 |
| **Total** | **1859** |

## Format

ShareGPT JSONL — Unsloth / SFTTrainer native.
Each record has `conversations` (system · user · assistant) and `metadata`.

## Coverage

- **14 industries** · **20 domains** · **7 levels** · **4 company sizes**
- Question types: [Technical] [Behavioral] [Situational] [Culture] [Career]
- 3-level repetition filtering (JD dedup · cross-sample staleness · within-sample Jaccard)

## Intended use

Fine-tuning Gemma 4 E4B via Unsloth QLoRA for automated interview-question generation.
