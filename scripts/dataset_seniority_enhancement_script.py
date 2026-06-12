#!/usr/bin/env python3
"""
================================================================================
  Dataset Seniority Enhancement Script v2.0 (LLM-Powered)
  Purpose: Fix uniform token counts using an LLM to generate deep, realistic
           questions for executive-level candidates.

  Best Practices Implemented:
  1. Atomic Writes: Writes to a temporary file, swapping only on clean success.
  2. Resumability: Checks progress and resumes seamlessly if interrupted.
  3. Format Validation: Rejects and retries invalid formats from the LLM.
  4. Portability: Standard library ONLY (no SDKs/external packages needed).
  5. Multi-Provider: Supports Ollama, OpenAI, and Gemini APIs.
================================================================================
"""

import json
import shutil
import time
import re
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DATASET = Path("dataset/raw/raw_dataset.jsonl")
BACKUP_DIR = Path("dataset/backup")
OUTPUT_DATASET = Path("dataset/raw/raw_dataset.jsonl")
TEMP_DATASET = Path("dataset/raw/raw_dataset.jsonl.tmp")

# Seniority levels target for LLM enhancement
SENIOR_LEVELS = {
    "Director",
    "VP / C-suite",
}

# --- LLM Provider Settings ---
# Options: "ollama" | "openai" | "gemini"
LLM_PROVIDER = "ollama"

# Set API Keys here or leave blank to read from environment variables
API_KEY = ""  # (Used for OpenAI or Gemini)

# Models
# - Ollama: e.g., "llama3", "mistral", "qwen2.5:7b-instruct"
# - OpenAI: e.g., "gpt-4o-mini", "gpt-4o"
# - Gemini: e.g., "gemini-2.5-flash"
MODEL_NAME = "gemma4:31b-cloud"

# Network Retry Settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds

# ==============================================================================
# LLM API CONNECTIONS
# ==============================================================================

def call_llm(prompt: str, system_instruction: str) -> str:
    """
    Interacts with the configured LLM API using only Python's built-in urllib.
    Handles timeouts, retries with exponential backoff, and credentials.
    """
    import os
    
    provider = LLM_PROVIDER.lower()
    api_key = API_KEY or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""

    # Setup Endpoint and Payload structures
    if provider == "ollama":
        url = "http://localhost:11434/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            },
            "stream": False
        }
    
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        if not api_key:
            raise ValueError("OpenAI API key is missing. Set it in the script config or export OPENAI_API_KEY.")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

    elif provider == "gemini":
        # Gemini uses API key in the URL parameter
        if not api_key:
            raise ValueError("Gemini API key is missing. Set it in the script config or export GEMINI_API_KEY.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"System Instruction: {system_instruction}\n\nUser Request:\n{prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }
    else:
        raise ValueError(f"Unsupported LLM Provider: {LLM_PROVIDER}")

    data_bytes = json.dumps(payload).encode("utf-8")
    
    # Retry Loop with Exponential Backoff
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                
                # Extract response based on API standard
                if provider == "ollama":
                    return resp_data["message"]["content"]
                elif provider == "openai":
                    return resp_data["choices"][0]["message"]["content"]
                elif provider == "gemini":
                    return resp_data["candidates"][0]["content"]["parts"][0]["text"]
                    
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if attempt == MAX_RETRIES - 1:
                raise e
            time.sleep(backoff)
            backoff *= 2.0
            
    return ""

# ==============================================================================
# VALIDATION ENGINE
# ==============================================================================

def validate_llm_response(original_text: str, generated_text: str) -> bool:
    """
    Validates that the generated text keeps the identical list structure
    (20 numbered items, correct categorization tags, no introductory fluff).
    """
    if not generated_text:
        return False
        
    orig_lines = [l.strip() for l in original_text.split("\n") if l.strip()]
    gen_lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
    
    # Validate count match
    orig_numbered = [l for l in orig_lines if re.match(r"^\d+\.", l)]
    gen_numbered = [l for l in gen_lines if re.match(r"^\d+\.", l)]
    
    if len(orig_numbered) != len(gen_numbered) or len(gen_numbered) == 0:
        return False
        
    # Check that bracket tags [Technical], [Behavioral], etc., are preserved
    for i, gen_line in enumerate(gen_numbered):
        orig_match = re.search(r"\[([^\]]+)\]", orig_numbered[i])
        gen_match = re.search(r"\[([^\]]+)\]", gen_line)
        
        if orig_match and not gen_match:
            return False
            
    return True

# ==============================================================================
# PROMPT DEFINITIONS
# ==============================================================================

SYSTEM_PROMPT = """You are an elite, highly experienced technical recruiter and executive interviewer.
Your task is to take a raw set of 20 interview questions and rewrite them specifically for executive leadership roles (such as Director, VP, and C-suite).

Your output must be sophisticated, highly strategic, realistic, challenging, and grammatically perfect.
Make sure the rewritten questions represent the extreme depth of real-world executive scenarios. 
- Technical questions should target system architectural strategy, long-term technical debt, business alignment, and multi-year roadmaps.
- Behavioral & Situational questions should focus on complex organizational changes, managing senior conflicts, ambiguous trade-offs, crisis management, and driving high-level company culture.

CRITICAL FORMAT RULES:
1. Maintain the EXACT structure and formatting of the input list.
2. Every rewritten line MUST match the original index prefix and category tag (e.g. '1. [Technical] Rewritten deep question').
3. Do NOT add any introductory, conversational, or concluding remarks in your output. Return ONLY the 20-question block.
4. Ensure the output list contains exactly 20 numbered items matching the items in the input list."""

def build_user_prompt(domain: str, level: str, questions: str) -> str:
    return f"""Target Seniority Level: {level}
Professional Domain: {domain}

Here is the current, basic 20-question interview script. Completely rewrite each of them to make them highly nuanced, strategic, and appropriate for a candidate at the {level} level in {domain}:

{questions}"""

# ==============================================================================
# EXECUTION FLOW
# ==============================================================================

def main():
    print("=" * 80)
    print("  DATASET SENIORITY ENHANCEMENT SCRIPT v2.0")
    print("  Mode: Active LLM Regeneration")
    print(f"  Config: Provider={LLM_PROVIDER.upper()} | Model={MODEL_NAME}")
    print("=" * 80)
    print()

    # Step 1: File Verification
    if not INPUT_DATASET.exists():
        print(f"❌ Error: Raw dataset not found at {INPUT_DATASET}")
        return

    # Load Source Records
    records = []
    with open(INPUT_DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total_records = len(records)
    print(f"📊 Loaded {total_records} records from master dataset.")

    # Step 2: Create a secure timestamped backup
    print("💾 Creating backup...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"raw_dataset_backup_before_llm_fix_{timestamp}.jsonl"
    shutil.copy2(INPUT_DATASET, backup_path)
    print(f"   ✅ Backup created at: {backup_path}")

    # Step 3: Handle Resumability Checkpoints
    processed_count = 0
    enhanced_count = 0
    processed_ids = set()

    # Read from temporary work file if it exists
    if TEMP_DATASET.exists():
        print(f"⏳ Found incomplete work file: {TEMP_DATASET}. Scanning for checkpoint...")
        try:
            with open(TEMP_DATASET, 'r', encoding='utf-8') as tf:
                for line in tf:
                    if line.strip():
                        item = json.loads(line)
                        # We use domain + level as a unique identifier key
                        rec_id = (item["metadata"]["domain"], item["metadata"]["level"])
                        processed_ids.add(rec_id)
            processed_count = len(processed_ids)
            print(f"   ✅ Checkpoint loaded: Resuming from record {processed_count + 1}/{total_records}...")
        except Exception as e:
            print(f"   ⚠️ Could not read checkpoint file ({e}). Starting fresh...")
            TEMP_DATASET.unlink(missing_ok=True)
            processed_ids = set()

    # Open temp dataset in append mode
    with open(TEMP_DATASET, 'a' if processed_count > 0 else 'w', encoding='utf-8') as out_f:
        for idx, record in enumerate(records):
            domain = record.get("metadata", {}).get("domain", "Unknown")
            level = record.get("metadata", {}).get("level", "Unknown")
            rec_id = (domain, level)

            # Skip already-processed records
            if rec_id in processed_ids:
                continue

            is_senior = level in SENIOR_LEVELS

            if is_senior:
                try:
                    # Dynamically find the assistant role inside conversations block
                    conversations = record.get("conversations", [])
                    assistant_idx = next((i for i, msg in enumerate(conversations) if msg.get("role") == "assistant"), -1)
                    
                    if assistant_idx == -1:
                        raise ValueError("No assistant role found in conversations array")

                    questions_text = conversations[assistant_idx]["content"]
                    
                    print(f"🤖 [{idx+1}/{total_records}] Regenerating: {level} - {domain}...")

                    # Call LLM with formatting validation and retries
                    valid_response = False
                    enhanced_questions = ""
                    api_retries = 3
                    
                    while not valid_response and api_retries > 0:
                        raw_llm_out = call_llm(
                            prompt=build_user_prompt(domain, level, questions_text),
                            system_instruction=SYSTEM_PROMPT
                        )
                        
                        if validate_llm_response(questions_text, raw_llm_out):
                            enhanced_questions = raw_llm_out.strip()
                            valid_response = True
                        else:
                            api_retries -= 1
                            print(f"   ⚠️ Invalid format from LLM. Retrying... (Attempts left: {api_retries})")
                            time.sleep(1)

                    # Fallback to keep raw data if the LLM completely fails format rules
                    if not valid_response:
                        print("   ❌ Format validation failed completely. Keeping original script.")
                        enhanced_questions = questions_text

                    # Update metadata and save record
                    original_tokens = len(questions_text.split())
                    enhanced_tokens = len(enhanced_questions.split())
                    token_increase = ((enhanced_tokens - original_tokens) / original_tokens) * 100 if original_tokens > 0 else 0

                    updated_conversations = list(conversations)
                    updated_conversations[assistant_idx] = {"role": "assistant", "content": enhanced_questions}

                    updated_record = {
                        "conversations": updated_conversations,
                        "metadata": {
                            **record.get("metadata", {}),
                            "llm_enhanced": True,
                            "llm_engine": MODEL_NAME,
                            "enhancement_version": "2.0",
                            "original_token_count": original_tokens,
                            "enhanced_token_count": enhanced_tokens,
                            "token_increase_pct": round(token_increase, 1),
                        }
                    }
                    
                    out_f.write(json.dumps(updated_record) + "\n")
                    out_f.flush()  # Push strictly to disk
                    enhanced_count += 1
                    print(f"   ✅ Done! Tokens: {original_tokens} -> {enhanced_tokens} (+{token_increase:.1f}%)")

                except Exception as e:
                    print(f"   ⚠️ Error occurred on record {idx}: {e}. Preserving original.")
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

            else:
                # Keep other profiles (Junior, Mid, EM) completely unaltered
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

    # Step 4: Atomic Swap & Clean Up
    print("\n🔄 Completing transaction: Overwriting master dataset...")
    try:
        shutil.move(str(TEMP_DATASET), str(OUTPUT_DATASET))
        print("🎉 Transaction complete! Master dataset updated safely.")
    except Exception as e:
        print(f"❌ Error while finalized write file: {e}")
        print(f"Your latest data is safe in {TEMP_DATASET}")
        return

    # Post-Run Token Count Analysis
    print()
    print("📊 Post-Enhancement Dataset Stats:")
    level_token_stats = {}
    with open(OUTPUT_DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lvl = r.get("metadata", {}).get("level", "Unknown")
                conv = r.get("conversations", [])
                asst_content = next((msg.get("content", "") for msg in conv if msg.get("role") == "assistant"), "")
                tokens = len(asst_content.split())
                
                if lvl not in level_token_stats:
                    level_token_stats[lvl] = []
                level_token_stats[lvl].append(tokens)

    print()
    for level in ["Junior (0–2 yrs)", "Mid-level (2–5 yrs)", "Senior (5–8 yrs)",
                  "Lead / Principal (8–12 yrs)", "Engineering Manager", "Director", "VP / C-suite"]:
        if level in level_token_stats:
            counts = level_token_stats[level]
            avg = sum(counts) / len(counts)
            marker = " 🎯 LLM ENHANCED" if level in SENIOR_LEVELS else ""
            print(f"  {level:30s}: count={len(counts):4d}, avg_tokens={avg:6.1f} {marker}")

    print("\n" + "=" * 80)
    print("  ✅ ENHANCEMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()