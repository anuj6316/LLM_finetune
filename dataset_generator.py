"""
================================================================================
  JD → 20 Interview Questions | Dataset Generator  v3.1
  Target : Gemma 4 E4B fine-tune via Unsloth (QLoRA)
  Format : ShareGPT JSONL  (Unsloth-native, production-ready)

  v3.1 updates
  ────────────
  ▸ Centralized CONFIG  — No external JSON required.
  ▸ Robustness          — Native KeyboardInterrupt handling + API retries.
  ▸ Performance         — Pre-compiled regexes & fixed L3 version tracking.
================================================================================
"""

import json, random, time, hashlib, re
from pathlib import Path
from collections import defaultdict, Counter
import litellm
litellm.set_verbose = False


# ══════════════════════════════════════════════════════════════════════════════
# 0.  PATHS & TUNEABLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH     = "checkpoint.json"
CHECKPOINT_EVERY    = 25        # save state every N accepted samples
MAX_STEM_REUSE      = 8         # L2: same question stem allowed this many times
MAX_STALE_PER_BATCH = 3         # L2: reject sample if > N stems are stale
JACCARD_THRESHOLD   = 0.60      # L3: within-sample near-duplicate ceiling
VERSION             = "3.1"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DIVERSITY MATRIX   14 × 20 × 7 × 4 = 7 840 unique combos
# ══════════════════════════════════════════════════════════════════════════════

INDUSTRIES = [
    "Software / SaaS", "FinTech / Payments", "Healthcare / MedTech",
    "E-commerce / D2C", "EdTech",
    "BFSI (Banking, Financial Services, Insurance)",
    "Manufacturing / Industry 4.0", "Consulting / Professional Services",
    "Media / Entertainment", "Logistics / Supply Chain",
    "Real Estate / PropTech", "Cybersecurity",
    "Gaming / Metaverse", "AgriTech / CleanTech",
]

DOMAINS = [
    "Backend Engineering", "Frontend Engineering", "Full Stack Engineering",
    "Mobile Engineering (iOS / Android)", "Data Science",
    "Machine Learning / AI Engineering", "MLOps / LLMOps",
    "DevOps / SRE / Platform Engineering", "Data Engineering",
    "QA / Automation Engineering", "Cloud Architecture",
    "Product Management", "UI / UX Design",
    "Technical Program Management", "Sales (B2B / Enterprise)",
    "Digital Marketing / Growth", "HR / People Operations",
    "Finance / FP&A", "Business Analyst", "Customer Success",
]

LEVELS = [
    "Junior (0–2 yrs)", "Mid-level (2–5 yrs)", "Senior (5–8 yrs)",
    "Lead / Principal (8–12 yrs)", "Engineering Manager",
    "Director", "VP / C-suite",
]

COMPANY_SIZES = [
    "Early-stage Startup (10–50 employees)",
    "Growth Startup (50–200 employees)",
    "Mid-size Company (200–1,000 employees)",
    "Large Enterprise (1,000+ employees)",
]

Q_DIST = {          # question type → required count
    "[Technical]":   7,
    "[Behavioral]":  5,
    "[Situational]": 4,
    "[Culture]":     2,
    "[Career]":      2,
}

STOP_WORDS = {
    "a","an","the","you","your","how","what","why","when","where","which",
    "who","in","of","for","to","and","or","that","this","at","by","from",
    "as","is","was","are","were","have","has","had","do","did","would",
    "could","should","can","will","be","been","being","with","it","if",
    "tell","me","about","time","situation","describe","experience","imagine",
    "approach","handling","faced","dealt","encountered","example","give",
    "share","walk","through","discuss","explain","using","use","used",
}

# Pre-compiled Regexes for Performance
STEM_PAT  = re.compile(r"^\d+\.\s*")
LABEL_PAT = re.compile(r"\[(?:Technical|Behavioral|Situational|Culture|Career)\]\s*")
WORD_PAT  = re.compile(r"[a-z0-9]+") # Includes numbers for proper version tracking (e.g. React 18)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert technical recruiter and hiring manager with 15+ years of "
    "experience across multiple industries. Given a job description, you generate "
    "targeted, insight-driven interview questions that help hiring teams assess "
    "both technical competence and cultural fit. Your questions are always "
    "specific to the role — never generic."
)

JD_PROMPT = """\
Generate a realistic Job Description for the role below.

Domain       : {domain}
Industry     : {industry}
Level        : {level}
Company size : {company_size}

Required sections:
1. Company Overview       (2–3 sentences; generic name: "a leading {industry} company")
2. About the Role         (3–4 sentences on ownership and impact)
3. Key Responsibilities   (6–8 action-verb bullet points)
4. Required Skills        (5–7 bullets — use real tool names, versions, frameworks)
5. Nice-to-Have Skills    (3–4 bullets)
6. What We Offer          (2–3 bullets, sized for {company_size})

Constraints:
- Skills must be hyper-specific (e.g. "FastAPI 0.110+", "dbt Core", "React 18")
- Length 350–500 words
- Return ONLY the JD text, no preamble
"""

Q_PROMPT = """\
Given the Job Description below, generate exactly 20 interview questions.

JD START
{jd}
JD END

Distribution (strict):
  7 × [Technical]   — test tools / skills explicitly listed in the JD
  5 × [Behavioral]  — start: "Tell me about a time…" or "Describe a situation…"
  4 × [Situational] — start: "Imagine…" or "How would you approach…"
  2 × [Culture]     — values, collaboration, team dynamics
  2 × [Career]      — motivation, growth, why this role

Rules:
- Every question must reference something concrete from the JD
- No question may repeat a concept already covered by another
- Behavioral prompts must name a realistic scenario for THIS role
- Questions should vary in length (8–35 words)

Format (no blank lines, no preamble):
1. [Technical] …?
2. [Behavioral] Tell me about a time …?
…
20. [Career] …?
"""


# ══════════════════════════════════════════════════════════════════════════════
# 3.  REPETITION TRACKER   (3 levels of deduplication)
# ══════════════════════════════════════════════════════════════════════════════

class RepetitionTracker:

    def __init__(self):
        self.jd_fps: set          = set()          # L1
        self.stem_counter: Counter = Counter()      # L2

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _jd_fp(jd: str) -> str:
        return hashlib.md5(jd[:300].encode("utf-8")).hexdigest()

    @staticmethod
    def _questions(text: str) -> list[str]:
        return [l.strip() for l in text.splitlines()
                if l.strip() and l.strip()[0].isdigit()]

    @staticmethod
    def _stem(line: str) -> str:
        t = STEM_PAT.sub("", line)
        t = LABEL_PAT.sub("", t)
        return t[:70].lower().strip()

    @staticmethod
    def _content_words(text: str) -> set:
        return {w for w in WORD_PAT.findall(text.lower()) if w not in STOP_WORDS}

    # ── L1: exact JD duplicate ───────────────────────────────────────────────

    def is_dup_jd(self, jd: str) -> bool:
        return self._jd_fp(jd) in self.jd_fps

    def register_jd(self, jd: str):
        self.jd_fps.add(self._jd_fp(jd))

    # ── L2: cross-sample question staleness ──────────────────────────────────

    def check_cross_sample(self, qtext: str) -> tuple[bool, str]:
        stems  = [self._stem(q) for q in self._questions(qtext)]
        stale  = sum(1 for s in stems if self.stem_counter[s] >= MAX_STEM_REUSE)
        if stale > MAX_STALE_PER_BATCH:
            return False, f"L2: {stale} overused stems (max {MAX_STALE_PER_BATCH})"
        return True, "OK"

    def register_questions(self, qtext: str):
        for q in self._questions(qtext):
            self.stem_counter[self._stem(q)] += 1

    # ── L3: within-sample near-duplicate ─────────────────────────────────────

    def check_within_sample(self, qtext: str) -> tuple[bool, str]:
        qs   = self._questions(qtext)
        sets = [self._content_words(q) for q in qs]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                if not a or not b:
                    continue
                j_score = len(a & b) / len(a | b)
                if j_score > JACCARD_THRESHOLD:
                    return False, f"L3: Q{i+1}≈Q{j+1} (J={j_score:.2f})"
        return True, "OK"

    # ── serialise / deserialise for checkpoint ───────────────────────────────

    def to_dict(self) -> dict:
        return {
            "jd_fps":       list(self.jd_fps),
            "stem_counter": dict(self.stem_counter),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RepetitionTracker":
        t = cls()
        t.jd_fps       = set(d.get("jd_fps", []))
        t.stem_counter = Counter(d.get("stem_counter", {}))
        return t


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CHECKPOINT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, path: str = CHECKPOINT_PATH):
        self.path = Path(path)

    def save(self, state: dict):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.rename(self.path)           # atomic on POSIX

    def load(self) -> dict | None:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return None

    def clear(self):
        self.path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class DatasetGenerator:

    def __init__(self):
        self.checkpoint = CheckpointManager()
        self.rep        = RepetitionTracker()
        self.stats      = defaultdict(int)
        self.failed_log: list[dict[str, str]] = []

    # ── LLM call (LiteLLM) ───────────────────────────────────────────────────

    def _call(self, prompt: str, cfg: dict, max_tokens: int) -> str:
        kwargs = dict(
            model    = cfg["model"],
            messages = [{"role": "user", "content": prompt}],
            max_tokens = max_tokens,
            temperature = 0.85,
        )
        if cfg.get("api_base"):
            kwargs["api_base"] = cfg["api_base"]
            
            # Inject dummy API key if routing to custom openAI endpoints
            # otherwise LiteLLM might raise an authentication validation error
            if kwargs["model"].startswith("openai/"):
                kwargs["api_key"] = "dummy"
                
        # 3-Attempt exponential backoff retry loop
        for attempt in range(3):
            try:
                r = litellm.completion(**kwargs)
                return r.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(2 ** attempt)

    def _gen_jd(self, domain, industry, level, size, cfg) -> str:
        return self._call(
            JD_PROMPT.format(domain=domain, industry=industry, level=level, company_size=size),
            cfg, max_tokens=950,
        )

    def _gen_questions(self, jd: str, cfg) -> str:
        return self._call(Q_PROMPT.format(jd=jd), cfg, max_tokens=1600)

    # ── validation ───────────────────────────────────────────────────────────

    @staticmethod
    def _validate_dist(qtext: str) -> tuple[bool, str]:
        lines    = [l.strip() for l in qtext.splitlines() if l.strip()]
        numbered = [l for l in lines if l and l[0].isdigit()]
        if not (18 <= len(numbered) <= 22):
            return False, f"count={len(numbered)}"
        for label, exp in Q_DIST.items():
            got = qtext.count(label)
            if abs(got - exp) > 1:
                return False, f"{label}={got} (want {exp})"
        return True, "OK"

    # ── quality score (0–100) ─────────────────────────────────────────────────

    def _quality_score(self, jd: str, qtext: str) -> int:
        score = 0

        # distribution compliance  (30 pts)
        ok, _ = self._validate_dist(qtext)
        if ok:
            score += 30

        # no within-sample repetition  (25 pts)
        ok, _ = self.rep.check_within_sample(qtext)
        if ok:
            score += 25

        # question specificity — each question shares ≥2 content words with JD  (25 pts)
        jd_words = RepetitionTracker._content_words(jd)
        qs = RepetitionTracker._questions(qtext)
        specific = sum(
            1 for q in qs
            if len(RepetitionTracker._content_words(q) & jd_words) >= 2
        )
        score += int(25 * specific / max(len(qs), 1))

        # question length variety  (20 pts)
        lengths = [len(q.split()) for q in qs]
        if lengths:
            avg = sum(lengths) / len(lengths)
            if 10 <= avg <= 30:
                score += 20
            elif 7 <= avg <= 35:
                score += 10

        return score

    # ── output format ────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(jd: str, qtext: str, meta: dict) -> dict:
        return {
            "conversations": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"Generate 20 interview questions for the following job description:\n\n{jd}"},
                {"role": "assistant", "content": qtext},
            ],
            "metadata": meta,
        }

    # ── checkpoint helpers ───────────────────────────────────────────────────

    def _ckpt(self, generated, combo_idx, output_path, target, seed):
        self.checkpoint.save({
            "generated":       generated,
            "combo_index":     combo_idx,
            "repetition":      self.rep.to_dict(),
            "failed_log":      self.failed_log,
            "output_path":     output_path,
            "target_size":     target,
            "shuffle_seed":    seed,
        })
        print(f"  💾  Checkpoint → {generated} samples saved.")

    # ── main generation loop ─────────────────────────────────────────────────

    def generate(
        self,
        target:      int   = 2500,
        output_path: str   = "raw_dataset.jsonl",
        sleep:       float = 1,
        seed:        int   = 42,
        jd_model:    str   = "anthropic/claude-3-haiku-20240307",
        jd_api_base: str | None = None,
        q_model:     str   = "anthropic/claude-3-haiku-20240307",
        q_api_base:  str | None = None,
    ) -> int:

        # build deterministic combo order
        combos = [
            (ind, dom, lvl, sz)
            for ind in INDUSTRIES
            for dom in DOMAINS
            for lvl in LEVELS
            for sz  in COMPANY_SIZES
        ]
        random.seed(seed)
        random.shuffle(combos)

        # ── resume? ──────────────────────────────────────────────────────────
        generated  = 0
        start_idx  = 0
        file_mode  = "w"

        ckpt = self.checkpoint.load()
        if ckpt:
            print(f"\n  ⏸  Checkpoint: {ckpt['generated']} / {ckpt['target_size']} done.")
            ans = input("     Resume? [y/n]: ").strip().lower()
            if ans == "y":
                generated        = ckpt["generated"]
                start_idx        = ckpt["combo_index"]
                self.rep         = RepetitionTracker.from_dict(ckpt["repetition"])
                self.failed_log  = ckpt.get("failed_log", [])
                file_mode        = "a"
                print(f"\n  ▶  Resuming from sample {generated}, combo {start_idx}\n")
            else:
                self.checkpoint.clear()
                print("  🔄  Starting fresh.\n")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combo_idx = start_idx

        # Local Configs based on params
        jd_cfg = {"model": jd_model, "api_base": jd_api_base}
        q_cfg  = {"model": q_model,  "api_base": q_api_base}

        with open(output_path, file_mode, encoding="utf-8") as fout:
            while generated < target and combo_idx < len(combos) * 2:
                try:
                    ind, dom, lvl, sz = combos[combo_idx % len(combos)]
                    combo_idx += 1

                    # 1 ── generate JD
                    jd = self._gen_jd(dom, ind, lvl, sz, jd_cfg)
                    time.sleep(sleep)

                    # 2 ── L1 JD dedup
                    if self.rep.is_dup_jd(jd):
                        self.stats["l1_dup"] += 1
                        continue
                    self.rep.register_jd(jd)

                    # 3 ── generate questions
                    qtext = self._gen_questions(jd, q_cfg)
                    time.sleep(sleep)

                    # 4 ── validate distribution
                    ok, reason = self._validate_dist(qtext)
                    if not ok:
                        self.stats["dist_fail"] += 1
                        self.failed_log.append({"reason": reason, "dom": dom})
                        continue

                    # 5 ── L2 cross-sample staleness
                    ok, reason = self.rep.check_cross_sample(qtext)
                    if not ok:
                        self.stats["l2_stale"] += 1
                        continue

                    # 6 ── L3 within-sample near-duplicate
                    ok, reason = self.rep.check_within_sample(qtext)
                    if not ok:
                        self.stats["l3_neardup"] += 1
                        self.failed_log.append({"reason": reason, "dom": dom})
                        continue

                    # 7 ── register questions (update stem counter)
                    self.rep.register_questions(qtext)

                    # 8 ── quality score
                    qscore = self._quality_score(jd, qtext)

                    # 9 ── write
                    meta = {
                        "domain": dom, "industry": ind,
                        "level": lvl, "company_size": sz,
                        "quality_score": qscore,
                        "q_counts": {lbl: qtext.count(lbl) for lbl in Q_DIST},
                        "version": VERSION,
                    }
                    fout.write(json.dumps(self._fmt(jd, qtext, meta), ensure_ascii=False) + "\n")
                    fout.flush()

                    generated += 1
                    self.stats[f"ind:{ind}"] += 1
                    self.stats[f"dom:{dom}"] += 1
                    self.stats[f"lvl:{lvl}"] += 1

                    print(
                        f"  ✅  [{generated:>4}/{target}]  "
                        f"Q={qscore:>3}  {dom:<35}  {ind:<30}  {lvl}"
                    )

                    if generated % CHECKPOINT_EVERY == 0:
                        self._ckpt(generated, combo_idx, output_path, target, seed)

                except KeyboardInterrupt:
                    print("\n\n  ⏸  KeyboardInterrupt caught. Saving checkpoint...")
                    self._ckpt(generated, combo_idx, output_path, target, seed)
                    return generated
                except Exception as exc:
                    self.stats["api_err"] += 1
                    self.failed_log.append({"reason": str(exc), "dom": dom})
                    print(f"  ❌  {dom} / {ind}: {exc}")
                    time.sleep(2)

        self.checkpoint.clear()
        print(f"\n{'─'*60}")
        print(f"  Generated : {generated}")
        print(f"  L1 dups   : {self.stats['l1_dup']}")
        print(f"  L2 stale  : {self.stats['l2_stale']}")
        print(f"  L3 neardup: {self.stats['l3_neardup']}")
        print(f"  API errors: {self.stats['api_err']}")
        print(f"{'─'*60}\n")
        return generated


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse(path: str):
    data = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    total        = len(data)
    label_counts = defaultdict(int)
    scores       = []
    word_lens    = []

    for item in data:
        ans = item["conversations"][2]["content"]
        word_lens.append(len(ans.split()))
        scores.append(item["metadata"].get("quality_score", 0))
        for lbl in Q_DIST:
            label_counts[lbl] += ans.count(lbl)

    avg_q  = sum(scores) / total if total else 0
    avg_wl = sum(word_lens) // total if total else 0

    print(f"\n{'═'*55}")
    print(f"  Samples         : {total}")
    print(f"  Avg quality     : {avg_q:.1f} / 100")
    print(f"  Avg output words: {avg_wl}")
    print(f"\n  Question-type distribution:")
    for lbl, exp in Q_DIST.items():
        avg = label_counts[lbl] / total
        ok  = "✅" if abs(avg - exp) < 0.5 else "⚠️ "
        print(f"    {ok} {lbl:<15} avg {avg:.2f}  (target {exp})")
    print(f"{'═'*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SPLIT  80 / 10 / 10
# ══════════════════════════════════════════════════════════════════════════════

def split(
    src:   str = "raw_dataset.jsonl",
    seed:  int = 42,
    train: float = 0.80,
    val:   float = 0.10,
    # test is remainder
):
    random.seed(seed)
    data = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    random.shuffle(data)

    n      = len(data)
    t_cut  = int(n * train)
    v_cut  = t_cut + int(n * val)

    splits = {
        "train.jsonl": data[:t_cut],
        "val.jsonl":   data[t_cut:v_cut],
        "test.jsonl":  data[v_cut:],
    }
    for fname, subset in splits.items():
        with open(fname, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {fname:<15} {len(subset):>5} samples")

    _write_card(n, splits)


def _write_card(total: int, splits: dict):
    card = f"""\
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

Generated with `dataset_generator.py` v{VERSION}.

## Stats

| Split | Samples |
|-------|---------|
| Train | {len(splits['train.jsonl'])} |
| Val   | {len(splits['val.jsonl'])} |
| Test  | {len(splits['test.jsonl'])} |
| **Total** | **{total}** |

## Format

ShareGPT JSONL — Unsloth / SFTTrainer native.
Each record has `conversations` (system · user · assistant) and `metadata`.

## Coverage

- **14 industries** · **20 domains** · **7 levels** · **4 company sizes**
- Question types: [Technical] [Behavioral] [Situational] [Culture] [Career]
- 3-level repetition filtering (JD dedup · cross-sample staleness · within-sample Jaccard)

## Intended use

Fine-tuning Gemma 4 E4B via Unsloth QLoRA for automated interview-question generation.
"""
    Path("README.md").write_text(card, encoding="utf-8")
    print("  README.md        dataset card written")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    CONFIG = {
        "target_samples": 2500,
        "output_path":    "raw_dataset.jsonl",
        "sleep_time":     0.6,
        "seed":           42,
        "train_split":    0.80,
        "val_split":      0.10,
        
        # ── Model Configuration ──
        # Updated to route to your local instance.
        # Note: litellm requires custom openAI endpoints to have the "openai/" prefix.
        # "local-model" can be replaced by the exact model name hosted on your server if required.
        "jd_model":       "openai/local-model",
        "jd_api_base":    "http://172.16.20.85:5174/v1/",  # /v1 is required by most local inference servers 
        
        "q_model":        "openai/local-model",
        # http://172.16.20.85:5174/v1/chat/completions
        "q_api_base":     "http://172.16.20.85:5174/v1/",
    }

    gen = DatasetGenerator()

    print(f"\n🚀 Starting generation: targeting {CONFIG['target_samples']} samples...")
    print(f"   JD Model: {CONFIG['jd_model']} | Q Model: {CONFIG['q_model']}\n")
    print(f"   Connecting to API Base: {CONFIG['jd_api_base']}\n")

    gen.generate(
        target      = CONFIG["target_samples"],
        output_path = CONFIG["output_path"],
        sleep       = CONFIG["sleep_time"],
        seed        = CONFIG["seed"],
        jd_model    = CONFIG["jd_model"],
        jd_api_base = CONFIG["jd_api_base"],
        q_model     = CONFIG["q_model"],
        q_api_base  = CONFIG["q_api_base"],
    )

    analyse(CONFIG["output_path"])

    split(
        src   = CONFIG["output_path"],
        seed  = CONFIG["seed"],
        train = CONFIG["train_split"],
        val   = CONFIG["val_split"],
    )

    print("✅  train.jsonl · val.jsonl · test.jsonl · README.md ready.")