"""
================================================================================
  JD → 20 Interview Questions | Dataset Generator  v3.3
  Target : Gemma 4 E4B fine-tune via Unsloth (QLoRA)
  Format : ShareGPT JSONL  (Unsloth-native, production-ready)

  v3.3 updates
  ────────────
  ▸ Level-Aware Prompts — Dynamically changes question depth based on seniority.
  ▸ Token Fix Engine    — Prevents uniform token distributions at generation time.
  ▸ Centralized CONFIG  — No external JSON required.
  ▸ Robustness          — Native KeyboardInterrupt handling + API retries.
================================================================================
"""

import asyncio, json, random, time, hashlib, re
from pathlib import Path
from collections import defaultdict, Counter
import litellm

try:
    import yaml
except ImportError:
    yaml = None  # config.yml loading will be skipped gracefully

litellm.set_verbose = False
litellm.suppress_debug_info = True


# ══════════════════════════════════════════════════════════════════════════════
# 0a. CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str = "config.yml") -> dict:
    """Load config.yml if present and PyYAML is installed, else return {}."""
    if yaml is None:
        print("  ⚠️  PyYAML not installed — using built-in defaults. Run: pip install pyyaml")
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"  ⚠️  {path} not found — using built-in defaults.")
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    print(f"  ✅  Loaded config from {path}")
    return data


_CFG = load_config()


# ══════════════════════════════════════════════════════════════════════════════
# 0b. PATHS & TUNEABLE CONSTANTS  (driven by config.yml, with defaults)
# ══════════════════════════════════════════════════════════════════════════════

_paths   = _CFG.get("paths",   {})
_quality = _CFG.get("quality", {})
_gen     = _CFG.get("generation", {})
_retry   = _CFG.get("models",  {}).get("retry", {})

CHECKPOINT_PATH     = _paths.get("checkpoint",        "checkpoint.json")
CHECKPOINT_EVERY    = _gen.get("checkpoint_every",    25)
MAX_STEM_REUSE      = _quality.get("max_stem_reuse",  8)
MAX_STALE_PER_BATCH = _quality.get("max_stale_per_batch", 3)
JACCARD_THRESHOLD   = _quality.get("jaccard_threshold",   0.60)
MAX_RETRY_ATTEMPTS  = _retry.get("max_attempts",      3)
BACKOFF_BASE        = _retry.get("backoff_base",       2)
VERSION             = "3.3"


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

_qdist_cfg = _CFG.get("question_distribution", {})
Q_DIST = {
    "[Technical]":   _qdist_cfg.get("Technical",   7),
    "[Behavioral]":  _qdist_cfg.get("Behavioral",  5),
    "[Situational]": _qdist_cfg.get("Situational", 4),
    "[Culture]":     _qdist_cfg.get("Culture",      2),
    "[Career]":      _qdist_cfg.get("Career",       2),
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

STEM_PAT  = re.compile(r"^\d+\.\s*")
LABEL_PAT = re.compile(r"\[(?:Technical|Behavioral|Situational|Culture|Career)\]\s*")
WORD_PAT  = re.compile(r"[a-z0-9]+") 


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DYNAMIC PROMPT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an elite executive technical recruiter and hiring architect with 15+ "
    "years of experience. Given a job description, you generate targeted, "
    "highly tailored interview questions that exactly match the expected maturity, "
    "scope, and scale of the target seniority level."
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

def build_q_prompt(jd: str, level: str, domain: str) -> str:
    """Dynamically builds a prompt that enforces appropriate complexity based on seniority."""
    
    # Define level-specific directives to break uniform token length patterns
    if level in ["Director", "VP / C-suite"]:
        complexity_directive = """\
CRITICAL DIRECTION FOR EXECUTIVE/SENIOR LEVEL:
- Your questions must reflect organizational strategy, multi-year engineering roadmaps, technical debt mitigation, and business alignment.
- Technical questions must focus on macro architectural governance, platform scaling trade-offs, and financial implications of technical decisions.
- Behavioral/Situational questions must deal with high-stakes organizational ambiguity, complex cross-functional alignment, executive influence, and structural crisis management.
- Ensure questions are deep, complex, and highly realistic, leading to longer, more detailed text profiles."""
    elif level in ["Lead / Principal (8–12 yrs)", "Engineering Manager"]:
        complexity_directive = """\
CRITICAL DIRECTION FOR LEAD/MANAGEMENT LEVEL:
- Focus on systems design, mentoring, code health governance, execution velocity, and delivery.
- Technical queries should target system bottlenecks, architectural trade-offs, and technical direction.
- Behavioral queries should target managing low performance, resolving engineering conflicts, and balancing execution speed with quality."""
    else:
        complexity_directive = """\
CRITICAL DIRECTION FOR JUNIOR/MID LEVEL:
- Focus heavily on core engineering capabilities, execution patterns, clean code principles, and implementation syntax of specified frameworks.
- Keep questions crisp, directly evaluating individual contribution and tactical execution."""

    return f"""\
Given the Job Description below, generate exactly 20 interview questions.

JD START
{jd}
JD END

Target Seniority Level: {level}
Target Domain: {domain}

{complexity_directive}

Distribution (strict):
  7 × [Technical]   — test tools / skills explicitly listed in the JD
  5 × [Behavioral]  — start: "Tell me about a time…" or "Describe a situation…"
  4 × [Situational] — start: "Imagine…" or "How would you approach…"
  2 × [Culture]     — values, collaboration, team dynamics
  2 × [Career]      — motivation, growth, why this role

Rules:
- Every question must reference something concrete from the JD
- No question may repeat a concept already covered by another
- Questions should vary in length based on the complexity required for this seniority level.

Format (no blank lines, no preamble):
1. [Technical] …?
2. [Behavioral] Tell me about a time …?
…
20. [Career] …?
"""


# ══════════════════════════════════════════════════════════════════════════════
# 3.  REPETITION TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class RepetitionTracker:
    def __init__(self):
        self.jd_fps: set          = set()          
        self.stem_counter: Counter = Counter()      

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

    def is_dup_jd(self, jd: str) -> bool:
        return self._jd_fp(jd) in self.jd_fps

    def register_jd(self, jd: str):
        self.jd_fps.add(self._jd_fp(jd))

    def check_cross_sample(self, qtext: str) -> tuple[bool, str]:
        stems  = [self._stem(q) for q in self._questions(qtext)]
        stale  = sum(1 for s in stems if self.stem_counter[s] >= MAX_STEM_REUSE)
        if stale > MAX_STALE_PER_BATCH:
            return False, f"L2: {stale} overused stems (max {MAX_STALE_PER_BATCH})"
        return True, "OK"

    def register_questions(self, qtext: str):
        for q in self._questions(qtext):
            self.stem_counter[self._stem(q)] += 1

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
        tmp.rename(self.path)           

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

    async def _acall(self, prompt: str, cfg: dict, max_tokens: int) -> str:
        kwargs = dict(
            model       = cfg["model"],
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = cfg.get("max_tokens", max_tokens),
            temperature = cfg.get("temperature", 0.85),
        )
        if cfg.get("api_base"):
            kwargs["api_base"] = cfg["api_base"]
            if kwargs["model"].startswith("openai/"):
                kwargs["api_key"] = "dummy"

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                async with self.semaphore:
                    r = await litellm.acompletion(**kwargs)
                content = r.choices[0].message.content
                if content is None:
                    raise ValueError(f"LLM returned None content")
                return content.strip()
            except Exception as e:
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(BACKOFF_BASE ** attempt)

    async def _gen_jd(self, domain, industry, level, size, cfg) -> str:
        return await self._acall(
            JD_PROMPT.format(domain=domain, industry=industry, level=level, company_size=size),
            cfg, max_tokens=950,
        )

    async def _gen_questions(self, jd: str, level: str, domain: str, cfg) -> str:
        # Dynamically formats the prompt injecting level-specific logic
        prompt = build_q_prompt(jd, level, domain)
        return await self._acall(prompt, cfg, max_tokens=2048)

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

    def _quality_score(self, jd: str, qtext: str) -> int:
        score = 0
        ok, _ = self._validate_dist(qtext)
        if ok: score += 30

        ok, _ = self.rep.check_within_sample(qtext)
        if ok: score += 25

        jd_words = RepetitionTracker._content_words(jd)
        qs = RepetitionTracker._questions(qtext)
        specific = sum(1 for q in qs if len(RepetitionTracker._content_words(q) & jd_words) >= 2)
        score += int(25 * specific / max(len(qs), 1))

        lengths = [len(q.split()) for q in qs]
        if lengths:
            avg = sum(lengths) / len(lengths)
            if 10 <= avg <= 35:
                score += 20
            elif 7 <= avg <= 45:
                score += 10
        return score

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

    def _ckpt(self, generated, started, output_path, target, seed):
        self.checkpoint.save({
            "generated":       generated,
            "started":         started,
            "repetition":      self.rep.to_dict(),
            "failed_log":      self.failed_log,
            "output_path":     output_path,
            "target_size":     target,
            "shuffle_seed":    seed,
        })
        print(f"  💾  Checkpoint → {generated} samples saved.")

    async def generate(
        self, target: int = 2500, output_path: str = "raw_dataset.jsonl",
        sleep: float = 1, seed: int = 42,
        jd_model: str = "anthropic/claude-3-haiku-20240307", jd_api_base: str | None = None,
        q_model: str = "anthropic/claude-3-haiku-20240307", q_api_base: str | None = None,
        concurrency: int = 4,
    ) -> int:

        combos = [
            (ind, dom, lvl, sz)
            for ind in INDUSTRIES for dom in DOMAINS for lvl in LEVELS for sz in COMPANY_SIZES
        ]
        random.seed(seed)
        random.shuffle(combos)

        generated  = 0
        start_idx  = 0
        file_mode  = "w"

        ckpt = self.checkpoint.load()
        if ckpt:
            print(f"\n  ⏸  Checkpoint: {ckpt['generated']} / {ckpt['target_size']} done.")
            ans = input("     Resume? [y/n]: ").strip().lower()
            if ans == "y":
                file_lines = 0
                if Path(output_path).exists():
                    with open(output_path, encoding="utf-8") as f:
                        file_lines = sum(1 for _ in f if _.strip())
                generated        = max(ckpt["generated"], file_lines)
                start_idx        = ckpt.get("started", ckpt.get("combo_index", 0))
                self.rep         = RepetitionTracker.from_dict(ckpt["repetition"])
                self.failed_log  = ckpt.get("failed_log", [])
                file_mode        = "a"
                print(f"\n  ▶  Resuming from sample {generated}, combo {start_idx}\n")
            else:
                self.checkpoint.clear()
                print("  🔄  Starting fresh.\n")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        jd_cfg = {"model": jd_model, "api_base": jd_api_base}
        q_cfg  = {"model": q_model,  "api_base": q_api_base}

        self.semaphore = asyncio.Semaphore(concurrency)
        index_lock = asyncio.Lock()
        write_lock = asyncio.Lock()
        current_idx = start_idx
        max_combos = len(combos) * 2

        async def worker():
            nonlocal current_idx, generated
            while generated < target:
                async with index_lock:
                    if current_idx >= max_combos: break
                    idx = current_idx
                    current_idx += 1

                ind, dom, lvl, sz = combos[idx % len(combos)]
                try:
                    jd = await self._gen_jd(dom, ind, lvl, sz, jd_cfg)

                    if self.rep.is_dup_jd(jd):
                        self.stats["l1_dup"] += 1
                        continue
                    self.rep.register_jd(jd)

                    # FIXED: Added lvl and dom context routing to prompt builder
                    qtext = await self._gen_questions(jd, lvl, dom, q_cfg)

                    ok, reason = self._validate_dist(qtext)
                    if not ok:
                        self.stats["dist_fail"] += 1
                        self.failed_log.append({"reason": reason, "dom": dom})
                        continue

                    ok, reason = self.rep.check_cross_sample(qtext)
                    if not ok:
                        self.stats["l2_stale"] += 1
                        continue

                    ok, reason = self.rep.check_within_sample(qtext)
                    if not ok:
                        self.stats["l3_neardup"] += 1
                        self.failed_log.append({"reason": reason, "dom": dom})
                        continue

                    self.rep.register_questions(qtext)
                    qscore = self._quality_score(jd, qtext)

                    meta = {
                        "domain": dom, "industry": ind,
                        "level": lvl, "company_size": sz,
                        "quality_score": qscore,
                        "q_counts": {lbl: qtext.count(lbl) for lbl in Q_DIST},
                        "version": VERSION,
                    }
                    line = json.dumps(self._fmt(jd, qtext, meta), ensure_ascii=False)

                    async with write_lock:
                        fout.write(line + "\n")
                        fout.flush()
                        generated += 1
                        self.stats[f"ind:{ind}"] += 1
                        self.stats[f"dom:{dom}"] += 1
                        self.stats[f"lvl:{lvl}"] += 1
                        print(f"  ✅  [{generated:>4}/{target}]  Q={qscore:>3}  {dom:<35}  {ind:<30}  {lvl}")
                        if generated % CHECKPOINT_EVERY == 0:
                            self._ckpt(generated, current_idx, output_path, target, seed)

                except Exception as exc:
                    self.stats["api_err"] += 1
                    self.failed_log.append({"reason": str(exc), "dom": dom})
                    print(f"  ❌  {dom} / {ind}: {exc}")

        with open(output_path, file_mode, encoding="utf-8") as fout:
            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            try:
                await asyncio.gather(*workers)
            except asyncio.CancelledError:
                for w in workers: w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                self._ckpt(generated, current_idx, output_path, target, seed)
                print(f"  ⏸  KeyboardInterrupt — checkpoint saved.\n")
                return generated

        self.checkpoint.clear()
        print(f"\n{'─'*60}\n  Generated : {generated}\n  L1 dups   : {self.stats['l1_dup']}\n  L2 stale  : {self.stats['l2_stale']}\n  L3 neardup: {self.stats['l3_neardup']}\n  API errors: {self.stats['api_err']}\n{'─'*60}\n")
        return generated


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ANALYSIS & SPLIT (Kept fully compatible)
# ══════════════════════════════════════════════════════════════════════════════

def analyse(path: str):
    data = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    total = len(data)
    label_counts = defaultdict(int)
    scores, word_lens = [], []

    for item in data:
        ans = item["conversations"][2]["content"]
        word_lens.append(len(ans.split()))
        scores.append(item["metadata"].get("quality_score", 0))
        for lbl in Q_DIST: label_counts[lbl] += ans.count(lbl)

    print(f"\n{'═'*55}\n  Samples         : {total}\n  Avg quality     : {sum(scores)/total if total else 0:.1f} / 100\n  Avg output words: {sum(word_lens)//total if total else 0}\n\n  Question-type distribution:")
    for lbl, exp in Q_DIST.items():
        avg = label_counts[lbl] / total
        print(f"    {'✅' if abs(avg - exp) < 0.5 else '⚠️ '} {lbl:<15} avg {avg:.2f}  (target {exp})")
    print(f"{'═'*55}\n")

def split(src: str = "raw_dataset.jsonl", seed: int = 42, train: float = 0.80, val: float = 0.10):
    random.seed(seed)
    data = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    random.shuffle(data)
    n = len(data)
    t_cut, v_cut = int(n * train), int(n * train) + int(n * val)
    splits = {"train.jsonl": data[:t_cut], "val.jsonl": data[t_cut:v_cut], "test.jsonl": data[v_cut:]}
    src_dir = Path(src).parent
    for fname, subset in splits.items():
        with open(src_dir / fname, "w", encoding="utf-8") as f:
            for item in subset: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    _write_card(n, splits, src_dir)

def _write_card(total: int, splits: dict, src_dir: Path):
    card = f"---\nlanguage: en\nlicense: apache-2.0\ntask_categories:\n  - text-generation\ntags:\n  - interview-questions\n---\n# JD → Interview Questions Dataset\nGenerated with v{VERSION}.\n## Stats\n| Split | Samples |\n|---|---|\n| Train | {len(splits['train.jsonl'])} |\n| Val | {len(splits['val.jsonl'])} |\n| Test | {len(splits['test.jsonl'])} |\n| **Total** | **{total}** |\n"
    Path(src_dir / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    _m = _CFG.get("models", {})
    _jd = _m.get("jd_model", {})
    _q = _m.get("q_model", {})
    _gen = _CFG.get("generation", {})
    _spl = _CFG.get("split", {})

    CONFIG = {
        "target_samples": _gen.get("target_samples", 2500),
        "output_path":    _gen.get("output_path",    "raw_dataset.jsonl"),
        "sleep_time":     _gen.get("sleep_time",     0.6),
        "seed":           _gen.get("seed",           42),
        "concurrency":    _gen.get("concurrency",    4),
        "train_split":    _spl.get("train", 0.80),
        "val_split":      _spl.get("val",   0.10),
        "jd_model":       _jd.get("name",     "openai/local-model"),
        "jd_api_base":    _jd.get("api_base", "http://172.16.20.85:5174/v1/"),
        "q_model":        _q.get("name",      "openai/local-model"),
        "q_api_base":     _q.get("api_base",  "http://172.16.20.85:5174/v1/"),
    }

    gen = DatasetGenerator()
    print(f"\n🚀 Starting generation: targeting {CONFIG['target_samples']} samples...\n   JD Model : {CONFIG['jd_model']}\n   Q  Model : {CONFIG['q_model']}\n")
    
    asyncio.run(gen.generate(
        target=CONFIG["target_samples"], output_path=CONFIG["output_path"],
        sleep=CONFIG["sleep_time"], seed=CONFIG["seed"],
        jd_model=CONFIG["jd_model"], jd_api_base=CONFIG["jd_api_base"],
        q_model=CONFIG["q_model"], q_api_base=CONFIG["q_api_base"],
        concurrency=CONFIG["concurrency"],
    ))

    analyse(CONFIG["output_path"])
    split(src=CONFIG["output_path"], seed=CONFIG["seed"], train=CONFIG["train_split"], val=CONFIG["val_split"])