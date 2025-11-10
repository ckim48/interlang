# app.py
import os
import json
import random
import sqlite3
import re, hashlib, time
import requests  # pip install requests
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash


from flask import (
    Flask, render_template, render_template_string,
    g, session, redirect, url_for, request, flash, jsonify
)

# ====== GPT client (optional; graceful fallback) ======
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# ==== CareerQuest weekly unlock helpers ====
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

KST = ZoneInfo("Asia/Seoul") if ZoneInfo else None

def _parse_sqlite_ts(ts: str) -> datetime:
    """
    Parses common SQLite TIMESTAMP formats into timezone-aware KST datetime.
    Accepts 'YYYY-MM-DD HH:MM:SS' or ISO-like strings.
    """
    if not ts:
        return datetime.now(tz=KST)
    fmt_candidates = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f")
    for fmt in fmt_candidates:
        try:
            dt = datetime.strptime(ts.split("+")[0], fmt)
            return dt.replace(tzinfo=KST) if KST else dt
        except Exception:
            pass
    # Fallback
    return datetime.now(tz=KST)

def compute_week_unlocks(created_at: str, weeks: int = 4):
    """
    Returns list of dicts:
      [{"week":1, "unlock_at": datetime, "is_unlocked": bool}, ...]
    Rule: Week 1 unlocks at creation time; each next week every +7 days.
    """
    base = _parse_sqlite_ts(created_at)
    now = datetime.now(tz=KST) if KST else datetime.now()
    states = []
    for w in range(1, weeks + 1):
        unlock_at = base + timedelta(days=(w - 1) * 7)
        states.append({
            "week": w,
            "unlock_at": unlock_at,
            "is_unlocked": now >= unlock_at
        })
    return states

def _fmt_kst(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M") if KST else dt.strftime("%Y-%m-%d %H:%M")

def get_openai():
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
except Exception:
    openai_client = None

# -------------------- Flask setup --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace_me_dev")
DB_PATH = os.environ.get("DB_PATH", "app.db")
app.permanent_session_lifetime = timedelta(days=30)

# -------------------- Achievements registry --------------------
# (store icons using Bootstrap Icons names)
ACHIEVEMENTS = [
    {
        "code": "FIRST_STEPS",
        "name": "First Steps",
        "description": "Complete your first board.",
        "icon": "bi-rocket-takeoff",
    },
    {
        "code": "SHARP_SHOOTER",
        "name": "Sharpshooter",
        "description": "Get 12 or more cells correct in a single board.",
        "icon": "bi-bullseye",
    },
    {
        "code": "PERFECTIONIST",
        "name": "Perfectionist",
        "description": "Get all 16 cells correct in a single board.",
        "icon": "bi-stars",
    },
    {
        "code": "LINE_TRIO",
        "name": "Line Trio",
        "description": "Complete 3 or more lines in a single board.",
        "icon": "bi-grid-3x3-gap",
    },
    {
        "code": "BINGO_MASTER",
        "name": "Bingo Master",
        "description": "Complete 6 or more lines in a single board.",
        "icon": "bi-trophy",
    },
    {
        "code": "WORDSMITH",
        "name": "Wordsmith",
        "description": "Answer at least one long-answer cell correctly in a board.",
        "icon": "bi-pencil-square",
    },
]

# Titles (auto-assigned from best earned achievement)
TITLES = {
    "PERFECTIONIST": {"title": "The Flawless", "icon": "bi-stars", "priority": 1},
    "BINGO_MASTER": {"title": "Line Lord",   "icon": "bi-trophy", "priority": 2},
    "SHARP_SHOOTER": {"title": "Hot Streak",  "icon": "bi-fire",   "priority": 3},
    "WORDSMITH":     {"title": "Wordsmith",   "icon": "bi-pen",    "priority": 4},
    "FIRST_STEPS":   {"title": "Newcomer",    "icon": "bi-rocket-takeoff", "priority": 5},
}
DEFAULT_TITLE = {"title": None, "icon": None, "priority": 999}
def _norm_prompt(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def title_priority(code):
    return TITLES.get(code, DEFAULT_TITLE)["priority"]

# -------------------- DB helpers --------------------
# -------------------- DB helpers --------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(
            DB_PATH,
            timeout=30,
            check_same_thread=False
        )
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA busy_timeout = 30000;")
        g.db.execute("PRAGMA journal_mode = WAL;")
        g.db.execute("PRAGMA synchronous = NORMAL;")
        g.db.execute("PRAGMA foreign_keys = ON;")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db:
        db.close()

def ensure_column(db, table, column, coldef):
    cols = [r["name"] for r in db.execute(f'PRAGMA table_info("{table}")').fetchall()]
    if column not in cols:
        db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {coldef}')
def dict_lookup_en_ko(word: str):
    """
    Tries: cache -> MyMemory (free) -> dictionaryapi.dev (EN def only).
    Returns (translation_ko, definition_en). Both may be None.
    """
    db = get_db()
    word = (word or "").strip().lower()
    if not word:
        return None, None

    # cache
    row = db.execute("SELECT translation, definition FROM dict_cache WHERE word=? AND (lang='en-ko' OR lang IS NULL)", (word,)).fetchone()
    if row and row["translation"] and row["definition"]:
        return row["translation"], row["definition"]

    trans_ko, defin_en = None, None

    # MyMemory (free; quality varies)
    try:
        r = requests.get("https://api.mymemory.translated.net/get", params={"q": word, "langpair": "en|ko"}, timeout=5)
        if r.ok:
            data = r.json()
            trans_ko = (data.get("responseData", {}) or {}).get("translatedText")
    except Exception:
        pass

    # English definition (dictionaryapi.dev)
    try:
        r = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        if r.ok:
            j = r.json()
            if isinstance(j, list) and j:
                meanings = j[0].get("meanings", [])
                if meanings:
                    defs = meanings[0].get("definitions", [])
                    if defs:
                        defin_en = defs[0].get("definition")
    except Exception:
        pass

    # Save cache (best effort)
    try:
        db.execute("INSERT OR REPLACE INTO dict_cache(word, lang, translation, definition, updated_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP)",
                   (word, "en-ko", trans_ko or "", defin_en or ""))
        db.commit()
    except Exception:
        pass

    return trans_ko, defin_en
def _stable_seed_from_args(path: str, q, category, theme, age, difficulty):
    """
    Build a deterministic daily seed from the current filters + path.
    Ensures random order stays stable for the day (and per filter set).
    """
    today = datetime.date.today().isoformat()
    key = f"{path}|{q}|{category}|{theme}|{age}|{difficulty}|{today}"
    # Stable 32-bit integer from SHA1
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF

@app.route("/stories", methods=["GET"])
def stories():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    # user header (unchanged)
    u = db.execute("SELECT level, xp, title_code FROM users WHERE username=?", (username,)).fetchone()
    user_level = u["level"] if u else 1
    user_xp    = u["xp"] if u else 0
    title_info = TITLES.get((u["title_code"] or ""), DEFAULT_TITLE)
    display_title = title_info["title"]
    title_icon    = title_info["icon"]

    # filters
    q        = (request.args.get("q") or "").strip()
    category = (request.args.get("category") or "").strip()
    theme    = (request.args.get("theme") or "").strip()
    try:
        age = int(request.args.get("age")) if request.args.get("age") else None
    except:
        age = None
    try:
        difficulty = int(request.args.get("difficulty")) if request.args.get("difficulty") else None
    except:
        difficulty = None

    # NEW: optional "ready only" toggle
    ready_only = (request.args.get("ready") == "1")

    # pagination inputs
    try:
        page = max(1, int(request.args.get("page", 1)))
    except:
        page = 1
    PAGE_SIZE = 12
    MAX_PAGES = 5

    # fetch all matches (no limit), then ensure records exist
    items_all = recommend_stories(
        age=age, difficulty=difficulty, q=q or None,
        category=category or None, theme=theme or None, limit=None
    )
    for s in items_all:
        try:
            ensure_story_record(s["slug"], s)
        except Exception:
            pass

    # Identify which are already generated (content non-empty)
    generated = set()
    slugs = tuple(s["slug"] for s in items_all)
    if slugs:
        qmarks = ",".join(["?"] * len(slugs))
        rows = db.execute(
            f"""
            SELECT slug FROM stories
            WHERE slug IN ({qmarks})
              AND COALESCE(NULLIF(content,''), '') <> ''
            """,
            slugs
        ).fetchall()
        generated = {r["slug"] for r in rows}

    # If user asked for "ready only", filter here
    if ready_only:
        items_all = [s for s in items_all if s["slug"] in generated]

    # ✅ NEW: Randomize ordering, but generated first
    from datetime import date
    import random

    gen_items   = [s for s in items_all if s["slug"] in generated]
    other_items = [s for s in items_all if s["slug"] not in generated]

    # stable seed
    seed_str = f"{date.today().isoformat()}|{username}|{q}|{category}|{theme}|{age}|{difficulty}"
    rng = random.Random(seed_str)

    rng.shuffle(gen_items)
    rng.shuffle(other_items)

    items_all = gen_items + other_items

    # paginate with a hard cap of 5 pages
    total_items = len(items_all)
    total_pages = min(MAX_PAGES, max(1, (total_items + PAGE_SIZE - 1) // PAGE_SIZE))
    page = min(page, total_pages)
    start = (page - 1) * PAGE_SIZE
    end   = start + PAGE_SIZE
    items = items_all[start:end]

    return render_template(
        "stories.html",
        username=username,
        user_level=user_level,
        user_xp=user_xp,
        display_title=display_title,
        title_icon=title_icon,
        items=items,
        q=q, category=category, theme=theme, age=age, difficulty=difficulty,
        page=page, total_pages=total_pages, page_size=PAGE_SIZE, total_items=total_items,
        generated_slugs=generated,
        ready_only=ready_only
    )

def seed_achievements(db):
    # tables
    db.execute("""
        CREATE TABLE IF NOT EXISTS achievements (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            icon TEXT NOT NULL
        );
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS user_achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            code TEXT NOT NULL,
            granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, code),
            FOREIGN KEY (code) REFERENCES achievements(code) ON DELETE CASCADE
        );
    """)

    # seed static registry
    for a in ACHIEVEMENTS:
        db.execute(
            "INSERT OR IGNORE INTO achievements(code,name,description,icon) VALUES(?,?,?,?)",
            (a["code"], a["name"], a["description"], a["icon"])
        )
    db.commit()

def init_db(seed=True):
    db = get_db()
    ensure_achievement_progress_schema(db)
    try:
        db.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass

    # ---------- Core user / quiz tables ----------
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            level INTEGER NOT NULL DEFAULT 1,
            xp INTEGER NOT NULL DEFAULT 0,
            title_code TEXT
        );
    """)

    # legacy MCQ pool (kept for backward compatibility)
    db.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER NOT NULL,
            question TEXT NOT NULL,
            choice_a TEXT NOT NULL,
            choice_b TEXT NOT NULL,
            choice_c TEXT NOT NULL,
            choice_d TEXT NOT NULL,
            correct TEXT NOT NULL
        );
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            level INTEGER NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            lines_completed INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            level INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            qtype TEXT NOT NULL,                 -- 'mcq' | 'short' | 'long'
            prompt TEXT NOT NULL,
            choice_a TEXT,
            choice_b TEXT,
            choice_c TEXT,
            choice_d TEXT,
            correct TEXT,                        -- mcq: 'A'..'D'; short: exact string; long: JSON list
            extra TEXT,                          -- JSON (pos, sentence, required_words, etc.)
            FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE
        );
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS quiz_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            response TEXT NOT NULL,
            is_correct INTEGER NOT NULL,         -- 0/1
            awarded_xp INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES quiz_questions(id) ON DELETE CASCADE
        );
    """)

    # ---------- Achievements (registry + user map) ----------
    # (tables + seeding happen here)
    def ensure_column(db, table, column, coldef):
        cols = [r["name"] for r in db.execute(f'PRAGMA table_info("{table}")').fetchall()]
        if column not in cols:
            db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {coldef}')

    def seed_achievements(db):
        db.execute("""
            CREATE TABLE IF NOT EXISTS achievements (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                icon TEXT NOT NULL
            );
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS user_achievements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                code TEXT NOT NULL,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, code),
                FOREIGN KEY (code) REFERENCES achievements(code) ON DELETE CASCADE
            );
        """)
        # pull from in-memory registry ACHIEVEMENTS if available
        try:
            for a in ACHIEVEMENTS:
                db.execute(
                    "INSERT OR IGNORE INTO achievements(code,name,description,icon) VALUES(?,?,?,?)",
                    (a["code"], a["name"], a["description"], a["icon"])
                )
        except Exception:
            # no registry available; skip seeding
            pass

    seed_achievements(db)

    # ---------- Stories / Vocabulary / Dictionary cache ----------
    db.execute("""
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            title TEXT,
            author TEXT,
            age_min INTEGER,
            age_max INTEGER,
            difficulty INTEGER,               -- 1..5
            categories TEXT,                  -- JSON
            themes TEXT,                      -- JSON
            provider TEXT,
            provider_id TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS story_reads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            story_id INTEGER NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP,
            words_learned INTEGER DEFAULT 0,
            FOREIGN KEY(story_id) REFERENCES stories(id) ON DELETE CASCADE
        );
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            word TEXT NOT NULL,
            translation TEXT,
            definition TEXT,
            source_story_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            review_count INTEGER DEFAULT 0,
            last_reviewed_at TIMESTAMP,
            UNIQUE(username, word, source_story_id),
            FOREIGN KEY(source_story_id) REFERENCES stories(id) ON DELETE SET NULL
        );
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS dict_cache (
            word TEXT PRIMARY KEY,
            lang TEXT,
            translation TEXT,
            definition TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # ---------- Migrations / safety (adds if missing) ----------
    ensure_column(db, "users", "xp", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(db, "users", "title_code", "TEXT")
    ensure_column(db, "attempts", "lines_completed", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(db, "users", "password_hash", "TEXT")
    ensure_column(db, "vocab", "bookmarked", "INTEGER NOT NULL DEFAULT 0")

    db.commit()

# -------------------- Local word pools --------------------
WORD_POOLS = {
    1: {
        "nouns": ["cat","dog","book","car","apple","school","water","song","room","park"],
        "verbs": ["run","eat","read","play","see","go","like","make","have","be"],
        "adjectives": ["big","small","happy","sad","red","fast","slow","hot","cold","nice"],
        "adverbs": ["quickly","slowly","happily","sadly","well","badly","now","then","here","there"],
        "preps": ["in","on","at","to","from","with","for","by","under","over"],
        "synonyms": {
            "big": ["large","huge","giant","enormous"],
            "small":["tiny","little","mini","petite"],
            "happy":["glad","joyful","cheerful","pleased"],
            "sad":["unhappy","down","blue","gloomy"]
        }
    },
    2: {
        "nouns": ["letter","story","teacher","friend","city","river","movie","market","phone","music"],
        "verbs": ["study","travel","watch","listen","write","build","open","close","help","learn"],
        "adjectives": ["bright","dark","tasty","boring","funny","clever","kind","strong","weak","loud"],
        "adverbs": ["often","always","usually","sometimes","rarely","quietly","carefully","easily","hardly","nearly"],
        "preps": ["between","among","above","below","through","across","into","onto","about","around"],
        "synonyms": {
            "funny": ["amusing","humorous","comic","witty"],
            "clever":["smart","bright","intelligent","sharp"],
            "tasty":["delicious","yummy","flavorful","savory"]
        }
    },
    3: {
        "nouns": ["evidence","choice","pattern","message","service","purpose","energy","history","camera","traffic"],
        "verbs": ["improve","discuss","explain","consider","prefer","create","discover","develop","deliver","reduce"],
        "adjectives": ["efficient","reliable","basic","complex","modern","ancient","typical","unique","public","private"],
        "adverbs": ["particularly","probably","clearly","simply","nearly","roughly","exactly","seriously","carefully","actively"],
        "preps": ["despite","during","within","without","beyond","against","toward","regarding","concerning","except"],
        "synonyms": {
            "improve": ["enhance","boost","refine","upgrade"],
            "reduce":  ["decrease","lower","cut","diminish"],
            "reliable":["dependable","trustworthy","steady","consistent"]
        }
    },
    4: {
        "nouns": ["strategy","concept","context","analysis","approach","argument","impact","network","feature","version"],
        "verbs": ["evaluate","integrate","optimize","justify","illustrate","assess","facilitate","implement","sustain","accelerate"],
        "adjectives": ["significant","adequate","robust","feasible","efficient","comprehensive","subtle","explicit","implicit","coherent"],
        "adverbs": ["substantially","consequently","ultimately","precisely","approximately","notably","deliberately","radically","readily","readily"],
        "preps": ["along","via","throughout","upon","amid","per","versus","plus","minus","barring"],
        "synonyms": {
            "optimize":["enhance","streamline","improve","refine"],
            "robust":  ["strong","resilient","solid","sturdy"],
            "coherent":["logical","consistent","rational","orderly"]
        }
    },
    5: {
        "nouns": ["hypothesis","paradigm","capability","constraint","phenomenon","criterion","equation","mechanism","consensus","inference"],
        "verbs": ["substantiate","calibrate","synthesize","extrapolate","corroborate","differentiate","articulate","approximate","validate","generalize"],
        "adjectives": ["systematic","inherent","marginal","ambiguous","explicit","implicit","probabilistic","deterministic","orthogonal","auxiliary"],
        "adverbs": ["theoretically","empirically","nominally","implicitly","explicitly","marginally","approximately","consequently","paradoxically","necessarily"],
        "preps": ["notwithstanding","circa","versus","hence","thereby","therein","thereof","whereas","wherein","per"],
        "synonyms": {
            "validate":["confirm","verify","corroborate","authenticate"],
            "synthesize":["combine","integrate","fuse","merge"],
            "ambiguous":["unclear","vague","equivocal","uncertain"]
        }
    }
}
# ==== UPDATED: gpt_generate_quiz (adds stronger system prompt + sanitization) ====
LEVEL_CONCEPTS = {
    1: "word",
    2: "phrase",
    3: "sentence",
    4: "structure",
    5: "mini_composition"
}

def concept_for_level(level: int) -> str:
    return LEVEL_CONCEPTS.get(int(level), "sentence")
import math

def _token_set(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z']{2,}", (s or "").lower()))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def recent_prompt_set(username: str, lookback_days: int = 14, max_prompts: int = 500) -> tuple[set[str], list[tuple[str,set[str]]]]:
    """
    Returns (norm_prompts, tokenized_prompts) from the user's recent quizzes.
    tokenized_prompts: list of (original_prompt, token_set) for fuzzy checks.
    """
    db = get_db()
    rows = db.execute(
        """
        SELECT qq.prompt
          FROM quiz_questions qq
          JOIN quizzes q ON q.id = qq.quiz_id
         WHERE q.username = ?
           AND q.created_at >= DATETIME('now', ?)
         ORDER BY q.created_at DESC
         LIMIT ?
        """,
        (username, f"-{lookback_days} days", max_prompts)
    ).fetchall()
    prompts = [r["prompt"] for r in rows if r["prompt"]]
    norm = {_norm_prompt(p) for p in prompts}
    toks = [(p, _token_set(p)) for p in prompts]
    return norm, toks

def is_too_similar(prompt: str, recent_tok: list[tuple[str,set[str]]], thr: float = 0.8) -> bool:
    ts = _token_set(prompt)
    for _, t in recent_tok:
        if jaccard(ts, t) >= thr:
            return True
    return False
def gpt_generate_quiz(level: int) -> dict:
    """
    LLM-only generator that ALWAYS returns exactly 16 items (per distribution_for_level),
    or raises RuntimeError. No local templating or padding — every item comes from GPT.

    Updates:
      • LONG: Prompt must explicitly list required_words, e.g.
        "Write one short sentence using these words: cat, run, fast."
      • SHORT: Must be truly single-answer. Answer is a single token (letters/digits only),
        no synonyms, alternates must be [].
    """
    import os, json, re
    from typing import Any, Dict, List

    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    TEMPERATURE = 0.4
    MAX_FULL_ATTEMPTS  = 3
    MAX_PATCH_ATTEMPTS = 6

    try:
        mcq_need, short_need, long_need = distribution_for_level(level)
    except Exception:
        mcq_need, short_need, long_need = (10, 4, 2)
    total_need = int(mcq_need) + int(short_need) + int(long_need)
    if total_need != 16:
        raise RuntimeError(f"distribution_for_level({level}) must sum to 16, got {total_need}")

    # ------- light guidance bank (never fabricate locally) -------
    DIFF_NAME = {1:"Beginner",2:"Novice",3:"Intermediate",4:"Advanced",5:"Expert"}
    def easy_bank():
        l1, l2 = WORD_POOLS.get(1, {}), WORD_POOLS.get(2, {})
        def pick(k): return list(dict.fromkeys((l1.get(k) or []) + (l2.get(k) or [])))
        return {"nouns":pick("nouns"),"verbs":pick("verbs"),"adjectives":pick("adjectives"),
                "adverbs":pick("adverbs"),"preps":pick("preps")}
    BANK = easy_bank() if level >= 2 else WORD_POOLS.get(level, WORD_POOLS.get(1, {}))
    diff_name = DIFF_NAME.get(int(level), "Intermediate")
    max_sentence_len = 6 if level <= 3 else 8

    # ------- small utils -------
    def _is_str(x): return isinstance(x, str) and x.strip() != ""
    def _norm(s): return (s or "").strip().lower()
    def extract_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text or "", flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return {}
        return {}

    # Single-token = letters/numbers only, no spaces/hyphens/slashes/commas.
    _single_token_re = re.compile(r"^[A-Za-z0-9]+$")

    def validate_block(data: Dict) -> Dict[str, List[Dict]]:
        """
        Accept only well-formed items. No synthesis here.
        LONG:
          - required_words length 3..4
          - prompt MUST explicitly list them in the text using phrase:
            'using these words: w1, w2, w3' (case-insensitive for checking)
        SHORT:
          - answer must be a single token (letters/digits only)
          - alternates forced to []
        """
        out = {"mcq": [], "short": [], "long": []}

        # MCQ
        for it in (data.get("mcq") or []):
            if not isinstance(it, dict): continue
            p, opts, ai = it.get("prompt",""), it.get("options",[]), it.get("answer_index",None)
            if not _is_str(p): continue
            if not (isinstance(opts, list) and len(opts) == 4 and all(_is_str(o) for o in opts)): continue
            if not (isinstance(ai, int) and 0 <= ai <= 3): continue
            out["mcq"].append({
                "prompt": p.strip(),
                "options": [str(o).strip() for o in opts],
                "answer_index": int(ai),
                "explanation": (it.get("explanation") or "").strip(),
                "hint": (it.get("hint") or "").strip()
            })

        # SHORT (enforce single-token answer, no alternates)
        for it in (data.get("short") or []):
            if not isinstance(it, dict): continue
            p, a = it.get("prompt",""), it.get("answer","")
            if not (_is_str(p) and _is_str(a)): continue
            a = a.strip()
            if not _single_token_re.fullmatch(a):  # reject multi-word or punctuated answers
                continue
            out["short"].append({
                "prompt": p.strip(),
                "answer": a,
                "alternates": [],  # force empty to keep one-and-only-one answer
                "explanation": (it.get("explanation") or "").strip(),
                "hint": (it.get("hint") or "").strip()
            })

        # LONG (explicit required words in prompt text)
        for it in (data.get("long") or []):
            if not isinstance(it, dict): continue
            req = (it.get("required_words",[]) or [])
            if not (isinstance(req, list) and 3 <= len(req) <= 4 and all(_is_str(w) for w in req)): continue
            req = [str(w).strip() for w in req[:4]]
            p = (it.get("prompt") or "").strip()

            # Prompt must explicitly show the list using a clear phrase
            # e.g., "Write one short sentence using these words: w1, w2, w3."
            ok_phrase = ("using these words:" in _norm(p))
            has_all = all(_norm(w) in _norm(p) for w in req)
            if not (_is_str(p) and ok_phrase and has_all):
                # If LLM didn't follow, reject this item
                continue

            out["long"].append({
                "prompt": p,
                "required_words": req,
                "explanation": (it.get("explanation") or "").strip(),
                "hint": (it.get("hint") or "").strip()
            })
        return out

    # ------- request payloads -------
    # JSON schema (guidance for the model)
    base_schema = {
        "type":"object","required":["mcq","short","long"],
        "properties":{
            "mcq":{"type":"array","items":{
                "type":"object","required":["prompt","options","answer_index","explanation","hint"],
                "properties":{
                    "prompt":{"type":"string"},
                    "options":{"type":"array","minItems":4,"maxItems":4,"items":{"type":"string"}},
                    "answer_index":{"type":"integer","minimum":0,"maximum":3},
                    "explanation":{"type":"string"},
                    "hint":{"type":"string"}}}},
            "short":{"type":"array","items":{
                "type":"object","required":["prompt","answer","alternates","explanation","hint"],
                "properties":{
                    "prompt":{"type":"string"},
                    "answer":{"type":"string"},
                    "alternates":{"type":"array","items":{"type":"string"}},
                    "explanation":{"type":"string"},
                    "hint":{"type":"string"}}}},
            "long":{"type":"array","items":{
                "type":"object","required":["prompt","required_words","hint"],
                "properties":{
                    "prompt":{"type":"string"},
                    "required_words":{"type":"array","minItems":3,"maxItems":4,"items":{"type":"string"}},
                    "hint":{"type":"string"}}}}
        }
    }

    sys_msg = (
        "You create quizzes for very young ESL learners (Pre-A1/A1). "
        "Return a SINGLE JSON object with keys mcq/short/long. No prose, no markdown."
    )

    # Shared audience rules
    audience_rules = {
        "cefr": "Pre-A1/A1",
        "avoid": ["passive voice","perfect tenses","conditionals","rare words","idioms"],
        "sentence_word_limit": max_sentence_len,
        "lowercase_single_words": True
    }

    # SHORT: restrict to unambiguous templates that yield a single-token answer.
    # The model MUST use one of these patterns.
    short_templates = [
        "Type the first letter of '{word}'. Answer is one letter.",
        "Type the last letter of '{word}'. Answer is one letter.",
        "Write the plural of '{regular_noun}'. Answer is one word (letters only).",
        "Type this number in words: {digit_0_to_10}. Answer is one word (letters only).",
        "Write the base form of this past tense: '{regular_past}'. Answer is one word (letters only)."
    ]

    long_rule_line = (
        "LONG items MUST use this exact prompt format: "
        "'Write one short sentence using these words: WORD1, WORD2, WORD3.' "
        "Use 3–4 very simple words from the word_bank. Do not add quotes around the list."
    )

    def payload_full():
        return {
            "task": "Generate English quiz items for a 4x4 Bingo board (TOTAL 16).",
            "hard_requirements": [
                "Return ONLY a JSON object with keys mcq/short/long. No markdown.",
                f"Produce EXACTLY mcq={mcq_need}, short={short_need}, long={long_need}.",
                "No placeholders, no empty strings, no duplicates.",
                "MCQ: exactly 4 options + valid answer_index (0..3).",
                # SHORT rules
                "SHORT items MUST be unambiguous and yield exactly ONE valid answer.",
                "SHORT answer MUST be a single token (letters/digits only, no spaces, hyphens, or punctuation).",
                "SHORT alternates MUST be an empty array [].",
                "SHORT prompts MUST use one of these patterns:",
                *short_templates,
                # LONG rules
                long_rule_line,
                "In every LONG prompt, explicitly show the required words using the phrase 'using these words:' "
                "followed by the comma-separated list. Those words MUST also appear in required_words.",
            ],
            "level": int(level),
            "difficulty_label": diff_name,
            "counts": {"mcq": mcq_need, "short": short_need, "long": long_need},
            "audience_rules": audience_rules,
            "word_bank": {
                "nouns": BANK.get("nouns", [])[:20],
                "verbs": BANK.get("verbs", [])[:20],
                "adjectives": BANK.get("adjectives", [])[:20],
                "adverbs": BANK.get("adverbs", [])[:20],
                "preps": BANK.get("preps", [])[:10],
            },
            "output_schema": base_schema
        }

    def payload_patch(missing: Dict[str,int], banlist_prompts: List[str]):
        return {
            "task": "PATCH: Provide ONLY the missing items to reach the exact totals.",
            "missing_counts": missing,
            "banlist_prompts": banlist_prompts,
            "instructions": [
                "Return ONLY a JSON object with mcq/short/long arrays.",
                "For any type with 0 missing, return an EMPTY array [].",
                "Every prompt must be NEW and must NOT be in banlist_prompts.",
                "No placeholders. No duplicates in this response.",
                # Re-assert critical constraints in PATCH
                "SHORT: single-token answer, alternates must be []. Use only the approved templates.",
                long_rule_line
            ],
            "audience_rules": audience_rules,
            "word_bank": {
                "nouns": BANK.get("nouns", [])[:20],
                "verbs": BANK.get("verbs", [])[:20],
                "adjectives": BANK.get("adjectives", [])[:20],
                "adverbs": BANK.get("adverbs", [])[:20],
                "preps": BANK.get("preps", [])[:10],
            },
            "output_schema": base_schema
        }

    def call_llm(payload: Dict) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            try:
                resp = client.chat.completions.create(
                    model=MODEL, temperature=TEMPERATURE,
                    response_format={"type":"json_object"},
                    messages=[{"role":"system","content":sys_msg},
                              {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=MODEL, temperature=TEMPERATURE,
                    messages=[{"role":"system","content":sys_msg},
                              {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
                )
            return resp.choices[0].message.content or ""
        except Exception:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            try:
                resp = openai.ChatCompletion.create(
                    model=MODEL, temperature=TEMPERATURE,
                    response_format={"type":"json_object"},
                    messages=[{"role":"system","content":sys_msg},
                              {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
                )
            except Exception:
                resp = openai.ChatCompletion.create(
                    model=MODEL, temperature=TEMPERATURE,
                    messages=[{"role":"system","content":sys_msg},
                              {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
                )
            return resp["choices"][0]["message"]["content"] or ""

    # ---- accumulate, track prompts to avoid during PATCH ----
    acc = {"mcq": [], "short": [], "long": []}
    seen_prompts_norm = set()

    def missing_counts():
        return {
            "mcq":  max(0, mcq_need   - len(acc["mcq"])),
            "short":max(0, short_need - len(acc["short"])),
            "long": max(0, long_need  - len(acc["long"]))
        }

    # Full attempts
    for a in range(1, MAX_FULL_ATTEMPTS+1):
        raw = call_llm(payload_full())
        data = extract_json(raw)
        block = validate_block(data)

        print(f"[GEN#{a}] got: mcq={len(block['mcq'])}, short={len(block['short'])}, long={len(block['long'])}")

        for t in ("mcq","short","long"):
            for it in block[t]:
                p = _norm(it.get("prompt"))
                if p and p not in seen_prompts_norm:
                    acc[t].append(it)
                    seen_prompts_norm.add(p)

        miss = missing_counts()
        if sum(miss.values()) == 0:
            acc["mcq"]   = acc["mcq"][:mcq_need]
            acc["short"] = acc["short"][:short_need]
            acc["long"]  = acc["long"] [:long_need]
            print(f"[GEN OK] Final: mcq={len(acc['mcq'])}, short={len(acc['short'])}, long={len(acc['long'])}")
            return acc

    # Patch attempts
    for pidx in range(1, MAX_PATCH_ATTEMPTS+1):
        miss = missing_counts()
        if sum(miss.values()) == 0:
            break
        banlist = list(seen_prompts_norm)
        raw = call_llm(payload_patch(miss, banlist))
        data = extract_json(raw)
        block = validate_block(data)

        # Enforce exact counts for each type in this PATCH
        for t in ("mcq","short","long"):
            need = miss[t]
            got  = len(block[t])
            if need == 0:
                block[t] = []
            elif got != need:
                print(f"[PATCH#{pidx}] type '{t}' expected {need} but got {got} → will retry")
                block[t] = []

        print(f"[PATCH#{pidx}] accepted: mcq={len(block['mcq'])}, short={len(block['short'])}, long={len(block['long'])}")

        for t in ("mcq","short","long"):
            for it in block[t]:
                p = _norm(it.get("prompt"))
                if p and p not in seen_prompts_norm:
                    acc[t].append(it)
                    seen_prompts_norm.add(p)

        if sum(missing_counts().values()) == 0:
            break

    miss = missing_counts()
    if sum(miss.values()) != 0:
        raise RuntimeError(f"LLM could not produce exactly 16 items. Missing -> {miss}")

    acc["mcq"]   = acc["mcq"][:mcq_need]
    acc["short"] = acc["short"][:short_need]
    acc["long"]  = acc["long"] [:long_need]
    print(f"[GEN OK] Final after PATCH: mcq={len(acc['mcq'])}, short={len(acc['short'])}, long={len(acc['long'])}")
    return acc


# ==== NEW HELPERS (place near your GPT functions) ====

def _is_ambiguous_mcq(item: dict) -> bool:
    """
    Heuristics to reject MCQs that aren't objectively right/wrong.
    Flags:
      - missing/invalid answer_index
      - fewer than 4 distinct options
      - vague 'best/most suitable' phrasing without a hard rule (e.g., not a preposition cloze)
      - 'adverb' fill-in MCQs (multiple can fit)
    """
    prompt = str(item.get("prompt", "")).lower()
    opts = item.get("options", [])
    ai = item.get("answer_index", None)

    # must have 4 distinct options and a valid answer index
    if not isinstance(opts, list) or len(opts) < 4 or len({(o or "").strip().lower() for o in opts}) < 4:
        return True
    if not isinstance(ai, int) or ai < 0 or ai > 3:
        return True

    # avoid vague “best/most suitable/sounds better”
    vague_words = ("best", "most suitable", "sounds better", "fits best")
    if any(v in prompt for v in vague_words):
        # allow if it's an objective grammar constraint (preposition cloze)
        if not (("preposition" in prompt) and ("blank" in prompt)):
            return True

    # avoid adverb-of-manner cloze MCQs (often multiple fit)
    if "adverb" in prompt and "fill in the blank" in prompt and "preposition" not in prompt:
        return True

    return False


def _downgrade_ambiguous_to_short(item: dict) -> dict:
    """
    Convert a shaky MCQ to a SHORT item when possible.
    Returns {} if conversion isn't safe.
    """
    prompt = str(item.get("prompt", "")).strip()
    opts = item.get("options", [])
    ai = item.get("answer_index", None)

    if not (isinstance(opts, list) and len(opts) >= 4 and isinstance(ai, int) and 0 <= ai < 4):
        return {}

    correct = str(opts[ai]).strip()
    alts = [str(o).strip() for i, o in enumerate(opts) if i != ai]

    # Only convert if it's clearly a cloze prompt
    if "blank" not in prompt.lower():
        return {}

    return {
        "qtype": "short",
        "prompt": prompt,
        "answer": correct,
        "alternates": alts[:3],
        "hint": "Choose the single best word for this exact sentence.",
        "explanation": "This word fits the sentence pattern; others are less precise here."
    }


def sanitize_gpt_quiz_payload(data: dict) -> dict:
    """
    Ensure objective MCQs; convert/drop ambiguous ones.
    De-duplicate options defensively.
    """
    out = {"mcq": [], "short": list(data.get("short", [])), "long": list(data.get("long", []))}
    mcqs = list(data.get("mcq", []))

    for item in mcqs:
        if _is_ambiguous_mcq(item):
            as_short = _downgrade_ambiguous_to_short(item)
            if as_short:
                out["short"].append(as_short)
            continue

        # clean duplicate options if any
        seen = set()
        cleaned = []
        for o in item.get("options", []):
            k = (o or "").strip()
            if k.lower() not in seen:
                cleaned.append(k)
                seen.add(k.lower())

        if len(cleaned) >= 4:
            item["options"] = cleaned[:4]
            out["mcq"].append(item)
        else:
            as_short = _downgrade_ambiguous_to_short(item)
            if as_short:
                out["short"].append(as_short)
            # else drop

    return out

def insert_gpt_short(db, quiz_id: int, pos: int, item: dict, used_norm_prompts: set) -> bool:
    """
    Insert a GPT-generated SHORT item.
    Supports optional alternates, hint, and explanation, all stored in `extra`.
    """
    try:
        prompt = str(item.get("prompt", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not prompt or not answer:
            return False

        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            return False

        alternates = []
        if isinstance(item.get("alternates"), list):
            alternates = [str(a).strip() for a in item["alternates"] if str(a).strip()]

        hint = str(item.get("hint") or "").strip() or None
        explanation = str(item.get("explanation") or "").strip() or None

        extra = {
            "mode": "gpt",
            "pos": pos,
            "alternates": alternates,
            "hint": hint,
            "explanation": explanation,
        }

        db.execute(
            """INSERT INTO quiz_questions (quiz_id,qtype,prompt,correct,extra)
               VALUES (?,?,?,?,?)""",
            (quiz_id, "short", prompt, answer, json.dumps(extra))
        )
        used_norm_prompts.add(n)
        return True
    except Exception:
        return False



def insert_gpt_long(db, quiz_id: int, pos: int, item: dict, used_norm_prompts: set) -> bool:
    """
    item = {"required_words": [str,...], "prompt": optional str}
    If 'prompt' is missing, we synthesize one from required_words.
    Returns True if inserted; False otherwise.
    """
    try:
        req = item.get("required_words", [])
        if not isinstance(req, list) or not req:
            return False

        prompt = (item.get("prompt") or f"Write a single clear sentence using ALL of these words: {', '.join(req)}.").strip()
        if not prompt:
            return False

        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            return False

        extra = {"required_words": req, "mode": "gpt", "pos": pos}
        db.execute(
            """INSERT INTO quiz_questions (quiz_id,qtype,prompt,correct,extra)
               VALUES (?,?,?,?,?)""",
            (quiz_id, "long", prompt, json.dumps(req), json.dumps(extra))
        )
        used_norm_prompts.add(n)
        return True
    except Exception:
        return False



    return False


# -------------------- Create 4×4 board --------------------
def distribution_for_level(level: int):
    if level <= 2:
        return (10, 4, 2)
    elif level == 3:
        return (8, 4, 4)
    else:
        return (6, 6, 4)
# ---------- Mock catalog helpers ----------
def make_slug(title: str, author: str = "") -> str:
    base = f"{(title or '').strip().lower()}|{(author or '').strip().lower()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

def build_mock_catalog(n=1000):
    random.seed(42)  # stable across restarts
    adjectives = [
        "Brave","Curious","Quiet","Shiny","Clever","Gentle","Merry","Swift","Tiny","Wandering",
        "Hidden","Lucky","Kind","Rusty","Sunny","Silver","Golden","Whistling","Giggly","Moonlit"
    ]
    nouns = [
        "Forest","Robot","River","Dragon","Library","Planet","Garden","Train","Pirate","Painter",
        "Notebook","Bridge","Lantern","Wizard","Explorer","Baker","Turtle","Meteor","Sparrow","Harbor"
    ]
    surnames = [
        "Lee","Kim","Park","Nguyen","Patel","Garcia","Brown","Carter","Ibrahim","Singh",
        "Johnson","Baker","Mori","Santos","Khan","Lopez","Davis","Jones","Chen","Li"
    ]
    categories_all = ["animals","adventure","friendship","mystery","science","sports","fairy tales","space","nature","school","family","magic"]
    themes_all     = ["bravery","kindness","teamwork","curiosity","perseverance","responsibility","imagination","problem-solving","change","confidence"]

    def age_for_diff(d):
        # approximate ranges by difficulty
        return {
            1: (6, 8),
            2: (8, 12),
            3: (10, 14),
            4: (12, 15),
            5: (13, 18),
        }.get(d, (8, 12))

    items = []

    # include the specific title you mentioned so search works as expected
    items.append({
        "slug": make_slug("Five Children and It", "Edith Nesbit"),
        "title": "Five Children and It",
        "author": "Edith Nesbit",
        "age_min": 9, "age_max": 13,
        "difficulty": 3,
        "categories": ["adventure","family"],
        "themes": ["curiosity","wishes"],
        "provider": "mock",
        "provider_id": "classic-five-children",
        "cover": None,
    })

    # synthesize the rest
    for i in range(n - 1):
        adj1, adj2 = random.choice(adjectives), random.choice(adjectives)
        noun1, noun2 = random.choice(nouns), random.choice(nouns)
        # several title patterns for variety
        pattern = random.choice([
            f"The {adj1} {noun1}",
            f"{adj1} {noun1} and the {adj2} {noun2}",
            f"A {adj1} {noun1} in {noun2} Town",
            f"The {noun1} of {noun2}",
            f"{adj1} Tales of the {noun1}",
        ])
        author = f"{random.choice(adjectives)} {random.choice(surnames)}"
        diff = 1 + (i % 5)
        a1, a2 = age_for_diff(diff)
        cats = random.sample(categories_all, k=2)
        ths  = random.sample(themes_all, k=2)
        items.append({
            "slug": make_slug(pattern, author),
            "title": pattern,
            "author": author,
            "age_min": a1, "age_max": a2,
            "difficulty": diff,
            "categories": cats,
            "themes": ths,
            "provider": "mock",
            "provider_id": f"mock-{i}",
            "cover": None,
        })
    return items

# Build once and reuse
MOCK_CATALOG = build_mock_catalog()
def create_board_quiz(username: str, level: int) -> int:
    """
    Create a new 4x4 board for `username` at `level`:
      1) Ask LLM for full set (via gpt_generate_quiz(level)).
      2) Strictly validate items.
      3) If any are invalid or counts are short, run PATCH calls to LLM
         to replace only the missing/invalid parts (NO local templating).
      4) Persist exactly 16 validated items with fixed positions (0..15).

    Raises RuntimeError if, after several retries, we still don't have valid 16 GPT items.
    """
    import os, json, re, hashlib, random
    from typing import Dict, List, Any

    db = get_db()

    # ---------- counts ----------
    try:
        need_mcq, need_short, need_long = distribution_for_level(level)
    except Exception:
        need_mcq, need_short, need_long = (10, 4, 2)
    if (need_mcq + need_short + need_long) != 16:
        raise RuntimeError("distribution_for_level(level) must sum to 16.")

    TOTAL = 16

    # ---------- helpers ----------
    def _is_str(x): return isinstance(x, str) and x.strip() != ""
    def _norm(s: str) -> str: return re.sub(r"\s+", " ", (s or "").strip().lower())

    def validate_block(block: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Return only valid items (no modification)."""
        out = {"mcq": [], "short": [], "long": []}

        # MCQ
        for it in (block.get("mcq") or []):
            if not isinstance(it, dict): continue
            prompt = it.get("prompt", "")
            options = it.get("options", [])
            ans_idx = it.get("answer_index", None)
            expl = (it.get("explanation") or "").strip()
            hint = (it.get("hint") or "").strip()
            if not _is_str(prompt): continue
            if not (isinstance(options, list) and len(options) == 4 and all(_is_str(o) for o in options)):
                continue
            if not (isinstance(ans_idx, int) and 0 <= ans_idx <= 3): continue
            out["mcq"].append({
                "prompt": prompt.strip(),
                "options": [str(o).strip() for o in options],
                "answer_index": int(ans_idx),
                "explanation": expl,
                "hint": hint
            })

        # SHORT
        for it in (block.get("short") or []):
            if not isinstance(it, dict): continue
            prompt = it.get("prompt", "")
            answer = it.get("answer", "")
            alts = it.get("alternates", []) or []
            expl = (it.get("explanation") or "").strip()
            hint = (it.get("hint") or "").strip()
            if not (_is_str(prompt) and _is_str(answer)): continue
            if not isinstance(alts, list) or any(not _is_str(a) for a in alts): alts = []
            out["short"].append({
                "prompt": prompt.strip(),
                "answer": answer.strip(),
                "alternates": [str(a).strip() for a in alts][:2],
                "explanation": expl,
                "hint": hint
            })

        # LONG (STRICT: require prompt AND 3–4 required_words)
        for it in (block.get("long") or []):
            if not isinstance(it, dict): continue
            prompt = it.get("prompt", "")
            req = it.get("required_words", []) or []
            hint = (it.get("hint") or "").strip()
            if not _is_str(prompt): continue
            if not (isinstance(req, list) and 3 <= len(req) <= 4 and all(_is_str(w) for w in req)): continue
            out["long"].append({
                "prompt": prompt.strip(),
                "required_words": [str(w).strip() for w in req[:4]],
                "explanation": "",
                "hint": hint
            })
        return out

    # Duplicate guards
    def sig_mcq(it: Dict) -> str:
        p = _norm(it.get("prompt", ""))
        opts = [ _norm(o) for o in (it.get("options") or []) ]
        return hashlib.md5(f"mcq|{p}|{','.join(sorted(opts))}".encode()).hexdigest()

    def sig_short(it: Dict) -> str:
        return hashlib.md5(f"short|{_norm(it.get('prompt',''))}|{_norm(it.get('answer',''))}".encode()).hexdigest()

    def sig_long(it: Dict) -> str:
        req = ",".join(sorted([_norm(w) for w in (it.get("required_words") or [])]))
        return hashlib.md5(f"long|{_norm(it.get('prompt',''))}|{req}".encode()).hexdigest()

    def dedup_merge(dest: Dict[str, List[Dict]], block: Dict[str, List[Dict]],
                    seen_prompts: set, seen_sigs: set):
        added = 0
        for t in ("mcq", "short", "long"):
            for it in block[t]:
                pn = _norm(it.get("prompt", ""))
                if not pn or pn in seen_prompts:  # prompt dup
                    continue
                if t == "mcq": s = sig_mcq(it)
                elif t == "short": s = sig_short(it)
                else: s = sig_long(it)
                if s in seen_sigs:  # structural dup
                    continue
                dest[t].append(it)
                seen_prompts.add(pn)
                seen_sigs.add(s)
                added += 1
        return added

    # LLM patcher (ONLY to fix missing pieces)
    def patch_missing(missing: Dict[str, int], banlist: List[str]) -> Dict[str, List[Dict]]:
        """
        Ask the LLM to return ONLY the requested counts, no extras, no prose.
        This relies on the same model/key your app already uses elsewhere.
        """
        payload = {
            "task": "PATCH: provide ONLY missing items to reach exact totals for a 4x4 Bingo (total 16).",
            "missing_counts": missing,
            "banlist_prompts": banlist,
            "hard_requirements": [
                "Return ONLY a JSON object with arrays mcq/short/long.",
                "For each requested type, return EXACTLY that many items.",
                "For types with 0 requested, return an EMPTY array [].",
                "All prompts must be NEW (not in banlist_prompts).",
                "No placeholders; all strings must be non-empty.",
                "MCQ: exactly 4 options, answer_index in 0..3.",
                "LONG: must include 'prompt' and 'required_words' (3–4 words)."
            ]
        }

        # Use the same client style as your gpt_generate_quiz
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            try:
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.4,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Return STRICT JSON. No prose, no markdown."},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                    ]
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.4,
                    messages=[
                        {"role": "system", "content": "Return STRICT JSON. No prose, no markdown."},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                    ]
                )
            txt = resp.choices[0].message.content or ""
        except Exception:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            try:
                resp = openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.4,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Return STRICT JSON. No prose, no markdown."},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                    ]
                )
            except Exception:
                resp = openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.4,
                    messages=[
                        {"role": "system", "content": "Return STRICT JSON. No prose, no markdown."},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                    ]
                )
            txt = resp["choices"][0]["message"]["content"] or ""

        # tolerant JSON extract
        def extract_json(text: str) -> Any:
            try:
                return json.loads(text)
            except Exception:
                m = re.search(r"\{(?:[^{}]|(?R))*\}", text or "", flags=re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return {}
                return {}

        return validate_block(extract_json(txt))

    # ---------- step 1: full generation ----------
    acc = {"mcq": [], "short": [], "long": []}
    seen_prompts, seen_sigs = set(), set()

    full = gpt_generate_quiz(level)  # your existing function
    block = validate_block(full)
    dedup_merge(acc, block, seen_prompts, seen_sigs)

    # ---------- step 2: patch missing (retry) ----------
    def missing_now():
        return {
            "mcq":  max(0, need_mcq  - len(acc["mcq"])),
            "short":max(0, need_short- len(acc["short"])),
            "long": max(0, need_long - len(acc["long"]))
        }

    PATCH_TRIES = 6
    for _ in range(PATCH_TRIES):
        miss = missing_now()
        if sum(miss.values()) == 0:
            break
        banlist = list(seen_prompts)
        patched = patch_missing(miss, banlist)

        # Enforce exact counts from the patch response
        for t in ("mcq", "short", "long"):
            if len(patched[t]) != miss[t]:
                # reject this type; ask again next iteration
                patched[t] = []

        dedup_merge(acc, patched, seen_prompts, seen_sigs)

    # Final check
    if not (len(acc["mcq"]) == need_mcq and len(acc["short"]) == need_short and len(acc["long"]) == need_long):
        raise RuntimeError(
            f"LLM could not provide exactly mcq={need_mcq}, short={need_short}, long={need_long} after patches. "
            f"Got mcq={len(acc['mcq'])}, short={len(acc['short'])}, long={len(acc['long'])}"
        )

    # ---------- persist ----------
    cur = db.execute("INSERT INTO quizzes(username, level) VALUES(?, ?)", (username, level))
    quiz_id = cur.lastrowid

    # Fixed board positions 0..15 (no shuffling -> your front-end places by pos)
    order: List[Dict] = (
        [dict(t="mcq",  item=it) for it in acc["mcq"]] +
        [dict(t="short",item=it) for it in acc["short"]] +
        [dict(t="long", item=it) for it in acc["long"]]
    )
    # If you want a simple interleave to avoid clumping, you can do that here,
    # but we’ll keep it deterministic by type order as above.

    # Debug summary
    def trunc(s, n=70):
        s = (s or "").replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s
    print(f"[create_board_quiz] L{level} totals: mcq={len(acc['mcq'])}, short={len(acc['short'])}, long={len(acc['long'])}")
    for i, it in enumerate(acc["mcq"]):
        print(f"  MCQ[{i:02d}] {trunc(it.get('prompt'))}  ans={['A','B','C','D'][int(it.get('answer_index',0))]}")
    for i, it in enumerate(acc["short"]):
        print(f"  SHORT[{i:02d}] {trunc(it.get('prompt'))}  ans={trunc(it.get('answer'))}")
    for i, it in enumerate(acc["long"]):
        print(f"  LONG[{i:02d}] {trunc(it.get('prompt'))}  req={','.join((it.get('required_words') or [])[:4])}")

    # Insert with extra.pos = 0..15
    for pos in range(TOTAL):
        t = order[pos]["t"]
        it = order[pos]["item"]
        prompt = (it.get("prompt") or "").strip()

        extra = {
            "pos": pos,
            "hint": (it.get("hint") or "").strip(),
            "explanation": (it.get("explanation") or "").strip()
        }

        choice_a = choice_b = choice_c = choice_d = None
        correct = None

        if t == "mcq":
            options = (it.get("options") or [])[:4]
            choice_a, choice_b, choice_c, choice_d = [str(o).strip() for o in options]
            ai = int(it.get("answer_index", 0))
            correct = ["A", "B", "C", "D"][ai]

        elif t == "short":
            correct = (it.get("answer") or "").strip()
            alts = it.get("alternates") or []
            if isinstance(alts, list) and alts:
                extra["alternates"] = [str(a).strip() for a in alts[:2]]

        else:  # long
            req = it.get("required_words") or []
            extra["required_words"] = [str(w).strip() for w in req[:4]]

        db.execute(
            """
            INSERT INTO quiz_questions
              (quiz_id, qtype, prompt, choice_a, choice_b, choice_c, choice_d, correct, extra)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                quiz_id, t, prompt,
                choice_a, choice_b, choice_c, choice_d,
                correct, json.dumps(extra, ensure_ascii=False)
            )
        )

    db.commit()
    return quiz_id


# -------------------- Auth --------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()
        remember = bool(request.form.get("rememberMe"))

        # validation
        if not username or not password:
            flash("User ID and password are required.", "danger")
            return render_template("login.html", form_username=username, form_remember=remember)

        db = get_db()
        row = db.execute(
            "SELECT username, password_hash, level, xp FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        if not row or not row["password_hash"]:
            flash("Account not found. Please register first.", "warning")
            # stay on login page
            return render_template("login.html", form_username=username, form_remember=remember)

        if not check_password_hash(row["password_hash"], password):
            flash("Incorrect password.", "danger")
            # stay on login page
            return render_template("login.html", form_username=username, form_remember=remember)

        # success
        session["username"] = username
        session.permanent = remember  # 30 days if checked
        flash(f"Logged in as {username}", "success")
        return redirect(url_for("home"))

    # GET
    return render_template("login.html")
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()
        confirm  = (request.form.get("confirm")  or "").strip()

        # Only these three analytics fields
        age_raw  = (request.form.get("age") or "").strip()
        grade    = (request.form.get("grade") or "").strip()
        gender   = (request.form.get("gender") or "").strip()  # '', 'female', 'male', 'other'

        if not username or not password:
            flash("User ID and password are required.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "warning")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        # Normalize inputs
        try:
            age = int(age_raw) if age_raw else None
        except ValueError:
            age = None

        if gender not in ("female", "male", "other", ""):
            gender = ""

        db = get_db()
        exists = db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
        if exists:
            flash("That User ID is already taken.", "danger")
            return redirect(url_for("register"))

        db.execute("""
            INSERT INTO users(username, level, xp, password_hash, age, grade, gender)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            1,                # level
            0,                # xp
            generate_password_hash(password),
            age,
            grade or None,
            gender or None
        ))
        db.commit()

        session["username"] = username
        flash("Account created. Welcome!", "success")
        return redirect(url_for("home"))

    return render_template("register.html")




@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("login"))

@app.route("/")
def index():
    return render_template('index.html')
# -------------------- Home --------------------
@app.route("/home")
def home():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    row = db.execute("SELECT level, xp, title_code FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        db.execute("INSERT INTO users(username, level, xp) VALUES(?, ?, ?)", (username, 1, 0))
        db.commit()
        user_level, user_xp, title_code = 1, 0, None
    else:
        user_level, user_xp, title_code = row["level"], row["xp"], row["title_code"]

    # If no title saved but user earned some, pick best and save
    if not title_code:
        codes = [r["code"] for r in db.execute(
            "SELECT code FROM user_achievements WHERE username=?", (username,)
        ).fetchall()]
        if codes:
            best = sorted(codes, key=lambda c: title_priority(c))[0]
            db.execute("UPDATE users SET title_code=? WHERE username=?", (best, username))
            db.commit()
            title_code = best

    title_info = TITLES.get(title_code or "", DEFAULT_TITLE)
    display_title = title_info["title"]
    title_icon = title_info["icon"]

    # Total achievements earned
    earned_count = db.execute(
        "SELECT COUNT(*) AS c FROM user_achievements WHERE username=?",
        (username,)
    ).fetchone()["c"]

    return render_template(
        "home.html",
        username=username,
        user_level=user_level,
        user_xp=user_xp,
        display_title=display_title,
        title_icon=title_icon,
        achievements_earned=earned_count,
    )
def recommend_stories(
    age: int | None,
    difficulty: int | None,
    q: str | None,
    category: str | None,
    theme: str | None,
    limit: int | None = 12
):
    """
    Pure local search over MOCK_CATALOG. No network.

    Filters by:
      - q (in title/author/category/theme, case-insensitive substring)
      - age (must fall within the item's [age_min, age_max])
      - difficulty (exact match if provided)
      - category/theme (exact token match, case-insensitive)

    Ranking:
      - If q is provided: exact title match first, then title prefix, then others.

    Returns:
      - Up to `limit` items if limit is an int.
      - All matching items if limit is None.
    """
    qlow = (q or "").strip().lower()
    catlow = (category or "").strip().lower()
    themelow = (theme or "").strip().lower()

    def matches(item: dict) -> bool:
        # difficulty
        if difficulty and int(item["difficulty"]) != int(difficulty):
            return False
        # age within item range
        if age is not None and not (item["age_min"] <= age <= item["age_max"]):
            return False
        # category & theme
        if catlow and catlow not in [c.lower() for c in (item.get("categories") or [])]:
            return False
        if themelow and themelow not in [t.lower() for t in (item.get("themes") or [])]:
            return False
        # query across fields
        if qlow:
            hay = [
                (item.get("title") or "").lower(),
                (item.get("author") or "").lower(),
                " ".join(item.get("categories") or []).lower(),
                " ".join(item.get("themes") or []).lower(),
            ]
            if not any(qlow in h for h in hay):
                return False
        return True

    results = [s for s in MOCK_CATALOG if matches(s)]

    # simple ranking: exact title hit first, then prefix, then substring/others
    def score(item: dict) -> int:
        title = (item.get("title") or "").lower()
        if qlow and title == qlow:
            return 0
        if qlow and title.startswith(qlow):
            return 1
        return 2

    results.sort(key=score)

    # Only slice if a numeric limit is provided
    ranked = results if limit is None else results[:max(0, int(limit))]

    # ensure minimal fields & types (mirrors old structure)
    items = []
    for r in ranked:
        items.append({
            "slug": r["slug"],
            "title": r["title"],
            "author": r["author"],
            "cover": r.get("cover"),
            "provider": r["provider"],
            "provider_id": r["provider_id"],
            "age_min": r["age_min"],
            "age_max": r["age_max"],
            "difficulty": r["difficulty"],
            "categories": r.get("categories", []),
            "themes": r.get("themes", []),
        })

    # fallback if nothing matched: pull near requested difficulty (or whole catalog)
    if not items:
        pool = [s for s in MOCK_CATALOG if (not difficulty or s["difficulty"] == difficulty)]
        if limit is None:
            items = pool if pool else list(MOCK_CATALOG)
        else:
            items = (pool[:limit] if pool else list(MOCK_CATALOG)[:limit])

    return items


def synthesize_story_text(level:int=2, title:str="A Small Adventure"):
    pool = WORD_POOLS.get(level, WORD_POOLS[2])
    nouns = random.sample(pool["nouns"], min(6, len(pool["nouns"])))
    verbs = random.sample(pool["verbs"], min(6, len(pool["verbs"])))
    adjs  = random.sample(pool["adjectives"], min(6, len(pool["adjectives"])))
    paras = []
    for p in range(3):
        sent = []
        for _ in range(5):
            n, v = random.choice(nouns), random.choice(verbs)
            a = random.choice(adjs)
            sent.append(f"The {a} {n} tried to {v} carefully.")
        paras.append(" ".join(sent))
    return f"{title}\n\n" + "\n\n".join(paras)
STOP = set("a,an,the,and,or,but,if,then,so,to,of,for,in,on,at,by,with,from,as,is,are,was,were,be,been,being,it,its,his,her,he,she,they,them,we,you,i,that,this,these,those".split(","))

def pick_vocab_from_text(text:str, max_words=12):
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    cand = [t for t in tokens if t not in STOP and len(t) >= 5]
    uniq = []
    for t in cand:
        if t not in uniq:
            uniq.append(t)
    random.shuffle(uniq)
    return uniq[:max_words]
def ensure_schema():
    """Create/patch minimal tables & columns this endpoint needs."""
    conn = get_db()
    c = conn.cursor()

    # Questions table (assumes it already exists in your app)
    # expected cols: id (PK), quiz_id, qtype ('mcq'|'short'|'long'), correct, explanation, extra (JSON, optional)

    # Answers table: log each attempt (one row per answer check)
    c.execute("""
    CREATE TABLE IF NOT EXISTS Answers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        quiz_id INTEGER,
        question_id INTEGER,
        response_text TEXT,
        was_correct INTEGER,
        confidence INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Users table: add xp/level if missing
    cols = {r["name"] for r in c.execute("PRAGMA table_info(Users)").fetchall()}
    if "user_xp" not in cols:
        c.execute("ALTER TABLE Users ADD COLUMN user_xp INTEGER DEFAULT 0")
    if "user_level" not in cols:
        c.execute("ALTER TABLE Users ADD COLUMN user_level INTEGER DEFAULT 1")

    conn.commit()
    conn.close()
def ensure_story_record(slug:str, meta:dict):
    db = get_db()
    row = db.execute("SELECT id FROM stories WHERE slug=?", (slug,)).fetchone()
    if row:
        # refresh missing meta, keep existing content as-is
        db.execute("""
            UPDATE stories
               SET title       = COALESCE(NULLIF(title,''), ?),
                   author      = COALESCE(NULLIF(author,''), ?),
                   age_min     = COALESCE(age_min, ?),
                   age_max     = COALESCE(age_max, ?),
                   difficulty  = COALESCE(difficulty, ?),
                   categories  = COALESCE(NULLIF(categories,''), ?),
                   themes      = COALESCE(NULLIF(themes,''), ?),
                   provider    = COALESCE(NULLIF(provider,''), ?),
                   provider_id = COALESCE(NULLIF(provider_id,''), ?)
             WHERE slug = ?
        """, (
            meta.get("title"), meta.get("author"),
            meta.get("age_min"), meta.get("age_max"),
            meta.get("difficulty"),
            json.dumps(meta.get("categories") or []),
            json.dumps(meta.get("themes") or []),
            meta.get("provider"), meta.get("provider_id"),
            slug
        ))
        db.commit()
        return row["id"]

    db.execute("""
        INSERT INTO stories(slug,title,author,age_min,age_max,difficulty,categories,themes,provider,provider_id,content)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
    """, (
        slug, meta.get("title"), meta.get("author"),
        meta.get("age_min"), meta.get("age_max"), meta.get("difficulty"),
        json.dumps(meta.get("categories") or []),
        json.dumps(meta.get("themes") or []),
        meta.get("provider"), meta.get("provider_id"),
        meta.get("content") or ""
    ))
    db.commit()
    return db.execute("SELECT id FROM stories WHERE slug=?", (slug,)).fetchone()["id"]
import json
def fetch_question(conn, qid):
    c = conn.cursor()
    row = c.execute("""
        SELECT id, quiz_id, qtype, correct, explanation,
               COALESCE(extra, '') AS extra
        FROM Questions WHERE id = ?
    """, (qid,)).fetchone()
    return row
def _fetch_ready_stories(db, limit=3):
    """
    Returns up to `limit` already-generated stories (content != '') as a list of dicts.
    Randomly selects stories from those available.
    """
    # 1) Confirm table and available columns
    cols = {r["name"] for r in db.execute("PRAGMA table_info(stories)").fetchall()}

    # 2) Run the query with RANDOM() ordering
    rows = db.execute(
        """
        SELECT slug, title, author, categories, themes
        FROM stories
        WHERE COALESCE(NULLIF(content,''), '') <> ''
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    # 3) Helper to safely parse JSON list strings
    def _safe_list(s):
        try:
            v = json.loads(s or "[]")
            return v if isinstance(v, list) else []
        except Exception:
            return []

    # 4) Build formatted output for the template
    out = []
    for r in rows:
        out.append({
            "slug": r["slug"],
            "title": r["title"],
            "author": r["author"] or "Unknown",
            "categories": _safe_list(r["categories"]),
            "themes": _safe_list(r["themes"]),
            "cover": None,  # your template already handles missing cover
        })
    return out


@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    # user basics
    u = db.execute("SELECT level, xp, title_code FROM users WHERE username=?", (username,)).fetchone()
    user_level = u["level"] if u else 1
    user_xp    = u["xp"] if u else 0
    title_info = TITLES.get((u["title_code"] or ""), DEFAULT_TITLE)
    display_title = title_info["title"]
    title_icon    = title_info["icon"]

    # achievements summary
    total_ach = db.execute("SELECT COUNT(*) AS c FROM achievements").fetchone()["c"]
    earned_codes = [r["code"] for r in db.execute("SELECT code FROM user_achievements WHERE username=?", (username,)).fetchall()]
    earned_ach = len(earned_codes)

    earned_titles = []
    for code in earned_codes:
        t = TITLES.get(code)
        if t and t.get("title"):
            earned_titles.append((t["priority"], t["title"]))
    earned_titles = [t for _, t in sorted(set(earned_titles), key=lambda x: x[0])]

    # recent quiz history
    attempts = db.execute(
        """
        SELECT level, score, total, lines_completed, created_at
        FROM attempts
        WHERE username=?
        ORDER BY created_at DESC
        LIMIT 12
        """,
        (username,),
    ).fetchall()

    # vocab notebook (bookmarked latest)
    vocab_rows = db.execute(
        """
        SELECT word, translation, definition, created_at, bookmarked
        FROM vocab
        WHERE username=? AND bookmarked=1
        ORDER BY created_at DESC
        LIMIT 30
        """,
        (username,),
    ).fetchall()

    # ——— NEW: pick exactly 3 "ready" stories or fallback to a few recommendations ———
    ready_items = _fetch_ready_stories(db, limit=3)
    if not ready_items:
        # fallback: grab 3 suggestions (no "Ready" badge in your template for these)
        fallback = recommend_stories(limit=3) or []
        for s in fallback:
            try:
                ensure_story_record(s["slug"], s)  # keep meta synced; content may still be empty
            except Exception:
                pass
        # normalize keys the template expects
        ready_items = [{
            "slug": s.get("slug"),
            "title": s.get("title"),
            "author": s.get("author") or "Unknown",
            "categories": s.get("categories") or [],
            "themes": s.get("themes") or [],
            "cover": s.get("cover"),
        } for s in fallback]

    return render_template(
        "profile.html",
        username=username,
        user_level=user_level,
        user_xp=user_xp,
        display_title=display_title,
        title_icon=title_icon,
        achievements_total=total_ach,
        achievements_earned=earned_ach,
        earned_titles=[t for t in earned_titles],
        attempts=attempts,
        vocab_rows=vocab_rows,
        ready_items=ready_items,  # <— pass to your template
    )

# app.py
def _short_explanation_text(ex: dict, correct_text: str, user_ans: str | None) -> str | None:
    """
    Pick the best explanation string for SHORT items.
    Priority: explicit 'explanation' > 'hint' > a tiny synthesized rule for a common pattern.
    """
    try:
        expl = (ex.get("explanation") or "").strip()
    except Exception:
        expl = ""
    if expl:
        return expl

    try:
        hint = (ex.get("hint") or "").strip()
    except Exception:
        hint = ""
    if hint:
        return hint

    # simple synthesis for a very common pattern we use in local fallback
    sent = (ex.get("sentence") or "").lower()
    if "walked ___ the store" in sent and correct_text == "to":
        return "Use 'to' for a destination (the store). 'Into' means going inside; 'toward' is only direction."
    return None

@app.get("/api/define")
def api_define():
    word = (request.args.get("word") or "").strip()
    if not word:
        return {"error": "missing word"}, 400

    # dictionary lookup
    ko, defin = dict_lookup_en_ko(word)

    bookmarked = None  # default when not logged in
    if "username" in session:
        db = get_db()
        username = session["username"]
        w = word.lower()

        # upsert a vocab row so we can track bookmark state reliably
        row = db.execute(
            "SELECT word, bookmarked FROM vocab WHERE username=? AND word=? ORDER BY created_at DESC LIMIT 1",
            (username, w),
        ).fetchone()

        if not row:
            db.execute(
                """
                INSERT INTO vocab(username, word, translation, definition, source_story_id, bookmarked)
                VALUES(?,?,?,?,?,?)
                """,
                (username, w, ko or "", defin or "", None, 0),
            )
        else:
            # keep definition/translation fresh for this user
            db.execute(
                "UPDATE vocab SET translation=?, definition=? WHERE username=? AND word=?",
                (ko or "", defin or "", username, w),
            )

        db.commit()

        # fetch the current bookmark state after upsert/update
        row2 = db.execute(
            "SELECT bookmarked FROM vocab WHERE username=? AND word=? ORDER BY created_at DESC LIMIT 1",
            (username, w),
        ).fetchone()
        bookmarked = bool(row2["bookmarked"]) if row2 else False

    # include 'bookmarked' only when we know the user
    payload = {"word": word, "ko": ko, "definition": defin}
    if bookmarked is not None:
        payload["bookmarked"] = bookmarked
    return payload

@app.get("/story/<slug>")
def read_story(slug):
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    # Try to load the story row by slug
    row = db.execute("SELECT * FROM stories WHERE slug=?", (slug,)).fetchone()

    # If it doesn't exist, build it from query params and generate with GPT
    if not row:
        title = (request.args.get("t") or "New Story").strip()

        try:
            age_min = int(request.args.get("a1")) if request.args.get("a1") else 8
        except Exception:
            age_min = 8
        try:
            age_max = int(request.args.get("a2")) if request.args.get("a2") else age_min + 4
        except Exception:
            age_max = age_min + 4
        try:
            difficulty = int(request.args.get("d")) if request.args.get("d") else 2
        except Exception:
            difficulty = 2

        cats = request.args.getlist("cat")
        ths  = request.args.getlist("th")

        content = gpt_generate_story(
            title=title,
            age_min=age_min,
            age_max=age_max,
            difficulty=difficulty,
            categories=cats,
            themes=ths
        )

        meta = {
            "slug": slug,
            "title": title,
            "author": "GPT",
            "age_min": age_min,
            "age_max": age_max,
            "difficulty": difficulty,
            "categories": cats,
            "themes": ths,
            "provider": "gpt",
            "provider_id": slug,
            "content": content,
        }
        story_id = ensure_story_record(slug, meta)
        row = db.execute("SELECT * FROM stories WHERE id=?", (story_id,)).fetchone()

    # If it exists but content is missing OR provider isn't 'gpt', generate once and persist.
    content = (row["content"] or "").strip()
    provider = (row["provider"] or "").strip().lower()

    def _parse_json_list(s):
        try:
            v = json.loads(s or "[]")
            return v if isinstance(v, list) else []
        except Exception:
            return []

    cats = _parse_json_list(row["categories"])
    ths  = _parse_json_list(row["themes"])

    needs_gen = (not content) or (provider != "gpt")
    if needs_gen:
        content = gpt_generate_story(
            title=row["title"] or "Story",
            age_min=row["age_min"] or 8,
            age_max=row["age_max"] or ((row["age_min"] or 8) + 4),
            difficulty=row["difficulty"] or 2,
            categories=cats,
            themes=ths
        )
        db.execute("UPDATE stories SET content=?, provider='gpt' WHERE id=?", (content, row["id"]))
        db.commit()
        row = db.execute("SELECT * FROM stories WHERE id=?", (row["id"],)).fetchone()

    # --- vocabulary extraction & notebook save ---
    vocab = pick_vocab_from_text(content, max_words=14)
    for w in vocab:
        ko, defin = dict_lookup_en_ko(w)
        try:
            db.execute("""
              INSERT OR IGNORE INTO vocab(username, word, translation, definition, source_story_id)
              VALUES(?,?,?,?,?)
            """, (username, w, ko or "", defin or "", row["id"]))
        except Exception:
            pass

    # record read
    db.execute("INSERT INTO story_reads(username, story_id) VALUES(?,?)", (username, row["id"]))
    db.commit()

    return render_template("story.html", story=row, content=content, vocab=vocab)

def gpt_generate_story(title: str,
                       age_min: int = 8,
                       age_max: int = 12,
                       difficulty: int = 2,
                       categories: list[str] | None = None,
                       themes: list[str] | None = None) -> str:
    """
    Returns plain-text story. First line is the title, then paragraphs separated by blank lines.
    Falls back to synthesize_story_text if GPT is unavailable.
    """
    categories = categories or []
    themes = themes or []
    level_map = {
        1: ("CEFR A1", "550–700"),
        2: ("CEFR A2", "650–850"),
        3: ("CEFR B1", "800–1000"),
        4: ("CEFR B1+", "900–1100"),
        5: ("CEFR B2", "1000–1200"),
    }
    level_tag, target_len = level_map.get(int(difficulty or 2))

    if not openai_client:
        return synthesize_story_text(difficulty or 2, title or "Story")

    level_map = {
        1: ("CEFR A1", "700–1100"),  # was 250–400
        2: ("CEFR A2", "1000–1500"),  # was 350–550
        3: ("CEFR B1", "1400–2000"),  # was 500–700
        4: ("CEFR B1+", "1800–2400"),  # was 600–800
        5: ("CEFR B2", "2100–2700"),  # was 700–900
    }

    system = (
        "You are an ESL children's story writer. "
        "Output ONLY the story text: the first line must be the exact title, "
        "then 9–14 short paragraphs separated by blank lines. No extra commentary."
    )
    user = f"""
Write an original English short story.

Title: {title or "Story"}
Audience age: {age_min}-{age_max}
Difficulty: {level_tag} (approx {difficulty}/5)
Target length: {target_len} words total.
Style: Warm, friendly, age-appropriate; simple, clear vocabulary matching difficulty.
Structure: 3–6 short paragraphs. Use active voice. Avoid slang and idioms that are too advanced.
Safety: Avoid violence, romance, religion, politics, and anything scary.

Optional categories: {", ".join(categories) if categories else "n/a"}
Optional themes: {", ".join(themes) if themes else "n/a"}
"""
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.8,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        # basic sanity fallback if response is too short
        if len(text.split()) < 120:
            text += "\n\n" + synthesize_story_text(difficulty or 2, title or "Story")
        return text
    except Exception:
        return synthesize_story_text(difficulty or 2, title or "Story")

# -------------------- Helpers for lines and achievements --------------------
def compute_completed_lines(correct_positions):
    lines = []
    for r in range(4):
        lines.append({4*r + c for c in range(4)})
    for c in range(4):
        lines.append({4*r + c for r in range(4)})
    lines.append({0,5,10,15})
    lines.append({3,6,9,12})

    count = 0
    for L in lines:
        if L.issubset(correct_positions):
            count += 1
    return count

def grant_achievement(db, username, code):
    db.execute(
        "INSERT OR IGNORE INTO user_achievements(username, code) VALUES(?, ?)",
        (username, code)
    )
    # Update title if this code outranks current title
    cur = db.execute("SELECT title_code FROM users WHERE username=?", (username,)).fetchone()
    current = cur["title_code"] if cur else None
    if current is None or title_priority(code) < title_priority(current):
        db.execute("UPDATE users SET title_code=? WHERE username=?", (code, username))

def evaluate_and_award_achievements(db, username, quiz_id, total_correct, lines_completed):
    # count completed boards (distinct quizzes with answers)
    boards = db.execute("""
        SELECT COUNT(DISTINCT a.quiz_id) AS c
        FROM quiz_answers a
        JOIN quizzes q ON q.id=a.quiz_id
        WHERE q.username=?
    """, (username,)).fetchone()["c"]

    # long-answer correct in this board?
    long_correct = db.execute("""
        SELECT COUNT(*) AS c
        FROM quiz_answers a
        JOIN quiz_questions qq ON qq.id=a.question_id
        WHERE a.quiz_id=? AND a.is_correct=1 AND qq.qtype='long'
    """, (quiz_id,)).fetchone()["c"]

    # grant
    if boards == 1:
        grant_achievement(db, username, "FIRST_STEPS")
    if total_correct >= 12:
        grant_achievement(db, username, "SHARP_SHOOTER")
    if total_correct >= 16:
        grant_achievement(db, username, "PERFECTIONIST")
    if lines_completed >= 3:
        grant_achievement(db, username, "LINE_TRIO")
    if lines_completed >= 6:
        grant_achievement(db, username, "BINGO_MASTER")
    if long_correct >= 1:
        grant_achievement(db, username, "WORDSMITH")

def recalc_user_achievements(db, username):
    """
    Safe recalculation (called on /achievements page load).
    It scans all of user's quizzes/answers and grants missed achievements.
    """
    # all quizzes for user
    quiz_ids = [r["id"] for r in db.execute(
        "SELECT id FROM quizzes WHERE username=?", (username,)
    ).fetchall()]

    for qid in quiz_ids:
        # correct answers count
        total_correct = db.execute(
            "SELECT COUNT(*) AS c FROM quiz_answers WHERE quiz_id=? AND is_correct=1", (qid,)
        ).fetchone()["c"]
        # correct positions
        rows = db.execute("""
            SELECT qq.extra
            FROM quiz_answers a
            JOIN quiz_questions qq ON qq.id=a.question_id
            WHERE a.quiz_id=? AND a.is_correct=1
        """, (qid,)).fetchall()
        pos_set = set()
        for r in rows:
            try:
                ex = json.loads(r["extra"]) if r["extra"] else {}
                if isinstance(ex.get("pos"), int):
                    pos_set.add(int(ex["pos"]))
            except Exception:
                pass
        lines_completed = compute_completed_lines(pos_set)
        evaluate_and_award_achievements(db, username, qid, total_correct, lines_completed)

    db.commit()
@app.route("/quiz/<int:level>", methods=["GET"])
def start_quiz(level):
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    # Gate locked levels using the user's persisted level
    row = db.execute("SELECT level, xp FROM users WHERE username = ?", (username,)).fetchone()
    current_level = row["level"] if row else 1
    if level > current_level:
        flash("That level is locked. Clear your current level first.", "warning")
        return redirect(url_for("home"))

    # Respect ?new=1 to force a fresh board
    force_new = (request.args.get("new") == "1")

    quiz_id = None
    if not force_new:
        # Try to reuse an unfinished quiz for this level
        unfinished = db.execute("""
            SELECT q.id AS quiz_id
            FROM quizzes q
            LEFT JOIN (
              SELECT quiz_id, COUNT(*) AS answered
              FROM quiz_answers
              GROUP BY quiz_id
            ) a ON a.quiz_id = q.id
            WHERE q.username=? AND q.level=?
            ORDER BY q.created_at DESC
        """, (username, level)).fetchall()

        for r in unfinished:
            total_q = db.execute(
                "SELECT COUNT(*) AS c FROM quiz_questions WHERE quiz_id=?",
                (r["quiz_id"],)
            ).fetchone()["c"]
            answered = db.execute(
                "SELECT COUNT(*) AS c FROM quiz_answers WHERE quiz_id=?",
                (r["quiz_id"],)
            ).fetchone()["c"]
            if total_q == 16 and answered < 16:
                quiz_id = r["quiz_id"]
                break

    # Create a new board if none reusable
    if not quiz_id:
        quiz_id = create_board_quiz(username, level)

    # Load questions + parse extras
    rows = db.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,)).fetchall()
    qs = []
    for r in rows:
        d = dict(r)
        d["extra_parsed"] = json.loads(d["extra"]) if d.get("extra") else {}
        qs.append(d)

    # Safety net: ensure exactly 16 questions
    if len(qs) < 16:
        print(len(qs))
        needed = 16 - len(qs)
        positions_taken = {q.get("extra_parsed", {}).get("pos") for q in qs if q.get("extra_parsed")}
        available = [p for p in range(16) if p not in positions_taken]
        filler_used_norm = set()
        for i in range(needed):
            pos = available[i]
        db.commit()
        rows = db.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,)).fetchall()
        qs = []
        for r in rows:
            d = dict(r)
            d["extra_parsed"] = json.loads(d["extra"]) if d.get("extra") else {}
            qs.append(d)

    # NEW: pass persisted profile to template so the header chip shows correct Lv/XP
    user_level = row["level"] if row else 1
    user_xp    = row["xp"] if row else 0

    return render_template(
        "quiz.html",
        level=level,
        quiz_id=quiz_id,
        qs=qs,
        user_level=user_level,
        user_xp=user_xp
    )

def _quiz_state(db, quiz_id: int):
    # Pull answers
    answers = db.execute("""
        SELECT qq.id AS qid, qq.qtype, qq.extra, a.is_correct
        FROM quiz_questions qq
        LEFT JOIN quiz_answers a ON a.question_id=qq.id AND a.quiz_id=qq.quiz_id
        WHERE qq.quiz_id=?
        ORDER BY qq.id
    """, (quiz_id,)).fetchall()

    answered = set()
    correct = set()
    pos_correct = set()
    wrong = 0

    for r in answers:
        ex = json.loads(r["extra"] or "{}")
        pos = ex.get("pos")
        if r["is_correct"] is not None:
            answered.add(pos)
            if int(r["is_correct"]) == 1:
                correct.add(pos)
                if pos is not None:
                    pos_correct.add(pos)
            else:
                wrong += 1

    # lives: 3 base minus wrong
    lives = max(0, 3 - wrong)

    # lines
    def _lines_from_positions(pos_set):
        lines = []
        for rr in range(4): lines.append({4*rr + c for c in range(4)})
        for cc in range(4): lines.append({4*r + cc for r in range(4)})
        lines.append({0,5,10,15})
        lines.append({3,6,9,12})
        cnt = 0
        awarded_idx = []
        for i, L in enumerate(lines):
            if L.issubset(pos_set):
                cnt += 1
                awarded_idx.append(i)
        return cnt, awarded_idx

    line_cnt, awarded_idx = _lines_from_positions(pos_correct)

    # xp: 10 per correct + 20 per line
    xp = (len(correct) * 10) + (line_cnt * 20)

    return {
        "answered": sorted([p for p in answered if p is not None]),
        "correct": sorted([p for p in correct if p is not None]),
        "lives": lives,
        "lines": line_cnt,
        "xp": xp
    }

@app.get("/api/quiz/state/<int:quiz_id>")
def api_quiz_state(quiz_id):
    if "username" not in session:
        return {"error":"auth"}, 401
    db = get_db()
    # Ensure this quiz belongs to the user
    owner = db.execute("SELECT username FROM quizzes WHERE id=?", (quiz_id,)).fetchone()
    if not owner or owner["username"] != session["username"]:
        return {"error":"not-found"}, 404
    return _quiz_state(db, quiz_id)
# --- API: answer a cell and persist XP/level delta ---
@app.route("/api/quiz/answer", methods=["POST"])
def api_quiz_answer():
    if "username" not in session:
        return jsonify({"ok": False, "error": "auth required"}), 401
    username = session["username"]
    db = get_db()

    data = request.get_json(silent=True) or {}
    quiz_id = int(data.get("quiz_id") or 0)
    qid     = int(data.get("question_id") or 0)
    resp    = (data.get("response") or "").strip()

    # Load question (must belong to this quiz)
    q = db.execute(
        "SELECT * FROM quiz_questions WHERE id=? AND quiz_id=?",
        (qid, quiz_id)
    ).fetchone()
    if not q:
        return jsonify({"ok": False, "error": "question not found"}), 404

    qtype = q["qtype"]
    prompt = q["prompt"]
    correct_field = q["correct"] or ""
    extra = {}
    try:
        extra = json.loads(q["extra"] or "{}")
    except Exception:
        extra = {}

    # Helper: normalize text
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    was_correct = False
    explanation = (extra.get("explanation") or "").strip()
    hint = (extra.get("hint") or "").strip()

    if qtype == "mcq":
        # Expect resp = 'A'|'B'|'C'|'D' (case-insensitive)
        if resp:
            letter = str(resp).strip().upper()
            corr   = str(correct_field).strip().upper()
            was_correct = (letter == corr)

    elif qtype == "short":
        gold = norm(correct_field)
        cand = norm(resp)
        alts = []
        if isinstance(extra.get("alternates"), list):
            alts = [norm(a) for a in extra.get("alternates")[:2] if isinstance(a, str)]
        was_correct = (cand == gold) or (cand in alts)

    else:  # long
        # All required words must appear as tokens (case-insensitive)
        req = extra.get("required_words") or []
        text = " " + norm(resp) + " "
        ok = True
        for w in req:
            w_norm = norm(w)
            # simple word-boundary check
            if not re.search(rf"(^|[\s\W]){re.escape(w_norm)}($|[\s\W])", text):
                ok = False
                break
        was_correct = ok

    # XP / lives / lines bookkeeping:
    #   +10 XP per correct, -1 life per wrong (server keeps only totals; client handles lines visually)
    # Persist the answer first
    db.execute(
        """
        INSERT INTO quiz_answers(quiz_id, question_id, response, is_correct, awarded_xp)
        VALUES(?,?,?,?,?)
        """,
        (quiz_id, qid, resp, 1 if was_correct else 0, 10 if was_correct else 0)
    )

    # Update user's XP/level
    u = db.execute("SELECT level, xp FROM users WHERE username=?", (username,)).fetchone()
    cur_level = u["level"] if u else 1
    cur_xp    = u["xp"] if u else 0

    if was_correct:
        cur_xp += 10
    # update level on 100-XP boundaries
    new_level = max(cur_level, 1 + (cur_xp // 100))

    db.execute("UPDATE users SET xp=?, level=? WHERE username=?", (cur_xp, new_level, username))
    db.commit()

    # Return status to client (the client tracks lives/lines locally)
    return jsonify({
        "ok": True,
        "was_correct": was_correct,
        "explanation": explanation or hint or "",
        "user_level": new_level,
        "user_xp": cur_xp
    })

# ==== NEW HELPERS (place near your GPT functions) ====

def _is_ambiguous_mcq(item: dict) -> bool:
    """
    Heuristics to reject MCQs that aren't objectively right/wrong.
    Flags:
      - missing/invalid answer_index
      - fewer than 4 distinct options
      - vague 'best/most suitable' phrasing without a hard rule (e.g., not a preposition cloze)
      - 'adverb' fill-in MCQs (multiple can fit)
    """
    prompt = str(item.get("prompt", "")).lower()
    opts = item.get("options", [])
    ai = item.get("answer_index", None)

    # must have 4 distinct options and a valid answer index
    if not isinstance(opts, list) or len(opts) < 4 or len({(o or "").strip().lower() for o in opts}) < 4:
        return True
    if not isinstance(ai, int) or ai < 0 or ai > 3:
        return True

    # avoid vague “best/most suitable/sounds better”
    vague_words = ("best", "most suitable", "sounds better", "fits best")
    if any(v in prompt for v in vague_words):
        # allow if it's an objective grammar constraint (preposition cloze)
        if not (("preposition" in prompt) and ("blank" in prompt)):
            return True

    # avoid adverb-of-manner cloze MCQs (often multiple fit)
    if "adverb" in prompt and "fill in the blank" in prompt and "preposition" not in prompt:
        return True

    return False


def _downgrade_ambiguous_to_short(item: dict) -> dict:
    """
    Convert a shaky MCQ to a SHORT item when possible.
    Returns {} if conversion isn't safe.
    """
    prompt = str(item.get("prompt", "")).strip()
    opts = item.get("options", [])
    ai = item.get("answer_index", None)

    if not (isinstance(opts, list) and len(opts) >= 4 and isinstance(ai, int) and 0 <= ai < 4):
        return {}

    correct = str(opts[ai]).strip()
    alts = [str(o).strip() for i, o in enumerate(opts) if i != ai]

    # Only convert if it's clearly a cloze prompt
    if "blank" not in prompt.lower():
        return {}

    return {
        "qtype": "short",
        "prompt": prompt,
        "answer": correct,
        "alternates": alts[:3],
        "hint": "Choose the single best word for this exact sentence.",
        "explanation": "This word fits the sentence pattern; others are less precise here."
    }


def sanitize_gpt_quiz_payload(data: dict) -> dict:
    """
    Ensure objective MCQs; convert/drop ambiguous ones.
    De-duplicate options defensively.
    """
    out = {"mcq": [], "short": list(data.get("short", [])), "long": list(data.get("long", []))}
    mcqs = list(data.get("mcq", []))

    for item in mcqs:
        if _is_ambiguous_mcq(item):
            as_short = _downgrade_ambiguous_to_short(item)
            if as_short:
                out["short"].append(as_short)
            continue

        # clean duplicate options if any
        seen = set()
        cleaned = []
        for o in item.get("options", []):
            k = (o or "").strip()
            if k.lower() not in seen:
                cleaned.append(k)
                seen.add(k.lower())

        if len(cleaned) >= 4:
            item["options"] = cleaned[:4]
            out["mcq"].append(item)
        else:
            as_short = _downgrade_ambiguous_to_short(item)
            if as_short:
                out["short"].append(as_short)
            # else drop

    return out

@app.route("/submit/quiz/<int:quiz_id>", methods=["POST"])
def submit_quiz_instance(quiz_id):
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    quiz = db.execute("SELECT * FROM quizzes WHERE id = ? AND username = ?", (quiz_id, username)).fetchone()
    if not quiz:
        flash("Quiz not found.", "danger")
        return redirect(url_for("home"))

    questions = db.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,)).fetchall()

    total = len(questions)
    total_correct = 0
    total_xp = 0
    correct_positions = set()

    for q in questions:
        qid = q["id"]
        qtype = q["qtype"]
        is_correct = 0
        awarded_xp = 0
        resp_text = ""

        if qtype == "mcq":
            key = f"q_{qid}"
            user_ans = (request.form.get(key) or "").upper()
            correct_letter = (q["correct"] or "").upper()
            is_correct = 1 if user_ans == correct_letter else 0
            awarded_xp = 10 if is_correct else 0
            resp_text = user_ans

        elif qtype == "short":
            user_ans = (request.form.get(f"q_{qid}") or "").strip().lower()
            correct_text = (q["correct"] or "").strip().lower()
            try:
                ex = json.loads(q["extra"] or "{}")
            except Exception:
                ex = {}
            alt_list = [str(a).strip().lower()
                        for a in (ex.get("alternates") or [])
                        if str(a).strip()]

            if user_ans and user_ans == correct_text:
                is_correct = 1
                awarded_xp = 10
            elif user_ans and user_ans in alt_list:
                is_correct = 1
                awarded_xp = 7
            else:
                is_correct = 0
                awarded_xp = 0

            resp_text = user_ans

        else:  # long
            try:
                required = set(json.loads(q["correct"]))
            except Exception:
                required = set()
            sentence = (request.form.get(f"q_{qid}") or "").strip()
            low = sentence.lower()
            hits = sum(1 for w in required if w.lower() in low)
            is_correct = 1 if hits == len(required) and len(sentence.split()) >= max(6, len(required) + 2) else 0
            awarded_xp = 15 if is_correct else int(round(15 * (hits / max(1, len(required)))))
            resp_text = sentence

        db.execute("""
            INSERT INTO quiz_answers(quiz_id, question_id, response, is_correct, awarded_xp)
            VALUES (?, ?, ?, ?, ?)
        """, (quiz_id, qid, resp_text, is_correct, awarded_xp))

        total_correct += is_correct
        total_xp += awarded_xp

        extra = json.loads(q["extra"]) if q["extra"] else {}
        pos = extra.get("pos", None)
        if is_correct and pos is not None:
            correct_positions.add(int(pos))

    # line bonus just for display (XP already handled incrementally in /api/quiz/answer)
    lines_completed = compute_completed_lines(correct_positions)
    bonus_xp = 20 * lines_completed
    total_xp += bonus_xp

    score_pct = round(100 * total_correct / total) if total else 0
    db.execute("""
        INSERT INTO attempts(username, level, score, total, lines_completed)
        VALUES (?, ?, ?, ?, ?)
    """, (username, quiz["level"], score_pct, total, lines_completed))

    # Award achievements (no XP changes here)
    evaluate_and_award_achievements(db, username, quiz_id, total_correct, lines_completed)
    db.commit()

    # Read current user level/xp (already updated during play)
    urow = db.execute("SELECT level, xp FROM users WHERE username = ?", (username,)).fetchone()
    new_level = urow["level"]
    xp_after  = urow["xp"]
    leveled_up = False  # level-ups occurred incrementally

    return render_template_string("""
    <!doctype html><html><head>
      <meta charset="utf-8"><title>Board Result</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head><body class="bg-light">
      <div class="container py-5">
        <div class="card shadow-sm">
          <div class="card-body">
            <h1 class="h4">Result – Level {{ level }}</h1>
            <p class="mb-1">Correct cells: <strong>{{ correct }}/{{ total }}</strong></p>
            <p class="mb-1">Completed lines: <strong>{{ lines_completed }}</strong> (+{{ 20 * lines_completed }} XP)</p>
            <p class="mb-1">XP earned (this board): <strong>{{ gained_xp }}</strong></p>
            <p class="mb-3">Your XP now (this level): <strong>{{ xp_after % 100 }}</strong> / 100 (Level {{ new_level }})</p>
            {% if leveled_up %}
              <div class="alert alert-success">Great job! You leveled up to Level {{ new_level }}.</div>
            {% endif %}
            <a href="{{ url_for('achievements') }}" class="btn btn-success">See Achievements</a>
            <a href="{{ url_for('home') }}" class="btn btn-primary ms-2">Back to Home</a>
          </div>
        </div>
      </div>
    </body></html>
    """, level=quiz["level"], correct=total_correct, total=total,
       lines_completed=lines_completed, gained_xp=total_xp,
       xp_after=xp_after, new_level=new_level, leveled_up=leveled_up)
def ensure_achievement_progress_schema(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS achievement_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            code TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            target   INTEGER NOT NULL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, code),
            FOREIGN KEY (code) REFERENCES achievements(code) ON DELETE CASCADE
        );
    """)
    db.commit()

def _ensure_progress_row(db, username: str, code: str, target: int | None = None):
    row = db.execute(
        "SELECT progress, target FROM achievement_progress WHERE username=? AND code=?",
        (username, code)
    ).fetchone()
    if not row:
        t = int(target or 0)
        db.execute(
            "INSERT INTO achievement_progress(username, code, progress, target) VALUES(?,?,?,?)",
            (username, code, 0, t)
        )
        db.commit()
        return 0, t
    # optionally bump target if caller supplies a bigger one
    if target is not None and int(target) > int(row["target"]):
        db.execute(
            "UPDATE achievement_progress SET target=?, last_updated=CURRENT_TIMESTAMP WHERE username=? AND code=?",
            (int(target), username, code)
        )
        db.commit()
        return int(row["progress"]), int(target)
    return int(row["progress"]), int(row["target"])

def grant_achievement(db, username: str, code: str):
    # idempotent grant
    db.execute("""
        INSERT OR IGNORE INTO user_achievements(username, code)
        VALUES(?, ?)
    """, (username, code))
    # if we know a target, snap progress to target
    db.execute("""
        UPDATE achievement_progress
           SET progress=CASE WHEN target>0 THEN target ELSE progress END,
               last_updated=CURRENT_TIMESTAMP
         WHERE username=? AND code=?
    """, (username, code))
    db.commit()

def set_achievement_progress(db, username: str, code: str, value: int, target: int | None = None):
    cur, tgt = _ensure_progress_row(db, username, code, target)
    val = max(0, int(value))
    db.execute("""
        UPDATE achievement_progress
           SET progress=?, last_updated=CURRENT_TIMESTAMP
         WHERE username=? AND code=?
    """, (val, username, code))
    db.commit()
    # auto-grant if complete
    if (target if target is not None else tgt) and val >= (target if target is not None else tgt):
        grant_achievement(db, username, code)

def add_achievement_progress(db, username: str, code: str, delta: int, target: int | None = None):
    cur, tgt = _ensure_progress_row(db, username, code, target)
    set_achievement_progress(db, username, code, cur + int(delta), target=tgt if target is None else target)
def update_achievement_progress_snapshot(db, username: str):
    """
    Compute user's best-so-far stats and snapshot them into achievement_progress.
    Called after finishing a board and on /achievements page load.
    Progress semantics:
      - SHARP_SHOOTER (12): best correct cells in any single board so far (cap at 12)
      - PERFECTIONIST (16): best correct cells in any single board so far (cap at 16)
      - LINE_TRIO (3):      best lines completed in any single board so far
      - BINGO_MASTER (6):   best lines completed in any single board so far
      - WORDSMITH (1):      1 if any long-answer was ever correct, else 0
      - FIRST_STEPS (1):    number of completed boards, capped at 1
    """

    # Completed boards (distinct quizzes that have answers) for this user
    boards = db.execute("""
        SELECT COUNT(DISTINCT a.quiz_id) AS c
          FROM quiz_answers a
          JOIN quizzes q ON q.id=a.quiz_id
         WHERE q.username=?
    """, (username,)).fetchone()["c"]

    # Best correct cells in a single board so far
    best_correct = db.execute("""
        SELECT MAX(c) AS m FROM (
          SELECT COUNT(*) AS c
            FROM quiz_answers a
            JOIN quizzes q ON q.id=a.quiz_id
           WHERE q.username=? AND a.is_correct=1
           GROUP BY a.quiz_id
        )
    """, (username,)).fetchone()["m"]
    best_correct = int(best_correct or 0)

    # Best lines completed in a single board so far (from attempts table)
    best_lines = db.execute("""
        SELECT COALESCE(MAX(lines_completed), 0) AS m
          FROM attempts
         WHERE username=?
    """, (username,)).fetchone()["m"]
    best_lines = int(best_lines or 0)

    # Ever had a correct long-answer?
    long_any = db.execute("""
        SELECT COUNT(*) AS c
          FROM quiz_answers a
          JOIN quiz_questions qq ON qq.id=a.question_id
          JOIN quizzes q ON q.id=a.quiz_id
         WHERE q.username=? AND a.is_correct=1 AND qq.qtype='long'
    """, (username,)).fetchone()["c"]
    long_any = 1 if int(long_any or 0) > 0 else 0

    # Write snapshot (and auto-grant when hitting target)
    set_achievement_progress(db, username, "FIRST_STEPS",   min(boards, 1), target=1)
    set_achievement_progress(db, username, "SHARP_SHOOTER", min(best_correct, 12), target=12)
    set_achievement_progress(db, username, "PERFECTIONIST", min(best_correct, 16), target=16)
    set_achievement_progress(db, username, "LINE_TRIO",     min(best_lines, 3),    target=3)
    set_achievement_progress(db, username, "BINGO_MASTER",  min(best_lines, 6),    target=6)
    set_achievement_progress(db, username, "WORDSMITH",     long_any,              target=1)
@app.route("/achievements")
def achievements():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    # Refresh all grants + per-badge progress snapshots
    recalc_user_achievements(db, username)
    update_achievement_progress_snapshot(db, username)

    # Header info (level / xp / title)
    u = db.execute(
        "SELECT level, xp, title_code FROM users WHERE username=?",
        (username,)
    ).fetchone()
    user_level = int(u["level"] if u else 1)
    user_xp    = int(u["xp"] if u else 0)
    title_code = (u["title_code"] if u else None) or ""

    # Map current title_code to display title/icon if you maintain TITLES/DEFAULT_TITLE
    title_info = TITLES.get(title_code, DEFAULT_TITLE)
    primary_title      = title_info.get("title")
    primary_title_icon = title_info.get("icon", "bi-award")

    # Pull all badges with progress + earned time if any
    rows = db.execute("""
        SELECT a.code, a.name, a.description, a.icon,
               ap.progress, ap.target,
               ua.granted_at
          FROM achievements a
          LEFT JOIN achievement_progress ap
                 ON ap.code=a.code AND ap.username=?
          LEFT JOIN user_achievements ua
                 ON ua.code=a.code AND ua.username=?
         ORDER BY a.code
    """, (username, username)).fetchall()

    items = [{
        "code":        r["code"],
        "name":        r["name"],
        "description": r["description"],
        "icon":        r["icon"],
        "progress":    int(r["progress"] or 0),
        "target":      int(r["target"] or 0),
        "unlocked_at": r["granted_at"],
    } for r in rows]

    # Overview counts and earned titles (chip list)
    earned_codes = [r["code"] for r in db.execute(
        "SELECT code FROM user_achievements WHERE username=? ORDER BY granted_at",
        (username,)
    ).fetchall()]
    achievements_total  = db.execute("SELECT COUNT(*) AS c FROM achievements").fetchone()["c"]
    achievements_earned = len(earned_codes)

    # If you also want to surface a list of earned title names:
    earned_titles = []
    for code in earned_codes:
        t = TITLES.get(code)
        if t and t.get("title"):
            earned_titles.append((t["priority"], t["title"]))
    earned_titles = [t for _, t in sorted(set(earned_titles), key=lambda x: x[0])]

    return render_template(
        "achievements.html",
        # user header
        username=username,
        user_level=user_level,
        user_xp=user_xp,
        primary_title=primary_title,
        primary_title_icon=primary_title_icon,
        # grid + stats
        items=items,
        achievements_total=achievements_total,
        achievements_earned=achievements_earned,
        earned_codes=earned_codes,
        earned_titles=earned_titles,
        TITLES=TITLES,  # <-- add this
    )


@app.post("/api/title/set")
def api_set_title():
    if "username" not in session:
        return jsonify({"ok": False, "error": "login_required"}), 401

    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    username = session["username"]

    # If you keep title metadata in a TITLES dict (recommended):
    title_meta = TITLES.get(code)
    if not title_meta or not title_meta.get("title"):
        return jsonify({"ok": False, "error": "not_a_title"}), 400

    db = get_db()
    # Ensure user actually earned this achievement / title
    owned = db.execute(
        "SELECT 1 FROM user_achievements WHERE username=? AND code=?",
        (username, code)
    ).fetchone()
    if not owned:
        return jsonify({"ok": False, "error": "not_owned"}), 403

    # Save current title
    db.execute("UPDATE users SET title_code=? WHERE username=?", (code, username))
    db.commit()

    return jsonify({
        "ok": True,
        "code": code,
        "title": title_meta.get("title"),
        "icon": title_meta.get("icon", "bi-award")
    })

@app.get("/api/user/state")
def api_user_state():
    if "username" not in session:
        return {"error": "auth"}, 401
    db = get_db()
    row = db.execute(
        "SELECT level, xp, title_code FROM users WHERE username=?",
        (session["username"],)
    ).fetchone()
    if not row:
        return {"level": 1, "xp": 0, "xp_mod": 0, "title": None, "title_icon": None}

    title_info = TITLES.get((row["title_code"] or ""), DEFAULT_TITLE)
    return {
        "level": int(row["level"] or 1),
        "xp": int(row["xp"] or 0),
        "xp_mod": int(row["xp"] or 0) % 100,
        "title": title_info["title"],
        "title_icon": title_info["icon"],
    }

@app.post("/api/vocab/bookmark")
def api_vocab_bookmark():
    if "username" not in session:
        return {"error": "auth"}, 401
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip().lower()
    toggle = bool(data.get("toggle"))
    set_to = data.get("bookmarked")  # optional explicit state

    if not word:
        return {"error": "missing-word"}, 400

    db = get_db()
    username = session["username"]

    # Ensure the vocab row exists (reading usually created it, but be safe)
    row = db.execute(
        "SELECT word, bookmarked FROM vocab WHERE username=? AND word=? ORDER BY created_at DESC LIMIT 1",
        (username, word)
    ).fetchone()
    if not row:
        # create a placeholder entry so user can bookmark manually
        try:
            db.execute("""
              INSERT OR IGNORE INTO vocab(username, word, translation, definition, source_story_id, bookmarked)
              VALUES(?,?,?,?,?,?)
            """, (username, word, "", "", None, 0))
            db.commit()
            row = db.execute(
                "SELECT word, bookmarked FROM vocab WHERE username=? AND word=? ORDER BY created_at DESC LIMIT 1",
                (username, word)
            ).fetchone()
        except Exception:
            pass

    current = int(row["bookmarked"]) if row else 0
    if toggle:
        new_state = 0 if current else 1
    else:
        new_state = 1 if str(set_to).lower() in ("1","true","yes","on") else 0

    db.execute(
        "UPDATE vocab SET bookmarked=? WHERE username=? AND word=?",
        (new_state, username, word)
    )
    db.commit()

    return {"ok": True, "word": word, "bookmarked": int(new_state) == 1}

# -------------------- Utilities --------------------
@app.route("/initdb")
def route_initdb():
    init_db(seed=True)
    flash("Database initialized.", "success")
    return redirect(url_for("login"))
def parse_correct_field(raw):
    """
    'correct' may be:
      - MCQ letter: 'A'|'B'|'C'|'D'
      - a single string
      - a JSON array of acceptable strings/words
    Return (kind, data)
    """
    if raw is None:
        return ("none", None)
    s = str(raw).strip()
    # try JSON
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return ("list", [str(x).strip() for x in val])
        if isinstance(val, dict):
            return ("dict", val)
    except Exception:
        pass
    # MCQ?
    if s.upper() in {"A","B","C","D"}:
        return ("mcq", s.upper())
    return ("text", s)

def normalize_text(s):
    return " ".join(str(s or "").strip().lower().split())

def check_correct(qtype, correct_raw, response_raw, extra_json):
    """
    Returns (was_correct: bool, explanation_from_extra: str|None)
    For SHORT/LONG we try to be lenient:
      - if 'correct' is a list, accept any exact match (case-insensitive)
      - if 'extra.required_words' exists (list), require all to appear as whole words
    """
    resp = normalize_text(response_raw)
    c_kind, c_val = parse_correct_field(correct_raw)

    # Optional explanation from extra
    exp = None
    try:
        ex = json.loads(extra_json) if extra_json else {}
        if isinstance(ex, dict):
            exp = (ex.get("explanation") or ex.get("hint") or None)
        else:
            ex = {}
    except Exception:
        ex = {}

    if qtype == "mcq":
        if c_kind == "mcq":
            return (str(response_raw).strip().upper() == c_val, exp)
        # fallback: treat textual options
        return (normalize_text(response_raw) == normalize_text(c_val), exp)

    # SHORT/LONG
    # 1) accept list of answers
    if c_kind == "list":
        valid = any(normalize_text(a) == resp for a in c_val)
        if valid:
            return (True, exp)

    # 2) required words
    req_words = []
    if isinstance(ex, dict):
        rw = ex.get("required_words")
        if isinstance(rw, list):
            req_words = [str(w).strip().lower() for w in rw if str(w).strip()]
    if req_words:
        ok = all((" " + resp + " ").find(" " + w + " ") != -1 for w in req_words)
        if ok:
            return (True, exp)

    # 3) plain text compare if available
    if c_kind in {"text"} and c_val:
        return (normalize_text(c_val) == resp, exp)

    # fallback unknown
    return (False, exp)

def update_user_xp_level(conn, username, delta_xp):
    c = conn.cursor()
    # read current
    cur = c.execute("SELECT COALESCE(user_xp,0) AS xp, COALESCE(user_level,1) AS lvl FROM Users WHERE username = ?",
                    (username,)).fetchone()
    if not cur:
        # create if missing
        c.execute("INSERT OR IGNORE INTO Users(username, user_xp, user_level) VALUES (?, 0, 1)", (username,))
        xp, lvl = 0, 1
    else:
        xp, lvl = int(cur["xp"]), int(cur["lvl"])
    xp += max(0, int(delta_xp))
    lvl = 1 + (xp // 100)
    c.execute("UPDATE Users SET user_xp = ?, user_level = ? WHERE username = ?", (xp, lvl, username))
    conn.commit()
    return xp, lvl
# =========================
# Admin Analytics (routes)
# =========================
from flask import render_template, abort

def _require_admin():
    role = session.get("role") or session.get("user_role")
    username = session.get("username")
    if (role not in {"Manager", "Admin", "admin"}) and (username != "testtest"):
        abort(403)

def _parse_date(s):
    # naive date parser YYYY-MM-DD -> keeps as text for sqlite compare
    try:
        s = str(s).strip()
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return s
    except Exception:
        pass
    return None
# --- Admin: per-student analytics JSON ---
# ---------- Admin Analytics (per-student) ----------
from collections import defaultdict
from datetime import date

@app.get("/admin/analytics")
def admin_analytics_page():
    # Simple guard: only logged-in users can view. (Adjust to your admin policy.)
    if "username" not in session:
        return redirect(url_for("login"))

    db = get_db()
    try:
        rows = db.execute("SELECT username FROM users ORDER BY username COLLATE NOCASE").fetchall()
        users = [r["username"] for r in rows] or []
    except Exception:
        users = []
    selected = users[0] if users else None
    return render_template("admin_analytics.html", users=users, selected_user=selected)

@app.get("/admin/analytics/data")
def admin_analytics_data():
    # JSON data for charts
    if "username" not in session:
        return jsonify({"error": "auth required"}), 401

    username = (request.args.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username is required"}), 400

    db = get_db()

    # ----- KPIs -----
    # Total quizzes / attempts / accuracy
    try:
        qz_rows = db.execute("SELECT id, created_at FROM quizzes WHERE username=?", (username,)).fetchall()
        quiz_ids = [r["id"] for r in qz_rows]
        total_quizzes = len(quiz_ids)
    except Exception:
        quiz_ids, total_quizzes = [], 0

    # Gather answers for accuracy
    total_answers = 0
    total_correct = 0
    try:
        if quiz_ids:
            qmarks = ",".join(["?"] * len(quiz_ids))
            ans_rows = db.execute(
                f"SELECT is_correct FROM quiz_answers WHERE quiz_id IN ({qmarks})",
                quiz_ids
            ).fetchall()
            total_answers = len(ans_rows)
            total_correct = sum(1 for r in ans_rows if int(r["is_correct"]) == 1)
    except Exception:
        pass

    overall_accuracy = (total_correct / total_answers) if total_answers else 0.0
    total_attempts = total_answers  # treats each answer as an attempt for “activity mix”

    # Stories read
    stories_read = 0
    try:
        stories_read = db.execute(
            "SELECT COUNT(*) AS c FROM story_reads WHERE username=?",
            (username,)
        ).fetchone()["c"]
    except Exception:
        stories_read = 0

    # New vocab entries as another activity indicator (optional)
    vocab_new = 0
    try:
        vocab_new = db.execute(
            "SELECT COUNT(*) AS c FROM vocab WHERE username=?",
            (username,)
        ).fetchone()["c"]
    except Exception:
        vocab_new = 0

    # Total XP
    total_xp = 0
    try:
        row = db.execute("SELECT xp FROM users WHERE username=?", (username,)).fetchone()
        total_xp = row["xp"] if row else 0
    except Exception:
        total_xp = 0

    # ----- Timeseries: last 30 days attempts & accuracy -----
    # Build date buckets
    today = date.today()
    days = [today - timedelta(days=i) for i in range(29, -1, -1)]
    labels = [d.strftime("%m-%d") for d in days]
    attempts_by_day = {d.isoformat(): 0 for d in days}
    correct_by_day  = {d.isoformat(): 0 for d in days}

    try:
        if quiz_ids:
            qmarks = ",".join(["?"] * len(quiz_ids))
            rows = db.execute(
                f"""
                SELECT qa.is_correct, qa.created_at
                FROM quiz_answers qa
                WHERE qa.quiz_id IN ({qmarks})
                """,
                quiz_ids
            ).fetchall()
            for r in rows:
                ts = (r["created_at"] or "")
                # be tolerant with timestamp parsing
                dkey = (ts[:10] if isinstance(ts, str) and len(ts)>=10 else today.isoformat())
                if dkey in attempts_by_day:
                    attempts_by_day[dkey] += 1
                    if int(r["is_correct"]) == 1:
                        correct_by_day[dkey] += 1
    except Exception:
        pass

    attempts_series = []
    accuracy_series = []
    for d in days:
        key = d.isoformat()
        a = attempts_by_day.get(key, 0)
        c = correct_by_day.get(key, 0)
        attempts_series.append(a)
        accuracy_series.append((c / a) if a else 0.0)

    # ----- Accuracy by level -----
    # If you store level on quizzes table, aggregate; else default single bucket.
    acc_map = defaultdict(lambda: {"a": 0, "c": 0})
    try:
        rows = db.execute(
            "SELECT q.level, qa.is_correct FROM quizzes q "
            "JOIN quiz_answers qa ON qa.quiz_id = q.id "
            "WHERE q.username=?", (username,)
        ).fetchall()
        for r in rows:
            lvl = int(r["level"]) if r["level"] is not None else 1
            acc_map[lvl]["a"] += 1
            acc_map[lvl]["c"] += (1 if int(r["is_correct"]) == 1 else 0)
    except Exception:
        pass

    lvls = sorted(acc_map.keys()) or [1,2,3,4,5]
    acc_by_level_labels = [f"L{l}" for l in lvls]
    acc_by_level_values = [(acc_map[l]["c"] / acc_map[l]["a"]) if acc_map[l]["a"] else 0.0 for l in lvls]

    # ----- Worksheets (4-week cycle) -----
    # Try to find the newest worksheet cycle for the user and compute unlock/submission.
    wk_labels = [f"Week {i}" for i in range(1,5)]
    wk_unlocked = [0,0,0,0]
    wk_submitted = [0,0,0,0]

    try:
        row = db.execute(
            "SELECT id, created_at, payload FROM career_worksheets WHERE username=? ORDER BY id DESC LIMIT 1",
            (username,)
        ).fetchone()
        if row:
            unlocks = compute_week_unlocks(row["created_at"], weeks=4)
            for st in unlocks:
                wk_unlocked[st["week"]-1] = 1 if st["is_unlocked"] else 0

            # If you later store per-week submissions, count them here.
            # Example (if you create such a table):
            # sub_rows = db.execute("SELECT week, COUNT(1) c FROM career_submissions WHERE worksheet_id=? GROUP BY week", (row["id"],)).fetchall()
            # for sr in sub_rows:
            #     wk_submitted[int(sr["week"])-1] = min(1, sr["c"])
            # For now, leave submitted as 0s unless you’ve implemented storage.
    except Exception:
        pass

    # ----- Stories over time (last 30 days) -----
    stories_by_day = {d.isoformat(): 0 for d in days}
    try:
        rows = db.execute(
            "SELECT started_at FROM story_reads WHERE username=?",
            (username,)
        ).fetchall()
        for r in rows:
            ts = (r["started_at"] or "")
            dkey = (ts[:10] if isinstance(ts, str) and len(ts)>=10 else today.isoformat())
            if dkey in stories_by_day:
                stories_by_day[dkey] += 1
    except Exception:
        pass
    stories_series = [stories_by_day[d.isoformat()] for d in days]

    # ----- XP cumulative (roughly by day) -----
    # If you don’t keep a history, show a flat line at current XP.
    xp_series = []
    try:
        current = total_xp
        step = (current / max(1, len(days)-1))
        for i, _ in enumerate(days):
            xp_series.append(round(step * i))
    except Exception:
        xp_series = [0 for _ in days]

    payload = {
        "kpis": {
            "total_quizzes": total_quizzes,
            "overall_accuracy": overall_accuracy,
            "stories_read": stories_read,
            "total_xp": total_xp,
            "total_attempts": total_attempts,
            "vocab_new": vocab_new
        },
        "timeseries": {
            "labels": labels,
            "attempts": attempts_series,
            "accuracy": accuracy_series
        },
        "acc_by_level": {
            "labels": acc_by_level_labels,
            "values": acc_by_level_values
        },
        "worksheets": {
            "labels": wk_labels,
            "unlocked": wk_unlocked,
            "submitted": wk_submitted
        },
        "stories": {
            "labels": labels,
            "counts": stories_series
        },
        "xp": {
            "labels": labels,
            "values": xp_series
        }
    }
    return jsonify(payload)

from datetime import datetime, timedelta

def _table_exists(db, name):
    row = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return bool(row)

def _columns(db, table):
    try:
        return {r["name"] for r in db.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()

def _pick(colset, candidates, default=None):
    for c in candidates:
        if c in colset:
            return c
    return default

from datetime import datetime, timedelta, timezone
from flask import jsonify, session

def _table_exists(db, name):
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return bool(row)

def _columns(db, table):
    try:
        return {r["name"] for r in db.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()

def _pick(colset, candidates):
    for c in candidates:
        if c in colset:
            return c
    return None

@app.get("/api/admin/analytics")
def api_admin_analytics():
    # --- admin gate ---
    if session.get("username") != "testtest":
        return jsonify({"ok": False, "error": "forbidden"}), 403

    db = get_db()

    # --- table/column detection ---
    has_users       = _table_exists(db, "users")
    has_attempts    = _table_exists(db, "attempts")
    has_qanswers    = _table_exists(db, "quiz_answers")
    has_quizzes     = _table_exists(db, "quizzes")
    has_story_reads = _table_exists(db, "story_reads")
    has_vocab       = _table_exists(db, "vocab")

    users_cols = _columns(db, "users") if has_users else set()
    attempts_cols = _columns(db, "attempts") if has_attempts else set()
    qa_cols = _columns(db, "quiz_answers") if has_qanswers else set()
    quizzes_cols = _columns(db, "quizzes") if has_quizzes else set()
    sr_cols = _columns(db, "story_reads") if has_story_reads else set()
    vb_cols = _columns(db, "vocab") if has_vocab else set()

    users_name_col = _pick(users_cols, ["username", "name", "user_name"])
    users_id_col   = _pick(users_cols, ["id", "user_id", "uid"])

    attempts_user_col = _pick(attempts_cols, ["username", "user", "user_name", "user_id", "uid"])
    attempts_time_col = _pick(attempts_cols, ["created_at", "submitted_at", "answered_at", "ts", "timestamp", "time"])

    qa_user_col    = _pick(qa_cols, ["username", "user", "user_name", "user_id", "uid"])
    qa_time_col    = _pick(qa_cols, ["created_at", "submitted_at", "answered_at", "ts", "timestamp", "time"])
    qa_correct_col = _pick(qa_cols, ["is_correct", "correct", "was_correct"])
    qa_conf_col    = _pick(qa_cols, ["confidence"])
    qa_quiz_col    = _pick(qa_cols, ["quiz_id", "qid"])

    qu_id_col    = _pick(quizzes_cols, ["id", "quiz_id"])
    qu_level_col = _pick(quizzes_cols, ["level", "lvl"])

    sr_time_col = _pick(sr_cols, ["started_at", "created_at", "ts", "timestamp", "time"])
    vb_time_col = _pick(vb_cols, ["created_at", "ts", "timestamp", "time"])

    # time window (last 30 days, UTC)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=29)
    dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
    start = dates[0]; end = dates[-1]

    def _date_clause(col):
        # safe wrapper: you must pass a real column name
        return f"date({col}) BETWEEN ? AND ?"

    # Helper to safely run a query only if every required (table, col) exists
    def safe_query(sql, params=()):
        try:
            return db.execute(sql, params).fetchall()
        except Exception:
            return []

    # ======== Timeseries: Attempts/day ========
    attempts_series = [0]*30
    if has_attempts and attempts_time_col:
        rows = safe_query(
            f"SELECT strftime('%Y-%m-%d',{attempts_time_col}) d, COUNT(*) c "
            f"FROM attempts WHERE {_date_clause(attempts_time_col)} GROUP BY d",
            (start, end),
        )
        amap = {r["d"]: r["c"] for r in rows}
        attempts_series = [amap.get(d, 0) for d in dates]

    # ======== Timeseries: Accuracy/day ========
    accuracy_series = [0.0]*30
    if has_qanswers and qa_time_col and qa_correct_col:
        rows = safe_query(
            f"SELECT strftime('%Y-%m-%d',{qa_time_col}) d, "
            f"AVG(CASE WHEN {qa_correct_col}=1 THEN 1.0 ELSE 0.0 END) acc "
            f"FROM quiz_answers WHERE {_date_clause(qa_time_col)} GROUP BY d",
            (start, end),
        )
        accmap = {r["d"]: (r["acc"] or 0.0) for r in rows}
        accuracy_series = [accmap.get(d, 0.0) for d in dates]

    # ======== Timeseries: DAU/day ========
    dau_series = [0]*30
    if has_attempts and attempts_time_col and attempts_user_col:
        rows = safe_query(
            "WITH base AS ("
            f"  SELECT strftime('%Y-%m-%d',{attempts_time_col}) d, {attempts_user_col} u "
            f"  FROM attempts WHERE {_date_clause(attempts_time_col)}"
            ") SELECT d, COUNT(DISTINCT u) dau FROM base GROUP BY d",
            (start, end),
        )
        daumap = {r["d"]: r["dau"] for r in rows}
        dau_series = [daumap.get(d, 0) for d in dates]
    elif has_qanswers and qa_time_col and qa_user_col:
        rows = safe_query(
            "WITH base AS ("
            f"  SELECT strftime('%Y-%m-%d',{qa_time_col}) d, {qa_user_col} u "
            f"  FROM quiz_answers WHERE {_date_clause(qa_time_col)}"
            ") SELECT d, COUNT(DISTINCT u) dau FROM base GROUP BY d",
            (start, end),
        )
        daumap = {r["d"]: r["dau"] for r in rows}
        dau_series = [daumap.get(d, 0) for d in dates]

    # ======== Activity mix (last 30 days) ========
    attempts_30 = 0
    if has_attempts and attempts_time_col:
        row = db.execute(
            f"SELECT COUNT(*) c FROM attempts WHERE {_date_clause(attempts_time_col)}", (start, end)
        ).fetchone()
        attempts_30 = row["c"] if row else 0

    reads_30 = 0
    if has_story_reads and sr_time_col:
        row = db.execute(
            f"SELECT COUNT(*) c FROM story_reads WHERE {_date_clause(sr_time_col)}", (start, end)
        ).fetchone()
        reads_30 = row["c"] if row else 0

    vocab_30 = 0
    if has_vocab and vb_time_col:
        row = db.execute(
            f"SELECT COUNT(*) c FROM vocab WHERE {_date_clause(vb_time_col)}", (start, end)
        ).fetchone()
        vocab_30 = row["c"] if row else 0

    # ======== Levels distribution ========
    level_labels, level_counts = [], []
    if has_users:
        lvl_col = _pick(users_cols, ["level", "lvl"])
        if lvl_col:
            rows = safe_query(
                f"SELECT {lvl_col} lvl, COUNT(*) c FROM users GROUP BY {lvl_col} ORDER BY {lvl_col}"
            )
            level_labels = [str(r["lvl"]) for r in rows]
            level_counts = [r["c"] for r in rows]

    # ======== Accuracy by level (requires quizzes + qa + join cols) ========
    acc_level_labels, acc_level_values = [], []
    if has_quizzes and qu_id_col and qu_level_col and has_qanswers and qa_quiz_col and qa_time_col and qa_correct_col:
        rows = safe_query(
            f"SELECT q.{qu_level_col} lvl, "
            f"AVG(CASE WHEN a.{qa_correct_col}=1 THEN 1.0 ELSE 0.0 END) acc "
            f"FROM quiz_answers a JOIN quizzes q ON q.{qu_id_col}=a.{qa_quiz_col} "
            f"WHERE {_date_clause('a.'+qa_time_col)} GROUP BY q.{qu_level_col} ORDER BY q.{qu_level_col}",
            (start, end),
        )
        acc_level_labels = [str(r["lvl"]) for r in rows]
        acc_level_values = [float(r["acc"] or 0.0) for r in rows]

    # ======== Top students (attempts + accuracy) ========
    top_users = []
    if (has_attempts and attempts_time_col and attempts_user_col) and (has_qanswers and qa_time_col and qa_user_col and qa_correct_col):
        # If IDs used, try to join users for pretty names
        if attempts_user_col in ("user_id", "uid") and qa_user_col in ("user_id", "uid") and has_users and users_id_col and users_name_col:
            rows = safe_query(
                f"""
                WITH A AS (
                  SELECT {attempts_user_col} uid, COUNT(*) attempts_cnt
                  FROM attempts
                  WHERE {_date_clause(attempts_time_col)}
                  GROUP BY {attempts_user_col}
                ),
                B AS (
                  SELECT {qa_user_col} uid, AVG(CASE WHEN {qa_correct_col}=1 THEN 1.0 ELSE 0.0 END) accuracy
                  FROM quiz_answers
                  WHERE {_date_clause(qa_time_col)}
                  GROUP BY {qa_user_col}
                )
                SELECT u.{users_name_col} AS username, A.attempts_cnt, COALESCE(B.accuracy, 0.0) accuracy
                FROM A
                LEFT JOIN B ON B.uid = A.uid
                JOIN users u ON u.{users_id_col} = A.uid
                ORDER BY A.attempts_cnt DESC
                LIMIT 10
                """,
                (start, end, start, end),
            )
            top_users = [
                {"username": r["username"], "attempts": r["attempts_cnt"], "accuracy": float(r["accuracy"] or 0.0)}
                for r in rows
            ]
        else:
            # Treat attempts_user_col as label (e.g., username)
            rows = safe_query(
                f"""
                WITH A AS (
                  SELECT {attempts_user_col} uname, COUNT(*) attempts_cnt
                  FROM attempts
                  WHERE {_date_clause(attempts_time_col)}
                  GROUP BY {attempts_user_col}
                ),
                B AS (
                  SELECT {qa_user_col} uname, AVG(CASE WHEN {qa_correct_col}=1 THEN 1.0 ELSE 0.0 END) accuracy
                  FROM quiz_answers
                  WHERE {_date_clause(qa_time_col)}
                  GROUP BY {qa_user_col}
                )
                SELECT A.uname AS username, A.attempts_cnt, COALESCE(B.accuracy, 0.0) accuracy
                FROM A LEFT JOIN B ON B.uname = A.uname
                ORDER BY A.attempts_cnt DESC
                LIMIT 10
                """,
                (start, end, start, end),
            )
            top_users = [
                {"username": r["username"], "attempts": r["attempts_cnt"], "accuracy": float(r["accuracy"] or 0.0)}
                for r in rows
            ]

    # ======== XP histogram ========
    xp_labels, xp_counts = [], []
    if has_users and "xp" in users_cols:
        rows = safe_query(
            """
            SELECT
              CASE
                WHEN xp <   50 THEN '0-49'
                WHEN xp <  100 THEN '50-99'
                WHEN xp <  200 THEN '100-199'
                WHEN xp <  400 THEN '200-399'
                WHEN xp <  800 THEN '400-799'
                ELSE '800+'
              END AS bucket,
              COUNT(*) c
            FROM users
            GROUP BY bucket
            ORDER BY
              CASE bucket
                WHEN '0-49' THEN 1
                WHEN '50-99' THEN 2
                WHEN '100-199' THEN 3
                WHEN '200-399' THEN 4
                WHEN '400-799' THEN 5
                ELSE 6
              END
            """
        )
        xp_labels = [r["bucket"] for r in rows]
        xp_counts = [r["c"] for r in rows]

    # ======== Confidence buckets (1–5) ========
    conf_buckets = ["1","2","3","4","5"]
    conf_attempts = [0,0,0,0,0]
    conf_accuracy = [0.0,0.0,0.0,0.0,0.0]
    if has_qanswers and qa_time_col and qa_correct_col and qa_conf_col:
        rowsA = safe_query(
            f"SELECT {qa_conf_col} c, COUNT(*) n FROM quiz_answers "
            f"WHERE {_date_clause(qa_time_col)} AND {qa_conf_col} BETWEEN 1 AND 5 "
            f"GROUP BY {qa_conf_col}",
            (start, end),
        )
        rowsB = safe_query(
            f"SELECT {qa_conf_col} c, AVG(CASE WHEN {qa_correct_col}=1 THEN 1.0 ELSE 0.0 END) acc FROM quiz_answers "
            f"WHERE {_date_clause(qa_time_col)} AND {qa_conf_col} BETWEEN 1 AND 5 "
            f"GROUP BY {qa_conf_col}",
            (start, end),
        )
        a_map = {int(r["c"]): r["n"] for r in rowsA if r["c"] is not None}
        b_map = {int(r["c"]): float(r["acc"] or 0.0) for r in rowsB if r["c"] is not None}
        conf_attempts = [a_map.get(i, 0) for i in range(1,6)]
        conf_accuracy = [b_map.get(i, 0.0) for i in range(1,6)]

    # ======== Overall summary ========
    total_attempts, total_corrects, overall_acc = 0, 0, 0.0
    if has_qanswers and qa_time_col and qa_correct_col:
        r1 = db.execute(
            f"SELECT COUNT(*) c FROM quiz_answers WHERE {_date_clause(qa_time_col)}",
            (start, end),
        ).fetchone()
        r2 = db.execute(
            f"SELECT COUNT(*) c FROM quiz_answers WHERE {_date_clause(qa_time_col)} AND {qa_correct_col}=1",
            (start, end),
        ).fetchone()
        total_attempts = (r1["c"] if r1 else 0) or 0
        total_corrects = (r2["c"] if r2 else 0) or 0
        overall_acc = (total_corrects / total_attempts) if total_attempts else 0.0

    return jsonify({
        "ok": True,
        "window": {"start": start, "end": end},
        "dates": dates,
        "summary": {
            "total_attempts": total_attempts,
            "total_corrects": total_corrects,
            "overall_accuracy": overall_acc,
        },
        "timeseries": {
            "attempts": attempts_series,
            "accuracy": accuracy_series,
            "dau": dau_series,
        },
        "activity_mix": {
            "attempts": attempts_30,
            "story_reads": reads_30,
            "vocab_new": vocab_30,
        },
        "levels": {
            "labels": level_labels,
            "counts": level_counts,
            "acc_labels": acc_level_labels,
            "acc_values": acc_level_values,
        },
        "top_students": top_users,
        "xp_histogram": {
            "labels": xp_labels,
            "counts": xp_counts,
        },
        "confidence": {
            "buckets": conf_buckets,
            "attempts": conf_attempts,
            "accuracy": conf_accuracy,
        },
    })
def _simple_sentence(level:int, words:list[str]) -> str:
    # keep very short for young learners
    # e.g., ["police","help","city"] -> "The police help the city."
    w = [w for w in words if isinstance(w, str) and w]
    if not w: return "This is a simple sentence."
    if len(w) == 1: return f"I see a {w[0]}."
    if len(w) == 2: return f"The {w[0]} and {w[1]} work together."
    return f"The {w[0]} {random.choice(['help','protect','build','cook','drive','teach'])} the {w[1]} in the {w[2]}."
def generate_career_worksheet_gpt_4w(job: str, level: int = 1, lang: str = "en") -> dict:
    """
    Generate 4 weekly worksheets for the given job (police, firefighter, etc.).
    Each week targets 40–60 minutes with:
      - Reading: 140–200 words (short, clear sentences, A1–A2)
      - Vocabulary: 8–10 target words [{word, meaning_ko, example_en}]
      - Comprehension: 6 short-answer Qs [{q, a}]
      - MCQ: 6 items (4 options, one correct)
      - Short blanks: 4–6 one-word answers
      - Writing: 2 prompts [{required_words: 3–5, prompt}]
    """
    import json, time, random

    job = (job or "").strip()
    level = max(1, min(int(level or 1), 5))

    sys = (
        "You are generating four weekly kid-friendly English worksheets for Korean EFL elementary learners "
        "(CEFR A1–A2), ages 7–12. Keep text concrete, short sentences, mostly present tense, and SAFE. "
        "Return STRICT JSON only (no code fences)."
    )

    user = {
        "job": job,
        "level": level,
        "lang": lang,
        "shape": {
            "weeks": 4,
            "per_week": {
                "reading_words": [140, 200],
                "vocab_count": [8, 10],
                "comprehension_q": 6,
                "mcq": 6,
                "short_blanks": [4, 6],
                "writing_prompts": 2
            },
            "rules": [
                "Short sentences (8–14 words).",
                "Mostly present tense; avoid idioms and figurative language.",
                "Add very short Korean glosses only when helpful.",
                "Avoid violence, politics, religion, romance.",
                "MCQs: exactly 4 options; exactly one correct.",
                "Short blanks: a single best ONE-WORD answer; 1–3 alternates ok."
            ],
            "schema": {
                "job": "lowercase string",
                "level": "1..5",
                "weeks": [{
                    "week": "1..4 (number)",
                    "title": "Week N • <topic>",
                    "reading": "140–200 word string",
                    "vocab": [{"word": "string", "meaning_ko": "string", "example_en": "string"}],
                    "comprehension": [{"q": "string", "a": "short string"}],
                    "mcq": [{"prompt": "string", "options": ["A","B","C","D"], "answer_index": 0}],
                    "short": [{"prompt":"string with ONE blank","answer":"string","alternates":["optional", "..."]}],
                    "writing": [{"required_words":["w1","w2","w3"], "prompt":"string"}]
                }]
            }
        }
    }

    client = get_openai()  # use your existing helper

    def _ask(max_retries=2):
        last = None
        for i in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0.4,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": json.dumps(user)}
                    ]
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last = e
                time.sleep(0.7 * (i + 1))
        raise last

    raw = _ask()
    # salvage JSON if model wraps it
    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s:
        raw = raw[s:e+1]
    data = json.loads(raw)

    # ---------- basic cleaners ----------
    def _int_1_4(x, default):
        try:
            v = int(float(str(x).strip()))
        except Exception:
            v = default
        return 1 if v < 1 else 4 if v > 4 else v

    def _clean_mcq(items, need=6):
        out = []
        for q in (items if isinstance(items, list) else []):
            if not isinstance(q, dict):
                continue
            prompt = str(q.get("prompt","")).strip()
            ops = q.get("options", [])
            if not isinstance(ops, list): ops = []
            ops = [str(o).strip() for o in ops if str(o).strip()][:4]
            # pad to 4 options if short
            while len(ops) < 4:
                for filler in ("to","in","on","at","with","from"):
                    if filler not in ops:
                        ops.append(filler)
                    if len(ops) == 4: break
            ai = q.get("answer_index", 0)
            try: ai = int(ai)
            except: ai = 0
            ai = max(0, min(3, ai))
            out.append({"prompt": prompt or "Choose the correct word.", "options": ops[:4], "answer_index": ai})
        # ensure at least `need`
        while len(out) < need:
            out.append({
                "prompt": "Choose the correct preposition: I go __ the station.",
                "options": ["to","in","on","at"], "answer_index": 0
            })
        return out[:need]

    def _clean_short(items, min_need=4, max_need=6):
        out = []
        for s in (items if isinstance(items, list) else []):
            if not isinstance(s, dict): continue
            prompt = str(s.get("prompt","")).strip() or "He walked __ the store."
            ans = str(s.get("answer","")).strip() or "to"
            alts = s.get("alternates", [])
            if not isinstance(alts, list): alts = []
            out.append({"prompt": prompt, "answer": ans, "alternates": [str(a).strip() for a in alts if str(a).strip()]})
        # pad
        base = [
            {"prompt":"He walked __ the store.","answer":"to","alternates":["into","toward"]},
            {"prompt":"They meet __ Monday.","answer":"on","alternates":["in","at"]},
            {"prompt":"The cat is __ the table.","answer":"under","alternates":["below","beneath"]},
            {"prompt":"We arrived __ 8 p.m.","answer":"at","alternates":["around"]},
            {"prompt":"Put it __ the box.","answer":"in","alternates":["inside","into"]},
            {"prompt":"She goes __ bus.","answer":"by","alternates":["on"]}
        ]
        i = 0
        while len(out) < min_need and i < len(base):
            out.append(base[i]); i += 1
        return out[:max_need]

    def _clean_vocab(items, need_min=8, need_max=10):
        out = []
        for v in (items if isinstance(items, list) else []):
            if not isinstance(v, dict): continue
            w = str(v.get("word","")).strip()
            if not w: continue
            out.append({
                "word": w,
                "meaning_ko": str(v.get("meaning_ko","")).strip(),
                "example_en": str(v.get("example_en","")).strip()
            })
        # pad with safe placeholders if needed
        fallback = [
            {"word":"uniform","meaning_ko":"제복","example_en":"Police officers wear a uniform at work."},
            {"word":"rescue","meaning_ko":"구조하다","example_en":"Firefighters rescue people from danger."},
            {"word":"community","meaning_ko":"지역사회","example_en":"They help the community every day."},
            {"word":"safety","meaning_ko":"안전","example_en":"Safety is the most important rule."},
            {"word":"teamwork","meaning_ko":"팀워크","example_en":"Teamwork makes hard jobs easier."},
            {"word":"emergency","meaning_ko":"긴급 상황","example_en":"Call 119 in an emergency."},
            {"word":"equipment","meaning_ko":"장비","example_en":"They check their equipment before work."},
            {"word":"protect","meaning_ko":"보호하다","example_en":"Officers protect people and places."},
            {"word":"report","meaning_ko":"보고하다","example_en":"They write a report after each call."},
            {"word":"signal","meaning_ko":"신호","example_en":"Follow the traffic signal at the street."}
        ]
        j = 0
        while len(out) < need_min and j < len(fallback):
            out.append(fallback[j]); j += 1
        return out[:need_max]

    def _clean_comp(items, need=6):
        out = []
        for c in (items if isinstance(items, list) else []):
            if not isinstance(c, dict): continue
            q = str(c.get("q","")).strip()
            a = str(c.get("a","")).strip()
            if q: out.append({"q": q, "a": a})
        # pad if short
        while len(out) < need:
            k = len(out) + 1
            out.append({"q": f"Q{k}. What is one important duty?", "a": "Answers may vary (e.g., help people)."})
        return out[:need]

    def _clean_writing(items, need=2):
        out = []
        for w in (items if isinstance(items, list) else []):
            if not isinstance(w, dict): continue
            req = w.get("required_words", [])
            if not isinstance(req, list) or not req: continue
            prompt = str(w.get("prompt") or f"Write one clear sentence using ALL of these words: {', '.join(req)}.").strip()
            out.append({"required_words": [str(r).strip() for r in req if str(r).strip()][:5], "prompt": prompt})
        # pad if short
        base = [
            {"required_words":["help","team","city"], "prompt":f"Write one sentence about how a {job} helps the city."},
            {"required_words":["safe","people","work"], "prompt":f"Write one sentence that shows how a {job} keeps people safe."},
        ]
        i = 0
        while len(out) < need and i < len(base):
            out.append(base[i]); i += 1
        return out[:need]

    # --------- synthesizers (self-healing when GPT is thin) ---------
    def _synth_reading(job, vocab, level, target_words=(140, 200)):
        # build short sentences and weave vocab examples
        base = [
            f"This unit is about the job of {job}.",
            "People in this job help the community every day.",
            "They work with a team and follow safety rules.",
            "They start with a plan and check their equipment.",
            "Clear steps help them do the job well.",
        ]
        for v in (vocab or [])[:8]:
            w = v.get("word", "work")
            ex = v.get("example_en", f"They often use the word {w} at work.")
            base.append(ex if 6 <= len(ex.split()) <= 16 else f"They often talk about {w}.")
        base += [
            "They talk to people in a simple and kind way.",
            "This job needs practice, focus, and teamwork.",
            "They try to keep everyone safe and calm.",
        ]
        text = " ".join(base)
        while len(text.split()) < target_words[0]:
            text += " They check again and do the next step."
        return text

    def _synth_comp_from_reading(reading, need=6):
        stems = [
            "What is this unit about?",
            "What do they do every day?",
            "What helps them do the job well?",
            "How do they talk to people?",
            "What do they check before work?",
            "What does this job need?",
            "Why are rules important?",
            "Who do they help?"
        ]
        out = []
        for q in stems:
            out.append({"q": q, "a": "Answers may vary."})
            if len(out) >= need: break
        return out[:need]

    def _synth_mcq_from_vocab(vocab, need=6):
        out = []
        options_pool = ["safety","teamwork","report","uniform","signal","rescue","protect","equipment","community"]
        for v in (vocab or []):
            word = v.get("word", "")
            meaning = v.get("meaning_ko", "")
            if not word: continue
            opts = [word]
            for o in options_pool:
                if o != word and o not in opts:
                    opts.append(o)
                if len(opts) == 4: break
            random.shuffle(opts)
            ai = opts.index(word) if word in opts else 0
            out.append({
                "prompt": f"Choose the word that matches: {meaning or 'meaning'}",
                "options": opts[:4],
                "answer_index": ai
            })
            if len(out) >= need: break
        while len(out) < need:
            out.append({
                "prompt": "Choose the correct preposition: I go __ the station.",
                "options": ["to","in","on","at"], "answer_index": 0
            })
        return out[:need]

    def _synth_short(need=5):
        return [
            {"prompt":"He walked __ the store.","answer":"to","alternates":["into","toward"]},
            {"prompt":"They meet __ Monday.","answer":"on","alternates":["in","at"]},
            {"prompt":"The cat is __ the table.","answer":"under","alternates":["below","beneath"]},
            {"prompt":"We arrived __ 8 p.m.","answer":"at","alternates":["around"]},
            {"prompt":"Put it __ the box.","answer":"in","alternates":["inside","into"]},
            {"prompt":"She goes __ bus.","answer":"by","alternates":["on"]},
        ][:need]

    def _synth_writing(job, level, need=2):
        bank = [
            {"required_words":["help","team","city"], "prompt":f"Write one sentence about how a {job} helps the city."},
            {"required_words":["safe","people","work"], "prompt":f"Write one sentence that shows how a {job} keeps people safe."},
            {"required_words":["plan","check","equipment"], "prompt":"Write one sentence using these words."},
        ]
        return bank[:need]

    # ---------- build cleaned weeks with hard minimums ----------
    weeks = data.get("weeks", [])
    if not isinstance(weeks, list): weeks = []

    cleaned = []
    for idx, w in enumerate(weeks, start=1):
        w = w if isinstance(w, dict) else {}
        wk = _int_1_4(w.get("week"), idx)
        title = str(w.get("title", f"Week {wk} • {job.title()}")).strip() or f"Week {wk}"
        reading = str(w.get("reading","")).strip()

        vocab = _clean_vocab(w.get("vocab", []))
        comp  = _clean_comp(w.get("comprehension", []))
        mcq   = _clean_mcq(w.get("mcq", []), need=6)
        short = _clean_short(w.get("short", []), min_need=4, max_need=6)
        writing = _clean_writing(w.get("writing", []), need=2)

        # synthesize when too thin
        if not reading or len(reading.split()) < 120:
            reading = _synth_reading(job, vocab, level)
        if len(vocab) < 8:
            vocab = _clean_vocab(vocab)  # pads with fallbacks
        if len(comp) < 6:
            comp = _synth_comp_from_reading(reading, need=6)
        if len(mcq) < 6:
            mcq = _synth_mcq_from_vocab(vocab, need=6)
        if len(short) < 4:
            short = _synth_short(need=5)
        if len(writing) < 2:
            writing = _synth_writing(job, level, need=2)

        cleaned.append({
            "week": wk,
            "title": title,
            "reading": reading,
            "vocab": vocab,
            "comprehension": comp,
            "mcq": mcq,
            "short": short,
            "writing": writing
        })

    # ensure exactly 4 weeks
    while len(cleaned) < 4:
        base = cleaned[-1] if cleaned else {
            "week": len(cleaned)+1, "title": f"Week {len(cleaned)+1}",
            "reading":"", "vocab":[], "comprehension":[], "mcq":[], "short":[], "writing":[]
        }
        dup = dict(base)
        dup["week"] = len(cleaned)+1
        if not dup.get("reading"):
            dup["reading"] = _synth_reading(job, [], level)
        if not dup.get("vocab"):
            dup["vocab"] = _clean_vocab([])
        if not dup.get("comprehension"):
            dup["comprehension"] = _synth_comp_from_reading(dup["reading"], need=6)
        if not dup.get("mcq"):
            dup["mcq"] = _synth_mcq_from_vocab(dup["vocab"], need=6)
        if not dup.get("short"):
            dup["short"] = _synth_short(need=5)
        if not dup.get("writing"):
            dup["writing"] = _synth_writing(job, level, need=2)
        cleaned.append(dup)
    cleaned = cleaned[:4]

    return {
        "job": (data.get("job") or job).lower(),
        "_level": level,
        "_lang": lang,
        "weeks": cleaned
    }


@app.route("/career", methods=["GET", "POST"])
def career():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]

    if request.method == "POST":
        job = (request.form.get("job") or "").strip()
        try:
            level = int(request.form.get("level") or 1)
        except:
            level = 1

        if not job:
            flash("Please enter a job.", "warning")
            return render_template("career_form.html")

        # try:
        ws = generate_career_worksheet_gpt_4w(job, level)
        # except Exception as e:
        #     flash(f"Could not generate worksheet: {e}", "danger")
        #     return render_template("career_form.html")

        db = get_db()
        cur = db.execute(
            "INSERT INTO career_worksheets(username, job, level, payload) VALUES(?,?,?,?)",
            (username, ws["job"], ws["_level"], json.dumps(ws))
        )
        db.commit()
        ws_id = cur.lastrowid
        # After generation → overview page with locks
        return redirect(url_for("career_overview", ws_id=ws_id))

    return render_template("career_form.html")
@app.get("/career/<int:ws_id>/overview")
def career_overview(ws_id):
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]

    db = get_db()
    row = db.execute("SELECT id, username, job, level, created_at, payload FROM career_worksheets WHERE id=?", (ws_id,)).fetchone()
    if not row or row["username"] != username:
        flash("Worksheet not found.", "danger")
        return redirect(url_for("career"))

    payload = json.loads(row["payload"])
    unlocks = compute_week_unlocks(row["created_at"], weeks=4)
    # Make a lightweight view model
    weeks_vm = []
    for st in unlocks:
        weeks_vm.append({
            "week": st["week"],
            "title": payload["weeks"][st["week"]-1].get("title", f"Week {st['week']}"),
            "unlock_at_str": _fmt_kst(st["unlock_at"]),
            "is_unlocked": st["is_unlocked"]
        })
    return render_template(
        "career_overview_4w.html",
        ws_id=row["id"],
        job=payload.get("job",""),
        level=payload.get("_level", row["level"]),
        created_at=row["created_at"],
        weeks_vm=weeks_vm
    )

@app.get("/career/<int:ws_id>")
def career_view(ws_id):
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]

    week = int(request.args.get("week") or 1)
    week = max(1, min(4, week))

    db = get_db()
    row = db.execute("SELECT id, username, job, level, created_at, payload FROM career_worksheets WHERE id=?", (ws_id,)).fetchone()
    if not row or row["username"] != username:
        flash("Worksheet not found.", "danger")
        return redirect(url_for("career"))

    payload = json.loads(row["payload"])
    unlocks = compute_week_unlocks(row["created_at"], weeks=4)
    st = next(s for s in unlocks if s["week"] == week)
    if not st["is_unlocked"]:
        flash(f"Week {week} unlocks at { _fmt_kst(st['unlock_at']) }.", "warning")
        return redirect(url_for("career_overview", ws_id=ws_id))

    W = payload["weeks"][week-1]
    return render_template(
        "career_worksheet_4w.html",
        ws_id=ws_id,
        job=payload.get("job", ""),
        level=payload.get("_level", row["level"]),
        created_at=row["created_at"],
        week=week,
        title=W["title"],
        reading=W["reading"],
        sections=W.get("sections", []),
        mcq=W.get("mcq", []),
        short=W.get("short", []),
        vocab=W.get("vocab", []),  # <-- add this
        comprehension=W.get("comprehension", []),  # <-- and this
        writing=W.get("writing", [])  # <-- and this
    )


@app.post("/career/<int:ws_id>/submit")
def career_submit(ws_id):
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    week = int(request.args.get("week") or 1)
    week = max(1, min(4, week))

    db = get_db()
    row = db.execute("SELECT id, username, created_at, payload FROM career_worksheets WHERE id=?", (ws_id,)).fetchone()
    if not row or row["username"] != username:
        flash("Worksheet not found.", "danger")
        return redirect(url_for("career"))

    unlocks = compute_week_unlocks(row["created_at"], weeks=4)
    st = next(s for s in unlocks if s["week"] == week)
    if not st["is_unlocked"]:
        flash(f"Week {week} is locked until { _fmt_kst(st['unlock_at']) }.", "warning")
        return redirect(url_for("career_overview", ws_id=ws_id))

    # ... your grading logic here ...
    flash(f"Week {week} submitted.", "success")
    return redirect(url_for("career_overview", ws_id=ws_id))
# Go to the most recent worksheet's overview, or to the generator form if none exist
@app.get("/career/latest")
def career_latest():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]

    db = get_db()
    row = db.execute(
        "SELECT id FROM career_worksheets WHERE username=? ORDER BY id DESC LIMIT 1",
        (username,)
    ).fetchone()
    if row:
        return redirect(url_for("career_overview", ws_id=row["id"]))
    # no worksheet yet → send them to the form
    return redirect(url_for("career"))

def grant_xp(db, username: str, amount: int):
    if amount <= 0:
        return 0
    db.execute("UPDATE users SET xp = COALESCE(xp,0) + ? WHERE username=?", (amount, username))
    return amount

def _normalize_bool(x):
    return 1 if x else 0

# === (C) LLM or heuristic quiz generator for a story ===
# We’ll generate 8 Qs: 6 MCQ + 2 LONG (open answer; correctness = contains all required words)
from flask import current_app
def generate_story_quiz_from_text(story_title: str, story_text: str) -> dict:
    """
    Returns:
      { "mcq":[ {id,prompt,options[4],answer_index} ] }
    Produces 8 high-quality reading-comprehension MCQs for kids.
    """

    # -------- TRIVIAL FALLBACK (no GPT) ----------
    def trivial(text):
        sents = [s.strip() for s in re.split(r"[.!?]", text or "") if len(s.strip().split()) >= 4]
        mcq = []
        for i, s in enumerate(sents[:8]):
            mcq.append({
                "id": f"m{i+1}",
                "prompt": f"What is something that happens in this part of the story?\n“{s}”",
                "options": ["It is mentioned", "It is not mentioned", "It is the opposite", "I don’t know"],
                "answer_index": 0
            })
        while len(mcq) < 8:
            mcq.append({
                "id": f"m{len(mcq)+1}",
                "prompt": "What is true according to the story?",
                "options": ["A detail from the story", "Not in story", "Wrong detail", "Random guess"],
                "answer_index": 0
            })
        return {"mcq": mcq}

    # -------- GPT ROUTE ----------
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        clean_story = story_text[:12000]  # safety cutoff

        system_msg = (
            "You are a professional ESL reading-comprehension quiz generator for young learners. "
            "You must make questions based ONLY on the story text. "
            "NO outside knowledge. Keep English simple. "
            "Each question should ask about characters, events, setting, feelings, or simple details."
        )

        user_msg = f"""
Create **exactly 8 multiple-choice reading-comprehension questions** for the following story.

Rules:
- EACH question must be answerable from the story.
- NO keyword questions. NO extract-the-first-word. NO grammar questions.
- Ask about story events, what characters do, who they are, where things happen,
  what objects appear, or why something happens.
- Difficulty: for 6–10 year-old ESL students.
- 4 options per question.
- One correct answer.
- Keep prompts very short (1–2 sentences max).
- Return ONLY JSON matching this schema:

{{
  "mcq":[
    {{
      "id":"m1",
      "prompt":"string",
      "options":["A","B","C","D"],
      "answer_index": 0
    }},
    ...
  ]
}}

Story Title: {story_title}

Story Text:
\"\"\"
{clean_story}
\"\"\"
        """

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.4,
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg},
            ],
            response_format={"type":"json_object"}
        )

        data = json.loads(resp.choices[0].message.content)
        mcq = data.get("mcq") or []

        # validate
        final = []
        for i, q in enumerate(mcq):
            opts = q.get("options", [])
            if not (isinstance(opts, list) and len(opts) == 4):
                continue
            ai = q.get("answer_index")
            if ai is None or not (0 <= ai < 4):
                continue
            final.append({
                "id": f"m{i+1}",
                "prompt": q.get("prompt","").strip(),
                "options": [str(o) for o in opts],
                "answer_index": int(ai)
            })

        # pad if less than 8 (rare)
        while len(final) < 8:
            k = len(final) + 1
            final.append({
                "id": f"m{k}",
                "prompt": "What is true according to the story?",
                "options": ["A story detail", "Not in story", "Wrong", "Random"],
                "answer_index": 0
            })

        return {"mcq": final[:8]}

    except Exception as e:
        print("GPT quiz error:", e)
        return trivial(story_text)

# === (D) Routes: fetch quiz for a story and submit answers ===

@app.route("/api/story/<slug>/quiz", methods=["GET"])
def api_story_quiz(slug):
    if "username" not in session:
        return jsonify({"error":"auth required"}), 401
    # You probably already load story content when rendering; fetch again or pass via client.
    # Minimal: accept ?title= and ?content= if you don’t have a story table:
    title = (request.args.get("title") or "").strip()
    content = (request.args.get("content") or "").strip()
    quiz = generate_story_quiz_from_text(title, content)
    return jsonify(quiz)

@app.route("/api/story_quiz/submit", methods=["POST"])
def api_story_quiz_submit():
    if "username" not in session:
        return jsonify({"error":"auth required"}), 401
    username = session["username"]
    data = request.get_json(force=True) or {}
    slug = (data.get("story_slug") or "").strip()

    mcq_items = data.get("mcq") or []        # [{id, answer_index(int)}]
    long_items = data.get("long") or []      # [{id, answer_text(str), required_words:[...]}]

    # Grade MCQ
    correct = 0
    total = 0

    # Expect client to echo back the quiz structure for grading (simple MVP).
    # Each mcq item should have: id, answer_index (user), solution_index (server/trusted)
    for it in mcq_items:
        try:
            total += 1
            if int(it.get("answer_index")) == int(it.get("solution_index")):
                correct += 1
        except Exception:
            pass

    # Grade LONG: correct iff answer contains ALL required words (case-insensitive, word-boundary-ish)
    def contains_all_words(ans: str, req: list[str]) -> bool:
        s = " " + (ans or "").lower() + " "
        return all((" " + str(w or "").lower().strip() + " ") in s for w in req if w)

    for it in long_items:
        req = it.get("required_words") or []
        ans = it.get("answer_text") or ""
        if req:
            total += 1
            if contains_all_words(ans, req):
                correct += 1

    score = 0.0 if total == 0 else round(100.0 * correct / total, 1)
    passed = score >= 80.0

    # Award once per attempt (no anti-farming here; add guards if you want daily/once)
    award = 0
    if passed:
        db = get_db()
        award = grant_xp(db, username, 50)   # +50 XP for pass
        db.execute("""INSERT INTO StoryQuizAttempts(username, story_slug, total, correct, score, passed, awarded_xp)
                      VALUES (?, ?, ?, ?, ?, ?, ?)""",
                   (username, slug or "-", total, correct, score, _normalize_bool(passed), award))
        db.commit()
    else:
        db = get_db()
        db.execute("""INSERT INTO StoryQuizAttempts(username, story_slug, total, correct, score, passed, awarded_xp)
                      VALUES (?, ?, ?, ?, ?, ?, 0)""",
                   (username, slug or "-", total, correct, score, _normalize_bool(passed)))
        db.commit()

    return jsonify({"total": total, "correct": correct, "score": score, "passed": passed, "xp_awarded": award})
def _normalize_bool(b): return 1 if bool(b) else 0

def _user_required():
    if "username" not in session:
        return None, (jsonify({"error":"auth required"}), 401)
    return session["username"], None
def _load_saved_quiz(db, username: str, slug: str):
    row = db.execute(
        "SELECT quiz_json, created_at FROM StoryQuizzes WHERE username=? AND story_slug=?",
        (username, slug)
    ).fetchone()
    if not row:
        return None, None
    try:
        return json.loads(row["quiz_json"]), row["created_at"]
    except Exception:
        return None, row["created_at"]

def _save_quiz(db, username: str, slug: str, quiz: dict):
    db.execute(
        """
        INSERT INTO StoryQuizzes(username, story_slug, quiz_json)
        VALUES(?,?,?)
        ON CONFLICT(username, story_slug)
        DO UPDATE SET quiz_json=excluded.quiz_json, created_at=CURRENT_TIMESTAMP
        """,
        (username, slug, json.dumps(quiz, ensure_ascii=False))
    )
    db.commit()


def _latest_status(db, username, slug):
    att = db.execute("""
        SELECT total, correct, score, passed
        FROM StoryQuizAttempts
        WHERE username=? AND story_slug=?
        ORDER BY id DESC LIMIT 1
    """, (username, slug)).fetchone()
    if not att:
        return {"state":"not_started"}
    st = {
        "state": "passed" if att["passed"] else "failed",
        "score": int(att["correct"]),
        "total": int(att["total"]),
        "xp": 0
    }
    return st

@app.get("/api/story_quiz/<slug>/status")
def api_story_quiz_status(slug):
    username, err = _user_required()
    if err: return err
    db = get_db()
    row = db.execute(
        """
        SELECT score, total, passed, xp, created_at
          FROM StoryQuizAttempts
         WHERE username=? AND story_slug=?
         ORDER BY id DESC LIMIT 1
        """, (username, slug)
    ).fetchone()
    if not row:
        return jsonify({"state": "not_started"})
    state = "passed" if row["passed"] else "failed"
    return jsonify({"state": state, "score": row["score"], "total": row["total"], "xp": row["xp"]})

# (Optional) allow client to POST a simple status blob after local grade
@app.post("/api/story_quiz/<slug>/status")
def api_story_quiz_status_set(slug):
    username, err = _user_required()
    if err: return err
    data = request.get_json(silent=True) or {}
    # store as an attempt so it shows up consistently
    total = int(data.get("total") or 0)
    score = int(data.get("score") or 0)
    passed = 1 if (data.get("state")=="passed") else 0
    db = get_db()
    db.execute("""INSERT INTO StoryQuizAttempts(username, story_slug, total, correct, score, passed, awarded_xp)
                  VALUES (?,?,?,?,?,?,?)""",
               (username, slug, total, score, float(0 if not total else 100.0*score/max(total,1)), passed, int(data.get("xp") or 0)))
    db.commit()
    return jsonify({"ok": True})

@app.get("/api/story_quiz/<slug>/get")
def api_story_quiz_get(slug):
    username, err = _user_required()
    if err: return err
    db = get_db()
    quiz, created_at = _load_saved_quiz(db, username, slug)
    if not quiz:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"quiz": quiz, "created_at": created_at})
@app.post("/api/story_quiz/<slug>/generate")
def api_story_quiz_generate(slug):
    """
    Generates quiz only if not cached, unless force=true is sent.
    Body (optional): { "title": "...", "content": "...", "force": true/false }
    """
    username, err = _user_required()
    if err: return err

    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force") or request.args.get("force") == "true")
    title   = (payload.get("title")   or request.args.get("title")   or "").strip()
    content = (payload.get("content") or request.args.get("content") or "").strip()

    db = get_db()

    # 1) Cache-first (skip if force)
    if not force:
        cached, _created = _load_saved_quiz(db, username, slug)
        if cached:
            # Ensure keys for front-end
            cached.setdefault("mcq", [])
            cached.setdefault("short", [])
            cached.setdefault("long", [])
            return jsonify(cached)

    # 2) Generate afresh (your existing generator; returns {"mcq":[], "short":[], "long":[]})
    quiz = generate_story_quiz_from_text(title, content)
    quiz.setdefault("mcq", [])
    quiz.setdefault("short", [])
    quiz.setdefault("long", [])

    # 3) Save
    _save_quiz(db, username, slug, quiz)

    return jsonify(quiz)


@app.post("/api/story_quiz/<slug>/status")
def api_story_quiz_status_post(slug):
    username, err = _user_required()
    if err: return err
    payload = request.get_json(silent=True) or {}
    score = int(payload.get("score") or 0)
    total = int(payload.get("total") or 0)
    passed = 1 if payload.get("state") == "passed" else 0
    xp = int(payload.get("xp") or 0)
    db = get_db()
    db.execute(
        "INSERT INTO StoryQuizAttempts(username, story_slug, score, total, passed, xp) VALUES(?,?,?,?,?,?)",
        (username, slug, score, total, passed, xp)
    )
    # also bump user XP if you track it
    if xp > 0:
        db.execute("UPDATE users SET xp = COALESCE(xp,0) + ? WHERE username=?", (xp, username))
    db.commit()
    return jsonify({"ok": True})
@app.post("/api/story_quiz/<slug>/submit")
def api_story_quiz_submit_slug(slug):
    """
    Front-end sends: { "answers": { "mcq_0": 1, "short_0": "word", "long_0": "..." } }
    We grade against the saved quiz for (username, slug).
    """
    username, err = _user_required()
    if err: return err
    data = request.get_json(force=True) or {}
    answers = data.get("answers") or {}

    db = get_db()
    quiz = _load_saved_quiz(db, username, slug)
    if not quiz:
        return jsonify({"error":"no_quiz"}), 400

    total = 0
    correct = 0

    # MCQ
    for i, it in enumerate(quiz.get("mcq", [])):
        total += 1
        key = f"mcq_{i}"
        user_idx = answers.get(key)
        try:
            ok = (user_idx is not None and int(user_idx) == int(it.get("answer_index")))
        except Exception:
            ok = False
        if ok: correct += 1

    # Short
    def _norm(s):
        return re.sub(r"\s+", " ", (s or "").strip().lower())
    for i, it in enumerate(quiz.get("short", [])):
        total += 1
        key = f"short_{i}"
        a = _norm(answers.get(key))
        gold = _norm(it.get("answer"))
        alts = [_norm(x) for x in (it.get("alternates") or [])]
        if a and (a == gold or a in alts): correct += 1

    # Long (must contain all required words)
    def _tokens(s): return re.findall(r"[a-z]+", (s or "").lower())
    for i, it in enumerate(quiz.get("long", [])):
        total += 1
        key = f"long_{i}"
        req = [str(w or "").lower() for w in (it.get("required_words") or []) if w]
        bag = set(_tokens(answers.get(key)))
        if all(w in bag for w in req): correct += 1

    score_pct = 0.0 if total == 0 else round(100.0 * correct / total, 1)
    passed = score_pct >= 80.0

    # XP on pass (match your earlier pattern)
    award = 0
    if passed:
        award = grant_xp(get_db(), username, 50)  # or your desired reward

    db.execute("""INSERT INTO StoryQuizAttempts(username, story_slug, total, correct, score, passed, awarded_xp)
                  VALUES (?,?,?,?,?,?,?)""",
               (username, slug, total, correct, score_pct, _normalize_bool(passed), award))
    db.commit()

    return jsonify({"score": correct, "total": total, "passed": passed, "xp_awarded": award})

# -------------------- Main --------------------
if __name__ == "__main__":
    with app.app_context():
        init_db(seed=True)
    app.run(debug=True)
