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
    try:    age = int(request.args.get("age")) if request.args.get("age") else None
    except: age = None
    try:    difficulty = int(request.args.get("difficulty")) if request.args.get("difficulty") else None
    except: difficulty = None

    # NEW: optional "ready only" toggle
    ready_only = (request.args.get("ready") == "1")

    # pagination inputs
    try:    page = max(1, int(request.args.get("page", 1)))
    except: page = 1
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

    # Sort: generated first, then by title for stability
    items_all.sort(key=lambda s: (0 if s["slug"] in generated else 1, (s.get("title") or "").lower()))

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
        # NEW: pass these so the template can show badges and keep the toggle state
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

def gpt_generate_quiz(level: int):
    """
    Ask GPT for three arrays: mcq, short, long, sized by distribution_for_level(level).
    Ages: 6–10 only; vocabulary and grammar must be age-appropriate.
    """
    if not openai_client:
        print("[GPT QUIZ] client not available -> returning None")
        return None

    mcq_need, short_need, long_need = distribution_for_level(level)

    level_specs = {
        1: dict(
            cefr="CEFR A0–A1",
            mcq_styles=["basic synonym (happy→joyful)", "very short fill-in (in/on/at/to)"],
            short_hint="prepositions (in/on/at/to) and easy adverbs (slowly, quickly)",
            long_len="8–14 words",
            long_words_range=(2, 3),
            examples_words=["dog","park","run","red","big","play"]
        ),
        2: dict(
            cefr="CEFR A1",
            mcq_styles=["synonym/antonym", "fill-in with common prepositions/adverbs"],
            short_hint="prepositions/adverbs only; 1-word answers",
            long_len="10–16 words",
            long_words_range=(3, 4),
            examples_words=["school","friend","story","river","quiet","walk","under","over"]
        ),
        3: dict(
            cefr="CEFR A1–A2",
            mcq_styles=["synonym + distractors", "short cloze (prepositions/adverbs)"],
            short_hint="exact 1-word answers, still simple",
            long_len="12–18 words",
            long_words_range=(3, 4),
            examples_words=["because","before","after","market","music","quickly","carefully"]
        ),
        4: dict(
            cefr="CEFR A2",
            mcq_styles=["synonym/closest meaning", "cloze with time/place prepositions"],
            short_hint="one-word answers; avoid rare words",
            long_len="14–20 words",
            long_words_range=(4, 5),
            examples_words=["adventure","help","learn","between","across","often","always"]
        ),
        5: dict(
            cefr="CEFR A2+",
            mcq_styles=["closest meaning; keep options concrete", "cloze with adverbs of manner/frequency"],
            short_hint="one-word answers (preposition/adverb/conjunction like 'because')",
            long_len="16–22 words",
            long_words_range=(4, 5),
            examples_words=["carefully","before","during","together","message","choice","bright","outside"]
        ),
    }
    spec = level_specs.get(int(level), level_specs[2])

    system = (
        "You are an English quiz generator for children aged 6–10.\n"
        "Return STRICT JSON with exactly three keys: 'mcq', 'short', and 'long'.\n"
        "No extra keys or commentary. Keep language simple, kid-friendly, and concrete.\n\n"
        "Formats:\n"
        "• mcq: array of items {prompt, options[4], answer_index(int 0..3)}\n"
        "  - Prompts must be short and clear. Options must be single words (no phrases), age-appropriate.\n"
        "• short: array of items {prompt, answer, alternates(optional array of strings), hint(optional string), explanation(optional string)}\n"
        "  - answer is the BEST one-word choice (preposition/adverb/conjunction like 'because').\n"
        "  - alternates are other ONE-WORD responses that are grammatically acceptable in some contexts (e.g., 'into' when 'to' is best).\n"
        "  - hint is one brief child-friendly sentence; explanation is a one-sentence reason why the BEST answer is correct.\n"
        "• long: array of items {required_words[list of 2..6 strings], prompt(optional)}\n"
        "  - If prompt is omitted, the app will show: 'Write one clear sentence using ALL of these words: ...'\n"
        "  - required_words must be easy words only; no proper nouns; no slang; no scary/violent themes.\n\n"
        "CRITICAL MCQ RULES:\n"
        "• MCQs MUST have exactly one objectively correct answer.\n"
        "• Avoid vague 'which sounds best/most suitable' phrasings.\n"
        "• DO NOT create 'adverb of manner' MCQs where multiple options (e.g., 'quickly' vs 'slowly') could fit.\n"
        "• Prefer synonyms/antonyms, simple definitions, or grammar rules with a single correct choice.\n"
        "• When multiple words can fit a sentence, use a SHORT item instead and put other acceptable words in 'alternates'.\n"
    )

    user_payload = {
        "age_range": "6–10",
        "cefr_target": spec["cefr"],
        "level": int(level),
        "counts": {"mcq": mcq_need, "short": short_need, "long": long_need},
        "constraints": {
            "use_only_kid_friendly_words": True,
            "avoid_topics": ["violence", "romance", "politics", "religion", "scary content"],
            "options_must_be_single_words": True,
            "short_answer_is_single_word_only": True,
        },
        "mcq_styles": spec["mcq_styles"],
        "short_rule": spec["short_hint"],
        "long_rule": {
            "target_sentence_length": spec["long_len"],
            "required_words_count_range": list(spec["long_words_range"]),
            "example_easy_words": spec["examples_words"],
        }
    }

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            temperature=0.4,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)

        # ---- NEW: sanitize / de-ambiguate ----
        data = sanitize_gpt_quiz_payload(data)

        # normalize keys + slice
        if "long" not in data and "sentence" in data and isinstance(data["sentence"], list):
            data["long"] = data["sentence"]
        if "short" not in data: data["short"] = []
        if "mcq" not in data:   data["mcq"]   = []

        if not isinstance(data.get("mcq", []), list):   data["mcq"] = []
        if not isinstance(data.get("short", []), list): data["short"] = []
        if not isinstance(data.get("long", []), list):  data["long"] = []

        data["mcq"]   = data["mcq"][:mcq_need]
        data["short"] = data["short"][:short_need]
        data["long"]  = data["long"][:long_need]

        return data
    except Exception as e:
        print("[GPT QUIZ] exception:", repr(e))
        return None

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


# -------------------- Local unique generators (fallbacks) --------------------
def add_mcq_question_unique(db, quiz_id: int, level: int, pos: int,
                            used_norm_prompts: set, max_tries: int = 8) -> bool:
    """
    Generates a unique MCQ using WORD_POOLS[level].
    Two modes:
      - synonym mode (preferred when available)
      - fill-in-the-blank (preposition/adverb)
    """
    for _ in range(max_tries):
        pool = WORD_POOLS[level]

        # Prefer synonym mode if available
        if random.random() < 0.6 and pool.get("synonyms"):
            base_word = random.choice(list(pool["synonyms"].keys()))
            correct = random.choice(pool["synonyms"][base_word])
            distractors = []
            for cat in ("nouns", "verbs", "adjectives"):
                distractors += random.sample(pool[cat], min(3, len(pool[cat])))
            distractors = [w for w in distractors if w != correct][:3]
            if len(distractors) < 3:
                continue

            options = [correct] + distractors
            random.shuffle(options)
            correct_letter = "ABCD"[options.index(correct)]
            prompt = f"Choose the synonym of “{base_word}”."
            extra = {"mode": "synonym", "base": base_word, "pos": pos}

        else:
            # Fill-in-the-blank: preposition or adverb
            preps = WORD_POOLS[level]["preps"]
            adverbs = WORD_POOLS[level]["adverbs"]

            if random.random() < 0.5 and len(preps) >= 4:
                correct = random.choice(preps)
                distractors = random.sample([p for p in preps if p != correct], 3)
                sentence = "I will meet you ___ the park."
                prompt = f"Fill in the blank with the best preposition:\n“{sentence}”"
                options = [correct] + distractors
            else:
                correct = random.choice(adverbs)
                distractors = random.sample([a for a in adverbs if a != correct], 3)
                base = random.choice(WORD_POOLS[level]["verbs"])
                sentence = f"She will {base} ___ to finish the task."
                prompt = f"Fill in the blank with the best adverb:\n“{sentence}”"
                options = [correct] + distractors

            random.shuffle(options)
            correct_letter = "ABCD"[options.index(correct)]
            extra = {"mode": "fill", "sentence": sentence, "pos": pos}

        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            continue

        db.execute(
            """INSERT INTO quiz_questions
               (quiz_id,qtype,prompt,choice_a,choice_b,choice_c,choice_d,correct,extra)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (quiz_id, "mcq", prompt, options[0], options[1], options[2], options[3], correct_letter, json.dumps(extra))
        )
        used_norm_prompts.add(n)
        return True

    return False
def add_short_question_unique(db, quiz_id: int, level: int, pos: int,
                              used_norm_prompts: set, max_tries: int = 8) -> bool:
    """
    Local fallback SHORT generator (kept for safety).
    Now includes an 'explanation' in extra for better feedback.
    """
    for _ in range(max_tries):
        pool = WORD_POOLS[level]
        use_prep = random.random() < 0.7

        if use_prep:
            correct = "to"
            sentence = "He walked ___ the store."
            prompt = f"Fill the blank with the best preposition:\n“{sentence}”"
            alternates = ["into", "toward"]
            hint = "Use 'to' for destination; 'into' means going inside; 'toward' means in the direction of."
            explanation = "Here the sentence describes a destination (the store), so 'to' fits best."
        else:
            correct = random.choice(pool["adverbs"])
            verb = random.choice(pool["verbs"])
            sentence = f"They {verb} ___ to catch the bus."
            prompt = f"Fill the blank with the best adverb:\n“{sentence}”"
            alternates = []
            hint = None
            explanation = None

        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            continue

        extra = {
            "mode": "short",
            "sentence": sentence,
            "pos": pos,
            "alternates": alternates,
            "hint": hint,
            "explanation": explanation,
        }
        db.execute(
            """INSERT INTO quiz_questions (quiz_id,qtype,prompt,correct,extra)
               VALUES (?,?,?,?,?)""",
            (quiz_id, "short", prompt, correct, json.dumps(extra))
        )
        used_norm_prompts.add(n)
        return True

    return False




def add_long_question_unique(db, quiz_id: int, level: int, pos: int,
                             used_norm_prompts: set, max_tries: int = 8) -> bool:
    """
    Generates a unique LONG item with 2–5 required words.
    """
    for _ in range(max_tries):
        pool = WORD_POOLS[level]
        required_count = {1: 2, 2: 3, 3: 3, 4: 4, 5: 5}[level]

        picks = []
        picks += random.sample(pool["nouns"], min(2, len(pool["nouns"])))
        picks += random.sample(pool["verbs"], min(2, len(pool["verbs"])))
        picks += random.sample(pool["adjectives"], min(2, len(pool["adjectives"])))
        random.shuffle(picks)
        required = picks[:required_count]

        prompt = f"Write a single clear sentence using ALL of these words: {', '.join(required)}."
        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            continue

        extra = {"required_words": required, "pos": pos}
        db.execute(
            """INSERT INTO quiz_questions (quiz_id,qtype,prompt,correct,extra)
               VALUES (?,?,?,?,?)""",
            (quiz_id, "long", prompt, json.dumps(required), json.dumps(extra))
        )
        used_norm_prompts.add(n)
        return True

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
    db = get_db()
    cur = db.execute("INSERT INTO quizzes(username, level) VALUES(?, ?)", (username, level))
    quiz_id = cur.lastrowid

    mcq_need, short_need, long_need = distribution_for_level(level)
    gpt = gpt_generate_quiz(level) or {}

    gpt_mcq   = list(gpt.get("mcq",   []))
    gpt_short = list(gpt.get("short", []))
    gpt_long  = list(gpt.get("long", gpt.get("sentence", [])))

    if len(gpt_mcq)   < mcq_need:   raise RuntimeError(f"GPT returned {len(gpt_mcq)} MCQ, need {mcq_need}")
    if len(gpt_short) < short_need: raise RuntimeError(f"GPT returned {len(gpt_short)} short, need {short_need}")
    if len(gpt_long)  < long_need:  raise RuntimeError(f"GPT returned {len(gpt_long)} long, need {long_need}")

    positions = list(range(16))
    random.shuffle(positions)

    used_norm_prompts = set()

    # MCQ
    for _ in range(mcq_need):
        pos = positions.pop()
        item = gpt_mcq.pop(0)
        prompt  = str(item.get("prompt", "")).strip()
        options = item.get("options", [])
        ai      = item.get("answer_index", None)
        if not (prompt and isinstance(options, list) and len(options) >= 4 and isinstance(ai, int) and 0 <= ai < 4):
            raise RuntimeError("Bad MCQ item from GPT")
        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            raise RuntimeError("Duplicate MCQ prompt from GPT")

        choice_a, choice_b, choice_c, choice_d = options[:4]
        correct_letter = "ABCD"[ai]
        extra = {"mode": "gpt", "pos": pos}
        db.execute(
            """INSERT INTO quiz_questions
               (quiz_id,qtype,prompt,choice_a,choice_b,choice_c,choice_d,correct,extra)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (quiz_id, "mcq", prompt, choice_a, choice_b, choice_c, choice_d, correct_letter, json.dumps(extra))
        )
        used_norm_prompts.add(n)

    # SHORT (captures alternates + hint + explanation)
    for _ in range(short_need):
        pos = positions.pop()
        item = gpt_short.pop(0)
        prompt = str(item.get("prompt", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not (prompt and answer):
            raise RuntimeError("Bad short item from GPT")
        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            raise RuntimeError("Duplicate short prompt from GPT")

        alternates = []
        if isinstance(item.get("alternates"), list):
            alternates = [str(a).strip() for a in item["alternates"] if str(a).strip()]
        hint = (item.get("hint") or "").strip() or None
        explanation = (item.get("explanation") or "").strip() or None

        extra = {"mode": "gpt", "pos": pos, "alternates": alternates, "hint": hint, "explanation": explanation}
        db.execute(
            """INSERT INTO quiz_questions (quiz_id,qtype,prompt,correct,extra)
               VALUES (?,?,?,?,?)""",
            (quiz_id, "short", prompt, answer, json.dumps(extra))
        )
        used_norm_prompts.add(n)

    # LONG
    for _ in range(long_need):
        pos = positions.pop()
        item = gpt_long.pop(0)
        req = item.get("required_words", [])
        prompt = str(item.get("prompt") or (f"Write a single clear sentence using ALL of these words: {', '.join(req)}.")).strip()
        if not (isinstance(req, list) and req and prompt):
            raise RuntimeError("Bad long item from GPT")
        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            raise RuntimeError("Duplicate long prompt from GPT")

        extra = {"required_words": req, "pos": pos}
        db.execute(
            """INSERT INTO quiz_questions (quiz_id,qtype,prompt,correct,extra)
               VALUES (?,?,?,?,?)""",
            (quiz_id, "long", prompt, json.dumps(req), json.dumps(extra))
        )
        used_norm_prompts.add(n)

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

        if not username or not password:
            flash("User ID and password are required.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "warning")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        db = get_db()
        exists = db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
        if exists:
            flash("That User ID is already taken.", "danger")
            return redirect(url_for("register"))

        db.execute(
            "INSERT INTO users(username, level, xp, password_hash) VALUES(?, ?, ?, ?)",
            (username, 1, 0, generate_password_hash(password))
        )
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
        1: ("CEFR A1", "250–400"),
        2: ("CEFR A2", "350–550"),
        3: ("CEFR B1", "500–700"),
        4: ("CEFR B1+", "600–800"),
        5: ("CEFR B2", "700–900"),
    }
    level_tag, target_len = level_map.get(int(difficulty or 2), ("CEFR A2", "350–550"))

    if not openai_client:
        return synthesize_story_text(difficulty or 2, title or "Story")

    system = (
        "You are an ESL children's story writer. "
        "Output ONLY the story text: the first line must be the exact title, "
        "then 3–6 short paragraphs separated by blank lines. No extra commentary."
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
        needed = 16 - len(qs)
        positions_taken = {q.get("extra_parsed", {}).get("pos") for q in qs if q.get("extra_parsed")}
        available = [p for p in range(16) if p not in positions_taken]
        filler_used_norm = set()
        for i in range(needed):
            pos = available[i]
            add_mcq_question_unique(db, quiz_id, level, pos, filler_used_norm)
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

@app.post("/api/quiz/answer")
def api_quiz_answer():
    """
    Expects JSON:
      {
        "quiz_id": <int>,
        "question_id": <int>,
        "response": <string>,
        "confidence": <int 1..5, optional>  // present when confidence popup was shown
      }
    Returns (example):
      {
        "ok": true,
        "was_correct": true,
        "lives": 3,
        "xp": 40,
        "lines": 1,
        "user_level": 2,
        "user_xp": 120,
        "correct": [ ...optional positions... ],
        "explanation": "optional"
      }
    """
    payload = request.get_json(force=True, silent=True) or {}
    quiz_id      = payload.get("quiz_id")
    question_id  = payload.get("question_id")
    response_txt = (payload.get("response") or "").strip()

    # Optional confidence (sent by the quiz page for ~20% of questions)
    confidence = payload.get("confidence")
    try:
        if confidence is not None:
            confidence = int(confidence)
            if confidence < 1 or confidence > 5:
                confidence = None
    except Exception:
        confidence = None

    # ---- VALIDATION ----
    if quiz_id is None or question_id is None:
        return jsonify({"ok": False, "error": "quiz_id and question_id are required"}), 400

    # ---- YOUR CORRECTNESS LOGIC HERE ----
    # TODO: Replace this block with your existing check logic.
    # Example stub:
    was_correct = False
    explanation = None
    try:
        conn = get_db()
        # Fetch the ground truth from your questions table (example)
        row = conn.execute("SELECT qtype, correct, explanation FROM Questions WHERE id=?", (question_id,)).fetchone()
        if row:
            qtype, correct_raw, ex_raw = row
            explanation = ex_raw
            if qtype == "mcq":
                # correct_raw expected like "A"/"B"/"C"/"D"
                was_correct = (str(correct_raw).strip().upper() == str(response_txt).strip().upper())
            elif qtype == "short":
                # simple normalize match
                was_correct = (str(correct_raw).strip().lower() == str(response_txt).strip().lower())
            else:
                # long: required keywords (JSON array in correct_raw) OR simple contains
                try:
                    import json as _json
                    kws = _json.loads(correct_raw) if correct_raw else []
                    if isinstance(kws, list) and kws:
                        ans = response_txt.lower()
                        was_correct = all(k.lower() in ans for k in kws)
                    else:
                        was_correct = (str(correct_raw).strip().lower() in str(response_txt).strip().lower())
                except Exception:
                    was_correct = False
    except Exception as e:
        app.logger.warning(f"Correctness check failed: {e}")

    # ---- UPDATE GAME STATE (XP/LIVES/LINES/LEVEL) ----
    # Keep your existing logic here. Below is a safe no-op skeleton that you can
    # replace with your real per-user round state tracking.
    lives = int(request.cookies.get("lives", 3))
    xp    = int(request.cookies.get("xp", 0))
    lines = int(request.cookies.get("lines", 0))

    if was_correct:
        xp += 10
    else:
        lives -= 1

    # Example: compute new profile level/xp (persist however you already do)
    username = session.get("username") or "guest"
    try:
        # Example Users table with total xp/level
        conn = get_db()
        # Upsert user total xp
        conn.execute("""
            INSERT INTO Users(username, password)
            VALUES(?, 'pw')
            ON CONFLICT(username) DO NOTHING;
        """, (username,))
        conn.execute("""
            UPDATE Users SET preferred_war = preferred_war
        """)  # harmless touch to keep SQLite happy when table has only NOT NULL cols
        # Suppose you track totals in a separate table or Users has columns user_xp/user_level
        # Replace these with your actual schema updates.
    except Exception as e:
        app.logger.warning(f"User XP/Level mock update skipped: {e}")

    # Compute profile numbers (replace with your real persistence)
    user_total_xp = session.get("user_total_xp", 0) + (10 if was_correct else 0)
    session["user_total_xp"] = user_total_xp
    user_level = max(1, (user_total_xp // 100) + 1)

    # ---- STORE ANALYTICS ROW (this powers the admin dashboard) ----
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO QuizAnswers(username, quiz_id, question_id, was_correct, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (username, int(quiz_id), int(question_id), 1 if was_correct else 0, confidence))
        conn.commit()
    except Exception as e:
        app.logger.warning(f"QuizAnswers insert failed: {e}")

    # Response payload (you probably add more fields normally)
    return jsonify({
        "ok": True,
        "was_correct": was_correct,
        "lives": lives,
        "xp": xp,
        "lines": lines,
        "user_level": user_level,
        "user_xp": user_total_xp,
        "explanation": explanation or None
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
@app.route("/admin/analytics")
def admin_analytics():
    _require_admin()
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT quiz_id FROM QuizAnswers ORDER BY quiz_id").fetchall()
    quiz_ids = [r[0] for r in rows]
    return render_template("admin_analytics.html", quiz_ids=quiz_ids)
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

# -------------------- Main --------------------
if __name__ == "__main__":
    with app.app_context():
        init_db(seed=True)
    app.run(debug=True)
