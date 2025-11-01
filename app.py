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
    g, session, redirect, url_for, request, flash
)

# ====== GPT client (optional; graceful fallback) ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
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
    # keep FK constraints on
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
def gpt_generate_quiz(level: int):
    """
    Ask GPT for three arrays: mcq, short, long, sized by distribution_for_level(level).
    Ages: 6–10 only; vocabulary and grammar must be age-appropriate.
    """
    if not openai_client:
        print("[GPT QUIZ] client not available -> returning None")
        return None

    # pull exact counts from your board distribution
    mcq_need, short_need, long_need = distribution_for_level(level)

    # map level to simple guardrails
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
        "• short: array of items {prompt, answer} where answer is ONE word (preposition/adverb/conjunction like 'because')\n"
        "  - Prompts are cloze ('Fill the blank: ... ___ ...'). Keep sentences short and concrete.\n"
        "• long: array of items {required_words[list of 2..6 strings], prompt(optional)}\n"
        "  - If prompt is omitted, the app will show: 'Write one clear sentence using ALL of these words: ...'\n"
        "  - required_words must be easy words only; no proper nouns; no slang; no scary/violent themes.\n"
    )

    # Build the exact request with counts bound to the chosen level
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
            temperature=0.4,  # low temp for cleaner options
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        print("\n================= [GPT QUIZ RAW] =================")
        print(raw)
        print("==================================================\n")

        data = json.loads(raw)

        # Back-compat if needed
        if "long" not in data and "sentence" in data and isinstance(data["sentence"], list):
            data["long"] = data["sentence"]
        if "short" not in data: data["short"] = []
        if "mcq" not in data:   data["mcq"]   = []

        # Sanity: arrays only
        if not isinstance(data.get("mcq", []), list):   data["mcq"] = []
        if not isinstance(data.get("short", []), list): data["short"] = []
        if not isinstance(data.get("long", []), list):  data["long"] = []

        # Trim if GPT oversupplies
        data["mcq"]   = data["mcq"][:mcq_need]
        data["short"] = data["short"][:short_need]
        data["long"]  = data["long"][:long_need]

        # Log counts
        print("[GPT QUIZ PARSED] keys:", list(data.keys()))
        print("[GPT QUIZ COUNTS]", "mcq:", len(data["mcq"]), "short:", len(data["short"]), "long:", len(data["long"]))

        return data
    except Exception as e:
        print("[GPT QUIZ] exception:", repr(e))
        return None


# -------------------- GPT inserters --------------------
def insert_gpt_mcq(db, quiz_id: int, pos: int, item: dict, used_norm_prompts: set) -> bool:
    """
    item = {"prompt": str, "options": [str,str,str,str], "answer_index": int}
    Returns True if inserted; False otherwise.
    """
    try:
        prompt = str(item.get("prompt", "")).strip()
        options = item.get("options", [])
        ai = item.get("answer_index", None)

        if not prompt or not isinstance(options, list) or len(options) < 4:
            return False
        if not isinstance(ai, int) or ai < 0 or ai > 3:
            return False

        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            return False

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
        return True
    except Exception:
        return False


def insert_gpt_short(db, quiz_id: int, pos: int, item: dict, used_norm_prompts: set) -> bool:
    """
    item = {"prompt": str, "answer": str}
    Returns True if inserted; False otherwise.
    """
    try:
        prompt = str(item.get("prompt", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not prompt or not answer:
            return False

        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            return False

        extra = {"mode": "gpt", "pos": pos}
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
    Generates a unique SHORT item. Answer is a single short token (preposition/adverb).
    """
    for _ in range(max_tries):
        pool = WORD_POOLS[level]
        if random.random() < 0.7:
            correct = random.choice(pool["preps"])
            sentence = "I have lived here ___ 2019."
            prompt = f"Fill the blank with the correct preposition:\n“{sentence}”"
        else:
            correct = random.choice(pool["adverbs"])
            verb = random.choice(pool["verbs"])
            sentence = f"They {verb} ___ to catch the bus."
            prompt = f"Fill the blank with the best adverb:\n“{sentence}”"

        n = _norm_prompt(prompt)
        if n in used_norm_prompts:
            continue

        extra = {"mode": "short", "sentence": sentence, "pos": pos}
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

    # Strictly use GPT output; no local fallbacks.
    gpt = gpt_generate_quiz(level) or {}

    gpt_mcq   = list(gpt.get("mcq",   []))
    gpt_short = list(gpt.get("short", []))
    # accept either 'long' (new) or 'sentence' (legacy) from GPT
    gpt_long  = list(gpt.get("long", gpt.get("sentence", [])))

    # Validate counts early so we never silently fill with local generators
    if len(gpt_mcq)   < mcq_need:   raise RuntimeError(f"GPT returned {len(gpt_mcq)} MCQ, need {mcq_need}")
    if len(gpt_short) < short_need: raise RuntimeError(f"GPT returned {len(gpt_short)} short, need {short_need}")
    if len(gpt_long)  < long_need:  raise RuntimeError(f"GPT returned {len(gpt_long)} long, need {long_need}")

    # fixed board positions 0..15 (your template places cells by extra_parsed.pos)
    positions = list(range(16))
    random.shuffle(positions)

    used_norm_prompts = set()

    # ---- MCQ ----
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

    # ---- SHORT ----
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

        extra = {"mode": "gpt", "pos": pos}
        db.execute(
            """INSERT INTO quiz_questions (quiz_id,qtype,prompt,correct,extra)
               VALUES (?,?,?,?,?)""",
            (quiz_id, "short", prompt, answer, json.dumps(extra))
        )
        used_norm_prompts.add(n)

    # ---- LONG ----
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

        if not username or not password:
            flash("User ID and password are required.", "danger")
            return redirect(url_for("login"))

        db = get_db()
        row = db.execute("SELECT username, password_hash, level, xp FROM users WHERE username = ?", (username,)).fetchone()

        if not row or not row["password_hash"]:
            flash("Account not found. Please register first.", "warning")
            return redirect(url_for("register"))

        if not check_password_hash(row["password_hash"], password):
            flash("Incorrect password.", "danger")
            return redirect(url_for("login"))

        session["username"] = username
        session.permanent = remember  # 30 days if checked
        flash(f"Logged in as {username}", "success")
        return redirect(url_for("home"))

    # render template file (see below)
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

# -------------------- Quiz flow: single 4x4 board --------------------
@app.route("/quiz/<int:level>", methods=["GET"])
def start_quiz(level):
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    row = db.execute("SELECT level FROM users WHERE username = ?", (username,)).fetchone()
    current_level = row["level"] if row else 1
    if level > current_level:
        flash("That level is locked. Clear your current level first.", "warning")
        return redirect(url_for("home"))

    # NEW: respect `?new=1`
    force_new = (request.args.get("new") == "1")

    quiz_id = None
    if not force_new:
        # Try to reuse an unfinished quiz
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

    # If forcing new OR nothing reusable found, create a new board
    if not quiz_id:
        quiz_id = create_board_quiz(username, level)

    rows = db.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,)).fetchall()
    qs = []
    for r in rows:
        d = dict(r)
        d["extra_parsed"] = json.loads(d["extra"]) if d.get("extra") else {}
        qs.append(d)

    # Ensure exactly 16 (safety net)
    if len(qs) < 16:
        needed = 16 - len(qs)
        positions_taken = {q.get("extra_parsed",{}).get("pos") for q in qs if q.get("extra_parsed")}
        available = [p for p in range(16) if p not in positions_taken]
        for i in range(needed):
            pos = available[i]
            add_mcq_question(db, quiz_id, level, pos)
        db.commit()
        rows = db.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,)).fetchall()
        qs = []
        for r in rows:
            d = dict(r)
            d["extra_parsed"] = json.loads(d["extra"]) if d.get("extra") else {}
            qs.append(d)

    return render_template("quiz.html", level=level, quiz_id=quiz_id, qs=qs)

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

@app.post("/api/quiz/answer")
def api_quiz_answer():
    if "username" not in session:
        return {"error":"auth"}, 401
    data = request.get_json(silent=True) or {}
    quiz_id = int(data.get("quiz_id") or 0)
    question_id = int(data.get("question_id") or 0)
    response_text = (data.get("response") or "").strip()

    db = get_db()
    row = db.execute("SELECT username FROM quizzes WHERE id=?", (quiz_id,)).fetchone()
    if not row or row["username"] != session["username"]:
        return {"error":"not-found"}, 404

    q = db.execute("SELECT * FROM quiz_questions WHERE id=? AND quiz_id=?", (question_id, quiz_id)).fetchone()
    if not q:
        return {"error":"question-not-found"}, 404

    # If already answered, return current state
    existing = db.execute("SELECT * FROM quiz_answers WHERE quiz_id=? AND question_id=?", (quiz_id, question_id)).fetchone()
    if existing:
        state = _quiz_state(db, quiz_id)
        state["already_answered"] = True
        return state

    qtype = q["qtype"]
    is_correct = 0
    awarded_xp = 0

    if qtype == "mcq":
        user_ans = response_text.upper()
        correct_letter = (q["correct"] or "").upper()
        is_correct = 1 if user_ans == correct_letter else 0
        awarded_xp = 10 if is_correct else 0

    elif qtype == "short":
        user_ans = response_text.lower()
        correct_text = (q["correct"] or "").strip().lower()
        is_correct = 1 if (user_ans and user_ans == correct_text) else 0
        awarded_xp = 10 if is_correct else 0

    else:  # long
        required = set(json.loads(q["correct"] or "[]"))
        sentence = response_text
        low = sentence.lower()
        hits = sum(1 for w in required if w.lower() in low)
        is_correct = 1 if hits == len(required) and len(sentence.split()) >= max(6, len(required) + 2) else 0
        awarded_xp = 15 if is_correct else int(round(15 * (hits / max(1, len(required)))))

    # Store answer (unique per uq_quiz_answers)
    db.execute("""
        INSERT INTO quiz_answers(quiz_id, question_id, response, is_correct, awarded_xp)
        VALUES (?,?,?,?,?)
    """, (quiz_id, question_id, response_text, is_correct, awarded_xp))
    db.commit()

    return _quiz_state(db, quiz_id)

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
            is_correct = 1 if (user_ans and user_ans == correct_text) else 0
            awarded_xp = 10 if is_correct else 0
            resp_text = user_ans

        elif qtype == "long":
            required = set(json.loads(q["correct"]))
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

    # line bonus (20 xp per completed line)
    lines_completed = compute_completed_lines(correct_positions)
    bonus_xp = 20 * lines_completed
    total_xp += bonus_xp

    score_pct = round(100 * total_correct / total) if total else 0
    db.execute("""
        INSERT INTO attempts(username, level, score, total, lines_completed)
        VALUES (?, ?, ?, ?, ?)
    """, (username, quiz["level"], score_pct, total, lines_completed))

    # Update XP & level
    urow = db.execute("SELECT level, xp FROM users WHERE username = ?", (username,)).fetchone()
    level_before, xp_before = urow["level"], urow["xp"]
    xp_after = xp_before + total_xp

    new_level = level_before
    while xp_after >= 100 and new_level < 5:
        xp_after -= 100
        new_level += 1

    db.execute("UPDATE users SET level = ?, xp = ? WHERE username = ?", (new_level, xp_after, username))

    # === NEW: award achievements for this board ===
    evaluate_and_award_achievements(db, username, quiz_id, total_correct, lines_completed)

    db.commit()

    leveled_up = new_level > level_before

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
            <p class="mb-1">XP earned: <strong>{{ gained_xp }}</strong></p>
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
@app.route("/achievements")
def achievements():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()

    # Make sure DB/tables exist and backfill any missed achievements
    recalc_user_achievements(db, username)

    # ---- user basics (so template can show level/xp/title) ----
    u = db.execute(
        "SELECT level, xp, title_code FROM users WHERE username=?",
        (username,)
    ).fetchone()
    user_level = u["level"] if u else 1
    user_xp    = u["xp"] if u else 0
    title_code = u["title_code"] if u else None

    title_info = TITLES.get(title_code or "", DEFAULT_TITLE)
    display_title = title_info["title"]
    title_icon    = title_info["icon"]

    # ---- achievements data for the page ----
    # All achievements with left join to this user's unlocks
    rows = db.execute("""
        SELECT a.code, a.name, a.description, a.icon,
               ua.granted_at AS unlocked_at
        FROM achievements a
        LEFT JOIN user_achievements ua
          ON ua.code=a.code AND ua.username=?
        ORDER BY a.name
    """, (username,)).fetchall()

    # Stats
    achievements_total  = db.execute("SELECT COUNT(*) AS c FROM achievements").fetchone()["c"]
    earned_codes_rows   = db.execute("SELECT code FROM user_achievements WHERE username=?", (username,)).fetchall()
    earned_codes        = [r["code"] for r in earned_codes_rows]
    achievements_earned = len(earned_codes)

    # Collected titles (names) from earned achievements
    earned_titles = []
    for code in earned_codes:
        t = TITLES.get(code)
        if t and t.get("title"):
            earned_titles.append((t["priority"], t["title"]))
    # sort by priority & dedupe
    earned_titles = [t for _, t in sorted(set(earned_titles), key=lambda x: x[0])]

    return render_template(
        "achievements.html",
        # user header
        username=username,
        user_level=user_level,
        user_xp=user_xp,
        display_title=display_title,
        title_icon=title_icon,
        # grid + stats
        items=rows,
        achievements_total=achievements_total,
        achievements_earned=achievements_earned,
        earned_codes=earned_codes,
        earned_titles=earned_titles,
    )

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

# -------------------- Main --------------------
if __name__ == "__main__":
    with app.app_context():
        init_db(seed=True)
    app.run(debug=True)
