#!/usr/bin/env python3
"""
Seed mock analytics data for InterLang admin dashboard, aligned to app.py schema.

Usage:
  python seed_mock_data.py --db static/database.db --days 30 --users 40 --quizzes 6 --wipe-last-days
"""

import argparse
import random
import sqlite3
from datetime import datetime, timedelta
import json
import string

# ------------------------- CLI -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="static/database.db", help="Path to SQLite DB (same as your app)")
    ap.add_argument("--days", type=int, default=30, help="How many past days to seed (inclusive of today)")
    ap.add_argument("--users", type=int, default=30, help="How many users to create (in addition to 'testtest')")
    ap.add_argument("--quizzes", type=int, default=5, help="How many quiz templates (levels variety)")
    ap.add_argument("--wipe-last-days", action="store_true",
                    help="Delete previous activity within the window before seeding")
    return ap.parse_args()

# ------------------------- Helpers -------------------------
def ts_in_day(base_day: datetime, hour_rng=(8, 22)) -> str:
    h = random.randint(*hour_rng)
    m = random.randint(0, 59)
    return (base_day.replace(hour=0, minute=0, second=0, microsecond=0) +
            timedelta(hours=h, minutes=m)).strftime("%Y-%m-%d %H:%M:%S")

def slug_word(n=5):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(n))

def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    # === Core user/quiz tables (match app.py) ===
    # users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            level INTEGER NOT NULL DEFAULT 1,
            xp INTEGER NOT NULL DEFAULT 0,
            title_code TEXT
        );
    """)  # :contentReference[oaicite:0]{index=0}

    # attempts (optional historical scoreboard rows)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            level INTEGER NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            lines_completed INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)  # :contentReference[oaicite:1]{index=1}

    # quizzes
    cur.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            level INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)  # :contentReference[oaicite:2]{index=2}

    # quiz_questions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            qtype TEXT NOT NULL,
            prompt TEXT NOT NULL,
            choice_a TEXT,
            choice_b TEXT,
            choice_c TEXT,
            choice_d TEXT,
            correct TEXT,
            extra TEXT,
            FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE
        );
    """)  # :contentReference[oaicite:3]{index=3}

    # quiz_answers
    cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            response TEXT NOT NULL,
            is_correct INTEGER NOT NULL,
            awarded_xp INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES quiz_questions(id) ON DELETE CASCADE
        );
    """)  # :contentReference[oaicite:4]{index=4}

    # === Stories / reading / vocabulary (match app.py) ===
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            title TEXT,
            author TEXT,
            age_min INTEGER,
            age_max INTEGER,
            difficulty INTEGER,
            categories TEXT,
            themes TEXT,
            provider TEXT,
            provider_id TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)  # :contentReference[oaicite:5]{index=5}

    cur.execute("""
        CREATE TABLE IF NOT EXISTS story_reads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            story_id INTEGER NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP,
            words_learned INTEGER DEFAULT 0,
            FOREIGN KEY(story_id) REFERENCES stories(id) ON DELETE CASCADE
        );
    """)  # :contentReference[oaicite:6]{index=6}

    cur.execute("""
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
    """)  # :contentReference[oaicite:7]{index=7}

    conn.commit()

def seed_users(conn: sqlite3.Connection, n_users: int) -> list[str]:
    cur = conn.cursor()
    # Make a friendly, memorable pool
    pool = ["alex","bailey","casey","dana","eli","fin","gia","haru","ivan","jules",
            "kai","leo","mina","noah","ori","pax","quinn","ryu","sara","tala",
            "umi","vale","will","xena","yuri","zara","mike","hana","eun","mark",
            "joon","sumi","nick","rhea","olga","bora","sena","teo","yena","sora"]
    users = []
    for i in range(n_users):
        name = pool[i % len(pool)]
        suffix = "" if i < len(pool) else str(100+i)
        username = f"{name}{suffix}"
        level = max(1, min(8, int(random.gauss(4.0, 1.8))))
        xp = max(0, int(level*120 + random.gauss(0, 100)))
        title = random.choice([None, "rookie", "sprout", "ranger", "scholar", None, None])
        users.append((username, level, xp, title))
    # ensure 'testtest' appears (your admin/testing account)
    users.append(("testtest", 6, 900 + random.randint(0,150), "mentor"))

    cur.executemany(
        "INSERT OR IGNORE INTO users (username, level, xp, title_code) VALUES (?,?,?,?)",
        users
    )
    conn.commit()
    return [u[0] for u in users]

def seed_stories(conn: sqlite3.Connection, n=30) -> list[int]:
    cur = conn.cursor()
    story_rows = []
    for i in range(n):
        slug = f"{slug_word(6)}-{i}"
        title = f"Story {i+1}: {slug_word(5).title()}"
        author = random.choice(["Lee", "Kim", "Park", "Choi", "Jang", "Yun", "Song"])
        age_min, age_max = random.choice([(7,9),(8,11),(10,13),(12,15)])
        diff = random.randint(1,5)
        categories = json.dumps(random.sample(["animal","space","friendship","adventure","school","mystery"], k=2))
        themes = json.dumps(random.sample(["bravery","kindness","curiosity","teamwork"], k=2))
        provider = random.choice(["internal","openlib"])
        provider_id = slug_word(8)
        content = f"Once upon a time in {slug_word(7)}..."
        story_rows.append((slug, title, author, age_min, age_max, diff, categories, themes, provider, provider_id, content))
    cur.executemany("""
        INSERT OR IGNORE INTO stories(slug,title,author,age_min,age_max,difficulty,categories,themes,provider,provider_id,content)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, story_rows)
    conn.commit()
    ids = [r[0] for r in cur.execute("SELECT id FROM stories").fetchall()]
    return ids

def create_quiz_with_questions(conn: sqlite3.Connection, username: str, level: int, created_at: str) -> tuple[int, list[int]]:
    """
    Create a quiz (for one user) and attach 12–16 questions:
      - ~8–10 MCQ with 4 choices
      - ~3–5 SHORT
      - ~1–2 LONG (free-form; extra JSON with required words)
    """
    cur = conn.cursor()
    cur.execute("INSERT INTO quizzes (username, level, created_at) VALUES (?,?,?)", (username, level, created_at))
    quiz_id = cur.lastrowid

    qids = []
    # MCQ
    for _ in range(random.randint(8, 10)):
        prompt = f"Choose the correct meaning of '{slug_word(6)}'"
        choices = [slug_word(5) for _ in range(4)]
        correct_letter = random.choice(["A","B","C","D"])
        cur.execute("""
            INSERT INTO quiz_questions(quiz_id,qtype,prompt,choice_a,choice_b,choice_c,choice_d,correct,extra)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (quiz_id,"mcq",prompt,choices[0],choices[1],choices[2],choices[3],correct_letter,None))
        qids.append(cur.lastrowid)

    # SHORT
    for _ in range(random.randint(3, 5)):
        target = slug_word(5)
        prompt = f"Type the word that means '{target}'"
        cur.execute("""
            INSERT INTO quiz_questions(quiz_id,qtype,prompt,correct,extra)
            VALUES (?,?,?,?,?)
        """, (quiz_id,"short",prompt,target,None))
        qids.append(cur.lastrowid)

    # LONG
    for _ in range(random.randint(1, 2)):
        required = [slug_word(5), slug_word(5)]
        prompt = f"Write a short sentence including: {', '.join(required)}"
        extra = json.dumps({"required_words": required})
        # 'correct' for long is a JSON list in your app's design
        cur.execute("""
            INSERT INTO quiz_questions(quiz_id,qtype,prompt,correct,extra)
            VALUES (?,?,?,?,?)
        """, (quiz_id,"long",prompt,json.dumps(required),extra))
        qids.append(cur.lastrowid)

    conn.commit()
    return quiz_id, qids

def seed_window_activity(conn: sqlite3.Connection, users: list[str], stories: list[int], days: int, quiz_levels_span=(1,8)):
    """
    For each day in the window:
      - choose a DAU set
      - for each active user:
          * create 1 quiz (level ~ user's level ±1)
          * answer 10–24 questions with correctness ~ level
          * add story_reads (0–5) and vocab (0–4)
          * add attempts rows for variety
      - award xp on correct answers
    """
    cur = conn.cursor()
    today = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    for d in range(days-1, -1, -1):
        day = today - timedelta(days=d)
        # DAU: weekday/weekend dynamics
        weekday = day.weekday()  # 0=Mon
        dau_target = int(random.gauss(18 if weekday < 5 else 12, 5))
        dau = max(6, min(len(users), dau_target))
        active = random.sample(users, k=min(dau, len(users)))

        # activity: story reads & vocab
        for u in active:
            sreads = max(0, int(random.gauss(2.0, 1.4)))
            vnew = max(0, int(random.gauss(1.4, 1.1)))

            for _ in range(sreads):
                sid = random.choice(stories)
                started = ts_in_day(day, (9, 21))
                finished = ts_in_day(day, (9, 22)) if random.random() < 0.75 else None
                words = max(0, int(random.gauss(30, 12)))
                cur.execute("""
                    INSERT INTO story_reads(username, story_id, started_at, finished_at, words_learned)
                    VALUES (?,?,?,?,?)
                """, (u, sid, started, finished, words))

            for _ in range(vnew):
                w = slug_word(random.randint(4,8))
                created = ts_in_day(day, (10, 22))
                cur.execute("""
                    INSERT INTO vocab(username, word, translation, definition, source_story_id, created_at, review_count)
                    VALUES (?,?,?,?,?,?,?)
                """, (u, w, None, None, random.choice(stories), created, random.randint(0,3)))

        # quizzes + answers + attempts
        for u in active:
            # user's current level & a quiz level near it
            urow = cur.execute("SELECT level, xp FROM users WHERE username=?", (u,)).fetchone()
            if not urow:
                continue
            ulevel, uxp = urow[0], urow[1]
            qlevel = max(quiz_levels_span[0], min(quiz_levels_span[1], ulevel + random.choice([-1,0,0,1])))

            q_created = ts_in_day(day, (9, 21))
            quiz_id, qids = create_quiz_with_questions(conn, u, qlevel, q_created)

            # how many Qs this user attempts today
            attempts_cnt = max(10, min(24, int(random.gauss(14, 5))))
            chosen_qs = random.choices(qids, k=attempts_cnt)

            corrects = 0
            for qid in chosen_qs:
                # correctness probability increases with level slightly
                base_p = 0.55 + 0.03*(qlevel-4) + random.uniform(-0.06, 0.06)
                p = max(0.2, min(0.95, base_p))
                ok = 1 if random.random() < p else 0
                corrects += ok

                # pick a plausible response string
                resp = random.choice([
                    slug_word(5), "A","B","C","D", "yes","no",
                    "I think it means " + slug_word(6),
                    "Because " + slug_word(5)
                ])
                created = ts_in_day(day, (9, 22))
                award = 10 if ok else 0  # simple XP rule
                cur.execute("""
                    INSERT INTO quiz_answers(quiz_id,question_id,response,is_correct,awarded_xp,created_at)
                    VALUES (?,?,?,?,?,?)
                """, (quiz_id, qid, resp, ok, award, created))
                if award:
                    cur.execute("UPDATE users SET xp = xp + ? WHERE username=?", (award, u))

            # add a compact attempts row (score/total)
            cur.execute("""
                INSERT INTO attempts(username, level, score, total, lines_completed, created_at)
                VALUES (?,?,?,?,?,?)
            """, (u, qlevel, corrects, attempts_cnt, random.randint(0,3), ts_in_day(day, (21, 22))))

    conn.commit()

def wipe_window(conn: sqlite3.Connection, start: datetime, end: datetime):
    cur = conn.cursor()
    s = start.strftime("%Y-%m-%d 00:00:00")
    e = end.strftime("%Y-%m-%d 23:59:59")
    # order matters due to FKs
    cur.execute("DELETE FROM quiz_answers WHERE created_at BETWEEN ? AND ?", (s, e))
    cur.execute("DELETE FROM quiz_questions WHERE quiz_id IN (SELECT id FROM quizzes WHERE created_at BETWEEN ? AND ?)", (s, e))
    cur.execute("DELETE FROM quizzes WHERE created_at BETWEEN ? AND ?", (s, e))
    cur.execute("DELETE FROM attempts WHERE created_at BETWEEN ? AND ?", (s, e))
    cur.execute("DELETE FROM story_reads WHERE started_at BETWEEN ? AND ?", (s, e))
    cur.execute("DELETE FROM vocab WHERE created_at BETWEEN ? AND ?", (s, e))
    conn.commit()

# ------------------------- Main -------------------------
def main():
    args = parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    # Keep constraints on (your app uses them)
    conn.execute("PRAGMA foreign_keys = ON;")

    ensure_schema(conn)

    # Seed users & stories
    users = seed_users(conn, args.users)
    stories = seed_stories(conn, n=40)

    # Determine window
    end = datetime.now()
    start = end - timedelta(days=args.days-1)

    if args.wipe_last_days:
        wipe_window(conn, start, end)

    # Activity across window
    seed_window_activity(conn, users, stories, args.days)

    # Quick stats print
    cur = conn.cursor()
    u = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    qa = cur.execute("SELECT COUNT(*) FROM quiz_answers").fetchone()[0]
    sr = cur.execute("SELECT COUNT(*) FROM story_reads").fetchone()[0]
    vb = cur.execute("SELECT COUNT(*) FROM vocab").fetchone()[0]
    at = cur.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    print(f"Seed ✓ users={u}, quiz_answers={qa}, story_reads={sr}, vocab={vb}, attempts={at}")

if __name__ == "__main__":
    main()
