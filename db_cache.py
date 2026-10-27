import sqlite3
import json
import os
import time
from typing import Optional, Dict, List

DB_PATH = os.path.join(os.path.dirname(__file__), "translations_cache.db")
_db_initialized = False


def init_db():
    global _db_initialized
    if _db_initialized:
        return
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=10000")
    # Cache for translations by manga, chapter, page, and box hash/index
    cur.execute("""
    CREATE TABLE IF NOT EXISTS page_translations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        manga_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        page_idx INTEGER NOT NULL,
        box_key TEXT NOT NULL,
        original TEXT,
        ipa TEXT,
        literal TEXT,
        natural TEXT,
        vocabulary TEXT,
        box_json TEXT,
        provider TEXT,
        created_at REAL,
        UNIQUE(manga_name, filename, page_idx, box_key)
    )
    """)
    
    # Pre-processed full page data (all detected bubbles on page with pre-computed translations)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS page_bubbles_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        manga_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        page_idx INTEGER NOT NULL,
        bubbles_json TEXT NOT NULL,
        is_fully_translated INTEGER DEFAULT 0,
        updated_at REAL,
        UNIQUE(manga_name, filename, page_idx)
    )
    """)

    # Audio TTS Cache (guarantees instantaneous playback and zero quota waste)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audio_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text_hash TEXT UNIQUE,
        audio_data BLOB NOT NULL,
        media_type TEXT NOT NULL,
        created_at REAL NOT NULL
    )
    """)
    conn.commit()
    conn.close()
    _db_initialized = True


def make_box_key(box: Optional[Dict]) -> str:
    if not box:
        return "full_page"
    # Round to 1 decimal place to tolerate tiny finger/touch selection offsets
    x = round(box.get("x", 0), 1)
    y = round(box.get("y", 0), 1)
    w = round(box.get("w", 0), 1)
    h = round(box.get("h", 0), 1)
    return f"{x}_{y}_{w}_{h}"


def get_cached_translation(manga_name: str, filename: str, page_idx: int, box: Optional[Dict]) -> Optional[Dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    box_key = make_box_key(box)
    
    cur.execute("""
        SELECT original, ipa, literal, natural, vocabulary, box_json, provider 
        FROM page_translations 
        WHERE manga_name=? AND filename=? AND page_idx=? AND box_key=?
    """, (manga_name, filename, page_idx, box_key))
    row = cur.fetchone()
    conn.close()
    
    if row:
        return {
            "original": row[0],
            "ipa": row[1] or "",
            "literal": row[2] or "",
            "natural": row[3] or "",
            "vocabulary": json.loads(row[4]) if row[4] else {},
            "box": json.loads(row[5]) if row[5] else box,
            "provider": row[6] + " (из кэша)",
            "cached": True
        }
    return None


def save_cached_translation(manga_name: str, filename: str, page_idx: int, box: Optional[Dict], data: Dict):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    box_key = make_box_key(box)
    
    cur.execute("""
        INSERT INTO page_translations 
        (manga_name, filename, page_idx, box_key, original, ipa, literal, natural, vocabulary, box_json, provider, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(manga_name, filename, page_idx, box_key) DO UPDATE SET
            original=excluded.original,
            ipa=excluded.ipa,
            literal=excluded.literal,
            natural=excluded.natural,
            vocabulary=excluded.vocabulary,
            box_json=excluded.box_json,
            provider=excluded.provider,
            created_at=excluded.created_at
    """, (
        manga_name,
        filename,
        page_idx,
        box_key,
        data.get("original", ""),
        data.get("ipa", ""),
        data.get("literal", ""),
        data.get("natural", data.get("translated", "")),
        json.dumps(data.get("vocabulary", {}), ensure_ascii=False),
        json.dumps(box, ensure_ascii=False) if box else None,
        data.get("provider", "Groq"),
        time.time()
    ))
    conn.commit()
    conn.close()


def get_cached_page_bubbles(manga_name: str, filename: str, page_idx: int) -> Optional[List[Dict]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT bubbles_json FROM page_bubbles_cache 
        WHERE manga_name=? AND filename=? AND page_idx=?
    """, (manga_name, filename, page_idx))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def is_page_fully_translated(manga_name: str, filename: str, page_idx: int) -> bool:
    """Return true only for a non-empty page whose positions and translations were saved."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT bubbles_json, is_fully_translated FROM page_bubbles_cache "
            "WHERE manga_name=? AND filename=? AND page_idx=?",
            (manga_name, filename, page_idx),
        ).fetchone()
        if not row or not row[1]:
            return False
        bubbles = json.loads(row[0])
        return bool(bubbles) and all(
            bubble.get("source") == "yolo26n-v3"
            and isinstance(bubble.get("translation"), dict)
            and bubble["translation"].get("natural")
            and bubble["translation"].get("natural") != bubble["translation"].get("original")
            for bubble in bubbles
        )
    finally:
        conn.close()


def delete_cached_translations_for_boxes(manga_name: str, filename: str, page_idx: int, boxes: List[Dict]):
    """Invalidate OCR/translation rows whose detected coordinates came from an old model pass."""
    keys = [make_box_key(box) for box in boxes if box]
    if not keys:
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            "DELETE FROM page_translations WHERE manga_name=? AND filename=? AND page_idx=? AND box_key=?",
            [(manga_name, filename, page_idx, key) for key in keys],
        )
        conn.commit()
    finally:
        conn.close()


def save_cached_page_bubbles(manga_name: str, filename: str, page_idx: int, bubbles: List[Dict], is_fully_translated: bool = False):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO page_bubbles_cache (manga_name, filename, page_idx, bubbles_json, is_fully_translated, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(manga_name, filename, page_idx) DO UPDATE SET
            bubbles_json=excluded.bubbles_json,
            is_fully_translated=excluded.is_fully_translated,
            updated_at=excluded.updated_at
    """, (manga_name, filename, page_idx, json.dumps(bubbles, ensure_ascii=False), 1 if is_fully_translated else 0, time.time()))
    conn.commit()
    conn.close()


def get_all_cached_translations_for_page(manga_name: str, filename: str, page_idx: int) -> List[Dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT original, ipa, literal, natural, vocabulary, box_json, box_key 
        FROM page_translations 
        WHERE manga_name=? AND filename=? AND page_idx=?
    """, (manga_name, filename, page_idx))
    rows = cur.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            "original": r[0],
            "ipa": r[1],
            "literal": r[2],
            "natural": r[3],
            "translated": r[3],
            "vocabulary": json.loads(r[4]) if r[4] else {},
            "box": json.loads(r[5]) if r[5] else None,
            "box_key": r[6]
        })
    return results


def get_cached_audio(text: str, voice_tag: str = "algieba") -> Optional[tuple]:
    """
    Returns (audio_bytes, media_type) if audio for this text is already generated and cached.
    """
    import hashlib
    h = hashlib.sha256(f"{voice_tag}:{text.strip().lower()}".encode('utf-8')).hexdigest()
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT audio_data, media_type FROM audio_cache WHERE text_hash=?", (h,))
        row = cur.fetchone()
        conn.close()
        if row:
            return (row[0], row[1])
    except Exception as e:
        print(f"[!] Error reading audio_cache: {e}")
    return None


def save_cached_audio(text: str, audio_bytes: bytes, media_type: str, voice_tag: str = "algieba"):
    """
    Persistently saves generated audio into SQLite cache.
    """
    import hashlib
    h = hashlib.sha256(f"{voice_tag}:{text.strip().lower()}".encode('utf-8')).hexdigest()
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO audio_cache (text_hash, audio_data, media_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (h, audio_bytes, media_type, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Error saving audio_cache: {e}")
