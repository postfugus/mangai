import sqlite3
import json
import os
import time
from typing import Optional, Dict, List
from text_quality import is_russian_translation, is_usable_ocr_text

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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS page_reading_order (
        manga_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        page_idx INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        order_json TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY(manga_name, filename, page_idx, fingerprint)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapter_voiceovers (
        manga_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        lang TEXT NOT NULL DEFAULT 'fr',
        fingerprint TEXT NOT NULL,
        audio_data BLOB NOT NULL,
        media_type TEXT NOT NULL,
        segments_json TEXT NOT NULL,
        bubbles_json TEXT NOT NULL,
        duration REAL NOT NULL,
        voice TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY(manga_name, filename, lang)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapter_voiceover_analysis (
        manga_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        items_json TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY(manga_name, filename)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapter_voiceover_audio_staging (
        manga_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        lang TEXT NOT NULL DEFAULT 'fr',
        fingerprint TEXT NOT NULL,
        audio_data BLOB NOT NULL,
        media_type TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL,
        PRIMARY KEY(manga_name, filename, lang)
    )
    """)
    staging_columns = {row[1] for row in cur.execute("PRAGMA table_info(chapter_voiceover_audio_staging)")}
    if "model" not in staging_columns:
        cur.execute("ALTER TABLE chapter_voiceover_audio_staging ADD COLUMN model TEXT NOT NULL DEFAULT ''")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapter_voiceover_tts_chunks (
        manga_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        model TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        input_hash TEXT NOT NULL,
        audio_data BLOB NOT NULL,
        media_type TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY(manga_name, filename, fingerprint, model, chunk_index)
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
        if not is_usable_ocr_text(row[0] or "") or not is_russian_translation(row[0] or "", row[3] or ""):
            with sqlite3.connect(DB_PATH, timeout=10) as cleanup:
                cleanup.execute(
                    "DELETE FROM page_translations WHERE manga_name=? AND filename=? AND page_idx=? AND box_key=?",
                    (manga_name, filename, page_idx, box_key),
                )
            return None
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
            and is_usable_ocr_text(bubble["translation"].get("original", ""))
            and is_russian_translation(
                bubble["translation"].get("original", ""),
                bubble["translation"].get("natural", ""),
            )
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
        if not is_usable_ocr_text(r[0] or "") or not is_russian_translation(r[0] or "", r[3] or ""):
            continue
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


def get_chapter_text_stats(manga_name: str, filename: str) -> Dict[str, int]:
    """Count unique OCR source text currently cached for one chapter."""
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        rows = conn.execute(
            "SELECT page_idx, box_key, original, natural FROM page_translations WHERE manga_name=? AND filename=?",
            (manga_name, filename),
        ).fetchall()
        bubble_rows = conn.execute(
            "SELECT page_idx, bubbles_json FROM page_bubbles_cache WHERE manga_name=? AND filename=?",
            (manga_name, filename),
        ).fetchall()

    source_texts = {
        (page_idx, box_key): original.strip()
        for page_idx, box_key, original, _ in rows
        if original and is_usable_ocr_text(original)
    }
    translation_status = {
        (page_idx, box_key): is_usable_ocr_text(original or "") and is_russian_translation(original or "", natural or "")
        for page_idx, box_key, original, natural in rows
        if original and is_usable_ocr_text(original)
    }
    for page_idx, raw_bubbles in bubble_rows:
        try:
            bubbles = json.loads(raw_bubbles)
        except (TypeError, json.JSONDecodeError):
            continue
        for bubble in bubbles:
            original = ((bubble.get("translation") or {}).get("original") or "").strip()
            box = bubble.get("box")
            if original and box and is_usable_ocr_text(original):
                key = (page_idx, make_box_key(box))
                source_texts.setdefault(key, original)
                translation_status.setdefault(
                    key, is_russian_translation(original, (bubble.get("translation") or {}).get("natural", ""))
                )
    return {
        "characters": sum(len(text) for text in source_texts.values()),
        "symbols": sum(sum(not char.isspace() for char in text) for text in source_texts.values()),
        "phrases": len(source_texts),
        "pages": len({page_idx for page_idx, _ in source_texts}),
        "invalid_translations": sum(not valid for valid in translation_status.values()),
    }


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


def get_cached_reading_order(manga_name: str, filename: str, page_idx: int, fingerprint: str):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        row = conn.execute(
            "SELECT order_json FROM page_reading_order WHERE manga_name=? AND filename=? AND page_idx=? AND fingerprint=?",
            (manga_name, filename, page_idx, fingerprint),
        ).fetchone()
    return json.loads(row[0]) if row else None


def save_cached_reading_order(manga_name: str, filename: str, page_idx: int, fingerprint: str, order):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO page_reading_order (manga_name, filename, page_idx, fingerprint, order_json, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (manga_name, filename, page_idx, fingerprint, json.dumps(order), time.time()),
        )
        conn.commit()


def get_cached_chapter_voiceover(manga_name: str, filename: str, fingerprint: Optional[str] = None,
                                include_audio: bool = True, lang: str = "fr"):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        fields = "fingerprint, audio_data, media_type, segments_json, bubbles_json, duration, voice, model, created_at, lang" if include_audio else "fingerprint, media_type, segments_json, bubbles_json, duration, voice, model, created_at, lang"
        row = conn.execute(f"SELECT {fields} FROM chapter_voiceovers WHERE manga_name=? AND filename=? AND lang=?",
                           (manga_name, filename, lang)).fetchone()
        if not row:
            row = conn.execute(f"SELECT {fields} FROM chapter_voiceovers WHERE manga_name=? AND filename=? ORDER BY created_at DESC LIMIT 1",
                               (manga_name, filename)).fetchone()
    if not row or (fingerprint and row[0] != fingerprint):
        return None
    if include_audio:
        audio_data, media_type = row[1], row[2]
        offset = 3
    else:
        audio_data, media_type = None, row[1]
        offset = 2
    return {
        "fingerprint": row[0], "audio_data": audio_data, "media_type": media_type,
        "segments": json.loads(row[offset]), "bubbles": json.loads(row[offset + 1]),
        "duration": row[offset + 2], "voice": row[offset + 3],
        "model": row[offset + 4], "created_at": row[offset + 5],
        "lang": row[offset + 6] if len(row) > offset + 6 else lang,
    }


def list_cached_chapter_voiceovers(manga_name: str, filename: str) -> List[Dict]:
    """Return all available localized voiceovers for this chapter."""
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        rows = conn.execute(
            "SELECT lang, voice, model, duration, created_at FROM chapter_voiceovers WHERE manga_name=? AND filename=? ORDER BY created_at DESC",
            (manga_name, filename)
        ).fetchall()
    return [{"lang": r[0], "voice": r[1], "model": r[2], "duration": r[3], "created_at": r[4]} for r in rows]


def save_cached_chapter_voiceover(manga_name: str, filename: str, fingerprint: str,
                                  audio_data: bytes, media_type: str, segments: Dict,
                                  bubbles: List[Dict], duration: float, voice: str, model: str,
                                  lang: str = "fr"):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chapter_voiceovers "
            "(manga_name, filename, lang, fingerprint, audio_data, media_type, segments_json, bubbles_json, duration, voice, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (manga_name, filename, lang, fingerprint, audio_data, media_type,
             json.dumps(segments, ensure_ascii=False), json.dumps(bubbles, ensure_ascii=False),
             duration, voice, model, time.time()),
        )
        conn.commit()


def get_staged_chapter_voiceover_audio(manga_name: str, filename: str, fingerprint: str, model: str, lang: str = "fr"):
    """Return chapter audio saved before local alignment, if it matches this text revision."""
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        row = conn.execute(
            "SELECT audio_data, media_type FROM chapter_voiceover_audio_staging "
            "WHERE manga_name=? AND filename=? AND lang=? AND fingerprint=? AND model=?",
            (manga_name, filename, lang, fingerprint, model),
        ).fetchone()
    return (row[0], row[1]) if row else None


def save_staged_chapter_voiceover_audio(manga_name: str, filename: str, fingerprint: str,
                                        audio_data: bytes, media_type: str, model: str, lang: str = "fr"):
    """Persist generated TTS before Whisper so alignment retries never spend TTS quota again."""
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chapter_voiceover_audio_staging "
            "(manga_name, filename, lang, fingerprint, audio_data, media_type, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (manga_name, filename, lang, fingerprint, audio_data, media_type, model, time.time()),
        )
        conn.commit()


def delete_staged_chapter_voiceover_audio(manga_name: str, filename: str, fingerprint: str, lang: str = "fr"):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute(
            "DELETE FROM chapter_voiceover_audio_staging WHERE manga_name=? AND filename=? AND lang=? AND fingerprint=?",
            (manga_name, filename, lang, fingerprint),
        )
        conn.commit()


def get_cached_chapter_voiceover_tts_chunk(manga_name: str, filename: str, fingerprint: str,
                                           model: str, chunk_index: int, input_hash: str):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        row = conn.execute(
            "SELECT audio_data FROM chapter_voiceover_tts_chunks "
            "WHERE manga_name=? AND filename=? AND fingerprint=? AND model=? AND chunk_index=? AND input_hash=?",
            (manga_name, filename, fingerprint, model, chunk_index, input_hash),
        ).fetchone()
    return row[0] if row else None


def save_cached_chapter_voiceover_tts_chunk(manga_name: str, filename: str, fingerprint: str,
                                            model: str, chunk_index: int, input_hash: str,
                                            audio_data: bytes, media_type: str = "audio/wav"):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chapter_voiceover_tts_chunks "
            "(manga_name, filename, fingerprint, model, chunk_index, input_hash, audio_data, media_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (manga_name, filename, fingerprint, model, chunk_index, input_hash, audio_data, media_type, time.time()),
        )
        conn.commit()


def delete_cached_chapter_voiceover_tts_chunks(manga_name: str, filename: str, fingerprint: str):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute(
            "DELETE FROM chapter_voiceover_tts_chunks WHERE manga_name=? AND filename=? AND fingerprint=?",
            (manga_name, filename, fingerprint),
        )
        conn.commit()


def get_cached_chapter_voiceover_analysis(manga_name: str, filename: str, fingerprint: str):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        row = conn.execute(
            "SELECT items_json FROM chapter_voiceover_analysis WHERE manga_name=? AND filename=? AND fingerprint=?",
            (manga_name, filename, fingerprint),
        ).fetchone()
    return json.loads(row[0]) if row else None


def save_cached_chapter_voiceover_analysis(manga_name: str, filename: str,
                                           fingerprint: str, items: List[Dict]):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chapter_voiceover_analysis "
            "(manga_name, filename, fingerprint, items_json, updated_at) VALUES (?, ?, ?, ?, ?)",
            (manga_name, filename, fingerprint, json.dumps(items, ensure_ascii=False), time.time()),
        )
        conn.commit()


def get_all_cached_chapter_bubbles(manga_name: str, filename: str) -> List[Dict]:
    """Collect OCR text and bubble boxes for every page, merging both local cache tables."""
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        bubble_rows = conn.execute(
            "SELECT page_idx, bubbles_json FROM page_bubbles_cache WHERE manga_name=? AND filename=?",
            (manga_name, filename),
        ).fetchall()
        translation_rows = conn.execute(
            "SELECT page_idx, original, natural, box_json FROM page_translations "
            "WHERE manga_name=? AND filename=?",
            (manga_name, filename),
        ).fetchall()
    collected = {}
    for page_idx, raw in bubble_rows:
        try:
            bubbles = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        for bubble in bubbles:
            box = bubble.get("box")
            translation = bubble.get("translation") or {}
            original = (translation.get("original") or "").strip()
            if box and is_usable_ocr_text(original):
                natural = (translation.get("natural") or "").strip()
                collected[(page_idx, make_box_key(box))] = {
                    "page_idx": page_idx, "box": box, "original": original,
                    "natural": natural if is_russian_translation(original, natural) else "",
                }
    for page_idx, original, natural, raw_box in translation_rows:
        if not original or not raw_box or not is_usable_ocr_text(original):
            continue
        try:
            box = json.loads(raw_box)
        except (TypeError, json.JSONDecodeError):
            continue
        key = (page_idx, make_box_key(box))
        translated = (natural or "").strip()
        translation_valid = is_russian_translation(original, translated)
        row_item = {"page_idx": page_idx, "box": box, "original": original.strip(),
                    "natural": translated if translation_valid else ""}
        if translation_valid or key not in collected:
            # The dedicated translation row is newer than the page snapshot;
            # prefer its validated correction instead of stale embedded text.
            collected[key] = row_item
    items = list(collected.values())
    items.sort(key=lambda item: (
        item["page_idx"],
        round(float(item["box"].get("y", 0)) / max(float(item["box"].get("h", 1)), 1) * 0.55),
        -float(item["box"].get("x", 0)),
    ))
    for index, item in enumerate(items):
        item["id"] = f"p{item['page_idx']}-b{index}"
    return items


def clear_offline_cache(manga_name: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, int]:
    """Clear cached translations, bubble locations and speech for a manga or chapter."""
    init_db()
    deleted = {}
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        if manga_name and filename:
            scope, params = "manga_name=? AND filename=?", (manga_name, filename)
        elif manga_name:
            scope, params = "manga_name=?", (manga_name,)
        else:
            scope, params = "1=1", ()
        for key, table in (("translations", "page_translations"), ("bubbles", "page_bubbles_cache"),
                           ("reading_orders", "page_reading_order"), ("voiceovers", "chapter_voiceovers"),
                           ("voiceover_analysis", "chapter_voiceover_analysis"),
                           ("staged_audio", "chapter_voiceover_audio_staging"),
                           ("tts_chunks", "chapter_voiceover_tts_chunks")):
            try:
                cur = conn.execute(f"DELETE FROM {table} WHERE {scope}", params)
                deleted[key] = cur.rowcount
            except sqlite3.OperationalError:
                deleted[key] = 0
        if not manga_name:
            cur = conn.execute("DELETE FROM audio_cache")
            deleted["audio"] = cur.rowcount
        conn.commit()
    return deleted

# ==================== MIHON FAVORITES / LIBRARY SHELF ====================

def init_library_tables():
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS favorites_library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                manga_id TEXT NOT NULL,
                title TEXT NOT NULL,
                cover_url TEXT DEFAULT '',
                status TEXT DEFAULT 'reading',
                category TEXT DEFAULT 'default',
                last_chapter TEXT DEFAULT '',
                unread_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, manga_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enabled_sources (
                source_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enabled INTEGER DEFAULT 1
            )
        """)
        # Initialize default enabled sources
        cursor.execute("INSERT OR IGNORE INTO enabled_sources (source_id, name, enabled) VALUES ('mangadex', 'MangaDex (Global)', 1)")
        cursor.execute("INSERT OR IGNORE INTO enabled_sources (source_id, name, enabled) VALUES ('mangakawaii', 'MangaKawaii (FR)', 1)")
        conn.commit()

def get_favorites_library() -> list:
    init_library_tables()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, source_id, manga_id, title, cover_url, status, category, last_chapter, unread_count, created_at
            FROM favorites_library
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "source_id": r[1],
                "manga_id": r[2],
                "title": r[3],
                "cover_url": r[4],
                "status": r[5],
                "category": r[6],
                "last_chapter": r[7],
                "unread_count": r[8],
                "created_at": r[9]
            }
            for r in rows
        ]

def toggle_favorite_manga(source_id: str, manga_id: str, title: str, cover_url: str = "") -> dict:
    init_library_tables()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM favorites_library WHERE source_id = ? AND manga_id = ?", (source_id, manga_id))
        existing = cursor.fetchone()
        if existing:
            cursor.execute("DELETE FROM favorites_library WHERE source_id = ? AND manga_id = ?", (source_id, manga_id))
            conn.commit()
            return {"favorite": False, "message": "Удалено с полки"}
        else:
            cursor.execute("""
                INSERT INTO favorites_library (source_id, manga_id, title, cover_url, status, category)
                VALUES (?, ?, ?, ?, 'reading', 'default')
            """, (source_id, manga_id, title, cover_url))
            conn.commit()
            return {"favorite": True, "message": "Добавлено на полку"}

def is_manga_in_library(source_id: str, manga_id: str) -> bool:
    init_library_tables()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM favorites_library WHERE source_id = ? AND manga_id = ?", (source_id, manga_id))
        return cursor.fetchone() is not None

def get_enabled_sources_list() -> list:
    init_library_tables()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT source_id, name, enabled FROM enabled_sources")
        rows = cursor.fetchall()
        return [{"source_id": r[0], "name": r[1], "enabled": bool(r[2])} for r in rows]

def set_source_enabled_state(source_id: str, name: str, enabled: bool):
    init_library_tables()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO enabled_sources (source_id, name, enabled)
            VALUES (?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET enabled = excluded.enabled
        """, (source_id, name, 1 if enabled else 0))
        conn.commit()
