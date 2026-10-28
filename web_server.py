import os
import time
import uuid
import json
import mimetypes
import re
import zipfile
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from mangakawaii_scraper import MangaKawaiiClient
from translator_service import MangaTranslatorService, MangaYOLOModel
from db_cache import get_all_cached_translations_for_page, get_cached_audio, save_cached_audio, clear_offline_cache, init_db, get_cached_page_bubbles, is_page_fully_translated, make_box_key

app = FastAPI(title="Manga Smart Reader & Cloud Hub")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOWNLOAD_DIR = os.getenv("MANGA_DOWNLOAD_DIR", "/root/manga_downloads")

init_db()

templates = Jinja2Templates(directory=TEMPLATES_DIR)
client = MangaKawaiiClient()

tasks: Dict[str, Dict] = {}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", value)]


def chapter_path(manga_name: str, filename: str) -> str:
    # Route parameters must stay inside the configured downloads directory.
    if os.path.basename(manga_name) != manga_name or os.path.basename(filename) != filename:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return os.path.join(DOWNLOAD_DIR, manga_name, filename)


def image_entries(archive: zipfile.ZipFile):
    return sorted(
        (name for name in archive.namelist()
         if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
         and not name.startswith("__MACOSX/")
         and not os.path.basename(name).startswith(".")),
        key=natural_key,
    )


class DownloadRequest(BaseModel):
    url_or_slug: str
    from_chapter: Optional[float] = None
    to_chapter: Optional[float] = None
    concurrency: int = 6


class TranslateRegionRequest(BaseModel):
    manga_name: str
    filename: str
    page_idx: int
    box: Optional[Dict[str, float]] = None
    src_lang: str = "fra"
    target_lang: str = "ru"


class BatchTranslateChapterRequest(BaseModel):
    manga_name: str
    filename: str
    start_page: int = 0
    end_page: Optional[int] = None


def get_all_manga_files():
    data = {}
    if not os.path.exists(DOWNLOAD_DIR):
        return data

    for manga_name in sorted(os.listdir(DOWNLOAD_DIR)):
        manga_path = os.path.join(DOWNLOAD_DIR, manga_name)
        if os.path.isdir(manga_path):
            files = []
            for f in os.listdir(manga_path):
                if f.endswith(".cbz"):
                    full_p = os.path.join(manga_path, f)
                    stat = os.stat(full_p)
                    files.append({
                        "filename": f,
                        "rel_path": f"{manga_name}/{f}",
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
                    })
            if files:
                def sort_key(item):
                    match = re.search(r"(?:Ch\.|Chapter\s*)(\d+(?:\.\d+)?)", item["filename"], re.I)
                    return (0, float(match.group(1)), natural_key(item["filename"])) if match else (1, natural_key(item["filename"]))
                files.sort(key=sort_key)
                data[manga_name] = files
    return data


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    mangas = get_all_manga_files()
    total_files = sum(len(v) for v in mangas.values())
    total_size_mb = round(sum(sum(f["size_mb"] for f in v) for v in mangas.values()), 1)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "mangas": mangas,
            "total_files": total_files,
            "total_size_mb": total_size_mb
        }
    )


@app.get("/reader/{manga_name}/{filename}", response_class=HTMLResponse)
def reader_page(request: Request, manga_name: str, filename: str):
    file_path = chapter_path(manga_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chapter not found")

    with zipfile.ZipFile(file_path, "r") as zf:
        pages = image_entries(zf)

    all_manga = get_all_manga_files()
    all_chapters = all_manga.get(manga_name, [])

    return templates.TemplateResponse(
        request=request,
        name="reader.html",
        context={
            "manga_name": manga_name,
            "filename": filename,
            "pages_list": pages,
            "all_chapters": all_chapters
        }
    )


@app.get("/page-image/{manga_name}/{filename}/{page_idx}")
def get_page_image(manga_name: str, filename: str, page_idx: int):
    file_path = chapter_path(manga_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chapter not found")

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            names = image_entries(zf)
            if page_idx < 0 or page_idx >= len(names):
                raise HTTPException(status_code=404, detail="Page index out of range")
            
            target = names[page_idx]
            media_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
            data = zf.read(target)
            return Response(
                content=data,
                media_type=media_type,
                headers={"Cache-Control": "private, max-age=86400"},
            )
    except HTTPException:
        raise
    except (zipfile.BadZipFile, OSError, KeyError) as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate-region")
def translate_region(req: TranslateRegionRequest):
    """
    OCRs and translates a user-selected crop box or dialog bubble.
    Checks local SQLite cache first before calling Groq Vision.
    """
    file_path = chapter_path(req.manga_name, req.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chapter not found")

    try:
        result = MangaTranslatorService.ocr_and_translate_region(
            cbz_path=file_path,
            manga_name=req.manga_name,
            filename=req.filename,
            page_idx=req.page_idx,
            box=req.box,
            src_lang=req.src_lang,
            target_lang=req.target_lang,
            use_cache=True
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detect-bubbles/{manga_name}/{filename}/{page_idx}")
def detect_bubbles(manga_name: str, filename: str, page_idx: int):
    """
    Detects all speech bubbles on the given page using YOLO26n.
    Results are cached locally.
    """
    file_path = chapter_path(manga_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chapter not found")

    if not MangaYOLOModel.get_instance().interpreter:
        raise HTTPException(status_code=503, detail="Bubble detector is unavailable. Install dependencies from requirements.txt.")

    try:
        bubbles = MangaTranslatorService.detect_bubbles(
            cbz_path=file_path,
            manga_name=manga_name,
            filename=filename,
            page_idx=page_idx
        )
        return bubbles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/page-overlays/{manga_name}/{filename}/{page_idx}")
def get_page_overlays(manga_name: str, filename: str, page_idx: int):
    """
    Returns all pre-cached translations for this page to display parallel Russian overlays!
    """
    items = get_all_cached_translations_for_page(manga_name, filename, page_idx)
    saved_bubbles = get_cached_page_bubbles(manga_name, filename, page_idx) or []
    if saved_bubbles and any(bubble.get("source") != "yolo26n-v3" for bubble in saved_bubbles):
        stale_keys = {make_box_key(bubble.get("box")) for bubble in saved_bubbles if bubble.get("box")}
        items = [item for item in items if make_box_key(item.get("box")) not in stale_keys]
    return items


# Deepgram Aura-2 French Male TTS (Hector)
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_VOICE_MODEL = os.environ.get("DEEPGRAM_VOICE_MODEL", "aura-2-hector-fr")
DEEPGRAM_VOICE_TAG = f"deepgram_{DEEPGRAM_VOICE_MODEL}"


def normalize_french_for_tts(text: str) -> str:
    """
    Normalizes manga speech text so Deepgram speaks in a continuous, natural cadence:
    1. Glues OCR-broken and line-wrapped hyphenated words (e.g. 'MAIN-TENANT' -> 'maintenant').
    2. Fixes spaced apostrophes and contractions (e.g. 'c ' est' -> 'c\'est', 'd \' un' -> 'd\'un').
    3. Normalizes Comic ALL-CAPS font to standard sentence casing to prevent syllable-by-syllable or spelled-out delivery.
    """
    if not text:
        return ""
    
    # Common legitimate French hyphenated words to preserve
    legit_hyphens = {
        "est-ce", "peut-être", "rendez-vous", "au-dessus", "au-dessous",
        "celui-ci", "celle-ci", "ceux-ci", "celles-ci", "là-bas", "par-dessus"
    }

    # Rejoin line-wrap hyphens with space: "CONNAIS- SIEZ" -> "CONNAISSIEZ"
    text = re.sub(r"([A-Za-zÀ-ÿ]{2,})-\s+([A-Za-zÀ-ÿ]{2,})", r"\1\2", text)

    # Rejoin word-break hyphens: "MAIN-TENANT" -> "MAINTENANT"
    def clean_hyphen(m):
        full = m.group(0).lower()
        if full in legit_hyphens or m.group(2).lower() in ["tu", "vous", "il", "elle", "ils", "elles", "on", "moi", "toi", "lui", "la", "le", "les", "ce", "en", "y"]:
            return m.group(0)
        return m.group(1) + m.group(2)

    text = re.sub(r"([A-Za-zÀ-ÿ]{3,})-([A-Za-zÀ-ÿ]{3,})", clean_hyphen, text)

    # Rejoin detached apostrophes: "C ' EST" -> "C'EST"
    text = re.sub(r"([A-Za-zÀ-ÿ])\s*['’`]\s*([A-Za-zÀ-ÿ])", r"\1'\2", text)

    # Collapse multiple spaces and newlines into a single clean space
    text = re.sub(r"\s+", " ", text).strip()

    # Convert ALL-CAPS manga lettering into natural sentence case
    if text.isupper() and len(text) > 3:
        parts = re.split(r"([.!?]+\s*)", text.lower())
        cased = ""
        for p in parts:
            if p:
                cased += p[0].upper() + p[1:] if len(p) > 1 else p.upper()
        text = cased

    return text


def call_deepgram_tts(clean_text: str) -> bytes:
    """Synthesize speech strictly with Deepgram (aura-2-hector-fr) and return MP3 audio bytes."""
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured")

    # Pass fluent, natural French text into Deepgram
    tts_text = normalize_french_for_tts(clean_text)
    if not tts_text:
        tts_text = clean_text

    url = f"https://api.deepgram.com/v1/speak?model={DEEPGRAM_VOICE_MODEL}"
    payload = json.dumps({"text": tts_text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    try:
        resp = urllib.request.urlopen(req, timeout=12)
        audio_bytes = resp.read()
        if len(audio_bytes) < 100:
            raise RuntimeError("Deepgram returned empty audio response")
        return audio_bytes
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Deepgram HTTP {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Deepgram TTS error: {e}")


@app.get("/api/voice-tts")
def voice_tts(text: str = Query(..., min_length=1), lang: str = Query("fr"), emotion: str = Query("neutral")):
    """Use Deepgram French male voice (Hector) with persistent SQLite audio cache."""
    clean_text = text.strip()[:400]
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=503, detail="Deepgram TTS is not configured")

    cached = get_cached_audio(clean_text, voice_tag=DEEPGRAM_VOICE_TAG)
    if cached:
        return Response(content=cached[0], media_type=cached[1])

    try:
        audio_bytes = call_deepgram_tts(clean_text)
        save_cached_audio(clean_text, audio_bytes, "audio/mpeg", voice_tag=DEEPGRAM_VOICE_TAG)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        print(f"[!] Deepgram speech error: {e}")
        raise HTTPException(status_code=502, detail=f"Deepgram speech synthesis failed: {e}")


@app.post("/api/clear-offline-cache")
def clear_offline_cache_endpoint(manga_name: Optional[str] = Query(None), filename: Optional[str] = Query(None)):
    """
    Clears local offline database (translations, bubbles, audio).
    Can target an entire manga/chapter or completely wipe all local cache.
    """
    try:
        stats = clear_offline_cache(manga_name, filename)
        return {"status": "ok", "deleted": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def bg_chapter_translation(task_id: str, req: BatchTranslateChapterRequest):
    tasks[task_id]["status"] = "in_progress"
    try:
        if not MangaYOLOModel.get_instance().interpreter:
            raise RuntimeError("Детектор пузырей не загружен. Установите зависимости из requirements.txt.")
        file_path = chapter_path(req.manga_name, req.filename)
        with zipfile.ZipFile(file_path, "r") as zf:
            pages = image_entries(zf)
        
        total_p = len(pages)
        end_p = min(total_p, req.end_page) if req.end_page is not None else total_p
        start_p = max(0, req.start_page)

        tasks[task_id]["total_pages"] = end_p - start_p
        tasks[task_id]["completed_pages"] = 0

        for idx in range(start_p, end_p):
            tasks[task_id]["current_page"] = idx + 1
            tasks[task_id]["message"] = f"Страница {idx+1}/{end_p}: Распознавание и перевод..."
            
            # 1. Translate bubbles on this page
            page_bubbles = get_cached_page_bubbles(req.manga_name, req.filename, idx)
            page_ready = is_page_fully_translated(req.manga_name, req.filename, idx)
            voice_tag = DEEPGRAM_VOICE_TAG
            audio_ready = bool(page_bubbles) and all(
                not (bubble.get("translation") or {}).get("original", "").strip()
                or get_cached_audio((bubble.get("translation") or {}).get("original", ""), voice_tag=voice_tag)
                for bubble in page_bubbles
            )
            if page_ready and audio_ready:
                tasks[task_id]["completed_pages"] += 1
                continue
            if not page_ready:
                page_bubbles = MangaTranslatorService.batch_translate_page(
                    cbz_path=file_path,
                    manga_name=req.manga_name,
                    filename=req.filename,
                    page_idx=idx
                )
            page_bubbles = page_bubbles or []

            # 2. Pre-generate available French speech once with Deepgram; future plays use SQLite cache.
            for b in page_bubbles:
                tr = b.get("translation") or {}
                orig_text = tr.get("original", "").strip()
                if orig_text and len(orig_text) > 1:
                    voice_tag = DEEPGRAM_VOICE_TAG
                    if DEEPGRAM_API_KEY and not get_cached_audio(orig_text, voice_tag=voice_tag):
                        try:
                            audio_bytes = call_deepgram_tts(orig_text)
                            save_cached_audio(orig_text, audio_bytes, "audio/mpeg", voice_tag=voice_tag)
                        except Exception as e:
                            print(f"[!] Deepgram speech cache skipped for '{orig_text[:20]}': {e}")

            tasks[task_id]["completed_pages"] += 1
            time.sleep(0.35)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = f"Обработано {end_p - start_p} страниц. Позиции, переводы и доступная озвучка сохранены локально."
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"Ошибка: {e}"


@app.post("/api/batch-translate-chapter")
def batch_translate_chapter(req: BatchTranslateChapterRequest, background_tasks: BackgroundTasks):
    """
    Pre-translates an entire chapter in the background, saving all bubbles and Russian overlays into SQLite.
    """
    for existing in tasks.values():
        if (existing.get("kind") == "batch_translate"
                and existing.get("manga_name") == req.manga_name
                and existing.get("filename") == req.filename
                and existing.get("status") in {"pending", "in_progress"}):
            return {"task_id": existing["id"], "reused": True}
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "kind": "batch_translate",
        "manga_name": req.manga_name,
        "filename": req.filename,
        "status": "pending",
        "message": "В очереди на пакетный перевод главы...",
        "total_pages": 0,
        "completed_pages": 0
    }
    background_tasks.add_task(bg_chapter_translation, task_id, req)
    return {"task_id": task_id}


@app.get("/download-file/{manga_name}/{filename}")
def download_cbz(manga_name: str, filename: str):
    file_path = chapter_path(manga_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.comicbook+zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


def bg_download_task(task_id: str, req: DownloadRequest):
    slug = req.url_or_slug.strip().split("/")[-1]
    if not re.fullmatch(r"[a-z0-9-]+", slug, re.I):
        tasks[task_id].update(status="failed", message="Укажите корректный slug манги.")
        return
    if req.from_chapter is not None and req.to_chapter is not None and req.from_chapter > req.to_chapter:
        tasks[task_id].update(status="failed", message="Начальная глава больше конечной.")
        return
    out_dir = os.path.join(DOWNLOAD_DIR, slug.replace("-", "_").title())
    os.makedirs(out_dir, exist_ok=True)

    tasks[task_id]["status"] = "in_progress"
    try:
        manga = client.get_manga_details(slug)
        manga_title = manga["title"] or slug
        chapters = manga["chapters"]

        targets = []
        for ch in chapters:
            try:
                num = float(ch["number"])
            except ValueError:
                num = 0.0
            if req.from_chapter is not None and num < req.from_chapter:
                continue
            if req.to_chapter is not None and num > req.to_chapter:
                continue
            ch_num = ch["number"]
            ch_filename = f"{manga_title} - Ch.{ch_num.zfill(2) if ch_num.isdigit() else ch_num}.cbz"
            local_path = os.path.join(out_dir, ch_filename)
            already_downloaded = False
            if os.path.isfile(local_path) and os.path.getsize(local_path) > 10000:
                try:
                    with zipfile.ZipFile(local_path) as existing_archive:
                        already_downloaded = bool(image_entries(existing_archive))
                except zipfile.BadZipFile:
                    already_downloaded = False
            if already_downloaded:
                tasks[task_id]["completed_chapters"] = tasks[task_id].get("completed_chapters", 0) + 1
            else:
                targets.append(ch)

        already_count = tasks[task_id].get("completed_chapters", 0)
        tasks[task_id]["total_chapters"] = len(targets) + already_count
        tasks[task_id]["completed_chapters"] = already_count
        if already_count:
            tasks[task_id]["message"] = f"Найдено локально глав: {already_count}. Проверяю, чего не хватает…"
        if not targets:
            if already_count:
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["message"] = f"Все {already_count} глав уже находятся локально."
                return
            raise ValueError("В выбранном диапазоне главы не найдены.")

        for idx, ch in enumerate(targets):
            tasks[task_id]["message"] = f"Скачивание главы {ch['number']} ({already_count + idx + 1}/{tasks[task_id]['total_chapters']})"
            client.download_chapter(
                chapter_info=ch,
                output_dir=out_dir,
                manga_title=manga_title,
                concurrency=req.concurrency
            )
            tasks[task_id]["completed_chapters"] += 1

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = f"Загружено {len(targets)} глав!"

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"Ошибка: {str(e)}"


@app.post("/api/download-task")
def trigger_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    requested_slug = req.url_or_slug.strip().rstrip("/").split("/")[-1].split("?")[0]
    for existing in tasks.values():
        if (existing.get("kind") == "download" and existing.get("slug") == requested_slug
                and existing.get("status") in {"pending", "in_progress"}):
            return {"task_id": existing["id"], "reused": True}
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "kind": "download",
        "slug": requested_slug,
        "status": "pending",
        "message": "В очереди...",
        "total_chapters": 0,
        "completed_chapters": 0
    }
    background_tasks.add_task(bg_download_task, task_id, req)
    return {"task_id": task_id}


@app.get("/api/task/{task_id}")
def check_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8080, reload=False)
