import os
import time
import uuid
import json
import mimetypes
import re
import zipfile
import hashlib
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from mangakawaii_scraper import MangaKawaiiClient
from text_quality import is_russian_translation, is_usable_ocr_text
from translator_service import MangaTranslatorService, MangaYOLOModel
from db_cache import (get_all_cached_translations_for_page, init_db, get_cached_page_bubbles,
                      is_page_fully_translated, make_box_key, get_cached_reading_order,
                      save_cached_reading_order, get_chapter_text_stats,
                      get_all_cached_chapter_bubbles, get_cached_chapter_voiceover,
                      save_cached_chapter_voiceover, get_cached_chapter_voiceover_analysis,
                      save_cached_chapter_voiceover_analysis, get_staged_chapter_voiceover_audio,
                      save_staged_chapter_voiceover_audio, delete_staged_chapter_voiceover_audio,
                      get_cached_chapter_voiceover_tts_chunk,
                      save_cached_chapter_voiceover_tts_chunk,
                      delete_cached_chapter_voiceover_tts_chunks, clear_offline_cache)
from keiyoushi_catalog import parse_keiyoushi_index
from manga_sources import multi_source_search, search_mangadex, search_mangakawaii, get_manga_details_and_chapters, get_chapter_pages
from db_cache import (get_favorites_library, toggle_favorite_manga, is_manga_in_library,
                      get_enabled_sources_list, set_source_enabled_state)
from chapter_voiceover import (DEEPGRAM_TTS_MODEL, GEMINI_FRENCH_MALE_VOICE,
                               GEMINI_TTS_MODEL, analyze_chapter_with_jev, has_pairwise_order_analysis,
                               chapter_fingerprint, order_chapter_items, synthesize_chapter_gemini,
                               transcribe_and_align)

app = FastAPI(title="Manga Smart Reader & Cloud Hub")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOWNLOAD_DIR = os.getenv("MANGA_DOWNLOAD_DIR", os.path.join(BASE_DIR, "downloads"))

init_db()

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
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


class InstantVoiceRequest(BaseModel):
    text: str


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


@app.get("/manifest.webmanifest")
def pwa_manifest():
    return JSONResponse({
        "id": "/", "name": "MangaReader", "short_name": "MangaReader",
        "description": "Локальная библиотека манги и ридер",
        "start_url": "/", "scope": "/", "display": "standalone",
        "background_color": "#111827", "theme_color": "#111827",
        "icons": [
            {"src": "/static/pwa-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": "/static/pwa-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }, media_type="application/manifest+json")


@app.get("/pwa-icon.svg")
def pwa_icon(size: int = Query(192, ge=192, le=512)):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 192 192"><rect width="192" height="192" rx="42" fill="#111827"/><path d="M38 42h53c12 0 20 8 20 20v90c0-12-8-20-20-20H38z" fill="#f97316"/><path d="M154 42h-33c-6 0-10 8-10 20v90c0-12 8-20 17-20h26z" fill="#fff"/><path d="M55 67h39M55 83h39M55 99h29" stroke="#fff" stroke-width="7" stroke-linecap="round"/></svg>'''
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/sw.js")
def pwa_service_worker():
    script = '''const CACHE = "mangai-shell-v1";
self.addEventListener("install", event => event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(["/", "/manifest.webmanifest", "/static/pwa-192.png", "/static/pwa-512.png"]))));
self.addEventListener("activate", event => event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key))))));
self.addEventListener("fetch", event => {
  const request = event.request;
  if (request.method !== "GET" || new URL(request.url).origin !== self.location.origin) return;
  const url = new URL(request.url);
  if (url.pathname.startsWith("/page-image/")) {
    event.respondWith(caches.open(CACHE).then(async cache => {
      const cached = await cache.match(request);
      if (cached) return cached;
      const response = await fetch(request);
      if (response.ok) cache.put(request, response.clone());
      return response;
    }));
    return;
  }
  if (url.pathname === "/" || url.pathname.startsWith("/reader/")) {
    event.respondWith(fetch(request).then(response => {
      if (!response.ok) return response;
      return caches.open(CACHE).then(cache => cache.put(request, response.clone()).then(() => response));
    }).catch(async () => (await caches.match(request)) || (await caches.match("/"))));
  }
});'''
    return Response(script, media_type="application/javascript", headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


@app.get("/api/proxy-image")
def proxy_image(url: str = Query(...)):
    """
    Proxies external manga chapter images with necessary Referer and User-Agent headers
    to bypass hotlinking protections (e.g. Remanga, MangaDex, MangaKawaii).
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Invalid URL protocol")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }
    if "remanga" in url or "reimg" in url:
        headers["Referer"] = "https://remanga.org"
        headers["Origin"] = "https://remanga.org"
    elif "mangadex" in url:
        headers["Referer"] = "https://mangadex.org"
    elif "mangakawaii" in url:
        headers["Referer"] = "https://www.mangakawaii.io"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type") or "image/jpeg"
            data = resp.read()
            return Response(
                content=data,
                media_type=content_type,
                headers={"Cache-Control": "public, max-age=86400"}
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image fetch error: {e}")


@app.get("/online-reader/{source_id}/{manga_id}/{chapter_id}", response_class=HTMLResponse)
def online_reader_page(request: Request, source_id: str, manga_id: str, chapter_id: str):
    """
    Online Mihon-style reader for direct chapter reading from MangaDex, Remanga, MangaKawaii
    without downloading local CBZ first.
    """
    pages = get_chapter_pages(source_id, manga_id, chapter_id)
    if not pages:
        raise HTTPException(status_code=404, detail="Не удалось получить страницы главы из источника")

    # Fetch manga details to get all chapters for dropdown navigation
    details = get_manga_details_and_chapters(source_id, manga_id)
    manga_title = details.get("title") or manga_id
    all_chapters = []
    current_chapter_title = f"Глава {chapter_id}"
    for ch in details.get("chapters", []):
        ch_id = str(ch.get("id"))
        ch_title = ch.get("title") or f"Глава {ch_id}"
        if ch_id == str(chapter_id):
            current_chapter_title = ch_title
        all_chapters.append({
            "filename": ch_title,
            "chapter_id": ch_id,
            "url": f"/online-reader/{source_id}/{manga_id}/{ch_id}",
            "size_mb": "онлайн"
        })

    return templates.TemplateResponse(
        request=request,
        name="reader.html",
        context={
            "manga_name": manga_title,
            "filename": current_chapter_title,
            "pages_list": pages,
            "all_chapters": all_chapters,
            "is_online": True,
            "source_id": source_id,
            "manga_id": manga_id,
            "chapter_id": chapter_id
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
    chapter_bubbles = get_all_cached_chapter_bubbles(manga_name, filename)
    if chapter_bubbles:
        cached_voiceover = get_cached_chapter_voiceover(
            manga_name, filename, chapter_fingerprint(chapter_bubbles), include_audio=False
        )
        if cached_voiceover and cached_voiceover.get("model") == GEMINI_TTS_MODEL:
            emotions = {(bubble["page_idx"], make_box_key(bubble["box"])): bubble.get("emotion", "neutral")
                        for bubble in cached_voiceover["bubbles"]}
            for item in items:
                item["emotion"] = emotions.get((page_idx, make_box_key(item.get("box"))), "neutral")
    return items


@app.get("/api/chapter-text-stats/{manga_name}/{filename}")
def chapter_text_stats(manga_name: str, filename: str):
    if not os.path.exists(chapter_path(manga_name, filename)):
        raise HTTPException(status_code=404, detail="Chapter not found")
    return get_chapter_text_stats(manga_name, filename)


@app.get("/api/reading-order/{manga_name}/{filename}/{page_idx}")
def get_reading_order(manga_name: str, filename: str, page_idx: int):
    """Return cached translations in a Jev-ranked narrative order for audio mode."""
    chapter = chapter_path(manga_name, filename)
    if not os.path.exists(chapter):
        raise HTTPException(status_code=404, detail="Chapter not found")
    page_bubbles = get_cached_page_bubbles(manga_name, filename, page_idx) or []
    translations_by_box = {
        make_box_key(item.get("box")): item
        for item in get_all_cached_translations_for_page(manga_name, filename, page_idx)
        if item.get("box")
    }
    items = []
    if page_bubbles:
        for bubble in page_bubbles:
            translation = translations_by_box.get(make_box_key(bubble.get("box"))) or bubble.get("translation") or {}
            if (bubble.get("box") and is_usable_ocr_text(translation.get("original", ""))
                    and is_russian_translation(translation.get("original", ""), translation.get("natural", ""))):
                items.append({"box": bubble["box"], **translation})
    if not items:
        items = [i for i in get_all_cached_translations_for_page(manga_name, filename, page_idx)
                 if i.get("box") and i.get("original")]
    if not items:
        raise HTTPException(status_code=409, detail="Сначала распознайте и переведите баблы на странице")

    def geometric_key(item):
        box = item["box"]
        # Manga panels are commonly read right-to-left; row grouping stays a fallback only.
        return (round(float(box.get("y", 0)) / max(float(box.get("h", 1)), 1) * 0.55), -float(box.get("x", 0)))

    fallback = sorted(items, key=geometric_key)
    digest_data = [{"box": i["box"], "original": i.get("original", ""), "natural": i.get("natural", "")} for i in items]
    fingerprint = hashlib.sha256(json.dumps(digest_data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cached = get_cached_reading_order(manga_name, filename, page_idx, fingerprint)
    if cached:
        ordered = [items[index] for index in cached if isinstance(index, int) and 0 <= index < len(items)]
        if len(ordered) == len(items):
            return ordered

    api_key = os.getenv("TYPESAFE_API_KEY", "")
    if api_key and len(items) > 1 and len(items) <= 50:
        ids = [str(n) for n in range(len(items))]
        state = {"speech_bubbles": [
            {"id": ids[n], "text": (item.get("natural") or item.get("original") or "")[:500],
             "position": {"x": item["box"].get("x"), "y": item["box"].get("y")}}
            for n, item in enumerate(items)
        ]}
        positions = {str(n + 1): f"Narrative position {n + 1}, from first to last." for n in range(len(items))}
        questions = {
            f"bubble_{n}": {
                "type": "choice",
                "instructions": (
                    f"For speech_bubbles[{n}] in state, choose its most likely position in the page's spoken narrative. "
                    "Use the dialogue meaning, pronoun references, question-answer links, cause and effect, and speaker turns. "
                    "Use spatial manga reading order as a tie-breaker: panels generally progress from top to bottom and right to left. "
                    "Assign a distinct sensible position where possible; do not put a reply before the line it answers."
                ),
                "criteria": positions,
            } for n in range(len(items))
        }
        payload = json.dumps({"model": "jev-latest", "state": state, "questions": questions}).encode()
        request = urllib.request.Request(
            "https://api.typesafe.ai/v1/systemone", data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                answers = json.loads(response.read()).get("answers", {})
            scores = []
            for n in range(len(items)):
                answer = answers.get(f"bubble_{n}", {})
                distribution = answer.get("probabilities") or {}
                scores.append(sum(int(position) * float(probability) for position, probability in distribution.items()
                                  if position.isdigit()) if distribution else int(answer.get("choice", n + 1)))
            ranked = sorted(range(len(items)), key=lambda n: (scores[n], geometric_key(items[n])))
            ordered = [items[n] for n in ranked]
            save_cached_reading_order(manga_name, filename, page_idx, fingerprint, ranked)
            return ordered
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
            print(f"[!] Jev reading order unavailable: {type(error).__name__}; using spatial order")
    return fallback


@app.get("/api/chapter-voiceover/{manga_name}/{filename}")
def chapter_voiceover_status(manga_name: str, filename: str):
    if not os.path.exists(chapter_path(manga_name, filename)):
        raise HTTPException(status_code=404, detail="Chapter not found")
    items = get_all_cached_chapter_bubbles(manga_name, filename)
    if not items:
        return {"status": "missing_bubbles", "bubbles": 0}
    fingerprint = chapter_fingerprint(items)
    cached = get_cached_chapter_voiceover(manga_name, filename, fingerprint, include_audio=False)
    if cached and cached.get("model") != GEMINI_TTS_MODEL:
        cached = None
    if cached:
        cached_bubbles = [dict(item) for item in cached["bubbles"]]
        analysis = get_cached_chapter_voiceover_analysis(manga_name, filename, fingerprint) or []
        annotations = {item["id"]: item for item in analysis}
        for bubble in cached_bubbles:
            annotation = annotations.get(bubble["id"], {})
            for key in ("pairwise_order", "emotion"):
                if key in annotation:
                    bubble[key] = annotation[key]
        cached_bubbles = order_chapter_items(cached_bubbles)
        return {"status": "ready", "fingerprint": fingerprint, "segments": cached["segments"],
                "bubbles": cached_bubbles, "duration": cached["duration"],
                "voice": cached["voice"], "model": cached["model"]}
    active = next((task for task in tasks.values()
                   if task.get("kind") == "chapter_voiceover" and task.get("manga_name") == manga_name
                   and task.get("filename") == filename and task.get("fingerprint") == fingerprint
                   and task.get("status") in {"pending", "in_progress"}), None)
    if active:
        return {"status": active["status"], "task_id": active["id"], "message": active.get("message"),
                "progress": active.get("progress", 0), "bubbles": len(items)}
    return {"status": "not_ready", "fingerprint": fingerprint, "bubbles": len(items)}


@app.post("/api/chapter-voiceover/{manga_name}/{filename}")
def start_chapter_voiceover(manga_name: str, filename: str, background_tasks: BackgroundTasks):
    chapter = chapter_path(manga_name, filename)
    if not os.path.exists(chapter):
        raise HTTPException(status_code=404, detail="Chapter not found")
    items = get_all_cached_chapter_bubbles(manga_name, filename)
    if not items:
        raise HTTPException(status_code=409, detail="Сначала распознай и переведи баблы главы.")
    fingerprint = chapter_fingerprint(items)
    cached = get_cached_chapter_voiceover(manga_name, filename, fingerprint, include_audio=False)
    if cached and cached.get("model") == GEMINI_TTS_MODEL:
        return {"status": "ready", "cached": True, "bubbles": len(items), "duration": cached["duration"]}
    staged_audio = get_staged_chapter_voiceover_audio(
        manga_name, filename, fingerprint, GEMINI_TTS_MODEL)
    analyzed = get_cached_chapter_voiceover_analysis(manga_name, filename, fingerprint)
    if not staged_audio and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="Добавь GEMINI_API_KEY в локальный .env для Gemini TTS.")
    if not analyzed and not os.getenv("TYPESAFE_API_KEY"):
        raise HTTPException(status_code=503, detail="Добавь TYPESAFE_API_KEY для порядка реплик Jev.")
    for task in tasks.values():
        if (task.get("kind") == "chapter_voiceover" and task.get("manga_name") == manga_name
                and task.get("filename") == filename and task.get("fingerprint") == fingerprint
                and task.get("status") in {"pending", "in_progress"}):
            return {"status": task["status"], "task_id": task["id"], "reused": True}
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"id": task_id, "kind": "chapter_voiceover", "manga_name": manga_name,
                      "filename": filename, "fingerprint": fingerprint, "status": "pending",
                      "progress": 0, "message": "Задача озвучки поставлена в очередь…", "bubbles": len(items)}
    background_tasks.add_task(bg_chapter_voiceover, task_id, manga_name, filename, fingerprint, items)
    return {"status": "pending", "task_id": task_id, "bubbles": len(items)}


@app.post("/api/instant-voice")
def instant_voice(req: InstantVoiceRequest):
    """Generate a non-cached single-bubble preview with Deepgram."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Текст реплики пуст.")
    if len(text) > 2000:
        raise HTTPException(status_code=413, detail="Мгновенная реплика превышает лимит Deepgram в 2000 символов.")
    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="Добавь DEEPGRAM_API_KEY для мгновенной озвучки Deepgram.")
    try:
        response = requests.post(
            "https://api.deepgram.com/v1/speak",
            params={"model": DEEPGRAM_TTS_MODEL, "encoding": "linear16", "container": "wav"},
            headers={"Authorization": f"Token {api_key}"}, json={"text": text}, timeout=(15, 180))
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Deepgram недоступен: {type(error).__name__}") from error
    if not response.ok:
        raise HTTPException(status_code=502, detail=f"Deepgram вернул HTTP {response.status_code}.")
    if not response.content.startswith(b"RIFF"):
        raise HTTPException(status_code=502, detail="Deepgram вернул некорректный WAV.")
    return Response(content=response.content, media_type="audio/wav",
                    headers={"Cache-Control": "no-store", "X-TTS-Provider": "Deepgram"})


@app.get("/api/chapter-voiceover/{manga_name}/{filename}/audio")
def get_chapter_voiceover_audio(manga_name: str, filename: str):
    if not os.path.exists(chapter_path(manga_name, filename)):
        raise HTTPException(status_code=404, detail="Chapter not found")
    items = get_all_cached_chapter_bubbles(manga_name, filename)
    cached = get_cached_chapter_voiceover(manga_name, filename, chapter_fingerprint(items)) if items else None
    if not cached or cached.get("model") != GEMINI_TTS_MODEL:
        raise HTTPException(status_code=404, detail="Озвучка главы ещё не собрана")
    return Response(content=cached["audio_data"], media_type=cached["media_type"],
                    headers={"Cache-Control": "private, max-age=31536000", "Accept-Ranges": "none"})


def bg_chapter_voiceover(task_id: str, manga_name: str, filename: str, fingerprint: str, items: List[Dict]):
    task = tasks[task_id]
    task.update(status="in_progress", progress=5, message="Jev расставляет реплики и отмечает эмоции…")
    try:
        analyzed = get_cached_chapter_voiceover_analysis(manga_name, filename, fingerprint)
        if analyzed and has_pairwise_order_analysis(items):
            items = order_chapter_items(analyzed)
            save_cached_chapter_voiceover_analysis(manga_name, filename, fingerprint, items)
            task.update(progress=24, message="Беру сохранённый порядок Jev и метки эмоций…")
        else:
            items = analyze_chapter_with_jev(items, os.getenv("TYPESAFE_API_KEY", ""))
            save_cached_chapter_voiceover_analysis(manga_name, filename, fingerprint, items)
        staged_audio = get_staged_chapter_voiceover_audio(
            manga_name, filename, fingerprint, GEMINI_TTS_MODEL)
        if staged_audio:
            audio_data, media_type = staged_audio
            task.update(progress=74, message="Использую сохранённую Gemini-дорожку; Whisper уточняет границы фраз…")
        else:
            task.update(progress=25, message="Gemini готовит офлайн-аудиоглаву и сохраняет пакеты локально…")

            def load_tts_chunk(index, digest):
                return get_cached_chapter_voiceover_tts_chunk(
                    manga_name, filename, fingerprint, GEMINI_TTS_MODEL, index, digest)

            def save_tts_chunk(index, digest, chunk_audio):
                save_cached_chapter_voiceover_tts_chunk(
                    manga_name, filename, fingerprint, GEMINI_TTS_MODEL, index, digest,
                    chunk_audio, "audio/wav")

            def update_tts_progress(done, total):
                progress = 30 + int(42 * done / max(1, total))
                task.update(progress=progress,
                            message=f"Gemini формирует офлайн-аудио: часть {done}/{total} ({GEMINI_TTS_MODEL})…")

            audio_data, media_type = synthesize_chapter_gemini(
                items, os.getenv("GEMINI_API_KEY", ""), model=GEMINI_TTS_MODEL,
                load_chunk=load_tts_chunk, save_chunk=save_tts_chunk,
                on_progress=update_tts_progress)
            save_staged_chapter_voiceover_audio(manga_name, filename, fingerprint, audio_data,
                                                media_type, GEMINI_TTS_MODEL)
            delete_cached_chapter_voiceover_tts_chunks(manga_name, filename, fingerprint)
        task.update(progress=75, message="Whisper Small локально снимает временные метки…")
        segments, duration = transcribe_and_align(audio_data, items, os.getenv("WHISPER_MODEL", "small"))
        task.update(progress=95, message="Сохраняю аудио и тайминги баблов локально…")
        public_bubbles = [{key: item[key] for key in ("id", "page_idx", "box", "original", "natural", "emotion", "pairwise_order") if key in item}
                          for item in items]
        save_cached_chapter_voiceover(manga_name, filename, fingerprint, audio_data, media_type,
                                      segments, public_bubbles, duration,
                                      GEMINI_FRENCH_MALE_VOICE, GEMINI_TTS_MODEL)
        delete_staged_chapter_voiceover_audio(manga_name, filename, fingerprint)
        task.update(status="completed", progress=100,
                    message=f"Глава готова: {len(items)} баблов, {duration / 60:.1f} минут. Аудио и тайминги сохранены локально.",
                    duration=duration, voice=GEMINI_FRENCH_MALE_VOICE)
    except Exception as error:
        task.update(status="failed", error=str(error), message=f"Не удалось собрать аудиоглаву: {error}")


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
            if not is_page_fully_translated(req.manga_name, req.filename, idx):
                MangaTranslatorService.batch_translate_page(
                    cbz_path=file_path,
                    manga_name=req.manga_name,
                    filename=req.filename,
                    page_idx=idx,
                )
            tasks[task_id]["completed_pages"] += 1
            time.sleep(0.35)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = f"Обработано {end_p - start_p} страниц. Позиции, распознанный текст и переводы сохранены локально."
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"Ошибка: {e}"


@app.post("/api/clear-offline-cache")
def clear_offline_cache_endpoint(manga_name: Optional[str] = Query(None), filename: Optional[str] = Query(None)):
    try:
        return {"status": "ok", "deleted": clear_offline_cache(manga_name, filename)}
    except Exception as error:
        raise HTTPException(status_code=500, detail="Не удалось очистить локальный кэш.") from error


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



# ==================== KEIYOUSHI EXTENSIONS & MIHON LIBRARY API ====================

@app.get("/api/keiyoushi/extensions")
def get_keiyoushi_extensions(query: Optional[str] = None, lang: Optional[str] = None):
    try:
        exts = parse_keiyoushi_index()
        enabled_map = {s["source_id"]: s["enabled"] for s in get_enabled_sources_list()}
        
        filtered = []
        for e in exts:
            if lang and lang != "all" and e["lang"] != lang:
                continue
            if query:
                q = query.lower()
                if q not in e["name"].lower() and q not in e["pkg"].lower():
                    continue
            e_copy = dict(e)
            e_copy["enabled"] = enabled_map.get(e["slug"], False)
            filtered.append(e_copy)
            
        return {"total": len(filtered), "extensions": filtered[:120]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources/enabled")
def get_active_sources():
    return {"sources": get_enabled_sources_list()}


class ToggleSourceReq(BaseModel):
    source_id: str
    name: str
    enabled: bool

@app.post("/api/sources/toggle")
def toggle_source(req: ToggleSourceReq):
    set_source_enabled_state(req.source_id, req.name, req.enabled)
    return {"status": "ok", "source_id": req.source_id, "enabled": req.enabled}


@app.get("/api/manga/details/{source_id}/{manga_id}")
def api_manga_details(source_id: str, manga_id: str):
    data = get_manga_details_and_chapters(source_id, manga_id)
    if "error" in data and not data.get("title"):
        raise HTTPException(status_code=500, detail=data["error"])
    data["in_library"] = is_manga_in_library(source_id, manga_id)
    return data

@app.get("/api/manga/search")
def api_search_manga(q: str = Query(..., min_length=1)):
    enabled = [s["source_id"] for s in get_enabled_sources_list() if s["enabled"]]
    # Always include mangadex and mangakawaii by default if none configured
    if not enabled:
        enabled = ["mangadex", "mangakawaii"]
    results = multi_source_search(q, enabled)
    # Check favorite status
    for item in results:
        item["in_library"] = is_manga_in_library(item["source_id"], item["id"])
    return {"query": q, "total": len(results), "results": results}


@app.get("/api/library")
def get_library():
    items = get_favorites_library()
    return {"total": len(items), "library": items}


class FavoriteReq(BaseModel):
    source_id: str
    manga_id: str
    title: str
    cover_url: Optional[str] = ""

@app.post("/api/library/toggle")
def toggle_favorite(req: FavoriteReq):
    res = toggle_favorite_manga(req.source_id, req.manga_id, req.title, req.cover_url or "")
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8080, reload=False)
