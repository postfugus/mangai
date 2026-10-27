import os
import io
import time
import uuid
import wave
import base64
import json
import zipfile
import urllib.request
import urllib.parse
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Automatically load .env if present
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from mangakawaii_scraper import MangaKawaiiClient
from translator_service import MangaTranslatorService
from db_cache import get_all_cached_translations_for_page, get_cached_audio, save_cached_audio, init_db

app = FastAPI(title="Manga Smart Reader & Cloud Hub")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOWNLOAD_DIR = os.getenv("MANGA_DOWNLOAD_DIR", "/root/manga_downloads")

init_db()

templates = Jinja2Templates(directory=TEMPLATES_DIR)
client = MangaKawaiiClient()

tasks: Dict[str, Dict] = {}


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
            for f in sorted(os.listdir(manga_path)):
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
                    name = item["filename"]
                    import re
                    m = re.search(r"Ch\.(\d+(?:\.\d+)?)", name)
                    return float(m.group(1)) if m else 999999.0
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
    file_path = os.path.join(DOWNLOAD_DIR, manga_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chapter not found")

    with zipfile.ZipFile(file_path, "r") as zf:
        pages = sorted([n for n in zf.namelist() if not n.startswith("__MACOSX") and "." in n])

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
    file_path = os.path.join(DOWNLOAD_DIR, manga_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chapter not found")

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            names = sorted([n for n in zf.namelist() if not n.startswith("__MACOSX") and "." in n])
            if page_idx < 0 or page_idx >= len(names):
                raise HTTPException(status_code=404, detail="Page index out of range")
            
            target = names[page_idx]
            ext = target.split(".")[-1].lower()
            media_type = "image/png" if ext == "png" else "image/webp" if ext == "webp" else "image/jpeg"
            data = zf.read(target)
            return Response(content=data, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate-region")
def translate_region(req: TranslateRegionRequest):
    """
    OCRs and translates a user-selected crop box or dialog bubble.
    Checks local SQLite cache first before calling Groq Vision.
    """
    file_path = os.path.join(DOWNLOAD_DIR, req.manga_name, req.filename)
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
    file_path = os.path.join(DOWNLOAD_DIR, manga_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chapter not found")

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
    return get_all_cached_translations_for_page(manga_name, filename, page_idx)


# Pure Google Gemini Flash Neural TTS Engine (Male Voice: Algieba)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Models on separate Gemini TTS rate limits
GEMINI_TTS_MODELS = [
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview"
]

def call_google_gemini_tts(clean_text: str) -> bytes:
    """
    Direct call strictly to Google Gemini Flash TTS (Voice: Algieba).
    Rotates between Google Gemini Flash TTS models if one hits a temporary rate limit.
    """
    payload = json.dumps({
        "contents": [{"parts": [{"text": clean_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": "algieba"
                    }
                }
            }
        }
    }).encode('utf-8')

    last_err = None

    for model in GEMINI_TTS_MODELS:
        gemini_tts_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    gemini_tts_url,
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                resp = urllib.request.urlopen(req, timeout=12)
                res_data = json.loads(resp.read().decode('utf-8'))
                candidates = res_data.get("candidates", [])
                
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "inlineData" in p:
                            mime_type = p["inlineData"].get("mimeType", "audio/wav")
                            audio_data = base64.b64decode(p["inlineData"]["data"])
                            
                            if "pcm" in mime_type.lower() or not audio_data.startswith(b"RIFF"):
                                wav_io = io.BytesIO()
                                with wave.open(wav_io, "wb") as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(24000)
                                    wf.writeframes(audio_data)
                                return wav_io.getvalue()
                            return audio_data
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429:
                    # Switch to next Google Gemini model immediately
                    break
                time.sleep(0.5)
            except Exception as e:
                last_err = e
                time.sleep(0.5)

    raise RuntimeError(f"Google Gemini Flash TTS error: {last_err}")


@app.get("/api/voice-tts")
async def voice_tts(text: str = Query(..., min_length=1), lang: str = Query("fr"), emotion: str = Query("neutral")):
    """
    STRICTLY Pure Google Gemini 3.8 Flash TTS (AI Male Voice: Algieba).
    All legacy/cheap TTS engines completely purged.
    """
    clean_text = text.strip()[:400]
    
    # Check if this exact text is already generated by Google Gemini TTS in local SQLite cache
    cached_audio = get_cached_audio(clean_text, voice_tag="gemini_algieba")
    if cached_audio:
        return Response(content=cached_audio[0], media_type=cached_audio[1])

    try:
        audio_bytes = call_google_gemini_tts(clean_text)
        save_cached_audio(clean_text, audio_bytes, "audio/wav", voice_tag="gemini_algieba")
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"[!] Google Gemini 3.8 Flash TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Google Gemini TTS error: {e}")


def bg_chapter_translation(task_id: str, req: BatchTranslateChapterRequest):
    tasks[task_id]["status"] = "in_progress"
    file_path = os.path.join(DOWNLOAD_DIR, req.manga_name, req.filename)
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            pages = sorted([n for n in zf.namelist() if not n.startswith("__MACOSX") and "." in n])
        
        total_p = len(pages)
        end_p = min(total_p, req.end_page) if req.end_page is not None else total_p
        start_p = max(0, req.start_page)

        tasks[task_id]["total_pages"] = end_p - start_p
        tasks[task_id]["completed_pages"] = 0

        for idx in range(start_p, end_p):
            tasks[task_id]["current_page"] = idx + 1
            tasks[task_id]["message"] = f"Страница {idx+1}/{end_p}: Распознавание и перевод..."
            
            # 1. Translate bubbles on this page
            page_bubbles = MangaTranslatorService.batch_translate_page(
                cbz_path=file_path,
                manga_name=req.manga_name,
                filename=req.filename,
                page_idx=idx
            )

            # 2. Pre-generate and cache ONLY Google Gemini 3.8 Flash TTS audio in background
            for b in page_bubbles:
                tr = b.get("translation") or {}
                orig_text = tr.get("original", "").strip()
                if orig_text and len(orig_text) > 1:
                    if not get_cached_audio(orig_text, voice_tag="gemini_algieba"):
                        try:
                            # Direct Google Gemini 3.8 Flash TTS generation
                            audio_bytes = call_google_gemini_tts(orig_text)
                            save_cached_audio(orig_text, audio_bytes, "audio/wav", voice_tag="gemini_algieba")
                            time.sleep(0.5) # respectful pace for Gemini free tier rate limits
                        except Exception as e:
                            print(f"[!] Background Gemini TTS skipped for '{orig_text[:20]}': {e}")

            tasks[task_id]["completed_pages"] += 1
            time.sleep(0.35)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = f"Все {end_p - start_p} страниц переведены и озвучены через Google Gemini 3.8 Flash TTS!"
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


@app.post("/api/batch-translate-chapter")
def batch_translate_chapter(req: BatchTranslateChapterRequest, background_tasks: BackgroundTasks):
    """
    Pre-translates an entire chapter in the background, saving all bubbles and Russian overlays into SQLite.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
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
    file_path = os.path.join(DOWNLOAD_DIR, manga_name, filename)
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
            targets.append(ch)

        tasks[task_id]["total_chapters"] = len(targets)
        tasks[task_id]["completed_chapters"] = 0

        for idx, ch in enumerate(targets):
            tasks[task_id]["message"] = f"Скачивание главы {ch['number']} ({idx+1}/{len(targets)})"
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
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
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
