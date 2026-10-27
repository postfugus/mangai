import os
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from mangakawaii_scraper import MangaKawaiiClient

app = FastAPI(
    title="MangaKawaii Downloader API",
    description="API for scraping and downloading manga from mangakawaii.fr into CBZ manga format.",
    version="1.0.0"
)

client = MangaKawaiiClient()
DEFAULT_DOWNLOAD_DIR = os.getenv("MANGA_DOWNLOAD_DIR", "/root/manga_downloads")

# In-memory download task tracker
tasks: Dict[str, Dict] = {}


class DownloadRequest(BaseModel):
    from_chapter: Optional[float] = None
    to_chapter: Optional[float] = None
    concurrency: int = 6
    output_dir: Optional[str] = None


def run_download_task(task_id: str, slug: str, req: DownloadRequest):
    out_dir = req.output_dir or os.path.join(DEFAULT_DOWNLOAD_DIR, slug)
    os.makedirs(out_dir, exist_ok=True)
    
    tasks[task_id]["status"] = "in_progress"
    try:
        manga = client.get_manga_details(slug)
        tasks[task_id]["manga_title"] = manga["title"]
        chapters = manga["chapters"]
        
        # Filter range if specified
        target_chapters = []
        for ch in chapters:
            try:
                num = float(ch["number"])
            except ValueError:
                num = 0.0
            
            if req.from_chapter is not None and num < req.from_chapter:
                continue
            if req.to_chapter is not None and num > req.to_chapter:
                continue
            target_chapters.append(ch)

        tasks[task_id]["total_chapters"] = len(target_chapters)
        tasks[task_id]["completed_chapters"] = 0
        tasks[task_id]["downloaded_files"] = []

        for idx, ch in enumerate(target_chapters):
            tasks[task_id]["current_chapter"] = ch["number"]
            tasks[task_id]["message"] = f"Downloading Chapter {ch['number']} ({idx+1}/{len(target_chapters)})"

            def on_progress(done, total, msg):
                tasks[task_id]["chapter_progress"] = {
                    "done": done,
                    "total": total,
                    "message": msg
                }

            cbz_path = client.download_chapter(
                chapter_info=ch,
                output_dir=out_dir,
                manga_title=manga["title"] or slug,
                concurrency=req.concurrency,
                progress_cb=on_progress
            )
            tasks[task_id]["completed_chapters"] += 1
            tasks[task_id]["downloaded_files"].append({
                "chapter": ch["number"],
                "path": cbz_path,
                "size_bytes": os.path.getsize(cbz_path) if os.path.exists(cbz_path) else 0
            })

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = f"All {len(target_chapters)} chapters downloaded successfully."

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"Error during download: {str(e)}"


@app.get("/")
def root():
    return {
        "service": "MangaKawaii Downloader API",
        "endpoints": {
            "GET /manga/{slug}": "Get manga info and chapters list",
            "GET /manga/{slug}/chapter/{number}/pages": "Get direct page image URLs for chapter",
            "POST /manga/{slug}/download": "Start background download to CBZ format",
            "GET /tasks/{task_id}": "Check download task progress",
            "GET /downloads": "List downloaded files"
        }
    }


@app.get("/manga/{slug}")
def get_manga(slug: str):
    try:
        return client.get_manga_details(slug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/manga/{slug}/chapter/{number}/pages")
def get_pages(slug: str, number: str):
    try:
        pages = client.get_chapter_pages(slug, number)
        return {
            "slug": slug,
            "chapter": number,
            "total_pages": len(pages),
            "pages": pages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/manga/{slug}/download")
def start_download(slug: str, req: DownloadRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "slug": slug,
        "status": "pending",
        "message": "Task queued",
        "created_at": time.time(),
        "total_chapters": 0,
        "completed_chapters": 0,
        "downloaded_files": []
    }
    background_tasks.add_task(run_download_task, task_id, slug, req)
    return {
        "task_id": task_id,
        "status": "queued",
        "check_url": f"/tasks/{task_id}"
    }


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/downloads")
def list_downloads(slug: Optional[str] = None):
    results = []
    base_dir = DEFAULT_DOWNLOAD_DIR
    if not os.path.exists(base_dir):
        return []

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".cbz"):
                full_path = os.path.join(root, f)
                if slug and slug not in full_path:
                    continue
                results.append({
                    "filename": f,
                    "path": full_path,
                    "size_mb": round(os.path.getsize(full_path) / (1024 * 1024), 2)
                })
    return sorted(results, key=lambda x: x["filename"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
