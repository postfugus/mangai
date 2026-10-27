import os
import re
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable
from curl_cffi import requests
from bs4 import BeautifulSoup


class MangaKawaiiClient:
    BASE_URL = "https://www.mangakawaii.fr"

    def __init__(self, impersonate: str = "chrome110", timeout: int = 25):
        self.impersonate = impersonate
        self.timeout = timeout
        self.session = requests.Session(impersonate=self.impersonate)

    def get_manga_details(self, slug: str) -> Dict:
        """
        Fetches manga metadata and all available chapters.
        """
        url = f"{self.BASE_URL}/manga/{slug}"
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch manga page: HTTP {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")

        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        desc_el = soup.select_one("[x-ref=desc]")
        description = desc_el.get_text(strip=True) if desc_el else ""

        cover_el = soup.select_one("img.shadow-2xl") or soup.select_one("meta[property='og:image']")
        cover_url = ""
        if cover_el:
            cover_url = cover_el.get("src") or cover_el.get("content") or ""

        # Find all chapters
        chapters = []
        for a in soup.select("a.ch-num"):
            href = a.get("href", "")
            ch_title = a.get_text(strip=True)
            m = re.search(r"/(\d+(?:\.\d+)?)$", href)
            ch_num = m.group(1) if m else ch_title.replace("Ch.", "").strip()
            chapters.append({
                "number": ch_num,
                "title": ch_title,
                "url": href if href.startswith("http") else f"{self.BASE_URL}{href}"
            })

        # Sort chapters numerically (1, 2, ..., 39)
        def parse_ch_num(x):
            try:
                return float(x["number"])
            except ValueError:
                return 999999.0

        chapters.sort(key=parse_ch_num)

        return {
            "slug": slug,
            "title": title,
            "description": description,
            "cover_url": cover_url,
            "chapters": chapters,
            "total_chapters": len(chapters)
        }

    def get_chapter_pages(self, chapter_url_or_slug: str, chapter_num: Optional[str] = None) -> List[str]:
        """
        Extracts clean image URLs for a specific chapter.
        """
        if chapter_num is not None:
            url = f"{self.BASE_URL}/manga/{chapter_url_or_slug}/{chapter_num}"
        else:
            url = chapter_url_or_slug

        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch chapter page: HTTP {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        state_el = soup.select_one("[x-data*=imgs]")
        if state_el and state_el.get("x-data"):
            raw_state = state_el["x-data"]
            m = re.search(r"JSON\.parse\((\x27|\")((?:(?!\1).|\\.)*)\1\)", raw_state)
            if m:
                inner = m.group(2).encode("utf-8").decode("unicode_escape")
                data = json.loads(inner)
                imgs = data.get("imgs", [])
                clean_imgs = [
                    img.replace(r"\/", "/")
                    for img in imgs
                    if "__mk_trap__" not in img
                ]
                if clean_imgs:
                    return clean_imgs

        # Fallback to direct img tags
        img_tags = soup.select("img[id^=pg-]")
        fallback_urls = []
        for img in img_tags:
            src = img.get("src") or img.get("data-src")
            if src and "__mk_trap__" not in src:
                fallback_urls.append(src if src.startswith("http") else f"{self.BASE_URL}{src}")

        if not fallback_urls:
            raise RuntimeError(f"Could not extract images from {url}")

        return fallback_urls

    def _download_single_image(self, index: int, img_url: str, referer: str, retries: int = 3) -> tuple:
        session = requests.Session(impersonate=self.impersonate)
        headers = {
            "Referer": referer,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
        }
        for attempt in range(retries):
            try:
                res = session.get(img_url, headers=headers, timeout=self.timeout)
                if res.status_code == 200 and len(res.content) > 0:
                    clean_url = img_url.split("?")[0]
                    ext = clean_url.split(".")[-1] if "." in clean_url else "jpg"
                    if len(ext) > 4:
                        ext = "jpg"
                    filename = f"{index:03d}.{ext}"
                    return index, filename, res.content
                time.sleep(1 + attempt)
            except Exception as e:
                time.sleep(1 + attempt)

        raise RuntimeError(f"Failed to download image: {img_url}")

    def download_chapter(
        self,
        chapter_info: Dict,
        output_dir: str,
        manga_title: str = "Manga",
        concurrency: int = 6,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """
        Downloads all pages of a chapter and packs them into a .cbz file.
        """
        os.makedirs(output_dir, exist_ok=True)
        ch_num = chapter_info["number"]
        cbz_name = f"{manga_title} - Ch.{ch_num.zfill(2) if ch_num.isdigit() else ch_num}.cbz"
        cbz_path = os.path.join(output_dir, cbz_name)

        # Check if already downloaded and valid
        if os.path.exists(cbz_path) and os.path.getsize(cbz_path) > 10000:
            try:
                with zipfile.ZipFile(cbz_path, "r") as test_zf:
                    if len(test_zf.namelist()) > 0:
                        if progress_cb:
                            progress_cb(1, 1, f"Already exists: {cbz_name}")
                        return cbz_path
            except zipfile.BadZipFile:
                os.remove(cbz_path)

        ch_url = chapter_info["url"]
        page_urls = self.get_chapter_pages(ch_url)
        total_pages = len(page_urls)

        results = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(self._download_single_image, i + 1, url, ch_url): i + 1
                for i, url in enumerate(page_urls)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx, filename, data = future.result()
                results[idx] = (filename, data)
                completed += 1
                if progress_cb:
                    progress_cb(completed, total_pages, f"Ch.{ch_num} ({completed}/{total_pages} pages)")

        temp_path = cbz_path + ".tmp"
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx in sorted(results.keys()):
                fname, img_bytes = results[idx]
                zf.writestr(fname, img_bytes)

        os.replace(temp_path, cbz_path)
        return cbz_path
