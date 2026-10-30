"""
Universal Multi-Source Manga Bridge for Keiyoushi & Popular Providers.
Supports:
1. MangaDex (global catalogue, official REST API, multiple languages)
2. MangaKawaii (direct French scanlation / original reader search)
Extensible architecture with unified JSON interface.
"""

import urllib.request
import urllib.parse
import json
import re
from typing import List, Dict, Any, Optional

def search_mangadex(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Search manga across MangaDex API"""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.mangadex.org/manga?title={encoded}&limit={limit}&includes[]=cover_art"
        req = urllib.request.Request(url, headers={"User-Agent": "MangaKawaii-Mihon/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        results = []
        for item in data.get("data", []):
            m_id = item.get("id")
            attrs = item.get("attributes", {})
            title_dict = attrs.get("title", {})
            title = title_dict.get("en") or title_dict.get("ru") or (list(title_dict.values())[0] if title_dict else "Unknown")
            
            # Find cover art file
            cover_file = ""
            for rel in item.get("relationships", []):
                if rel.get("type") == "cover_art":
                    cover_file = rel.get("attributes", {}).get("fileName", "")
                    break
            
            cover_url = f"https://uploads.mangadex.org/covers/{m_id}/{cover_file}.256.jpg" if cover_file else ""
            desc = attrs.get("description", {}).get("ru") or attrs.get("description", {}).get("en") or ""
            
            results.append({
                "source": "MangaDex",
                "source_id": "mangadex",
                "id": m_id,
                "title": title,
                "cover_url": cover_url,
                "description": desc[:160] + ("..." if len(desc) > 160 else ""),
                "status": attrs.get("status", "unknown"),
                "year": attrs.get("year"),
                "url": f"https://mangadex.org/title/{m_id}"
            })
        return results
    except Exception as e:
        print(f"[MangaDex search error]: {e}")
        return []

def search_mangakawaii(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Search manga across MangaKawaii French catalog"""
    try:
        import requests
        from bs4 import BeautifulSoup
        encoded = urllib.parse.quote(query)
        url = f"https://www.mangakawaii.fr/recherche?q={encoded}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        seen = set()
        for a in soup.select('a[href*="/manga/"]'):
            href = a.get("href", "")
            if not href or href in seen:
                continue
            slug_match = re.search(r'/manga/([a-zA-Z0-9_\-]+)', href)
            if not slug_match:
                continue
            slug = slug_match.group(1)
            seen.add(href)
            
            # Extract title and image
            img = a.find("img")
            cover = img.get("src") or img.get("data-src") if img else ""
            raw_text = a.get_text(separator=" ", strip=True)
            # Remove leading "Manga" label if present
            clean_title = re.sub(r'^Manga\s*', '', raw_text)
            
            results.append({
                "source": "MangaKawaii",
                "source_id": "mangakawaii",
                "id": slug,
                "title": clean_title or slug.replace("-", " ").title(),
                "cover_url": cover,
                "description": "Французский сканлейт & оригиналы",
                "status": "ongoing",
                "year": None,
                "url": href if href.startswith("http") else f"https://www.mangakawaii.fr{href}"
            })
            if len(results) >= limit:
                break
        return results
    except Exception as e:
        print(f"[MangaKawaii search error]: {e}")
        return []

def multi_source_search(query: str, enabled_sources: List[str] = None) -> List[Dict[str, Any]]:
    """Search query across all currently enabled manga sources"""
    if not query or not query.strip():
        return []
    
    q = query.strip()
    sources = enabled_sources or ["mangadex", "mangakawaii"]
    combined = []
    
    if "mangadex" in sources:
        combined.extend(search_mangadex(q))
    if "mangakawaii" in sources:
        combined.extend(search_mangakawaii(q))
        
    return combined

if __name__ == "__main__":
    res = multi_source_search("Kaze to Ki no Uta")
    print(f"Multi-search returned {len(res)} results:")
    for r in res[:6]:
        print(f"[{r['source']}] {r['title']} -> {r['cover_url']}")
