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
    sources = enabled_sources or ["mangadex", "mangakawaii", "remanga"]
    combined = []
    
    if "mangadex" in sources:
        combined.extend(search_mangadex(q))
    if "mangakawaii" in sources:
        combined.extend(search_mangakawaii(q))
    if "remanga" in sources:
        combined.extend(search_remanga(q))
        
    return combined

if __name__ == "__main__":
    res = multi_source_search("Kaze to Ki no Uta")
    print(f"Multi-search returned {len(res)} results:")
    for r in res[:6]:
        print(f"[{r['source']}] {r['title']} -> {r['cover_url']}")

def search_remanga(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Search manga across Remanga API (Russian scanlation)"""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.remanga.org/api/search/?query={encoded}&count={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        results = []
        for item in data.get("content", []):
            dir_slug = item.get("dir")
            title = item.get("rus_name") or item.get("en_name") or dir_slug
            img = item.get("img", {}).get("high") or item.get("img", {}).get("mid") or ""
            cover = f"https://remanga.org{img}" if img.startswith("/") else img
            
            results.append({
                "source": "Remanga",
                "source_id": "remanga",
                "id": dir_slug,
                "title": title,
                "cover_url": cover,
                "description": f"Русский перевод · Оценка: {item.get('avg_rating', '—')} ⭐",
                "status": "ongoing",
                "year": None,
                "url": f"https://remanga.org/manga/{dir_slug}"
            })
        return results
    except Exception as e:
        print(f"[Remanga search error]: {e}")
        return []

def get_manga_details_and_chapters(source_id: str, manga_id: str) -> Dict[str, Any]:
    """
    Fetch comprehensive manga details & chapter list (Mihon details screen)
    """
    if source_id == "mangadex":
        try:
            url = f"https://api.mangadex.org/manga/{manga_id}?includes[]=cover_art&includes[]=author"
            req = urllib.request.Request(url, headers={"User-Agent": "MangaKawaiiApp/1.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                d = json.loads(r.read().decode())
            attrs = d["data"]["attributes"]
            title_dict = attrs.get("title", {})
            title = title_dict.get("ru") or title_dict.get("en") or (list(title_dict.values())[0] if title_dict else manga_id)
            desc = attrs.get("description", {}).get("ru") or attrs.get("description", {}).get("en") or ""
            
            cover_file = ""
            for rel in d["data"].get("relationships", []):
                if rel.get("type") == "cover_art":
                    cover_file = rel.get("attributes", {}).get("fileName", "")
                    break
            cover_url = f"https://uploads.mangadex.org/covers/{manga_id}/{cover_file}.512.jpg" if cover_file else ""
            
            # Fetch chapters list (Russian, English, French or all)
            ch_url = f"https://api.mangadex.org/manga/{manga_id}/feed?limit=96&order[chapter]=desc"
            with urllib.request.urlopen(urllib.request.Request(ch_url, headers={"User-Agent": "MangaKawaiiApp/1.0"}), timeout=12) as cr:
                cd = json.loads(cr.read().decode())
            
            chapters = []
            for ch in cd.get("data", []):
                ch_attrs = ch["attributes"]
                ch_num = ch_attrs.get("chapter") or "0"
                ch_name = ch_attrs.get("title") or f"Глава {ch_num}"
                lang = ch_attrs.get("translatedLanguage") or "all"
                chapters.append({
                    "id": ch["id"],
                    "number": ch_num,
                    "title": f"Гл. {ch_num} - {ch_name}" if ch_name != f"Глава {ch_num}" else f"Глава {ch_num}",
                    "lang": lang.upper(),
                    "date": (ch_attrs.get("publishAt") or "")[:10]
                })
            
            return {
                "source": "MangaDex",
                "source_id": "mangadex",
                "id": manga_id,
                "title": title,
                "cover_url": cover_url,
                "description": desc,
                "status": attrs.get("status", "ongoing"),
                "genres": [t.get("attributes", {}).get("name", {}).get("en", "") for t in attrs.get("tags", [])][:6],
                "chapters": chapters
            }
        except Exception as e:
            print(f"[MangaDex details error]: {e}")
            return {"error": str(e), "chapters": []}

    elif source_id == "remanga":
        try:
            url = f"https://api.remanga.org/api/titles/{manga_id}/"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
            content = data.get("content", {})
            title = content.get("rus_name") or content.get("en_name") or manga_id
            img = content.get("img", {}).get("high") or content.get("img", {}).get("mid") or ""
            cover_url = f"https://remanga.org{img}" if img.startswith("/") else img
            desc = content.get("description", "")
            
            chapters = []
            branches = content.get("branches", [])
            if branches:
                b_id = branches[0].get("id")
                ch_url = f"https://api.remanga.org/api/titles/chapters/?branch_id={b_id}&count=150"
                ch_req = urllib.request.Request(ch_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(ch_req, timeout=12) as cr:
                    ch_data = json.loads(cr.read().decode())
                
                for c in ch_data.get("content", []):
                    c_num = c.get("chapter", "0")
                    c_title = c.get("name") or f"Глава {c_num}"
                    chapters.append({
                        "id": str(c.get("id")),
                        "number": str(c_num),
                        "title": f"Гл. {c_num} - {c_title}" if c_title != f"Глава {c_num}" else f"Глава {c_num}",
                        "lang": "RU",
                        "date": (c.get("upload_date") or "")[:10]
                    })
            
            genres = [g.get("name") for g in content.get("genres", [])][:6]
            return {
                "source": "Remanga",
                "source_id": "remanga",
                "id": manga_id,
                "title": title,
                "cover_url": cover_url,
                "description": desc,
                "status": content.get("status", {}).get("name", "ongoing"),
                "genres": genres,
                "chapters": chapters
            }
        except Exception as e:
            print(f"[Remanga details error]: {e}")
            return {"error": str(e), "chapters": []}

    elif source_id == "mangakawaii":
        try:
            from mangakawaii_scraper import MangaKawaiiClient
            c = MangaKawaiiClient()
            details = c.get_manga_details(manga_id)
            chapters = []
            for ch in details.get("chapters", []):
                chapters.append({
                    "id": ch.get("number") or ch.get("name"),
                    "number": ch.get("number"),
                    "title": ch.get("name") or f"Chapitre {ch.get('number')}",
                    "lang": "FR",
                    "date": ""
                })
            return {
                "source": "MangaKawaii",
                "source_id": "mangakawaii",
                "id": manga_id,
                "title": details.get("title") or manga_id.replace("-", " ").title(),
                "cover_url": details.get("cover") or "",
                "description": details.get("description") or "Французский сканлейт & оригиналы",
                "status": "ongoing",
                "genres": ["Французский сканлейт"],
                "chapters": chapters
            }
        except Exception as e:
            print(f"[MangaKawaii details error]: {e}")
            return {"error": str(e), "chapters": []}

    return {"error": "Unsupported source", "chapters": []}
