"""
Keiyoushi Repository Index Parser & Catalog Manager.
Fetches https://raw.githubusercontent.com/keiyoushi/extensions/repo/index.pb (gzipped Protobuf),
extracts all 1,400+ extensions with their name, package, APK release URL, icon, and language.
Cached locally with lightweight TTL.
"""

import time
import gzip
import re
import urllib.request
from typing import List, Dict, Optional

INDEX_PB_URL = "https://raw.githubusercontent.com/keiyoushi/extensions/repo/index.pb"

_CACHE: Dict[str, any] = {
    "timestamp": 0,
    "extensions": []
}

def parse_keiyoushi_index() -> List[Dict[str, any]]:
    now = time.time()
    # Cache for 6 hours
    if _CACHE["extensions"] and (now - _CACHE["timestamp"] < 21600):
        return _CACHE["extensions"]

    req = urllib.request.Request(
        INDEX_PB_URL,
        headers={"User-Agent": "MangaKawaii-MihonBridge/1.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    
    decompressed = gzip.decompress(raw)
    
    # Extract extensions using pattern search across protobuf stream
    # Each entry contains: Name, Package (eu.kanade.tachiyomi.extension.<lang>.<id>), APK download URL, Icon URL
    results = []
    
    # Pattern to match package and adjacent fields
    pkg_pattern = re.compile(rb'eu\.kanade\.tachiyomi\.extension\.([a-zA-Z0-9_\-]+)\.([a-zA-Z0-9_\-]+)')
    
    for match in pkg_pattern.finditer(decompressed):
        lang = match.group(1).decode("utf-8", errors="ignore")
        ext_slug = match.group(2).decode("utf-8", errors="ignore")
        pkg_full = match.group(0).decode("utf-8", errors="ignore")
        
        # Scan backward from match for the extension Name
        start_scope = max(0, match.start() - 100)
        prefix_chunk = decompressed[start_scope:match.start()]
        
        # Look for the last string token before the package name
        # Protobuf string tag: \n<len><name>
        name = ext_slug.capitalize()
        name_match = re.findall(rb'[\x0a\x12]([\x01-\x50])([A-Za-z0-9\s\.\-\(\)\'\[\]]{2,50})', prefix_chunk)
        if name_match:
            # Last matched string token before the package name is usually the human name
            cand_name = name_match[-1][1].decode("utf-8", errors="ignore").strip()
            if cand_name and not cand_name.startswith("http") and cand_name != pkg_full:
                name = cand_name

        # Scan forward for APK download url and icon url
        end_scope = min(len(decompressed), match.end() + 350)
        suffix_chunk = decompressed[match.end():end_scope]
        
        apk_match = re.search(rb'https://github\.com/keiyoushi/extensions/releases/download/[^\x00-\x1f\s]+?\.apk', suffix_chunk)
        apk_url = apk_match.group(0).decode("utf-8", errors="ignore") if apk_match else ""
        
        icon_match = re.search(rb'https://cdn\.jsdelivr\.net/gh/keiyoushi/extensions@repo/icon/[^\x00-\x1f\s]+?\.png', suffix_chunk)
        icon_url = icon_match.group(0).decode("utf-8", errors="ignore") if icon_match else ""

        version_match = re.search(r'-v([0-9\.]+)\.apk', apk_url)
        version = version_match.group(1) if version_match else "1.0.0"

        results.append({
            "name": name,
            "pkg": pkg_full,
            "slug": ext_slug,
            "lang": lang,
            "version": version,
            "apk_url": apk_url,
            "icon_url": icon_url,
            "installed": False # Will be merged with enabled list
        })
        
    # Deduplicate by pkg
    unique_map = {}
    for r in results:
        unique_map[r["pkg"]] = r
    
    unique_list = sorted(list(unique_map.values()), key=lambda x: (x["lang"] not in ("all", "ru", "en", "fr"), x["lang"], x["name"]))
    _CACHE["timestamp"] = now
    _CACHE["extensions"] = unique_list
    return unique_list

if __name__ == "__main__":
    exts = parse_keiyoushi_index()
    print(f"Parsed {len(exts)} Keiyoushi extensions.")
    for e in exts[:10]:
        print(f"[{e['lang'].upper()}] {e['name']} ({e['version']}) -> {e['pkg']}")
