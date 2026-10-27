import argparse
import sys
import os
import time
from mangakawaii_scraper import MangaKawaiiClient


def main():
    parser = argparse.ArgumentParser(description="MangaKawaii Manga Downloader CLI (CBZ format)")
    parser.add_argument("--slug", type=str, default="kaze-to-ki-no-uta", help="Manga slug (e.g. kaze-to-ki-no-uta)")
    parser.add_argument("--output", type=str, default="/root/manga_downloads/Kaze_to_Ki_no_Uta", help="Output directory")
    parser.add_argument("--start", type=float, default=None, help="Start chapter number")
    parser.add_argument("--end", type=float, default=None, help="End chapter number")
    parser.add_argument("--concurrency", type=int, default=6, help="Download concurrency per chapter")

    args = parser.parse_args()

    print(f"[*] Initializing MangaKawaii client for '{args.slug}'...")
    client = MangaKawaiiClient()

    try:
        manga = client.get_manga_details(args.slug)
    except Exception as e:
        print(f"[!] Error fetching manga info: {e}")
        sys.exit(1)

    print(f"[+] Title: {manga['title']}")
    print(f"[+] Available chapters: {manga['total_chapters']}")
    print(f"[+] Destination: {args.output}")

    # Filter chapters
    target_chapters = []
    for ch in manga["chapters"]:
        try:
            num = float(ch["number"])
        except ValueError:
            num = 0.0

        if args.start is not None and num < args.start:
            continue
        if args.end is not None and num > args.end:
            continue
        target_chapters.append(ch)

    print(f"[+] Chapters to download: {len(target_chapters)}")
    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()
    for idx, ch in enumerate(target_chapters, 1):
        print(f"\n[{idx}/{len(target_chapters)}] Downloading Chapter {ch['number']}...")

        def on_progress(done, total, msg):
            pct = int((done / total) * 100) if total > 0 else 0
            bar = ("#" * (pct // 5)).ljust(20, ".")
            sys.stdout.write(f"\r    [{bar}] {pct}% ({done}/{total}) {msg}")
            sys.stdout.flush()

        try:
            cbz_path = client.download_chapter(
                chapter_info=ch,
                output_dir=args.output,
                manga_title=manga["title"] or args.slug,
                concurrency=args.concurrency,
                progress_cb=on_progress
            )
            size_mb = os.path.getsize(cbz_path) / (1024 * 1024)
            print(f"\n    Saved: {os.path.basename(cbz_path)} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"\n    [!] Failed to download chapter {ch['number']}: {e}")

    total_time = time.time() - start_time
    print(f"\n==========================================")
    print(f"[+] All requested chapters processed in {total_time:.1f}s.")
    print(f"[+] Files location: {args.output}")


if __name__ == "__main__":
    main()
