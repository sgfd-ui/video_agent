"""yt-dlp based downloader/search helpers."""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL


def search_videos(keyword: str, limit: int = 5) -> list[dict]:
    query = f"ytsearch{limit}:{keyword}"
    ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(query, download=False)

    entries = result.get("entries", []) if isinstance(result, dict) else []
    rows = []
    for item in entries:
        if not item:
            continue
        rows.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url") or item.get("webpage_url", ""),
                "duration": item.get("duration") or 0,
                "cover": item.get("thumbnail") or "",
            }
        )
    return [r for r in rows if r["url"]]


def download_video(url: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_template = str(Path(output_dir) / "%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "quiet": True,
        "format": "mp4/bestvideo+bestaudio/best",
        "noplaylist": True,
        "merge_output_format": "mp4",
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)

    file = Path(file_path)
    if file.suffix != ".mp4":
        mp4 = file.with_suffix(".mp4")
        if mp4.exists():
            return str(mp4)
    return str(file)
