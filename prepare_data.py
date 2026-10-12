"""下载公开文本，并保留原文条款；下载失败时不伪造数据。"""
import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from runtime import ROOT, sha256, write_json

SOURCE_URL = "https://www.gutenberg.org/ebooks/11.txt.utf-8"


def extract_body(text):
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("未找到 Gutenberg 头尾标记；请核验来源，不自动猜测截断位置")
    return text[start.end():end.start()].strip() + "\n"


def prepare(directory=ROOT / "data"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "alice_original.txt"
    reused_cache = raw_path.exists()
    if reused_cache:
        raw = raw_path.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=45) as response:
            raw = response.read()
        # 先验证内容再保存，避免把网络错误页面当数据。
        extract_body(raw.decode("utf-8-sig"))
        raw_path.write_bytes(raw)
    body = extract_body(raw.decode("utf-8-sig").replace("\r\n", "\n"))
    (directory / "input.txt").write_text(body, encoding="utf-8", newline="\n")
    metadata_path = directory / "source.json"
    previous = None
    if metadata_path.exists():
        import json
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {"title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "ebook": 11, "source_url": SOURCE_URL, "source_page": "https://www.gutenberg.org/ebooks/11", "retrieved_utc": previous["retrieved_utc"] if reused_cache and previous and previous.get("raw_sha256") == sha256(raw_path) else datetime.now(timezone.utc).isoformat(), "copyright_source_statement": "Public domain in the USA.", "terms": "Complete original and Project Gutenberg terms are preserved in alice_original.txt. Check applicable local law before redistribution.", "transform": "UTF-8 decode; CRLF to LF; content between Gutenberg START/END markers; strip outer whitespace; append LF", "raw_sha256": sha256(raw_path), "training_sha256": sha256(directory / "input.txt"), "characters": len(body)}
    write_json(metadata_path, metadata)
    print(f"Prepared {len(body)} characters: {directory / 'input.txt'}", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data")
    prepare(parser.parse_args().directory)
