#!/usr/bin/env python3
"""Offline indexer: embeds every GIF in ./gifs/ with gemini-embedding-2.

For each GIF it extracts 3 evenly spaced keyframes (start / middle / end),
embeds each frame as RETRIEVAL_DOCUMENT, averages the vectors, L2-normalizes
the result and writes everything to embeddings_db.json. It also pre-embeds a
few demo queries (RETRIEVAL_QUERY) so the live demo works without API calls.

Usage:
    export GEMINI_API_KEY="your-key"
    python index_gifs.py            # incremental (skips already-indexed GIFs)
    python index_gifs.py --force    # re-embed everything
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types
from PIL import Image

MODEL = "gemini-embedding-2"
DIMENSIONS = 768
NUM_FRAMES = 3
GIF_DIR = Path("gifs")
DB_PATH = Path("embeddings_db.json")

SAMPLE_QUERIES = [
    "kejriwal karo na meme",
    "paisa hi paisa hoga",
    "awkward silence",
    "celebrating a big win",
    "confused math calculation",
]

REQUEST_DELAY = 1.0  # seconds between successful calls (free-tier friendly)
MAX_RETRIES = 6
RETRYABLE_CODES = {429, 500, 503}


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def embed(client: genai.Client, content, task_type: str) -> np.ndarray:
    """Embed one piece of content (str or types.Part) with retry on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=MODEL,
                contents=content,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=DIMENSIONS,
                ),
            )
            time.sleep(REQUEST_DELAY)
            return np.asarray(response.embeddings[0].values, dtype=np.float32)
        except errors.APIError as exc:
            if exc.code in RETRYABLE_CODES and attempt < MAX_RETRIES - 1:
                wait = min(60, 5 * 2**attempt)
                print(f"    HTTP {exc.code}; retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def extract_keyframes(path: Path, n: int = NUM_FRAMES) -> list[bytes]:
    """Return PNG bytes for n evenly spaced frames (start, middle, end)."""
    frames: list[bytes] = []
    with Image.open(path) as im:
        total = getattr(im, "n_frames", 1)
        indices = sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))
        for idx in indices:
            im.seek(idx)
            frame = im.convert("RGB")
            frame.thumbnail((512, 512))  # keep payloads small
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            frames.append(buf.getvalue())
    return frames


def load_db() -> dict:
    if DB_PATH.exists():
        try:
            db = json.loads(DB_PATH.read_text(encoding="utf-8"))
            if db.get("dimensions") == DIMENSIONS and db.get("model") == MODEL:
                return db
        except json.JSONDecodeError:
            pass
    return {"model": MODEL, "dimensions": DIMENSIONS, "gifs": {}, "cached_queries": {}}


def save_db(db: dict) -> None:
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(db), encoding="utf-8")
    tmp.replace(DB_PATH)  # atomic: a crash never leaves a corrupt file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-embed all GIFs")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set the GEMINI_API_KEY environment variable.", file=sys.stderr)
        return 1
    client = genai.Client(api_key=api_key)

    gif_paths = sorted(GIF_DIR.glob("*.gif"))
    if not gif_paths:
        print(f"Error: no .gif files found in {GIF_DIR}/", file=sys.stderr)
        return 1

    db = load_db()
    if args.force:
        db["gifs"] = {}

    for i, path in enumerate(gif_paths, 1):
        key = path.as_posix()
        if key in db["gifs"]:
            print(f"[{i}/{len(gif_paths)}] skip {key} (already indexed)")
            continue
        print(f"[{i}/{len(gif_paths)}] indexing {key}")
        try:
            vectors = [
                embed(client, types.Part.from_bytes(data=png, mime_type="image/png"),
                      "RETRIEVAL_DOCUMENT")
                for png in extract_keyframes(path)
            ]
        except Exception as exc:  # keep going; one bad GIF shouldn't kill the run
            print(f"    FAILED: {exc}", file=sys.stderr)
            continue
        db["gifs"][key] = l2_normalize(np.mean(vectors, axis=0)).tolist()
        save_db(db)  # checkpoint after every GIF

    print("Pre-embedding sample queries...")
    for query in SAMPLE_QUERIES:
        if query in db["cached_queries"] and not args.force:
            continue
        print(f"  {query!r}")
        vec = embed(client, query, "RETRIEVAL_QUERY")
        db["cached_queries"][query] = l2_normalize(vec).tolist()
        save_db(db)

    print(f"Done: {len(db['gifs'])} GIFs, {len(db['cached_queries'])} cached queries "
          f"-> {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
