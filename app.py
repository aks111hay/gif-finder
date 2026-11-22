"""Semantic GIF & Meme Finder: FastAPI backend + plain HTML frontend."""
import io
import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import finder

logging.basicConfig(level=logging.INFO)
BASE_DIR = Path(__file__).resolve().parent

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", 12))    # per IP, shared key only
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_SEARCHES", 2))

app = FastAPI(title="Semantic GIF & Meme Finder", docs_url="/docs", redoc_url=None)
app.mount("/media", StaticFiles(directory=str(finder.CACHE_DIR)), name="media")

_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


# ------------------------------------------------------------------- helpers
def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")


def _check_rate_limit(request: Request, user_key: str | None) -> None:
    """Protect the shared key. Visitors using their own key are not limited."""
    if (user_key or "").strip():
        return
    ip, now = _client_ip(request), time.time()
    with _hits_lock:
        window = _hits[ip]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT:
            raise HTTPException(429, "Too many searches from your IP. Wait a minute "
                                     "or paste your own Gemini API key.")
        window.append(now)
        if len(_hits) > 5000:  # keep the dict small
            for stale in [k for k, v in _hits.items() if not v]:
                del _hits[stale]


def _run(fn, *args):
    """Run a search with a concurrency cap and map errors to HTTP responses."""
    if not _slots.acquire(timeout=20):
        raise HTTPException(503, "Server is busy with other searches. Try again shortly.")
    try:
        matches, status = fn(*args)
        results = finder.build_gallery(matches)
    except finder.FinderError as exc:
        raise HTTPException(exc.status, str(exc))
    except finder.RateLimited:
        raise HTTPException(429, "Gemini quota is busy (HTTP 429). Wait a few "
                                 "seconds, try a preset, or paste your own API key.")
    except finder.EmbedFailed:
        raise HTTPException(502, "Gemini could not embed that input. Try something else.")
    finally:
        _slots.release()
    if not results:
        raise HTTPException(502, "Found matches but none of the image links could be "
                                 "loaded (expired or hotlink-blocked). Try again.")
    return {"status": status, "results": results}


# ----------------------------------------------------------------------- API
class TextQuery(BaseModel):
    query: str = Field(min_length=1, max_length=200)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/presets")
def presets():
    return {"queries": finder.PRESET_QUERIES}


@app.post("/api/search/text")
def search_text(body: TextQuery, request: Request,
                x_gemini_key: str | None = Header(default=None)):
    """Text / Hinglish search. Optional header `X-Gemini-Key` overrides the env key."""
    _check_rate_limit(request, x_gemini_key)
    return _run(finder.text_search, body.query, x_gemini_key)


@app.post("/api/search/image")
def search_image(request: Request, file: UploadFile = File(...),
                 hint: str = Form(default="", max_length=100),
                 x_gemini_key: str | None = Header(default=None)):
    """Reverse image search (multipart upload, 8 MB max)."""
    _check_rate_limit(request, x_gemini_key)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 8 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > finder.MAX_PIXELS:
            raise ValueError("too large")
        image.load()
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    finally:
        del data
    return _run(finder.image_search, image, hint, x_gemini_key)


# ------------------------------------------------------------------ frontend
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))