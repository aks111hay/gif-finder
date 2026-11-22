"""Core logic: DuckDuckGo candidate discovery + gemini-embedding-2 re-ranking.

Flow for a search:
  1. Embed the query (text or image) as RETRIEVAL_QUERY.
  2. Fetch candidate GIF/meme URLs from DuckDuckGo (free, via `ddgs`).
  3. Download each candidate safely (timeouts, size cap, public hosts only),
     extract up to 3 keyframes with Pillow, embed them as RETRIEVAL_DOCUMENT,
     average + L2-normalize.
  4. Rank by cosine similarity and return the best matches.

No UI code lives here, so it can be reused by app.py and warm_cache.py.
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import logging
import os
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
from ddgs import DDGS
from google import genai
from google.genai import errors, types
from PIL import Image

from cache_db import CacheDB

log = logging.getLogger("gif_finder")

# ------------------------------------------------------------------ settings
MODEL = "gemini-embedding-2"
DIMENSIONS = 768
NUM_FRAMES = 3
TOP_K = 6

MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", 12))       # per live search
MAX_BYTES = int(float(os.environ.get("MAX_DOWNLOAD_MB", 4)) * 1024 * 1024)
SEARCH_BUDGET = float(os.environ.get("SEARCH_BUDGET_SEC", 50))   # max seconds per search
WORKERS = int(os.environ.get("WORKERS", 3))                      # parallel candidates
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", 0.3))      # pause after each API call
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "gemini-2.5-flash")

CONNECT_TIMEOUT, READ_TIMEOUT, TOTAL_DOWNLOAD_TIMEOUT = 4, 8, 15
MAX_PIXELS = 20_000_000          # reject decompression bombs
MAX_RETRIES = 4
RETRYABLE = {429, 500, 503}
DISK_CACHE_LIMIT = 150 * 1024 * 1024
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

BASE_DIR = Path(__file__).resolve().parent
PRESET_PATH = BASE_DIR / "cache" / "preset_results.json"
CACHE_DIR = Path(tempfile.gettempdir()) / "gif_finder_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRESET_QUERIES = [
    "kejriwal karo na meme",
    "paisa hi paisa hoga",
    "awkward silence",
    "celebrating a big win",
    "confused math calculation",
]
EXT = {"GIF": "gif", "PNG": "png", "JPEG": "jpg", "WEBP": "webp"}


# ---------------------------------------------------------------- exceptions
class FinderError(Exception):
    """User-facing error (bad key, no results, ...) with an HTTP status."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class RateLimited(Exception):
    """Gemini free-tier quota exhausted."""


class EmbedFailed(Exception):
    """One embedding call failed for a non-fatal reason (skip that candidate)."""


# -------------------------------------------------------------------- caches
db = CacheDB(dim=DIMENSIONS)  # SQLite: embeddings + ranked results
RESULT_TTL = 6 * 3600


def _load_presets() -> dict:
    try:
        data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        return {k.strip().lower(): v for k, v in data.get("queries", {}).items()}
    except (OSError, json.JSONDecodeError):
        return {}


PRESETS = _load_presets()


# ------------------------------------------------------------------- helpers
def l2(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def get_client(api_key: str | None) -> genai.Client:
    key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise FinderError("No Gemini API key available. Paste your own key "
                          "in the API key box.", 401)
    return genai.Client(api_key=key)


def embed(client: genai.Client, content, task_type: str) -> np.ndarray:
    """One embedding call with retry/backoff. Returns a normalized vector."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.embed_content(
                model=MODEL,
                contents=content,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=DIMENSIONS),
            )
            time.sleep(REQUEST_DELAY)
            return l2(np.asarray(resp.embeddings[0].values, dtype=np.float32))
        except errors.APIError as exc:
            msg = (getattr(exc, "message", "") or "").lower()
            if exc.code in (401, 403) or "api key" in msg:
                raise FinderError("Gemini rejected the API key. Check "
                                  "GEMINI_API_KEY or paste a valid key.", 401) from exc
            if exc.code in RETRYABLE and attempt < MAX_RETRIES - 1:
                log.warning("Gemini HTTP %s, retrying in %.0fs", exc.code, delay)
                time.sleep(delay)
                delay = min(delay * 2, 20)
                continue
            if exc.code == 429:
                raise RateLimited() from exc
            raise EmbedFailed(str(exc)) from exc
        except Exception as exc:  # network blips etc.
            raise EmbedFailed(str(exc)) from exc
    raise RateLimited()


def _is_public_host(host: str) -> bool:
    """Block private / loopback / link-local targets (basic SSRF guard)."""
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return False
        return True
    except (OSError, ValueError):
        return False


def download(url: str, referer: str | None = None) -> bytes | None:
    """Fetch an image with timeouts and a hard size cap. None on any problem."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    if not _is_public_host(parsed.hostname):
        return None
    headers = {"User-Agent": USER_AGENT,
               "Accept": "image/gif,image/*;q=0.8,*/*;q=0.5"}
    if referer:
        headers["Referer"] = referer
    started = time.monotonic()
    try:
        with requests.get(url, headers=headers, stream=True,
                          timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
            if resp.status_code != 200:
                return None
            final_host = urlparse(resp.url).hostname
            if final_host and not _is_public_host(final_host):
                return None
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype and not (ctype.startswith("image/") or "octet-stream" in ctype):
                return None
            length = resp.headers.get("Content-Length", "")
            if length.isdigit() and int(length) > MAX_BYTES:
                return None
            buf = bytearray()
            for chunk in resp.iter_content(65536):
                buf.extend(chunk)
                if len(buf) > MAX_BYTES or time.monotonic() - started > TOTAL_DOWNLOAD_TIMEOUT:
                    return None
            return bytes(buf) if buf else None
    except requests.RequestException:
        return None


def extract_frames(data: bytes, n: int = NUM_FRAMES):
    """Return (list of PNG bytes for n evenly spaced frames, format) or None."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in EXT or im.width * im.height > MAX_PIXELS:
                return None
            fmt = im.format
            total = getattr(im, "n_frames", 1)
            idxs = (sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))
                    if total > 1 else [0])
            frames = []
            for idx in idxs:
                im.seek(idx)
                frame = im.convert("RGB")
                frame.thumbnail((384, 384))
                buf = io.BytesIO()
                frame.save(buf, format="PNG")
                frames.append(buf.getvalue())
            return frames, fmt
    except Exception:
        return None


def image_to_png(image: Image.Image, max_side: int = 768) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def _local_path(url: str) -> Path | None:
    return next(CACHE_DIR.glob(_url_hash(url) + ".*"), None)


def _save(url: str, data: bytes, fmt: str) -> Path:
    path = CACHE_DIR / f"{_url_hash(url)}.{EXT[fmt]}"
    path.write_bytes(data)
    return path


def prune_disk_cache() -> None:
    try:
        files = sorted((p for p in CACHE_DIR.iterdir() if p.is_file()),
                       key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while files and total > DISK_CACHE_LIMIT:
            victim = files.pop(0)
            total -= victim.stat().st_size
            victim.unlink(missing_ok=True)
    except OSError:
        pass


# ----------------------------------------------------------------- discovery
def search_candidates(query: str, n: int = MAX_CANDIDATES) -> list[dict]:
    """Free candidate fetch from DuckDuckGo image search (animated GIFs first)."""
    seen: set[str] = set()
    out: list[dict] = []
    for q, kind in ((f"{query} gif", "gif"), (f"{query} meme gif", None)):
        try:
            rows = DDGS(timeout=10).images(
                q, max_results=n * 2, type_image=kind, safesearch="moderate")
        except Exception as exc:  # rate limits, network errors, parser changes
            log.warning("DuckDuckGo search failed: %s", exc)
            rows = []
        for row in rows or []:
            url = row.get("image")
            if url and url not in seen:
                seen.add(url)
                out.append({"image": url, "page": row.get("url"),
                            "title": row.get("title") or ""})
        if len(out) >= n:
            break
    return out[:n]


def process_candidate(client: genai.Client, cand: dict) -> dict | None:
    """Download -> keyframes -> RETRIEVAL_DOCUMENT embeddings. None if unusable."""
    url = cand["image"]
    vec = db.get_vec("doc:" + url)
    if vec is None:
        data = download(url, cand.get("page"))
        if data is None:
            return None
        extracted = extract_frames(data)
        if extracted is None:
            return None
        frames, fmt = extracted
        try:
            _save(url, data, fmt)
        except OSError:
            pass
        del data  # raw bytes are no longer needed; free them right away
        try:
            vecs = [embed(client, types.Part.from_bytes(data=png, mime_type="image/png"),
                          "RETRIEVAL_DOCUMENT") for png in frames]
        except EmbedFailed:
            return None
        finally:
            del frames
        vec = l2(np.mean(vecs, axis=0))
        db.set_vec("doc:" + url, vec)
    return {"url": url, "page": cand.get("page"), "title": cand.get("title", ""),
            "vec": vec}


def discover_and_rank(client, qvec: np.ndarray, search_query: str, progress=None):
    """Fetch candidates, embed them, rank by cosine. Returns (matches, status)."""
    cb = progress or (lambda frac, desc: None)
    cb(0.05, "Searching DuckDuckGo…")
    cands = search_candidates(search_query)
    if not cands:
        raise FinderError("DuckDuckGo returned no candidates (it may be "
                          "rate-limiting). Wait a minute and try again.", 503)

    results, done, rate_limited, timed_out = [], 0, False, False
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    futures = [pool.submit(process_candidate, client, c) for c in cands]
    try:
        for fut in as_completed(futures, timeout=SEARCH_BUDGET):
            done += 1
            try:
                item = fut.result()
            except RateLimited:
                rate_limited = True
                break
            except FinderError:
                raise
            except Exception as exc:
                log.warning("candidate failed: %s", exc)
                item = None
            if item:
                results.append(item)
            cb(0.1 + 0.85 * done / len(futures), f"Embedded {done}/{len(futures)} candidates…")
    except FuturesTimeout:
        timed_out = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if not results:
        raise FinderError(
            "Gemini quota is busy (HTTP 429). Wait a bit, try a preset chip, or "
            "paste your own API key." if rate_limited else
            "None of the candidate images could be downloaded. Try rephrasing.",
            429 if rate_limited else 502)

    ranked = sorted(
        ({"url": r["url"], "page": r["page"], "title": r["title"],
          "score": float(r["vec"] @ qvec)} for r in results),
        key=lambda m: m["score"], reverse=True)

    status = (f"🌐 Live search: {len(cands)} candidates from DuckDuckGo → "
              f"{len(results)} usable, ranked by gemini-embedding-2.")
    if rate_limited:
        status += " ⚠️ Gemini quota ran out partway, so some candidates were skipped."
    if timed_out:
        status += " ⏱️ Time budget reached; ranked what was ready."
    return ranked[:TOP_K * 2], status


# ----------------------------------------------------------------- public API
def embed_query_text(client, query: str) -> np.ndarray:
    key = query.lower()
    vec = db.get_vec("query:" + key)
    if vec is None:
        vec = embed(client, query, "RETRIEVAL_QUERY")
        db.set_vec("query:" + key, vec)
    return vec


def text_search(query: str, api_key: str | None = None, progress=None):
    """Hinglish/English text search. Returns (matches, status)."""
    query = " ".join((query or "").split())
    if not query:
        raise FinderError("Type something to search for.")
    key = query.lower()
    if key in PRESETS:
        return PRESETS[key]["matches"], "⚡ Preset cache (no Gemini or DuckDuckGo calls)."
    cached = db.get_results(key, RESULT_TTL)
    if cached:
        return cached, "⚡ Cached from an earlier search (no API calls)."
    client = get_client(api_key)
    qvec = embed_query_text(client, query)
    try:
        matches, status = discover_and_rank(client, qvec, query, progress)
    except RateLimited:
        raise FinderError("Gemini quota is busy (HTTP 429). Wait a few seconds, "
                          "try a preset chip, or paste your own API key.", 429)
    db.set_results(key, matches)
    return matches, status


def caption_image(client, png: bytes) -> str:
    """Short web-search phrase describing an image (best effort)."""
    try:
        resp = client.models.generate_content(
            model=CAPTION_MODEL,
            contents=[types.Part.from_bytes(data=png, mime_type="image/png"),
                      "Write a 3-6 word web search phrase that would find GIFs "
                      "similar in meaning to this image (include any visible "
                      "text or character names). Output only the phrase."])
        return " ".join((resp.text or "").split())[:80]
    except Exception as exc:
        log.warning("captioning failed: %s", exc)
        return ""


def image_search(image: Image.Image, hint: str = "", api_key: str | None = None,
                 progress=None):
    """Reverse image -> GIF search. Returns (matches, status)."""
    client = get_client(api_key)
    png = image_to_png(image)
    qvec = embed(client, types.Part.from_bytes(data=png, mime_type="image/png"),
                 "RETRIEVAL_QUERY")
    hint = " ".join((hint or "").split())
    search_query = hint or caption_image(client, png) or "funny reaction meme"
    del png
    try:
        matches, status = discover_and_rank(client, qvec, search_query, progress)
    except RateLimited:
        raise FinderError("Gemini quota is busy (HTTP 429). Wait a few seconds "
                          "or paste your own API key.", 429)
    return matches, f'{status} Search phrase used: "{search_query}".'


# --------------------------------------------------------------- presentation
def _domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host or "unknown"


def _filename(url: str) -> str:
    name = unquote(Path(urlparse(url).path).name) or _url_hash(url)
    return name if len(name) <= 40 else name[:37] + "..."


def _ensure_local(match: dict) -> Path | None:
    path = _local_path(match["url"])
    if path:
        return path
    data = download(match["url"], match.get("page"))
    if data is None:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, too_big = im.format, im.width * im.height > MAX_PIXELS
        if fmt not in EXT or too_big:
            return None
        return _save(match["url"], data, fmt)
    except Exception:
        return None
    finally:
        del data


def build_gallery(matches: list[dict], k: int = TOP_K) -> list[dict]:
    """Rank -> JSON-ready cards. Images are served from our own /media route.

    Dead links are skipped and backfilled from the extra ranked candidates.
    """
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(_ensure_local, matches[:k * 2]))
    items = []
    for match, path in zip(matches, paths):
        if path is None or len(items) >= k:
            continue
        source = match.get("page") or match["url"]
        items.append({
            "src": f"/media/{path.name}",
            "score": round(max(match["score"], 0.0) * 100, 1),
            "domain": _domain(source),
            "filename": _filename(match["url"]),
            "source": source,
        })
    prune_disk_cache()
    return items