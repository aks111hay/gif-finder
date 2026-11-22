# GIF & Meme Finder

Search GIFs by meaning (English or Hinglish) or by uploading a screenshot.
Candidates come live from DuckDuckGo (`ddgs`, free) and are re-ranked by
Google's `gemini-embedding-2` (768-dim, L2-normalized, cosine similarity) via the
`google-genai` SDK. FastAPI backend, plain HTML/CSS/JS frontend, SQLite cache.
No GIFs are stored in the repo.

```
gif-finder/
├── app.py               # FastAPI server (binds 0.0.0.0, reads PORT)
├── finder.py            # DuckDuckGo discovery, safe download, embedding, ranking
├── cache_db.py          # SQLite cache for embeddings + ranked results
├── warm_cache.py        # one-off: precompute the 5 preset queries
├── static/index.html    # the whole UI (no framework, no build step)
├── cache/preset_results.json   # made by warm_cache.py: ranked results for the 5 presets
├── cache/preset_media/         # made by warm_cache.py: the preset images (~1 MB each max)
├── requirements.txt
└── README.md
```

## How a search works

1. The query (text → `RETRIEVAL_QUERY`, or an uploaded image) is embedded.
2. DuckDuckGo returns up to 12 candidate GIF URLs (`MAX_CANDIDATES`).
3. Each candidate is downloaded with timeouts and a 4 MB cap. Three keyframes
   (start / middle / end) are extracted with Pillow, embedded as
   `RETRIEVAL_DOCUMENT`, averaged and L2-normalized. Raw bytes are discarded
   right after.
4. Candidates are ranked by cosine similarity. The top 6 are shown with
   `Match: 84.2%`, filename and a link to the source page. Images are served
   from the app's own `/media` route, so hotlink protection can't break the page.

Broken, slow, oversized or hotlink-blocked URLs are skipped and backfilled from
the next-best candidates. HTTP 429s are retried with backoff, then reported as
a friendly error.

## Caching

| Layer | Stores | Notes |
|---|---|---|
| `cache/preset_results.json` + `cache/preset_media/` | Ranked results and images for the 5 preset chips | Committed to git |
| SQLite (`gif_finder_cache.db` in the temp dir) | Query vectors, candidate vectors, ranked results (6 h) | Vectors live on disk, not RAM |
| Temp image folder | Downloaded images, auto-pruned to 150 MB | Served at `/media/...` |

A preset chip makes zero Gemini and DuckDuckGo calls. A fresh live search costs
about `candidates × 3` embedding calls (about 36 by default); lower
`MAX_CANDIDATES` if your key hits the free-tier limit. On Render's free tier the
disk is wiped on every restart or spin-down, so the SQLite cache only helps
while the instance is awake. The committed preset file is what survives.

## Why production can look different from localhost

DuckDuckGo treats a home IP and Render's datacenter IP differently, so the same
live search can return good GIFs locally and off-topic images on Render. The
ranker (`gemini-embedding-2`) can only order what it is given. Three safeguards
are built in:

1. **Presets are baked, not searched.** `warm_cache.py` runs the full pipeline
   on your machine and saves the results *and the images* in `cache/`. Commit
   them and the 5 preset chips never touch DuckDuckGo, Gemini or a hotlinked
   image in production. After deploying, open `/healthz` and check that
   `presets_loaded` is `5`. If it is `0`, `cache/` was not committed.
2. **Better candidates for live searches.** The backend is pinned (DuckDuckGo
   GIF filter first, Bing only as a fallback), the pool is 3× larger, and
   candidates are pre-sorted by word overlap with the query and GIF-looking
   URLs before any Gemini call is spent.
3. **Junk is flagged and never cached.** If the best score is below
   `MIN_CONFIDENCE` (default 0.30), the response carries a low-confidence
   warning and the result is not stored. Calibrate this number from your own
   local scores for good vs bad searches.

If live search on Render is still poor, set `DDGS_PROXY` to a proxy URL (read
automatically by `ddgs`) or run the app somewhere with a residential-style IP.

## 1. Run locally

Get a free key at https://aistudio.google.com/apikey, then:

```bash
cd gif-finder
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key-here"   # PowerShell: $env:GEMINI_API_KEY="your-key-here"

python warm_cache.py     # run this on your own machine; read the scores it prints
python app.py            # open http://localhost:8000
```

## API

Interactive docs live at `/docs`. A visitor's own key can be sent in the
`X-Gemini-Key` header; without it the server's `GEMINI_API_KEY` is used.

```bash
curl http://localhost:8000/api/presets

curl -X POST http://localhost:8000/api/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "awkward silence"}'

curl -X POST http://localhost:8000/api/search/image \
  -H "X-Gemini-Key: YOUR_KEY" \
  -F "file=@screenshot.png" -F "hint=confused math"
```

Response:

```json
{
  "status": "Live search: 12 candidates ... ranked by gemini-embedding-2.",
  "results": [
    {"src": "/media/ab12cd34.gif", "score": 61.4, "domain": "giphy.com",
     "filename": "awkward.gif", "source": "https://giphy.com/..."}
  ]
}
```

Errors return `{"detail": "..."}` with 400 (bad input), 401 (missing or bad key),
429 (quota or rate limit), 502/503 (no usable candidates, search busy).

## 2. Deploy free on Render (Web Service)

1. Run `python warm_cache.py` locally and check its output: every preset should show
   sensible results. Use `--no-bundle` if you do not want image files in the repo
   (those presets then hotlink, which can fail from Render).
2. Push to GitHub (the repo stays tiny):

```bash
cd gif-finder
git init
git add app.py finder.py cache_db.py warm_cache.py requirements.txt README.md static/ cache/
git status --short cache/      # must list preset_results.json (and preset_media/)
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

3. On https://render.com click **New → Web Service** and connect the repo.
4. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app.py`
   - **Instance type:** Free
5. Under **Environment** add:
   - `GEMINI_API_KEY` = your key
   - `PYTHON_VERSION` = `3.12.3`
6. Click **Create Web Service**. Your app appears at `https://<name>.onrender.com`.
7. Visit `https://<name>.onrender.com/healthz` and confirm `"presets_loaded": 5`.

Render's free instance spins down after about 15 minutes without traffic and
takes about a minute to wake, so open the URL before a live demo.

## Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | none | Shared key (visitors can paste their own) |
| `MAX_CANDIDATES` | 12 | Candidates embedded per live search |
| `MAX_DOWNLOAD_MB` | 4 | Per-image download cap |
| `SEARCH_BUDGET_SEC` | 50 | Max seconds per live search |
| `WORKERS` | 3 | Parallel candidate workers (keep low for 512 MB) |
| `MAX_CONCURRENT_SEARCHES` | 2 | Searches running at once |
| `RATE_LIMIT_PER_MIN` | 12 | Searches per IP per minute on the shared key |
| `REQUEST_DELAY` | 0.3 | Pause after each Gemini call (seconds) |
| `MIN_CONFIDENCE` | 0.30 | Best score below this = low-confidence warning, result not cached |
| `DDG_REGION` | `in-en` | DuckDuckGo region (e.g. `us-en`, `wt-wt`) |
| `DDGS_PROXY` | none | Proxy for DuckDuckGo requests (read by `ddgs`) |
| `CAPTION_MODEL` | `gemini-2.5-flash` | Guesses a search phrase for uploaded images |
| `CACHE_DB` | temp dir | Path of the SQLite file |

## Notes

- **DuckDuckGo is unofficial.** `ddgs` scrapes it, so it can be rate-limited or
  break. `pip install -U ddgs` usually fixes it.
- **Image rights.** Results are third-party images. The app temporarily caches
  them, shows source links and doesn't redistribute them.
- **Image search** needs words to find candidates: type a hint, or leave it empty
  and Gemini writes a short phrase. If that fails, a generic "funny reaction
  meme" search is used.
- Scores are raw cosine similarity ×100. Text-to-image scores run lower than
  image-to-image ones, so the ranking matters more than the percentage.
- If `task_type` or `output_dimensionality` is rejected for `gemini-embedding-2`,
  run `pip install -U google-genai` and check
  https://ai.google.dev/gemini-api/docs/embeddings
- Preset links die over time. Re-run `python warm_cache.py` and push to refresh.