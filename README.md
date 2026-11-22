---
title: Semantic GIF & Meme Finder
emoji: 🎞️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# 🎞️ Multimodal Semantic GIF & Meme Finder

Search GIFs by meaning (English or Hinglish) or by uploading a screenshot, using
Google's `gemini-embedding-2` and the official `google-genai` SDK.

```
gif-finder/
├── gifs/                # your .gif files (use generic names: meme_01.gif ...)
├── index_gifs.py        # offline indexer -> embeddings_db.json
├── app.py               # Gradio web UI
├── requirements.txt
└── README.md
```

## 1. Run locally

Get a free API key at https://aistudio.google.com/apikey, then:

```bash
cd gif-finder
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put your GIFs in ./gifs/ (e.g. meme_01.gif, meme_02.gif, ...)

export GEMINI_API_KEY="your-key-here"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
python index_gifs.py                    # builds embeddings_db.json (re-run safely; it skips done GIFs)
python app.py                           # open http://localhost:7860
```

Useful flags: `python index_gifs.py --force` re-embeds every GIF.

The indexer sleeps ~1s between calls and retries with exponential backoff on
HTTP 429/500/503, and checkpoints after every GIF, so you can safely re-run it if
you hit the free-tier limit.

## 2. Deploy free on Hugging Face Spaces

1. Create an account at https://huggingface.co and a **write** token at
   https://huggingface.co/settings/tokens.
2. Create a new Space: https://huggingface.co/new-space → SDK **Gradio**, hardware
   **CPU basic (free)**, visibility Public.
3. Push the project (run `index_gifs.py` locally first so `embeddings_db.json` exists):

```bash
git lfs install
cd gif-finder
git init
git lfs track "*.gif"
git add .gitattributes
git add app.py index_gifs.py requirements.txt README.md embeddings_db.json gifs/
git commit -m "Initial commit"
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push --force space HEAD:main     # use your HF username and the write token as the password
```

4. In the Space, open **Settings → Variables and secrets → New secret**, name it
   `GEMINI_API_KEY` and paste your key. Restart the Space.

Visitors can also paste their own key in the "Use your own API key" accordion
if your shared quota runs out. The five example chips are served from the
pre-computed cache, so they never touch the API.

## Notes

- Scores shown are raw cosine similarity ×100. Text-to-image similarities are
  typically lower than image-to-image ones, so ranking matters more than the
  absolute percentage.
- If you change `DIMENSIONS` or the model, re-run `python index_gifs.py --force`.
- Keep GIFs under ~10 MB each for snappy loading on the free tier.
- If `task_type` or `output_dimensionality` is rejected for `gemini-embedding-2`,
  upgrade the SDK (`pip install -U google-genai`) and check the current embeddings
  docs: https://ai.google.dev/gemini-api/docs/embeddings
