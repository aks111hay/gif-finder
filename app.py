"""Multimodal Semantic GIF & Meme Finder (Gradio + gemini-embedding-2)."""
import io
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from google import genai
from google.genai import errors, types
from PIL import Image

MODEL = "gemini-embedding-2"
DIMENSIONS = 768
TOP_K = 6
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "embeddings_db.json"

# ---------------------------------------------------------------- load index
if not DB_PATH.exists():
    raise SystemExit("embeddings_db.json not found. Run `python index_gifs.py` first.")

_db = json.loads(DB_PATH.read_text(encoding="utf-8"))
GIF_PATHS = list(_db["gifs"].keys())
GIF_MATRIX = np.asarray([_db["gifs"][p] for p in GIF_PATHS], dtype=np.float32)
CACHED = {k.strip().lower(): np.asarray(v, dtype=np.float32)
          for k, v in _db.get("cached_queries", {}).items()}
EXAMPLE_QUERIES = list(_db.get("cached_queries", {}).keys())

if len(GIF_PATHS) == 0:
    raise SystemExit("The index is empty. Add GIFs to ./gifs/ and re-run index_gifs.py.")


# ------------------------------------------------------------------- helpers
def get_client(user_key: str | None) -> genai.Client:
    key = (user_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise gr.Error("No Gemini API key available. Paste one under "
                       "'Use your own API key'.")
    return genai.Client(api_key=key)


def embed_query(client: genai.Client, content) -> np.ndarray:
    try:
        resp = client.models.embed_content(
            model=MODEL,
            contents=content,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=DIMENSIONS,
            ),
        )
    except errors.APIError as exc:
        if exc.code == 429:
            raise gr.Error("Shared Gemini quota is busy (HTTP 429). Wait a few "
                           "seconds, try a cached example, or paste your own API key.")
        raise gr.Error(f"Gemini API error ({exc.code}): {exc.message}")
    vec = np.asarray(resp.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def top_matches(qvec: np.ndarray) -> list[tuple[str, str]]:
    sims = GIF_MATRIX @ qvec  # both sides are unit vectors -> cosine similarity
    order = np.argsort(-sims)[:TOP_K]
    return [
        (str(BASE_DIR / GIF_PATHS[i]),
         f"Match: {max(float(sims[i]), 0.0) * 100:.1f}%  ·  {Path(GIF_PATHS[i]).name}")
        for i in order
    ]


# ------------------------------------------------------------------- actions
def search_text(query: str, user_key: str):
    query = (query or "").strip()
    if not query:
        raise gr.Error("Type something to search for.")
    cached = CACHED.get(query.lower())
    if cached is not None:
        return top_matches(cached), "⚡ Served from cache (no API call)."
    qvec = embed_query(get_client(user_key), query)
    return top_matches(qvec), "🔍 Embedded live with gemini-embedding-2."


def search_image(image: Image.Image | None, user_key: str):
    if image is None:
        raise gr.Error("Upload a screenshot or meme image first.")
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
    qvec = embed_query(get_client(user_key), part)
    return top_matches(qvec), "🖼️ Image embedded live with gemini-embedding-2."


# ------------------------------------------------------------------------ UI
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#chips {flex-wrap: wrap; gap: 8px;}
#chips button {flex: 0 1 auto; min-width: 0;}
@media (max-width: 640px) {
  #chips button {flex: 1 1 100%;}
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Semantic GIF Finder") as demo:
    gr.Markdown(
        "# 🎞️ Semantic GIF & Meme Finder\n"
        "Search by *meaning*, not filename. Powered by Google's "
        "`gemini-embedding-2` multimodal embeddings — it understands the visuals, "
        "on-screen text, and Hinglish. Every file is named generically "
        "(`meme_01.gif`), so matches come purely from the content."
    )

    with gr.Accordion("🔑 Use your own API key (optional)", open=False):
        gr.Markdown("If the shared free quota is busy, paste your own key from "
                    "[Google AI Studio](https://aistudio.google.com/apikey). "
                    "It is used only for your requests and never stored.")
        api_key_box = gr.Textbox(label="Gemini API key", type="password",
                                 placeholder="AIza...")

    with gr.Tabs():
        with gr.Tab("💬 Text / Hinglish Search"):
            with gr.Row():
                query_box = gr.Textbox(label="Describe the GIF", scale=4,
                                       placeholder="e.g. awkward silence")
                search_btn = gr.Button("Search", variant="primary", scale=1)
            with gr.Row(elem_id="chips"):
                chips = [gr.Button(q, size="sm") for q in EXAMPLE_QUERIES]
            text_status = gr.Markdown()
            text_gallery = gr.Gallery(label="Top matches", columns=3, rows=2,
                                      height="auto", object_fit="contain",
                                      show_label=False)

            search_btn.click(search_text, [query_box, api_key_box],
                             [text_gallery, text_status])
            query_box.submit(search_text, [query_box, api_key_box],
                             [text_gallery, text_status])
            for chip, q in zip(chips, EXAMPLE_QUERIES):
                chip.click(lambda q=q: q, None, query_box).then(
                    search_text, [query_box, api_key_box],
                    [text_gallery, text_status])

        with gr.Tab("🖼️ Reverse Image → GIF"):
            image_in = gr.Image(type="pil", label="Upload a screenshot or meme",
                                sources=["upload", "clipboard"])
            image_btn = gr.Button("Find similar GIFs", variant="primary")
            image_status = gr.Markdown()
            image_gallery = gr.Gallery(label="Top matches", columns=3, rows=2,
                                       height="auto", object_fit="contain",
                                       show_label=False)
            image_btn.click(search_image, [image_in, api_key_box],
                            [image_gallery, image_status])

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        allowed_paths=[str(BASE_DIR / "gifs")],
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
