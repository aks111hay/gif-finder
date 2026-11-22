#!/usr/bin/env python3
"""Bake the 5 preset queries into the repo so production never depends on
DuckDuckGo for them.

Run this on your own machine (a home IP gets much better DuckDuckGo results
than Render's datacenter IP), check the output, then commit `cache/`.

It writes:
  cache/preset_results.json   ranked URLs + scores (a few KB)
  cache/preset_media/*        the top images, copied into the repo so Render
                              serves them with zero network calls
                              (skip with --no-bundle; files over --max-kb are
                              left as hotlinks)

Usage:
    export GEMINI_API_KEY="your-key"
    python warm_cache.py                 # refresh all presets
    python warm_cache.py --no-bundle     # URLs only, no image files
    python warm_cache.py --max-kb 800    # smaller bundled files
    python warm_cache.py --force         # overwrite even if the new result is worse
"""
import argparse
import json
import shutil
import sys
import time

import finder

MEDIA_DIR = finder.PRESET_PATH.parent / "preset_media"


def load_existing() -> dict:
    try:
        return json.loads(finder.PRESET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def bundle(matches: list[dict], max_bytes: int) -> int:
    """Copy up to TOP_K images into cache/preset_media and tag them with 'local'."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for match in matches:
        if count >= finder.TOP_K:
            break
        path = finder._ensure_local(match)
        if path is None or path.stat().st_size > max_bytes:
            continue
        shutil.copyfile(path, MEDIA_DIR / path.name)
        match["local"] = f"preset_media/{path.name}"
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-bundle", action="store_true")
    parser.add_argument("--max-kb", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        client = finder.get_client(None)
    except finder.FinderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    existing = load_existing().get("queries", {})
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model": finder.MODEL, "queries": {}}
    failed = []

    for query in finder.PRESET_QUERIES:
        print(f"\n→ {query!r}")
        try:
            qvec = finder.embed(client, query, "RETRIEVAL_QUERY")
            matches, status = finder.discover_and_rank(
                client, qvec, query, progress=lambda frac, desc: None)
        except (finder.FinderError, finder.RateLimited) as exc:
            print(f"   FAILED: {exc or 'rate limited'}", file=sys.stderr)
            failed.append(query)
            if query in existing:
                out["queries"][query] = existing[query]
                print("   kept the previous cached result")
            continue

        top = matches[0]["score"]
        print(f"   {status}")
        for i, m in enumerate(matches[:finder.TOP_K], 1):
            print(f"   {i}. {m['score'] * 100:5.1f}%  {m['url'][:90]}")

        if top < finder.MIN_CONFIDENCE and query in existing and not args.force:
            print(f"   best score {top * 100:.1f}% is below MIN_CONFIDENCE; "
                  "kept the previous result (use --force to overwrite)")
            out["queries"][query] = existing[query]
            failed.append(query)
            continue

        if not args.no_bundle:
            n = bundle(matches, args.max_kb * 1024)
            print(f"   bundled {n} image(s) into cache/preset_media/")
        out["queries"][query] = {"matches": matches}
        time.sleep(2)

    # drop bundled files that no preset references any more
    if MEDIA_DIR.exists():
        used = {m["local"].split("/", 1)[1] for q in out["queries"].values()
                for m in q["matches"] if m.get("local")}
        for f in MEDIA_DIR.iterdir():
            if f.is_file() and f.name not in used:
                f.unlink()

    finder.PRESET_PATH.parent.mkdir(exist_ok=True)
    finder.PRESET_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in MEDIA_DIR.glob("*")) / 1e6 if MEDIA_DIR.exists() else 0
    print(f"\nSaved {len(out['queries'])}/{len(finder.PRESET_QUERIES)} presets "
          f"-> {finder.PRESET_PATH}  (bundled media: {size_mb:.1f} MB)")
    if failed:
        print(f"Review these before deploying: {failed}", file=sys.stderr)
    return 1 if len(out["queries"]) < len(finder.PRESET_QUERIES) else 0


if __name__ == "__main__":
    sys.exit(main())