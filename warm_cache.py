#!/usr/bin/env python3
"""Pre-compute results for the 5 preset queries and save them to
cache/preset_results.json (a few KB: URLs, titles, scores - no images).

Commit that file so the preset chips load instantly on Render without calling
Gemini or DuckDuckGo. Re-run it any time to refresh the presets (link rot).

Usage:
    export GEMINI_API_KEY="your-key"
    python warm_cache.py
"""
import json
import sys
import time

import finder


def main() -> int:
    try:
        client = finder.get_client(None)
    except finder.FinderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model": finder.MODEL, "queries": {}}
    for query in finder.PRESET_QUERIES:
        print(f"→ {query!r}")
        try:
            qvec = finder.embed(client, query, "RETRIEVAL_QUERY")
            matches, status = finder.discover_and_rank(
                client, qvec, query,
                progress=lambda frac, desc: print(f"   {desc}"))
        except (finder.FinderError, finder.RateLimited) as exc:
            print(f"   skipped: {exc or 'rate limited'}", file=sys.stderr)
            continue
        out["queries"][query] = {"matches": matches}
        print(f"   {status}")
        time.sleep(2)

    finder.PRESET_PATH.parent.mkdir(exist_ok=True)
    finder.PRESET_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {len(out['queries'])}/{len(finder.PRESET_QUERIES)} presets "
          f"-> {finder.PRESET_PATH}")
    return 0 if out["queries"] else 1


if __name__ == "__main__":
    sys.exit(main())