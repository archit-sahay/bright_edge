from __future__ import annotations

from typing import Any, Dict

from .fetcher import fetch_url
from .parser import parse_content
from .classifier import classify_page, PageType
from .candidates import generate_candidates
from .scoring import score_candidates, diversify


def extract_topics(url: str, top_k: int = 8, timeout: float = 8.0, respect_robots: bool = True, render: bool = False, include_css_topics: bool = False) -> Dict[str, Any]:
    fetch = fetch_url(url, timeout=timeout, respect_robots=respect_robots, render=render)
    if fetch.error or not fetch.text:
        return {
            "url": url,
            "error": fetch.error or "fetch-failed",
            "status_code": fetch.status_code,
            "topics": [],
        }

    content = parse_content(fetch.text)
    page_type = classify_page(content)
    candidates = generate_candidates(content, url=fetch.url, include_css_topics=include_css_topics)
    scored = score_candidates(candidates)
    diversified = diversify(scored)
    top = diversified[:top_k]

    return {
        "url": fetch.url,
        "page_type": page_type.value,
        "topics": [
            {"text": t.text, "score": round(float(t.score), 4), "sources": t.sources}
            for t in top
        ],
    }


