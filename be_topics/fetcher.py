from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse
from urllib import robotparser

import chardet
import requests


DEFAULT_HEADERS = {
    "User-Agent": "BrightEdge-TopicExtractor/0.1 (+contact@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


@dataclass
class FetchResult:
    url: str
    status_code: int
    content_type: str
    text: Optional[str]
    error: Optional[str] = None


def is_fetch_allowed(url: str, user_agent: str) -> Tuple[bool, str]:
    parsed = urlparse(url)
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(user_agent, url)
    except Exception:
        # If robots cannot be read, default to allow but proceed politely
        allowed = True
    return allowed, robots_url


def _render_with_playwright(url: str, timeout: float) -> Tuple[str, str, int]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        raise RuntimeError("playwright not installed; run `pip install playwright` and `playwright install chromium`") from e

    with sync_playwright() as p:  # type: ignore
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], locale="en-US")
        page = context.new_page()
        resp = page.goto(url, wait_until="load", timeout=int(timeout * 1000))
        # Wait for network to be idle briefly
        try:
            page.wait_for_load_state("networkidle", timeout=int(timeout * 1000))
        except Exception:
            pass
        html = page.content()
        final_url = page.url
        status = resp.status if resp else 200
        context.close()
        browser.close()
        return final_url, html, status


def fetch_url(url: str, timeout: float = 8.0, max_retries: int = 2, respect_robots: bool = True, render: bool = False) -> FetchResult:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    if respect_robots:
        allowed, robots_url = is_fetch_allowed(url, DEFAULT_HEADERS["User-Agent"])
        if not allowed:
            return FetchResult(url=url, status_code=999, content_type="", text=None, error=f"Disallowed by robots: {robots_url}")

    if render:
        try:
            final_url, html, status = _render_with_playwright(url, timeout=timeout)
            return FetchResult(url=final_url, status_code=status, content_type="text/html", text=html)
        except Exception as e:
            return FetchResult(url=url, status_code=0, content_type="", text=None, error=f"render-failed: {e}")

    backoff = 0.5
    last_exc: Optional[Exception] = None
    try:
        for attempt in range(max_retries + 1):
            try:
                head = session.head(url, allow_redirects=True, timeout=timeout)
                if head.status_code in (401, 403, 429):
                    return FetchResult(url=url, status_code=head.status_code, content_type="", text=None, error="Access restricted")
                # Some sites don't support HEAD (405/501). Fallback to GET.
                if head.status_code >= 400 and head.status_code not in (405, 501):
                    return FetchResult(url=url, status_code=head.status_code, content_type="", text=None, error="HTTP error on HEAD")
                ctype = head.headers.get("Content-Type", "").lower()
                if head.status_code < 400:
                    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                        return FetchResult(url=url, status_code=head.status_code, content_type=ctype, text=None, error="Non-HTML content")

                resp = session.get(url, allow_redirects=True, timeout=timeout)
                ctype = resp.headers.get("Content-Type", "").lower()
                if resp.status_code >= 400:
                    return FetchResult(url=url, status_code=resp.status_code, content_type=ctype, text=None, error="HTTP error on GET")

                # Decode content
                raw = resp.content
                enc = resp.encoding or (chardet.detect(raw)["encoding"] if raw else None) or "utf-8"
                text = raw.decode(enc, errors="replace")
                return FetchResult(url=resp.url, status_code=resp.status_code, content_type=ctype, text=text)
            except requests.RequestException as e:
                last_exc = e
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise
    except Exception as e:  # type: ignore[no-redef]
        return FetchResult(url=url, status_code=0, content_type="", text=None, error=str(e if last_exc is None else last_exc))


