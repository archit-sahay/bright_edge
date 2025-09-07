# BrightEdge Topic Extraction Engine (NLTK-based)

Extracts relevant topics from any URL by fetching the page, removing boilerplate, classifying page type, generating candidate phrases, and ranking them using in-house scoring (no third-party density libraries).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

## CLI

```bash
be-topics extract --url "https://www.amazon.com/..." --top-k 8 --timeout 8
```

Outputs JSON with `page_type` and `topics`.

## Development
- Python 3.9+
- Libraries: requests, bs4, lxml, nltk, tldextract, chardet

## Notes
- Honors robots.txt and basic preflight checks.
- JS-heavy pages are best-effort; optional headless fetch could be added later.

---

## Design document

### Requirements
- Classify any page and return relevant topics (no third‑party density analyzers).
- Use NLTK or basic NLP, avoid LLMs for core extraction.
- Handle busy pages and strip clutter/boilerplate.
- Provide good OOP design, reliability, performance, and scale.
- Include CLI and structured error handling.

### High-level approach
- Pipeline: Fetch → Parse/Sanitize → Boilerplate removal → Classification → Candidate generation (n‑grams) → Scoring/Ranking → Dedup → Output.
- Components: `UrlFetcher`, `HtmlParser/ContentExtractor`, `PageClassifier`, `CandidateGenerator`, `TopicScorer`, `Pipeline`.

### Fetching
- requests.Session with timeouts and retries; HEAD→GET fallback for 405/501.
- robots.txt preflight (politeness + compliance).
- Optional rendering via Playwright (`--render`) for JS-heavy pages.

### Parsing and boilerplate removal
- BeautifulSoup+lxml; drop `script/style/noscript/iframe/svg/link` and comments.
- Main-content heuristic (text length, paragraph count, link density) to prefer content blocks.
- Extract: title, meta/OG/Twitter, h1–h6, p, li, filtered `a`, button/input placeholders, image alt, JSON‑LD.
- Product extras: bullets (e.g., Amazon About this item), spec tables (key/value), highlighted text.

### Page classification (rule-based)
- Product signals: price/add-to-cart patterns, SKU/model cues.
- News/article signals: bylines/keywords; article if long lead paragraphs.
- Fallback to other.

### Candidate generation (SAN: Source-aware Sanitized N‑gram)
- Regex tokenization (no punkt dependency); minimal fallback stopwords if NLTK data missing.
- Preprocessing: punctuation removal, stopwords/pronoun/verb filtering, number handling, unit normalization (in, lb, watts, volts; cm/mm/ft), dimension merging.
- N‑grams (1–3) from sources: title, headings, body, bullets, specs, alts, lists, anchors, URL tokens, JSON‑LD; optional CSS-derived topics via `--css-topics`.
- Domain noise filters: e‑commerce UI terms; Wikipedia TOC/citation cleanup.
- Spec parsing keeps values and drops labels to prefer meaningful phrases.

Pros: fast, deterministic, explainable, tuned for products and articles.  
Cons: heuristic (limited semantics/synonyms), requires per-domain tuning for best results.

### Scoring (SW‑TF: Source‑Weighted TF with product signals)
- Score = TF × (1 + source boosts) × n‑gram length boost (favor 2–3 words).
- Extra multipliers for model-like patterns and unit-bearing phrases.
- Diversification: normalized Jaccard similarity; subset suppression; title shingle suppression to reduce repeats.

Pros: simple, interpretable, tunable by source; promotes spec/title phrases.  
Cons: page-local (no global IDF), residual noise can rise if repeated; no deep semantics.

### Error handling
- Structured errors: invalid URL, robots disallow, HTTP errors (403/429/5xx), timeouts, render failures.
- Clear messages and status codes in the JSON output.

### Performance & scale
- Session pooling; lxml parsing; minimal allocations.
- Optional render path only when requested (`--render`).
- Planned (future): async batching, caching, per-domain rate limits.

### Hurdles overcome
- NLTK SSL/corpus issues: switched to regex tokenization and minimal fallback stopwords.
- HEAD 405 handling: added automatic fallback to GET.
- JS/HTTP2 quirks and blocked pages: render fallback; domain-specific noise filtering (e‑com, wiki).
- Topic duplication: canonicalization + Jaccard dedup; title shingle suppression.
- Spec fragmentation: unit normalization and dimension merging to keep coherent phrases.

### Future enhancements
- Stronger dimension consolidation into single phrases (e.g., `6.5 x 11 x 7 in`).
- Collection-aware weighting (IDF) across crawls; per-domain scoring profiles.
- Async fetch/caching; ETag/Last‑Modified adherence; rate‑limit governance.
- Optional Web Unlocker integration as conditional fallback (API/proxy) with routing heuristics, observability, and budget caps.
- Lightweight entity recognition (brand/model/category) and multilingual support.

### CLI usage examples
```bash
# Product page
be-topics extract --url "https://www.amazon.com/..." --top-k 10 --timeout 12

# Article page (render if needed)
be-topics extract --url "https://en.wikipedia.org/wiki/Edward_Snowden" --top-k 10 --timeout 12

# Optional
be-topics extract --url "https://example.com" --render --css-topics
```

### Output
```json
{
  "url": "https://...",
  "page_type": "product|article|news|other",
  "topics": [
    { "text": "cuisinart 2-slice toaster", "score": 0.0123, "sources": {"title": 1} }
  ]
}
```

