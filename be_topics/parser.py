from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import itertools

from bs4 import BeautifulSoup, Comment


MAIN_TAGS = {"article", "main", "section", "div"}


@dataclass
class PageContent:
    title: str
    meta_description: str
    og_title: str
    og_description: str
    tw_title: str
    tw_description: str
    h_tags: List[str]
    paragraphs: List[str]
    list_items: List[str]
    anchor_texts: List[str]
    button_texts: List[str]
    input_placeholders: List[str]
    images_alt: List[str]
    json_ld: List[str]
    semantic_classes: List[str]
    semantic_ids: List[str]
    highlighted_texts: List[str]
    bullets: List[str]
    specs: List[str]


def clean_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    # Remove scripts, styles, and noisy elements
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "link"]):
        tag.decompose()
    # print(f"Residual HTML Soup: \n\n\n[{soup}]\n\n\n\n")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    return soup


def _link_density(element) -> float:
    text_len = len(element.get_text(" ", strip=True)) or 1
    link_text = " ".join(a.get_text(" ", strip=True) for a in element.find_all("a"))
    return len(link_text) / text_len


def _score_block(element) -> float:
    text = element.get_text(" ", strip=True)
    text_len = len(text)
    num_p = len(element.find_all("p"))
    ld = _link_density(element)
    # Favor longer, paragraph-rich, low-link-density blocks
    return (text_len / 100.0) + (num_p * 2.0) - (ld * 50.0)


def extract_main_block(soup: BeautifulSoup):
    candidates = [
        e for e in soup.find_all(MAIN_TAGS)
        if e.name in MAIN_TAGS
    ]
    if not candidates:
        candidates = soup.find_all("div")

    best = None
    best_score = float("-inf")
    for e in candidates:
        score = _score_block(e)
        if score > best_score:
            best = e
            best_score = score
    return best or soup


_TAILWIND_PATTERNS = [
    re.compile(r"^(sm|md|lg|xl|2xl|hover|focus|active|disabled|dark|motion-safe|motion-reduce|aria|data):"),
    re.compile(r"^(flex|grid|block|inline|hidden|sr-only|container)$"),
    re.compile(r"^(items|justify|content|place)-(start|end|center|between|around|evenly)$"),
    re.compile(r"^(w|h|max-w|max-h|min-w|min-h)-(full|screen|\d+|\[)"),
    re.compile(r"^(p|px|py|pt|pr|pb|pl|m|mx|my|mt|mr|mb|ml)-(\d+|px|\[)"),
    re.compile(r"^(space-[xy])-(\d+|px|\[)"),
    re.compile(r"^(text|bg|border|ring|fill|stroke|placeholder|accent|caret)-(?:[a-z-]+|\d+|\[[^\]]+\])(?:/\d+)?$"),
    re.compile(r"^(font)-(?:thin|extralight|light|normal|medium|semibold|bold|extrabold|black|\d+)$"),
    re.compile(r"^(rounded|shadow)(?:-[a-z0-9]+|)$"),
    re.compile(r"^(leading|tracking|align|whitespace|break)-"),
    re.compile(r"^(z|opacity|inset|top|right|bottom|left|order)-\d+"),
    re.compile(r"^(from|to|via)-[a-z0-9\-\[\]]+"),
    re.compile(r"^object-(cover|contain|fill|none|scale-down)$"),
    re.compile(r"^(absolute|relative|fixed|sticky)$"),
    re.compile(r"^(overflow|truncate|transition|duration|ease|animate)-"),
]

_CLASS_VENDOR_BLOCKLIST = {"lucide", "fa", "fas", "far", "fab"}
_CLASS_SEMANTIC_ALLOW_TOKENS = {
    "product","title","subtitle","headline","heading","content","article","news","author","date",
    "price","brand","model","category","rating","review","spec","specs","features","summary","body",
}


def _is_tailwind_class(name: str) -> bool:
    if ":" in name:
        return True
    for pat in _TAILWIND_PATTERNS:
        if pat.match(name):
            return True
    return False


def _extract_semantic_classes(root: BeautifulSoup, limit: int = 40, max_freq: int = 5) -> List[str]:
    # First pass: count class frequencies
    freq: Dict[str, int] = {}
    for el in root.find_all(True):
        for c in (el.get("class") or []):
            cname = str(c).strip()
            if cname:
                freq[cname] = freq.get(cname, 0) + 1

    collected: List[str] = []
    seen = set()
    for el in root.find_all(True):
        cls = el.get("class")
        if not cls:
            continue
        for c in cls:
            cname = str(c).strip()
            if not cname or cname in seen:
                continue
            if freq.get(cname, 0) > max_freq:
                continue
            # Drop framework/vendor classes
            if _is_tailwind_class(cname) or any(cname.startswith(v + "-") or cname == v for v in _CLASS_VENDOR_BLOCKLIST):
                continue
            # Keep only somewhat descriptive names
            if len(cname) < 4 or re.fullmatch(r"[a-zA-Z]{1,3}\d{0,2}", cname):
                continue
            # Require at least one semantic token in the class name
            tokens = re.split(r"[-_]+", cname.lower())
            if not any(tok in _CLASS_SEMANTIC_ALLOW_TOKENS for tok in tokens):
                continue
            seen.add(cname)
            collected.append(cname)
            if len(collected) >= limit:
                return collected
    return collected


def _extract_semantic_ids(root: BeautifulSoup, limit: int = 20) -> List[str]:
    ids: List[str] = []
    seen = set()
    for el in root.find_all(True):
        idv = el.get("id")
        if not idv:
            continue
        name = str(idv).strip()
        if not name or name in seen:
            continue
        if len(name) < 4 or name.isdigit():
            continue
        if any(name.startswith(v + "-") or name == v for v in _CLASS_VENDOR_BLOCKLIST):
            continue
        tokens = re.split(r"[-_]+", name.lower())
        if not any(tok in _CLASS_SEMANTIC_ALLOW_TOKENS for tok in tokens):
            continue
        seen.add(name)
        ids.append(name)
        if len(ids) >= limit:
            break
    return ids


def parse_content(html: str) -> PageContent:
    soup = clean_html(html)
    # print(soup)
    title = (soup.title.string or "").strip() if soup.title else ""
    md = soup.find("meta", attrs={"name": "description"})
    meta_description = (md.get("content", "").strip() if md else "")
    og_title = (soup.find("meta", property="og:title") or {}).get("content", "") if soup else ""
    og_description = (soup.find("meta", property="og:description") or {}).get("content", "") if soup else ""
    tw_title = (soup.find("meta", attrs={"name": "twitter:title"}) or {}).get("content", "") if soup else ""
    tw_description = (soup.find("meta", attrs={"name": "twitter:description"}) or {}).get("content", "") if soup else ""
    h_tags = [
        *[h.get_text(" ", strip=True) for h in soup.find_all("h1")],
        *[h.get_text(" ", strip=True) for h in soup.find_all("h2")],
        *[h.get_text(" ", strip=True) for h in soup.find_all("h3")],
        *[h.get_text(" ", strip=True) for h in soup.find_all("h4")],
        *[h.get_text(" ", strip=True) for h in soup.find_all("h5")],
        *[h.get_text(" ", strip=True) for h in soup.find_all("h6")],
    ]

    # print(f"{len(h_tags)} H Tags Found: [{'\n\n'.join(h_tags)}]\n\n")

    main = extract_main_block(soup)
    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]

    # print(f"{len(paragraphs)} P Tags Found: [{'\n\n'.join(paragraphs)}]")

    highlighted_text = [
        *[p.get_text(" ", strip=True) for p in soup.find_all("b")],
        *[s.get_text(" ", strip=True) for s in soup.find_all("strong")],
        *[u.get_text(" ", strip=True) for u in soup.find_all("u")],
        # *[bl.get_text(" ", strip=True) for bl in soup.find_all("bold")],
    ]
    # print(f"{len(highlighted_text)} Tags Highlighted: [{'\n\n'.join(highlighted_text)}]\n\n\n")

    list_items = [li.get_text(" ", strip=True) for li in main.find_all("li")][:30]
    # Anchor texts that look content-like (exclude menus/nav via short length and repetitive items)
    anchor_texts = [a.get_text(" ", strip=True) for a in main.find_all("a") if len(a.get_text(" ", strip=True)) >= 3][:50]
    # Buttons and input placeholders
    button_texts = [b.get_text(" ", strip=True) for b in main.find_all("button") if b.get_text(" ", strip=True)]
    input_placeholders = [inp.get("placeholder", "").strip() for inp in main.find_all("input") if inp.get("placeholder")]
    images_alt = [img.get("alt", "").strip() for img in main.find_all("img") if img.get("alt")]
    semantic_classes = _extract_semantic_classes(main)
    # print(f"Residual HTML Soup: \n\n\n[{main}]\n\n\n\n")
    semantic_ids = _extract_semantic_ids(main)
    # print(f"Residual HTML Soup: \n\n\n[{soup}]\n\n\n\n")

    # Extract simple JSON-LD strings (names) to boost topics
    json_ld_texts: List[str] = []
    for script in soup.find_all("script", type=re.compile(r"application/(ld\+json|json)")):
        content = script.string
        if content:
            json_ld_texts.append(content[:5000])  # cap to keep small

    # Extract product bullets (e.g., Amazon "About this item")
    bullets: List[str] = []
    fb = soup.find(id=re.compile(r"feature-bullets|featurebullets", re.I))
    if fb:
        bullets.extend([li.get_text(" ", strip=True) for li in fb.find_all("li")])
    else:
        about = soup.find(lambda t: t.name in ("h1","h2","h3","h4","h5","h6") and re.search(r"about this item", t.get_text(" ", strip=True), re.I))
        if about:
            ul = about.find_next("ul")
            if ul:
                bullets.extend([li.get_text(" ", strip=True) for li in ul.find_all("li")])

    # Extract simple spec tables (key-value rows)
    specs: List[str] = []
    spec_containers = [
        soup.find(id=re.compile(r"productDetails_techSpec_section_1", re.I)),
        soup.find(id=re.compile(r"productDetails_detailBullets_sections1", re.I)),
        soup.find(id=re.compile(r"productOverview_feature_div", re.I)),
    ]
    for container in filter(None, spec_containers):
        for row in container.find_all("tr"):
            th = row.find(["th","td"])
            tds = row.find_all("td")
            if th and tds:
                key = th.get_text(" ", strip=True)
                val = tds[-1].get_text(" ", strip=True)
                if key and val:
                    specs.append(f"{key}: {val}")
        for dl in container.find_all("dl"):
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                key = dt.get_text(" ", strip=True)
                val = dd.get_text(" ", strip=True)
                if key and val:
                    specs.append(f"{key}: {val}")

    # Price extraction (Amazon and generic): prefer visible accessible price spans
    price_candidates: List[str] = []
    try:
        for sp in main.select('span.a-offscreen, span.a-color-price, span.p13n-sc-price'):
            txt = sp.get_text(" ", strip=True)
            if re.match(r"^\$\s?\d{1,4}(?:[.,]\d{2})$", txt):
                price_candidates.append(txt)
    except Exception:
        pass
    if not price_candidates:
        # Fallback: search main text for a price-like pattern
        main_text = main.get_text(" ", strip=True)
        found = re.findall(r"\$\s?\d{1,4}(?:[.,]\d{2})", main_text)
        if found:
            price_candidates.append(found[0])
    if price_candidates:
        specs.append(f"Price: {price_candidates[0]}")

    return PageContent(
        title=title,
        meta_description=meta_description,
        og_title=og_title,
        og_description=og_description,
        tw_title=tw_title,
        tw_description=tw_description,
        h_tags=h_tags,
        paragraphs=paragraphs,
        list_items=list_items,
        anchor_texts=anchor_texts,
        button_texts=button_texts,
        input_placeholders=input_placeholders,
        images_alt=images_alt,
        json_ld=json_ld_texts,
        semantic_classes=semantic_classes,
        semantic_ids=semantic_ids,
        highlighted_texts=highlighted_text,
        bullets=bullets,
        specs=specs,
    )


