from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import ssl
import nltk

from nltk.stem import PorterStemmer

from .parser import PageContent


try:
    # nltk.download("punkt")
    # nltk.download("punkt_tab")
    # nltk.download("stopwords")
    # ssl._create_default_https_context = _create_unverified_https_context
    from nltk.corpus import stopwords  # type: ignore

    # nltk.download("punkt")
    # nltk.download("punkt_tab")
    # nltk.download("stopwords")
    # print("Downloaded")

    ok = {
        'any', 'haven', 'aren', 'himself', 'ain', 'mustn', 'doesn', 'herself', "isn't", 'can', 'both', 'needn', 'myself', "he's", 'once',
        "a","an","the","and","or","but","if","then","else","for","to","of","in","on","with","by","from",
        "at","as","is","are","was","were","be","been","being","this","that","these","those","it","its",
        "you","your","we","our","they","their","he","she","his","her","them","us","i","me","my", "control", "becoming"
        "about","into","over","under","up","down","out","off","so","not","no","yes","can","will","make","makes", "feature","featured",
        'doing', 'same', 'is', "he'll", 'down', 'themselves', 'own', "couldn't", "she'd", 'o', 'into', 'was', 'yourselves', "you've", 'and', 'd', 'about', "we'll", 'where', "won't", "they're", "i'm", 'weren', 'hers', 'above', 'we', 'my', 'off', "i'll", 'shouldn', 'those', 'theirs', 'just', 'itself', 'again', 'here', 'his', 'all', 'hadn', 'while', 'or', "should've", 'whom', "she's", 'she', 'why', 'he', 'through', 'during', 'each', "hadn't", 'had', "you'll", 'at', "doesn't", 'these', 'how', 'but', "we've", 'isn', 'him', "wasn't", 'were', 'has', "we'd", 'me', "they've", 'did', 'wouldn', 'against', 'will', "hasn't", 'between', 'are', 'the', 'your', "didn't", "aren't", "he'd", 've', 'which', 'very', 'mightn', 'until', "shan't", "you'd", 'because', "she'll", 'other', 'don', 'in', "they'd", 'wasn', 'from', 'won', 'having', 'our', 'couldn', 'for', 'to', "they'll", 'their', 'then', 'ma', 'too', 'y', 'a', "that'll", "i'd", 'when', "we're", "wouldn't", 'as', 'what', 'you', 'does', 'than', 'it', 'shan', 'now', 'of', 'i', 'below', 're', 'ours', "it's", 'yourself', 'before', 'few', 'll', 'didn', "i've", 'on', 'out', 'that', 'after', "it'd", "needn't", 'have', 'such', "shouldn't", 'so', 'who', 'more', 'should', 'under', 'them', "mustn't", "it'll", 'this', "weren't", 'hasn', 'further', 'yours', 'they', 'am', 'with', 'there', "haven't", 'some', 'by', 'over', 'an', 'its', 'up', 'been', 'being', 't', "you're", 'no', 'do', 'most', "don't", 'if', 'her', 'm', 'be', 'not', 'only', 's', "mightn't", 'nor', 'ourselves'}

    _STOPWORDS = set(ok)  # requires corpus; may fail
except Exception as e:
    print(f"Exception: {e}")
    # Minimal fallback stopword list to avoid NLTK downloads
    # _STOPWORDS = {
    #     "a","an","the","and","or","but","if","then","else","for","to","of","in","on","with","by","from",
    #     "at","as","is","are","was","were","be","been","being","this","that","these","those","it","its",
    #     "you","your","we","our","they","their","he","she","his","her","them","us","i","me","my",
    #     "about","into","over","under","up","down","out","off","so","not","no","yes","can","will","make","makes", "feature","featured"
    # }
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")
_STEMMER = PorterStemmer()
_STOP_VERBS = {
    "meet","start","enter","change","unmute","learn","click","submit","contact",
}
_PRONOUNS_DET = {"my","your","our","his","her","their","this","that","these","those"}

# Generic e-commerce/navigation noise tokens and phrases
_ECOM_NOISE_TOKENS = {
    "amazon","com","hello","sign","account","lists","returns","orders","cart","all",
    "today","deals","prime","video","registry","customer","service","gift","cards",
    "sell","home","kitchen","share","sponsored","learn","more","search","shift","alt",
    "view","history","keyboard","shortcuts","add","buying","options","compare","similar",
    "items","previous","next","set","slides","ratings","stars","price","prices","usd",
    "deliver","delivery","india","united","states","watch","now","download","pdf","link",
}
_ECOM_NOISE_PHRASES = {
    "add to cart","buying options","compare with similar","keyboard shortcuts","hello sign in",
    "returns & orders","gift cards","customer service","today's deals","prime video","0 cart",
    "see more product details","report an issue","product description","product information",
    "warranty & support","from the manufacturer","user manual","visit the store","learn more",
}

# Wikipedia/navigation/UI specific noise
_WIKI_NOISE_TOKENS = {
    "edit","jump","navigation","sidebar","toc","table","contents","toggle","subsection",
    "move","hide","show","top","category","talk","read","view","history","source",
    "wikipedia","wikidata","mediawiki",
}
_WIKI_NOISE_PHRASES = {
    "move to sidebar","table of contents","edit this at wikidata","toggle subsection",
    "download as pdf","printable version","permanent link","page information","cite this page",
}


@dataclass
class Candidate:
    text: str
    source: str  # title, h, body, url, meta, jsonld, alt


def _normalize_phrase(tokens: List[str]) -> str:
    # Lowercase, collapse dashes/spaces, strip punctuation ends (no stemming for display)
    words = [t.lower() for t in tokens]
    phrase = " ".join(words)
    phrase = re.sub(r"\s+", " ", phrase).strip(" -_")
    return phrase


def _clean_title(title: str, url: str) -> str:
    from urllib.parse import urlparse
    t = title.strip()
    host = (urlparse(url).hostname or "").lower()
    # Amazon: drop leading prefix and trailing department
    if "amazon." in host:
        t = re.sub(r"^amazon(?:\.\w+)?\s*:\s*", "", t, flags=re.I)
        # Drop trailing department segment after colon
        t = re.sub(r"\s*:\s*[^:]+$", "", t)
    # Wikipedia: drop trailing site suffix
    if "wikipedia.org" in host:
        t = re.sub(r"\s*[\-\–]\s*wikipedia.*$", "", t, flags=re.I)
    return t.strip()



def _extract_tokens(text: str) -> List[str]:
    # Regex tokenization only; avoids NLTK punkt dependency
    toks = _TOKEN_RE.findall(text)
    return [t for t in toks if t]


def _generate_ngrams(tokens: List[str], n: int) -> Iterable[List[str]]:
    L = len(tokens)
    for i in range(L - n + 1):
        yield tokens[i : i + n]


def _is_valid_phrase(tokens: List[str]) -> bool:
    if not tokens:
        return False
    # Drop phrases that are purely numeric
    if all(re.fullmatch(r"\d+(?:\.\d+)?", t) for t in tokens):
        return False
    # Filter stopwords-only phrases and very short tokens
    if all(t.lower() in _STOPWORDS for t in tokens):
        return False
    if all(len(t) <= 1 for t in tokens):
        return False
    # Leading/trailing stop/pronoun tokens reduce quality
    if len(tokens) >= 2:
        if tokens[0].lower() in _STOPWORDS | _PRONOUNS_DET:
            return False
        if tokens[-1].lower() in _STOPWORDS | _PRONOUNS_DET:
            return False
        # Drop phrases dominated by stopwords
        sw_ratio = sum(1 for t in tokens if t.lower() in _STOPWORDS) / len(tokens)
        if sw_ratio >= 0.5:
            return False
    # Single-word: avoid stop verbs/pronouns and too short
    if len(tokens) == 1:
        t = tokens[0].lower()
        # Avoid numeric-only single tokens
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            return False
        if t in _STOPWORDS or t in _STOP_VERBS or t in _PRONOUNS_DET or len(t) < 3:
            return False
    # Avoid phrases largely composed of verbs/UI actions
    if any(t.lower() in _STOP_VERBS for t in tokens):
        return False
    # Allow model numbers like CPT-122
    return True


def _preprocess_text(text: str, source: str) -> str:
    t = text
    # If spec key-value, keep value only
    if source == "spec" and ":" in t:
        t = t.split(":", 1)[1]
    # Normalize measurement units like 6.5"D x 11"W x 7"H → 6.5 in x 11 in x 7 in
    t = re.sub(r"(\d+(?:\.\d+)?)\s*[\"”']?\s*[dDwWhH]\b", r"\1 in", t)
    # Normalize standalone inches like 1.5-inch, 7\" → 1.5 in
    t = re.sub(r"(\d+(?:\.\d+)?)(?:\s*[-\"]\s*|\s*)(?:inch|in|\")\b", r"\1 in", t, flags=re.I)
    # Normalize pounds
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:pounds|lbs?)\b", r"\1 lb", t, flags=re.I)
    # Normalize metric/other length units
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:centimeters|centimetres|cm)\b", r"\1 cm", t, flags=re.I)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:millimeters|millimetres|mm)\b", r"\1 mm", t, flags=re.I)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:feet|foot|ft)\b", r"\1 ft", t, flags=re.I)
    # Normalize watts/volts abbreviations
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:watts?)\b", r"\1 watts", t, flags=re.I)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:volts?|v)\b", r"\1 v", t, flags=re.I)
    # Strip Wikipedia-style citation markers like [ 123 ] and caret references
    t = re.sub(r"\[\s*\d+\s*\]", "", t)
    t = re.sub(r"\s*\^\s*", " ", t)
    return t


def _is_noise_phrase(phrase: str) -> bool:
    p = phrase.lower().strip()
    if p in _ECOM_NOISE_PHRASES:
        return True
    if p in _WIKI_NOISE_PHRASES:
        return True
    toks = p.split()
    if not toks:
        return True
    # Dimension-style phrases like "6.5 x 11 x 7 in" → drop entirely
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*(?:in|inch|cm|mm|ft))?(?:\s*x\s*\d+(?:\.\d+)?(?:\s*(?:in|inch|cm|mm|ft))?)+\s*(?:in|inch|cm|mm|ft)?", p, flags=re.I):
        return True
    # High ratio of noise tokens
    noise_ratio = sum(1 for t in toks if t in _ECOM_NOISE_TOKENS or t in _WIKI_NOISE_TOKENS) / len(toks)
    if noise_ratio >= 0.5:
        return True
    # Keyboard shortcut patterns
    if re.search(r"\bshift\b.*\balt\b|\bctrl\b|\bopt\b", p):
        return True
    # Drop phrases that include Wikipedia brand markers
    if "wikipedia" in p or "wikidata" in p or "mediawiki" in p:
        return True
    # TOC-like stubs such as "1.2", "1", "2.1.3"
    if re.fullmatch(r"\d+(?:\.\d+)*", p):
        return True
    # Phrases ending/starting with bare 'x' from dimensions are noisy in general extraction
    if re.fullmatch(r"\d+(?:\.\d+)?\s*x", p) or re.fullmatch(r"x\s*\d+(?:\.\d+)?", p):
        return True
    return False


def _is_wiki_toc_item(text: str) -> bool:
    s = text.strip()
    if re.match(r"^\(?top\)?$", s, flags=re.I):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s", s):
        return True
    sl = s.lower()
    if any(tok in sl for tok in ("toggle", "subsection", "table of contents")):
        return True
    return False


def _extract_phrases_from_text(text: str, source: str, max_ngram: int = 3) -> List[Candidate]:
    pre = _preprocess_text(text, source)
    tokens = [t for t in _extract_tokens(pre) if t.lower() not in _ECOM_NOISE_TOKENS]
    phrases: List[Candidate] = []
    for n in range(1, max_ngram + 1):
        for ngram in _generate_ngrams(tokens, n):
            if _is_valid_phrase(ngram):
                norm = _normalize_phrase(ngram)
                if not _is_noise_phrase(norm):
                    phrases.append(Candidate(text=norm, source=source))
    return phrases


def _extract_url_phrases(url: str) -> List[Candidate]:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path or ""
    # Keep only meaningful path segments
    segs = [s for s in path.split('/') if s]
    kept_tokens: List[str] = []
    for s in segs:
        # Keep brand/model/product tokens and words
        if re.fullmatch(r"[A-Za-z0-9\-]+", s):
            # Skip known noise segments
            if s.lower() in {"dp","ref","gp","s","bestsellers","bestseller","sr"}:
                continue
            kept_tokens.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", s))
    text = " ".join(kept_tokens)
    return _extract_phrases_from_text(text, source="url")


def generate_candidates(content: PageContent, url: str, include_css_topics: bool = False) -> List[Candidate]:
    candidates: List[Candidate] = []
    # Title and headings
    if content.title:
        candidates += _extract_phrases_from_text(_clean_title(content.title, url), source="title")
    if content.og_title:
        candidates += _extract_phrases_from_text(_clean_title(content.og_title, url), source="og")
    if content.tw_title:
        candidates += _extract_phrases_from_text(_clean_title(content.tw_title, url), source="twitter")
    HEADING_NOISE_SUBSTR = {
        "feedback","price","product description","product information","options available",
        "keyboard shortcuts","customers who viewed this item also viewed","similar brands on amazon",
        "warranty & support","product videos","product guidance & documents","from the manufacturer",
        "top brand","safety documents","image unavailable","sorry, there was a problem",
        "product summary","about this item","deals on related products","brands you might like",
        # Generic wiki headings
        "personal life","career","early life","references","external links","see also",
        "bibliography","notes","further reading","works","media","writing","filmography",
    }
    for h in content.h_tags[:5]:
        h_norm = h.strip().lower()
        if any(substr in h_norm for substr in HEADING_NOISE_SUBSTR):
            continue
        candidates += _extract_phrases_from_text(h, source="h")

    # Early paragraphs only
    for p in content.paragraphs[:6]:
        candidates += _extract_phrases_from_text(p, source="body")

    # Product bullets and specs (high-signal for products)
    for b in getattr(content, "bullets", [])[:12]:
        candidates += _extract_phrases_from_text(b, source="bullet")
    for s in getattr(content, "specs", [])[:20]:
        candidates += _extract_phrases_from_text(s, source="spec")

    # Meta description
    if content.meta_description and not re.search(r"amazon|online shopping", content.meta_description, re.I):
        candidates += _extract_phrases_from_text(content.meta_description, source="meta")
    if content.og_description:
        candidates += _extract_phrases_from_text(content.og_description, source="og")
    if content.tw_description:
        candidates += _extract_phrases_from_text(content.tw_description, source="twitter")

    # Image alts (cap)
    for alt in content.images_alt[:10]:
        candidates += _extract_phrases_from_text(alt, source="alt")

    # List items and anchor texts for sparse pages
    for li in content.list_items[:20]:
        if _is_wiki_toc_item(li):
            continue
        candidates += _extract_phrases_from_text(li, source="li")
    for a in content.anchor_texts[:30]:
        if _is_wiki_toc_item(a):
            continue
        candidates += _extract_phrases_from_text(a, source="a")
    for b in content.button_texts[:10]:
        candidates += _extract_phrases_from_text(b, source="button")
    for ph in content.input_placeholders[:10]:
        candidates += _extract_phrases_from_text(ph, source="placeholder")

    # Non-tailwind semantic classes can hint at topics on sparse pages
    if include_css_topics:
        for css in content.semantic_classes[:20]:
            # Convert kebab/camel to space-separated words
            text = re.sub(r"[-_]+", " ", css)
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
            candidates += _extract_phrases_from_text(text, source="class")
        for ident in content.semantic_ids[:10]:
            text = re.sub(r"[-_]+", " ", ident)
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
            candidates += _extract_phrases_from_text(text, source="id")

    # URL path tokens (cleaned)
    candidates += _extract_url_phrases(url)

    # JSON-LD raw text (very limited)
    for j in content.json_ld[:2]:
        candidates += _extract_phrases_from_text(j, source="jsonld")

    # Deduplicate
    seen = set()
    unique: List[Candidate] = []
    for c in candidates:
        key = c.text
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


