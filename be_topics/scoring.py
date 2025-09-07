from __future__ import annotations

import math
from collections import Counter, defaultdict
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .candidates import Candidate
from nltk.stem import PorterStemmer


SourceBoost = {
    "title": 2.5,
    "h": 2.0,
    "meta": 0.3,
    "og": 2.5,
    "twitter": 2.0,
    "body": 0.5,
    "bullet": 1.1,
    "spec": 1.2,
    "url": 1.0,
    "alt": 0.5,
    "li": 1.8,
    "a": 0.3,
    "button": 0.2,
    "placeholder": 0.8,
    "class": 0.6,
    "id": 0.8,
    "jsonld": 1.2,
    "strong": 1.5,
    "b": 1.5,
    "p": 0.5
}


@dataclass
class ScoredTopic:
    text: str
    score: float
    sources: Dict[str, int]


def _compute_tf(candidates: List[Candidate]) -> Dict[str, float]:
    # print(f"Computing TF for {len(candidates)} candidates: \n\n\n{candidates}\n\n\n")
    stemmer = PorterStemmer()

    def key(s: str) -> str:
        tokens = s.lower().split()
        # Stem only alpha tokens; keep digits/models unchanged
        norm = [stemmer.stem(t) if t.isalpha() and len(t) > 2 else t for t in tokens]
        return " ".join(norm)
    counts = Counter(key(c.text) for c in candidates)
    total = sum(counts.values()) or 1
    # Map back to display text by picking one representative per key
    rep_map: Dict[str, str] = {}
    for c in candidates:
        k = key(c.text)
        if k not in rep_map:
            rep_map[k] = c.text
    return {rep_map[t]: c / total for t, c in counts.items()}


def _boost_from_sources(candidates: List[Candidate]) -> Dict[str, Dict[str, int]]:
    stemmer = PorterStemmer()
    def key(s: str) -> str:
        tokens = s.lower().split()
        norm = [stemmer.stem(t) if t.isalpha() and len(t) > 2 else t for t in tokens]
        return " ".join(norm)
    # Group sources by stemmed key
    rep_map: Dict[str, str] = {}
    for c in candidates:
        k = key(c.text)
        if k not in rep_map:
            rep_map[k] = c.text
    src_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in candidates:
        k = key(c.text)
        disp = rep_map[k]
        src_counts[disp][c.source] += 1
    return src_counts


def score_candidates(candidates: List[Candidate]) -> List[ScoredTopic]:
    tf = _compute_tf(candidates)
    src_counts = _boost_from_sources(candidates)

    scored: List[ScoredTopic] = []
    for phrase, tf_val in tf.items():
        srcs = src_counts.get(phrase, {})
        boost = sum(SourceBoost.get(src, 0.0) * count for src, count in srcs.items())
        # N-gram length boost to prefer 2-3 word phrases
        n_words = max(1, len(phrase.split()))
        length_boost = {1: 0.8, 2: 1.2, 3: 1.4}.get(n_words, 1.0)
        score = tf_val * (1.0 + boost) * length_boost
        # Category-agnostic product boosts: model patterns and units
        if re.search(r"\b[A-Z]{2,}\d{2,}\b|\b\d{1,2}(-|\s)?slice\b", phrase):
            score *= 1.35
        if re.search(r"\b(\d+(?:\.\d+)?\s?(inch|in|w|v|watts|lbs|pounds))\b", phrase):
            score *= 1.2
        scored.append(ScoredTopic(text=phrase, score=score, sources=dict(srcs)))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def diversify(scored: List[ScoredTopic], similarity_threshold: float = 0.8) -> List[ScoredTopic]:
    def _canon(s: str) -> str:
        s = s.lower().replace('-', ' ')
        return re.sub(r"\s+", " ", s).strip()

    def jaccard(a: str, b: str) -> float:
        sa, sb = set(_canon(a).split()), set(_canon(b).split())
        inter = len(sa & sb)
        union = len(sa | sb) or 1
        return inter / union

    selected: List[ScoredTopic] = []
    for cand in scored:
        is_dup = False
        for s in selected:
            # High textual overlap
            if jaccard(cand.text, s.text) >= similarity_threshold:
                is_dup = True
                break
            ca, cb = _canon(cand.text), _canon(s.text)
            # Subset/superset suppression
            if ca in cb or cb in ca:
                is_dup = True
                break
            # Title shingle suppression: if both are primarily from title, keep the earlier (higher score)
            cand_title = any(k in cand.sources for k in ("title", "og", "twitter"))
            sel_title = any(k in s.sources for k in ("title", "og", "twitter"))
            if cand_title and sel_title and jaccard(cand.text, s.text) >= 0.5:
                is_dup = True
                break
        if not is_dup:
            selected.append(cand)
    return selected


