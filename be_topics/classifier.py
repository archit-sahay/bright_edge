from __future__ import annotations

import re
from enum import Enum
from typing import Dict

from .parser import PageContent


class PageType(str, Enum):
    PRODUCT = "product"
    ARTICLE = "article"
    NEWS = "news"
    OTHER = "other"


def classify_page(content: PageContent) -> PageType:
    t = " ".join([content.title] + content.h_tags[:1] + content.paragraphs[:2]).lower()

    # Product signals
    product_signals = [
        re.search(r"\badd to cart\b|\bbuy now\b|\bsku\b|\bmodel\b|\bprice\b", t),
        re.search(r"\$\s?\d+(?:[.,]\d{2})?", t),
    ]
    if any(product_signals):
        return PageType.PRODUCT

    # News signals
    if re.search(r"\bnews\b|\bbreaking\b|\bcnn\b|\bassociated press\b|\bby [A-Z][a-z]+\b", t):
        return PageType.NEWS

    # Article/blog signals
    if len(" ".join(content.paragraphs[:3]).split()) > 60:
        return PageType.ARTICLE

    return PageType.OTHER


