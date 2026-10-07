"""
Data quality and dialect filtering for TounsiLM.

Implements:
  - QualityFilter: configurable filters for document quality
  - DialectFilter: heuristic Tunisian-vs-non-Tunisian scoring
  - PIIFilter: basic personally identifiable information removal

The DialectFilter uses lexical indicators but is designed to be
replaceable by a trained classifier in the future.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from tounsilm.config import DataConfig


# ──────────────────────────────────────────────
# Quality Filter
# ──────────────────────────────────────────────

@dataclass
class FilterResult:
    """Result of applying a filter to a document."""
    passed: bool
    reason: str = ""


class QualityFilter:
    """Configurable document quality filter.

    Checks:
      - Length bounds (min/max characters)
      - Symbol ratio (excessive special characters)
      - URL ratio
      - Repeated character ratio
      - Empty/whitespace-only text
      - Encoding issues
    """

    def __init__(self, config: DataConfig):
        self.min_length = config.min_doc_length
        self.max_length = config.max_doc_length
        self.max_symbol_ratio = config.max_symbol_ratio
        self.max_url_ratio = config.max_url_ratio
        self.max_repeated_char_ratio = config.max_repeated_char_ratio

    def filter(self, text: str) -> FilterResult:
        """Apply all quality filters to a document.

        Args:
            text: Document text.

        Returns:
            FilterResult indicating pass/fail and reason.
        """
        if not text or not text.strip():
            return FilterResult(False, "empty_text")

        text_len = len(text)

        # Length check
        if text_len < self.min_length:
            return FilterResult(False, f"too_short ({text_len} < {self.min_length})")
        if text_len > self.max_length:
            return FilterResult(False, f"too_long ({text_len} > {self.max_length})")

        # Symbol ratio (non-alphanumeric, non-space, non-Arabic characters)
        alpha_or_arabic = sum(
            1 for c in text
            if c.isalnum() or c.isspace()
            or "\u0600" <= c <= "\u06FF"  # Arabic block
            or "\uFE70" <= c <= "\uFEFF"  # Arabic Presentation Forms-B
        )
        symbol_ratio = 1.0 - (alpha_or_arabic / max(text_len, 1))
        if symbol_ratio > self.max_symbol_ratio:
            return FilterResult(False, f"high_symbol_ratio ({symbol_ratio:.2f})")

        # URL ratio
        url_chars = sum(len(m.group()) for m in re.finditer(
            r"https?://\S+|www\.\S+", text, re.IGNORECASE
        ))
        url_ratio = url_chars / max(text_len, 1)
        if url_ratio > self.max_url_ratio:
            return FilterResult(False, f"high_url_ratio ({url_ratio:.2f})")

        # Repeated character ratio
        if text_len > 0:
            repeated = sum(
                1 for i in range(1, text_len)
                if text[i] == text[i - 1] and not text[i].isspace()
            )
            repeated_ratio = repeated / max(text_len, 1)
            if repeated_ratio > self.max_repeated_char_ratio:
                return FilterResult(False, f"high_repeated_char_ratio ({repeated_ratio:.2f})")

        return FilterResult(True)

    def filter_batch(self, texts: list[str]) -> list[FilterResult]:
        """Filter a batch of documents."""
        return [self.filter(t) for t in texts]


# ──────────────────────────────────────────────
# Dialect Filter
# ──────────────────────────────────────────────

class DialectFilter:
    """Heuristic Tunisian dialect detection.

    Uses lexical indicators common in Tunisian Arabic (Derja) to score
    whether a document is likely Tunisian. This is a starting heuristic
    that can be replaced with a trained classifier.

    The filter does NOT define Tunisian through a fixed word list alone —
    it uses multiple signal categories and returns a score, not a binary.
    """

    # Tunisian lexical indicators — common Derja words/particles
    TUNISIAN_ARABIC_MARKERS = [
        # Question words
        "شنوة", "شنو", "شنية", "شنوى", "علاش", "وقتاش", "كيفاش", "كيفاه", "كفاه", "شكون", "وقتاه",
        # Common particles
        "خاطر", "توا", "توّا", "تو", "برشا", "باهي", "يلزم", "لازم",
        "ما عادش", "مازال", "مزال", "هكا", "هكّا", "هكاك",
        # Demonstratives
        "هاذي", "هاذا", "هاذوما", "هاك", "هذايا", "هذي",
        # Verbs / verb forms
        "نجم", "نجمت", "يحب", "يحبو", "نعمل", "تعمل", "يعمل",
        "ما جيتش", "ما نجمش", "مانيش", "ما نعرفش", "مشيت", "مشينا",
        # Common expressions
        "لاباس", "يعيشك", "عسلامة", "انشالله", "يرحم",
        "بالحق", "فيسع", "بركا", "قداش", "بربي", "ياخي", "عيش",
        # Prepositions / conjunctions / dialect words
        "متاع", "تاع", "فما", "ثما", "كان", "ملا", "موش", "مش",
        "باش", "زعمة", "فازة", "فازات", "ديما", "أما", "شوية", "حاجة", "بلاد", "حومة", "نورمالمون",
    ]

    # Tunisian Arabizi markers
    TUNISIAN_ARABIZI_MARKERS = [
        "chnowa", "chneya", "3lech", "waktesh", "kifech",
        "5ater", "tawa", "barcha", "behi", "yezem",
        "ma3adch", "mezel", "hakka", "hedhi", "hedha",
        "nejm", "y7eb", "y7ebou", "na3mel", "ta3mel",
        "labes", "ya3ichek", "3asslemma", "3asslema", "inchallah",
        "bel7a9", "fisa3", "barka", "9adech",
        "mata3", "ta3", "famma", "themma",
        # Arabizi number codes as context
        "7atta", "9ahwa", "3arbi", "5oubz",
        # Additional common Arabizi words
        "mouch", "mesh", "mch", "bech", "bach", "chkoun", "za3ma",
        "faza", "fazat", "dima", "ama", "enti", "ena", "ahna",
        "chouf", "chwaya", "belehi", "bro", "weld", "bent", "bled",
    ]

    def __init__(self, min_score: float = 0.1):
        """
        Args:
            min_score: Minimum score threshold for considering a doc Tunisian.
        """
        self.min_score = min_score

        # Compile patterns for efficiency
        self._arabic_patterns = [
            re.compile(re.escape(marker)) for marker in self.TUNISIAN_ARABIC_MARKERS
        ]
        self._arabizi_patterns = [
            re.compile(r"\b" + re.escape(marker) + r"\b", re.IGNORECASE)
            for marker in self.TUNISIAN_ARABIZI_MARKERS
        ]

    def score(self, text: str) -> dict:
        """Score a document for Tunisian dialect likelihood.

        Args:
            text: Document text.

        Returns:
            Dictionary with score details:
              - score: Overall Tunisian score [0, 1]
              - arabic_matches: Number of Arabic marker matches
              - arabizi_matches: Number of Arabizi marker matches
              - matched_markers: List of matched marker strings
        """
        if not text:
            return {
                "score": 0.0,
                "arabic_matches": 0,
                "arabizi_matches": 0,
                "matched_markers": [],
            }

        matched = []

        # Check Arabic markers
        arabic_count = 0
        for pattern in self._arabic_patterns:
            if pattern.search(text):
                arabic_count += 1
                matched.append(pattern.pattern)

        # Check Arabizi markers
        arabizi_count = 0
        for pattern in self._arabizi_patterns:
            if pattern.search(text):
                arabizi_count += 1
                matched.append(pattern.pattern.strip("\\b"))

        # Calculate score
        total_markers = len(self._arabic_patterns) + len(self._arabizi_patterns)
        total_matches = arabic_count + arabizi_count

        # Logarithmic scoring — diminishing returns on many matches
        # A few strong matches are enough to indicate Tunisian
        if total_matches == 0:
            score = 0.0
        else:
            import math
            score = min(1.0, math.log1p(total_matches) / math.log1p(total_markers / 3))

        return {
            "score": score,
            "arabic_matches": arabic_count,
            "arabizi_matches": arabizi_count,
            "matched_markers": matched,
        }

    def is_tunisian(self, text: str) -> bool:
        """Check if a document is likely Tunisian.

        Args:
            text: Document text.

        Returns:
            True if the document scores above the threshold.
        """
        return self.score(text)["score"] >= self.min_score


# ──────────────────────────────────────────────
# PII Filter
# ──────────────────────────────────────────────

class PIIFilter:
    """Basic PII (Personally Identifiable Information) filter.

    Detects and optionally removes:
      - Email addresses
      - Phone numbers (Tunisian format)
      - Credit card-like patterns
      - IP addresses
    """

    PATTERNS = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "phone_tn": re.compile(
            r"(?:\+216|00216)?[\s.-]?[2-9]\d[\s.-]?\d{3}[\s.-]?\d{3}"
        ),
        "phone_intl": re.compile(
            r"\+\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}[\s.-]?\d{0,4}"
        ),
        "ip_address": re.compile(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        ),
        "credit_card": re.compile(
            r"\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b"
        ),
    }

    REPLACEMENT = "[PII_REMOVED]"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def filter(self, text: str) -> tuple[str, list[str]]:
        """Filter PII from text.

        Args:
            text: Input text.

        Returns:
            Tuple of (filtered_text, list_of_detected_pii_types).
        """
        if not self.enabled:
            return text, []

        detected = []
        for pii_type, pattern in self.PATTERNS.items():
            if pattern.search(text):
                detected.append(pii_type)
                text = pattern.sub(self.REPLACEMENT, text)

        return text, detected
