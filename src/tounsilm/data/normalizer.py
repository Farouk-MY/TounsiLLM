"""
Tunisian-aware text normalization.

Handles:
  - Arabic Unicode normalization (careful to preserve dialect distinctions)
  - Whitespace normalization
  - Repeated punctuation/character reduction
  - HTML remnant removal
  - URL / username cleaning
  - Malformed Unicode cleanup

Arabizi is treated separately — numeric codes (3, 7, 5, 9, etc.) are
preserved since they represent Arabic letters in Arabizi convention.

IMPORTANT: This normalizer does NOT blindly convert all Arabic letter
variants, because certain distinctions may carry dialect information.
"""

from __future__ import annotations

import re
import unicodedata


class TunisianNormalizer:
    """Text normalizer for Tunisian Arabic / Arabizi / mixed content."""

    # ── Arabic normalization maps ──
    # We normalize only the variants that are purely cosmetic.
    # We do NOT collapse letter distinctions useful in dialect.
    ARABIC_NORMALIZATION = {
        "\u0622": "\u0627",  # آ → ا (alef with madda above → plain alef)
        "\u0623": "\u0627",  # أ → ا (alef with hamza above)
        "\u0625": "\u0627",  # إ → ا (alef with hamza below)
        "\u0671": "\u0627",  # ٱ → ا (alef wasla)
        "\u0624": "\u0648",  # ؤ → و (waw with hamza) — optional, configurable
        "\u0626": "\u064A",  # ئ → ي (ya with hamza) — optional, configurable
    }

    # Diacritics (tashkeel) to optionally remove
    DIACRITICS = re.compile(
        "[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
    )

    # Tatweel (kashida)
    TATWEEL = "\u0640"

    # HTML patterns
    HTML_TAG = re.compile(r"<[^>]+>")
    HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")

    # URL pattern
    URL_PATTERN = re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE
    )

    # Username / mention pattern
    MENTION_PATTERN = re.compile(r"@[\w]+")

    # Hashtag pattern (preserve the text, remove #)
    HASHTAG_PATTERN = re.compile(r"#([\w]+)")

    # Repeated character limiter (3+ of the same char → 2)
    REPEATED_CHAR = re.compile(r"(.)\1{2,}")

    # Repeated punctuation limiter
    REPEATED_PUNCT = re.compile(r"([!?.,؟،])\1{2,}")

    # Excessive whitespace
    MULTI_SPACE = re.compile(r"[ \t]+")
    MULTI_NEWLINE = re.compile(r"\n{3,}")

    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        normalize_alef: bool = True,
        normalize_hamza: bool = False,  # Conservative default
        remove_urls: bool = True,
        remove_mentions: bool = True,
        limit_repeated_chars: bool = True,
        limit_repeated_punct: bool = True,
        remove_html: bool = True,
    ):
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.normalize_alef = normalize_alef
        self.normalize_hamza = normalize_hamza
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.limit_repeated_chars = limit_repeated_chars
        self.limit_repeated_punct = limit_repeated_punct
        self.remove_html = remove_html

    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline.

        Args:
            text: Raw input text.

        Returns:
            Normalized text.
        """
        if not text:
            return ""

        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Remove null bytes and control characters (except newline/tab)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # HTML cleanup
        if self.remove_html:
            text = self.HTML_TAG.sub("", text)
            text = self.HTML_ENTITY.sub("", text)

        # URL removal
        if self.remove_urls:
            text = self.URL_PATTERN.sub("", text)

        # Mention removal
        if self.remove_mentions:
            text = self.MENTION_PATTERN.sub("", text)

        # Hashtag: keep text, remove #
        text = self.HASHTAG_PATTERN.sub(r"\1", text)

        # Arabic-specific normalization
        if self.remove_diacritics:
            text = self.DIACRITICS.sub("", text)

        if self.remove_tatweel:
            text = text.replace(self.TATWEEL, "")

        if self.normalize_alef:
            for old, new in self.ARABIC_NORMALIZATION.items():
                # Only apply alef normalization, skip hamza-on-carrier
                if old in ("\u0622", "\u0623", "\u0625", "\u0671"):
                    text = text.replace(old, new)

        if self.normalize_hamza:
            for old, new in self.ARABIC_NORMALIZATION.items():
                if old in ("\u0624", "\u0626"):
                    text = text.replace(old, new)

        # Repeated character/punctuation limiting
        if self.limit_repeated_chars:
            text = self.REPEATED_CHAR.sub(r"\1\1", text)

        if self.limit_repeated_punct:
            text = self.REPEATED_PUNCT.sub(r"\1\1", text)

        # Whitespace normalization
        text = self.MULTI_SPACE.sub(" ", text)
        text = self.MULTI_NEWLINE.sub("\n\n", text)
        text = text.strip()

        return text

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalize a batch of texts."""
        return [self.normalize(t) for t in texts]
