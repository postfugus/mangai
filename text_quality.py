"""Small language and OCR sanity checks for cached manga translations."""

import re
import unicodedata


_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_LETTERS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]")
_MOJIBAKE = re.compile(r"(?:Ã.|Â.|â€|\ufffd)")


def clean_ocr_and_code_artifacts(text: str) -> str:
    """Strip OCR parsing debris, code markers (e.g. 7718ц9, +!1!#), brackets, and stray noise."""
    if not text:
        return ""
    t = text.strip()
    # Replace code brackets, markup, and standalone junk symbols with spaces
    t = re.sub(r"[{}\[\]<>#~°^|+=_\\/`]+", " ", t)
    # Remove alphanumeric parsing hashes/codes e.g. 7718ц9, 47D2, F7, nr
    t = re.sub(r"\b(?:\d+[A-Za-zА-Яа-яЁё]+|[A-Za-zА-Яа-яЁё]+\d+)[A-Za-zА-Яа-яЁё\d]*\b", "", t)
    # Remove punctuation-digit codes e.g. !1! or +!1!#
    t = re.sub(r"(?:^|\s)[!?\-_+*]+(?:\d+[!?\-_+*]*)+(?:\s|$)", " ", t)
    # Remove isolated 1-2 Latin letter artifacts inside Cyrillic text
    t = re.sub(r"(?<=\s)[A-Za-z]{1,2}(?=\s|$)", "", t)
    t = re.sub(r"^[A-Za-z]{1,2}(?=\s)", "", t)
    # Strip boundary punctuation/debris
    t = re.sub(r"^\s*[:;,\.\-–—\s]+", "", t)
    t = re.sub(r"[:;,\-–—\s]+$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_russian_translation(original: str, translated: str) -> bool:
    """Reject empty, copied-source, non-Russian, or mostly unparsed output."""
    source = (original or "").strip()
    result = (translated or "").strip()
    if not result or result.casefold() == source.casefold():
        return False
    if _MOJIBAKE.search(result):
        return False

    cyrillic = len(_CYRILLIC.findall(result))
    letters = len(_LETTERS.findall(result))
    if cyrillic == 0:
        return False
    # Very short replies (e.g. «Да», «О!») are valid with one or two letters.
    if letters <= 3:
        return cyrillic >= 1
    latin = len(_LATIN.findall(result))
    if cyrillic / letters < 0.72 or latin > max(2, int(cyrillic * 0.12)):
        return False

    digits = sum(char.isdigit() for char in result)
    if digits >= 4 and digits / max(1, len(result)) > 0.35:
        return False
    # A useful translation needs words; reject OCR debris such as «___ 7 )».
    words = re.findall(r"[А-Яа-яЁё]+", result)
    return bool(words) and (len(words) > 1 or len(words[0]) >= 2)


def is_usable_ocr_text(text: str) -> bool:
    """Reject mostly punctuation/number noise before sending OCR to translation."""
    value = (text or "").strip()
    if not value or _MOJIBAKE.search(value):
        return False
    letters = len(_LETTERS.findall(value))
    if letters < 2:
        return False
    meaningful = sum(char.isalnum() for char in value)
    if meaningful / max(1, len(value)) < 0.38:
        return False
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", value)
    digits = sum(char.isdigit() for char in value)
    if len(value) > 8 and digits / max(1, meaningful) >= 0.18:
        return False
    multi_letter = sum(len(token) >= 2 for token in tokens)
    if len(tokens) >= 4 and multi_letter / len(tokens) < 0.35:
        return False
    if len(value) > 8 and multi_letter < 2:
        return False
    if re.search(r"\b(?:chapitre|chapter|volume)\s*\d+", value, re.IGNORECASE):
        return False
    if any(unicodedata.category(char) == "Cc" for char in value):
        return False
    return True
