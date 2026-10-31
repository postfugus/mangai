"""Narrative analysis, Deepgram chapter TTS, and local word alignment."""

import difflib
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import time
import unicodedata
import wave
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-hector-fr")
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_MALE_VOICE = os.getenv("GEMINI_MALE_VOICE", "Puck")
GEMINI_FRENCH_MALE_VOICE = os.getenv("GEMINI_FRENCH_MALE_VOICE", "fr-fr-podcaster-2")
GEMINI_CHUNK_MAX_WORDS = 2050  # ~14 minutes at 145 words/min, below the reported ~16 min ceiling.
GEMINI_RETRIES = 3
DEEPGRAM_MAX_CHARS = 2000
DEEPGRAM_SAFE_CHARS = 1990
DEEPGRAM_RETRIES = 4
JEV_MAX_BUBBLES_PER_REQUEST = 48
# Jev 1.14 limits each question to 32k tokens and the whole request to 64k.
# UTF-8 bytes are a conservative upper bound for byte-BPE token count, so stay
# well below both context ceilings even when French/Russian OCR has odd symbols.
JEV_STATE_BUDGET_BYTES = 28 * 1024
JEV_REQUEST_BUDGET_BYTES = 56 * 1024
EMOTION_OPTIONS = {
    "neutral": "No marked emotion",
    "joy": "Joy or excitement",
    "sadness": "Sadness or despair",
    "anger": "Anger or irritation",
    "fear": "Fear or anxiety",
    "surprise": "Surprise or shock",
    "tenderness": "Warmth or compassion",
    "sarcasm": "Sarcasm or irony",
    "whisper": "Whisper or secrecy",
}


def chapter_fingerprint(items: List[Dict]) -> str:
    stable = [{"id": item["id"], "text": item["original"].strip(),
               "translation": item.get("natural", ""), "box": item["box"]} for item in items]
    return hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def geometry_key(item: Dict) -> Tuple[float, float, float]:
    box = item["box"]
    y = float(box.get("y", 0.0))
    x = float(box.get("x", 0.0))
    # Manga reading order: panels and text progress strictly from top to bottom (y),
    # and within the same horizontal band/panel from right to left (-x).
    return (int(item["page_idx"]), round(y / 7.5), -x)


def order_chapter_items(items: List[Dict], min_confidence: float = 0.55) -> List[Dict]:
    """Keep page order fixed and topologically apply confident Jev pair choices."""
    pages: Dict[int, List[Dict]] = {}
    for item in items:
        pages.setdefault(int(item["page_idx"]), []).append(item)
    output = []
    for page_idx in sorted(pages):
        page_items = sorted(pages[page_idx], key=geometry_key)
        by_id = {item["id"]: item for item in page_items}
        incoming = {item["id"]: set() for item in page_items}
        outgoing = {item["id"]: set() for item in page_items}
        for left in page_items:
            for right_id, vote in (left.get("pairwise_order") or {}).items():
                if right_id not in by_id or float(vote.get("confidence", 0.0) or 0.0) < min_confidence:
                    continue
                choice = vote.get("choice")
                if choice == "left_before_right":
                    before, after = left["id"], right_id
                elif choice == "right_before_left":
                    before, after = right_id, left["id"]
                else:
                    continue
                if after not in outgoing[before]:
                    outgoing[before].add(after)
                    incoming[after].add(before)
        positions = {item["id"]: index for index, item in enumerate(page_items)}
        remaining = set(by_id)
        ordered_page = []
        while remaining:
            available = [item_id for item_id in remaining
                         if not (incoming[item_id] & remaining)]
            if not available:
                # Contradictory strong judgments should never scramble a page.
                ordered_page = page_items
                break
            item_id = min(available, key=positions.get)
            ordered_page.append(by_id[item_id])
            remaining.remove(item_id)
        output.extend(ordered_page)
    return output


def has_pairwise_order_analysis(items: List[Dict]) -> bool:
    """Return whether each adjacent same-page scan pair has a Jev decision."""
    pages: Dict[int, List[Dict]] = {}
    for item in items:
        pages.setdefault(int(item["page_idx"]), []).append(item)
    for page_items in pages.values():
        page_items.sort(key=geometry_key)
        for left, right in zip(page_items, page_items[1:]):
            if right["id"] not in (left.get("pairwise_order") or {}):
                return False
    return True


def prepare_french_speech(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    letters = [char for char in clean if char.isalpha()]
    if letters and sum(char.isupper() for char in letters) / len(letters) >= 0.6:
        clean = clean.lower()
        clean = re.sub(r"[a-zà-öø-ÿ]", lambda match: match.group(0).upper(), clean, count=1)
    return clean


def _build_jev_payload(items: List[Dict], batch: List[Dict], spatial_rank: Dict[str, int],
                       adjacent: Dict[str, Optional[str]], context_chars: int) -> Tuple[bytes, int]:
    state = {
        "schema": "chapter_dialogue_rows=[bubble_id,page,scan_position,French_text]; target_russian_translations=[bubble_id,Russian_text]",
        "chapter_dialogue_rows": [
            [item["id"], int(item["page_idx"]) + 1, spatial_rank[item["id"]],
             prepare_french_speech(item.get("original", ""))[:context_chars]]
            for item in items
        ],
        "target_russian_translations": [[item["id"], item.get("natural", "")[:300]] for item in batch],
    }
    questions = {}
    for item in batch:
        bubble_id = item["id"]
        questions[f"emotion_{bubble_id}"] = {
            "type": "choice",
            "instructions": f"Choose bubble {bubble_id}'s strongest spoken emotion from its French and Russian text; ignore punctuation alone.",
            "criteria": EMOTION_OPTIONS,
        }
        right_id = adjacent.get(bubble_id)
        if right_id:
            questions[f"compare_{bubble_id}__{right_id}"] = {
                "type": "choice",
                "instructions": (
                    f"For bubbles {bubble_id} and {right_id} on the same page, which should be heard first? "
                    "Use question-answer links, references, sentence continuation and speaker turns. "
                    "Use scan order only if dialogue does not clearly determine the order."
                ),
                "criteria": {
                    "left_before_right": f"Bubble {bubble_id} comes before {right_id}.",
                    "right_before_left": f"Bubble {right_id} comes before {bubble_id}.",
                    "uncertain": "The dialogue does not establish an order; keep spatial order.",
                },
            }
    body = json.dumps({"model": "jev-latest", "state": state, "questions": questions},
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    state_bytes = len(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return body, state_bytes


def analyze_chapter_with_jev(items: List[Dict], api_key: str) -> List[Dict]:
    """Compare neighboring dialogue bubbles and classify emotions in bounded batches."""
    if not api_key:
        raise RuntimeError("Не задан TYPESAFE_API_KEY для Jev.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Full chapter context is French-only; translations are supplied only for the
    # batch being judged. This avoids repeating a second full copy in every call.
    page_groups = {}
    for item in items:
        page_groups.setdefault(int(item["page_idx"]), []).append(item)
    spatial_rank = {}
    for page, page_items in page_groups.items():
        on_page = sorted(page_items, key=geometry_key)
        spatial_rank.update({item["id"]: rank + 1 for rank, item in enumerate(on_page)})
    context_order = sorted(items, key=lambda item: (int(item["page_idx"]), spatial_rank[item["id"]]))
    adjacent = {}
    for page_items in page_groups.values():
        on_page = sorted(page_items, key=geometry_key)
        adjacent.update({left["id"]: right["id"] for left, right in zip(on_page, on_page[1:])})

    def evaluate(batch: List[Dict], start_index: int, context_chars: int = 180):
        body, state_bytes = _build_jev_payload(context_order, batch, spatial_rank, adjacent, context_chars)
        if state_bytes > JEV_STATE_BUDGET_BYTES or len(body) > JEV_REQUEST_BUDGET_BYTES:
            if len(batch) > 1:
                middle = len(batch) // 2
                evaluate(batch[:middle], start_index, context_chars)
                evaluate(batch[middle:], start_index + middle, context_chars)
                return
            if context_chars > 32:
                evaluate(batch, start_index, max(32, context_chars // 2))
                return
            raise RuntimeError("Контекст главы не помещается в безопасный лимит Jev даже после сокращения.")

        response = requests.post(JEV_URL, data=body, headers=headers, timeout=120)
        if not response.ok:
            error_text = " ".join(response.text.split())[:500]
            if "max_tokens_exceeded" in error_text.lower():
                if len(batch) > 1:
                    middle = len(batch) // 2
                    evaluate(batch[:middle], start_index, context_chars)
                    evaluate(batch[middle:], start_index + middle, context_chars)
                    return
                if context_chars > 32:
                    evaluate(batch, start_index, max(32, context_chars // 2))
                    return
            raise RuntimeError(f"Jev не обработал пакет около реплики {start_index + 1} (HTTP {response.status_code}): {error_text}")

        payload = response.json()
        answers = payload.get("answers", {})
        usage = payload.get("usage", {})
        print(f"[Jev] bubbles={len(batch)} request_bytes={len(body)} state_bytes={state_bytes} input_tokens={usage.get('input_tokens', 'n/a')}")
        for item in batch:
            emotion = answers.get(f"emotion_{item['id']}", {}).get("choice", "neutral")
            item["emotion"] = emotion if emotion in EMOTION_OPTIONS else "neutral"
            right_id = adjacent.get(item["id"])
            if right_id:
                answer = answers.get(f"compare_{item['id']}__{right_id}", {})
                choice = answer.get("choice", "uncertain")
                if choice not in {"left_before_right", "right_before_left", "uncertain"}:
                    choice = "uncertain"
                item.setdefault("pairwise_order", {})[right_id] = {
                    "choice": choice,
                    "confidence": float(answer.get("confidence", 0.0) or 0.0),
                }

    def largest_safe_batch(start: int) -> Tuple[int, int]:
        high = min(len(items) - start, JEV_MAX_BUBBLES_PER_REQUEST)
        low, best_size, best_context = 1, 0, 0
        while low <= high:
            candidate_size = (low + high) // 2
            candidate = items[start:start + candidate_size]
            context_chars = 180
            body, state_bytes = _build_jev_payload(context_order, candidate, spatial_rank, adjacent, context_chars)
            while state_bytes > JEV_STATE_BUDGET_BYTES and context_chars > 32:
                context_chars = max(32, context_chars // 2)
                body, state_bytes = _build_jev_payload(context_order, candidate, spatial_rank, adjacent, context_chars)
            safe = state_bytes <= JEV_STATE_BUDGET_BYTES and len(body) <= JEV_REQUEST_BUDGET_BYTES
            if safe:
                best_size, best_context = candidate_size, context_chars
                low = candidate_size + 1
            else:
                high = candidate_size - 1
        return best_size, best_context

    start = 0
    while start < len(items):
        batch_size, context_chars = largest_safe_batch(start)
        if not batch_size:
            raise RuntimeError("Даже одна реплика не помещается в безопасный контекст Jev.")
        evaluate(items[start:start + batch_size], start, context_chars)
        start += batch_size
    return order_chapter_items(items)


def pack_deepgram_chunks(items: List[Dict], max_chars: int = DEEPGRAM_SAFE_CHARS) -> List[str]:
    """Pack whole bubble lines under Aura's 2,000-character hard input limit."""
    chunks, current = [], ""
    for item in items:
        text = prepare_french_speech(item.get("original", ""))
        if not text:
            continue
        if len(text) > max_chars:
            raise RuntimeError(f"Реплика {item.get('id', '?')} длиннее безопасного лимита Deepgram ({max_chars} символов).")
        candidate = f"{current}\n{text}" if current else text
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = text
        else:
            current = candidate
    if current:
        chunks.append(current)
    if any(len(chunk) > DEEPGRAM_MAX_CHARS for chunk in chunks):
        raise AssertionError("Упаковщик сформировал запрос длиннее документированного лимита Deepgram.")
    return chunks


def pack_gemini_chapter_chunks(items: List[Dict], max_words: int = GEMINI_CHUNK_MAX_WORDS) -> List[List[Dict]]:
    """Keep one Gemini request below the user's ~16 minute output ceiling."""
    chunks, current, words = [], [], 0
    for item in items:
        count = max(1, len(normalize_words(prepare_french_speech(item.get("original", "")))))
        if current and words + count > max_words:
            chunks.append(current)
            current, words = [], 0
        current.append(item)
        words += count
    if current:
        chunks.append(current)
    return chunks


def _wav_from_gemini_result(result: Dict) -> bytes:
    encoded = [block["data"] for step in result.get("steps", [])
               for block in step.get("content", [])
               if block.get("type") == "audio" and block.get("data")]
    if not encoded and result.get("output_audio", {}).get("data"):
        encoded = [result["output_audio"]["data"]]
    if not encoded:
        raise RuntimeError("Gemini завершил запрос без аудиодорожки.")
    return _combine_wav_chunks([base64.b64decode(value) for value in encoded], pause_ms=90)


def get_voice_for_lang(lang: str) -> Tuple[str, str]:
    """Returns (voice_name, prompt_language_name) for Gemini male voiceover."""
    l = (lang or "ru").lower()
    if l.startswith("fr"):
        return GEMINI_FRENCH_MALE_VOICE, "French"
    elif l.startswith("ru"):
        return GEMINI_MALE_VOICE, "Russian"
    elif l.startswith("de"):
        return GEMINI_MALE_VOICE, "German"
    elif l.startswith("it"):
        return GEMINI_MALE_VOICE, "Italian"
    elif l.startswith("en"):
        return GEMINI_MALE_VOICE, "English"
    return GEMINI_MALE_VOICE, "Russian"


def prepare_item_speech(item: Dict, lang: str = "ru") -> str:
    """Extract and sanitize speech text for the specified language."""
    l = (lang or "ru").lower()
    if l.startswith("ru"):
        text = (item.get("natural") or item.get("translated") or item.get("original") or "").strip()
    else:
        text = (item.get("original") or item.get("natural") or item.get("translated") or "").strip()
    return prepare_french_speech(text)


def synthesize_chapter_gemini(
    items: List[Dict], api_key: str, model: str = GEMINI_TTS_MODEL,
    voice: Optional[str] = None,
    lang: str = "fr",
    load_chunk: Optional[Callable[[int, str], Optional[bytes]]] = None,
    save_chunk: Optional[Callable[[int, str, bytes], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[bytes, str]:
    """Generate offline audiobook audio with Gemini male voice across languages, caching each bounded request."""
    if not api_key:
        raise RuntimeError("Не задан GEMINI_API_KEY для Gemini TTS.")
    default_voice, lang_name = get_voice_for_lang(lang)
    target_voice = voice or default_voice
    requests_for_chapter = pack_gemini_chapter_chunks(items)
    outputs = []
    for index, batch in enumerate(requests_for_chapter):
        text = "\n".join(prepare_item_speech(item, lang=lang) for item in batch)
        digest = hashlib.sha256(f"{model}\0{target_voice}\0{lang}\0{text}".encode("utf-8")).hexdigest()
        cached = load_chunk(index, digest) if load_chunk else None
        if cached:
            audio = cached
        else:
            style_instruction = f"Natural audiobook narration in {lang_name}; keep a steady adult male voice, finish every word, and let the final syllable decay naturally."
            content = [{
                "type": "text", "text": prepare_item_speech(item, lang=lang),
                "annotations": [{"type": "speech_metadata", "style": style_instruction}],
            } for item in batch if prepare_item_speech(item, lang=lang)]
            body = {
                "model": model,
                "input": [{"type": "user_input", "content": content}],
                "response_format": {"type": "audio", "mime_type": "audio/wav"},
                "generation_config": {"speech_config": [{"voice": target_voice}]},
            }
            response = None
            for attempt in range(GEMINI_RETRIES):
                try:
                    response = requests.post(
                        GEMINI_TTS_URL, headers={"x-goog-api-key": api_key}, json=body,
                        timeout=(20, 1200))
                except requests.RequestException as error:
                    if attempt + 1 == GEMINI_RETRIES:
                        raise RuntimeError(f"Gemini TTS: ошибка соединения после {GEMINI_RETRIES} попыток: {error}") from error
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if response.ok:
                    break
                if response.status_code not in {429, 500, 502, 503, 504} or attempt + 1 == GEMINI_RETRIES:
                    message = " ".join(response.text.split())[:350]
                    raise RuntimeError(f"Gemini TTS вернул HTTP {response.status_code}: {message}")
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(max(float(retry_after), 0.5), 60.0) if retry_after else min(2 ** attempt, 8)
                except ValueError:
                    delay = min(2 ** attempt, 8)
                time.sleep(delay)
            if response is None or not response.ok:
                raise RuntimeError("Gemini TTS не вернул аудио после повторных попыток.")
            audio = _wav_from_gemini_result(response.json())
            if save_chunk:
                save_chunk(index, digest, audio)
        outputs.append(audio)
        if on_progress:
            on_progress(index + 1, len(requests_for_chapter))
    return _combine_wav_chunks(outputs, pause_ms=240), "audio/wav"


def _combine_wav_chunks(audio_blocks: List[bytes], pause_ms: int = 180) -> bytes:
    """Join same-format linear16 WAV blocks with a short silent inter-chunk pause."""
    if not audio_blocks:
        raise RuntimeError("Deepgram не вернул аудиоблоки.")
    if len(audio_blocks) == 1:
        return audio_blocks[0]
    params = None
    frames = []
    for block in audio_blocks:
        try:
            with wave.open(io.BytesIO(block), "rb") as wav_file:
                current = wav_file.getparams()
                if params is None:
                    params = current
                elif (current.nchannels, current.sampwidth, current.framerate, current.comptype) != (
                    params.nchannels, params.sampwidth, params.framerate, params.comptype
                ):
                    raise RuntimeError("Deepgram вернул WAV-блоки с разными форматами; безопасная склейка невозможна.")
                frames.append(wav_file.readframes(wav_file.getnframes()))
        except (wave.Error, EOFError) as error:
            raise RuntimeError("Deepgram вернул некорректный WAV-блок.") from error
    pause = b"\x00" * int(params.framerate * pause_ms / 1000) * params.nchannels * params.sampwidth
    output = io.BytesIO()
    with wave.open(output, "wb") as combined:
        combined.setnchannels(params.nchannels)
        combined.setsampwidth(params.sampwidth)
        combined.setframerate(params.framerate)
        for index, pcm in enumerate(frames):
            if index:
                combined.writeframesraw(pause)
            combined.writeframesraw(pcm)
    return output.getvalue()


def synthesize_chapter_deepgram(
    items: List[Dict], api_key: str, model: str = DEEPGRAM_TTS_MODEL,
    load_chunk: Optional[Callable[[int, str], Optional[bytes]]] = None,
    save_chunk: Optional[Callable[[int, str, bytes], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[bytes, str]:
    if not api_key:
        raise RuntimeError("Не задан DEEPGRAM_API_KEY для Deepgram TTS.")
    chunks = pack_deepgram_chunks(items)
    audio_blocks = []
    for index, text in enumerate(chunks):
        digest = hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()
        cached = load_chunk(index, digest) if load_chunk else None
        if cached:
            audio = cached
        else:
            response = None
            for attempt in range(DEEPGRAM_RETRIES):
                try:
                    response = requests.post(
                        DEEPGRAM_TTS_URL,
                        params={"model": model, "encoding": "linear16", "container": "wav"},
                        headers={"Authorization": f"Token {api_key}"},
                        json={"text": text}, timeout=(20, 240),
                    )
                except requests.RequestException as error:
                    if attempt + 1 == DEEPGRAM_RETRIES:
                        raise RuntimeError(f"Deepgram TTS: ошибка соединения после {DEEPGRAM_RETRIES} попыток: {error}") from error
                    time.sleep(min(2 ** attempt, 12))
                    continue
                if response.ok:
                    break
                if response.status_code not in {429, 500, 502, 503, 504} or attempt + 1 == DEEPGRAM_RETRIES:
                    message = " ".join(response.text.split())[:350]
                    raise RuntimeError(f"Deepgram TTS вернул HTTP {response.status_code}: {message}")
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(max(float(retry_after), 0.5), 30.0) if retry_after else min(2 ** attempt, 12)
                except ValueError:
                    delay = min(2 ** attempt, 12)
                time.sleep(delay)
            if response is None or not response.ok:
                raise RuntimeError("Deepgram TTS не вернул WAV после повторных попыток.")
            audio = response.content
            if not audio.startswith(b"RIFF"):
                raise RuntimeError("Deepgram TTS ответил без WAV-аудио.")
            if save_chunk:
                save_chunk(index, digest, audio)
        audio_blocks.append(audio)
        if on_progress:
            on_progress(index + 1, len(chunks))
    return _combine_wav_chunks(audio_blocks), "audio/wav"


def normalize_words(text: str) -> List[str]:
    text = (text or "").casefold().replace("’", "'").replace("œ", "oe").replace("æ", "ae").replace("ё", "е")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Support all alphabets: Cyrillic, Latin, Greek, Kanji, numbers, etc.
    return re.findall(r"[\w]+", text)


def _token_sequence_similarity(target: List[str], sample: List[str]) -> float:
    """Fuzzy ordered token alignment for common OCR/Whisper spelling variants."""
    if not target or not sample:
        return 0.0
    rows, cols = len(target) + 1, len(sample) + 1
    costs = [[0.0] * cols for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = float(row)
    for col in range(1, cols):
        costs[0][col] = float(col)
    for row in range(1, rows):
        for col in range(1, cols):
            similarity = difflib.SequenceMatcher(
                a=target[row - 1], b=sample[col - 1], autojunk=False).ratio()
            can_fuzzy_match = min(len(target[row - 1]), len(sample[col - 1])) >= 4
            if target[row - 1] == sample[col - 1]:
                substitution = 0.0
            else:
                substitution = 1.0 - similarity if can_fuzzy_match and similarity >= 0.45 else 1.0
            costs[row][col] = min(
                costs[row - 1][col] + 1.0,
                costs[row][col - 1] + 1.0,
                costs[row - 1][col - 1] + substitution,
            )
    return max(0.0, 1.0 - costs[-1][-1] / max(len(target), len(sample)))


def _audio_rms_envelope(audio: bytes, frame_ms: int = 10):
    """Return a 10 ms RMS envelope for PCM WAV; used to retain the audible phrase tail."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
        if width != 2 or not raw:
            return None, None
        samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768.0
        frame_samples = max(1, round(sample_rate * frame_ms / 1000))
        remainder = len(samples) % frame_samples
        if remainder:
            samples = np.pad(samples, ((0, frame_samples - remainder), (0, 0)))
        framed = samples.reshape(-1, frame_samples, channels)
        rms = np.sqrt(np.mean(np.square(framed), axis=(1, 2)))
        return rms, frame_samples / sample_rate
    except (wave.Error, ValueError, ZeroDivisionError):
        return None, None


def _refine_phrase_boundaries(raw_start: float, raw_end: float, next_start: float,
                              duration: float, rms, frame_seconds: Optional[float]):
    """Use the waveform envelope to include the complete onset and audible decay.

    Whisper word times are anchors rather than hard cut points. Find the low-energy
    gap around each anchor, retaining voiced consonants and the breathy tail.
    """
    raw_start = min(duration, max(0.0, raw_start))
    raw_end = min(duration, max(raw_start, raw_end))
    if rms is None or not frame_seconds or len(rms) == 0:
        return max(0.0, raw_start - 0.08), min(duration, raw_end + 0.35)

    frame_count = len(rms)
    anchor_start = min(frame_count - 1, max(0, int(raw_start / frame_seconds)))
    anchor_end = min(frame_count - 1, max(anchor_start, int(raw_end / frame_seconds)))

    # A short pre-roll scan catches weak French consonants that Whisper places late.
    onset_left = max(0, anchor_start - int(0.18 / frame_seconds))
    onset_right = min(frame_count, anchor_start + int(0.12 / frame_seconds) + 1)
    onset_signal = rms[onset_left:onset_right]
    onset_peak = float(np.max(onset_signal)) if len(onset_signal) else 0.0
    onset_floor = float(np.percentile(onset_signal, 20)) if len(onset_signal) else 0.0
    onset_threshold = max(onset_floor * 1.15, onset_peak * 0.006, 0.00018)
    quiet_run = max(1, int(0.045 / frame_seconds))
    phrase_start = max(0, anchor_start - int(0.025 / frame_seconds))
    # Find the closest sustained quiet stretch before the Whisper start. The first
    # energetic frame after it is the acoustic onset; keep a small consonant pre-roll.
    for frame in range(anchor_start, onset_left + quiet_run - 1, -1):
        left = max(onset_left, frame - quiet_run)
        if frame <= anchor_start and np.all(rms[left:frame] <= onset_threshold):
            active = frame
            while active < onset_right and rms[active] <= onset_threshold:
                active += 1
            phrase_start = max(0, active * frame_seconds - 0.025)
            break
    else:
        active_candidates = np.flatnonzero(onset_signal > onset_threshold)
        if len(active_candidates):
            phrase_start = max(0, (onset_left + int(active_candidates[0])) * frame_seconds - 0.025)

    # Search independently of the next Whisper timestamp: it may be early too.
    tail_right = min(frame_count, anchor_end + int(1.6 / frame_seconds) + 1)
    tail_signal = rms[max(0, anchor_end - int(0.20 / frame_seconds)):tail_right]
    tail_peak = float(np.max(tail_signal)) if len(tail_signal) else 0.0
    tail_floor = float(np.percentile(tail_signal, 20)) if len(tail_signal) else 0.0
    tail_threshold = max(tail_floor * 1.35, tail_peak * 0.008, 0.00022)
    quiet_frames = max(1, int(0.12 / frame_seconds))
    phrase_end = min(duration, raw_end + 0.45)
    found_quiet = False
    for frame in range(anchor_end + 1, max(anchor_end + 1, tail_right - quiet_frames + 1)):
        if np.all(rms[frame:frame + quiet_frames] <= tail_threshold):
            # Include the decay and a little room tone; don't end on the first quiet
            # sample. This also avoids a hard click when the browser stops the segment.
            phrase_end = min(duration, (frame + max(1, int(0.10 / frame_seconds))) * frame_seconds)
            found_quiet = True
            break
    if not found_quiet:
        # If there is no quiet gap, stop just before the next recognized onset only
        # when it is clearly after this word; otherwise favor completing this phrase.
        if next_start > raw_end + 0.10:
            phrase_end = min(phrase_end, max(raw_end + 0.12, next_start - 0.025))
    return min(raw_start, phrase_start), max(raw_end, phrase_end)


def align_bubbles_to_words(items: List[Dict], words: List[Dict], duration: float,
                           rms=None, frame_seconds: Optional[float] = None,
                           lang: str = "ru") -> Dict[str, Dict[str, float]]:
    """Align bubbles to Whisper words, then refine cuts against the actual waveform."""
    if duration <= 0 or not items:
        raise RuntimeError("Аудиоглава не содержит звука или диалоговых реплик.")

    # Если Whisper не распознал слова или не установлен, производим прямое
    # взвешенное выравнивание по огибающей RMS и объёму фраз
    if not words:
        words = []

    transcript = []
    for word in words:
        tokens = normalize_words(word.get("text", ""))
        for token in tokens:
            transcript.append((token, float(word["start"]), float(word["end"])))
    cursor = 0
    aligned = [None] * len(items)
    for item_index, item in enumerate(items):
        target = normalize_words(prepare_item_speech(item, lang=lang))
        if not target:
            continue
        best = None
        search_end = min(len(transcript), cursor + max(36, len(target) * 4))
        min_width = max(1, len(target) - max(2, len(target) // 4))
        max_width = len(target) + max(2, len(target) // 4)
        for start in range(cursor, search_end):
            for width in range(min_width, min(max_width, len(transcript) - start) + 1):
                sample = [token[0] for token in transcript[start:start + width]]
                ratio = _token_sequence_similarity(target, sample)
                if best is None or ratio > best[0]:
                    best = (ratio, start, start + width - 1)
        if best and best[0] >= 0.46:
            _, first, last = best
            aligned[item_index] = (max(0.0, transcript[first][1]), min(duration, transcript[last][2]))
            cursor = last + 1

    index = 0
    while index < len(items):
        if aligned[index] is not None:
            index += 1
            continue
        gap_start = index
        while index < len(items) and aligned[index] is None:
            index += 1
        gap_end = index
        left = aligned[gap_start - 1][1] if gap_start else 0.0
        right = aligned[gap_end][0] if gap_end < len(items) else duration
        right = max(left, right)
        weights = [max(1, len(normalize_words(prepare_item_speech(items[i], lang=lang)))) for i in range(gap_start, gap_end)]
        weight_sum = sum(weights) or 1
        offset = 0.0
        for i, weight in zip(range(gap_start, gap_end), weights):
            start = left + (right - left) * offset / weight_sum
            offset += weight
            end = left + (right - left) * offset / weight_sum
            aligned[i] = (start, max(start + 0.06, end))

    result = {}
    raw_times = [timing or (0.0, duration) for timing in aligned]
    refined = []
    for index, (item, timing) in enumerate(zip(items, raw_times)):
        start, raw_end = timing
        next_start = next((candidate[0] for candidate in raw_times[index + 1:]
                           if candidate[0] > start + 0.08), duration)
        start, end = _refine_phrase_boundaries(start, raw_end, next_start, duration, rms, frame_seconds)
        refined.append((start, end))
    previous_start = 0.0
    for index, (item, timing) in enumerate(zip(items, raw_times)):
        start, end = refined[index]
        start = max(previous_start, start)
        next_start = next((candidate[0] for candidate in refined[index + 1:]
                           if candidate[0] > start + 0.04), duration)
        if end > next_start:
            # Whisper can place adjacent word times on top of one another. Stop
            # this bubble at the next acoustically refined onset, never inside a
            # later bubble merely because that bubble's word match began early.
            end = max(start + 0.06, next_start - 0.015)
        end = min(duration, max(start + 0.06, end))
        result[item["id"]] = {"start": round(start, 3), "end": round(end, 3)}
        previous_start = start
    return result


@lru_cache(maxsize=1)
def _whisper_model(model_name: str):
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except ImportError:
        return None


def transcribe_and_align(audio: bytes, items: List[Dict], lang: str = "ru", model_name: str = "tiny") -> Tuple[Dict, float]:
    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        duration = wav_file.getnframes() / float(wav_file.getframerate())
    rms, frame_seconds = _audio_rms_envelope(audio)

    model = _whisper_model(model_name)
    words = []
    whisper_lang = (lang or "ru")[:2].lower()
    if model is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
                audio_file.write(audio)
                audio_file.flush()
                segments, _ = model.transcribe(audio_file.name, language=whisper_lang, beam_size=1,
                                               word_timestamps=True, vad_filter=True,
                                               condition_on_previous_text=False)
                for segment in segments:
                    for word in segment.words or []:
                        words.append({"text": word.word, "start": word.start, "end": word.end})
                    if not segment.words:
                        words.append({"text": segment.text, "start": segment.start, "end": segment.end})
        except Exception as e:
            print(f"[!] Whisper transcription skipped, falling back to proportional alignment: {e}")
            words = []
    else:
        print("[*] faster-whisper is not installed. Using acoustic RMS & proportional alignment.")

    return align_bubbles_to_words(items, words, duration, rms, frame_seconds, lang=lang), duration

def mp3_to_wav_bytes(mp3_bytes: bytes) -> bytes:
    """Converts MP3 audio bytes to linear PCM WAV bytes using miniaudio."""
    import miniaudio, wave, io
    decoded = miniaudio.mp3_read_s16(mp3_bytes)
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav:
        wav.setnchannels(decoded.nchannels)
        wav.setsampwidth(2)
        wav.setframerate(decoded.sample_rate)
        wav.writeframes(decoded.samples)
    return wav_io.getvalue()


def synthesize_bubble_multilingual(text: str, lang: str = 'ru', api_key: str = '', gemini_api_key: str = '') -> Tuple[bytes, str]:
    """
    Instant single bubble voiceover with Gemini male voice as primary, falling back to Deepgram/neural TTS.
    Supports 'ru', 'de', 'it', 'fr', 'en'.
    """
    clean = text.strip()
    if not clean:
        raise ValueError("Empty text")

    lang = (lang or 'ru').lower()
    g_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    # 1. Try Gemini TTS with male voice
    if g_key:
        voice, lang_name = get_voice_for_lang(lang)
        body = {
            "model": GEMINI_TTS_MODEL,
            "input": [{"type": "user_input", "content": [{
                "type": "text", "text": clean,
                "annotations": [{"type": "speech_metadata", "style": f"Natural narration in {lang_name}; steady adult male voice."}],
            }]}],
            "response_format": {"type": "audio", "mime_type": "audio/wav"},
            "generation_config": {"speech_config": [{"voice": voice}]},
        }
        try:
            res = requests.post(
                GEMINI_TTS_URL, headers={"x-goog-api-key": g_key}, json=body, timeout=(5, 30))
            if res.ok:
                audio = _wav_from_gemini_result(res.json())
                return audio, "audio/wav"
        except Exception:
            pass

    # 2. Try French Deepgram fallback if key provided
    if lang == 'fr' and api_key:
        try:
            res = requests.post(
                "https://api.deepgram.com/v1/speak",
                params={"model": "aura-2-hector-fr", "encoding": "linear16", "container": "wav"},
                headers={"Authorization": f"Token {api_key}"}, json={"text": clean}, timeout=(10, 60))
            if res.ok and res.content.startswith(b"RIFF"):
                return res.content, "audio/wav"
        except Exception:
            pass

    # 3. Neural gTTS fallback
    from gtts import gTTS
    import io
    gtts_lang = 'ru'
    if lang.startswith('de'):
        gtts_lang = 'de'
    elif lang.startswith('it'):
        gtts_lang = 'it'
    elif lang.startswith('fr'):
        gtts_lang = 'fr'
    elif lang.startswith('en'):
        gtts_lang = 'en'
    elif lang.startswith('ru'):
        gtts_lang = 'ru'
    else:
        gtts_lang = lang[:2]

    tts = gTTS(text=clean, lang=gtts_lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    mp3_data = fp.getvalue()
    wav_data = mp3_to_wav_bytes(mp3_data)
    return wav_data, "audio/wav"


def synthesize_chapter_multilingual(
    items: List[Dict],
    lang: str = 'ru',
    api_key: str = '',
    on_progress: Optional[Callable[[int, int], None]] = None,
    load_chunk: Optional[Callable[[int, str], Optional[bytes]]] = None,
    save_chunk: Optional[Callable[[int, str, bytes], None]] = None,
) -> Tuple[bytes, Dict[str, Dict[str, float]], float, str, str]:
    """
    Synthesizes a full chapter with speech for all speech bubbles in the requested language.
    Primary engine: Gemini TTS adult male voice with chunk-level caching.
    Fallback: individual bubble synthesis + concatenation if Gemini API quota/network is exhausted.
    Returns: (audio_bytes, segments_dict, duration, voice_label, model_label)
    """
    lang = (lang or 'ru').lower()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    target_voice, lang_name = get_voice_for_lang(lang)

    # Attempt unified Gemini chapter synthesis first
    if gemini_key:
        try:
            audio_wav, _ = synthesize_chapter_gemini(
                items, gemini_key, model=GEMINI_TTS_MODEL,
                voice=target_voice, lang=lang,
                load_chunk=load_chunk, save_chunk=save_chunk,
                on_progress=on_progress
            )
            # Perform word/acoustic boundary alignment with Whisper
            segments, duration = transcribe_and_align(audio_wav, items, lang=lang)
            voice_label = target_voice
            model_label = GEMINI_TTS_MODEL
            return audio_wav, segments, duration, voice_label, model_label
        except Exception as gemini_err:
            print(f"[!] Gemini chapter TTS unavailable ({gemini_err}), using robust fallback synthesis...")

    # Fallback bubble-by-bubble synthesis (guarantees completion regardless of quota limits)
    audio_blocks = []
    durations = []
    pause_ms = 220
    
    total = len(items)
    for idx, item in enumerate(items):
        text = prepare_item_speech(item, lang=lang)
        if not text:
            text = '...'

        wav_bytes, _ = synthesize_bubble_multilingual(text, lang=lang, api_key=api_key, gemini_api_key=gemini_key)
        audio_blocks.append(wav_bytes)
        with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
            dur = w.getnframes() / float(w.getframerate())
            durations.append(dur)
        if on_progress:
            on_progress(idx + 1, total)

    combined_wav = _combine_wav_chunks(audio_blocks, pause_ms=pause_ms)
    pause_sec = pause_ms / 1000.0
    segments = {}
    current_time = 0.0

    for idx, item in enumerate(items):
        dur = durations[idx]
        start_t = round(current_time, 3)
        end_t = round(current_time + dur, 3)
        segments[item['id']] = {'start': start_t, 'end': end_t}
        current_time += dur + pause_sec

    total_duration = round(current_time - pause_sec if current_time > pause_sec else current_time, 3)
    voice_label = f"Gemini Male / {lang.upper()} Voice"
    model_label = f"gemini-male-tts-{lang}"
    return combined_wav, segments, total_duration, voice_label, model_label
