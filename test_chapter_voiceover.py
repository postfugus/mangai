import io
import json
import os
import tempfile
import unittest
import wave
from unittest.mock import Mock, patch

import db_cache
from db_cache import get_all_cached_chapter_bubbles, save_cached_translation
from chapter_voiceover import (
    DEEPGRAM_MAX_CHARS,
    DEEPGRAM_SAFE_CHARS,
    JEV_REQUEST_BUDGET_BYTES,
    JEV_STATE_BUDGET_BYTES,
    _audio_rms_envelope,
    _combine_wav_chunks,
    align_bubbles_to_words,
    analyze_chapter_with_jev,
    has_pairwise_order_analysis,
    normalize_words,
    order_chapter_items,
    pack_deepgram_chunks,
    synthesize_chapter_deepgram,
)
from text_quality import is_russian_translation, is_usable_ocr_text


def make_wav(sample=120, samples=240, sample_rate=24000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sample.to_bytes(2, "little", signed=True) * samples)
    return output.getvalue()


class ChapterVoiceoverTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"id": "p0-b0", "page_idx": 0, "box": {"x": 10, "y": 5, "w": 20, "h": 6},
             "original": "Bonjour, tu es là ?", "natural": "Привет, ты здесь?"},
            {"id": "p0-b1", "page_idx": 0, "box": {"x": 45, "y": 14, "w": 20, "h": 6},
             "original": "Oui ! Je suis là.", "natural": "Да! Я здесь."},
        ]

    @patch("chapter_voiceover.requests.post")
    def test_jev_orders_and_tags_all_bubbles_in_one_request(self, post):
        post.return_value = Mock(ok=True, json=lambda: {"answers": {
            "compare_p0-b0__p0-b1": {"choice": "right_before_left", "confidence": 0.9},
            "emotion_p0-b0": {"choice": "surprise"}, "emotion_p0-b1": {"choice": "joy"},
        }, "usage": {"input_tokens": 1234}})
        ordered = analyze_chapter_with_jev([dict(item) for item in self.items], "jev-secret")
        self.assertEqual([item["id"] for item in ordered], ["p0-b1", "p0-b0"])
        self.assertEqual([item["emotion"] for item in ordered], ["joy", "surprise"])
        self.assertEqual(post.call_count, 1)
        sent = json.loads(post.call_args.kwargs["data"])
        self.assertIn("compare_p0-b0__p0-b1", sent["questions"])
        self.assertIn("emotion_p0-b1", sent["questions"])
        self.assertIn("target_russian_translations", sent["state"])
        self.assertEqual(len(sent["state"]["chapter_dialogue_rows"][0]), 4)

    def test_jev_pairwise_order_is_confidence_gated_and_never_reorders_pages(self):
        items = [
            {"id": "late-page0", "page_idx": 0, "box": {"x": 10, "y": 5, "h": 6},
             "pairwise_order": {"early-page0": {"choice": "right_before_left", "confidence": 0.9}}},
            {"id": "early-page0", "page_idx": 0, "box": {"x": 40, "y": 20, "h": 6},
             "pairwise_order": {}},
            {"id": "page1", "page_idx": 1, "box": {"x": 5, "y": 1, "h": 6},
             "pairwise_order": {"late-page0": {"choice": "right_before_left", "confidence": 1.0}}},
        ]
        self.assertEqual([x["id"] for x in order_chapter_items(items)],
                         ["early-page0", "late-page0", "page1"])
        uncertain = [dict(x) for x in items]
        uncertain[0]["pairwise_order"] = {"early-page0": {"choice": "right_before_left", "confidence": 0.4}}
        self.assertEqual([x["id"] for x in order_chapter_items(uncertain)],
                         ["late-page0", "early-page0", "page1"])
        self.assertTrue(has_pairwise_order_analysis(items[:2]))
        self.assertTrue(has_pairwise_order_analysis(uncertain[:2]))

    @patch("chapter_voiceover.requests.post")
    def test_jev_splits_max_token_error_without_losing_answers(self, post):
        def response_for(url, data, **kwargs):
            request = json.loads(data)
            if len(request["questions"]) > 2:
                return Mock(ok=False, status_code=400, text='{"error_type":"max_tokens_exceeded"}')
            answers = {}
            for question_id, question in request["questions"].items():
                if question["type"] == "score":
                    answers[question_id] = {"score": 0.5}
                else:
                    answers[question_id] = {"choice": "neutral"}
            return Mock(ok=True, json=lambda: {"answers": answers})
        post.side_effect = response_for
        result = analyze_chapter_with_jev([dict(item) for item in self.items], "jev-secret")
        self.assertEqual(len(result), 2)
        self.assertEqual(post.call_count, 3)

    @patch("chapter_voiceover.requests.post")
    def test_large_jev_chapter_requests_stay_below_context_budgets(self, post):
        items = [{
            "id": f"p{n // 7}-b{n % 7}", "page_idx": n // 7,
            "box": {"x": (n % 7) * 12, "y": (n % 5) * 15, "w": 10, "h": 5},
            "original": f"Bonjour, voici la réplique numéro {n}, et voici encore du contexte français.",
            "natural": f"Привет, это реплика номер {n}, дополнительный контекст для понимания сцены.",
        } for n in range(355)]

        def response_for(url, data, **kwargs):
            request = json.loads(data)
            answers = {}
            for question_id, question in request["questions"].items():
                answers[question_id] = {"score": 0.5} if question["type"] == "score" else {"choice": "neutral"}
            return Mock(ok=True, json=lambda: {"answers": answers})
        post.side_effect = response_for
        result = analyze_chapter_with_jev(items, "jev-secret")
        self.assertEqual(len(result), 355)
        self.assertLessEqual(post.call_count, 12)
        for call in post.call_args_list:
            raw = call.kwargs["data"]
            request = json.loads(raw)
            state_bytes = len(json.dumps(request["state"], ensure_ascii=False, separators=(",", ":")).encode())
            self.assertLessEqual(len(raw), JEV_REQUEST_BUDGET_BYTES)
            self.assertLessEqual(state_bytes, JEV_STATE_BUDGET_BYTES)

    def test_deepgram_packing_respects_documented_hard_character_limit(self):
        items = [dict(self.items[0], original="Bonjour. " * 70) for _ in range(18)]
        chunks = pack_deepgram_chunks(items)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= DEEPGRAM_SAFE_CHARS <= DEEPGRAM_MAX_CHARS for chunk in chunks))
        self.assertEqual(sum(len(chunk.split("\n")) for chunk in chunks), len(items))

    @patch("chapter_voiceover.requests.post")
    def test_deepgram_synthesizes_single_male_french_voice_and_reuses_cached_chunk(self, post):
        audio = make_wav()
        post.return_value = Mock(ok=True, content=audio, status_code=200, headers={}, text="")
        cached = {}
        saved = []

        def load_chunk(index, digest):
            value = cached.get((index, digest))
            return value

        def save_chunk(index, digest, chunk_audio):
            cached[(index, digest)] = chunk_audio
            saved.append(index)

        progress = []
        result, media_type = synthesize_chapter_deepgram(
            self.items, "deepgram-secret", load_chunk=load_chunk,
            save_chunk=save_chunk, on_progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(media_type, "audio/wav")
        self.assertEqual(result, audio)
        self.assertEqual(saved, [0])
        self.assertEqual(progress, [(1, 1)])
        call = post.call_args
        self.assertEqual(call.args[0], "https://api.deepgram.com/v1/speak")
        self.assertEqual(call.kwargs["params"]["model"], "aura-2-hector-fr")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Token deepgram-secret")
        self.assertEqual(call.kwargs["json"]["text"], "Bonjour, tu es là ?\nOui ! Je suis là.")
        second, _ = synthesize_chapter_deepgram(self.items, "deepgram-secret", load_chunk=load_chunk)
        self.assertEqual(second, audio)
        self.assertEqual(post.call_count, 1)

    def test_joined_tts_chunks_have_a_small_pause_without_format_change(self):
        a, b = make_wav(samples=240), make_wav(sample=-120, samples=240)
        joined = _combine_wav_chunks([a, b], pause_ms=90)
        with wave.open(io.BytesIO(joined), "rb") as reader:
            self.assertEqual(reader.getnframes(), 240 + int(24000 * 0.09) + 240)
            self.assertEqual(reader.getframerate(), 24000)

    def test_whisper_word_boundaries_keep_a_natural_speech_tail(self):
        sample_rate = 24000
        samples = bytearray(b"\x00\x00" * sample_rate)
        tone = (900).to_bytes(2, "little", signed=True)
        for index in range(int(0.30 * sample_rate), int(0.50 * sample_rate)):
            samples[index * 2:index * 2 + 2] = tone
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav_file:
            wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples)
        rms, frame_seconds = _audio_rms_envelope(audio.getvalue())
        result = align_bubbles_to_words(
            [dict(self.items[0], original="Bonjour")],
            [{"text": " Bonjour", "start": 0.30, "end": 0.50}], 1.0, rms, frame_seconds)
        self.assertGreaterEqual(result["p0-b0"]["end"], 0.58)
        self.assertLessEqual(result["p0-b0"]["end"], 1.0)
        self.assertLessEqual(result["p0-b0"]["start"], 0.30)

    def test_waveform_refines_both_ends_and_does_not_trust_early_next_whisper_time(self):
        sample_rate = 24000
        samples = bytearray(b"\x00\x00" * sample_rate)
        tone = (1100).to_bytes(2, "little", signed=True)
        # Simulate quiet consonant onset before Whisper's late word start, followed
        # by the phrase tail and room tone after Whisper's early word end.
        for index in range(int(0.20 * sample_rate), int(0.66 * sample_rate)):
            samples[index * 2:index * 2 + 2] = tone
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav_file:
            wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples)
        rms, frame_seconds = _audio_rms_envelope(audio.getvalue())
        result = align_bubbles_to_words(
            [dict(self.items[0], original="Bonjour")],
            [{"text": " Bonjour", "start": 0.30, "end": 0.50}], 1.0, rms, frame_seconds)
        self.assertLess(result["p0-b0"]["start"], 0.30)
        self.assertGreaterEqual(result["p0-b0"]["end"], 0.74)

    def test_whisper_words_map_back_to_each_bubble_and_preserve_tail(self):
        words = [
            {"text": " Bonjour", "start": 0.0, "end": 0.35},
            {"text": " tu", "start": 0.36, "end": 0.5},
            {"text": " es", "start": 0.51, "end": 0.62},
            {"text": " là", "start": 0.63, "end": 0.9},
            {"text": " Oui", "start": 1.2, "end": 1.45},
            {"text": " Je", "start": 1.46, "end": 1.58},
            {"text": " suis", "start": 1.59, "end": 1.8},
            {"text": " là", "start": 1.81, "end": 2.0},
        ]
        result = align_bubbles_to_words(self.items, words, 2.1)
        self.assertLess(result["p0-b0"]["start"], result["p0-b1"]["start"])
        self.assertAlmostEqual(result["p0-b0"]["start"], 0.0)
        self.assertGreater(result["p0-b0"]["end"], 0.9)
        self.assertLessEqual(result["p0-b0"]["end"], result["p0-b1"]["start"])
        self.assertGreaterEqual(result["p0-b1"]["end"], 1.8)
        self.assertLessEqual(result["p0-b1"]["end"], 2.1)

    def test_missing_whisper_transcript_fails_instead_of_mapping_whole_chapter(self):
        with self.assertRaisesRegex(RuntimeError, "Whisper не распознал"):
            align_bubbles_to_words(self.items, [], 30)

    def test_french_alignment_normalizes_accents_and_apostrophe_tokens(self):
        self.assertEqual(normalize_words("HE n’est pas allé"), ["he", "n", "est", "pas", "alle"])
        self.assertEqual(normalize_words("CŒUR, déjà"), ["coeur", "deja"])

    def test_ocr_spelling_variants_still_align_to_whisper_words(self):
        item = dict(self.items[0], id="p0-short", original="MONSIELIR REUE !")
        result = align_bubbles_to_words(
            [item], [{"text": "Monsieur", "start": 0.40, "end": 0.85},
                     {"text": "Roux", "start": 0.86, "end": 1.20}], 2.0)
        self.assertGreaterEqual(result["p0-short"]["start"], 0.25)
        self.assertLessEqual(result["p0-short"]["start"], 0.40)
        self.assertLess(result["p0-short"]["end"], 1.6)

    def test_translation_and_ocr_quality_reject_mixed_script_and_character_soup(self):
        self.assertTrue(is_russian_translation("Bonjour", "Здравствуйте, Серж."))
        self.assertFalse(is_russian_translation("Bonjour", "Rage te Ri ne uta Глава: Резес"))
        self.assertFalse(is_russian_translation("Bonjour", "ÐŸÑ€Ð¸Ð²ÐµÑ‚"))
        self.assertFalse(is_usable_ocr_text("N D 2 J 7 L Li #k"))
        self.assertFalse(is_usable_ocr_text("Rage te Ri ne uta Chapitre 1 : Les reses"))
        self.assertTrue(is_usable_ocr_text("Hé !"))

    def test_chapter_bubble_collection_skips_ocr_noise_and_cleans_bad_translation(self):
        old_path, old_init = db_cache.DB_PATH, db_cache._db_initialized
        handle, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        try:
            db_cache.DB_PATH = path
            db_cache._db_initialized = False
            save_cached_translation("m", "c", 0, {"x": 1, "y": 1, "w": 10, "h": 5},
                                    {"original": "Bonjour", "natural": "Привет"})
            save_cached_translation("m", "c", 0, {"x": 20, "y": 1, "w": 10, "h": 5},
                                    {"original": "N D 2 J 7 L Li #k", "natural": "N D 2 J 7 L Li #k"})
            save_cached_translation("m", "c", 0, {"x": 40, "y": 1, "w": 10, "h": 5},
                                    {"original": "Merci", "natural": "Thank you very much"})
            bubbles = get_all_cached_chapter_bubbles("m", "c")
            self.assertEqual(len(bubbles), 2)
            by_source = {bubble["original"]: bubble for bubble in bubbles}
            self.assertIn("Bonjour", by_source)
            self.assertEqual(by_source["Merci"]["natural"], "")
        finally:
            db_cache.DB_PATH = old_path
            db_cache._db_initialized = old_init
            os.remove(path)

    def test_tts_chunk_cache_persists_only_matching_input_hash(self):
        old_path, old_init = db_cache.DB_PATH, db_cache._db_initialized
        handle, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        try:
            db_cache.DB_PATH = path
            db_cache._db_initialized = False
            audio = make_wav()
            db_cache.save_cached_chapter_voiceover_tts_chunk("m", "c", "fp", "model", 0, "hash", audio)
            self.assertEqual(db_cache.get_cached_chapter_voiceover_tts_chunk("m", "c", "fp", "model", 0, "hash"), audio)
            self.assertIsNone(db_cache.get_cached_chapter_voiceover_tts_chunk("m", "c", "fp", "model", 0, "changed"))
            db_cache.delete_cached_chapter_voiceover_tts_chunks("m", "c", "fp")
            self.assertIsNone(db_cache.get_cached_chapter_voiceover_tts_chunk("m", "c", "fp", "model", 0, "hash"))
        finally:
            db_cache.DB_PATH = old_path
            db_cache._db_initialized = old_init
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
