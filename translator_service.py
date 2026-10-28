import os
import io
import re
import json
import base64
import zipfile
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, ImageEnhance
import requests
import pytesseract
from pathlib import Path

from db_cache import get_cached_translation, save_cached_translation, get_cached_page_bubbles, save_cached_page_bubbles, delete_cached_translations_for_boxes
from text_quality import is_russian_translation, is_usable_ocr_text

try:
    import ai_edge_litert.interpreter as litert
    HAS_LITERT = True
except ImportError:
    HAS_LITERT = False

try:
    import ctranslate2
    import sentencepiece as spm
except ImportError:
    ctranslate2 = None
    spm = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_LAST_ERROR = ""
LOCAL_MODELS_DIR = Path(__file__).parent / ".local_models"
TESSDATA_DIR = LOCAL_MODELS_DIR / "tessdata"


class OfflineFrenchRussianTranslator:
    """Small, persistent CPU translator: French → English → Russian."""
    _instance = None

    def __init__(self):
        self.stages = []
        if ctranslate2 is None or spm is None:
            return
        for name in ("fr_en", "en_ru"):
            root = LOCAL_MODELS_DIR / name
            model_path = root / "model"
            tokenizer_path = root / "sentencepiece.model"
            if not model_path.is_dir() or not tokenizer_path.is_file():
                self.stages = []
                return
            self.stages.append((
                ctranslate2.Translator(str(model_path), device="cpu", compute_type="int8"),
                spm.SentencePieceProcessor(model_file=str(tokenizer_path)),
            ))

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def translate(self, text: str) -> Optional[str]:
        if not self.stages:
            return None
        current = text.strip()
        for translator, tokenizer in self.stages:
            tokens = tokenizer.encode(current, out_type=str)
            result = translator.translate_batch(
                [tokens], beam_size=2, max_decoding_length=160
            )[0].hypotheses[0]
            current = tokenizer.decode(result).replace("▁", " ").strip()
        return current or None


class MangaYOLOModel:
    _instance = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        if HAS_LITERT and os.path.exists(model_path):
            try:
                self.interpreter = litert.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                print(f"[!] Warning: Failed to initialize LiteRT model: {e}")
                self.interpreter = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            default_path = os.path.join(os.path.dirname(__file__), "models", "manga_panel_detector_int8.tflite")
            cls._instance = cls(default_path)
        return cls._instance

    def predict(self, img: Image.Image, conf_threshold: float = 0.25) -> List[Dict]:
        if not self.interpreter:
            return []

        resized = img.resize((640, 640), Image.BILINEAR)
        input_data = np.array(resized, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        detections = []
        for det in output:
            x1_norm, y1_norm, x2_norm, y2_norm, conf, cls_id = det
            if conf >= (0.12 if int(cls_id) == 1 else conf_threshold):
                cls_int = int(cls_id)
                label = "panel" if cls_int == 0 else "text"
                x1 = max(0.0, min(1.0, float(x1_norm)))
                y1 = max(0.0, min(1.0, float(y1_norm)))
                x2 = max(0.0, min(1.0, float(x2_norm)))
                y2 = max(0.0, min(1.0, float(y2_norm)))

                if x2 > x1 and y2 > y1:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    area = box_w * box_h
                    if cls_int == 1 and (box_w < 0.012 or box_h < 0.008 or area < 0.00025 or area > 0.28):
                        continue
                    detections.append({
                        "label": label,
                        "class_id": cls_int,
                        "confidence": round(float(conf), 3),
                        "box": {
                            "x": round(x1 * 100, 2),
                            "y": round(y1 * 100, 2),
                            "w": round((x2 - x1) * 100, 2),
                            "h": round((y2 - y1) * 100, 2)
                        }
                    })

        # The exported model sometimes returns several overlapping boxes for one
        # speech area, especially at the lower text threshold. Keep the strongest.
        kept = []
        for candidate in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            a = candidate["box"]
            suppressed = False
            for existing in kept:
                if existing["class_id"] != candidate["class_id"]:
                    continue
                b = existing["box"]
                left, top = max(a["x"], b["x"]), max(a["y"], b["y"])
                right = min(a["x"] + a["w"], b["x"] + b["w"])
                bottom = min(a["y"] + a["h"], b["y"] + b["h"])
                intersection = max(0, right - left) * max(0, bottom - top)
                union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
                if union and intersection / union > 0.5:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(candidate)
        return sorted(kept, key=lambda d: d["box"]["y"])


class MangaTranslatorService:
    @staticmethod
    def extract_image_from_cbz(cbz_path: str, page_idx: int) -> Image.Image:
        with zipfile.ZipFile(cbz_path, "r") as zf:
            image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
            names = sorted(
                (n for n in zf.namelist()
                 if os.path.splitext(n)[1].lower() in image_exts
                 and not n.startswith("__MACOSX/")
                 and not os.path.basename(n).startswith(".")),
                key=lambda value: [int(part) if part.isdigit() else part.casefold()
                                   for part in re.split(r"(\d+)", value)],
            )
            if page_idx < 0 or page_idx >= len(names):
                raise ValueError(f"Page index {page_idx} out of range (total: {len(names)})")
            img_data = zf.read(names[page_idx])
            return Image.open(io.BytesIO(img_data)).convert("RGB")

    @classmethod
    def clean_and_validate_translation(cls, result: Dict) -> Dict:
        """
        Guarantees translation quality:
        - Filters out empty/nonsense OCR artifacts (stray punctuation, single symbols).
        - Strips broken characters, brackets, raw markdown junk.
        - Ensures the Russian translation is genuine and contains Cyrillic letters.
        - Fixes OCR typos and syntax if needed.
        """
        if not result:
            return result

        orig = (result.get("original") or "").strip()
        nat = (result.get("natural") or "").strip()
        lit = (result.get("literal") or "").strip()

        # Clean noise, brackets, extra quotes
        orig = re.sub(r'^[|_\-—~`\'"“«\s]+|[|_\-—~`\'"”»\s]+$', '', orig).strip()
        nat = re.sub(r'^[|_\-—~`\'"“«\s]+|[|_\-—~`\'"”»\s]+$', '', nat).strip()

        # Check for pure noise (no actual letters)
        has_letters_orig = bool(re.search(r'[a-zA-Zа-яА-Я]', orig))
        has_cyrillic_nat = bool(re.search(r'[а-яА-ЯёЁ]', nat))

        if not has_letters_orig or len(orig) < 2:
            result["original"] = ""
            result["natural"] = ""
            result["literal"] = ""
            return result

        # If natural translation has no Cyrillic or repeats original French, trigger fast LLM correction
        if not has_cyrillic_nat or nat.lower() == orig.lower():
            corrected = cls.translate_text_with_groq_llm(orig)
            if corrected and corrected.get("natural"):
                nat = corrected["natural"]
                if not lit and corrected.get("literal"):
                    lit = corrected["literal"]

        result["original"] = orig
        result["natural"] = nat
        result["translated"] = nat
        result["literal"] = lit
        return result

    @classmethod
    def ocr_and_translate_with_groq_vision(cls, crop_img: Image.Image) -> Optional[Dict]:
        """Read a French manga bubble and translate it to Russian with Groq vision."""
        if not GROQ_API_KEY:
            return None
        buf = io.BytesIO()
        crop_img.save(buf, format="JPEG", quality=92)
        prompt = """Read the text in this French manga bubble and translate it into natural Russian.
Correct obvious OCR mistakes using the visible letters and sentence context. Preserve the speaker's meaning, tone, names, and meaningful numbers. Do not invent a line when the image contains only an unreadable sound effect.
Return only a JSON object with exactly these keys:
{"original":"complete French text","natural":"complete Russian translation in Cyrillic"}
The natural field must be Russian. Do not return French or English there. Do not put OCR debris, isolated punctuation, or meaningless digits in the translation. If the crop is not readable dialogue, return empty strings for both fields."""
        image_data = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = {
            "model": GROQ_VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]}],
            "temperature": 0.1,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
                timeout=35,
            )
            if not response.ok:
                global GROQ_LAST_ERROR
                GROQ_LAST_ERROR = f"Groq API вернул HTTP {response.status_code}. Проверьте GROQ_API_KEY и доступность модели."
                print(f"[!] Groq Vision error: HTTP {response.status_code}")
                return None
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            original = parsed.get("original", "")
            natural = parsed.get("natural", "")
            if not isinstance(original, str) or not isinstance(natural, str):
                return None
            original, natural = original.strip(), natural.strip()
            return {
                "original": original,
                "ipa": "",
                "literal": "",
                "natural": natural,
                "translated": natural,
                "emotion": "neutral",
                "vocabulary": {},
                "provider": f"Groq ({GROQ_VISION_MODEL})",
            }
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            print(f"[!] Groq Vision failed: {type(error).__name__}")
            return None

    @classmethod
    def ocr_and_translate_region(
        cls,
        cbz_path: str,
        manga_name: str,
        filename: str,
        page_idx: int,
        box: Optional[Dict[str, float]] = None,
        src_lang: str = "fra",
        target_lang: str = "ru",
        use_cache: bool = True
    ) -> Dict:
        global GROQ_LAST_ERROR
        GROQ_LAST_ERROR = ""
        # 1. Check local persistent SQLite cache first!
        if use_cache:
            cached = get_cached_translation(manga_name, filename, page_idx, box)
            if cached and cached.get("original"):
                return cached

        full_img = cls.extract_image_from_cbz(cbz_path, page_idx)
        w_img, h_img = full_img.size

        if box and all(k in box for k in ["x", "y", "w", "h"]):
            pad_x = 1.0
            pad_y = 1.0
            x1 = max(0, int(((box["x"] - pad_x) / 100.0) * w_img))
            y1 = max(0, int(((box["y"] - pad_y) / 100.0) * h_img))
            x2 = min(w_img, int(((box["x"] + box["w"] + pad_x) / 100.0) * w_img))
            y2 = min(h_img, int(((box["y"] + box["h"] + pad_y) / 100.0) * h_img))
            crop = full_img.crop((x1, y1, x2, y2)) if (x2 > x1 + 5 and y2 > y1 + 5) else full_img
        else:
            crop = full_img

        # 2. Call Groq vision for OCR and translation.
        result = cls.ocr_and_translate_with_groq_vision(crop)
        
        # Reject non-Russian answers and OCR strings dominated by symbols/noise.
        if result:
            source = result.get("original", "")
            translated = result.get("natural", "")
            if not is_usable_ocr_text(source):
                result = None
            elif not is_russian_translation(source, translated):
                repaired = cls.translate_text_with_groq(source)
                if repaired and is_russian_translation(source, repaired.get("natural", "")):
                    result = {**result, **repaired, "provider": f"{result.get('provider', 'Groq Vision')} + validated translation"}
                else:
                    result = None

        # Local OCR is the fallback when vision parsing is malformed or untranslated.
        if not result:
            gray = crop.convert("L")
            enhancer = ImageEnhance.Contrast(gray)
            proc_img = enhancer.enhance(1.8)
            tess_lang = {"fra": "fra", "fr": "fra", "eng": "eng", "en": "eng"}.get(src_lang, "fra")
            tess_config = f'--tessdata-dir "{TESSDATA_DIR}" --psm 6' if (TESSDATA_DIR / f"{tess_lang}.traineddata").is_file() else "--psm 6"
            if max(proc_img.size) < 1400:
                scale = 1400 / max(proc_img.size)
                proc_img = proc_img.resize((int(proc_img.width * scale), int(proc_img.height * scale)), Image.Resampling.LANCZOS)
            raw_text = pytesseract.image_to_string(proc_img, lang=tess_lang, config=tess_config).strip()
            lines = [re.sub(r"[|_\\/]+", " ", line).strip() for line in raw_text.splitlines()]
            cleaned = re.sub(r"\s+", " ", " ".join(line for line in lines if len(line) > 1)).strip()

            translated_result = cls.translate_text_with_groq(cleaned) if is_usable_ocr_text(cleaned) else None

            if translated_result and is_russian_translation(cleaned, translated_result.get("natural", "")):
                result = translated_result
            else:
                result = {
                    "original": cleaned,
                    "ipa": "",
                    "literal": "",
                    "natural": "",
                    "translated": "",
                    "vocabulary": {},
                    "provider": "OCR (перевод не прошёл проверку)",
                    "translation_error": GROQ_LAST_ERROR or (
                        "Добавьте GROQ_API_KEY в .env." if not GROQ_API_KEY
                        else "Groq не вернул корректный русский перевод. Проверьте исходный текст и ответ API."
                    ),
                }

        # Clean, validate and eliminate nonsensical output
        result = cls.clean_and_validate_translation(result)
        result["box"] = box

        # Save to local SQLite cache only if we have a real Russian translation
        if is_usable_ocr_text(result.get("original", "")) and is_russian_translation(
            result.get("original", ""), result.get("natural", "")
        ):
            save_cached_translation(manga_name, filename, page_idx, box, result)

        return result

    @classmethod
    def translate_text_with_groq(cls, text: str) -> Optional[Dict]:
        """Translate OCR text with Groq and accept only Russian output."""
        global GROQ_LAST_ERROR
        if not GROQ_API_KEY:
            GROQ_LAST_ERROR = "Добавьте GROQ_API_KEY в .env."
            return None
        if not is_usable_ocr_text(text):
            return None
        prompt = (
            "Translate the following French manga dialogue into one complete, natural Russian line. "
            "Use Cyrillic for Russian words; preserve character names and meaningful numbers. "
            "Do not answer in French or English, do not give variants or commentary. "
            "If the text is unreadable OCR noise, return an empty string. "
            "Respond only as JSON with the key natural.\nFrench: " + text
        )
        payload = {
            "model": GROQ_VISION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
                timeout=30,
            )
            if not response.ok:
                GROQ_LAST_ERROR = f"Groq API вернул HTTP {response.status_code}. Проверьте GROQ_API_KEY и доступность модели."
                print(f"[!] Groq text translation error: HTTP {response.status_code}")
                return None
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            natural = parsed.get("natural", "")
            if not isinstance(natural, str):
                return None
            natural = natural.strip()
            if not is_russian_translation(text, natural):
                GROQ_LAST_ERROR = "Groq вернул ответ, который не прошёл проверку русского языка."
                return None
            return {
                "original": text,
                "ipa": "",
                "literal": "",
                "natural": natural,
                "translated": natural,
                "vocabulary": {},
                "provider": f"Groq ({GROQ_VISION_MODEL})",
            }
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            GROQ_LAST_ERROR = f"Ошибка запроса к Groq: {type(error).__name__}."
            print(f"[!] Groq text translation failed: {type(error).__name__}")
            return None

    @classmethod
    def detect_bubbles(cls, cbz_path: str, manga_name: str, filename: str, page_idx: int) -> List[Dict]:
        """
        Finds all dialog bubbles on the page using YOLO26n.
        Uses cached bubble list if available.
        """
        cached = get_cached_page_bubbles(manga_name, filename, page_idx)
        if cached and all(b.get("source") == "yolo26n-v3" for b in cached):
            return cached

        if cached:
            delete_cached_translations_for_boxes(
                manga_name, filename, page_idx,
                [bubble.get("box") for bubble in cached],
            )

        full_img = cls.extract_image_from_cbz(cbz_path, page_idx)
        yolo = MangaYOLOModel.get_instance()
        yolo_dets = yolo.predict(full_img, conf_threshold=0.25)
        text_dets = [d for d in yolo_dets if d["label"] == "text"]

        results = []
        for idx, d in enumerate(text_dets, 1):
            results.append({
                "id": idx,
                "confidence": d["confidence"],
                "source": "yolo26n-v3",
                "box": d["box"]
            })

        # Empty detector output can mean the model is missing or below threshold;
        # don't poison the persistent cache with a false "no bubbles" result.
        if results:
            save_cached_page_bubbles(manga_name, filename, page_idx, results, is_fully_translated=False)
        return results

    @classmethod
    def batch_translate_page(cls, cbz_path: str, manga_name: str, filename: str, page_idx: int) -> List[Dict]:
        """
        Translates all speech bubbles on a page and saves them to local cache.
        """
        bubbles = cls.detect_bubbles(cbz_path, manga_name, filename, page_idx)
        enriched = []
        for b in bubbles:
            tr = cls.ocr_and_translate_region(
                cbz_path=cbz_path,
                manga_name=manga_name,
                filename=filename,
                page_idx=page_idx,
                box=b["box"],
                use_cache=True
            )
            b_copy = dict(b)
            b_copy["translation"] = tr
            enriched.append(b_copy)
        
        if enriched:
            fully_translated = all(
                is_russian_translation(
                    (b.get("translation") or {}).get("original", ""),
                    (b.get("translation") or {}).get("natural", ""),
                )
                for b in enriched
            )
            save_cached_page_bubbles(manga_name, filename, page_idx, enriched, is_fully_translated=fully_translated)
        return enriched
