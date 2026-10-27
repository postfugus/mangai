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

from db_cache import get_cached_translation, save_cached_translation, get_cached_page_bubbles, save_cached_page_bubbles

try:
    import ai_edge_litert.interpreter as litert
    HAS_LITERT = True
except ImportError:
    HAS_LITERT = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")


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
            if conf >= conf_threshold:
                cls_int = int(cls_id)
                label = "panel" if cls_int == 0 else "text"
                x1 = max(0.0, min(1.0, float(x1_norm)))
                y1 = max(0.0, min(1.0, float(y1_norm)))
                x2 = max(0.0, min(1.0, float(x2_norm)))
                y2 = max(0.0, min(1.0, float(y2_norm)))

                if x2 > x1 and y2 > y1:
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

        detections.sort(key=lambda d: d["box"]["y"])
        return detections


class MangaTranslatorService:
    @staticmethod
    def extract_image_from_cbz(cbz_path: str, page_idx: int) -> Image.Image:
        with zipfile.ZipFile(cbz_path, "r") as zf:
            names = sorted([n for n in zf.namelist() if not n.startswith("__MACOSX") and "." in n])
            if page_idx < 0 or page_idx >= len(names):
                raise ValueError(f"Page index {page_idx} out of range (total: {len(names)})")
            img_data = zf.read(names[page_idx])
            return Image.open(io.BytesIO(img_data)).convert("RGB")

    @classmethod
    def ocr_and_translate_with_groq_vision(cls, crop_img: Image.Image) -> Optional[Dict]:
        """
        Uses Groq Vision (qwen3.8-27b) with advanced linguistic tutoring prompt:
        - Exact transcript
        - MFA / IPA phonetic transcription
        - Word-by-word literal gloss (дословный подстрочник)
        - Natural literary Russian translation
        - Vocabulary & grammatical breakdown
        """
        if not GROQ_API_KEY:
            return None

        buf = io.BytesIO()
        crop_img.save(buf, format="JPEG", quality=95)
        b64_img = base64.b64encode(buf.getvalue()).decode()

        prompt = """You are an elite polyglot tutor optimizing rapid language acquisition through manga reading.
Analyze this cropped manga speech bubble image in its dramatic context.

Your mission: Make the Russian reader effortlessly assimilate the French grammatical patterns, syntax, and natural reflex.

Output a STRICT JSON object with these exact keys:
{
  "original": "exact original speech text without omissions",
  "ipa": "/MFA phonetic transcription with liaisons and pauses/",
  "literal": "поморфемный и пословный дословный подстрочник на русском, отражающий структуру французской фразы (например: 'C'est ça' -> 'Это есть это')",
  "natural": "литературный русский перевод с сохранением точной интонации и характера персонажа",
  "emotion": "одно слово интонации для озвучки: question, surprise, whisper, excited, calm, anger, sad, neutral",
  "vocabulary": {
    "mot": "значение, грамматическая форма, контекст"
  }
}
Output only the JSON object."""

        payload = {
            "model": GROQ_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 600
        }

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=12
            )
            if resp.status_code == 200:
                raw_json = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)
                return {
                    "original": parsed.get("original", "").strip(),
                    "ipa": parsed.get("ipa", "").strip(),
                    "literal": parsed.get("literal", "").strip(),
                    "natural": parsed.get("natural", "").strip() or parsed.get("translated", "").strip(),
                    "translated": parsed.get("natural", "").strip(), # alias for backward compatibility
                    "emotion": parsed.get("emotion", "neutral").strip().lower(),
                    "vocabulary": parsed.get("vocabulary", {}),
                    "provider": f"Groq Vision ({GROQ_VISION_MODEL})"
                }
            else:
                print(f"[!] Groq Vision error: HTTP {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"[!] Groq Vision exception: {e}")

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

        # 2. Call Groq Vision
        result = cls.ocr_and_translate_with_groq_vision(crop)
        
        # 3. Fallback to Tesseract OCR + Groq Text Translation (openai/gpt-oss-120b)
        if not result or not result.get("original") or not result.get("natural"):
            gray = crop.convert("L")
            enhancer = ImageEnhance.Contrast(gray)
            proc_img = enhancer.enhance(1.8)
            tess_lang = src_lang if src_lang in ["fra", "eng", "jpn"] else "fra"
            raw_text = pytesseract.image_to_string(proc_img, lang=tess_lang).strip()
            lines = [re.sub(r"[|_\\/]+", " ", line).strip() for line in raw_text.splitlines()]
            cleaned = " ".join([l for l in lines if len(l) > 1])

            # Translate through Groq LLM (gpt-oss-120b) to ensure Russian translation is ALWAYS produced
            translated_result = cls.translate_text_with_groq_llm(cleaned) if cleaned else None
            
            if translated_result and translated_result.get("natural"):
                result = translated_result
            else:
                result = {
                    "original": cleaned,
                    "ipa": "",
                    "literal": cleaned,
                    "natural": (translated_result.get("natural") if translated_result else "") or cleaned,
                    "translated": (translated_result.get("natural") if translated_result else "") or cleaned,
                    "vocabulary": {},
                    "provider": "Tesseract + Groq Fallback"
                }

        result["box"] = box
        # Save to local SQLite cache only if we have a real Russian translation
        if result.get("original") and result.get("natural") and result.get("natural") != result.get("original"):
            save_cached_translation(manga_name, filename, page_idx, box, result)

        return result

    @classmethod
    def translate_text_with_groq_llm(cls, text: str) -> Optional[Dict]:
        """
        Translates French OCR text to Russian with grammar and vocab using Groq text models.
        Uses clean single-line prompt to guarantee 100% robust translation without JSON schema rejects.
        """
        if not GROQ_API_KEY or not text.strip():
            return None

        prompt = (
            "You are an expert French-to-Russian translator for manga. "
            "Given French speech (which may have minor OCR typos), translate it accurately into natural literary Russian. "
            "Output ONLY the final Russian translation on a single line, nothing else.\n\n"
            f"French: {text}\nRussian:"
        )

        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 150
        }

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=10
            )
            if resp.status_code == 200:
                raw_translation = resp.json()["choices"][0]["message"]["content"].strip()
                # Clean up any surrounding quotes or markdown
                clean_ru = re.sub(r'^["`\']+|["`\']+$', '', raw_translation).strip()
                if clean_ru:
                    return {
                        "original": text,
                        "ipa": "",
                        "literal": clean_ru,
                        "natural": clean_ru,
                        "translated": clean_ru,
                        "vocabulary": {},
                        "provider": "Groq Fallback (qwen/qwen3.8-27b)"
                    }
            else:
                print(f"[!] Groq LLM translation error: {resp.status_code} - {resp.text[:150]}")
        except Exception as e:
            print(f"[!] Groq LLM translation exception: {e}")

        return None

    @classmethod
    def detect_bubbles(cls, cbz_path: str, manga_name: str, filename: str, page_idx: int) -> List[Dict]:
        """
        Finds all dialog bubbles on the page using YOLO26n.
        Uses cached bubble list if available.
        """
        cached = get_cached_page_bubbles(manga_name, filename, page_idx)
        if cached is not None:
            return cached

        full_img = cls.extract_image_from_cbz(cbz_path, page_idx)
        yolo = MangaYOLOModel.get_instance()
        yolo_dets = yolo.predict(full_img, conf_threshold=0.25)
        text_dets = [d for d in yolo_dets if d["label"] == "text"]

        results = []
        for idx, d in enumerate(text_dets, 1):
            results.append({
                "id": idx,
                "confidence": d["confidence"],
                "source": "yolo26n",
                "box": d["box"]
            })

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
        
        save_cached_page_bubbles(manga_name, filename, page_idx, enriched, is_fully_translated=True)
        return enriched
