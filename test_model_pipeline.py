import os
import sys
import zipfile
import io
import time
from PIL import Image

sys.path.insert(0, "/root/mangakawaii-downloader")

from translator_service import MangaYOLOModel, MangaTranslatorService


def run_tests():
    print("=" * 60)
    print("🔬 ЗАПУСК ТЕСТОВ МОДЕЛИ YOLO26n И ПАЙПЛАЙНА ПЕРЕВОДА")
    print("=" * 60)

    # 1. Проверка модели
    model_path = "/root/mangakawaii-downloader/models/manga_panel_detector_int8.tflite"
    assert os.path.exists(model_path), f"Файл модели {model_path} не найден!"
    print(f"[OK] 1. Файл модели найден: {model_path} ({os.path.getsize(model_path)} байт)")

    # 2. Инициализация модели
    yolo = MangaYOLOModel.get_instance()
    assert yolo.interpreter is not None, "Ошибка инициализации LiteRT интерпертатора!"
    print(f"[OK] 2. LiteRT интерпретатор успешно создан и выделил тензоры")

    # 3. Проверка формы тензоров
    input_shape = yolo.input_details[0]["shape"].tolist()
    output_shape = yolo.output_details[0]["shape"].tolist()
    print(f"[OK] 3. Тензоры модели:")
    print(f"       Вход: {input_shape} (тип: {yolo.input_details[0]['dtype']})")
    print(f"       Выход: {output_shape} (тип: {yolo.output_details[0]['dtype']})")
    assert input_shape == [1, 640, 640, 3], f"Ожидался вход [1, 640, 640, 3], получен {input_shape}"
    assert output_shape == [1, 300, 6], f"Ожидался выход [1, 300, 6], получен {output_shape}"

    # 4. Тест детекции на страницах манги
    cbz_path = "/root/manga_downloads/Kaze_to_Ki_no_Uta/Kaze to Ki no Uta - Ch.01.cbz"
    assert os.path.exists(cbz_path), "Файл CBZ Ch.01 не найден!"

    test_pages = [8, 11, 14, 19] # Индексы страниц с репликами
    total_text_bubbles = 0
    total_panels = 0

    print("\n" + "-" * 60)
    print("📄 ТЕСТ ДЕТЕКЦИИ НА СТРАНИЦАХ ГЛАВЫ 1:")
    print("-" * 60)

    for p_idx in test_pages:
        t0 = time.time()
        img = MangaTranslatorService.extract_image_from_cbz(cbz_path, p_idx)
        w, h = img.size
        
        detections = yolo.predict(img, conf_threshold=0.25)
        dt = (time.time() - t0) * 1000

        panels = [d for d in detections if d["label"] == "panel"]
        texts = [d for d in detections if d["label"] == "text"]
        total_text_bubbles += len(texts)
        total_panels += len(panels)

        print(f"  Страница {p_idx+1} ({w}x{h}): {len(panels)} панелей, {len(texts)} реплик [время: {dt:.1f}ms]")
        for idx, t in enumerate(texts[:2], 1):
            box = t["box"]
            print(f"    * Реплика #{idx} (conf={t['confidence']}): x={box['x']}%, y={box['y']}%, w={box['w']}%, h={box['h']}%")

    assert total_text_bubbles > 0, "Модель не нашла ни одного текстового бабла!"
    print(f"\n[OK] 4. Детекция успешна! Всего найдено {total_panels} панелей и {total_text_bubbles} баблов диалогов.")

    # 5. Тест сквозного OCR и перевода детекции
    print("\n" + "-" * 60)
    print("🔤 ТЕСТ СКВОЗНОГО ПЕРЕВОДА НАЙДЕННОГО БАБЛА (Страница 9):")
    print("-" * 60)

    # Берём детекцию со страницы 9 (индекс 8)
    bubbles = MangaTranslatorService.detect_bubbles(cbz_path, page_idx=8, src_lang="fra", target_lang="ru")
    assert len(bubbles) > 0, "Не найдены баблы через detect_bubbles!"
    
    first_bubble = bubbles[0]
    print(f"  Тестируем бабл ID {first_bubble['id']} (conf={first_bubble.get('confidence')}):")
    print(f"  Координаты: {first_bubble['box']}")

    t_res = MangaTranslatorService.ocr_and_translate_region(
        cbz_path=cbz_path,
        page_idx=8,
        box=first_bubble["box"],
        src_lang="fra",
        target_lang="ru"
    )

    print(f"\n  [FR] Распознанный оригинал: \"{t_res['original']}\"")
    print(f"  [RU] Перевод:              \"{t_res['translated']}\"")
    print(f"  [📚] Словарный разбор:     {t_res['vocabulary']}")

    assert len(t_res["original"]) > 0, "Оригинал пустой!"
    assert len(t_res["translated"]) > 0, "Перевод пустой!"
    print("\n" + "=" * 60)
    print("✅ ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ! МОДЕЛЬ РАБОТАЕТ БЕЗУПРЕЧНО.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
