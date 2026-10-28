# MangaKawaii Downloader & API

API и инструмент для скачивания манги с сайта `mangakawaii.fr` в стандартный манга-формат **CBZ** (Comic Book Zip).

## Особенности
- **Обход Cloudflare**: использование браузерного TLS-fingerprint (Chrome impersonation via `curl_cffi`).
- **Формат CBZ**: архив со сжатием, корректная нумерация страниц (`001.jpg`, `002.png`), совместим с Tachiyomi, Mihon, Moon+ Reader, Panels, CDisplayEx.
- **Фильтрация ловушек**: автоматическое удаление скрытых anti-bot изображений (`__mk_trap__`).
- **REST API (FastAPI)**: эндпоинты для получения метаданных, страниц глав, запуска фоновых скачиваний и мониторинга прогресса.
- **CLI**: консольная утилита с индикатором прогресса и возобновлением закачки (пропуск уже скачанных глав).

---

## Структура проекта
```
/root/mangakawaii-downloader/
├── mangakawaii_scraper.py   # Клиент парсера и загрузчика
├── api.py                   # FastAPI REST API сервис
├── cli.py                   # Консольная утилита скачивания
└── README.md                # Документация
```

---

## Запуск REST API
Для скачивания глав через REST API:

```bash
PYTHONPATH=. uvicorn api:app --host 0.0.0.0 --port 8000
```

### Документация Swagger UI
После запуска доступна по адресу: `http://localhost:8000/docs`

---

## Эндпоинты API

### 1. Информация о манге и список глав
`GET /manga/{slug}`
Пример:
```bash
curl http://localhost:8000/manga/kaze-to-ki-no-uta
```
Ответ:
```json
{
  "slug": "kaze-to-ki-no-uta",
  "title": "Kaze to Ki no Uta",
  "description": "L'histoire se déroule en France...",
  "cover_url": "https://cdn.mangakawaii.fr/uploads/...",
  "total_chapters": 39,
  "chapters": [
    {"number": "1", "title": "Ch. 1", "url": "https://www.mangakawaii.fr/manga/kaze-to-ki-no-uta/1"},
    ...
  ]
}
```

### 2. Получение ссылок на страницы главы
`GET /manga/{slug}/chapter/{number}/pages`
Пример:
```bash
curl http://localhost:8000/manga/kaze-to-ki-no-uta/chapter/1/pages
```

### 3. Запуск фоновой выкачки манги в CBZ
`POST /manga/{slug}/download`
Тело запроса (JSON):
```json
{
  "from_chapter": 1,
  "to_chapter": 39,
  "concurrency": 6,
  "output_dir": "/root/manga_downloads/Kaze_to_Ki_no_Uta"
}
```
Ответ:
```json
{
  "task_id": "9a7b6a12-...",
  "status": "queued",
  "check_url": "/tasks/9a7b6a12-..."
}
```

### 4. Проверка статуса задачи
`GET /tasks/{task_id}`

### 5. Список скачанных файлов
`GET /downloads?slug=kaze-to-ki-no-uta`

---

## Первичная настройка веб-ридера

Установите зависимости и один раз загрузите локальные модели распознавания французского текста и перевода:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_local_models.py
./start_server.sh
```

Для локальной OCR-обработки также нужен системный `tesseract`. Jev (`TYPESAFE_API_KEY`) сравнивает порядок соседних реплик на странице и помечает эмоции; уверенные сравнения исправляют порядок, неуверенные оставляют порядок чтения страницы. Офлайн-аудиоглаву создаёт только Gemini TTS (`GEMINI_API_KEY`, мужской французский голос); длинные главы делятся на части по безопасному объёму текста, а ответы кэшируются локально. Whisper Small локально выставляет тайминги, а анализ RMS-огибающей уточняет начало и конец каждой фразы. Дорожка, порядок, эмоции и привязки к баблам сохраняются в `translations_cache.db`. Нажмите «Аудио» в ридере, чтобы запустить подготовку главы; локальную модель Whisper приложение скачает при первом таком запуске.

Модели детектора хранятся в `.local_models/`, а главы, переводы, координаты пузырей и офлайн-аудиоглавы — локально. Groq Vision распознаёт французский текст и переводит его на русский. Для отдельной реплики без готовой Gemini-аудиоглавы используется мгновенный Deepgram TTS; он не создаёт и не подменяет локальный аудиокэш главы. При повторном открытии неизменённой главы готовая Gemini-дорожка берётся из локальной базы без повторных запросов к TTS.

## Запуск через CLI

Выкачать все части манги:
```bash
PYTHONPATH=. python3 cli.py --slug kaze-to-ki-no-uta --output ./downloads/Kaze_to_Ki_No_Uta
```

Выкачать диапазон глав (например, с 1 по 10):
```bash
PYTHONPATH=. python3 cli.py --slug kaze-to-ki-no-uta --start 1 --end 10
```
