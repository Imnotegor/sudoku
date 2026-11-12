# Sudoku

![Sudoku hero](assets/hero.jpg)

[Русский](#русский) | [English](#english)

## Русский

Минималистичная судоку‑игра в браузере с офлайн‑режимом, адаптивной сеткой и сканированием бумажных судоку.

На главной странице есть быстрая партия, личная карта города и командная битва. В битве четыре игрока делятся на две команды. За 25 минут нужно удержать больше районов. В личной кампании решённая задача открывает соседние районы, а классические партии хранятся отдельно.

Запуск всех режимов локально: `docker compose up --build`, затем откройте `http://localhost:8080`. PostgreSQL хранит состояние в томе `city-data`. При запуске Rust-сервер создаёт схему `city` с 22 таблицами и загружает 72 судоку. Для локального доступа к админке задайте `CITY_ADMIN_KEY` (случайный ключ длиной от 32 символов) перед запуском Compose. Игрок получает гостевой профиль в браузере.

**Что есть в быстрой партии:**
- офлайн‑режим (PWA) и автосохранение прогресса
- адаптивная верстка для телефона и десктопа
- уровни сложности
- заметки (несколько чисел в клетке), отмена хода, очистка
- подсветка ошибок и связанных клеток
- ручной ввод с импортом/экспортом строки 9x9
- сканирование с камеры или фото
- решение судоку (локальный solver + серверный API)
- OCR‑сервис на Python с CNN‑моделью для печатных цифр и fallback на Tesseract

## English

A minimal Sudoku game for the browser with offline mode, responsive layout, and scanning paper puzzles.

**What’s included:**
- offline mode (PWA) with autosave
- responsive UI for mobile and desktop
- difficulty levels
- notes (multiple candidates per cell), undo, clear
- conflict highlighting and related cell emphasis
- manual input with 9x9 import/export string
- camera/photo scanning
- Sudoku solving (local solver + server API)
- Python OCR service with a CNN model for printed digits and a Tesseract fallback
