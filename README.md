# Sudoku

[Играть онлайн / Play online](https://sockketsukker.hansatmp.de/) · [Русский](#русский) · [English](#english)

![Sudoku](assets/hero.jpg)

## Русский

Sudoku — браузерная игра, в которой можно решить одну головоломку, пройти личную карту или собрать четверых друзей для битвы за город.

| Режим | Как играть |
| --- | --- |
| [Быстрая партия](https://sockketsukker.hansatmp.de/quick.html) | Выберите сложность, вводите цифры и заметки, отменяйте ходы. Ошибки подсвечиваются. Поле можно ввести вручную, импортировать и экспортировать строкой или распознать по фото с камеры. Партия автоматически сохраняется в браузере. После первой загрузки можно играть офлайн. Для распознавания фото нужен интернет. |
| [Мой город](https://sockketsukker.hansatmp.de/?mode=solo) | Решите судоку стартового района, чтобы открыть соседние, и проложите путь до финиша. Загруженную карту можно проходить офлайн. Прогресс синхронизируется при следующем открытии карты с подключением к серверу. Быстрая партия сохраняется отдельно. |
| [Битва за город](https://sockketsukker.hansatmp.de/?mode=battle) | Создайте комнату и отправьте ссылку трём игрокам. Когда в обеих командах занято по два места, стартует матч на 25 минут. Каждое подтверждённое решение даёт команде одно очко влияния в выбранном районе. Побеждает команда, которая к концу матча удерживает больше районов. |

В каждом матче пять районов и от пяти до семи дорог. Сервер сохраняет планировку комнаты, поэтому игроки и зрители видят одну карту даже после перезагрузки. При равном влиянии район не засчитывается ни одной команде. Если команды удерживают поровну районов, матч заканчивается вничью.

На главной можно переключать три анимированных пейзажа — речные кварталы, архипелаг и горную долину. Вокзал, парк, университет, порт и маяк отмечены своими зданиями. Анимацию можно поставить на паузу, а при включённом в системе уменьшении движения карта статична.

### Локальный запуск

Из корня репозитория запустите приложение, сервис распознавания и PostgreSQL одной командой. Понадобится Docker с Compose.

```sh
docker compose up --build -d
```

Откройте [игру на localhost:8080](http://localhost:8080/). При первом входе в городской режим сервер создаст гостевой профиль для этого браузера. Готовность подключения к базе можно проверить через [городской API](http://localhost:8080/api/city/health).

Для входа в [локальную админку](http://localhost:8080/?mode=admin) задайте `CITY_ADMIN_KEY` длиной не менее 32 символов в файле `.env` в корне проекта. Случайное значение можно получить командой `openssl rand -hex 32`. После изменения `.env` выполните `docker compose up -d app` и введите этот же ключ в форму входа. Организатор может публиковать версии карт, выбирать длительность матча от 5 до 60 минут и снимать спорные задачи с пересчётом очков.

### Устройство проекта

- **Клиент** написан на TypeScript. Классическая игра хранит партию в `localStorage`, а у личной кампании отдельные сохранения и очередь офлайн-результатов. [Классическая игра](src/client/main.ts) · [Городские режимы](src/client/city/)
- **Сервер** на Rust и Axum раздаёт клиент, ведёт матчи и проверяет решения. Фото он передаёт отдельному сервису OCR на Python и FastAPI. [Городской API](src/server/city/mod.rs) · [API решения и сканирования](API.md) · [OCR](ocr/)
- **PostgreSQL 16** хранит в схеме `city` 22 таблицы и два представления. При первом запуске приложение загружает 72 судоку. Данные сохраняются в томе Compose `city-data`, координаты районов и дороги матча — в `matches.layout` типа `jsonb`. [Схема](src/server/city/schema.sql) · [Инициализация и миграция](src/server/city/mod.rs) · [Генератор планировки](src/server/city/match_layout.rs)

## English

Sudoku is a browser game with three modes. In **Quick game**, choose a difficulty, take notes, undo moves or import and export a grid as text. The board highlights mistakes. Your puzzle is saved in the browser and works offline after the first load. Scanning a photo uses the online OCR service. In **My city**, solving a district unlocks its neighbours. The downloaded campaign works offline and syncs results the next time you open the map online.

In **City battle**, share a room link with three other players. The 25-minute match starts when both teams have two players. Each verified solution earns one point in its district. The team with more points controls that district. Tied districts count for neither team, and an equal number of controlled districts ends the match in a draw. Each room keeps its own layout of five districts and five to seven roads. Everyone sees the same landmarks and landscape.

### Run locally

Run `docker compose up --build -d` from the repository root, then open [localhost:8080](http://localhost:8080/). Compose starts the Rust application, the Python OCR service and PostgreSQL 16. The server initializes 22 tables, two views and 72 puzzles. The `city-data` volume persists game data between restarts.

To use the [local admin panel](http://localhost:8080/?mode=admin), set `CITY_ADMIN_KEY` in the root `.env` file to a random key of at least 32 characters. Generate one with `openssl rand -hex 32`, run `docker compose up -d app` and enter the same key in the browser. Organizers can publish maps, host timed matches and review disputed scores.
