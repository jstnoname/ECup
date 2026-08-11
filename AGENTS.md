# AGENTS.md

## Project

ECup 2026 — контейнерное соревнование по матчингу товаров (дубли карточек на маркетплейсе). Код обучается/валидируется **только удалённо** (Kaggle, Colab через PyCharm); локальная машина — для EDA, вёрстки кода и сборки Docker-образа.

- Метрика: **Macro PR-AUC** по 20 категориям (`sklearn.metrics.average_precision_score` по каждой категории, затем среднее). НЕ использовать `auc()`/`roc_auc` для PR-кривой.
- Лимиты рантайма: Check 1 мин (1000 пар), Public 6 мин (~115K пар), Private 13 мин (~275K пар). Ресурсы: 20 CPU, 200GB RAM, H100 80GB.
- Архив сабмита ≤ 5GB, Docker-образ ≤ 15GB, ≤ 5 сабмитов/день. Контейнер работает **без интернета** — все библиотеки в образе, веса модели в архиве/образе.
- Данные: приватный HF-датасет **`well-please/ecup-data`** (org `well-please`, раздел Data ниже); локальная копия — `C:\Users\Alexander\Downloads\`. Файлы: `items.parquet` (13M товаров), `items_human.parquet` (подмножество для эмуляции), `matches.parquet` (365K пар, ручная разметка 0/1), `matches_llm.parquet` (11M пар, LLM-таргеты 0..1). Тестовые товары не пересекаются с обучающими.

## Architecture

### Карта репозитория

```
ECup/
├── pyproject.toml      # 1) зависимости (core/train/dev) — единый источник, нет requirements.txt
│                       # 2) [tool.ecup] — конфиг обучения (model_name, epochs, lr, веса, batch_size...)
├── uv.lock             # lock только для локальной установки (uv); платформы ставят pip из pyproject
├── .env                # секреты (Dagshub MLflow URI+креды, HF_TOKEN) — gitignored, только train-путь
├── .dockerignore       # страховка: .env/.venv/.git/данные/ноутбуки не попадают в образ
├── metadata.json       # контракт сабмита: {"image": "jstnoname/ecup-solution:V", "entry_point": "python -u run.py"}
├── run.py              # entry point соревнования: CLI-инференс (--items_path/-i, --matches_path/-m,
│                       #   --output-path/-o → submit.csv). Тонкий: без MLflow, без чтения .env
├── ecup/               # общий пакет (локально: editable через uv; платформы: %pip install -e .)
│   ├── config.py       # загрузка секретов (.env / kaggle_secrets / userdata) — импортируется ТОЛЬКО train-кодом
│   ├── data.py         # ensure_data(cfg, dest) — идемпотентное скачивание данных с HF (token из секретов)
│   ├── text.py         # сборка текста пары name+attributes — ЕДИНЫЙ формат train=inference, менять только тут
│   └── model.py        # CrossEncoder wrapper + pyfunc flavor: predict(DataFrame[id1,id2,text1,text2]) -> scores
├── train/
│   └── train_ce.py     # fit_cross_encoder(cfg) -> mlflow run_id — вся логика обучения, ноутбук = оркестрация
├── notebooks/
│   ├── 01_eda.ipynb    # EDA локально (uv run jupyter lab, ядро из .venv, polars)
│   └── 02_train.ipynb  # оркестрация обучения на Kaggle/Colab: %pip install -e .[train], tomllib-конфиг
└── scripts/
    └── emulate.py      # удалённая эмуляция run.py на items_human.parquet + срез пар:
                        #   валидация формата вывода + замер времени (запуск на Kaggle/Colab, не локально)
```

- `pyproject.toml` — **единственный источник зависимостей и конфига обучения**:
  - группы: `core` (polars, pyarrow, python-dotenv — то, что реально нужно контейнеру), `train` (torch, transformers, sentence-transformers, mlflow==3.5.1, scikit-learn), `dev` (jupyterlab, matplotlib, ipykernel). НЕТ requirements.txt.
  - `[tool.ecup]` — конфиг обучения (model_name, epochs, lr, human_weight, llm_weight, batch_size, max_len, seed...). Читается ноутбуком через `tomllib`, при желании переопределяется словарём в ячейке, целиком логируется в MLflow.
  - `requires-python = ">=3.11"` (на Kaggle Python 3.11 — ужесточать нельзя).
- `ecup/` — общий пакет: `text.py` (сборка текста пары name+attributes — **единый формат train=inference**, менять только здесь), `model.py` (обёртка CrossEncoder + pyfunc flavor: `predict(DataFrame[id1,id2,text1,text2]) -> scores` — контракт для mlflow docker), `config.py` (загрузка секретов — только train-путь), `data.py` (скачивание данных с HF — только dev/train-путь).
- `train/train_ce.py` — `fit_cross_encoder(cfg) -> mlflow run_id`; вся логика обучения в файлах, ноутбук = оркестрация.
- `run.py` — entry point соревнования: `--items_path/-i`, `--matches_path/-m`, `--output-path/-o` → `submit.csv` с колонками `id1,id2,predict` (сырые скоры, ВСЕ пары без исключения). Тонкий и чистый: без обучения, без MLflow, **без чтения `.env`**.
- `notebooks/01_eda.ipynb` (локально), `notebooks/02_train.ipynb` (Kaggle/Colab), `scripts/emulate.py` (удалённая эмуляция run.py на items_human + срез пар: валидация формата + замер времени), `metadata.json` (`{"image": "jstnoname/ecup-solution:V", "entry_point": "python -u run.py"}`).

## Data

- Источник: приватный датасет **`well-please/ecup-data`** (repo_type=dataset, private) в HF-организации `well-please`. Файлы: `items.parquet`, `items_human.parquet`, `matches.parquet`, `matches_llm.parquet` — сырые parquet, без конверсии в Arrow/HF-формат.
- Доступ: членство в org `well-please`. Токены **личные у каждого участника**: для скачивания достаточно read-скоупа, для (пере)заливки датасета — write.
- Скачивание в коде: `ecup/data.py` → `ensure_data(cfg, dest)` через `hf_hub_download` (token из секретов/`.env`), идемпотентно — пропускает уже скачанные файлы; данные кладём в `/kaggle/working/data` / `/content/data`.
- Загрузка данных запрещена правилами соревнования: датасет НЕ публиковать, файлы не коммитить в git, не включать в docker-образ/архив (в рантайме контейнеру данные приходят аргументами).

## Commands

- `uv sync --extra train --extra dev` — локальная установка (Windows → CPU-торч из PyPI).
- `uv run jupyter lab` — EDA (ядро из `.venv`).
- На Kaggle/Colab: `%pip install -e .[train]` — ТОЛЬКО `%pip`, не `!pip` (гарантия установки в окружение ядра). На платформах uv нет — только pip из pyproject.
- `mlflow models build-docker -m runs:/<run_id>/model -n ecup-solution:<tag>` (mlflow 2.x; в новых версиях — `container-build`) → затем `Dockerfile: FROM ecup-solution:<tag>` + `COPY run.py` + `ENTRYPOINT ["python","-u","run.py"]` (штатный entrypoint mlflow — HTTP-сервер, для соревнования нужен CLI).
- `docker push jstnoname/ecup-solution:<tag>`.

## Conventions

- polars для данных везде (не pandas/pyspark): ленивое чтение parquet, джойны 11M пар укладываются в память платформ; `.to_pandas()` только на стыке со sklearn.
- Ноутбук — оркестрация: конфиг вверху, вызовы функций из файлов. Все параметры эксперимента — в MLflow.
- Тренировка: BCE со взвешиванием (human=1.0, llm≈0.2), 1–2 эпохи, fp16.
- Валидация времени/формата — только удалённо (`scripts/emulate.py` на Kaggle/Colab), не локально.
- Тестовые данные имеют другие товары — без меморизации по id, только текст/атрибуты.

## Secrets

- `.env` (gitignored): `MLFLOW_TRACKING_URI` (Dagshub), `MLFLOW_TRACKING_USERNAME/PASSWORD`, `HF_TOKEN` — креды только для обучения. Никогда не коммитить, не печатать, не включать в образ/архив/логи.
- `HF_TOKEN`: для скачивания данных хватает read-токена (на платформах — личный токен участника).
- На платформах секреты не в ячейках: Kaggle → `kaggle_secrets`, Colab → `userdata`; локальный fallback — `.env`.
- `run.py` и `ecup/model.py` не импортируют `ecup/config.py`-загрузку секретов и не падают без `.env` (все чтения через `os.environ.get` с дефолтами).
- `.dockerignore` и сборка архива по белому списку (только `metadata.json`) — страховка от утечки.

## Gotchas

- mlflow **всех версий ≥2 требует `pandas<3`** (граф с `pandas>=3` неразрешим) — pandas в стеке держим `<3`, mlflow зафиксирован `==3.5.1` (совпадает с версией сервера Dagshub).
- `.env` должен быть **без BOM**: Windows-редакторы могут сохранить файл с UTF-8 BOM, и dotenv прочитает первый ключ как `\ufeffKEY` — креды молча не загрузятся. Проверка: первые байты файла `EF BB BF`.
- Время загрузки модели входит в лимит Check (1 мин) — веса грузить с локального файла, без HF-download в рантайме.
- Docker-образ собирается на локальной машине (Docker Desktop), артефакт модели качается из Dagshub — не на Kaggle/Colab.
- Число пар в выводе должно совпадать с числом пар на входе; `predict` — сырые скоры (не 0/1).
