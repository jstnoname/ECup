# AGENTS.md

## Project

ECup 2026 — контейнерное соревнование по матчингу товаров (дубли карточек на маркетплейсе). Код обучается/валидируется **только удалённо** (Kaggle, Colab через PyCharm); локальная машина — для EDA, вёрстки кода и сборки Docker-образа.

- Метрика: **Macro PR-AUC** по 20 категориям (`sklearn.metrics.average_precision_score` по каждой категории, затем среднее). НЕ использовать `auc()`/`roc_auc` для PR-кривой.
- Лимиты рантайма: Check 1 мин (1000 пар), Public 6 мин (~115K пар), Private 13 мин (~275K пар). Ресурсы: 20 CPU, 200GB RAM, H100 80GB.
- Архив сабмита ≤ 5GB, Docker-образ ≤ 15GB, ≤ 5 сабмитов/день. Контейнер работает **без интернета** — все библиотеки в образе, веса модели в архиве/образе.
- Данные: приватный HF-датасет **`well-please/ecup-data`** (org `well-please`, раздел Data ниже); локальная копия — `C:\Users\Alexander\Downloads\`. Файлы: `items.parquet` (13 397 761 товаров, 20 категорий), `items_human.parquet` (711 304 — подмножество items, подтверждено анти-джойном; выделен для эмуляции и human-разметки), `matches.parquet` (365 654 пары, ручная разметка 0/1, positive rate 25.7%), `matches_llm.parquet` (11 187 780 пар, LLM-таргеты 0..1). Тестовые товары не пересекаются с обучающими. Итоги EDA — `notebooks/eda.ipynb`, секция «10. Выводы анализа данных».
- Baseline организаторов (архив в `data/baseline/`, код `src/utils.py`): CLS-эмбеддинги cross-encoder MS-Marco-MiniLM-L12 (**английская модель**) + LogReg. Оценка на полных human-данных (ноутбук `02_baseline_eval`): **global AP 0.4397, macro PR-AUC 0.3635** (random 0.257, jaccard 0.319) — наш эталон для превзойти; слабые категории: ювелирка 0.112, обувь 0.135, одежда 0.182.

## Architecture

### Карта репозитория

```
ECup/
├── pyproject.toml      # 1) зависимости (core/train/dev) — единый источник, нет requirements.txt
│                       # 2) [tool.ecup.data]/[tool.ecup.model] — конфиг (пути данных; параметры
│                       #    обучения model_name, epochs, lr, веса... появятся перед 02_train)
├── uv.lock             # lock только для локальной установки (uv); платформы ставят pip из pyproject
├── .env                # секреты (Dagshub MLflow URI+креды, HF_TOKEN) — gitignored, только train-путь
├── .dockerignore       # страховка: .env/.venv/.git/данные/ноутбуки не попадают в образ
├── metadata.json       # контракт сабмита: {"image": "jstnoname/ecup-solution:V", "entry_point": "python -u run.py"}
├── run.py              # entry point соревнования: CLI-инференс (--items_path/-i, --matches_path/-m,
│                       #   --output-path/-o → submit.csv). Тонкий: без MLflow, без чтения .env
├── utility/             # общий пакет (локально: editable через uv; платформы: %pip install -e .)
│   ├── config.py        # load_config ([tool.ecup.*] через tomllib), set_secrets (подгрузка секретов
│   │                    #   в os.environ: kaggle_secrets/userdata/.env), data_dir (пути данных)
│   └── __init__.py      # Env (dataclass) + utility.load() — единая инициализация: секреты+конфиг+данные
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
  - группы: `core` (pandas<3, polars, python-dotenv, mlflow==3.5.1 — то, что реально нужно контейнеру), `dev` (jupyterlab, matplotlib, ipykernel). НЕТ requirements.txt. `train`-группа (torch, transformers, sentence-transformers...) добавится перед обучением.
  - `[tool.ecup.data]` — пути данных (data_repo и имена файлов); `[tool.ecup.model]` — параметры обучения (model_name, epochs, lr, human_weight, llm_weight, batch_size, max_len, seed...; сейчас пустой, типизация — при 02_train). Читается ноутбуком через `tomllib`, при желании переопределяется словарём в ячейке, целиком логируется в MLflow.
  - `requires-python = ">=3.11"` (на Kaggle Python 3.11 — ужесточать нельзя).
- `utility/` — общий пакет: `config.py` (`load_config` — `[tool.ecup.*]` через tomllib, `set_secrets` — подгрузка `SECRET_KEYS` в os.environ из kaggle_secrets/userdata/.env, `data_dir`), `__init__.py` (`Data`/`Model`/`Config`/`Env` dataclass + `utility.load()` — единая инициализация: секреты + конфиг + каталог данных). Данные читаются НЕ из пакета, а явно в ноутбуках (см. Data).
- `train/train_ce.py` — `fit_cross_encoder(cfg) -> mlflow run_id`; вся логика обучения в файлах, ноутбук = оркестрация.
- `run.py` — entry point соревнования: `--items_path/-i`, `--matches_path/-m`, `--output-path/-o` → `submit.csv` с колонками `id1,id2,predict` (сырые скоры, ВСЕ пары без исключения). Тонкий и чистый: без обучения, без MLflow, **без чтения `.env`**.
- `notebooks/01_eda.ipynb` (локально), `notebooks/02_train.ipynb` (Kaggle/Colab), `scripts/emulate.py` (удалённая эмуляция run.py на items_human + срез пар: валидация формата + замер времени), `metadata.json` (`{"image": "jstnoname/ecup-solution:V", "entry_point": "python -u run.py"}`).

## Data

- Источник: приватный датасет **`well-please/ecup-data`** (repo_type=dataset, private) в HF-организации `well-please`. Файлы: `items.parquet`, `items_human.parquet`, `matches.parquet`, `matches_llm.parquet` — сырые parquet, без конверсии в Arrow/HF-формат.
- **Пулы товаров не пересекаются** (вывод EDA): shared id между human- и llm-парами = 0, общих пар = 0; все llm-пары покрывают `items`, все human-пары — `items_human`. Кросс-валидация LLM/human невозможна, тестовые товары другие — модель генерализует по тексту, без памяти по id. `items_human` — подмножество `items` (анти-джойн по id = 0, подтверждено в 02_baseline_eval).
- Атрибуты (EDA): 17 200 ключей, avg 12.6 непустых атрибута/товар; топ: тип 61%, бренд 52%, страна 40%, комплектация 38%, цвет 36%. Топ ключей в items смещён в ювелирку — набор ключей не фиксировать. «нет бренда» — спец-значение отсутствия бренда.
- Доступ: членство в org `well-please`. Токены **личные у каждого участника**: для скачивания достаточно read-скоупа, для (пере)заливки датасета — write.
- Чтение данных — **явно в ноутбуках**, напрямую с HF нативным polars (`hf://`), без локальной копии и без `huggingface_hub`. Достаточно один раз вызвать `set_secrets()` (HF_TOKEN в os.environ), дальше:
  `pl.read_parquet(f"hf://datasets/{env.config.data.data_repo}/{file}.parquet")`.
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
- EDA: все простые признаки (exact_name 0.258, jaccard 0.302, атрибутные match ≈0.256) на уровне baseline 0.257 — лексика и совпадения атрибутов не разделяют классы, основная модель — семантический cross-encoder.
- Случайный sample до чтения parquet невозможен (у `LazyFrame` нет `sample` в polars 1.43); для сэмпла тяжёлых колонок — `slice(off, n)` по нескольким смещениям (см. ячейку `items_attr_compare`).

## Secrets

- `.env` (gitignored): `MLFLOW_TRACKING_URI` (Dagshub), `MLFLOW_TRACKING_USERNAME/PASSWORD`, `HF_TOKEN` — креды только для обучения. Никогда не коммитить, не печатать, не включать в образ/архив/логи.
- `HF_TOKEN`: для скачивания данных хватает read-токена (на платформах — личный токен участника).
- На платформах секреты не в ячейках: Kaggle → `kaggle_secrets`, Colab → `userdata`; локальный fallback — `.env`.
- `run.py` и `utility/model.py` не импортируют `utility/config.py`-загрузку секретов и не падают без `.env` (все чтения через `os.environ.get` с дефолтами).
- `.dockerignore` и сборка архива по белому списку (только `metadata.json`) — страховка от утечки.

## Gotchas

- mlflow **всех версий ≥2 требует `pandas<3`** (граф с `pandas>=3` неразрешим) — pandas в стеке держим `<3`, mlflow зафиксирован `==3.5.1` (совпадает с версией сервера Dagshub).
- `.env` должен быть **без BOM**: Windows-редакторы могут сохранить файл с UTF-8 BOM, и dotenv прочитает первый ключ как `\ufeffKEY` — креды молча не загрузятся. Проверка: первые байты файла `EF BB BF`.
- Время загрузки модели входит в лимит Check (1 мин) — веса грузить с локального файла, без HF-download в рантайме.
- Docker-образ собирается на локальной машине (Docker Desktop), артефакт модели качается из Dagshub — не на Kaggle/Colab.
- Число пар в выводе должно совпадать с числом пар на входе; `predict` — сырые скоры (не 0/1).
