# AGENTS.md

## Project

ECup 2026 — контейнерное соревнование по матчингу товаров (дубли карточек на маркетплейсе). Код обучается/валидируется **только удалённо** (Kaggle, Colab через PyCharm); локальная машина — для EDA, вёрстки кода и сборки Docker-образа.

- Метрика: **Macro PR-AUC** по 20 категориям (`sklearn.metrics.average_precision_score` по каждой категории, затем среднее). НЕ использовать `auc()`/`roc_auc` для PR-кривой.
- Лимиты рантайма: Check 1 мин (1000 пар), Public 6 мин (~115K пар), Private 13 мин (~275K пар). Ресурсы: 20 CPU, 200GB RAM, H100 80GB.
- Архив сабмита ≤ 5GB, Docker-образ ≤ 15GB, ≤ 5 сабмитов/день. Контейнер работает **без интернета** — все библиотеки в образе, веса модели в архиве/образе.
- Данные: приватный HF-датасет **`well-please/ecup-data`** (org `well-please`, раздел Data ниже); локальная копия — `C:\Users\Alexander\Downloads\`. Файлы: `items.parquet` (13 397 761 товаров, 20 категорий), `items_human.parquet` (711 304 — подмножество items, подтверждено анти-джойном; выделен для эмуляции и human-разметки), `matches.parquet` (365 654 пары, ручная разметка 0/1, positive rate 25.7%), `matches_llm.parquet` (11 187 780 пар, LLM-таргеты 0..1). Тестовые товары не пересекаются с обучающими. Итоги EDA — `notebooks/01_eda.ipynb`, секция «10. Выводы анализа данных».
- Baseline организаторов (архив в `data/baseline/`, код `src/utils.py`): CLS-эмбеддинги cross-encoder MS-Marco-MiniLM-L12 (**английская модель**) + LogReg. Оценка на полных human-данных (ноутбук `02_baseline_eval`): **global AP 0.4397, macro PR-AUC 0.3635** (random 0.257, jaccard 0.319) — наш эталон для превзойти; слабые категории: ювелирка 0.112, обувь 0.135, одежда 0.182.
- Наше решение (дистилляция, `03_model_distillation`, прогон 2026-08-21): **student ruBert-base full fine-tune — macro PR-AUC 0.5117 / global AP 0.5186** на всех human-парах (365K); teacher head-only 0.3378 (выборка 20K). Схема и артефакты — README «Результаты дистилляции».
- Сабмит: первый **Success на Check — Mean PR-AUC 0.3114** (2026-08-22, `jstnoname/distill_model_solution:v6`). Контрольный тест инференс-пути расхождений не дал; зазор к 0.5117 — шум малой выборки Check (~50 пар/категорию) и/или сдвиг распределения теста, сравнивать с платформенным скором baseline организаторов на той же фазе.

## Architecture

### Карта репозитория

```
ECup/
├── pyproject.toml      # 1) зависимости репо (core/train/dev) — единый источник; у образа свой requirements.txt
│                       # 2) [tool.ecup.data]/[tool.ecup.model] — конфиг данных и параметров обучения
├── uv.lock             # lock только для локальной установки (uv); платформы ставят pip из pyproject
├── .env                # секреты (Dagshub MLflow URI+креды, HF_TOKEN) — gitignored, только train-путь
├── data/               # gitignored: baseline организаторов (data/baseline/), локальные копии данных
├── utility/            # общий пакет для ноутбуков (локально editable через uv; платформы: %pip install -e .)
│   ├── config.py       # load_config ([tool.ecup.*] через tomllib), set_secrets (kaggle_secrets/userdata/.env), data_dir
│   ├── model.py        # CrossEncoder (encoder+[CLS]+head, freeze_encoder), product_text(attr_cap=1500)
│   ├── eval.py         # macro_pr_auc(pos_min=2)
│   └── __init__.py     # Env (dataclass) + utility.load() — единая инициализация: секреты+конфиг+данные
├── notebooks/
│   ├── 01_eda.ipynb                # EDA локально (uv run jupyter lab, ядро из .venv, polars)
│   ├── 02_baseline_eval.ipynb      # оценка baseline организаторов на полных human-данных
│   └── 03_model_distillation.ipynb # дистилляция teacher→student (Colab/Kaggle): обучение → MLflow-артефакты
└── models/
    └── baseline/       # сабмит-решение: образ = окружение; код+веса дублируются в архив (без utility/mlflow/.env)
        ├── run.py          # entry point инференса: --items_path/--matches_path/--output_path
        │                   #   → CSV id1,id2,predict (ВСЕ пары); cuda-only fp16, пути /baseline/*, bucketing по длине
        ├── Dockerfile      # python:3.12.13-slim, deps из requirements.txt, БЕЗ ENTRYPOINT (команду подаёт чекер)
        ├── requirements.txt # пины: torch==2.11.0+cu128 (--extra-index-url pytorch), transformers==5.15.0, polars==1.43.2
        ├── metadata.json   # {"image": "...:vN", "entry_point": "python -u run.py"} — копия едет в корне архива
        ├── model/          # student_state.pt fp16 (~340MB) + config.json (в образе и в архиве)
        └── tokenizer/      # токенизатор sbert_large_mt_nlu_ru (в образе и в архиве)
```

- `pyproject.toml` — **единственный источник зависимостей и конфига**:
  - группы: core (pandas<3, polars, numpy, python-dotenv, mlflow==3.5.1, ipykernel, transformers>=5.15.0, torch==2.11.0), dev (jupyterlab, matplotlib, scikit-learn), train (sentence-transformers, tqdm, joblib). requirements.txt в корне репо нет; у образа свой `models/baseline/requirements.txt` с рантайм-пинами (+cu128).
  - `[tool.ecup.data]` — data_repo и имена файлов HF-датасета; `[tool.ecup.model]` — параметры обучения: student_name=`ai-forever/ruBert-base`, teacher_name/tokenizer_name=`ai-forever/sbert_large_mt_nlu_ru`, seed 69, epochs 2, batch_size 16, lr 1e-5, max_len 256, alpha 0.5, llm_weight 0.2, temperature 1.0, test_size 0.2, infer_batch_size 512. Читается ноутбуком через `utility.load()`, целиком логируется в MLflow.
  - `requires-python = ">=3.11"` (на Kaggle Python 3.11 — ужесточать нельзя).
- `utility/` — общий пакет для ноутбуков; в контейнер НЕ попадает (у `models/baseline/run.py` свои инлайн-копии `product_text` и загрузки `CrossEncoder`).
- `models/baseline/run.py` — entry point соревнования: самодостаточный инференс без обучения/MLflow/`.env`; CLI — подчёркнутые флаги `--items_path/--matches_path/--output_path` (без шорткатов); `parse_args` возвращает кортеж `(items, matches, output)`, main распаковывает; **ВСЕ пары на выходе** (отсутствующий текст → пустая строка через fill_null, счётчик пар сохраняется); **cuda-only fp16**, модель/токенизатор грузятся с абсолютных путей `/baseline/{model,tokenizer}` (не от `__file__`: чекер исполняет скрипт из своего воркспейса); lazy semi-join по нужным id (`needed.lazy()` обязателен — см. Gotchas); батчи по длине (bucketing) с восстановлением исходного порядка.
- Подготовка весов (одноразово, до build): хелперы `utility/model.py` — `download_model_from_mlflow(run_id, save_path)` → `<save_path>/model/`, `download_tokenizer_from_hf(name, save_path)` → `<save_path>/tokenizer/`; temp-каталоги самоудаляются в `finally`.
- `metadata.json` = `{"image": "jstnoname/distill_model_solution:<vN>", "entry_point": "python -u run.py"}`; лежит в `models/baseline/`, копия едет в корне архива. Теги версионные (v1…vN), уже запушенные НЕ мутировать.

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
- Подготовка весов перед сборкой — из python вызвать хелперы `utility.model`: `download_model_from_mlflow(<student_run_id>, <путь>)` и `download_tokenizer_from_hf("ai-forever/sbert_large_mt_nlu_ru", <путь>)` (креды из `.env`).
- `docker build -t jstnoname/distill_model_solution:vN models/baseline` — контекст сборки только `models/baseline/`.
- GPU-free проверка собранного образа (CUDA-мостик, инференс не запускает): `docker run --rm jstnoname/distill_model_solution:vN python -c "import torch; print(torch.__version__, torch.version.cuda)"` → ожидание `2.11.0+cu128 12.8`. Локальный инференс невозможен (cuda-only); CPU-логика (`build_pairs`) проверяется смоуком на синтетических parquet.
- `docker push jstnoname/distill_model_solution:vN`.

## Conventions

- polars для данных везде (не pandas/pyspark): ленивое чтение parquet, джойны 11M пар укладываются в память платформ; `.to_pandas()` только на стыке со sklearn.
- Ноутбук — оркестрация: конфиг вверху, вызовы функций из файлов. Все параметры эксперимента — в MLflow.
- Тренировка: fp32 master weights + `torch.autocast(fp16)` + `GradScaler` + `clip_grad_norm_(10.0)`; обучение в чистом fp16 (`.half()` на модели) даёт NaN loss.
- Валидация времени/формата — только удалённо (Kaggle/Colab: `items_human.parquet` + срез `matches`), не локально.
- Локальный инференс невозможен (cuda-only); перед сабмитом гонять CPU-смоук чистых функций (`build_pairs` на синтетических parquet) — слой ошибок дешевле ловить до попытки.
- Тестовые данные имеют другие товары — без меморизации по id, только текст/атрибуты.
- EDA: все простые признаки (exact_name 0.258, jaccard 0.302, атрибутные match ≈0.256) на уровне baseline 0.257 — лексика и совпадения атрибутов не разделяют классы, основная модель — семантический cross-encoder.
- Случайный sample до чтения parquet невозможен (у `LazyFrame` нет `sample` в polars 1.43); для сэмпла тяжёлых колонок — `slice(off, n)` по нескольким смещениям (см. ячейку `items_attr_compare`).

## Secrets

- `.env` (gitignored): `MLFLOW_TRACKING_URI` (Dagshub), `MLFLOW_TRACKING_USERNAME/PASSWORD`, `HF_TOKEN` — креды только для обучения. Никогда не коммитить, не печатать, не включать в образ/архив/логи.
- `HF_TOKEN`: для скачивания данных хватает read-токена (на платформах — личный токен участника).
- На платформах секреты не в ячейках: Kaggle → `kaggle_secrets`, Colab → `userdata`; локальный fallback — `.env`.
- `run.py` и `utility/model.py` не импортируют `utility/config.py`-загрузку секретов и не падают без `.env` (все чтения через `os.environ.get` с дефолтами).
- `.dockerignore` в `models/baseline/` (исключает prep-артефакты) — страховка от утечки; в архив кладём только нужное для рантайма (`metadata.json`, `run.py`, `model/`, `tokenizer/`), без `.env`/служебных файлов.

## Gotchas

- mlflow **всех версий ≥2 требует `pandas<3`** (граф с `pandas>=3` неразрешим) — pandas в стеке держим `<3`, mlflow зафиксирован `==3.5.1` (совпадает с версией сервера Dagshub).
- Dagshub НЕ поддерживает поле `model_id` в LogBatch: `mlflow.*.log_model` падает **400 BAD_REQUEST**. Логирование моделей — `mlflow.pytorch.save_model(dir)` + `mlflow.log_artifacts(dir, artifact_path=...)`; в рантайме веса грузим вручную (`torch.load` state_dict), mlflow в контейнер не ставим.
- Обучение в чистом fp16 (`.half()` на модели + forward) даёт NaN loss уже на 1-й эпохе — только fp32 master + autocast + GradScaler (см. Conventions).
- Веса/токенизатор (`models/baseline/model/`, `tokenizer/`) в git не идут — только в Docker-образ/архив; игнорятся корневым `.gitignore` (`models/*/model`, `models/*/tokenizer`, `**.zip` — вложенного `.gitignore` в models/baseline нет).
- `.env` должен быть **без BOM**: Windows-редакторы могут сохранить файл с UTF-8 BOM, и dotenv прочитает первый ключ как `\ufeffKEY` — креды молча не загрузятся. Проверка: первые байты файла `EF BB BF`.
- Время загрузки модели входит в лимит Check (1 мин) — веса грузить с локального файла, без HF-download в рантайме.
- Docker-образ собирается на локальной машине (Docker Desktop), артефакт модели качается из Dagshub — не на Kaggle/Colab.
- Число пар в выводе должно совпадать с числом пар на входе; `predict` — сырые скоры (не 0/1). Baseline организаторов молча дропает пары без текста — нам так нельзя.

## Контракт проверочной системы (выстрадано шестью попытками)

- **Архив возит код, образ — окружение.** Чекер распаковывает архив в `/workspace_<hash>` и исполняет `entry_point` оттуда: `run.py` ОБЯЗАТЕЛЬНО лежит в корне архива (иначе `python: can't open file .../run.py`). Веса/токенизатор дублируются в архив как страховка.
- **В образе НЕ должно быть ENTRYPOINT/CMD с командой.** Строка `entry_point` из metadata.json приходит как docker-команда поверх ENTRYPOINT → дубль (`unrecognized arguments: python -u run.py`). Правильно: Dockerfile вообще без ENTRYPOINT/CMD.
- **Пути к весам — абсолютные `/baseline/{model,tokenizer}`**, не от `__file__`: скрипт исполняется из чужого cwd.
- **Драйвер стенда держит максимум CUDA 12.8** («found version 12080»). Дефолтный PyPI-torch скомпилирован под более новую CUDA → ставить строго `torch==X+cu128` через `--extra-index-url https://download.pytorch.org/whl/cu128`. У baseline организаторов то же: база nvidia/cuda:12.8 + cuda-compat. Проверка сборки без GPU — `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- **polars ≥1.43: `LazyFrame.join` требует LazyFrame** строго (`require_same_type`): eager DataFrame → TypeError в рантайме. IDE-подсветка тут права — всегда `needed.lazy()`.
- Токенизация инференса = обучению: парный режим `tokenizer(text1_list, text2_list)`, truncation longest_first, max_len 256; `padding=True` vs `'max_length'` на скоры НЕ влияет (attention_mask маскирует), но быстрее.
- Логи чекера содержат наши print'ы (`pairs:` / `model loaded` / батчевые rates / `done:`) — инструментировать для диагностики таймингов.
