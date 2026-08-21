# ECup 2026 — матчинг товаров

Репозиторий команды для контейнерного соревнования **ECup 2026** (ODS AI): классификация пар товаров маркетплейса на дубли. Задача — по готовым парам `(id1, id2)` из Retrieval-части определить, является ли это один и тот же товар (карточка-дубль).

Решение оформляется как Docker-контейнер, который автономно запускается на закрытых тестовых данных и сохраняет предсказания в CSV.

## Задача

- **Вход решения**: `--items_path` (товары: `id`, `name`, `attributes`, `category`) и `--matches_path` (пары `id1`, `id2`) — в виде parquet-файлов.
- **Выход**: CSV с колонками `id1, id2, predict` (сырые скоры, для ВСЕХ пар без исключения).
- **Метрика**: Macro PR-AUC по 20 категориям — `sklearn.metrics.average_precision_score` по каждой категории, затем среднее.
- **Лимиты**: Check 1 мин (1000 пар), Public 6 мин (~115K пар), Private 13 мин (~275K пар); 20 CPU, 200 GB RAM, H100 80 GB. Архив ≤ 5 GB, Docker-образ ≤ 15 GB, ≤ 5 сабмитов/день. Контейнер работает **без интернета** — все зависимости и веса уже внутри.

## Данные

- Приватный HF-датасет **`well-please/ecup-data`** (доступ — через org `well-please`, личный токен каждого участника).
- `items.parquet` — 13 397 761 товар (20 категорий); `items_human.parquet` — 711 304 товара (подмножество `items`, для эмуляции и human-разметки); `matches.parquet` — 365 654 пары с ручной разметкой 0/1 (positive rate 25.7%); `matches_llm.parquet` — 11 187 780 пар с LLM-таргетами 0..1.
- Товары, участвующие в human- и llm-парах, не пересекаются (общих id и пар — 0), поэтому прямая сверка разметок невозможна; тестовые товары тоже другие — модель генерализует по тексту.
- Публиковать данные запрещено правилами соревнования.

## Результаты EDA и baseline

Полный анализ — `notebooks/eda.ipynb`, секция «10. Выводы анализа данных»; оценка baseline организаторов — `notebooks/02_baseline_eval.ipynb`. Ключевое:

- **Простые признаки не работают**: exact_name 0.258, name_jaccard 0.302 (лучший), name_len_ratio 0.251, совпадения атрибутов ≈0.256 — все на уровне baseline 0.257 (Macro PR-AUC). Лексика и атрибуты не разделяют классы.
- **Baseline организаторов** (MS-Marco-MiniLM-L12 CLS + LogReg, полный прогон на human-данных): **global AP 0.4397, macro PR-AUC 0.3635** — эталон для превзойти; слабые категории: ювелирка 0.112, обувь 0.135, одежда 0.182.
- **Наша цель — семантический cross-encoder** на русском backbone (текст `name + attributes + category`), обучение — BCE по LLM-таргетам (11.2M пар).
- Атрибуты: 17 200 ключей, 12.6 в среднем на товар; набор ключей не фиксировать (топ смещён в ювелирку в items); «нет бренда» — спец-значение.
- Human-разметка (365K пар) — для эмуляции пайплайна и валидации скоров, пулы не пересекаются с llm.

## Результаты дистилляции (`notebooks/03_model_distillation.ipynb`)

Прогон 2026-08-21 (Colab T4), MLflow experiment `ecup` (Dagshub):

- **Student (`ai-forever/ruBert-base`, full fine-tune): macro PR-AUC 0.5117, global AP 0.5186** — на всех human-парах (365K). Baseline организаторов: 0.3635 / 0.4397 → **+14.8pp macro (+41% относительно)**.
- Teacher (`ai-forever/sbert_large_mt_nlu_ru`, frozen encoder + head): **0.3378** на выборке human 20K — слабее baseline; его soft targets сжаты (mean 0.236, std 0.095, q01..q99 = 0.082..0.456), loss студента по teacher-компоненте стоит на плато. Дистилляция сработала как регуляризация — основной вклад дал full fine-tune студента на llm-таргетах.
- Схема: учитель head-only, 2 эпохи на 80K llm-пар (сэмпл 100K, test_size 0.2), lr 1e-5, bs 16, max_len 256 → студент ruBert-base, loss = `ALPHA·BCE(teacher_soft) + (1−ALPHA)·LLM_WEIGHT·BCE(llm)` = 0.5 / 0.1.
- Стабильность: обучение в чистом fp16 даёт NaN → fp32 master weights + `autocast(fp16)` + `GradScaler` + clip 10.0.
- Артефакты: child-runs `teacher training` / `student distill` → `teacher_model/`, `student_model/` (MLmodel-формат, веса в `data/model.pth`, уже в fp16 после `.half()`).

## Архитектура

```
ECup/
├── pyproject.toml      # единственный источник зависимостей + [tool.ecup.data]/[tool.ecup.model] — конфиг
├── uv.lock             # локальный lock (uv); на платформах — pip из pyproject
├── .env                # секреты (Dagshub MLflow, HF_TOKEN) — gitignored, только train-путь
├── data/               # gitignored: baseline организаторов (data/baseline/), локальные копии данных
├── utility/            # общий пакет для ноутбуков (локально editable через uv, на платформах pip -e .)
│   ├── config.py       # load_config ([tool.ecup.*]), set_secrets (секреты в environ), data_dir
│   ├── model.py        # CrossEncoder (encoder + CLS + head), product_text(name, category, attributes)
│   ├── eval.py         # macro_pr_auc
│   └── __init__.py     # Data/Model/Config/Env + utility.load() — секреты+конфиг+каталог данных
├── notebooks/
│   ├── 01_eda.ipynb                # EDA локально (JupyterLab, polars)
│   ├── 02_baseline_eval.ipynb      # оценка baseline организаторов на human-данных
│   └── 03_model_distillation.ipynb # дистилляция teacher→student (Colab/Kaggle), артефакты → MLflow
└── models/
    └── baseline/       # Docker-решение для сабмита: самодостаточно, без utility/mlflow/.env
        ├── run.py          # entry point инференса: --items_path --matches_path --output_path
        ├── prepare_model.py    # скачивание student_model из MLflow → fp16 state_dict + токенизатор
        ├── Dockerfile      # python:3.11-slim + torch cu128; веса/токенизатор COPY внутрь образа
        ├── model/          # веса студента fp16 (gitignored, попадает в образ)
        └── tokenizer/      # токенизатор sbert_large_mt_nlu_ru (gitignored, в образе)
```

Ключевая идея: **обучение и валидация — только удалённо** (Kaggle, Colab); локальная машина — для EDA, вёрстки кода и сборки Docker-образа из MLflow-артефакта (Dagshub). `metadata.json` (`{"image": ..., "entry_point": "python -u run.py"}`) создаётся в корне при сабмите.

## Быстрый старт (локально)

```bash
uv sync --extra dev                   # установка (сейчас достаточно core+dev)
uv run jupyter lab                  # EDA (ядро из .venv)
```

## Обучение (Kaggle / Colab)

Ноутбук `notebooks/02_train.ipynb` — оркестрация:

1. `%pip install -e .[train]` — установка из pyproject (на платформах нет uv, только pip).
2. Конфиг читается из `[tool.ecup.data]` / `[tool.ecup.model]` в `pyproject.toml` (`tomllib`), при желании переопределяется в ячейке; целиком логируется в MLflow.
3. Секреты (MLflow/Dagshub, HF) — через секреты платформ (`kaggle_secrets` / `userdata`), не в ячейках.
4. Данные — чтение напрямую с HF нативным polars (`hf://`): в первой ячейке `env = utility.load()` (секреты → environ, конфиг, каталог данных), дальше `pl.read_parquet(f"hf://datasets/{env.config.data.data_repo}/{file}.parquet")`.
5. Обучение → модель логируется в MLflow (Dagshub) через `save_model` + `log_artifacts` (`log_model` на Dagshub падает 400 — см. AGENTS Gotchas) → артефакт используется для Docker-сборки.

## Сабмит

1. `python models/baseline/prepare_model.py` — скачать `student_model` из MLflow (Dagshub, креды из `.env`), перепаковать в fp16 `state_dict` → `models/baseline/model/`, скачать токенизатор → `models/baseline/tokenizer/`.
2. `docker build -t jstnoname/ecup-solution:v1 models/baseline` — образ ~8GB (torch cu128), лимит 15GB.
3. Смоук формата локально: `docker run --rm -v <данные>:/data jstnoname/ecup-solution:v1 --items_path /data/items.parquet --matches_path /data/matches.parquet --output_path /data/submit.csv`; замер времени — только удалённо (Kaggle/Colab).
4. `docker push jstnoname/ecup-solution:v1`.
5. Архив: `metadata.json` = `{"image": "jstnoname/ecup-solution:v1", "entry_point": "python -u run.py"}`.

## Для агентов

Файл `AGENTS.md` — инструкции для OpenCode/агентов: конвенции, gotcha, команды. README — для людей.
