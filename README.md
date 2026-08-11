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
- `items.parquet` — 13M товаров; `items_human.parquet` — подмножество для эмуляции; `matches.parquet` — 365K пар с ручной разметкой (0/1); `matches_llm.parquet` — 11M пар с LLM-таргетами (0..1).
- Публиковать данные запрещено правилами соревнования.

## Архитектура

```
ECup/
├── pyproject.toml      # единственный источник зависимостей (группы core/train/dev)
│                       # + [tool.ecup.data]/[tool.ecup.model] — конфиг (пути данных; параметры
│                       #   обучения model_name, epochs, lr, веса... появятся перед 02_train)
├── uv.lock             # локальный lock (uv); на платформах — pip из pyproject
├── .env                # секреты (Dagshub MLflow, HF_TOKEN) — gitignored, только train-путь
├── run.py              # entry point соревнования: --items_path/-i --matches_path/-m
│                       #   --output-path/-o → submit.csv. Тонкий, без MLflow, без .env
├── metadata.json       # контракт сабмита: {"image": "jstnoname/ecup-solution:V",
│                       #   "entry_point": "python -u run.py"}
├── utility/             # общий пакет (локально editable через uv, на платформах pip -e .)
│   ├── config.py        # load_config ([tool.ecup.*]), set_secrets (секреты в environ), data_dir
│   └── __init__.py      # Data/Model/Config/Env + utility.load() — секреты+конфиг+каталог данных
├── train/
│   └── train_ce.py     # fit_cross_encoder(cfg) -> mlflow run_id (вся логика обучения)
├── notebooks/
│   ├── 01_eda.ipynb    # EDA локально (JupyterLab, polars)
│   └── 02_train.ipynb  # оркестрация обучения на Kaggle/Colab
└── scripts/
    └── emulate.py      # удалённая эмуляция run.py: формат вывода + замер времени
```

Ключевая идея: **обучение и валидация — только удалённо** (Kaggle, Colab); локальная машина — для EDA, вёрстки кода и сборки Docker-образа из MLflow-артефакта (Dagshub).

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
5. Обучение → модель (pyfunc) логируется в MLflow (Dagshub) → артефакт используется для Docker-сборки.

## Сабмит

1. `mlflow models build-docker -m runs:/<run_id>/model -n ecup-solution:<tag>` — образ с весами и зависимостями.
2. `Dockerfile: FROM ecup-solution:<tag>` + `COPY run.py` + `ENTRYPOINT ["python","-u","run.py"]`.
3. `docker push jstnoname/ecup-solution:<tag>`.
4. Архив: `metadata.json` с указанием образа и entry point.

## Для агентов

Файл `AGENTS.md` — инструкции для OpenCode/агентов: конвенции, gotcha, команды. README — для людей.
