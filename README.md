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

Полный анализ — `notebooks/01_eda.ipynb`, секция «10. Выводы анализа данных»; оценка baseline организаторов — `notebooks/02_baseline_eval.ipynb`. Ключевое:

- **Простые признаки не работают**: exact_name 0.258, name_jaccard 0.302 (лучший), name_len_ratio 0.251, совпадения атрибутов ≈0.256 — все на уровне baseline 0.257 (Macro PR-AUC). Лексика и атрибуты не разделяют классы.
- **Baseline организаторов** (MS-Marco-MiniLM-L12 CLS + LogReg, полный прогон на human-данных): **global AP 0.4397, macro PR-AUC 0.3635** — эталон для превзойти; слабые категории: ювелирка 0.112, обувь 0.135, одежда 0.182.
- **Наша цель — семантический cross-encoder** на русском backbone (текст `name + attributes + category`), обучение — LoRA fine-tune на human-метках (365K пар, BCE по `target` 0/1), опционально LLM-дозаливка через `sampling.extend`.
- Атрибуты: 17 200 ключей, 12.6 в среднем на товар; набор ключей не фиксировать (топ смещён в ювелирку в items); «нет бренда» — спец-значение.
- Human-разметка (365K пар) — для эмуляции пайплайна и валидации скоров, пулы не пересекаются с llm.

## Результаты дистилляции (`notebooks/03_model_distillation.ipynb`)

Прогон 2026-08-21 (Colab T4), MLflow experiment `ecup` (Dagshub):

- **Student (`ai-forever/ruBert-base`, full fine-tune): macro PR-AUC 0.5117, global AP 0.5186** — на всех human-парах (365K). Baseline организаторов: 0.3635 / 0.4397 → **+14.8pp macro (+41% относительно)**.
- Teacher (`ai-forever/sbert_large_mt_nlu_ru`, frozen encoder + head): **0.3378** на выборке human 20K — слабее baseline; его soft targets сжаты (mean 0.236, std 0.095, q01..q99 = 0.082..0.456), loss студента по teacher-компоненте стоит на плато. Дистилляция сработала как регуляризация — основной вклад дал full fine-tune студента на llm-таргетах.
- Схема: учитель head-only, 2 эпохи на 80K llm-пар (сэмпл 100K, test_size 0.2), lr 1e-5, bs 16, max_len 256 → студент ruBert-base, loss = `ALPHA·BCE(teacher_soft) + (1−ALPHA)·LLM_WEIGHT·BCE(llm)` = 0.5 / 0.1.
- Стабильность: обучение в чистом fp16 даёт NaN → fp32 master weights + `autocast(fp16)` + `GradScaler` + clip 10.0.
- Артефакты: child-runs `teacher training` / `student distill` → `teacher_model/`, `student_model/` (MLmodel-формат, веса в `data/model.pth`, уже в fp16 после `.half()`).

## Результаты fine-tuning (`notebooks/04_fine_tuning_2.ipynb`)

Прогон 2026-08 (Kaggle/Colab), MLflow experiment `ecup-student-fine-tuning` (Dagshub). Парадигма сменена с дистилляции на **прямое LoRA-дообучение** студента на human-разметке: классический KD не даёт выигрыша при скалярной голове (нет распределений по классам), а калибровка под целевую метрику на 365K human-парах важнее.

- **Студент**: `well-please/student_model` (backbone ruBert-base, cross-encoder CLS+head). **Учитель не используется** в лоссе — берётся только токенизатор (`well-please/teacher_model`); обучение = supervised BCE на `target` (0/1).
- **LoRA**: `r=16, alpha=32, lora_dropout=0.05`, `target_modules=['clf','query','key','value']`, `layers_to_transform=range(6,12)` (верхние 6 слоёв энкодера + классификатор).
- **Данные**: `matches.parquet` (human, 365K пар) → стратифицированный сплит по `(category1, category2, target)` через `utility.sampling.train_test_split`: **train 229 645 / test 14 910**. Батчи отсортированы по суммарной длине (length-bucketing) для ускорения.
- **Гиперпараметры**: `student_epochs=2, lr=1e-5, max_len=320, student_batch=224, seed=69`; fp32 master + `autocast(fp16)` + `GradScaler` + clip 10.0.
- **Платформенный скор**: **Public Mean PR-AUC = 0.3999233091** (сабмит v7, `models/fine-tuning`). Для сравнения: дистилляция v6 Check 0.3114 (+0.0886); офлайн human macro PR-AUC дистилляции = 0.5117 (зазор — шум малой выборки Check/Public и сдвиг распределения теста).
- Артефакты: `student.merge_and_unload().push_to_hub('well-please/student_model', private=True)`; MLflow `transformers.log_model`.

## Архитектура

```
ECup/
├── pyproject.toml      # зависимости репо + [tool.ecup.data]/[tool.ecup.model] — конфиг (у образа свой requirements.txt)
├── uv.lock             # локальный lock (uv); на платформах — pip из pyproject
├── .env                # секреты (Dagshub MLflow, HF_TOKEN) — gitignored, только train-путь
├── data/               # gitignored: baseline организаторов (data/baseline/), локальные копии данных
├── utility/            # общий пакет для ноутбуков (локально editable через uv, на платформах pip -e .)
│   ├── config.py       # load_config ([tool.ecup.*]), set_secrets (секреты в environ), data_dir
│   ├── model.py        # CrossEncoder + HFCrossEncoder.from_pretrained, product_text(attr_cap=1500, total_cap)
│   ├── sampling.py     # train_test_split (стратификация по категориям) + extend (LLM-дозаливка)
│   ├── eval.py         # macro_pr_auc(y_true, scores, categories) — строго как на платформе
│   └── __init__.py     # Data/Model/Config/Env + utility.load() — секреты+конфиг+каталог данных
├── notebooks/
│   ├── 01_eda.ipynb                   # EDA локально (JupyterLab, polars)
│   ├── 02_baseline_eval.ipynb         # оценка baseline организаторов на human-данных
│   ├── 03_model_distillation.ipynb    # дистилляция teacher→student (Colab/Kaggle), артефакты → MLflow
│   └── 04_fine_tuning_2.ipynb         # LoRA fine-tune студента на human-парах (Kaggle/Colab)
└── models/
    ├── baseline/       # историческое сабмит-решение v6 (дистилляция, пути /baseline)
    │   ├── run.py          # entry point инференса: --items_path --matches_path --output_path;
    │   │                   #   cuda-only fp16, пути весов /baseline/*, lazy semi-join, bucketing по длине
    │   ├── Dockerfile      # python:3.12.13-slim, БЕЗ ENTRYPOINT (команду подаёт проверочная система)
    │   ├── requirements.txt # пины образа: torch==2.11.0+cu128 (--extra-index-url pytorch), transformers==5.15.0, polars==1.43.2
    │   ├── metadata.json   # {"image": "...:v6", "entry_point": "python -u run.py"} — копия едет в корне архива
    │   ├── model/          # student_state.pt fp16 (~340MB) + config.json (в образе и в архиве)
    │   └── tokenizer/      # токенизатор sbert_large_mt_nlu_ru (в образе и в архиве)
    └── fine-tuning/    # активное сабмит-решение v7 (LoRA fine-tune, пути /fine-tuning)
        ├── run.py          # entry point: --items_path/--matches_path/--output_path
        │                   #   → csv id1,id2,predict; cuda-only fp16, пути /fine-tuning/*, bucketing по длине
        ├── Dockerfile      # python:3.12.13-slim, WORKDIR /fine-tuning, БЕЗ ENTRYPOINT
        ├── requirements.txt # torch==2.11.0+cu128, transformers==5.15.0, polars==1.43.2
        ├── metadata.json   # {"image": "...:v7", "entry_point": "python -u run.py"}
        ├── model/          # model.pt fp16 (~340MB) + config.json
        └── tokenizer/      # токенизатор (в образе и в архиве)
```

Ключевая идея: **обучение и валидация — только удалённо** (Kaggle, Colab); локальная машина — для EDA, вёрстки кода и сборки Docker-образа из MLflow-артефакта (Dagshub). `metadata.json` (`{"image": ..., "entry_point": "python -u run.py"}`) лежит в `models/baseline/` (исторический v6) или `models/fine-tuning/` (активный v7) и пакуется в zip-архив сабмита вместе с `run.py` и весами.

## Обучение (Kaggle / Colab)

Ноутбук `notebooks/03_model_distillation.ipynb` — оркестрация дистилляции; `notebooks/04_fine_tuning_2.ipynb` — LoRA fine-tune (активный пайплайн):

1. `%pip install -e .[train]` — установка из pyproject (на платформах нет uv, только pip). Включает `peft` и `torchao` (для LoRA).
2. Конфиг читается из `[tool.ecup.data]` / `[tool.ecup.model]` в `pyproject.toml` (`tomllib`), при желании переопределяется в ячейке; целиком логируется в MLflow.
3. Секреты (MLflow/Dagshub, HF) — через секреты платформ (`kaggle_secrets` / `userdata`), не в ячейках.
4. Данные — чтение напрямую с HF нативным polars (`hf://`): в первой ячейке `env = utility.load()` (секреты → environ, конфиг, каталог данных), дальше `pl.read_parquet(f"hf://datasets/{env.config.data.data_repo}/{file}.parquet")`.
5. Обучение → модель логируется в MLflow (Dagshub) через `save_model` + `log_artifacts` (`log_model` на Dagshub падает 400 — см. AGENTS Gotchas) → артефакт используется для Docker-сборки.

## Сабмит

### Активный пайплайн (v7, `models/fine-tuning`)

1. Веса: модель обучается в ноутбуке 04 → `push_to_hub('well-please/student_model', private)`. Экспорт: `download_model_from_mlflow` → `models/fine-tuning/model/model.pt` + `config.json`; токенизатор: `download_tokenizer_from_hf("ai-forever/sbert_large_mt_nlu_ru", ...)` → `models/fine-tuning/tokenizer/`.
2. `docker build -t jstnoname/distill_model_solution:v7 models/fine-tuning` — torch ставится строго `+cu128`: драйверы стенда держат максимум CUDA 12.8, дефолтный PyPI-wheel падает с «NVIDIA driver too old».
3. GPU-free проверка сборки: `docker run --rm jstnoname/distill_model_solution:v7 python -c "import torch; print(torch.__version__, torch.version.cuda)"` → ожидание `2.11.0+cu128 12.8`.
4. `docker push jstnoname/distill_model_solution:v7`. Теги версионные; уже запушенные не мутировать.
5. Архив (`models/fine-tuning/` → zip): `metadata.json` (v7) + `run.py` + `model/` + `tokenizer/`. Чекер распаковывает архив и запускает `entry_point` оттуда.
6. Статус: **Public Mean PR-AUC = 0.3999233091** (v7).

### Исторический пайплайн (v6, `models/baseline`)

1. `docker build -t jstnoname/distill_model_solution:v6 models/baseline`.
2. Статус: **Check Success, Mean PR-AUC 0.3114** (2026-08-22, дистилляция v6).

## Для агентов

Файл `AGENTS.md` — инструкции для OpenCode/агентов: конвенции, gotcha, команды. README — для людей.
