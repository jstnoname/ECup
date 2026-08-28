import polars as pl


def _hash_sort(df: pl.LazyFrame, by: tuple[str, ...], seed: int) -> pl.LazyFrame:
    return df \
        .with_columns(pl.struct("id1", "id2").hash(seed=seed).alias("__h")) \
        .sort("__h") \
        .drop("__h") \
        .with_columns(pl.int_range(0, pl.len()).over(list(by)).alias("_rk"))


def train_test_split(
    pairs: pl.DataFrame | pl.LazyFrame,
    strata_cols: tuple[str, ...] = ("category",),
    test_size: float = 0.2,
    seed: int = 69
) -> tuple[pl.LazyFrame | pl.DataFrame, pl.LazyFrame | pl.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    was_lazy = isinstance(pairs, pl.LazyFrame)
    src = pairs.lazy()
    by = list(strata_cols)

    items = pl.concat([src.select(pl.col("id1").alias("id")), src.select(pl.col("id2").alias("id"))]).unique()
    n_items = items.select(pl.len()).collect().item()
    n_test = int(n_items * test_size)
    if n_test < 1:
        raise ValueError(f"test_size={test_size} yields 0 test items over {n_items} items")

    test_ids = items \
        .with_columns(pl.col("id").hash(seed=seed).alias("__h")) \
        .sort("__h") \
        .slice(0, n_test) \
        .select(pl.col("id").alias("__t_id"))
    test = src \
        .join(test_ids, left_on="id1", right_on="__t_id", how="semi") \
        .join(test_ids, left_on="id2", right_on="__t_id", how="semi")
    train = src \
        .join(test_ids, left_on="id1", right_on="__t_id", how="anti") \
        .join(test_ids, left_on="id2", right_on="__t_id", how="anti")

    wants = test \
        .group_by(by) \
        .agg(pl.len()) \
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share")) \
        .select(
            *by,
            (pl.col("share") * train.select(pl.len()).collect().item()).round(0).cast(pl.Int32).alias('keep')
        )
    train = _hash_sort(train, strata_cols, seed) \
        .join(wants, on=by, how="left") \
        .with_columns(pl.col("keep").fill_null(0)) \
        .filter(pl.col("_rk") < pl.col("keep")) \
        .drop("_rk", "keep")

    if not was_lazy:
        return train.collect(), test.collect()
    return train, test


def extend(
    train: pl.DataFrame | pl.LazyFrame,
    test: pl.DataFrame | pl.LazyFrame,
    llm_pool: pl.DataFrame | pl.LazyFrame,
    strata_cols: tuple[str, ...] = ("category",),
    fill_weight: float = 0.5,
    seed: int = 69,
) -> pl.DataFrame | pl.LazyFrame:
    was_lazy = isinstance(train, pl.LazyFrame)
    train_lf, test_lf, pool_lf = train.lazy(), test.lazy(), llm_pool.lazy()
    by = list(strata_cols)

    have = train_lf.group_by(by).agg(pl.len()).rename({"len": "n"}).collect()
    shares = test_lf \
        .group_by(by).agg(pl.len()) \
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share")) \
        .select(*by, "share") \
        .collect()
    merged = have.join(shares, on=by, how="full", coalesce=True).with_columns(pl.col("n").fill_null(0))
    dead_keys = merged.filter(pl.col("share").is_null()).select(by)
    if dead_keys.height:
        print(f"sampling.extend: dropping {dead_keys.height} strata with zero test mass")
        train_lf = train_lf.join(dead_keys.lazy(), on=by, how="anti")
        merged = merged.filter(pl.col("share").is_not_null())

    k_scale = merged.select((pl.col("n") / pl.col("share")).max()).item()
    deficits = merged \
        .with_columns((pl.col("share") * k_scale).round(0).cast(pl.Int64).alias("want")) \
        .with_columns(pl.max_horizontal("want", "n").alias("want")) \
        .with_columns((pl.col("want") - pl.col("n")).clip(0).cast(pl.Int64).alias("deficit")) \
        .filter(pl.col("deficit") > 0) \
        .select(*by, "deficit")

    forbidden = pl.concat([test_lf.select(pl.col("id1").alias("id")), test_lf.select(pl.col("id2").alias("id"))]) \
        .unique() \
        .select(pl.col("id").alias("__f_id"))
    clean_pool = pool_lf \
        .join(forbidden, left_on="id1", right_on="__f_id", how="anti") \
        .join(forbidden, left_on="id2", right_on="__f_id", how="anti")
    filled = _hash_sort(clean_pool, strata_cols, seed) \
        .join(deficits.lazy(), on=by, how="inner") \
        .filter(pl.col("_rk") < pl.col("deficit")) \
        .drop("_rk", "deficit")

    human_part = train_lf.with_columns(
pl.col("target").cast(pl.Float32), pl.lit("human").alias("source"), pl.lit(1.0).alias("w")
    )
    filled = filled.with_columns(
        pl.col("target").cast(pl.Float32),
        pl.lit("llm").alias("source"),
        pl.lit(float(fill_weight)).alias("w"),
    ).select(human_part.collect_schema().names())
    result = pl.concat([human_part, filled], how="vertical")

    taken = filled.group_by(by).agg(pl.len()).rename({"len": "n"}).collect()
    short = deficits \
        .join(taken, on=by, how="left") \
        .with_columns(pl.col("n").fill_null(0)) \
        .filter(pl.col("n") < pl.col("deficit")) \
        .sort("deficit", descending=True)

    if short.height:
        print(f"sampling.extend: {short.height} strata under-filled (pool exhausted):")
        for row in short.iter_rows(named=True):
            key = ", ".join(f"{c}={row[c]!r}" for c in by)
            print(f"  [{key}] wanted {row['deficit']}, got {row['n']} "
                  f"(short {row['deficit'] - row['n']})")

    if not was_lazy:
        return result.collect()
    return result
