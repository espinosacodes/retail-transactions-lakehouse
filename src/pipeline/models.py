"""Models: segmentación K-Means + recomendador (FP-Growth y ALS).

Produce tres data marts adicionales en Gold:

    gold/cluster_assignments       (customer_id, cluster_id)
    gold/cluster_profiles          (cluster_id, n_customers, métricas medias)
    gold/product_rules             (antecedent, consequent, support, confidence, lift)
    gold/customer_recommendations  (customer_id, product_id, score, rank)

Además persiste los modelos entrenados en `data/models/` (pyspark.ml savers)
para que el módulo de ingesta los pueda recargar al regenerar resultados.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.fpm import FPGrowth, FPGrowthModel
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, functions as F

from .paths import DATA, GOLD, SILVER
from .spark_session import get_spark


MODELS_DIR = DATA / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_FEATURES = [
    "frequency",
    "units_total",
    "distinct_products",
    "distinct_categories",
    "avg_basket_size",
    "recency_days",
]

# Valores por defecto: configurables vía args si llegase a hacer falta.
KMEANS_K_RANGE = (3, 4, 5, 6)
# FP-Growth opera sobre ~1.1M canastas. Un min_support muy bajo dispara una explosión
# combinatoria del árbol de candidatos. Con 5% (≈55k canastas) seguimos obteniendo
# reglas accionables y mantenemos el costo bajo control en un nodo único.
FP_MIN_SUPPORT = 0.05
FP_MIN_CONFIDENCE = 0.30
FP_MAX_BASKET_SIZE = 30        # evita canastas patológicas que inflan FP-tree
FP_TOP_N_PRODUCTS = 200        # nos quedamos con los 200 productos más vendidos
ALS_RANK = 16
ALS_MAX_ITER = 10
ALS_REG_PARAM = 0.05
ALS_TOP_N = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _overwrite_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _silhouette(model: KMeansModel, df: DataFrame, features_col: str = "features",
                prediction_col: str = "cluster_id") -> float:
    preds = model.transform(df)
    evaluator = ClusteringEvaluator(
        predictionCol=prediction_col,
        featuresCol=features_col,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    return float(evaluator.evaluate(preds))


# ---------------------------------------------------------------------------
# Segmentación de clientes con K-Means
# ---------------------------------------------------------------------------
def run_kmeans(spark) -> None:
    print("[models] segmentación K-Means → leyendo dim_customer_features ...")
    customers = spark.read.parquet(str(GOLD / "dim_customer_features"))

    assembler = VectorAssembler(inputCols=CLUSTER_FEATURES, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)
    prep = Pipeline(stages=[assembler, scaler]).fit(customers)
    feats = prep.transform(customers).select("customer_id", "features", *CLUSTER_FEATURES).cache()

    # Selección de k por silhouette sobre una muestra (acelera la evaluación).
    sample = feats.sample(False, 0.10, seed=42).cache()
    best = None
    scores = []
    for k in KMEANS_K_RANGE:
        km = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster_id",
                    maxIter=30, tol=1e-4)
        model = km.fit(sample)
        sil = _silhouette(model, sample)
        scores.append((k, sil))
        print(f"[models]   k={k}  silhouette={sil:.4f}")
        if best is None or sil > best[1]:
            best = (k, sil, model)
    sample.unpersist()

    k_best, sil_best, _ = best
    print(f"[models] k seleccionado = {k_best} (silhouette={sil_best:.4f})")

    # Reentrenamos sobre el dataset completo con el k ganador para obtener asignaciones definitivas.
    final = KMeans(k=k_best, seed=42, featuresCol="features", predictionCol="cluster_id",
                   maxIter=50, tol=1e-4).fit(feats)
    assignments = final.transform(feats).select("customer_id", "cluster_id",
                                                *CLUSTER_FEATURES)

    # Re-ordenamos cluster_id por tamaño descendente para que el "cluster 0" sea siempre el mayoritario.
    sizes = (assignments.groupBy("cluster_id").count()
             .orderBy(F.col("count").desc()).collect())
    remap = {row["cluster_id"]: i for i, row in enumerate(sizes)}
    remap_expr = F.create_map([F.lit(x) for kv in remap.items() for x in kv])
    assignments = assignments.withColumn("cluster_id", remap_expr[F.col("cluster_id")])

    _overwrite_dir(GOLD / "cluster_assignments")
    (assignments.select("customer_id", "cluster_id")
        .coalesce(4).write.mode("overwrite").parquet(str(GOLD / "cluster_assignments")))

    # Perfil de cada cluster (medias).
    profiles = (
        assignments.groupBy("cluster_id")
        .agg(
            F.count("*").alias("n_customers"),
            *[F.avg(F.col(c)).alias(f"avg_{c}") for c in CLUSTER_FEATURES],
            *[F.expr(f"percentile_approx({c}, 0.5)").alias(f"median_{c}")
              for c in CLUSTER_FEATURES],
        )
        .orderBy("cluster_id")
    )

    _overwrite_dir(GOLD / "cluster_profiles")
    (profiles.coalesce(1).write.mode("overwrite").parquet(str(GOLD / "cluster_profiles")))

    # Histórico de la búsqueda de k (silhouette por k) para mostrarlo en el dashboard.
    sil_df = spark.createDataFrame(
        [(int(k), float(s)) for k, s in scores] + [(-1, float(sil_best))],  # -1 = best (marker)
        ["k", "silhouette"],
    )
    _overwrite_dir(GOLD / "kmeans_search")
    (sil_df.coalesce(1).write.mode("overwrite").parquet(str(GOLD / "kmeans_search")))

    # Persistimos el modelo escalado para reuso al ingerir datos nuevos.
    model_path = MODELS_DIR / "kmeans_pipeline"
    if model_path.exists():
        shutil.rmtree(model_path)
    prep.write().overwrite().save(str(MODELS_DIR / "kmeans_preprocessor"))
    final.write().overwrite().save(str(model_path))

    feats.unpersist()
    print(f"[models] cluster_assignments y cluster_profiles escritos (k={k_best})")


# ---------------------------------------------------------------------------
# Recomendador por canasta (FP-Growth)
# ---------------------------------------------------------------------------
def run_fpgrowth(spark) -> None:
    print("[models] FP-Growth → construyendo canastas ...")
    items = spark.read.parquet(str(SILVER / "transactions_items"))

    # Limitamos al top-N de productos por volumen: esto poda fuertemente el espacio
    # de búsqueda sin perder señal (los productos en la "long tail" no formarían
    # reglas con soporte ≥ FP_MIN_SUPPORT de todas formas).
    top_products = (
        items.groupBy("product_id")
        .agg(F.sum("qty").alias("units"))
        .orderBy(F.col("units").desc())
        .limit(FP_TOP_N_PRODUCTS)
        .select("product_id")
    )

    filtered = items.join(F.broadcast(top_products), "product_id", "inner")

    baskets = (
        filtered.groupBy("transaction_id")
        .agg(F.collect_set("product_id").alias("items"))
        .filter((F.size("items") >= 2) & (F.size("items") <= FP_MAX_BASKET_SIZE))
    )

    print(f"[models] FP-Growth: top-{FP_TOP_N_PRODUCTS} productos, "
          f"min_support={FP_MIN_SUPPORT}, min_confidence={FP_MIN_CONFIDENCE}")

    fp = FPGrowth(itemsCol="items",
                  minSupport=FP_MIN_SUPPORT,
                  minConfidence=FP_MIN_CONFIDENCE)
    model = fp.fit(baskets)

    rules = model.associationRules
    # FP-Growth devuelve antecedent/consequent como array<int>; los pasamos a
    # filas (product_id antecedente, product_id consecuente) para que sea trivial
    # consultarlas desde el dashboard.
    flat = (
        rules
        .withColumn("antecedent_product_id", F.explode("antecedent"))
        .withColumn("consequent_product_id", F.explode("consequent"))
        .select("antecedent_product_id", "consequent_product_id",
                F.col("confidence").cast("double"),
                F.col("lift").cast("double"))
    )

    # Enriquecemos con métricas de producto para mostrarlas bonito.
    prod = spark.read.parquet(str(GOLD / "dim_product_features"))
    enriched = (
        flat
        .join(prod.select(F.col("product_id").alias("consequent_product_id"),
                           F.col("category_name").alias("consequent_category")),
              "consequent_product_id", "left")
        .join(prod.select(F.col("product_id").alias("antecedent_product_id"),
                           F.col("category_name").alias("antecedent_category")),
              "antecedent_product_id", "left")
        .select(
            "antecedent_product_id", "antecedent_category",
            "consequent_product_id", "consequent_category",
            "confidence", "lift",
        )
    )

    _overwrite_dir(GOLD / "product_rules")
    (enriched.coalesce(2).write.mode("overwrite").parquet(str(GOLD / "product_rules")))

    n_rules = enriched.count()
    print(f"[models] FP-Growth: {n_rules:,} reglas (min_sup={FP_MIN_SUPPORT}, "
          f"min_conf={FP_MIN_CONFIDENCE})")

    model_path = MODELS_DIR / "fpgrowth"
    if model_path.exists():
        shutil.rmtree(model_path)
    model.write().overwrite().save(str(model_path))


# ---------------------------------------------------------------------------
# Recomendador cliente→producto (ALS implicit)
# ---------------------------------------------------------------------------
def run_als(spark) -> None:
    print("[models] ALS implicit → matriz cliente×producto ...")
    items = spark.read.parquet(str(SILVER / "transactions_items"))

    interactions = (
        items.groupBy("customer_id", "product_id")
        .agg(F.sum("qty").cast("double").alias("rating"))
    )

    als = ALS(
        userCol="customer_id",
        itemCol="product_id",
        ratingCol="rating",
        rank=ALS_RANK,
        maxIter=ALS_MAX_ITER,
        regParam=ALS_REG_PARAM,
        implicitPrefs=True,
        coldStartStrategy="drop",
        seed=42,
    )
    model = als.fit(interactions)

    # top-N recomendaciones por cliente.
    top_per_user = model.recommendForAllUsers(ALS_TOP_N)
    exploded = (
        top_per_user
        .withColumn("rec", F.explode("recommendations"))
        .select(
            F.col("customer_id"),
            F.col("rec.product_id").alias("product_id"),
            F.col("rec.rating").alias("score"),
        )
    )
    # Rango por cliente (1..N).
    from pyspark.sql.window import Window
    w = Window.partitionBy("customer_id").orderBy(F.col("score").desc())
    ranked = exploded.withColumn("rank", F.row_number().over(w))

    _overwrite_dir(GOLD / "customer_recommendations")
    (ranked.coalesce(4).write.mode("overwrite").parquet(str(GOLD / "customer_recommendations")))

    print(f"[models] ALS: top-{ALS_TOP_N} recomendaciones por cliente persistidas")

    model_path = MODELS_DIR / "als"
    if model_path.exists():
        shutil.rmtree(model_path)
    model.write().overwrite().save(str(model_path))


def run() -> None:
    spark = get_spark("models")
    try:
        run_kmeans(spark)
        run_fpgrowth(spark)
        run_als(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
