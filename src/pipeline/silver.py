"""Silver: explota la canasta a (transacción, producto) y enriquece con catálogo."""
from __future__ import annotations

from pyspark.sql import functions as F

from .paths import BRONZE, SILVER
from .spark_session import get_spark


def run():
    spark = get_spark("silver")

    tx_bronze = spark.read.parquet(str(BRONZE / "transactions"))
    cats = spark.read.parquet(str(BRONZE / "categories"))
    pc_raw = spark.read.parquet(str(BRONZE / "product_category"))

    # En el catálogo un producto puede mapear a varias categorías.
    # Para evitar duplicar filas al hacer el join, nos quedamos con la categoría
    # de id más bajo como categoría principal (criterio determinístico).
    pc = (
        pc_raw
        .groupBy("product_id")
        .agg(F.min("category_id").alias("category_id"))
    )

    transactions = (
        tx_bronze
        .filter(F.col("date").isNotNull() & F.col("product_list_raw").isNotNull())
        .withColumn(
            "transaction_id",
            F.sha2(
                F.concat_ws("|",
                            F.col("date").cast("string"),
                            F.col("store_id").cast("string"),
                            F.col("customer_id").cast("string"),
                            F.col("product_list_raw")),
                256,
            ),
        )
        .select(
            "transaction_id",
            "date",
            "store_id",
            "customer_id",
            "product_list_raw",
        )
    )

    items = (
        transactions
        .withColumn("product_id_str", F.explode(F.split(F.col("product_list_raw"), r"\s+")))
        .withColumn("product_id_str", F.trim("product_id_str"))
        .filter(F.col("product_id_str") != "")
        .withColumn("product_id", F.col("product_id_str").cast("int"))
        .drop("product_id_str", "product_list_raw")
        .groupBy("transaction_id", "date", "store_id", "customer_id", "product_id")
        .agg(F.count(F.lit(1)).alias("qty"))
    )

    enriched = (
        items
        .join(pc, "product_id", "left")
        .join(cats, "category_id", "left")
        .select(
            "transaction_id",
            "date",
            "store_id",
            "customer_id",
            "product_id",
            "category_id",
            "category_name",
            "qty",
        )
    )

    (enriched.write.mode("overwrite")
        .partitionBy("store_id")
        .parquet(str(SILVER / "transactions_items")))

    n = enriched.count()
    print(f"[silver] transactions_items written: {n:,} rows")
    spark.stop()


if __name__ == "__main__":
    run()
