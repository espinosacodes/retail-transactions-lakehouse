# Análisis y Modelado Analítico de Transacciones de Supermercado

**Curso:** Procesamiento Distribuido de Datos — G1
**Autores:** Santiago Espinosa · Cristian Molina
**Entrega 3 (05-jun-2026):** Código fuente ejecutable + Informe técnico
**Entrega 4 (09/10-jun-2026):** Sustentación funcional del Análisis Avanzado

## Stack

PySpark 3.5 (ETL + MLlib) → Parquet medallion Bronze/Silver/Gold → DuckDB (consulta) → Streamlit (dashboard).

## Estructura

```
proyecto/
├── data/
│   ├── landing/{Transactions,Products}/  # CSV de entrada
│   ├── bronze/                           # Parquet crudo
│   ├── silver/transactions_items/        # (transacción, producto)
│   ├── gold/                             # data marts
│   │   ├── fact_kpis/
│   │   ├── fact_sales_daily/
│   │   ├── dim_customer_features/
│   │   ├── dim_product_features/
│   │   ├── fact_category_metrics/
│   │   ├── cluster_assignments/          # ← entrega 3
│   │   ├── cluster_profiles/             # ← entrega 3
│   │   ├── kmeans_search/                # ← entrega 3
│   │   ├── product_rules/                # ← entrega 3 (FP-Growth)
│   │   └── customer_recommendations/     # ← entrega 3 (ALS)
│   └── models/                           # modelos pyspark.ml persistidos
├── src/pipeline/
│   ├── bronze.py · silver.py · gold.py
│   ├── models.py                         # K-Means + FP-Growth + ALS
│   ├── ingest.py                         # incorporación de nuevos datos (RF-8)
│   ├── run.py                            # CLI orquestador
│   └── spark_session.py · paths.py
├── app/streamlit_app.py                  # dashboard (5 páginas)
├── docs/
│   ├── arquitectura.md
│   ├── resumen_ejecutivo.md
│   └── informe_tecnico.md                # ← entrega 3
├── requirements.txt
└── Makefile
```

## Cómo correrlo

> El dataset **no** está versionado. Antes de correr el pipeline hay que dejar los CSV en `data/landing/`:
>
> ```
> data/landing/
> ├── Transactions/{102_Tran.csv, 103_Tran.csv, 107_Tran.csv, 110_Tran.csv}
> └── Products/{Categories.csv, ProductCategory.csv}
> ```
>
> Los archivos vienen del dataset del curso (separados por `|`, sin header en transacciones).

```bash
make install         # crea .venv e instala dependencias (PySpark, Streamlit, DuckDB, ...)
make pipeline        # bronze -> silver -> gold -> models (≈ 3 min sobre el dataset completo)
make app             # abre el dashboard en http://localhost:8501
```

Comandos adicionales:

```bash
make models          # sólo re-entrena K-Means + FP-Growth + ALS
make ingest-check    # reporta qué archivos en data/landing/ son nuevos / cambiaron
make ingest          # detecta cambios y relanza el pipeline si los hay
```

## Volúmenes procesados

| Capa | Filas |
|---|---|
| Landing (4 archivos CSV) | 1.108.987 canastas |
| Bronze | 1.108.987 transacciones + 50 categorías + 112.010 product↔category |
| Silver `transactions_items` | **10.591.792** filas (canasta × producto, con qty agregada) |
| Gold `fact_sales_daily` | 724 (día × tienda) |
| Gold `dim_customer_features` | 131.186 clientes |
| Gold `dim_product_features` | 449 productos |
| Gold `fact_category_metrics` | 21 categorías |
| Gold `cluster_assignments` | 131.186 clientes etiquetados con cluster_id ∈ [0..4] |
| Gold `cluster_profiles` | 5 (un perfil por cluster) |
| Gold `product_rules` | 327 reglas con `min_support=0.05, min_confidence=0.30` |
| Gold `customer_recommendations` | 131.186 × 10 = 1.31M recomendaciones top-10 por cliente |

Período cubierto: **2013-01-01 → 2013-06-30** (6 meses) · 4 tiendas (102, 103, 107, 110).

## Qué muestra el dashboard

El sidebar permite navegar entre 5 páginas y aplica filtros globales (tiendas + rango de fechas).

**1) Resumen Ejecutivo**
- KPIs: total de ventas (unidades), número de transacciones, clientes únicos, tiendas activas.
- Top 10 productos por unidades vendidas.
- Top 10 clientes por número de transacciones.
- Días pico de compra (serie de tiempo + heatmap calendario).
- Categorías más rentables (barras + pie).

**2) Visualizaciones Analíticas**
- Serie de tiempo de ventas (granularidad diaria / semanal).
- Boxplot de la distribución por cliente o categoría (escala log opcional).
- Heatmap de correlación entre 6 features de cliente (frecuencia, volumen, diversidad de productos / categorías, tamaño promedio de canasta, recencia).

**3) Segmentación de Clientes (K-Means)**
- Selección de k por silhouette score (k ∈ {3, 4, 5, 6}; ganador: k=5).
- Distribución de tamaños y perfil medio de cada cluster (heatmap normalizado).
- Etiquetado de negocio (VIP, regulares activos, ocasionales recientes, inactivos, canasta grande).
- Buscador: dado un `customer_id`, devuelve el cluster asignado y sus features.

**4) Recomendador de Productos**
- *Producto → productos asociados* (FP-Growth): reglas con soporte ≥ 5% y confianza ≥ 30% sobre el top-200 de productos.
- *Cliente → productos sugeridos* (ALS implicit): top-10 recomendaciones por cliente comparadas con su historial real.
- Tabla de las 30 reglas con mayor `lift` global.

**5) Generación de nuevos resultados**
- Subir un CSV nuevo al `data/landing/` directamente desde el dashboard.
- Botones para `--check`, `--run`, `--force` del pipeline incremental.
- Histórico de corridas leídas desde `data/landing/_runs.jsonl`.

## Reproducción del análisis avanzado

```bash
make pipeline    # ejecuta bronze → silver → gold → models en cadena
make app         # navegar al sidebar → "Segmentación de Clientes" / "Recomendador" / "Generación de nuevos resultados"
```

El informe técnico completo está en [`docs/informe_tecnico.md`](docs/informe_tecnico.md).
