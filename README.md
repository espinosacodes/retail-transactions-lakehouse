# Análisis y Modelado Analítico de Transacciones de Supermercado

**Curso:** Procesamiento Distribuido de Datos — G1
**Autores:** Santiago Espinosa · Cristian Molina
**Entrega 2 (29-may-2026):** Resumen Ejecutivo + Visualizaciones Analíticas

## Stack

PySpark 3.5 (ETL) → Parquet medallion Bronze/Silver/Gold → DuckDB (consulta) → Streamlit (dashboard).

## Estructura

```
proyecto/
├── data/
│   ├── landing/{Transactions,Products}/  # CSV de entrada
│   ├── bronze/                           # Parquet crudo
│   ├── silver/transactions_items/        # (transacción, producto)
│   └── gold/                             # data marts
│       ├── fact_kpis/
│       ├── fact_sales_daily/
│       ├── dim_customer_features/
│       ├── dim_product_features/
│       └── fact_category_metrics/
├── src/pipeline/
│   ├── bronze.py · silver.py · gold.py · run.py
│   └── spark_session.py · paths.py
├── app/streamlit_app.py                  # dashboard
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
make install      # crea .venv e instala dependencias (PySpark, Streamlit, DuckDB, ...)
make pipeline     # bronze -> silver -> gold (≈ 1 minuto sobre el dataset completo)
make app          # abre el dashboard en http://localhost:8501
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

Período cubierto: **2013-01-01 → 2013-06-30** (6 meses) · 4 tiendas (102, 103, 107, 110).

## Qué muestra el dashboard

**Resumen Ejecutivo**
- KPIs: total de ventas (unidades), número de transacciones, clientes únicos, tiendas activas.
- Top 10 productos por unidades vendidas.
- Top 10 clientes por número de transacciones.
- Días pico de compra (serie de tiempo + heatmap calendario).
- Categorías más rentables (barras + pie).

**Visualizaciones Analíticas**
- Serie de tiempo de ventas (granularidad diaria / semanal).
- Boxplot de la distribución por cliente o categoría (escala log opcional).
- Heatmap de correlación entre 6 features de cliente (frecuencia, volumen, diversidad de productos / categorías, tamaño promedio de canasta, recencia).

Todos los gráficos respetan los filtros globales del panel lateral (tiendas y rango de fechas).
