# Informe Técnico — Análisis y Modelado Analítico de Transacciones de Supermercado

**Curso:** Procesamiento Distribuido de Datos · G1
**Autores:** Santiago Espinosa · Cristian Molina
**Entrega:** 05 de junio de 2026
**Repositorio:** `proyecto/` (este directorio)

---

## Resumen ejecutivo del informe

Se diseñó y desarrolló una solución de datos completa que ingiere archivos CSV de transacciones de un supermercado, los procesa de forma distribuida con **Apache Spark (PySpark)** bajo una **arquitectura medallion (Bronze/Silver/Gold)**, calcula indicadores y *features* de cliente y producto, entrena tres modelos analíticos (**K-Means** para segmentación, **FP-Growth** para reglas de asociación, **ALS implícito** para recomendaciones personalizadas) y los expone en un **dashboard web Streamlit** con cinco páginas funcionales. La solución incorpora automáticamente nuevos archivos vía un módulo de **ingesta incremental** con manifest de hashes. Todo es reproducible con tres comandos: `make install`, `make pipeline`, `make app`.

---

## 1. Descripción de los datos

### 1.1 Origen

El dataset entregado por el curso se compone de seis archivos CSV con separador `|`:

| Dominio | Archivo | Registros | Esquema observado |
|---|---|---:|---|
| Catálogo | `Products/Categories.csv` | 50 | `category_id \| category_name` |
| Catálogo | `Products/ProductCategory.csv` | 112.010 | `v.Code_pr \| v.code` (producto → categoría) |
| Transacciones | `Transactions/102_Tran.csv` | 314.286 | `fecha \| tienda \| cliente \| lista_productos` |
| Transacciones | `Transactions/103_Tran.csv` | 407.130 | idem |
| Transacciones | `Transactions/107_Tran.csv` | 254.633 | idem |
| Transacciones | `Transactions/110_Tran.csv` | 132.938 | idem |
| **Total transacciones** |  | **1.108.987 canastas** |  |

### 1.2 Particularidades relevantes

- Cada fila de transacciones representa **una canasta** (cliente × tienda × fecha). El ID de transacción es **implícito** y se reconstruye en el ETL mediante un `sha2` sobre `(date, store_id, customer_id, product_list_raw)`.
- El último campo es una **lista de IDs de producto separada por espacios** que debe **explotarse** a formato largo. Tras el `explode` la tabla pasa de 1.108.987 filas (canastas) a **10.591.792 filas (canasta × producto, con `qty` agregada)** — un orden de magnitud más, lo que justifica el motor distribuido.
- **No hay precios ni montos**: todas las métricas son **relativas** — volumen (unidades), frecuencia, diversidad, recencia.
- La fecha mínima observada es `2013-01-01`, la máxima `2013-06-30` → **6 meses** (181 días).
- No hay columna de cantidad: si un `product_id` aparece *N* veces en la lista de una canasta, se interpreta como *N* unidades compradas (aplicado en Silver vía `count(lit(1))` por `transaction_id × product_id`).
- En el catálogo `ProductCategory.csv` un mismo producto puede mapear a varias categorías. Para garantizar joins 1:1 en Silver se conserva la **categoría de id más bajo** como categoría principal (criterio determinístico).
- El `store_id` también aparece en el nombre del archivo (`102_Tran.csv`), lo que se usa como llave de partición en Bronze.

### 1.3 Indicadores globales (sobre el dataset completo)

| Indicador | Valor |
|---|---:|
| Total de unidades vendidas | **10.591.793** |
| Número de transacciones | **1.108.986** |
| Clientes únicos | **131.186** |
| Productos transaccionados | **449** (de 95k en catálogo) |
| Categorías observadas | **20** (de 50 en catálogo) |
| Tiendas activas | 4 (102, 103, 107, 110) |
| Tamaño promedio de canasta | **9,55 ítems** |

> Lectura: la operación es de **alto volumen y canasta pequeña pero diversa** (~10 ítems por compra). El 99,5% del catálogo de SKUs no se vendió en el semestre y el 60% de las categorías no se movió — hallazgo importante de calidad de datos / merchandising.

---

## 2. Metodología de análisis

### 2.1 Arquitectura medallion

El procesamiento sigue una arquitectura **Bronze → Silver → Gold** materializada en Parquet sobre el sistema de archivos local.

```
data/landing/Transactions/*.csv      data/landing/Products/*.csv
              │                                  │
              ▼                                  ▼
    ┌─────────────────────────────────────────────────────┐
    │  Bronze (PySpark, particionado por store_id)        │
    │   bronze/transactions    bronze/categories          │
    │   bronze/product_category                           │
    └─────────────────────────────────────────────────────┘
              │
              ▼  explode(product_list) + join catálogo
    ┌─────────────────────────────────────────────────────┐
    │  Silver: transactions_items                          │
    │   (transaction_id, date, store_id, customer_id,      │
    │    product_id, category_id, category_name, qty)      │
    │   10.591.792 filas, particionado por store_id        │
    └─────────────────────────────────────────────────────┘
              │
              ▼  agregaciones por dimensión
    ┌─────────────────────────────────────────────────────┐
    │  Gold (data marts analíticos)                        │
    │   fact_kpis · fact_sales_daily                       │
    │   dim_customer_features · dim_product_features       │
    │   fact_category_metrics                              │
    │   cluster_assignments · cluster_profiles · kmeans_search │
    │   product_rules · customer_recommendations           │
    └─────────────────────────────────────────────────────┘
              │
              ▼
    Streamlit + DuckDB (lectura analítica sin levantar Spark)
```

### 2.2 Pipeline de ingesta

`src/pipeline/bronze.py` lee los CSV con un esquema declarado y persiste copias fieles en Parquet particionadas por `store_id`. No hay transformación de negocio en esta capa más allá del parseo de fechas y la inferencia de `store_id` desde el nombre del archivo cuando falta el campo. `mode("overwrite")` garantiza idempotencia frente a re-ejecuciones.

### 2.3 Silver

`src/pipeline/silver.py` realiza tres operaciones clave:

1. **Construcción del `transaction_id`** como `sha2(date|store_id|customer_id|product_list_raw, 256)`. Esto asegura que canastas idénticas se identifiquen igual y permite hacer joins idempotentes.
2. **Explode + agregación de cantidad**: la cadena de IDs se `split` por espacios, se `explode` y luego se `groupBy(transaction_id, product_id).count()` para producir `qty`.
3. **Enriquecimiento con catálogo**: join izquierdo contra el mapeo `product → category` (resuelto a 1:1 vía `min(category_id)` por producto) y luego contra `Categories` para traer el nombre.

Resultado: una tabla larga `transactions_items` con 10,59 M filas, particionada por `store_id` para acelerar consultas filtradas por tienda en el dashboard.

### 2.4 Gold

`src/pipeline/gold.py` calcula cinco data marts:

- `fact_kpis`: un único registro con totales globales (unidades, transacciones, clientes únicos, productos, categorías, rango de fechas).
- `fact_sales_daily`: agregado `(date, store_id) → (units, txn_count, customers)`.
- `dim_customer_features`: por cliente, **6 features de comportamiento** que alimentan K-Means:
  - `frequency`: número de transacciones distintas.
  - `units_total`: suma de unidades compradas.
  - `distinct_products`: diversidad de SKUs.
  - `distinct_categories`: diversidad de categorías.
  - `avg_basket_size`: `units_total / frequency`.
  - `recency_days`: días desde la última compra hasta `max(date)`.
- `dim_product_features`: por producto, `(units_total, txn_count, distinct_customers, category)`.
- `fact_category_metrics`: por categoría, métricas agregadas para el panel de rentabilidad relativa.

### 2.5 Modelos analíticos

`src/pipeline/models.py` entrena y persiste tres modelos:

| Modelo | Algoritmo | Entrada | Hiperparámetros relevantes | Salida persistida |
|---|---|---|---|---|
| Segmentación | **K-Means** (`pyspark.ml.clustering.KMeans`) | `dim_customer_features` con 6 columnas, escaladas con `StandardScaler` (`withMean=True, withStd=True`) | `k ∈ {3,4,5,6}` seleccionado por silhouette sobre muestra del 10%; `maxIter=50, seed=42` | `cluster_assignments`, `cluster_profiles`, `kmeans_search`, modelo en `data/models/kmeans*` |
| Recomendador producto–producto | **FP-Growth** (`pyspark.ml.fpm.FPGrowth`) | Canastas como `array<int>` desde Silver, filtradas al **top-200 productos por volumen** y tamaño 2–30 | `minSupport=0.05`, `minConfidence=0.30` | `product_rules` con `(antecedent, consequent, confidence, lift)` |
| Recomendador cliente–producto | **ALS implícito** (`pyspark.ml.recommendation.ALS`) | Matriz `(customer_id, product_id, qty)` desde Silver | `rank=16, maxIter=10, regParam=0.05, implicitPrefs=True, coldStartStrategy='drop'` | `customer_recommendations` con top-10 por cliente (con `score` y `rank`) |

#### Notas de diseño

- **K-Means con escalado**: las features tienen rangos muy distintos (`frequency` ~1–535, `units_total` ~1–2860, `recency_days` ~0–181). Sin escalar, K-Means se dominaría por `units_total`. Se usa `StandardScaler` z-score.
- **K-Means: re-etiquetado por tamaño**: los IDs nativos de cluster son arbitrarios. Tras entrenar, se renumeran de modo que `cluster_id=0` sea el mayoritario, lo que estabiliza el dashboard entre re-entrenamientos.
- **FP-Growth: poda del espacio**: el dataset original tiene 449 productos transaccionados; con 1,1 M canastas y `minSupport=0.01`, el árbol FP estalla en memoria local (OutOfMemoryError reproducible). Se filtra a top-200 productos y se sube `minSupport` a 0,05 (≈55 k canastas como umbral); con esto el modelo entrena en ~30 s y produce 327 reglas accionables.
- **ALS implícito**: la `qty` se interpreta como *confianza* del feedback positivo, no como rating explícito. `coldStartStrategy='drop'` evita NaNs en `recommendForAllUsers`.

### 2.6 Ingesta incremental (RF-8)

`src/pipeline/ingest.py` implementa la incorporación de nuevos datos como sigue:

1. Calcula el `sha256` de todos los CSV bajo `data/landing/Transactions/` y `data/landing/Products/`.
2. Compara contra `data/landing/_manifest.json` (mapping `archivo → sha256`).
3. Si hay archivos nuevos, modificados o eliminados → relanza el pipeline **completo** (bronze → silver → gold → models) y actualiza el manifest.
4. Si no hay cambios → reporta "sin novedades" y termina sin reprocesar.
5. Registra cada corrida (timestamps, qué archivos cambiaron, tiempos por etapa) en `data/landing/_runs.jsonl`.

La elección de "**relanzar todo**" en lugar de procesar diffs por partición es deliberada: Bronze ya está en modo `overwrite`, el dataset cabe en memoria local y reprocesar 6 meses toma <3 minutos. La consistencia entre marts y modelos queda garantizada por construcción, sin lógica de delta que mantener.

El dashboard expone esta funcionalidad en la página **"Generación de nuevos resultados"**: el usuario sube CSV directamente desde el navegador, dispara `ingest --check / --run / --force` y ve el log en vivo del proceso.

### 2.7 Stack tecnológico

| Capa | Herramienta | Justificación |
|---|---|---|
| Procesamiento ETL | PySpark 3.5 (modo local, `local[*]`) | Eje del curso; escala a cluster sin reescribir; SQL declarativo; particionado nativo |
| Almacenamiento | Parquet + medallion | Columnar, comprimido, particionable; lectura analítica eficiente |
| ML | Spark MLlib (KMeans, FPGrowth, ALS) | Mantiene el pipeline distribuido extremo a extremo, sin reescribir feature engineering en sklearn |
| Visualización | Streamlit | Multipágina nativa, despliegue trivial, no requiere notebook |
| Consulta dashboard | DuckDB sobre Parquet | Lecturas analíticas en milisegundos sin levantar Spark cada vez |
| Orquestación | Makefile + CLI Python (`python -m src.pipeline.run`) | Suficiente para alcance académico; sin sobre-ingeniería |

---

## 3. Principales hallazgos visuales

### 3.1 Distribución por tienda

| Tienda | Unidades | Transacciones | % del total |
|---|---:|---:|---:|
| **103** | 4.234.392 | 407.129 | **37 %** |
| 102 | 2.562.327 | 314.286 | 28 % |
| 107 | 2.410.968 | 254.633 | 23 % |
| 110 | 1.384.106 | 132.938 | 12 % |

La tienda **103 mueve 3× más volumen que la 110**. Cualquier modelo agregado debe controlar por tienda para no enmascarar comportamientos locales.

### 3.2 Top productos

Los 5 productos más vendidos están todos entre **254 k y 300 k unidades** en el semestre — sin un único "ganador" sino un núcleo de productos básicos que rotan de forma similar:

| Producto | Unidades |
|---|---:|
| Prod 5 | 300.526 |
| Prod 10 | 290.313 |
| Prod 3 | 269.855 |
| Prod 4 | 260.418 |
| Prod 6 | 254.644 |

### 3.3 Top clientes y outlier de actividad

El cliente **336296** registra **535 transacciones** en el semestre — el resto del top-10 está entre 129 y 163. Es 3 a 4× más activo que el siguiente cliente; candidato natural a verificación (cuenta empresarial?) o programa VIP.

### 3.4 Estacionalidad semanal

| Día | Transacciones | Índice (Mié = 100) |
|---|---:|---:|
| **Domingo** | 191.406 | 140 |
| **Sábado** | 189.015 | 138 |
| Viernes | 158.766 | 116 |
| Lunes | 142.445 | 104 |
| Martes | 150.739 | 110 |
| Miércoles | 137.245 | 100 |
| Jueves | 139.370 | 102 |

La operación está **fuertemente sesgada al fin de semana** (sáb + dom ~40 % más volumen que un miércoles). Día pico del semestre: **2013-06-15 (sáb)** con 9.476 transacciones. Mínimo: 2013-01-01 (martes, año nuevo) con 2.860.

### 3.5 Categorías

El supermercado es fuertemente **fresco-vegetal**: las tres categorías top son verduras y jugos. Sin embargo, **206 IDs de producto (46 % de los transaccionados) no tienen mapeo a categoría**: aparecen como "(sin categoría)" en el dashboard y forman la barra más alta del gráfico. Es un hallazgo importante de calidad de datos del catálogo.

### 3.6 Correlaciones de comportamiento de cliente

Pearson sobre las 6 features del `dim_customer_features`:

|  | freq | units | prods | cats | basket | recency |
|---|---:|---:|---:|---:|---:|---:|
| frequency | 1.00 | 0.84 | 0.73 | 0.63 | 0.17 | -0.44 |
| units_total | 0.84 | 1.00 | 0.85 | 0.69 | 0.46 | -0.39 |
| distinct_products | 0.73 | 0.85 | 1.00 | 0.69 | 0.58 | -0.46 |
| distinct_categories | 0.63 | 0.69 | 0.69 | 1.00 | 0.57 | -0.48 |
| avg_basket_size | 0.17 | 0.46 | 0.58 | 0.57 | 1.00 | -0.15 |
| recency_days | -0.44 | -0.39 | -0.46 | -0.48 | -0.15 | 1.00 |

Lecturas clave:

1. **Frecuencia y diversidad de productos correlacionan fuerte (0,73–0,85):** los clientes que vienen más también compran cosas más variadas.
2. **Tamaño de canasta es relativamente independiente de la frecuencia (0,17):** un cliente puede ser frecuente con canastas pequeñas o esporádico con canastas grandes — esto sugiere **dos modos de compra diferenciables** y motiva un k > 2 en K-Means.
3. **Recencia correlaciona negativamente con todo (-0,39 a -0,48):** los clientes recientes son los más activos. Las features tienen contenido predictivo.

---

## 4. Resultados del modelo de segmentación y recomendación

### 4.1 Segmentación de clientes con K-Means

**Selección de k** (silhouette sobre muestra del 10 % de clientes):

| k | Silhouette |
|---:|---:|
| 3 | 0,4993 |
| 4 | 0,4787 |
| **5** | **0,5063** ← elegido |
| 6 | 0,4679 |

El silhouette más alto (k=5) se reentrenó sobre los 131.186 clientes completos. Los IDs de cluster se renumeraron por tamaño descendente.

**Perfil de los 5 clusters** (promedios sobre features sin escalar):

| Cluster | n (clientes) | % | Frecuencia | Unidades totales | Productos distintos | Categorías distintas | Canasta media | Recencia (días) | Etiqueta de negocio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **0** | 51.078 | 39 % | 3,7 | 17,9 | 13,2 | 3,4 | 4,9 | 28,5 | 🟡 **Ocasionales recientes** — compran poco pero estuvieron hace ~1 mes |
| **1** | 32.023 | 24 % | 14,4 | 120,4 | 51,8 | 7,7 | 9,0 | 12,7 | 🟢 **Regulares activos** — varios sets de compra, canasta promedio, recencia baja |
| **2** | 28.001 | 21 % | 1,5 | 7,7 | 7,0 | 2,2 | 4,8 | 124,6 | ⚫ **Inactivos / dormidos** — compraron al inicio del semestre y no volvieron |
| **3** | 10.447 | 8 % | 35,1 | 432,2 | 106,4 | 11,0 | 13,8 | 4,8 | 🔵 **VIPs / power users** — alta frecuencia, alta diversidad, recencia mínima |
| **4** | 9.637 | 7 % | 4,8 | 113,4 | 53,6 | 7,5 | **24,5** | 45,3 | 🟠 **Canasta grande** — compran esporádicamente pero llenan la canasta |

**Validación cualitativa**: el cliente outlier 336296 (535 transacciones) cae correctamente en el cluster 3 (VIP). El cluster 4 (canasta grande, baja frecuencia) corrobora la lectura de la sección 3.6 sobre la baja correlación entre `frequency` y `avg_basket_size`.

**Aplicación inmediata**: las acciones recomendadas por segmento se muestran en la tabla de "Lectura de negocio" del dashboard. Resumen:

| Cluster | Acción de negocio |
|---|---|
| 0 — Ocasionales recientes | Cupón de segunda compra con vigencia corta |
| 1 — Regulares activos | Programa de puntos / fidelización estándar |
| 2 — Inactivos | Campaña de re-activación; segmento de mayor riesgo de churn |
| 3 — VIPs | Atención preferente, verificación de cuentas atípicas |
| 4 — Canasta grande | Promociones por monto mínimo / pickup de pedidos grandes |

### 4.2 Recomendador por reglas de asociación (FP-Growth)

Configuración: `minSupport=0.05` (≈ 55.450 canastas), `minConfidence=0.30`, sobre el top-200 productos por volumen, canastas de 2–30 ítems.

**327 reglas** producidas. Top-10 por `lift`:

| Antecedente | Consecuente | Confianza | Lift |
|---:|---:|---:|---:|
| Prod 1 | Prod 2 (VERDURAS DE FRUTOS) | 0,528 | **4,44** |
| Prod 2 (VERD. FRUTOS) | Prod 1 | 0,423 | 4,44 |
| Prod 16 | Prod 21 | 0,575 | 3,14 |
| Prod 3 (VERD. RAÍZ/TUBÉRC.) | Prod 21 | 0,575 | 3,14 |
| Prod 21 | Prod 16 | 0,605 | 2,94 |
| Prod 3 | Prod 16 | 0,605 | 2,94 |
| Prod 31 | Prod 3 | 0,731 | 2,90 |
| Prod 4 | Prod 7 | 0,557 | 2,62 |
| Prod 5 (AROMÁTICAS) | Prod 7 (VERD. FRUTOS) | 0,557 | 2,62 |
| Prod 21 | Prod 16 | 0,532 | 2,58 |

**Lectura**: las reglas con mayor lift cruzan verduras de raíz (papa, cebolla, zanahoria), verduras de fruto (tomate, pimentón) y aromáticas, lo cual es coherente con la lectura de la sección 3.5 (operación dominada por frescos) y refleja **comportamiento de canasta de mercado tradicional**: vegetales que se compran en conjunto para preparar comida casera. Una `lift = 4,4` indica que comprar Prod 1 multiplica por 4× la probabilidad de comprar Prod 2 respecto al baseline.

**Uso en el dashboard**: la página "Recomendador → Producto → productos asociados" permite seleccionar un `antecedent_product_id` y ver las reglas que disparan, ordenadas por lift. Sirve directamente para *cross-selling* en góndola física o en e-commerce.

### 4.3 Recomendador cliente–producto (ALS implícito)

Configuración: `rank=16, maxIter=10, regParam=0.05, implicitPrefs=True`. Top-10 productos por cliente.

**1.311.860 recomendaciones** persistidas (131.186 clientes × 10).

**Distribución de scores por rango de recomendación**:

| Rank | min(score) | max(score) | avg(score) |
|---:|---:|---:|---:|
| 1 | 0,007 | 1,475 | 0,751 |
| 2 | 0,006 | 1,373 | 0,713 |
| 5 | 0,005 | 1,314 | 0,650 |
| 10 | 0,005 | 1,217 | 0,586 |

Los scores decrecen suavemente con el rank, lo que indica que el modelo está discriminando y no produciendo recomendaciones planas.

**Validación cualitativa**: para el cliente VIP 336296, ALS recomienda los productos {18, 3, 50, 8, 41, 31, 33, 45, 47, 6}. Esto incluye Prod 3 (VERDURAS RAÍZ — la categoría #1 del supermercado) en posición #2, lo cual es consistente con un cliente cuyo historial está dominado por frescos.

**Uso en el dashboard**: la página "Recomendador → Cliente → productos sugeridos" muestra lado a lado el historial real del cliente y las recomendaciones ALS, con barras horizontales coloreadas por score.

### 4.4 Limitaciones reconocidas

- **Cold-start ALS**: clientes que no aparecen en Silver no reciben recomendaciones (`coldStartStrategy='drop'`). Estrategia futura: fallback a top-N por categoría favorita del cluster K-Means asignado al cliente.
- **FP-Growth limitado al top-200**: por restricción de memoria local. En cluster real se podría bajar `minSupport` para capturar reglas de productos de cola larga.
- **No hay split train/test**: dado el alcance académico y la ausencia de timestamps con granularidad menor a día, se entregan métricas internas (silhouette, support/confidence, score) y se valida cualitativamente. Una iteración productiva agregaría hold-out temporal.
- **Pesos K-Means**: las 6 features se ponderan igual tras el escalado. Si la operación quisiera priorizar — por ejemplo — recencia, una variante razonable sería re-escalar con pesos de negocio antes de `KMeans.fit`.

---

## 5. Conclusiones y posibles aplicaciones empresariales

### 5.1 Hallazgos principales

1. **Operación dominada por frescos y fines de semana.** Verduras y jugos representan el grueso del volumen; sábado y domingo concentran ~40 % más transacciones que un miércoles. Impacta directamente staffing, abastecimiento y rotación de inventario.
2. **Catálogo muy subutilizado.** Sólo 449 SKUs de ~95.000 catalogados se vendieron en 6 meses (0,5 %); 60 % de las categorías permanecen inactivas. Oportunidad de depuración o re-merchandising.
3. **Calidad de datos del catálogo:** 46 % de los productos transaccionados no tienen categoría asignada — pendiente de saneamiento con el cliente.
4. **Pareto clásico en clientes:** 8 % de los clientes (cluster VIP) capturan 35 % del volumen; 21 % son inactivos y 24 % son regulares activos. Hay un tercio del volumen capturable con políticas de retención dirigidas.
5. **Outlier extremo:** el cliente 336296 (535 transacciones, cluster VIP) merece revisión — posible cuenta empresarial o canal de revendedor.
6. **Asociaciones fuertes en frescos:** reglas de FP-Growth con lift > 3 entre verduras y aromáticas validan la hipótesis de "canasta de mercado tradicional" y son insumo directo para góndola.

### 5.2 Aplicaciones empresariales inmediatas

| Decisión de negocio | Insumo del análisis |
|---|---|
| **Staffing y abastecimiento por día** | Estacionalidad semanal (sáb+dom = 140 % de un miércoles) |
| **Cross-selling en góndola / e-commerce** | Reglas FP-Growth (sección 4.2): emparejar productos con lift > 2 |
| **Programa de fidelización VIP** | Cluster 3 (8 % de clientes, alta frecuencia + diversidad) |
| **Campaña de re-activación** | Cluster 2 (21 % de clientes, recencia >120 días) |
| **Recomendaciones personalizadas en app/correo** | Tabla `customer_recommendations` (top-10 ALS por cliente) |
| **Promociones por monto mínimo** | Cluster 4 (canasta grande, baja frecuencia) |
| **Saneamiento de catálogo** | 206 productos sin categoría, 95 k SKUs nunca vendidos |
| **Detección de cuentas atípicas** | Cliente 336296 (535 txns vs media de 8,45) |

### 5.3 Próximos pasos sugeridos

- **Incorporar precios** cuando estén disponibles: convierte todas las métricas relativas en *revenue real*, lo que abriría análisis de margen y métricas de elasticidad.
- **Train/test temporal** para validar la calidad de ALS: hold-out del último mes y métricas tipo `precision@10`, `recall@10`.
- **A/B test del recomendador** en una vertical pequeña (1 tienda × 1 categoría) antes de extender.
- **Re-entrenamiento incremental** de K-Means con `initialModel` para preservar la semántica de los clusters al ingerir nuevos meses.
- **Externalizar el dashboard** a una imagen Docker o desplegarlo en Streamlit Community Cloud para que el negocio lo consuma sin tocar Python.

### 5.4 Conclusión

La solución entregada cumple los requerimientos funcionales (RF-1 a RF-8) y no funcionales (escalabilidad, modularidad, reproducibilidad, idempotencia, observabilidad mínima) descritos en la propuesta de arquitectura. Es **completamente funcional fuera del notebook**: tres comandos (`make install`, `make pipeline`, `make app`) producen un dashboard web navegable con KPIs, visualizaciones analíticas, segmentación de clientes, recomendaciones personalizadas y ruta de ingesta de nuevos datos. El stack Spark + Parquet + DuckDB + Streamlit es defensible como solución académica y a la vez ofrece un camino claro de promoción a producción (cluster Spark / Delta Lake / contenedores) sin reescribir lógica de negocio.

---

## 6. Reproducción end-to-end

```bash
# 1) Setup
make install              # crea .venv e instala dependencias

# 2) Pipeline completo (≈ 3 min en una laptop con 16 GB RAM)
make pipeline             # bronze → silver → gold → models

# 3) Dashboard
make app                  # abre http://localhost:8501

# 4) Demostrar incorporación de nuevos datos (RF-8)
#    - Copiar un CSV adicional a data/landing/Transactions/
#    - O subirlo desde la página "Generación de nuevos resultados"
make ingest-check         # ver qué cambió
make ingest               # ejecutar pipeline si hay cambios
```

Las tablas Gold quedan en `data/gold/`. Los modelos pyspark.ml persistidos quedan en `data/models/`. El log de corridas de ingesta queda en `data/landing/_runs.jsonl`.

---

**Anexos:**

- [Documento de arquitectura completo](arquitectura.md)
- [Resumen ejecutivo de la entrega 2](resumen_ejecutivo.md)
- Código fuente: `src/pipeline/{bronze,silver,gold,models,ingest,run}.py`
- Dashboard: `app/streamlit_app.py`
