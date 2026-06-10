# Guion de sustentación — 10-jun-2026

**Autores:** Santiago Espinosa · Cristian Molina
**Duración objetivo:** 10–15 min

Preparado a partir del feedback de los grupos que ya sustentaron. El profe va a profundizar especialmente en:

1. Cómo funciona K-Means
2. **La estrategia utilizada** (es el bloque más importante)
3. Diagrama de deployment
4. Demostración cargando un dataset nuevo que él entregará

---

## 1) Cómo funciona K-Means — versión "para defender ante el profe"

### El algoritmo en sí (responder esto si pregunta "explíquenme K-Means")

K-Means es un algoritmo de **clustering no supervisado**: no le decimos a qué grupo pertenece cada cliente; el algoritmo descubre los grupos a partir de la geometría de los datos.

Pasos del algoritmo:

1. **Definir k** — el número de clusters que queremos. Es el hiperparámetro principal.
2. **Inicializar k centroides** — k puntos en el espacio de features, elegidos al azar (Spark usa `k-means||`, una variante de `k-means++`, que los siembra de forma más inteligente para que la convergencia sea estable).
3. **Asignar** cada punto al centroide más cercano (distancia euclidiana al cuadrado).
4. **Mover** cada centroide al promedio de los puntos que se le asignaron en el paso anterior.
5. **Repetir** los pasos 3 y 4 hasta que las asignaciones dejen de cambiar o se llegue al `maxIter`. Esto define la convergencia.
6. **Resultado:** cada punto queda con un `cluster_id` y cada cluster tiene un centroide que representa su perfil promedio.

**Lo que K-Means optimiza** es la **suma de distancias cuadradas intra-cluster** (WCSS — *within-cluster sum of squares*). Cada iteración no aumenta esa métrica; por eso converge.

**Limitaciones que el profe puede mencionar y cómo responder:**
- *"K-Means asume clusters esféricos."* — Sí, por eso usamos `StandardScaler` antes: z-score por columna para que las 6 features estén en la misma escala. Sin escalar, `units_total` (rango 1–2860) dominaría a `recency_days` (0–181) y el algoritmo se sesgaría a una sola dimensión.
- *"¿Cómo eligieron k?"* — Por **silhouette score** (ver siguiente sección). No por método del codo, porque WCSS siempre baja con k creciente y elegir el "codo" es subjetivo; silhouette da un número comparable.
- *"¿Y la sensibilidad a la inicialización?"* — Spark usa `k-means||`, fijamos `seed=42` para reproducibilidad y corremos hasta `maxIter=50` con tolerancia 1e-4. Es estable entre ejecuciones (lo verificamos al re-correr el pipeline).

### Lo que hicimos nosotros con K-Means (esto es lo que tienen que saber al dedillo)

- **Tabla de entrada:** `dim_customer_features` — 131.186 filas, una por cliente.
- **Las 6 features** que usamos (saber explicar por qué cada una):
  1. `frequency` — número de transacciones distintas → captura *cuán seguido viene*.
  2. `units_total` — suma de unidades compradas → captura *cuánto compra en total*.
  3. `distinct_products` — diversidad de SKUs → captura *qué tan variada es la compra*.
  4. `distinct_categories` — diversidad de categorías → captura *en cuántos rubros compra*.
  5. `avg_basket_size` — `units_total / frequency` → captura *cómo es la canasta típica*.
  6. `recency_days` — días desde la última compra → captura *qué tan activo está hoy*.
- **Pipeline Spark:**
  `VectorAssembler` → `StandardScaler(withMean=True, withStd=True)` → `KMeans`.
- **Selección de k:** corremos K-Means para `k ∈ {3, 4, 5, 6}` sobre una **muestra del 10 %** de los clientes y comparamos por **silhouette score** (`pyspark.ml.evaluation.ClusteringEvaluator`). El k ganador se reentrena sobre los 131.186 clientes completos.
  - Silhouette mide para cada punto: qué tan cerca está de los puntos de su propio cluster vs. qué tan cerca está de los del cluster más cercano. Va de –1 (mal) a 1 (excelente). Lo que reportamos suele caer en 0.47–0.51, que para datos transaccionales reales es muy razonable.
- **Re-numeración por tamaño:** después de entrenar, renombramos los `cluster_id` para que **el cluster 0 sea siempre el mayoritario, el 1 el segundo, etc.** Esto estabiliza la interfaz: al re-entrenar, las etiquetas no se barajan al azar.
- **Salida (Gold):**
  - `cluster_assignments` (customer_id, cluster_id) — para uso en el dashboard.
  - `cluster_profiles` — un registro por cluster con promedio y mediana de cada feature → permite hablar de los segmentos en lenguaje de negocio.
  - `kmeans_search` — la búsqueda de k, para mostrar en el dashboard cómo se eligió.
  - El modelo `pyspark.ml` se persiste en `data/models/kmeans_*` para reuso.

### Si el profe pregunta "¿qué encontraron?"

Cinco segmentos típicos (los nombres y % varían por run; ver el dashboard en vivo):

| Cluster | Etiqueta de negocio | Cómo se distingue |
|---|---|---|
| 0 | 🟡 Ocasionales recientes | Baja frecuencia, canasta chica, recencia ≤ 30 días |
| 1 | 🟢 Regulares activos | Frecuencia media (~14 txns), canasta media, recencia baja |
| 2 | ⚫ Inactivos / dormidos | Frecuencia baja, recencia > 120 días |
| 3 | 🔵 VIPs / power users | Frecuencia alta (>30 txns), alta diversidad, recencia mínima |
| 4 | 🟠 Canasta grande | Baja frecuencia pero canasta media > 20 ítems |

**Validación cualitativa:** el cliente outlier 336296 (535 transacciones) cae correctamente en el cluster VIP.

---

## 2) La estrategia utilizada — este es el bloque crítico

> Esta sección es la que el otro grupo dijo que es la **más importante**. Hay que defenderla bien, no recitarla.

La estrategia es una cadena de decisiones de diseño concatenadas. Cada una resuelve un problema concreto del enunciado.

### 2.1 Por qué procesamiento distribuido (PySpark) y no pandas

- Cada canasta viene como una sola fila con la lista de productos pegada con espacios.
- Al hacer `explode` para tener `(transacción, producto)`, **pasamos de 1.108.987 filas a 10.591.792** — un orden de magnitud más.
- Pandas en memoria local soportaría esto, pero **el curso es de procesamiento distribuido** y la decisión es de futuro: si añaden 2 meses más, ya no entra. Spark escala a cluster sin reescribir el código de negocio.

### 2.2 Por qué medallion (Bronze / Silver / Gold)

- **Bronze = fidelidad a la fuente.** No transformamos nada de negocio. Si el profe nos pide ver el dato original, está ahí, en Parquet, particionado por `store_id`.
- **Silver = limpio y unido.** Aquí pasa el `explode`, se construye el `transaction_id` y se hace el join con el catálogo. Es la tabla "verdad operativa".
- **Gold = data marts para consumo.** Pre-agregados por dimensión (cliente, producto, categoría, día). El dashboard nunca lee Silver directamente para los KPIs — siempre Gold.
- **¿Para qué sirve esto?** Si mañana cambiamos cómo se calcula un KPI, sólo rehacemos Gold; Silver no se toca. Si cambia la fuente, sólo Bronze.

### 2.3 Por qué Parquet y no, por ejemplo, una base de datos

- Parquet es **columnar y comprimido**: las consultas analíticas leen sólo las columnas que necesitan.
- Es **particionable**: `store_id` se vuelve un directorio y filtrar por tienda no escanea las demás.
- No requiere un servicio corriendo (Postgres, MySQL): es portable, vive en el sistema de archivos.
- **DuckDB sobre Parquet** le da al dashboard la sensación de "base SQL" con latencia <100 ms, sin levantar Spark.

### 2.4 Decisiones puntuales que son nuestras (defendibles)

| Decisión | Por qué |
|---|---|
| `transaction_id = sha256(date|store|cust|product_list)` | El dataset no trae ID. Necesitamos uno **determinístico** para que re-procesar el mismo archivo no duplique. El hash sobre la canasta cumple. |
| `qty = count(*)` por `(transaction_id, product_id)` | El producto puede aparecer N veces en la lista de una canasta; interpretarlo como N unidades es la única lectura coherente con "no hay precios pero hay volumen". |
| `category = min(category_id) por product_id` | El catálogo tiene productos asignados a varias categorías. Joinear 1:N infla el conteo. Tomar el id menor es **determinístico** y no cambia las métricas relativas. |
| `StandardScaler` antes de K-Means | Las 6 features tienen rangos muy distintos. Sin escalar el algoritmo se sesga a una dimensión. |
| Re-numeración de clusters por tamaño | Estabiliza la UI: cluster 0 = mayoritario siempre. |
| FP-Growth sobre **top-200 productos**, `min_support = 0.05` | Con 449 productos y 1.1M canastas, `min_support = 0.01` revienta la JVM (OOM reproducible). Top-200 cubre el 99 % del volumen; subimos el soporte para mantener reglas accionables. |
| ALS con `coldStartStrategy='drop'` | Para no devolver NaNs cuando un cliente o producto no estaba en el entrenamiento. Documentado como limitación; fallback futuro = recomendar el top del cluster K-Means del cliente. |
| Reprocesar **todo** en cada ingesta nueva | El dataset cabe en memoria local (3 min de pipeline). Mantener delta-merge por particiones añade complejidad sin ganar nada todavía. Decisión deliberada de no sobre-ingeniería. |
| Manifest de `sha256` por archivo | Forma sencilla de detectar archivos nuevos o **modificados** (no sólo nuevos). Es el "estado" del lakehouse. |

### 2.5 La idempotencia (el profe seguro pregunta)

> *"¿Qué pasa si cargo dos veces el mismo archivo?"*

- **Bronze:** `mode("overwrite")` y partición por `store_id` → al sobrescribir queda en estado limpio.
- **Silver:** `transaction_id` es un hash determinístico → si la misma canasta entra dos veces, queda una sola tras el groupBy.
- **Gold:** se rehace desde Silver → no acumula nada.
- **Manifest:** detecta que el archivo no cambió y omite el reproceso completo.

Resultado: re-ejecutar el pipeline sobre el mismo input produce **bit a bit el mismo Gold**.

---

## 3) Diagrama de deployment

> Mostrar el diagrama del PDF (`docs/DIAGRAMA ARQUITECTURA.pdf`) en pantalla. El equivalente Mermaid está en `docs/arquitectura.md` sección 4.2.

### Qué hay que saber explicar (de izquierda a derecha)

```
┌─────────────────────────── Laptop / Workstation ────────────────────────────┐
│                                                                              │
│   Usuario / Profesor                                                         │
│         │ http://localhost:8501                                              │
│         ▼                                                                    │
│  ┌──────────────┐         ┌───────────────────────────────────────────────┐  │
│  │  Streamlit   │ ──SQL──▶│  DuckDB (in-process)                          │  │
│  │  (Python)    │         │   read_parquet('data/gold/**/*.parquet')      │  │
│  └──────┬───────┘         └───────────────────────────────────────────────┘  │
│         │ subprocess                                                         │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────┐                                │
│  │  CLI:  python -m src.pipeline.ingest     │                                │
│  └──────┬───────────────────────────────────┘                                │
│         │                                                                    │
│         ▼                                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  PySpark 3.5 — modo local[*]   (driver memory 8g)                     │   │
│  │   ├─ bronze.py  (lee CSV → Parquet raw)                                │   │
│  │   ├─ silver.py  (explode + join catálogo)                              │   │
│  │   ├─ gold.py    (data marts agregadas)                                 │   │
│  │   └─ models.py  (KMeans + FP-Growth + ALS, Spark MLlib)                │   │
│  └──────┬────────────────────────────────────────────────────────────────┘   │
│         │ R/W                                                                │
│         ▼                                                                    │
│  ┌────────────────────────── Sistema de archivos ─────────────────────────┐  │
│  │  data/landing/{Transactions,Products}/   ← entrada (drop zone)        │  │
│  │  data/landing/_manifest.json             ← estado (sha256 por archivo) │  │
│  │  data/landing/_runs.jsonl                ← log de corridas             │  │
│  │  data/bronze/                            ← Parquet raw                 │  │
│  │  data/silver/transactions_items/         ← Parquet limpio + explode    │  │
│  │  data/gold/                              ← 10 data marts               │  │
│  │  data/models/                            ← K-Means, FP-Growth, ALS     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Puntos a recalcar al mostrarlo

- **Todo corre en un solo host** (laptop). Modo `local[*]` de Spark = un proceso JVM con tantos workers como cores.
- **Una sola JVM** se levanta por etapa del pipeline. No hay cluster manager separado, no hay HDFS — el sistema de archivos local sirve como "data lake".
- **El dashboard NO levanta Spark.** Lee Parquet con DuckDB. Esto es importante porque Spark tarda ~5 s en arrancar; DuckDB es instantáneo. Por eso el dashboard se siente fluido.
- **El subprocess** desde Streamlit al CLI de ingesta es la única "frontera" entre componentes; le pasamos `cwd=ROOT` y el python del venv para que la invocación sea limpia.
- **Promoción a producción** (si el profe pregunta): el mismo código corre en un cluster YARN/Kubernetes cambiando una sola línea de `master("local[*]")` a `master("yarn")`. Los Parquet pueden vivir en S3/GCS. El dashboard se containeriza con Docker.

---

## 4) Demo: cargar un dataset nuevo

> Esto es lo que el profe quiere ver con sus propios ojos.

### Checklist previa (hacer 15 min antes de la sustentación)

```bash
cd /Users/santiagoespinosa/Documents/dataProcess/semana13/proyecto

# 1. Estado del manifest debería estar consistente — si dice "needs_run": false, vamos bien
.venv/bin/python -m src.pipeline.ingest --check

# 2. Arrancar el dashboard
make app
#   se abre en http://localhost:8501
```

Dejar la página de Streamlit lista en la sección **"Generación de nuevos resultados"**.

### Script de la demo (≈ 4 min)

1. **Mostrar el estado inicial** (30 s)
   - En el dashboard, sección "Resumen Ejecutivo": señalar el KPI de transacciones (`1.108.986`) y de clientes (`131.186`). Esto es la "foto antes de cargar el archivo".

2. **Subir el archivo del profe** (30 s)
   - Pasar a "Generación de nuevos resultados".
   - Si es un archivo de transacciones: arrastrarlo al uploader "Archivo de transacciones".
   - Confirmar que aparece: *"Guardado en `data/landing/Transactions/XXX_Tran.csv`"*.

3. **Comprobar el cambio detectado** (30 s)
   - Click en "🔍 Comprobar cambios".
   - En el log debería aparecer:
     ```json
     {
       "new": ["Transactions/XXX_Tran.csv"],
       "needs_run": true
     }
     ```

4. **Ejecutar el pipeline incremental** (≈ 2.5 min)
   - Click en "▶️ Ejecutar si hay cambios".
   - El spinner se queda activo y el log va mostrando, por orden:
     - `=== ingest: running bronze ===` (~ 7 s)
     - `=== ingest: running silver ===` (~ 10 s)
     - `=== ingest: running gold ===` (~ 20 s)
     - `=== ingest: running models ===` (~ 90 s) — aquí podemos comentar: "se está reentrenando K-Means con los datos del nuevo archivo incluidos, lo mismo para FP-Growth y ALS".

5. **Validar el resultado** (30 s)
   - Volver a "Resumen Ejecutivo".
   - Mostrar que los KPIs cambiaron (los números suben con la nueva tienda).
   - Si la nueva tienda tiene un `store_id` distinto, aparece en el sidebar de filtros automáticamente. Lo seleccionamos para mostrar las gráficas filtradas.
   - Pasar a "Segmentación de Clientes" → mostrar que los clusters se recalcularon (en `kmeans_search` puede aparecer otro k ganador, lo cual es **esperado** y deseable: el modelo se adapta a los datos nuevos).

### Plan B — si el profe NO trae archivo

Hay uno listo en `docs/demo_assets/888_Tran.csv` (tienda 888, julio 2013, 3.010 canastas sintéticas). Arrastrarlo en el uploader y seguir el mismo guion.

### Qué decir si algo falla en la demo

- **Si el dashboard se ve "atascado":** el spinner sigue activo, **es normal** — la fase `models` toma ~90 s.
- **Si el upload falla por tamaño:** Streamlit por defecto permite hasta 200 MB. Para archivos más grandes, copiar el archivo manualmente a `data/landing/Transactions/` desde la terminal y luego usar "🔍 Comprobar cambios" → el manifest lo detecta igual.
- **Si el subprocess da error:** alternar a la terminal y ejecutar `.venv/bin/python -m src.pipeline.ingest --run`. El efecto es idéntico.

---

## 5) Posibles preguntas adicionales (preparadas)

**P: ¿Por qué K-Means y no DBSCAN / jerárquico?**
R: K-Means escala bien y es el algoritmo "estándar" en MLlib. DBSCAN requiere elegir `eps` y `minPts`, y no está en `pyspark.ml`. Para el alcance del curso, K-Means con silhouette para elegir k es la decisión más defendible.

**P: ¿Por qué `implicitPrefs=True` en ALS?**
R: No tenemos ratings explícitos (no hay un "me gusta"). Sólo tenemos cantidades compradas, que son señales de **interés positivo**, no de preferencia ordenada. Eso es exactamente lo que el modelo implícito de Hu, Koren & Volinsky modela: la cantidad se trata como *confianza* en que al cliente le interesa el producto.

**P: ¿Cómo evitan duplicados al re-procesar?**
R: Manifest de `sha256` + `transaction_id` hashed + `mode("overwrite")` en todas las capas. Idempotente por construcción.

**P: ¿Cómo monitorean el pipeline?**
R: Cada corrida graba un JSON en `data/landing/_runs.jsonl` con timestamps de inicio/fin, qué archivos cambiaron y los tiempos por etapa. El dashboard lo muestra en una tabla en la página de "Generación de nuevos resultados".

**P: ¿Cuánto tarda el pipeline completo?**
R: ~3 min en la laptop de desarrollo (16 GB RAM, modo `local[*]`). Bronze 7 s, Silver 7 s, Gold 15 s, Models 90 s. El 50 % del tiempo es el reentrenamiento de los tres modelos.

**P: ¿Y si vienen 20 archivos nuevos a la vez?**
R: El manifest los detecta a todos en un solo escaneo. El pipeline se ejecuta una sola vez sobre el conjunto. No tiene problema porque Bronze lee todos los archivos con un solo `read.csv(glob)`.

**P: ¿Y si el profe quiere ver el código en vivo?**
R: Los archivos clave son:
- `src/pipeline/models.py` → K-Means + FP-Growth + ALS
- `src/pipeline/ingest.py` → ingesta incremental
- `app/streamlit_app.py` → dashboard (las 3 páginas nuevas están al final)

---

## Apertura y cierre

**Apertura (30 s):**
> "Buenos días. Soy Santiago, junto con Cristian construimos esta solución. En las dos entregas pasadas mostramos la arquitectura y el resumen ejecutivo. Hoy nos toca el análisis avanzado: segmentación con K-Means, recomendaciones con FP-Growth y ALS, y la incorporación automática de nuevos datos. Lo vamos a mostrar todo corriendo en vivo, terminando con una demo donde cargamos un archivo nuevo y vemos el pipeline reaccionar."

**Cierre (30 s):**
> "Recapitulando: tenemos una segmentación de cinco clusters que distingue claramente VIPs, regulares, ocasionales, inactivos y compradores de canasta grande; 327 reglas de asociación con lift superior a 4 entre verduras; recomendaciones personalizadas para los 131 mil clientes; y un mecanismo de ingesta incremental que detecta cambios por hash y reprocesa de forma idempotente. La solución cumple los ocho requerimientos funcionales del enunciado y los cinco no funcionales. Quedamos atentos a preguntas."
