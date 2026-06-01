"""Dashboard de análisis de transacciones de supermercado.

Lee las tablas Gold (Parquet) con DuckDB y las muestra en dos páginas:
1) Resumen Ejecutivo
2) Visualizaciones Analíticas
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
SILVER = ROOT / "data" / "silver"

st.set_page_config(
    page_title="Supermercado · Analítica",
    layout="wide",
)


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    optional = {
        "cluster_assignments",
        "cluster_profiles",
        "kmeans_search",
        "product_rules",
        "customer_recommendations",
    }
    # vistas perezosas sobre los Parquet en disco
    for table in [
        "fact_kpis",
        "fact_sales_daily",
        "dim_customer_features",
        "dim_product_features",
        "fact_category_metrics",
        "cluster_assignments",
        "cluster_profiles",
        "kmeans_search",
        "product_rules",
        "customer_recommendations",
    ]:
        path = GOLD / table
        if not path.exists():
            if table in optional:
                continue
        con.execute(
            f"create view {table} as select * from read_parquet('{path}/**/*.parquet')"
        )
    items_path = SILVER / "transactions_items"
    con.execute(
        f"create view items as select * from read_parquet('{items_path}/**/*.parquet')"
    )
    return con


def _has_table(name: str) -> bool:
    return (GOLD / name).exists()


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    return get_con().execute(sql).df()


# ============================================================
# Sidebar — filtros globales
# ============================================================
st.sidebar.title("Supermercado")
st.sidebar.caption("Análisis y Modelado Analítico de Transacciones")
st.sidebar.markdown("**Autores:** Santiago Espinosa · Cristian Molina")
st.sidebar.divider()

kpis = q("select * from fact_kpis").iloc[0]
date_min = pd.to_datetime(kpis["date_min"]).date()
date_max = pd.to_datetime(kpis["date_max"]).date()

stores = sorted([int(s) for s in q("select distinct store_id from fact_sales_daily order by store_id")["store_id"]])
selected_stores = st.sidebar.multiselect("Tiendas", options=stores, default=stores)
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = date_range
else:
    d0, d1 = date_min, date_max

store_filter_sql = f"store_id in ({','.join(str(s) for s in selected_stores)})" if selected_stores else "1=0"
date_filter_sql = f"date between DATE '{d0}' and DATE '{d1}'"

page = st.sidebar.radio(
    "Sección",
    options=[
        "Resumen Ejecutivo",
        "Visualizaciones Analíticas",
        "Segmentación de Clientes",
        "Recomendador de Productos",
        "Generación de nuevos resultados",
    ],
    index=0,
)


# ============================================================
# Helpers
# ============================================================
def kpi_card(col, label: str, value, helptext: str | None = None):
    with col:
        st.metric(label=label, value=value, help=helptext)


def filtered(table: str, store_col: str = "store_id", date_col: str = "date") -> str:
    return f"select * from {table} where {date_col} {date_filter_sql.split('date ')[1]} and {store_col} in ({','.join(str(s) for s in selected_stores) or 'NULL'})"


# ============================================================
# Página 1 — Resumen Ejecutivo
# ============================================================
if page == "Resumen Ejecutivo":
    st.title("Resumen Ejecutivo")
    st.caption(f"Período disponible: **{date_min} → {date_max}** · Tiendas: {', '.join(map(str, stores))}")

    if not selected_stores:
        st.warning("Selecciona al menos una tienda en el panel izquierdo.")
        st.stop()

    # --- KPI cards (filtrados) ---
    base = q(f"""
        select
            sum(units) as total_units,
            sum(txn_count) as total_transactions,
            sum(customers) as total_customers_naive
        from fact_sales_daily
        where {date_filter_sql} and {store_filter_sql}
    """).iloc[0]

    distinct_customers = q(f"""
        select count(distinct customer_id) as n
        from items
        where {date_filter_sql} and {store_filter_sql}
    """).iloc[0]["n"]

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Total de ventas (unidades)", f"{int(base['total_units']):,}",
             "Suma de cantidades vendidas en el período filtrado.")
    kpi_card(c2, "Número de transacciones", f"{int(base['total_transactions']):,}",
             "Conteo de canastas únicas (cliente × tienda × fecha).")
    kpi_card(c3, "Clientes únicos", f"{int(distinct_customers):,}")
    kpi_card(c4, "Tiendas activas", f"{len(selected_stores)}")

    st.divider()

    # --- Top 10 productos y Top 10 clientes ---
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top 10 productos por unidades vendidas")
        top_p = q(f"""
            select cast(i.product_id as varchar) as pid,
                   coalesce(p.category_name, '(sin categoría)') as categoria,
                   sum(i.qty) as unidades,
                   count(distinct i.transaction_id) as transacciones
            from items i
            left join (select distinct product_id, category_name from dim_product_features) p
                on i.product_id = p.product_id
            where {date_filter_sql.replace('date', 'i.date')}
              and i.store_id in ({','.join(str(s) for s in selected_stores)})
            group by 1, 2
            order by unidades desc
            limit 10
        """)
        top_p["producto"] = "Prod " + top_p["pid"].astype(str) + " · " + top_p["categoria"].astype(str)
        order_p = top_p.sort_values("unidades")["producto"].tolist()
        fig = px.bar(top_p.sort_values("unidades"), x="unidades", y="producto",
                     orientation="h", text="unidades",
                     color="unidades", color_continuous_scale="Blues",
                     category_orders={"producto": order_p},
                     hover_data={"transacciones": ":,", "categoria": True, "pid": True})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis_title="Producto · Categoría", xaxis_title="Unidades", height=460,
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(type="category"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Top 10 clientes por número de transacciones")
        top_c = q(f"""
            select cast(customer_id as varchar) as cliente,
                   count(distinct transaction_id) as transacciones,
                   sum(qty) as unidades
            from items
            where {date_filter_sql} and {store_filter_sql}
            group by customer_id
            order by transacciones desc
            limit 10
        """)
        top_c["cliente"] = "Cli " + top_c["cliente"].astype(str)
        order_c = top_c.sort_values("transacciones")["cliente"].tolist()
        fig = px.bar(top_c.sort_values("transacciones"), x="transacciones", y="cliente",
                     orientation="h", text="transacciones",
                     color="transacciones", color_continuous_scale="Greens",
                     category_orders={"cliente": order_c})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis_title="Cliente", xaxis_title="Transacciones", height=420,
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(type="category"))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Días pico de compra ---
    st.subheader("Días pico de compra")
    tabs = st.tabs(["Serie de tiempo", "Heatmap diario"])

    with tabs[0]:
        ts = q(f"""
            select date, sum(txn_count) as transacciones, sum(units) as unidades
            from fact_sales_daily
            where {date_filter_sql} and {store_filter_sql}
            group by date
            order by date
        """)
        fig = px.line(ts, x="date", y="transacciones", markers=False,
                      labels={"date": "Fecha", "transacciones": "Transacciones"})
        # marca el top-5 días
        top_days = ts.nlargest(5, "transacciones")
        fig.add_scatter(x=top_days["date"], y=top_days["transacciones"],
                        mode="markers", marker=dict(size=10, color="red"),
                        name="Top 5 días")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        cal = q(f"""
            select date,
                   dayofweek(date) as dow,
                   weekofyear(date) as week_n,
                   sum(txn_count) as transacciones
            from fact_sales_daily
            where {date_filter_sql} and {store_filter_sql}
            group by date
        """)
        dow_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        cal["dow_name"] = cal["dow"].apply(lambda i: dow_names[(int(i) - 1) % 7])
        pivot = cal.pivot_table(index="dow_name", columns="week_n",
                                values="transacciones", aggfunc="sum")
        pivot = pivot.reindex(dow_names)
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                        labels=dict(x="Semana del año", y="Día de la semana",
                                    color="Transacciones"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Categorías más "rentables" (volumen + frecuencia) ---
    st.subheader("Categorías más rentables (proxy: volumen y frecuencia)")
    cat_df = q(f"""
        select
            coalesce(c.category_name, '(sin categoría)') as categoria,
            sum(i.qty) as unidades,
            count(distinct i.transaction_id) as transacciones,
            count(distinct i.customer_id) as clientes
        from items i
        left join (select distinct category_id, category_name
                   from dim_product_features
                   where category_name is not null) c
            on i.category_id = c.category_id
        where {date_filter_sql.replace('date', 'i.date')}
          and i.store_id in ({','.join(str(s) for s in selected_stores)})
        group by 1
        order by unidades desc
    """)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig = px.bar(cat_df.head(15).sort_values("unidades"),
                     x="unidades", y="categoria", orientation="h",
                     color="unidades", color_continuous_scale="Purples",
                     hover_data=["transacciones", "clientes"])
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=480, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="", xaxis_title="Unidades")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig_pie = px.pie(cat_df.head(8), values="unidades", names="categoria",
                         hole=0.45)
        fig_pie.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                              showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================
# Página 2 — Visualizaciones Analíticas
# ============================================================
elif page == "Visualizaciones Analíticas":
    st.title("Visualizaciones Analíticas")
    st.caption("Exploración de estructura y comportamiento de los datos.")

    if not selected_stores:
        st.warning("Selecciona al menos una tienda en el panel izquierdo.")
        st.stop()

    # --- Serie de tiempo ---
    st.subheader("Serie de tiempo · Ventas por día y semana")
    col_g, col_m = st.columns(2)
    granularity = col_g.radio("Granularidad", ["Diaria", "Semanal"], horizontal=True)
    metric_choice = col_m.radio("Métrica", ["Transacciones", "Unidades", "Ambas (ejes separados)"],
                                horizontal=True)

    if granularity == "Diaria":
        ts = q(f"""
            select date, sum(units) as unidades, sum(txn_count) as transacciones
            from fact_sales_daily
            where {date_filter_sql} and {store_filter_sql}
            group by date order by date
        """)
        x = "date"
    else:
        ts = q(f"""
            select date_trunc('week', date) as semana,
                   sum(units) as unidades,
                   sum(txn_count) as transacciones
            from fact_sales_daily
            where {date_filter_sql} and {store_filter_sql}
            group by 1 order by 1
        """)
        x = "semana"

    if metric_choice == "Transacciones":
        fig = px.line(ts, x=x, y="transacciones",
                      labels={x: "Período", "transacciones": "Transacciones"})
        # marca los 5 días pico
        top_pts = ts.nlargest(5, "transacciones")
        fig.add_scatter(x=top_pts[x], y=top_pts["transacciones"],
                        mode="markers", marker=dict(size=10, color="red"),
                        name="Top 5 días")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          showlegend=True)
    elif metric_choice == "Unidades":
        fig = px.line(ts, x=x, y="unidades",
                      labels={x: "Período", "unidades": "Unidades"})
        top_pts = ts.nlargest(5, "unidades")
        fig.add_scatter(x=top_pts[x], y=top_pts["unidades"],
                        mode="markers", marker=dict(size=10, color="red"),
                        name="Top 5 días")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          showlegend=True)
    else:  # Ambas con ejes separados
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=ts[x], y=ts["transacciones"], name="Transacciones",
                                 line=dict(color="#1f77b4")), secondary_y=False)
        fig.add_trace(go.Scatter(x=ts[x], y=ts["unidades"], name="Unidades",
                                 line=dict(color="#ff7f0e")), secondary_y=True)
        fig.update_yaxes(title_text="Transacciones", secondary_y=False)
        fig.update_yaxes(title_text="Unidades", secondary_y=True)
        fig.update_xaxes(title_text="Período")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.1))

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Interpretación"):
        st.write(
            "Permite identificar **tendencias** (crecimiento o caída sostenida) y "
            "**estacionalidad** (picos recurrentes por día de la semana o por semana del mes). "
            "Los días con mayor volumen suelen coincidir con fines de semana."
        )

    st.divider()

    # --- Boxplot ---
    st.subheader("Boxplot · Distribución por cliente / categoría")
    box_dim = st.radio("Distribución de:",
                       ["Unidades por cliente", "Transacciones por cliente",
                        "Unidades por categoría"],
                       horizontal=True)

    if box_dim == "Unidades por cliente":
        df_box = q(f"""
            select customer_id, sum(qty) as valor
            from items
            where {date_filter_sql} and {store_filter_sql}
            group by customer_id
        """)
        title = "Unidades compradas por cliente (escala log)"
    elif box_dim == "Transacciones por cliente":
        df_box = q(f"""
            select customer_id, count(distinct transaction_id) as valor
            from items
            where {date_filter_sql} and {store_filter_sql}
            group by customer_id
        """)
        title = "Transacciones por cliente"
    else:
        df_box = q(f"""
            select category_id, sum(qty) as valor
            from items
            where {date_filter_sql} and {store_filter_sql} and category_id is not null
            group by category_id
        """)
        title = "Unidades por categoría"

    use_log = st.checkbox("Eje Y en escala logarítmica", value=True)
    fig = px.box(df_box, y="valor", points="outliers")
    if use_log:
        fig.update_yaxes(type="log")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), title=title)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Interpretación"):
        st.write(
            "Los **outliers** representan comportamientos atípicos: clientes con compras "
            "muy por encima de la mediana, o categorías que dominan el volumen. "
            "El uso de escala logarítmica suele revelar mejor la distribución cuando hay "
            "alta asimetría."
        )
        st.write(df_box["valor"].describe().to_frame("valor").T)

    st.divider()

    # --- Heatmap de correlación ---
    st.subheader("Heatmap · Correlación entre variables de cliente")
    feats = q("""
        select frequency,
               units_total,
               distinct_products,
               distinct_categories,
               avg_basket_size,
               recency_days
        from dim_customer_features
    """)
    corr = feats.corr(method="pearson")
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Interpretación"):
        st.write(
            "Variables consideradas: frecuencia (transacciones), volumen total (unidades), "
            "diversidad de productos, diversidad de categorías, tamaño promedio de canasta y "
            "recencia. Una correlación alta entre **frecuencia y diversidad de productos** "
            "sugiere que los clientes que más vienen también compran cosas más variadas, "
            "lo cual es insumo directo para la segmentación de la próxima entrega."
        )

    st.divider()
    st.caption("Tablas Gold leídas con DuckDB · Pipeline ETL en PySpark · Medallion Bronze/Silver/Gold")


# ============================================================
# Página 3 — Segmentación de Clientes (K-Means)
# ============================================================
elif page == "Segmentación de Clientes":
    st.title("Segmentación de Clientes · K-Means")
    st.caption("Cinco segmentos descubiertos sobre 6 features de comportamiento.")

    if not _has_table("cluster_assignments"):
        st.warning("Los modelos aún no se han ejecutado. Corre `make models` (o "
                   "`python -m src.pipeline.run --step models`) para generarlos.")
        st.stop()

    # --- Resumen general ---
    search = q("select * from kmeans_search order by k")
    best_row = search[search["k"] == -1].iloc[0] if (search["k"] == -1).any() else None
    search = search[search["k"] != -1]
    best_k = int(search.loc[search["silhouette"].idxmax(), "k"])

    profiles = q("select * from cluster_profiles order by cluster_id")
    n_total = int(profiles["n_customers"].sum())

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Clientes segmentados", f"{n_total:,}")
    kpi_card(c2, "Clusters (k)", f"{best_k}", "Seleccionado por silhouette score")
    kpi_card(c3, "Silhouette", f"{float(best_row['silhouette']):.3f}" if best_row is not None else "—",
             "Score de cohesión/separación de los clusters")
    kpi_card(c4, "Features", "6", "frecuencia, unidades, productos, categorías, basket, recencia")

    st.divider()

    # --- Selección de k ---
    st.subheader("Búsqueda del k óptimo (silhouette por k)")
    fig = px.bar(search, x="k", y="silhouette", text="silhouette",
                 color="silhouette", color_continuous_scale="Viridis")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.add_hline(y=float(search["silhouette"].max()), line_dash="dash", line_color="red",
                  annotation_text=f"máximo (k={best_k})")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False, coloraxis_showscale=False,
                      yaxis_title="Silhouette", xaxis_title="k")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("La búsqueda se hizo sobre una muestra del 10% de clientes; el modelo "
               "final se reentrena sobre todo el dataset con el k ganador.")

    st.divider()

    # --- Distribución de tamaños y perfiles ---
    st.subheader("Tamaño de cada cluster")
    sizes = profiles[["cluster_id", "n_customers"]].copy()
    sizes["pct"] = (sizes["n_customers"] / n_total * 100).round(1)
    sizes["label"] = "Cluster " + sizes["cluster_id"].astype(str)
    col_a, col_b = st.columns([2, 3])
    with col_a:
        fig_pie = px.pie(sizes, values="n_customers", names="label", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        fig_bar = px.bar(sizes.sort_values("n_customers"),
                         x="n_customers", y="label", orientation="h",
                         text="pct",
                         color="n_customers", color_continuous_scale="Teal")
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                              showlegend=False, coloraxis_showscale=False,
                              yaxis_title="", xaxis_title="Clientes")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- Perfil medio de cada cluster (heatmap normalizado) ---
    st.subheader("Perfil medio de cada cluster")
    avg_cols = [c for c in profiles.columns if c.startswith("avg_") and c != "avg_n_customers"]
    profile_means = profiles.set_index("cluster_id")[avg_cols].copy()
    profile_means.columns = [c.removeprefix("avg_") for c in profile_means.columns]
    # Normalizamos por columna (min-max) para que los 6 ejes sean comparables visualmente.
    norm = (profile_means - profile_means.min()) / (profile_means.max() - profile_means.min())
    fig = px.imshow(norm, text_auto=".2f", aspect="auto",
                    color_continuous_scale="YlGnBu",
                    labels=dict(x="Feature (normalizada 0-1)", y="Cluster",
                                color="Intensidad"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Valores normalizados min-max entre clusters para resaltar qué "
               "feature domina cada segmento.")

    with st.expander("Tabla cruda de promedios y medianas"):
        st.dataframe(profiles, use_container_width=True)

    # --- Interpretación de negocio ---
    st.subheader("Lectura de negocio")
    interpretations = []
    for _, row in profiles.iterrows():
        cid = int(row["cluster_id"])
        n = int(row["n_customers"])
        pct = round(n / n_total * 100, 1)
        freq = float(row["avg_frequency"])
        units = float(row["avg_units_total"])
        basket = float(row["avg_avg_basket_size"])
        rec = float(row["avg_recency_days"])

        if freq >= 25:
            tag = "🔵 VIP / power users"
        elif freq >= 8 and rec <= 20:
            tag = "🟢 Regulares activos"
        elif basket >= 18:
            tag = "🟠 Canasta grande (compra esporádica)"
        elif rec >= 60:
            tag = "⚫ Inactivos / dormidos"
        else:
            tag = "🟡 Ocasionales recientes"

        interpretations.append({
            "Cluster": cid,
            "Segmento": tag,
            "Clientes": f"{n:,} ({pct}%)",
            "Frecuencia media (txns)": round(freq, 1),
            "Unidades medias": round(units, 0),
            "Canasta media": round(basket, 1),
            "Recencia media (días)": round(rec, 0),
        })
    st.dataframe(pd.DataFrame(interpretations), use_container_width=True, hide_index=True)

    st.subheader("Buscar el cluster de un cliente")
    cust_id = st.number_input("Customer ID", min_value=1, step=1, value=336296)
    res = q(f"""
        select ca.cluster_id, c.frequency, c.units_total, c.distinct_products,
               c.distinct_categories, c.avg_basket_size, c.recency_days
        from cluster_assignments ca
        join dim_customer_features c using (customer_id)
        where ca.customer_id = {int(cust_id)}
    """)
    if res.empty:
        st.info(f"No se encontró el cliente {int(cust_id)} en la tabla de clusters.")
    else:
        r = res.iloc[0]
        st.success(f"Cliente {int(cust_id)} → **Cluster {int(r['cluster_id'])}**")
        st.dataframe(res, use_container_width=True, hide_index=True)


# ============================================================
# Página 4 — Recomendador de Productos
# ============================================================
elif page == "Recomendador de Productos":
    st.title("Recomendador de Productos")
    st.caption("Filtrado colaborativo (ALS) cliente→producto + reglas de asociación "
               "(FP-Growth) producto→producto.")

    if not _has_table("product_rules") and not _has_table("customer_recommendations"):
        st.warning("Los modelos aún no se han ejecutado. Corre `make models`.")
        st.stop()

    tab_a, tab_b, tab_c = st.tabs([
        "Producto → productos asociados",
        "Cliente → productos sugeridos",
        "Top reglas de asociación",
    ])

    # --- producto → productos asociados (FP-Growth) ---
    with tab_a:
        st.subheader("Reglas producto → producto (FP-Growth)")
        st.caption("Reglas mineadas con min_support=0.05 y min_confidence=0.30 sobre el "
                   "top-200 de productos por volumen.")

        products = q("""
            select distinct antecedent_product_id as product_id
            from product_rules
            order by product_id
        """)["product_id"].tolist()

        if not products:
            st.info("No hay reglas disponibles para mostrar.")
        else:
            pid = st.selectbox("Selecciona un producto antecedente",
                               options=products, index=0,
                               format_func=lambda p: f"Producto {p}")
            recs = q(f"""
                select consequent_product_id as product_id,
                       coalesce(consequent_category, '(sin categoría)') as categoria,
                       confidence, lift
                from product_rules
                where antecedent_product_id = {int(pid)}
                order by lift desc, confidence desc
                limit 20
            """)
            if recs.empty:
                st.info("Este producto no genera reglas con el umbral configurado.")
            else:
                recs["label"] = "Prod " + recs["product_id"].astype(str) + " · " + recs["categoria"]
                fig = px.bar(recs.sort_values("lift"),
                             x="lift", y="label", orientation="h",
                             text="confidence", color="lift",
                             color_continuous_scale="Oranges",
                             hover_data={"confidence": ":.2f", "lift": ":.2f",
                                         "product_id": True, "categoria": True,
                                         "label": False})
                fig.update_traces(texttemplate="conf %{text:.2f}", textposition="outside")
                fig.update_layout(height=460, showlegend=False, coloraxis_showscale=False,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  yaxis_title="", xaxis_title="Lift",
                                  yaxis=dict(type="category"))
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Reglas crudas"):
                    st.dataframe(recs, use_container_width=True, hide_index=True)

    # --- cliente → productos sugeridos (ALS) ---
    with tab_b:
        st.subheader("Recomendaciones personalizadas (ALS implicit)")
        st.caption("rank=16, regParam=0.05, implicitPrefs=True. Top-10 productos por cliente.")

        cust_default = 336296
        cust_id = st.number_input("Customer ID", min_value=1, step=1,
                                  value=cust_default, key="als_cust_id")

        # Lo que ya compró
        ya = q(f"""
            select i.product_id,
                   coalesce(p.category_name, '(sin categoría)') as categoria,
                   sum(i.qty) as unidades,
                   count(distinct i.transaction_id) as veces
            from items i
            left join (select distinct product_id, category_name from dim_product_features) p
                on i.product_id = p.product_id
            where i.customer_id = {int(cust_id)}
            group by 1, 2
            order by unidades desc
            limit 10
        """)
        recs = q(f"""
            select cr.product_id, cr.score, cr.rank,
                   coalesce(p.category_name, '(sin categoría)') as categoria
            from customer_recommendations cr
            left join (select distinct product_id, category_name from dim_product_features) p
                on cr.product_id = p.product_id
            where cr.customer_id = {int(cust_id)}
            order by cr.rank
        """)

        if ya.empty and recs.empty:
            st.info(f"No se encontró historial ni recomendaciones para el cliente {int(cust_id)}.")
        else:
            col_x, col_y = st.columns(2)
            with col_x:
                st.markdown("**Historial top-10**")
                if ya.empty:
                    st.write("_(sin historial en el rango filtrado)_")
                else:
                    st.dataframe(ya, use_container_width=True, hide_index=True)
            with col_y:
                st.markdown("**Top-10 recomendados (ALS)**")
                if recs.empty:
                    st.write("_(este cliente no recibió recomendaciones — cold start)_")
                else:
                    st.dataframe(recs, use_container_width=True, hide_index=True)

            if not recs.empty:
                fig = px.bar(recs.sort_values("score"),
                             x="score", y="product_id", orientation="h",
                             color="score", color_continuous_scale="Blues",
                             text="rank",
                             hover_data={"categoria": True, "rank": True, "score": ":.3f"})
                fig.update_traces(texttemplate="#%{text}", textposition="outside")
                fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  yaxis_title="Producto", xaxis_title="Score ALS",
                                  yaxis=dict(type="category"))
                st.plotly_chart(fig, use_container_width=True)

    # --- top reglas globales ---
    with tab_c:
        st.subheader("Reglas globales con mayor lift")
        top_rules = q("""
            select antecedent_product_id as antecedente,
                   coalesce(antecedent_category, '(sin categoría)') as cat_a,
                   consequent_product_id as consecuente,
                   coalesce(consequent_category, '(sin categoría)') as cat_c,
                   confidence, lift
            from product_rules
            order by lift desc, confidence desc
            limit 30
        """)
        st.dataframe(top_rules, use_container_width=True, hide_index=True)
        st.caption("**Lift > 1** indica que el consecuente aparece junto al antecedente "
                   "con mayor frecuencia de la que cabría esperar por azar.")


# ============================================================
# Página 5 — Generación de nuevos resultados (ingesta incremental)
# ============================================================
else:  # "Generación de nuevos resultados"
    import json
    import subprocess
    import sys

    st.title("Generación de nuevos resultados")
    st.caption("Incorporación de nuevos datos (RF-8). Sube un archivo o lanza el "
               "pipeline manualmente cuando hayas dejado nuevos CSV en `data/landing/`.")

    LANDING = ROOT / "data" / "landing"
    LANDING_TX = LANDING / "Transactions"
    LANDING_PROD = LANDING / "Products"

    st.subheader("Estado del manifest")
    manifest_path = LANDING / "_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    # Escaneo en vivo de los archivos presentes
    files_now = []
    for d in (LANDING_TX, LANDING_PROD):
        if d.exists():
            for p in sorted(d.glob("*.csv")):
                rel = str(p.relative_to(LANDING))
                files_now.append({
                    "archivo": rel,
                    "tamaño (MB)": round(p.stat().st_size / (1024 * 1024), 2),
                    "en manifest": "✓" if rel in manifest else "—",
                })
    if files_now:
        st.dataframe(pd.DataFrame(files_now), use_container_width=True, hide_index=True)
    else:
        st.info("No hay archivos en `data/landing/`. Sube uno o cópialo manualmente.")

    st.divider()

    st.subheader("1) Subir un archivo nuevo")
    col_t, col_p = st.columns(2)
    with col_t:
        uploaded_tx = st.file_uploader(
            "Archivo de transacciones (formato `YYY_Tran.csv` con separador `|`)",
            type=["csv"], key="upload_tx",
            help="Se guarda en data/landing/Transactions/")
        if uploaded_tx is not None:
            target = LANDING_TX / uploaded_tx.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded_tx.getbuffer())
            st.success(f"Guardado en `{target.relative_to(ROOT)}`")
    with col_p:
        uploaded_prod = st.file_uploader(
            "Archivo de catálogo (Categories.csv / ProductCategory.csv)",
            type=["csv"], key="upload_prod",
            help="Se guarda en data/landing/Products/")
        if uploaded_prod is not None:
            target = LANDING_PROD / uploaded_prod.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded_prod.getbuffer())
            st.success(f"Guardado en `{target.relative_to(ROOT)}`")

    st.divider()

    st.subheader("2) Detectar cambios y ejecutar el pipeline")
    col_c, col_r, col_f = st.columns(3)
    skip_models = st.checkbox(
        "Omitir reentrenamiento de modelos (más rápido)", value=False,
        help="Útil para iterar sólo sobre KPIs y visualizaciones; las páginas de "
             "segmentación / recomendador seguirán usando los modelos previos.")

    log_box = st.empty()
    run_log = st.session_state.setdefault("ingest_log", [])

    def _run_subprocess(args: list[str]) -> int:
        env_python = sys.executable
        # Forzamos el python del venv del proyecto
        proc = subprocess.Popen(
            [env_python, "-m", "src.pipeline.ingest", *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            run_log.append(line.rstrip())
            # Mantenemos solo las últimas 200 líneas para no romper la UI
            del run_log[:-200]
            log_box.code("\n".join(run_log[-80:]), language="text")
        proc.wait()
        return proc.returncode

    with col_c:
        if st.button("🔍 Comprobar cambios", use_container_width=True):
            run_log.clear()
            rc = _run_subprocess(["--check"])
            st.toast(f"Comprobación terminada (rc={rc})", icon="✅" if rc == 0 else "❌")

    with col_r:
        if st.button("▶️ Ejecutar si hay cambios", use_container_width=True, type="primary"):
            run_log.clear()
            args = ["--run"] + (["--skip-models"] if skip_models else [])
            with st.spinner("Ejecutando pipeline..."):
                rc = _run_subprocess(args)
            if rc == 0:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.toast("Pipeline OK — caché del dashboard refrescada", icon="✅")
            else:
                st.toast(f"Pipeline falló (rc={rc})", icon="❌")

    with col_f:
        if st.button("⚡ Forzar reproceso completo", use_container_width=True):
            run_log.clear()
            args = ["--force"] + (["--skip-models"] if skip_models else [])
            with st.spinner("Reprocesando..."):
                rc = _run_subprocess(args)
            if rc == 0:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.toast("Reproceso OK", icon="✅")
            else:
                st.toast(f"Falló (rc={rc})", icon="❌")

    st.divider()

    st.subheader("Histórico de corridas")
    runs_log_path = LANDING / "_runs.jsonl"
    if runs_log_path.exists():
        rows = []
        for line in runs_log_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if rows:
            df_runs = pd.DataFrame(rows).sort_values("started_at", ascending=False)
            keep_cols = [c for c in
                         ["started_at", "finished_at", "ran", "forced", "skip_models",
                          "files_count", "new", "changed", "removed", "timings_s"]
                         if c in df_runs.columns]
            st.dataframe(df_runs[keep_cols].head(20),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay corridas registradas.")
    else:
        st.info("Aún no hay corridas registradas. La primera ejecución creará el log.")
