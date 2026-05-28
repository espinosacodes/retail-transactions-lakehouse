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
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    # vistas perezosas sobre los Parquet en disco
    for table in [
        "fact_kpis",
        "fact_sales_daily",
        "dim_customer_features",
        "dim_product_features",
        "fact_category_metrics",
    ]:
        path = GOLD / table
        con.execute(
            f"create view {table} as select * from read_parquet('{path}/**/*.parquet')"
        )
    items_path = SILVER / "transactions_items"
    con.execute(
        f"create view items as select * from read_parquet('{items_path}/**/*.parquet')"
    )
    return con


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    return get_con().execute(sql).df()


# ============================================================
# Sidebar — filtros globales
# ============================================================
st.sidebar.title("🛒 Supermercado")
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
    options=["📊 Resumen Ejecutivo", "🔬 Visualizaciones Analíticas"],
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
if page.startswith("📊"):
    st.title("📊 Resumen Ejecutivo")
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
        st.subheader("🏆 Top 10 productos por unidades vendidas")
        top_p = q(f"""
            select cast(product_id as varchar) as producto,
                   sum(qty) as unidades,
                   count(distinct transaction_id) as transacciones
            from items
            where {date_filter_sql} and {store_filter_sql}
            group by product_id
            order by unidades desc
            limit 10
        """)
        top_p["producto"] = "Prod " + top_p["producto"].astype(str)
        order_p = top_p.sort_values("unidades")["producto"].tolist()
        fig = px.bar(top_p.sort_values("unidades"), x="unidades", y="producto",
                     orientation="h", text="unidades",
                     color="unidades", color_continuous_scale="Blues",
                     category_orders={"producto": order_p})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis_title="Producto", xaxis_title="Unidades", height=420,
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(type="category"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("👥 Top 10 clientes por número de transacciones")
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
    st.subheader("📅 Días pico de compra")
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
    st.subheader("🏷️ Categorías más rentables (proxy: volumen y frecuencia)")
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
else:
    st.title("🔬 Visualizaciones Analíticas")
    st.caption("Exploración de estructura y comportamiento de los datos.")

    if not selected_stores:
        st.warning("Selecciona al menos una tienda en el panel izquierdo.")
        st.stop()

    # --- Serie de tiempo ---
    st.subheader("📈 Serie de tiempo · Ventas por día y semana")
    granularity = st.radio("Granularidad", ["Diaria", "Semanal"], horizontal=True)

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

    fig = px.line(ts, x=x, y=["unidades", "transacciones"],
                  labels={"value": "Cantidad", x: "Período", "variable": "Métrica"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Interpretación"):
        st.write(
            "Permite identificar **tendencias** (crecimiento o caída sostenida) y "
            "**estacionalidad** (picos recurrentes por día de la semana o por semana del mes). "
            "Los días con mayor volumen suelen coincidir con fines de semana."
        )

    st.divider()

    # --- Boxplot ---
    st.subheader("📦 Boxplot · Distribución por cliente / categoría")
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

    with st.expander("ℹ️ Interpretación"):
        st.write(
            "Los **outliers** representan comportamientos atípicos: clientes con compras "
            "muy por encima de la mediana, o categorías que dominan el volumen. "
            "El uso de escala logarítmica suele revelar mejor la distribución cuando hay "
            "alta asimetría."
        )
        st.write(df_box["valor"].describe().to_frame("valor").T)

    st.divider()

    # --- Heatmap de correlación ---
    st.subheader("🔥 Heatmap · Correlación entre variables de cliente")
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

    with st.expander("ℹ️ Interpretación"):
        st.write(
            "Variables consideradas: frecuencia (transacciones), volumen total (unidades), "
            "diversidad de productos, diversidad de categorías, tamaño promedio de canasta y "
            "recencia. Una correlación alta entre **frecuencia y diversidad de productos** "
            "sugiere que los clientes que más vienen también compran cosas más variadas, "
            "lo cual es insumo directo para la segmentación de la próxima entrega."
        )

    st.divider()
    st.caption("Tablas Gold leídas con DuckDB · Pipeline ETL en PySpark · Medallion Bronze/Silver/Gold")
