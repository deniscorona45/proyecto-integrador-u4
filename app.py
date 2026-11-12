"""
Hito 4 (Unidad 6): Dashboard Streamlit
Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego en Cultivos de Invernadero

Ejecutar:  py -3.13 -m streamlit run app.py
(requiere que analisis.py esté en la misma carpeta)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from scipy import stats

from analisis import ALPHA, ajustar_ols, cargar_datos, matriz_correlacion, prueba_tratamientos

st.set_page_config(page_title="Rendimiento Agrícola en Invernadero", page_icon="🌿", layout="wide")


@st.cache_data
def obtener_datos():
    return cargar_datos()


df_total = obtener_datos()

# ---------------------------------------------------------------- Encabezado
st.title("🌿 Optimización Inferencial del Rendimiento Agrícola")
st.caption("Eficiencia de riego en cultivos de invernadero (jitomate/aguacate) · "
           "Análisis y Exploración de Datos (INC-2505)")

# ------------------------------------------------- Paso 4.2: filtros dinámicos
st.sidebar.header("Filtros")
tratamientos = st.sidebar.multiselect(
    "Tipo de tratamiento",
    options=["Tratamiento_A", "Tratamiento_B"],
    default=["Tratamiento_A", "Tratamiento_B"],
)
riego_min, riego_max = float(df_total["volumen_riego"].min()), float(df_total["volumen_riego"].max())
rango_riego = st.sidebar.slider(
    "Volumen de riego (m³/ha)", min_value=riego_min, max_value=riego_max,
    value=(riego_min, riego_max),
)

df = df_total[
    df_total["tipo_tratamiento"].isin(tratamientos)
    & df_total["volumen_riego"].between(rango_riego[0], rango_riego[1])
]
st.sidebar.write(f"Parcelas mostradas: **{len(df)}** de {len(df_total)}")

if len(df) < 10:
    st.warning("Hay muy pocas parcelas con estos filtros para ajustar el modelo. Amplía los filtros.")
    st.stop()

# ------------------------------------------------------------ Cálculos
modelo, rmse = ajustar_ols(df)
hay_ambos = df["tipo_tratamiento"].nunique() == 2
resultado = prueba_tratamientos(df) if hay_ambos else None

# ------------------------------------------------------------ Tarjetas KPI
k1, k2, k3 = st.columns(3)
k1.metric("Rendimiento promedio global", f"{df['rendimiento_ton'].mean():.2f} ton/ha")
k2.metric("R² ajustado del modelo", f"{modelo.rsquared_adj * 100:.1f} %")
if resultado:
    k3.metric("P-valor de la prueba inferencial", f"{resultado['p']:.2e}",
              delta="Se rechaza H0" if resultado["rechaza_h0"] else "No se rechaza H0",
              delta_color="normal" if resultado["rechaza_h0"] else "off")
else:
    k3.metric("P-valor de la prueba inferencial", "—")
    k3.caption("Selecciona ambos tratamientos para ejecutar la prueba t.")

st.divider()

# ------------------------------------------------- Paso 4.1: cuadrícula 2x2
fila1_col1, fila1_col2 = st.columns(2)
fila2_col1, fila2_col2 = st.columns(2)

# Gráfica 1: Heatmap de Pearson
with fila1_col1:
    st.subheader("1. Mapa de calor de correlación (Pearson)")
    corr = matriz_correlacion(df).rename(
        index={"volumen_riego": "Riego", "fertilizante_kg": "Fertilizante",
               "temperatura_c": "Temperatura", "rendimiento_ton": "Rendimiento"},
        columns={"volumen_riego": "Riego", "fertilizante_kg": "Fertilizante",
                 "temperatura_c": "Temperatura", "rendimiento_ton": "Rendimiento"})
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    r_riego = corr.loc["Riego", "Rendimiento"]
    r_fert = corr.loc["Fertilizante", "Rendimiento"]
    dominante = "el riego" if abs(r_riego) > abs(r_fert) else "el fertilizante"
    st.caption(f"Factor dominante: **{dominante}** (r riego = {r_riego:.2f}, r fertilizante = {r_fert:.2f}).")

# Gráfica 2: Dispersión con regresión OLS
with fila1_col2:
    st.subheader("2. Dispersión con ajuste OLS")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", ci=95,
                scatter_kws={"alpha": 0.6, "s": 25}, line_kws={"color": "red"}, ax=ax)
    ax.set_xlabel("Volumen de Riego (m³/ha)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Modelo: Rendimiento = {modelo.params['const']:.3f} + "
               f"{modelo.params['volumen_riego']:.4f}·Riego + {modelo.params['fertilizante_kg']:.4f}·Fertilizante "
               f"· RMSE = {rmse:.3f} ton/ha")

# Gráfica 3: Boxplot por tratamiento
with fila2_col1:
    st.subheader("3. Boxplot de tratamientos")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    orden = [t for t in ["Tratamiento_A", "Tratamiento_B"] if t in df["tipo_tratamiento"].unique()]
    sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", order=orden,
                hue="tipo_tratamiento", palette="Set2", legend=False, ax=ax)
    medias = df.groupby("tipo_tratamiento")["rendimiento_ton"].mean()
    for i, t in enumerate(orden):
        ax.scatter(i, medias[t], color="black", marker="D", zorder=5, label="Media" if i == 0 else None)
    ax.set_xlabel("Tipo de Tratamiento (A vs B)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    if resultado:
        st.caption(f"Diferencia B − A = {resultado['dif']:.2f} ton/ha · "
                   f"IC 95% [{resultado['ic'][0]:.2f}, {resultado['ic'][1]:.2f}] · "
                   f"t = {resultado['t']:.2f}, p = {resultado['p']:.2e}")

# Gráfica 4: Histograma de residuos
with fila2_col2:
    st.subheader("4. Histograma de residuos")
    residuos = modelo.resid
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.histplot(residuos, bins=15, stat="density", color="steelblue", alpha=0.6, ax=ax)
    sns.kdeplot(residuos, color="red", ax=ax, label="KDE")
    x = np.linspace(residuos.min(), residuos.max(), 200)
    ax.plot(x, stats.norm.pdf(x, 0, residuos.std()), "k--", label="Normal teórica")
    ax.set_xlabel("Residuo (Y real − Ŷ predicho)")
    ax.set_ylabel("Densidad")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Si el histograma es aproximadamente simétrico y centrado en cero, "
               "se cumple de forma razonable el supuesto de normalidad de los errores.")

st.divider()

# ------------------------------------------------------------ Veredicto
st.subheader("Veredicto de ingeniería")
if resultado:
    if resultado["rechaza_h0"]:
        st.success(f"Se rechaza H0 (p = {resultado['p']:.2e} < {ALPHA}). El Tratamiento_B incrementa "
                   "significativamente el rendimiento: se recomienda autorizar la inversión para escalarlo.")
    else:
        st.info(f"No se rechaza H0 (p = {resultado['p']:.4f} ≥ {ALPHA}). No hay evidencia suficiente "
                "para autorizar la inversión en el Tratamiento_B.")
else:
    st.info("Selecciona ambos tratamientos en el panel lateral para ver el veredicto.")

with st.expander("Ver datos de telemetría y resumen estadístico del modelo OLS"):
    st.dataframe(df, width="stretch")
    st.text(modelo.summary().as_text())
