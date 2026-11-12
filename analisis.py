"""
Proyecto Integrador U4 - Optimización Inferencial del Rendimiento Agrícola
Hitos 1, 2 y 3: telemetría (DuckDB), regresión/correlación (4.2) e inferencia (4.3).

Se puede ejecutar solo:  py -3.13 analisis.py
y también lo importa el dashboard (app.py).
"""
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ALPHA = 0.05


# =============================================================================
# HITO 1: CARGA Y PREPARACIÓN DE TELEMETRÍA (DuckDB)
# =============================================================================
def cargar_datos() -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.execute("SELECT setseed(0.42);")  # Paso 1.1: semilla global

    # Paso 1.2 (pasos 2 a 5 del algoritmo): variables base de la parcela
    con.execute("""
        CREATE TABLE base AS
        SELECT
            parcela_id,
            ROUND(100.0 + random() * 200.0, 2)  AS volumen_riego,   -- 100 a 300 m3/ha
            ROUND(50.0  + random() * 100.0, 2)  AS fertilizante_kg,  -- 50 a 150 kg/ha
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A'
                 ELSE 'Tratamiento_B' END       AS tipo_tratamiento,
            (random() * 1.6) - 0.8              AS ruido,            -- -0.8 a +0.8 ton/ha
            ROUND(18.0 + random() * 14.0, 2)    AS temperatura_c     -- 18 a 32 C (para el heatmap)
        FROM generate_series(1, 100) AS t(parcela_id);
    """)

    # Paso 6 del algoritmo: variable de respuesta (ecuación físico-matemática)
    con.execute("""
        CREATE TABLE telemetria_invernadero AS
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            temperatura_c,
            ROUND(
                5.0
                + (0.035 * volumen_riego)
                + (0.020 * fertilizante_kg)
                + CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END
                + ruido
            , 2) AS rendimiento_ton
        FROM base;
    """)

    # Paso 1.3: extracción SQL y paso a pandas
    df = con.execute("""
        SELECT parcela_id, volumen_riego, fertilizante_kg, temperatura_c,
               tipo_tratamiento, rendimiento_ton
        FROM telemetria_invernadero
        ORDER BY parcela_id
    """).df()
    con.close()
    return df


# =============================================================================
# HITO 2: CORRELACIÓN DE PEARSON Y REGRESIÓN OLS (Subtema 4.2)
# =============================================================================
def matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["volumen_riego", "fertilizante_kg", "temperatura_c", "rendimiento_ton"]
    return df[cols].corr(method="pearson")


def ajustar_ols(df: pd.DataFrame):
    X = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])  # agrega beta_0
    y = df["rendimiento_ton"]
    modelo = sm.OLS(y, X).fit()
    rmse = float(np.sqrt(np.mean(modelo.resid ** 2)))
    return modelo, rmse


# =============================================================================
# HITO 3: PRUEBA t DE DOS MUESTRAS INDEPENDIENTES (Subtema 4.3)
# H0: mu_B - mu_A <= 0      H1: mu_B - mu_A > 0
# =============================================================================
def prueba_tratamientos(df: pd.DataFrame, alpha: float = ALPHA) -> dict:
    a = df.loc[df["tipo_tratamiento"] == "Tratamiento_A", "rendimiento_ton"]
    b = df.loc[df["tipo_tratamiento"] == "Tratamiento_B", "rendimiento_ton"]

    t_stat, p_value = stats.ttest_ind(b, a, equal_var=True, alternative="greater")

    # IC 95% para (mu_B - mu_A), varianza combinada (t de Student)
    n_a, n_b = len(a), len(b)
    gl = n_a + n_b - 2
    sp2 = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / gl
    ee = np.sqrt(sp2 * (1 / n_a + 1 / n_b))
    dif = b.mean() - a.mean()
    t_crit = stats.t.ppf(1 - alpha / 2, gl)
    ic = (dif - t_crit * ee, dif + t_crit * ee)

    return {
        "n_a": n_a, "n_b": n_b,
        "media_a": a.mean(), "media_b": b.mean(),
        "dif": dif, "t": t_stat, "p": p_value, "gl": gl, "ic": ic,
        "rechaza_h0": p_value < alpha,
    }


# =============================================================================
# EJECUCIÓN POR CONSOLA (para capturas del reporte)
# =============================================================================
if __name__ == "__main__":
    df = cargar_datos()

    print("=== HITO 1: PRIMEROS 10 REGISTROS (DuckDB) ===")
    print(df.head(10).to_string(index=False))
    print("\nRango de rendimiento: "
          f"{df['rendimiento_ton'].min():.2f} a {df['rendimiento_ton'].max():.2f} ton/ha")
    print("\n" + "=" * 60 + "\n")

    print("=== HITO 2.1: MATRIZ DE CORRELACIÓN DE PEARSON (r) ===")
    print(matriz_correlacion(df).round(4))
    print("\n=== HITO 2.2 y 2.3: MODELO OLS ===")
    modelo, rmse = ajustar_ols(df)
    print(modelo.summary())
    print(f"\nR2 ajustado: {modelo.rsquared_adj:.4f}  ({modelo.rsquared_adj*100:.2f}% de la variabilidad explicada)")
    print(f"RMSE: {rmse:.4f} ton/ha")
    print("\n" + "=" * 60 + "\n")

    print("=== HITO 3: PRUEBA t (Tratamiento_B vs Tratamiento_A) ===")
    r = prueba_tratamientos(df)
    print("H0: mu_B - mu_A <= 0   |   H1: mu_B - mu_A > 0   |   alpha = 0.05")
    print(f"n_A = {r['n_a']}  media_A = {r['media_a']:.4f} ton/ha")
    print(f"n_B = {r['n_b']}  media_B = {r['media_b']:.4f} ton/ha")
    print(f"Diferencia de medias (B - A): {r['dif']:.4f} ton/ha")
    print(f"Estadístico t: {r['t']:.4f}  (gl = {r['gl']})")
    print(f"Valor p: {r['p']:.6e}")
    print(f"IC 95% de la diferencia: [{r['ic'][0]:.4f}, {r['ic'][1]:.4f}] ton/ha")
    print("\n=== VEREDICTO DE INGENIERÍA ===")
    if r["rechaza_h0"]:
        print(f"Se RECHAZA H0 (p = {r['p']:.6e} < {ALPHA}). El Tratamiento_B incrementa "
              "significativamente el rendimiento. SE RECOMIENDA ESCALAR EL TRATAMIENTO_B.")
    else:
        print(f"NO se rechaza H0 (p = {r['p']:.4f} >= {ALPHA}). No hay evidencia suficiente "
              "para autorizar la inversión en el Tratamiento_B.")
