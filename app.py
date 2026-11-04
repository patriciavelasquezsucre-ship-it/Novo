import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import urllib3

# Desactivar advertencias de certificados SSL autofirmados (común en servidores públicos locales)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Negocios en Venezuela",
    page_icon="📊",
    layout="wide"
)

st.title("🇻🇪 Simulador Financiero en Tiempo Real (Multidivisa / FCP)")
st.markdown("Herramienta diseñada para evaluar la viabilidad de **proyectos nuevos** y **negocios en marcha** bajo condiciones de volatilidad cambiaria, inflación y esquemas multidivisa.")

# ---------------------------------------------------------
# FUNCIÓN DE WEB SCRAPING: TASA BCV EN TIEMPO REAL
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # Guarda en caché la tasa durante 1 hora para evitar saturar el sitio del BCV
def obtener_tasa_bcv():
    url = "https://www.bcv.org.ve/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=6)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # El portal del BCV almacena el valor del dólar dentro de un div con id 'dolar' en una etiqueta strong
            div_dolar = soup.find("div", {"id": "dolar"})
            if div_dolar:
                tasa_str = div_dolar.find("strong").text.strip()
                # Limpiar el formato de texto (ej. cambiar coma decimal por punto)
                tasa_limpia = float(tasa_str.replace(".", "").replace(",", "."))
                return tasa_limpia
    except Exception as e:
        # Si ocurre un error de conexión, retorna None para activar el respaldo
        pass
    return None

# Intentar obtener la tasa real del BCV
tasa_bcv_en_vivo = obtener_tasa_bcv()
tasa_fallback = tasa_bcv_en_vivo if tasa_bcv_en_vivo else 45.0

# ---------------------------------------------------------
# BARRA LATERAL: ENTRADAS DEL EVALUADOR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Negocio")

tipo_modelo = st.sidebar.selectbox(
    "Estado del Negocio",
    ["Negocio en Marcha", "Proyecto Nuevo"]
)

st.sidebar.subheader("1. Indicadores Macroeconómicos")
if tasa_bcv_en_vivo:
    st.sidebar.success(f"🟢 Tasa BCV Oficial en Vivo: **{tasa_bcv_en_vivo} VES/USD**")
else:
    st.sidebar.warning("⚠️ No se pudo conectar al BCV. Usando tasa referencial manual.")

tasa_referencia_inicial = st.sidebar.number_input(
    "Tasa de Cambio Base Inicial (VES/USD)", 
    min_value=1.0, 
    value=float(tasa_fallback), 
    step=0.5
)

st.sidebar.subheader("2. Capital e Inventario")
capital_inicial = st.sidebar.number_input("Capital Inicial de Trabajo (USD)", min_value=500, value=12000, step=500)
inventario_inicial = st.sidebar.number_input("Inventario Inicial (Unidades)", min_value=10, value=400, step=50)
costo_unitario_usd = st.sidebar.number_input("Costo de Reposición Unitario (USD)", min_value=0.5, value=10.0, step=0.5)
precio_venta_usd = st.sidebar.number_input("Precio de Venta Unitario (USD)", min_value=1.0, value=18.0, step=0.5)

st.sidebar.subheader("3. Estructura de Pagos (Multidivisa)")
pct_divisas = st.sidebar.slider("% Pagos en Divisas (Zelle / Efectivo)", min_value=0, max_value=100, value=40) / 100.0
pct_bolivares = 1.0 - pct_divisas
st.sidebar.caption(f"Porcentaje automático en Bolívares (Pago Móvil): **{int(pct_bolivares*100)}%**")

dias_rezago = st.sidebar.slider("Días de Rezago en Uso/Conversión de VES", min_value=0, max_value=5, value=2)

st.sidebar.subheader("4. Parámetros Operativos y Macro")
gasto_fijo_diario = st.sidebar.number_input("Gastos Operativos Diarios Indexados (USD)", min_value=5.0, value=50.0, step=5.0)
devaluacion_diaria = st.sidebar.slider("Devaluación Promedio Diaria Estimada (%)", min_value=0.0, max_value=2.0, value=0.3, step=0.1) / 100.0
dias_simulacion = st.sidebar.select_slider("Horizonte de Simulación (Días)", options=[30, 60, 90], value=30)
iteraciones = st.sidebar.slider("Iteraciones de Monte Carlo", min_value=100, max_value=1000, value=300, step=100)

# ---------------------------------------------------------
# MOTOR DE CÁLCULO (MONTE CARLO + FCP)
# ---------------------------------------------------------
@st.cache_data
def ejecutar_motor_simulacion(
    dias, capital, inv_inicial, costo_unit, precio_unit, 
    mix_divisas, mix_bolivares, rezago, gasto_fijo, dev_diaria, n_iter, es_proyecto, tasa_base
):
    np.random.seed(42)
    matriz_caja = []
    
    factor_demanda = 0.7 if es_proyecto else 1.0
    
    for _ in range(n_iter):
        caja_usd = capital
        inventario = inv_inicial
        tasa_cambio = tasa_base # Usa la tasa obtenida en vivo o referencial
        serie_caja = [caja_usd]
        
        for dia in range(1, dias + 1):
            dev_dia = np.random.normal(loc=dev_diaria, scale=dev_diaria * 0.7)
            tasa_cambio *= (1 + max(0, dev_dia))
            
            demanda_unidades = min(np.random.poisson(lam=14 * factor_demanda), inventario)
            venta_bruta_usd = demanda_unidades * precio_unit
            
            ingreso_divisas = (venta_bruta_usd * mix_divisas) * (1 - 0.01)
            
            tasa_futura = tasa_cambio * ((1 + max(0, dev_dia)) ** rezago)
            factor_erosion = tasa_cambio / tasa_futura if tasa_futura > 0 else 1.0
            ingreso_bolivares = (venta_bruta_usd * mix_bolivares) * (1 - 0.015) * factor_erosion
            
            ingreso_neto_real = ingreso_divisas + ingreso_bolivares
            gasto_operativo_dia = gasto_fijo * (1 + max(0, dev_dia))
            
            inventario -= demanda_unidades
            caja_usd += ingreso_neto_real - gasto_operativo_dia
            
            if inventario < 120:
                lote_reposicion = 300
                costo_compra = lote_reposicion * costo_unit
                if caja_usd >= costo_compra:
                    caja_usd -= costo_compra
                    inventario += lote_reposicion
                    
            serie_caja.append(caja_usd)
            
        matriz_caja.append(serie_caja)
        
    return np.percentile(matriz_caja, [10, 50, 90], axis=0)

es_proyecto = True if tipo_modelo == "Proyecto Nuevo" else False
percentiles = ejecutar_motor_simulacion(
    dias=dias_simulacion,
    capital=capital_inicial,
    inv_inicial=inventario_inicial,
    costo_unit=costo_unitario_usd,
    precio_unit=precio_venta_usd,
    mix_divisas=pct_divisas,
    mix_bolivares=pct_bolivares,
    rezago=dias_rezago,
    gasto_fijo=gasto_fijo_diario,
    dev_diaria=devaluacion_diaria,
    n_iter=iteraciones,
    es_proyecto=es_proyecto,
    tasa_base=tasa_referencia_inicial
)

# ---------------------------------------------------------
# INTERFAZ PRINCIPAL: MÉTRICAS Y GRÁFICAS
# ---------------------------------------------------------
st.subheader("📈 Resultados de Proyección de Caja (USD)")

col1, col2, col3 = st.columns(3)
p10_final = round(percentiles[0][-1], 2)
p50_final = round(percentiles[1][-1], 2)
p90_final = round(percentiles[2][-1], 2)

col1.metric("Escenario Pesimista (P10)", f"${p10_final:,.2f}", delta=f"{p10_final - capital_inicial:,.2f} USD")
col2.metric("Escenario Esperado (P50)", f"${p50_final:,.2f}", delta=f"{p50_final - capital_inicial:,.2f} USD")
col3.metric("Escenario Optimista (P90)", f"${p90_final:,.2f}", delta=f"{p90_final - capital_inicial:,.2f} USD")

st.markdown("---")

dias_eje = list(range(0, dias_simulacion + 1))
fig = go.Figure()

fig.add_trace(go.Scatter(x=dias_eje, y=percentiles[2], mode='lines', name='Optimista (P90)', line=dict(color='#2ecc71', width=2)))
fig.add_trace(go.Scatter(x=dias_eje, y=percentiles[1], mode='lines', name='Esperado (P50)', line=dict(color='#3498db', width=3)))
fig.add_trace(go.Scatter(x=dias_eje, y=percentiles[0], mode='lines', name='Pesimista (P10)', line=dict(color='#e74c3c', width=2)))
fig.add_hline(y=capital_inicial, line_dash="dash", line_color="gray", annotation_text="Capital Inicial")

fig.update_layout(
    xaxis_title="Días de Operación",
    yaxis_title="Caja Disponible (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# SECCIÓN DE DIAGNÓSTICO FINANCIERO Y EXPORTACIÓN
# ---------------------------------------------------------
st.subheader("💡 Diagnóstico y Recomendaciones del Evaluador")

fcp_neto = round((pct_divisas * 0.99) + (pct_bolivares * 0.985 * (1 / ((1 + devaluacion_diaria) ** dias_rezago))), 4)

if p10_final < capital_inicial:
    st.warning(f"⚠️ **Riesgo de Erosión de Caja:** Bajo el escenario pesimista, el capital final disminuye. El Factor de Conversión Ponderado (FCP) retiene **${fcp_neto}** por cada dólar debido a comisiones y los {dias_rezago} días de rezago en bolívares. Se recomienda reducir los plazos de liquidación o incrementar el porcentaje de cobro en divisas.")
else:
    st.success(f"✅ **Modelo Robusto:** El negocio muestra capacidad de absorción ante la devaluación simulada. El Factor de Conversión Ponderado estimado es de **{fcp_neto}**.")

# Tabla y Descarga CSV
st.markdown("---")
df_resultados = pd.DataFrame({
    "Día": dias_eje,
    "Pesimista (P10) USD": percentiles[0],
    "Esperado (P50) USD": percentiles[1],
    "Optimista (P90) USD": percentiles[2]
})

with st.expander("Ver tabla completa de proyección día a día"):
    st.dataframe(df_resultados.set_index("Día"), use_container_width=True)

csv_data = df_resultados.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Proyección Completa (CSV / Excel)",
    data=csv_data,
    file_name="simulacion_financiera_venezuela.csv",
    mime="text/csv",
)