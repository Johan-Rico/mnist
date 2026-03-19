"""
MNIST Digit Classifier — Streamlit App
Dibuja un número y el modelo lo predice en tiempo real.

Instalación:
    pip install streamlit scikit-learn numpy pillow streamlit-drawable-canvas

Ejecutar:
    streamlit run main_app.py
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# ── Streamlit drawable canvas ────────────────────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo oscuro con textura sutil */
.stApp {
    background: #0A0E1A;
    color: #E2E8F0;
}

/* Header principal */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.1;
}

.subtitle {
    color: #64748B;
    font-size: 1rem;
    margin-top: 4px;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Card contenedor */
.card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Resultado de predicción */
.pred-digit {
    font-family: 'Space Mono', monospace;
    font-size: 7rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
    background: linear-gradient(135deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.pred-label {
    text-align: center;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #475569;
    margin-top: -8px;
}

/* Barra de confianza */
.conf-bar-wrap {
    background: #1E293B;
    border-radius: 99px;
    height: 8px;
    width: 100%;
    margin: 4px 0 12px 0;
    overflow: hidden;
}

.conf-bar-fill {
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, #38BDF8, #818CF8);
    transition: width 0.5s ease;
}

/* Etiqueta dígito en la distribución */
.digit-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.digit-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #94A3B8;
    width: 18px;
    text-align: right;
}

/* Badge accuracy */
.badge {
    display: inline-block;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 99px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    color: #38BDF8;
    margin-left: 8px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1220 !important;
    border-right: 1px solid #1E293B;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p {
    color: #94A3B8 !important;
}

/* Botón */
.stButton > button {
    background: linear-gradient(135deg, #38BDF8, #818CF8) !important;
    color: #0A0E1A !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.5rem !important;
    letter-spacing: 1px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Canvas frame */
.canvas-wrapper {
    border: 2px dashed #334155;
    border-radius: 14px;
    padding: 8px;
    display: flex;
    justify-content: center;
}

/* Métricas */
.metric-box {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38BDF8;
}
.metric-lbl {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
}

div[data-testid="stHorizontalBlock"] { gap: 1rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODELO — entrenamiento cacheado
# ════════════════════════════════════════════════════════════════════════════
MODELOS_DISPONIBLES = {
    "SVM (RBF)":            ("svm",  True),
    "Random Forest":        ("rf",   False),
    "Red Neuronal (MLP)":   ("mlp",  True),
    "K-Nearest Neighbors":  ("knn",  False),
    "Regresión Logística":  ("lr",   True),
}

@st.cache_resource(show_spinner="⚙️ Entrenando modelo…")
def entrenar_modelo(nombre_modelo: str):
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    _, necesita_escalado = MODELOS_DISPONIBLES[nombre_modelo]

    if nombre_modelo == "SVM (RBF)":
        modelo = SVC(kernel="rbf", C=10, gamma="scale",
                     probability=True, random_state=42)
    elif nombre_modelo == "Random Forest":
        modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    elif nombre_modelo == "Red Neuronal (MLP)":
        modelo = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                               random_state=42, early_stopping=True)
    elif nombre_modelo == "K-Nearest Neighbors":
        modelo = KNeighborsClassifier(n_neighbors=5)
    else:
        modelo = LogisticRegression(max_iter=1000, random_state=42)

    Xtr = X_train_sc if necesita_escalado else X_train
    Xte = X_test_sc  if necesita_escalado else X_test
    modelo.fit(Xtr, y_train)

    acc = accuracy_score(y_test, modelo.predict(Xte))
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]

    return modelo, scaler, necesita_escalado, acc, n_train, n_test


def imagen_a_vector(img_array: np.ndarray) -> np.ndarray:
    """
    Convierte el array del canvas (RGBA, 280×280) al vector de 64 features
    que espera load_digits (8×8, escala 0–16).
    """
    img = Image.fromarray(img_array.astype(np.uint8))
    # Quedarse solo con canal alfa (lo que se dibujó)
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        img = a
    else:
        img = img.convert("L")

    img = img.resize((8, 8), Image.LANCZOS)
    img = ImageOps.invert(img)           # fondo blanco → negro
    arr = np.array(img, dtype=np.float64)
    arr = arr / arr.max() * 16 if arr.max() > 0 else arr   # rango 0–16
    return arr.flatten().reshape(1, -1)


def predecir(modelo, scaler, necesita_escalado, vec):
    Xp = scaler.transform(vec) if necesita_escalado else vec
    pred    = modelo.predict(Xp)[0]
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(Xp)[0]
    else:
        proba = np.zeros(10)
        proba[pred] = 1.0
    return int(pred), proba


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 Configuración")
    st.markdown("---")

    nombre_modelo = st.selectbox(
        "Algoritmo de clasificación",
        list(MODELOS_DISPONIBLES.keys()),
        index=0,
    )

    grosor = st.slider("Grosor del trazo", 10, 40, 22, step=2)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset")
    st.markdown("""
- **sklearn** `load_digits`
- **1 797** muestras · **10** clases
- Resolución: **8 × 8** píxeles
- Features: **64** por imagen
    """)
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.75rem; color:#475569; line-height:1.6'>
Dibuja un dígito (0–9) en el canvas.<br>
El modelo lo clasifica en tiempo real.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ════════════════════════════════════════════════════════════════════════════
modelo, scaler, necesita_escalado, acc_test, n_train, n_test = entrenar_modelo(nombre_modelo)


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="main-title">MNIST Classifier</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Dibuja un número · el modelo lo reconoce al instante</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Métricas rápidas
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{acc_test*100:.1f}%</div><div class="metric-lbl">Accuracy test</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{n_train}</div><div class="metric-lbl">Muestras train</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{n_test}</div><div class="metric-lbl">Muestras test</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-box"><div class="metric-val">64</div><div class="metric-lbl">Features (8×8)</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Columnas principales
col_canvas, col_result = st.columns([1.1, 1], gap="large")

# ── Canvas ───────────────────────────────────────────────────────────────────
with col_canvas:
    st.markdown("#### ✏️ Dibuja un dígito")
    st.markdown('<p style="color:#475569;font-size:0.85rem;margin-top:-8px">Traza el número en el recuadro negro</p>', unsafe_allow_html=True)

    if not CANVAS_OK:
        st.error(
            "Instala el componente del canvas:\n\n"
            "```\npip install streamlit-drawable-canvas\n```",
            icon="⚠️"
        )
        st.stop()

    canvas_result = st_canvas(
        fill_color    = "rgba(0,0,0,0)",
        stroke_width  = grosor,
        stroke_color  = "#FFFFFF",
        background_color = "#000000",
        width         = 280,
        height        = 280,
        drawing_mode  = "freedraw",
        key           = "canvas",
        display_toolbar = True,
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        predecir_btn = st.button("🔍 Clasificar", use_container_width=True)
    with col_b2:
        limpiar_btn  = st.button("🗑️ Limpiar",    use_container_width=True)

    if limpiar_btn:
        st.rerun()


# ── Resultado ────────────────────────────────────────────────────────────────
with col_result:
    st.markdown("#### 📊 Resultado")
    st.markdown('<p style="color:#475569;font-size:0.85rem;margin-top:-8px">Predicción y distribución de probabilidades</p>', unsafe_allow_html=True)

    canvas_tiene_datos = (
        canvas_result.image_data is not None
        and canvas_result.image_data.sum() > 0
    )

    if predecir_btn and canvas_tiene_datos:
        with st.spinner("Analizando…"):
            vec = imagen_a_vector(canvas_result.image_data)
            digito, proba = predecir(modelo, scaler, necesita_escalado, vec)

        # Dígito predicho grande
        st.markdown(f'<div class="pred-digit">{digito}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-label">dígito predicho · {proba[digito]*100:.1f}% confianza</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Distribución completa
        st.markdown("**Distribución de probabilidades**")
        for i in range(10):
            pct = proba[i] * 100
            color_bar = "#38BDF8" if i == digito else "#1E293B"
            fill_color = "linear-gradient(90deg,#38BDF8,#818CF8)" if i == digito else "#334155"
            st.markdown(f"""
            <div class="digit-row">
                <span class="digit-num">{i}</span>
                <div class="conf-bar-wrap" style="flex:1">
                    <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{fill_color}"></div>
                </div>
                <span style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#64748B;width:42px;text-align:right">
                    {pct:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Clasificación top-3
        top3 = np.argsort(proba)[::-1][:3]
        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        for col, rank, idx in zip([t1,t2,t3], ["🥇","🥈","🥉"], top3):
            with col:
                st.markdown(f"""
                <div style='text-align:center;background:#111827;border:1px solid #1E293B;
                            border-radius:10px;padding:10px 6px'>
                    <div style='font-size:1.1rem'>{rank}</div>
                    <div style='font-family:Space Mono,monospace;font-size:1.5rem;
                                color:#E2E8F0;font-weight:700'>{idx}</div>
                    <div style='font-size:0.72rem;color:#475569'>{proba[idx]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

    elif predecir_btn and not canvas_tiene_datos:
        st.info("✏️ Dibuja un dígito primero en el canvas de la izquierda.")

    else:
        st.markdown("""
        <div style='
            border:2px dashed #1E293B;border-radius:14px;
            padding:3rem 2rem;text-align:center;margin-top:1rem'>
            <div style='font-size:3.5rem;margin-bottom:0.5rem'>🖊️</div>
            <div style='color:#475569;font-size:0.9rem;line-height:1.6'>
                Dibuja un número en el canvas<br>
                y pulsa <strong style='color:#94A3B8'>Clasificar</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN INFORMATIVA
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
with st.expander("🔬 ¿Cómo funciona el pipeline?"):
    st.markdown("""
    1. **Canvas** — dibujas sobre un lienzo 280×280 px (fondo negro, trazo blanco).
    2. **Resize** — la imagen se redimensiona a **8×8 px** con `PIL.Image.LANCZOS`.
    3. **Inversión** — se invierte la escala de grises para que coincida con el formato de `load_digits` (fondo negro, dígito claro).
    4. **Normalización** — los valores se escalan al rango **0–16** (igual que el dataset original).
    5. **Flatten** — la imagen 8×8 se aplana a un vector de **64 features**.
    6. **Escalado** (opcional) — si el modelo lo requiere, se aplica `StandardScaler`.
    7. **Predicción** — el clasificador entrenado devuelve la clase y la distribución de probabilidades.

    ```
    Canvas 280×280 → resize 8×8 → flatten → [64 features] → modelo → dígito
    ```
    """)
