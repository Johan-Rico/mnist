"""
╔══════════════════════════════════════════════════════════════════════╗
║        MNIST Digit Classifier — EAFIT AI Playground                 ║
║        Reconocimiento de Dígitos Manuscritos con ML Clásico          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# ─────────────────────────── PAGE CONFIG ────────────────────────────
st.set_page_config(
    page_title="MNIST Classifier · EAFIT AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CUSTOM CSS ─────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .stApp { background-color: #0d0f14; color: #e8eaf0; }

  [data-testid="stSidebar"] {
      background: #13161f !important;
      border-right: 1px solid #1e2130;
  }

  .hero-title {
      font-family: 'Space Mono', monospace;
      font-size: 2.6rem;
      font-weight: 700;
      background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 60%, #f472b6 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 0;
  }
  .hero-sub { color: #6b7280; font-size: 1rem; margin-top: 0.3rem; }

  .mcard {
      background: #181c28;
      border: 1px solid #1e2435;
      border-radius: 14px;
      padding: 1.1rem 1.4rem;
      text-align: center;
  }
  .mcard .val {
      font-family: 'Space Mono', monospace;
      font-size: 2rem;
      font-weight: 700;
      background: linear-gradient(90deg, #00d4ff, #7c3aed);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
  }
  .mcard .lbl { color: #6b7280; font-size: 0.82rem; margin-top: 0.25rem; }

  .pred-box {
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
      border: 2px solid #7c3aed;
      border-radius: 18px;
      padding: 2rem;
      text-align: center;
  }
  .pred-digit {
      font-family: 'Space Mono', monospace;
      font-size: 5rem;
      font-weight: 700;
      line-height: 1;
  }
  .pred-label { color: #a78bfa; font-size: 1rem; margin-top: 0.5rem; }

  .pill {
      display: inline-block;
      background: #1a1f2e;
      border: 1px solid #2d3450;
      border-radius: 20px;
      padding: 0.3rem 0.9rem;
      font-size: 0.8rem;
      color: #8892b0;
      margin: 0.15rem;
  }

  .sec-header {
      font-family: 'Space Mono', monospace;
      font-size: 1.1rem;
      color: #00d4ff;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      border-bottom: 1px solid #1e2435;
      padding-bottom: 0.4rem;
      margin-bottom: 1rem;
  }

  .info-card {
      background: #0f1520;
      border-left: 3px solid #7c3aed;
      border-radius: 0 10px 10px 0;
      padding: 0.8rem 1rem;
      font-size: 0.88rem;
      color: #9ca3af;
      margin: 0.6rem 0;
  }

  .stTabs [data-baseweb="tab-list"] { background: #13161f; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #6b7280 !important; }
  .stTabs [aria-selected="true"] { background: #1e2435 !important; color: #00d4ff !important; border-radius: 7px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── DATA ────────────────────────────────────
@st.cache_data
def load_data():
    digits = load_digits()
    return digits.data, digits.target, digits.images, digits

X, y, images, digits_data = load_data()


# ─────────────────────────── CLASSIFIERS CATALOG ─────────────────────
CLASSIFIERS = {
    "SVM — RBF Kernel": {
        "class": SVC,
        "description": "Encuentra el **hiperplano óptimo** en un espacio de alta dimensión usando el kernel RBF. Excelente precisión en imágenes de baja resolución.",
        "tags": ["#Alta precisión", "#Kernel trick", "#Estable"],
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "description": "Ensemble de **múltiples árboles de decisión** que votan en conjunto. Robusto al ruido y paralelizable.",
        "tags": ["#Robusto", "#Feature importance", "#Ensemble"],
    },
    "K-Nearest Neighbors": {
        "class": KNeighborsClassifier,
        "description": "Clasifica buscando los **K vecinos más parecidos** en el espacio de pixeles. Intuitivo y sin entrenamiento explícito.",
        "tags": ["#Lazy learner", "#Intuitivo", "#Sin parámetros"],
    },
    "Regresión Logística": {
        "class": LogisticRegression,
        "description": "Modelo lineal con salida **softmax** para clasificación multiclase. Interpretable y rápido de entrenar.",
        "tags": ["#Lineal", "#Interpretable", "#Probabilístico"],
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "description": "Construye modelos secuencialmente **corrigiendo errores previos**. Alto rendimiento, entrenamiento más lento.",
        "tags": ["#Boosting", "#Alta precisión", "#Secuencial"],
    },
}


# ─────────────────────────── SIDEBAR ─────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ MNIST Classifier")
    st.markdown('<p style="color:#6b7280;font-size:0.8rem;">EAFIT · Machine Learning Lab</p>', unsafe_allow_html=True)
    st.markdown("---")

    clf_name = st.selectbox("🤖 Clasificador", list(CLASSIFIERS.keys()))

    st.markdown("### 🎛️ Hiperparámetros")
    custom_params = {}
    if clf_name == "SVM — RBF Kernel":
        custom_params["C"] = st.slider("C (Regularización)", 0.1, 50.0, 10.0, 0.5)
        custom_params["gamma"] = st.select_slider("Gamma", [0.0001, 0.001, 0.01, 0.1], value=0.001)
        custom_params["probability"] = True
    elif clf_name == "Random Forest":
        custom_params["n_estimators"] = st.slider("Árboles", 10, 300, 200, 10)
        raw_depth = st.slider("Max Depth (0 = sin límite)", 0, 30, 0)
        custom_params["max_depth"] = raw_depth if raw_depth > 0 else None
        custom_params["random_state"] = 42
    elif clf_name == "K-Nearest Neighbors":
        custom_params["n_neighbors"] = st.slider("K vecinos", 1, 15, 3)
        custom_params["weights"] = st.selectbox("Pesos", ["uniform", "distance"])
    elif clf_name == "Regresión Logística":
        custom_params["max_iter"] = st.slider("Max iteraciones", 200, 2000, 1000, 100)
        custom_params["solver"] = "lbfgs"
        custom_params["multi_class"] = "multinomial"
    elif clf_name == "Gradient Boosting":
        custom_params["n_estimators"] = st.slider("Estimadores", 50, 300, 150, 25)
        custom_params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        custom_params["max_depth"] = st.slider("Max Depth", 2, 8, 3)
        custom_params["random_state"] = 42

    st.markdown("### 🔀 División de datos")
    test_size = st.slider("Conjunto de prueba (%)", 10, 35, 20) / 100
    normalize = st.checkbox("Normalizar (StandardScaler)", value=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem;color:#4b5563;">Dataset: sklearn digits<br>1,797 muestras · 8×8 px · 10 clases (0–9)</p>', unsafe_allow_html=True)


# ─────────────────────────── TRAIN ───────────────────────────────────
import json

@st.cache_data
def train_model(clf_name, params_json, test_size, normalize):
    params = json.loads(params_json)          # tipos originales preservados
    clf_class = CLASSIFIERS[clf_name]["class"]
    clf = clf_class(**params)

    steps = [("scaler", StandardScaler()), ("clf", clf)] if normalize else [("clf", clf)]
    pipe = Pipeline(steps)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="macro")
    cm   = confusion_matrix(y_te, y_pred)
    rpt  = classification_report(y_te, y_pred, output_dict=True)
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

    return pipe, X_tr, X_te, y_tr, y_te, y_pred, y_prob, acc, f1, cm, rpt, cv

# Serializar con json para preservar int/float/bool/None correctamente
params_json = json.dumps(custom_params, sort_keys=True)

with st.spinner("⚙️ Entrenando modelo..."):
    pipe, X_tr, X_te, y_tr, y_te, y_pred, y_prob, acc, f1, cm, rpt, cv = train_model(
        clf_name, params_json, test_size, normalize
    )


# ─────────────────────────── HEADER ──────────────────────────────────
st.markdown('<p class="hero-title">✍️ MNIST Digit Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Reconocimiento de dígitos manuscritos · sklearn digits dataset · EAFIT AI Lab</p>', unsafe_allow_html=True)
st.markdown("")

m1, m2, m3, m4 = st.columns(4)
for col, val, lbl in zip(
    [m1, m2, m3, m4],
    [f"{acc:.1%}", f"{f1:.1%}", f"{cv.mean():.1%}", str(len(X_te))],
    ["Accuracy (test)", "F1-Score macro", "CV 5-fold mean", "Muestras de prueba"]
):
    col.markdown(f'<div class="mcard"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
st.markdown("")


# ─────────────────────────── TABS ────────────────────────────────────
tabs = st.tabs([
    "🔭 Explorar Dataset",
    "📊 Métricas de Desempeño",
    "🔮 Predicción en Vivo",
    "🏆 Comparar Modelos",
    "📖 Acerca del Modelo",
])


# ═══════════════ TAB 1 — EXPLORAR DATASET ═══════════════
with tabs[0]:
    st.markdown('<p class="sec-header">Visualización del Dataset</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">El dataset <b>sklearn digits</b> contiene 1,797 imágenes de dígitos 0–9 escritos a mano, cada una de <b>8×8 píxeles</b> (64 características). Es una versión compacta del famoso MNIST original (28×28 px).</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Galería de muestras por dígito")
        fig_g, axes_g = plt.subplots(2, 5, figsize=(10, 4.5))
        fig_g.patch.set_facecolor("#0d0f14")
        for d in range(10):
            idx = np.where(y == d)[0][0]
            ax = axes_g[d // 5][d % 5]
            ax.imshow(images[idx], cmap="plasma", interpolation="nearest")
            ax.set_title(f"Dígito: {d}", color="#00d4ff", fontsize=10, pad=4)
            ax.axis("off")
            ax.set_facecolor("#0d0f14")
        fig_g.tight_layout(pad=1.0)
        st.pyplot(fig_g)
        plt.close()

    with c2:
        st.markdown("#### Distribución de clases")
        counts = pd.Series(y).value_counts().sort_index()
        fig_d = px.bar(x=counts.index, y=counts.values,
                        color=counts.values, color_continuous_scale="Plasma",
                        text=counts.values,
                        labels={"x": "Dígito", "y": "Cantidad"})
        fig_d.update_traces(textposition="outside", marker_line_width=0)
        fig_d.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                             font_color="#c8ccdc", showlegend=False,
                             coloraxis_showscale=False, height=310,
                             xaxis=dict(tickmode="linear", gridcolor="#1e2435"),
                             yaxis=dict(gridcolor="#1e2435"))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("#### Explorador interactivo de muestras")
    ea, eb = st.columns([1, 3])
    with ea:
        sel_d = st.selectbox("Selecciona dígito", list(range(10)))
        d_idxs = np.where(y == sel_d)[0]
        s_i = st.slider("Muestra #", 0, len(d_idxs) - 1, 0)
        r_i = d_idxs[s_i]
        st.markdown(f"**Índice global:** `{r_i}`")
        st.markdown(f"**Rango pixeles:** `{images[r_i].min():.0f}` – `{images[r_i].max():.0f}`")

    with eb:
        fig_s, ax_s = plt.subplots(1, 3, figsize=(10, 3.2))
        fig_s.patch.set_facecolor("#0d0f14")

        ax_s[0].imshow(images[r_i], cmap="plasma", interpolation="nearest")
        ax_s[0].set_title("Original 8×8", color="#00d4ff", fontsize=10)
        ax_s[0].axis("off")

        sns.heatmap(images[r_i], annot=True, fmt=".0f", cmap="plasma",
                    ax=ax_s[1], cbar=False, linewidths=0.5, linecolor="#0d0f14",
                    annot_kws={"size": 8, "color": "white"})
        ax_s[1].set_title("Valores de píxeles", color="#00d4ff", fontsize=10)
        ax_s[1].tick_params(colors="#4b5563", labelsize=7)

        ax_s[2].imshow(images[r_i], cmap="plasma", interpolation="bilinear")
        ax_s[2].set_title("Interpolado (bilinear)", color="#00d4ff", fontsize=10)
        ax_s[2].axis("off")

        for ax in ax_s:
            ax.set_facecolor("#0d0f14")
        fig_s.tight_layout()
        st.pyplot(fig_s)
        plt.close()


# ═══════════════ TAB 2 — MÉTRICAS ═══════════════
with tabs[1]:
    st.markdown('<p class="sec-header">Métricas de Desempeño</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Matriz de Confusión")
        fig_cm = px.imshow(cm, text_auto=True,
                            x=[str(i) for i in range(10)],
                            y=[str(i) for i in range(10)],
                            color_continuous_scale="Plasma",
                            labels=dict(x="Predicción", y="Real"), aspect="auto")
        fig_cm.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                              font_color="#c8ccdc", height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.markdown("#### Precision · Recall · F1 por dígito")
        rpt_df = pd.DataFrame(rpt).T
        cls_df = rpt_df.loc[[str(i) for i in range(10)]].round(3)
        fig_r = go.Figure()
        for metric, color in [("precision","#00d4ff"),("recall","#7c3aed"),("f1-score","#f472b6")]:
            fig_r.add_trace(go.Bar(
                name=metric.capitalize(),
                x=[f"Dígito {i}" for i in range(10)],
                y=cls_df[metric],
                marker_color=color, opacity=0.85,
                text=[f"{v:.2f}" for v in cls_df[metric]],
                textposition="outside", textfont=dict(size=9)
            ))
        fig_r.update_layout(barmode="group", paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                             font_color="#c8ccdc", height=420,
                             yaxis=dict(range=[0,1.15], gridcolor="#1e2435"),
                             xaxis=dict(gridcolor="#1e2435"),
                             legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("#### Validación Cruzada (5-fold)")
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)], y=cv,
        marker_color=["#00d4ff","#3b82f6","#7c3aed","#a855f7","#f472b6"],
        text=[f"{v:.2%}" for v in cv], textposition="outside",
    ))
    fig_cv.add_hline(y=cv.mean(), line_dash="dot", line_color="#f59e0b",
                     annotation_text=f"Media: {cv.mean():.2%}",
                     annotation_font_color="#f59e0b")
    fig_cv.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                          font_color="#c8ccdc", height=300, showlegend=False,
                          yaxis=dict(range=[0.8,1.05], gridcolor="#1e2435"),
                          xaxis=dict(gridcolor="#1e2435"))
    st.plotly_chart(fig_cv, use_container_width=True)

    # Error samples
    errors = np.where(y_pred != y_te)[0]
    st.markdown(f"#### Muestras mal clasificadas — {len(errors)} errores ({len(errors)/len(y_te):.1%})")
    if len(errors) > 0:
        show_n = min(15, len(errors))
        fig_e, axes_e = plt.subplots(3, 5, figsize=(11, 7))
        fig_e.patch.set_facecolor("#0d0f14")
        for i, ax in enumerate(axes_e.flatten()):
            if i < show_n:
                ax.imshow(X_te[errors[i]].reshape(8,8), cmap="plasma", interpolation="nearest")
                ax.set_title(f"Real:{y_te[errors[i]]} Pred:{y_pred[errors[i]]}",
                             color="#f87171", fontsize=9, pad=3)
            ax.axis("off")
            ax.set_facecolor("#0d0f14")
        fig_e.tight_layout()
        st.pyplot(fig_e)
        plt.close()


# ═══════════════ TAB 3 — PREDICCIÓN EN VIVO ═══════════════
with tabs[2]:
    st.markdown('<p class="sec-header">Predicción en Vivo</p>', unsafe_allow_html=True)

    mode = st.radio("Modo de prueba", ["📂 Muestra del dataset", "🎲 Muestra aleatoria"], horizontal=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        if mode == "📂 Muestra del dataset":
            dig_filter = st.selectbox("Filtrar por dígito real", ["Todos"] + list(range(10)))
            available = list(range(len(X_te))) if dig_filter == "Todos" else list(np.where(y_te == dig_filter)[0])
            sample_choice = st.slider("Índice de muestra", 0, len(available) - 1, 0)
            test_idx = available[sample_choice]
        else:
            if st.button("🎲 Nueva muestra aleatoria"):
                st.session_state["rand_idx"] = int(np.random.randint(0, len(X_te)))
            if "rand_idx" not in st.session_state:
                st.session_state["rand_idx"] = 0
            test_idx = st.session_state["rand_idx"]

        sample_flat = X_te[test_idx]
        sample_img  = sample_flat.reshape(8, 8)
        true_label  = y_te[test_idx]

        fig_live, ax_live = plt.subplots(1, 2, figsize=(7, 3.5))
        fig_live.patch.set_facecolor("#0d0f14")
        ax_live[0].imshow(sample_img, cmap="plasma", interpolation="nearest")
        ax_live[0].set_title("Original 8×8", color="#00d4ff", fontsize=11)
        ax_live[0].axis("off")
        ax_live[1].imshow(sample_img, cmap="plasma", interpolation="bilinear")
        ax_live[1].set_title("Interpolado", color="#00d4ff", fontsize=11)
        ax_live[1].axis("off")
        for ax in ax_live:
            ax.set_facecolor("#0d0f14")
        fig_live.tight_layout()
        st.pyplot(fig_live)
        plt.close()

        st.markdown(f'<div class="info-card">🏷️ <b>Etiqueta real:</b> <code>{true_label}</code> &nbsp;|&nbsp; Índice test: <code>{test_idx}</code></div>', unsafe_allow_html=True)

    with col_r:
        pred = pipe.predict(sample_flat.reshape(1, -1))[0]
        correct = pred == true_label
        status_color = "#00d4ff" if correct else "#f87171"
        icon = "✅" if correct else "❌"

        st.markdown(f"""
        <div class="pred-box">
            <div style="color:#6b7280;font-size:0.85rem;margin-bottom:0.5rem;">PREDICCIÓN DEL MODELO</div>
            <div class="pred-digit" style="color:{status_color};">{pred}</div>
            <div class="pred-label">{icon} {"Correcto" if correct else f"Incorrecto — Real: {true_label}"}</div>
            <div style="margin-top:1rem;color:#4b5563;font-size:0.78rem;">Modelo: {clf_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if y_prob is not None:
            probs = pipe.predict_proba(sample_flat.reshape(1, -1))[0]
            colors_p = ["#f87171" if i != pred else "#00d4ff" for i in range(10)]
            fig_p = go.Figure(go.Bar(
                x=list(range(10)), y=probs,
                marker_color=colors_p,
                text=[f"{p:.1%}" for p in probs],
                textposition="outside", textfont=dict(size=9),
            ))
            fig_p.update_layout(
                paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                font_color="#c8ccdc", height=270, showlegend=False,
                margin=dict(t=10),
                yaxis=dict(range=[0, 1.15], gridcolor="#1e2435", title="Probabilidad"),
                xaxis=dict(title="Dígito", tickmode="linear", gridcolor="#1e2435"),
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Este modelo no entrega probabilidades. Usa SVM con probability=True o KNN/RF.")


# ═══════════════ TAB 4 — COMPARAR MODELOS ═══════════════
with tabs[3]:
    st.markdown('<p class="sec-header">Comparación de Clasificadores</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">Evalúa todos los clasificadores con <b>parámetros por defecto</b> y validación cruzada 5-fold. Puede tardar ~30 segundos.</div>', unsafe_allow_html=True)

    if st.button("🚀 Ejecutar comparación completa", type="primary"):
        bench = {
            "SVM — RBF": SVC(C=10, gamma=0.001, probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "KNN (k=3)": KNeighborsClassifier(n_neighbors=3, weights="distance"),
            "Reg. Logística": LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Naive Bayes": GaussianNB(),
        }
        results = []
        prog = st.progress(0)
        status = st.empty()
        for i, (name, clf_b) in enumerate(bench.items()):
            status.text(f"⚙️ Entrenando {name}...")
            pipe_b = Pipeline([("scaler", StandardScaler()), ("clf", clf_b)])
            scores_b = cross_val_score(pipe_b, X, y, cv=5, scoring="accuracy")
            pipe_b.fit(X_tr, y_tr)
            f1_b = f1_score(y_te, pipe_b.predict(X_te), average="macro")
            results.append({"Modelo": name, "CV Accuracy": scores_b.mean(),
                             "CV Std": scores_b.std(), "F1 Macro": f1_b})
            prog.progress((i + 1) / len(bench))

        status.empty(); prog.empty()

        res_df = pd.DataFrame(results).sort_values("CV Accuracy", ascending=False).reset_index(drop=True)
        medals = ["🥇","🥈","🥉"] + [""] * (len(res_df) - 3)
        res_df.insert(0, "", medals)

        fig_cmp = make_subplots(rows=1, cols=2,
                                 subplot_titles=["CV Accuracy (5-fold)", "F1 Macro (test)"])
        bar_colors = ["#00d4ff","#3b82f6","#7c3aed","#a855f7","#f472b6","#f87171"]
        for ci, metric in enumerate(["CV Accuracy", "F1 Macro"], start=1):
            fig_cmp.add_trace(go.Bar(
                x=res_df["Modelo"], y=res_df[metric],
                error_y=dict(array=res_df["CV Std"]) if metric == "CV Accuracy" else None,
                marker_color=bar_colors,
                text=[f"{v:.2%}" for v in res_df[metric]],
                textposition="outside", showlegend=False,
            ), row=1, col=ci)
        fig_cmp.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                               font_color="#c8ccdc", height=420)
        for ci in [1, 2]:
            fig_cmp.update_yaxes(range=[0.8, 1.05], gridcolor="#1e2435", row=1, col=ci)
            fig_cmp.update_xaxes(gridcolor="#1e2435", row=1, col=ci)
        st.plotly_chart(fig_cmp, use_container_width=True)

        display = res_df.copy()
        display["CV Accuracy"] = display["CV Accuracy"].map("{:.2%}".format)
        display["CV Std"]      = display["CV Std"].map("±{:.2%}".format)
        display["F1 Macro"]    = display["F1 Macro"].map("{:.2%}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.markdown(f"**Modelo actual:** `{clf_name}` → Accuracy CV: `{cv.mean():.2%}` ± `{cv.std():.2%}`")
        st.info("Haz clic en el botón para comparar todos los modelos disponibles.")


# ═══════════════ TAB 5 — ACERCA DEL MODELO ═══════════════
with tabs[4]:
    st.markdown('<p class="sec-header">Acerca del Modelo Seleccionado</p>', unsafe_allow_html=True)

    info = CLASSIFIERS[clf_name]
    st.markdown(f"### {clf_name}")
    st.markdown(f'<div class="info-card">{info["description"]}</div>', unsafe_allow_html=True)

    tags_html = " ".join([f'<span class="pill">{t}</span>' for t in info["tags"]])
    st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("#### Hiperparámetros activos")
    if custom_params:
        hdf = pd.DataFrame([(k, str(v)) for k, v in custom_params.items()], columns=["Parámetro", "Valor"])
        st.dataframe(hdf, use_container_width=True, hide_index=True)

    st.markdown("#### Pipeline de entrenamiento")
    if normalize:
        st.markdown("**1. StandardScaler** — Normaliza cada pixel: $x' = (x - \\mu) / \\sigma$")
        st.markdown(f"**2. {clf_name}** — Entrena sobre los 64 pixeles normalizados")
    else:
        st.markdown(f"**1. {clf_name}** — Entrena sobre los 64 pixeles crudos")
    st.markdown("**Último paso. Predicción** — Asigna dígito 0–9 a cada imagen nueva")

    st.markdown("#### Dataset en detalle")
    ic = st.columns(4)
    for col, val, lbl in zip(ic, ["1,797","64 (8×8)","10 (0–9)","0 – 16"],
                                  ["Total muestras","Características","Clases","Rango pixeles"]):
        col.metric(lbl, val)

    st.markdown('<div class="info-card">💡 El dataset <b>sklearn digits</b> es una versión compacta del MNIST original (28×28 px, 70,000 imágenes). Ideal para prototipar clasificadores de visión porque el entrenamiento toma solo segundos en CPU.</div>', unsafe_allow_html=True)
