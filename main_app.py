"""
mnist_app.py — Aplicación Streamlit para clasificación de dígitos MNIST con CNN
Ejecutar: streamlit run mnist_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageOps, ImageFilter
import io
import time
import os

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Fondo oscuro con textura sutil */
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(236,72,153,0.06) 0%, transparent 40%);
}

/* Header principal */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    background: linear-gradient(135deg, #818cf8 0%, #e879f9 50%, #fb923c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: #6b7280;
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 2rem;
}

/* Cards métricas */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(129,140,248,0.4); }
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

/* Badge de predicción */
.pred-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    color: #e879f9;
    text-align: center;
    line-height: 1;
    text-shadow: 0 0 40px rgba(232,121,249,0.4);
}
.pred-conf {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #6b7280;
    margin-top: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10,10,20,0.95);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Botones */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #6b7280;
    border-radius: 7px;
}
.stTabs [aria-selected="true"] {
    background: rgba(129,140,248,0.2) !important;
    color: #818cf8 !important;
}

/* Separador con estilo */
hr { border-color: rgba(255,255,255,0.06); margin: 1.5rem 0; }

/* Canvas de dibujo */
.canvas-container {
    border: 2px dashed rgba(129,140,248,0.3);
    border-radius: 12px;
    padding: 8px;
    background: rgba(255,255,255,0.02);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Importaciones pesadas con caché ──────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cargar_librerias():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    return tf, keras, layers, models, EarlyStopping, ReduceLROnPlateau, \
           classification_report, confusion_matrix, accuracy_score


@st.cache_resource(show_spinner=False)
def cargar_datos():
    tf, keras, *_ = cargar_librerias()
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    return X_train, y_train, X_test, y_test


@st.cache_resource(show_spinner=False)
def construir_cnn():
    tf, keras, layers, models, *_ = cargar_librerias()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', input_shape=(28,28,1), name='conv1'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ], name='MNIST_CNN')

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


@st.cache_resource(show_spinner=False)
def entrenar_modelo(_model_ref, epochs=15):
    tf, keras, *rest = cargar_librerias()
    EarlyStopping_cls  = rest[3]
    ReduceLROnPlateau_cls = rest[4]

    X_train, y_train, X_test, y_test = cargar_datos()

    X_tr = X_train.astype('float32') / 255.0
    X_te = X_test.astype('float32') / 255.0
    X_tr = X_tr[..., np.newaxis]
    X_te = X_te[..., np.newaxis]
    y_tr_ohe = keras.utils.to_categorical(y_train, 10)
    y_te_ohe = keras.utils.to_categorical(y_test, 10)

    model = construir_cnn()
    history = model.fit(
        X_tr, y_tr_ohe,
        validation_split=0.1,
        epochs=epochs,
        batch_size=256,
        callbacks=[
            EarlyStopping_cls(monitor='val_accuracy', patience=4, restore_best_weights=True),
            ReduceLROnPlateau_cls(monitor='val_loss', factor=0.5, patience=2),
        ],
        verbose=0,
    )
    _, acc = model.evaluate(X_te, y_te_ohe, verbose=0)
    return model, history, acc, X_te, y_te_ohe, y_test


# ─── Funciones auxiliares ─────────────────────────────────────────────────────

def predecir(imagen_28x28, model):
    img = imagen_28x28.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    probs = model.predict(img, verbose=0)[0]
    return int(np.argmax(probs)), float(np.max(probs)), probs


def fig_barras_prob(probs, pred):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    colores = ['#e879f9' if i == pred else '#2d2d3f' for i in range(10)]
    bars = ax.bar(range(10), probs * 100, color=colores, width=0.7, edgecolor='none')
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(0, 110)
    ax.set_xticks(range(10))
    ax.tick_params(colors='#6b7280', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    for bar, p in zip(bars, probs):
        if p > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{p*100:.0f}%', ha='center', va='bottom',
                    color='#9ca3af', fontsize=7.5, fontfamily='monospace')
    ax.set_xlabel('Dígito', color='#6b7280', fontsize=9)
    plt.tight_layout(pad=0.3)
    return fig


def fig_curvas(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor('#0a0a0f')
    for ax in axes:
        ax.set_facecolor('#111118')
        for spine in ax.spines.values():
            spine.set_color('#2d2d3f')
        ax.tick_params(colors='#6b7280', labelsize=9)

    ep = range(1, len(history.history['accuracy']) + 1)
    axes[0].plot(ep, np.array(history.history['accuracy'])*100,   color='#818cf8', lw=2, label='Train')
    axes[0].plot(ep, np.array(history.history['val_accuracy'])*100, color='#e879f9', lw=2, label='Val', linestyle='--')
    axes[0].set_title('Accuracy (%)', color='#d1d5db', fontsize=10)
    axes[0].set_xlabel('Época', color='#6b7280', fontsize=9)
    axes[0].legend(fontsize=8, labelcolor='#9ca3af', framealpha=0)
    axes[0].grid(True, color='#1f1f2e', linewidth=0.8)

    axes[1].plot(ep, history.history['loss'],     color='#818cf8', lw=2, label='Train')
    axes[1].plot(ep, history.history['val_loss'], color='#e879f9', lw=2, label='Val', linestyle='--')
    axes[1].set_title('Loss', color='#d1d5db', fontsize=10)
    axes[1].set_xlabel('Época', color='#6b7280', fontsize=9)
    axes[1].legend(fontsize=8, labelcolor='#9ca3af', framealpha=0)
    axes[1].grid(True, color='#1f1f2e', linewidth=0.8)

    plt.tight_layout(pad=1.0)
    return fig


def fig_confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax,
                linewidths=0.3, linecolor='#1a1a2e',
                xticklabels=range(10), yticklabels=range(10),
                annot_kws={'size': 8})
    ax.set_xlabel('Predicción', color='#9ca3af', fontsize=10)
    ax.set_ylabel('Real', color='#9ca3af', fontsize=10)
    ax.tick_params(colors='#6b7280', labelsize=9)
    ax.set_title('Matriz de Confusión', color='#d1d5db', fontsize=11, pad=10)
    plt.tight_layout()
    return fig


def normalizar_imagen_usuario(pil_img):
    """Convierte imagen PIL del usuario a array 28x28."""
    img = pil_img.convert('L')           # Escala de grises
    img = ImageOps.invert(img)           # Invertir (fondo negro, dígito blanco)
    img = img.filter(ImageFilter.GaussianBlur(1))
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img)
    # Auto-contraste
    if arr.max() > 0:
        arr = (arr / arr.max() * 255).astype(np.uint8)
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">MNIST Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Convolutional Neural Network · TensorFlow / Keras</div>',
            unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    epochs_slider = st.slider("Épocas de entrenamiento", min_value=3, max_value=30, value=15, step=1)
    st.caption("EarlyStopping puede detener antes si converge.")

    st.markdown("---")
    entrenar_btn = st.button("🚀 Entrenar modelo")

    st.markdown("---")
    st.markdown("### 📐 Arquitectura CNN")
    st.markdown("""
<small style="color:#6b7280; font-family:'JetBrains Mono',monospace; line-height:1.8">
Input (28×28×1)<br>
↓ Conv2D 32 + BN<br>
↓ Conv2D 32 + BN<br>
↓ MaxPool + Drop<br>
↓ Conv2D 64 + BN<br>
↓ Conv2D 64 + BN<br>
↓ MaxPool + Drop<br>
↓ Conv2D 128 + BN<br>
↓ GlobalAvgPool<br>
↓ Dense 256 + BN<br>
↓ Dense 10 (softmax)
</small>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<small style="color:#4b5563">
<b style="color:#6b7280">Dataset:</b> MNIST<br>
<b style="color:#6b7280">Train:</b> 60,000 imgs<br>
<b style="color:#6b7280">Test:</b> 10,000 imgs<br>
<b style="color:#6b7280">Clases:</b> 10 dígitos (0–9)<br>
<b style="color:#6b7280">Input:</b> 28×28 px grayscale
</small>
""", unsafe_allow_html=True)


# ─── Estado de sesión ─────────────────────────────────────────────────────────
if 'model'   not in st.session_state: st.session_state.model   = None
if 'history' not in st.session_state: st.session_state.history = None
if 'test_acc' not in st.session_state: st.session_state.test_acc = None
if 'X_te'    not in st.session_state: st.session_state.X_te    = None
if 'y_te_ohe' not in st.session_state: st.session_state.y_te_ohe = None
if 'y_te'    not in st.session_state: st.session_state.y_te    = None


# ─── Entrenamiento ────────────────────────────────────────────────────────────
if entrenar_btn or st.session_state.model is None:
    with st.spinner("⏳ Entrenando CNN... esto puede tardar 1-3 minutos."):
        t0 = time.time()
        model, history, acc, X_te, y_te_ohe, y_te = entrenar_modelo(None, epochs=epochs_slider)
        elapsed = time.time() - t0
        st.session_state.model    = model
        st.session_state.history  = history
        st.session_state.test_acc = acc
        st.session_state.X_te     = X_te
        st.session_state.y_te_ohe = y_te_ohe
        st.session_state.y_te     = y_te
    st.success(f"✅ Modelo entrenado en {elapsed:.0f}s — Accuracy: **{acc*100:.2f}%**")


model    = st.session_state.model
history  = st.session_state.history
test_acc = st.session_state.test_acc
X_te     = st.session_state.X_te
y_te     = st.session_state.y_te

if model is None:
    st.info("Presiona **Entrenar modelo** en el sidebar para comenzar.")
    st.stop()


# ─── Métricas rápidas ─────────────────────────────────────────────────────────
epocas_reales = len(history.history['accuracy'])
mejor_val_acc = max(history.history['val_accuracy'])
total_params  = model.count_params()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{test_acc*100:.2f}%</div>
        <div class="metric-label">Test Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{mejor_val_acc*100:.2f}%</div>
        <div class="metric-label">Best Val Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{epocas_reales}</div>
        <div class="metric-label">Épocas</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_params/1e6:.2f}M</div>
        <div class="metric-label">Parámetros</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Clasificar",
    "📈 Entrenamiento",
    "📊 Evaluación",
    "🔍 Explorar Dataset",
])


# ══ TAB 1: CLASIFICAR ════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Clasificar un dígito")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        modo = st.radio("Fuente de la imagen",
                        ["📂 Imagen del dataset de prueba", "🖼️ Subir imagen propia"],
                        horizontal=True)

        if "📂" in modo:
            idx = st.slider("Índice en el conjunto de prueba", 0, len(X_te)-1, 0, 1)

            # Recuperar imagen en escala [0,255] para visualizar
            X_train_raw, y_train_raw, X_test_raw, _ = cargar_datos()
            img_28 = X_test_raw[idx]

            fig_img, ax = plt.subplots(figsize=(3, 3))
            fig_img.patch.set_facecolor('#0a0a0f')
            ax.imshow(img_28, cmap='gray', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f"Etiqueta real: {y_te[idx]}",
                         color='#9ca3af', fontsize=11, pad=8)
            st.pyplot(fig_img, use_container_width=False)

            pred_digit, pred_conf, probs = predecir(img_28, model)

        else:
            uploaded = st.file_uploader(
                "Sube una imagen de un dígito (PNG, JPG)",
                type=['png', 'jpg', 'jpeg'],
                help="Idealmente: fondo blanco, dígito negro, cuadrada."
            )
            if uploaded:
                pil_img = Image.open(uploaded)
                img_28  = normalizar_imagen_usuario(pil_img)

                col_o, col_p = st.columns(2)
                with col_o:
                    st.image(pil_img, caption="Original", use_container_width=True)
                with col_p:
                    fig_p, ax = plt.subplots(figsize=(3,3))
                    fig_p.patch.set_facecolor('#0a0a0f')
                    ax.imshow(img_28, cmap='gray')
                    ax.axis('off')
                    ax.set_title("Preprocesada (28×28)", color='#9ca3af', fontsize=9)
                    st.pyplot(fig_p, use_container_width=False)

                pred_digit, pred_conf, probs = predecir(img_28, model)
            else:
                st.info("⬆️ Sube una imagen para clasificar.")
                st.stop()

    with col_right:
        st.markdown("#### Resultado")
        st.markdown(f'<div class="pred-badge">{pred_digit}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-conf">Confianza: {pred_conf*100:.1f}%</div>',
                    unsafe_allow_html=True)

        if pred_conf > 0.95:
            st.success("🟢 Alta confianza")
        elif pred_conf > 0.75:
            st.warning("🟡 Confianza media")
        else:
            st.error("🔴 Baja confianza")

        st.markdown("#### Probabilidades por clase")
        fig_prob = fig_barras_prob(probs, pred_digit)
        st.pyplot(fig_prob, use_container_width=True)

        # Top-3
        top3_idx = np.argsort(probs)[::-1][:3]
        st.markdown("**Top-3 predicciones:**")
        for rank, i in enumerate(top3_idx):
            bar = "█" * int(probs[i] * 20)
            st.markdown(
                f"`{rank+1}.` **{i}** — `{probs[i]*100:5.1f}%` `{bar}`"
            )


# ══ TAB 2: CURVAS DE ENTRENAMIENTO ═══════════════════════════════════════════
with tab2:
    st.markdown("### Curvas de entrenamiento")
    st.pyplot(fig_curvas(history), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Accuracy por época:**")
        acc_data = {
            "Época": list(range(1, epocas_reales+1)),
            "Train Acc (%)": [f"{v*100:.2f}" for v in history.history['accuracy']],
            "Val Acc (%)":   [f"{v*100:.2f}" for v in history.history['val_accuracy']],
        }
        st.dataframe(acc_data, use_container_width=True, height=280)
    with col_b:
        st.markdown("**Loss por época:**")
        loss_data = {
            "Época": list(range(1, epocas_reales+1)),
            "Train Loss": [f"{v:.4f}" for v in history.history['loss']],
            "Val Loss":   [f"{v:.4f}" for v in history.history['val_loss']],
        }
        st.dataframe(loss_data, use_container_width=True, height=280)


# ══ TAB 3: EVALUACIÓN ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Evaluación completa en test set")

    from sklearn.metrics import classification_report

    y_pred_proba = model.predict(X_te, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    col_cm, col_rep = st.columns([1, 1], gap="large")

    with col_cm:
        st.markdown("#### Matriz de Confusión")
        st.pyplot(fig_confusion(y_te, y_pred), use_container_width=True)

    with col_rep:
        st.markdown("#### Métricas por clase")
        report = classification_report(y_te, y_pred, output_dict=True)
        rows = []
        for dig in range(10):
            r = report[str(dig)]
            rows.append({
                "Clase": f"Dígito {dig}",
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1-Score":  f"{r['f1-score']:.3f}",
                "Support":   int(r['support']),
            })
        st.dataframe(rows, use_container_width=True, height=380)

    st.markdown("---")
    # Ejemplos de errores
    st.markdown("#### ❌ Ejemplos mal clasificados")
    X_train_raw, _, X_test_raw, _ = cargar_datos()
    wrong_idx = np.where(y_pred != y_te)[0]

    if len(wrong_idx) == 0:
        st.success("¡Sin errores en el conjunto de prueba!")
    else:
        n_show = min(15, len(wrong_idx))
        fig_err, axes = plt.subplots(2, n_show // 2, figsize=(14, 4))
        fig_err.patch.set_facecolor('#0a0a0f')
        axes = axes.flatten()
        for i, idx in enumerate(wrong_idx[:n_show]):
            axes[i].imshow(X_test_raw[idx], cmap='Reds')
            axes[i].set_title(
                f"R:{y_te[idx]} P:{y_pred[idx]}\n{y_pred_proba[idx,y_pred[idx]]*100:.0f}%",
                fontsize=7, color='#ef4444'
            )
            axes[i].axis('off')
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_err, use_container_width=True)


# ══ TAB 4: EXPLORAR DATASET ══════════════════════════════════════════════════
with tab4:
    st.markdown("### Explorar el dataset MNIST")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = cargar_datos()

    col_e1, col_e2 = st.columns([1, 2])

    with col_e1:
        digito_sel = st.selectbox("Selecciona un dígito", list(range(10)), index=0)
        n_ejemplos = st.slider("Número de ejemplos", 5, 20, 10)

    with col_e2:
        st.markdown(f"#### Muestras del dígito **{digito_sel}** (train set)")
        indices = np.where(y_train_raw == digito_sel)[0]
        sel     = np.random.choice(indices, n_ejemplos, replace=False)

        cols_per_row = 5
        rows_needed  = (n_ejemplos + cols_per_row - 1) // cols_per_row
        fig_g, axes  = plt.subplots(rows_needed, cols_per_row,
                                     figsize=(cols_per_row*2, rows_needed*2.1))
        fig_g.patch.set_facecolor('#0a0a0f')
        if rows_needed == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

        for i, ax in enumerate(axes_flat):
            if i < n_ejemplos:
                ax.imshow(X_train_raw[sel[i]], cmap='gray_r')
                ax.axis('off')
            else:
                ax.set_visible(False)

        plt.tight_layout(pad=0.3)
        st.pyplot(fig_g, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Distribución de clases")
    fig_dist, axes_d = plt.subplots(1, 2, figsize=(12, 4))
    fig_dist.patch.set_facecolor('#0a0a0f')
    for ax in axes_d:
        ax.set_facecolor('#111118')
        for sp in ax.spines.values():
            sp.set_color('#2d2d3f')
        ax.tick_params(colors='#6b7280', labelsize=9)

    cmap_ = plt.cm.plasma(np.linspace(0.2, 0.9, 10))
    u, c = np.unique(y_train_raw, return_counts=True)
    axes_d[0].bar(u, c, color=cmap_, edgecolor='none')
    axes_d[0].set_title('Train (60,000)', color='#d1d5db', fontsize=10)
    axes_d[0].set_xlabel('Dígito', color='#6b7280')
    axes_d[0].set_ylabel('Cantidad', color='#6b7280')
    axes_d[0].set_xticks(range(10))

    u2, c2 = np.unique(y_test_raw, return_counts=True)
    axes_d[1].bar(u2, c2, color=cmap_, edgecolor='none')
    axes_d[1].set_title('Test (10,000)', color='#d1d5db', fontsize=10)
    axes_d[1].set_xlabel('Dígito', color='#6b7280')
    axes_d[1].set_xticks(range(10))

    plt.tight_layout()
    st.pyplot(fig_dist, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#374151; font-family:'JetBrains Mono',monospace; font-size:0.75rem; padding: 0.5rem 0">
  MNIST Classifier · CNN (TensorFlow/Keras) · Streamlit
</div>
""", unsafe_allow_html=True)
