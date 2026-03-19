import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import io
import base64
from streamlit_drawable_canvas import st_canvas

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Neural Net",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0e0e0e;
    color: #f0f0f0;
  }
  .stApp { background-color: #0e0e0e; }

  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

  .title-block {
    text-align: center;
    padding: 2rem 0 1rem 0;
  }
  .title-block h1 {
    font-size: 3rem;
    background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 0;
  }
  .title-block p {
    color: #888;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.3rem;
  }

  .metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
  }
  .metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2px;
  }
  .metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #ff6b35;
    line-height: 1.1;
  }

  .result-banner {
    background: linear-gradient(135deg, #1a1a1a, #1f1a14);
    border: 1px solid #ff6b35;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-banner .digit {
    font-size: 5rem;
    font-weight: 800;
    color: #ff6b35;
    line-height: 1;
  }
  .result-banner .confidence {
    font-family: 'Space Mono', monospace;
    color: #888;
    font-size: 0.9rem;
  }

  .stButton > button {
    background: #ff6b35 !important;
    color: #0e0e0e !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s ease !important;
    width: 100%;
  }
  .stButton > button:hover {
    background: #e55a25 !important;
    transform: translateY(-1px);
  }

  .stProgress > div > div > div > div {
    background: #ff6b35 !important;
  }

  .sidebar-section {
    background: #1a1a1a;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #2a2a2a;
  }

  hr { border-color: #2a2a2a; }

  [data-testid="stSidebar"] {
    background-color: #111111 !important;
  }

  .bar-container { margin: 0.3rem 0; }
  .bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #aaa;
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
  }
  .bar-outer {
    background: #2a2a2a;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
  }
  .bar-inner {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #ff6b35, #f7c59f);
    transition: width 0.5s ease;
  }
</style>
""", unsafe_allow_html=True)


# ─── CNN MODEL ──────────────────────────────────────────────────────────────────
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── TRAINING / LOADING ─────────────────────────────────────────────────────────
MODEL_PATH = "mnist_cnn.pth"

@st.cache_resource(show_spinner=False)
def get_model():
    """Load or train the CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistCNN().to(device)

    import os
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device, None  # already trained

    return model, device, "needs_training"


def train_model(model, device, epochs, progress_bar, status_text):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(train_loader)
    step = 0
    history = []

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            step += 1
            progress_bar.progress(step / total_steps)

        acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "acc": acc})
        status_text.markdown(
            f"<span style='font-family:Space Mono;color:#ff6b35'>Época {epoch+1}/{epochs} — "
            f"Loss: {avg_loss:.4f} — Acc: {acc:.2f}%</span>",
            unsafe_allow_html=True
        )
        scheduler.step()

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return history


# ─── INFERENCE ──────────────────────────────────────────────────────────────────
def preprocess_canvas(image_data: np.ndarray) -> torch.Tensor:
    """Convert canvas RGBA numpy array → normalized 1×28×28 tensor."""
    img = Image.fromarray(image_data.astype(np.uint8), "RGBA")
    img = img.convert("L")                        # grayscale
    img = ImageOps.invert(img)                    # white digit on black → black bg
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081                 # same normalization as training
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # 1×1×28×28
    return tensor


def predict(model, device, tensor) -> tuple[int, np.ndarray]:
    with torch.no_grad():
        output = model(tensor.to(device))
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs


# ─── APP LAYOUT ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>🧠 MNIST CNN</h1>
  <p>Clasificador de dígitos — Red Neuronal Convolucional</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    epochs = st.slider("Épocas de entrenamiento", 1, 15, 5)
    stroke_width = st.slider("Grosor del trazo", 8, 35, 20)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono;font-size:0.75rem;color:#555;line-height:1.8'>
    <b style='color:#888'>Arquitectura CNN</b><br>
    Conv 32 → Conv 32 → Pool<br>
    Conv 64 → Conv 64 → Pool<br>
    FC 512 → Softmax 10<br><br>
    <b style='color:#888'>Dataset</b><br>
    MNIST 60k train / 10k test
    </div>
    """, unsafe_allow_html=True)

# ── Model init ──
model, device, status = get_model()

# ── Training section ──
st.markdown("## 1 · Entrenar el Modelo")

col_a, col_b = st.columns([3, 1])
with col_a:
    import os
    if os.path.exists(MODEL_PATH):
        st.success("✅ Modelo cargado desde archivo guardado")
    else:
        st.info("ℹ️ No hay modelo guardado. Entrena primero.")

with col_b:
    train_btn = st.button("🚀 Entrenar")

if train_btn:
    prog = st.progress(0.0)
    stat = st.empty()
    with st.spinner("Descargando MNIST y entrenando…"):
        history = train_model(model, device, epochs, prog, stat)
    st.success(f"✅ Entrenamiento completado — Accuracy final: {history[-1]['acc']:.2f}%")

    # mini loss chart
    losses = [h["loss"] for h in history]
    accs   = [h["acc"]  for h in history]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card"><div class="label">Loss final</div>'
                    f'<div class="value">{losses[-1]:.4f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="label">Accuracy final</div>'
                    f'<div class="value">{accs[-1]:.1f}%</div></div>', unsafe_allow_html=True)
    st.line_chart({"Loss": losses, "Accuracy (%)": accs})

# ── Drawing section ──
st.markdown("---")
st.markdown("## 2 · Dibuja un Dígito")
st.markdown("<p style='color:#666;font-size:0.9rem'>Dibuja un número del 0 al 9 en el pad y pulsa <b>Clasificar</b></p>",
            unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    classify_btn = st.button("🔍 Clasificar")

with col2:
    if classify_btn:
        if not os.path.exists(MODEL_PATH):
            st.error("⚠️ Primero entrena el modelo")
        elif canvas_result.image_data is None or canvas_result.image_data.sum() == 0:
            st.warning("✏️ Dibuja algo primero")
        else:
            tensor = preprocess_canvas(canvas_result.image_data)
            digit, probs = predict(model, device, tensor)

            st.markdown(f"""
            <div class="result-banner">
              <div style='font-family:Space Mono;font-size:0.8rem;color:#666;text-transform:uppercase;
                          letter-spacing:3px;margin-bottom:0.5rem'>Predicción</div>
              <div class="digit">{digit}</div>
              <div class="confidence">{probs[digit]*100:.1f}% confianza</div>
            </div>
            """, unsafe_allow_html=True)

            # probability bars
            st.markdown("**Probabilidades por clase:**")
            sorted_idx = np.argsort(probs)[::-1]
            for i in sorted_idx:
                pct = probs[i] * 100
                bar_color = "#ff6b35" if i == digit else "#3a3a3a"
                st.markdown(f"""
                <div class="bar-container">
                  <div class="bar-label"><span>{i}</span><span>{pct:.1f}%</span></div>
                  <div class="bar-outer">
                    <div class="bar-inner" style="width:{pct}%;background:{bar_color}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#1a1a1a;border:1px dashed #333;border-radius:12px;
                    height:200px;display:flex;align-items:center;justify-content:center;
                    margin-top:1rem;'>
          <span style='color:#444;font-family:Space Mono;font-size:0.85rem'>
            ← dibuja y presiona Clasificar
          </span>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:Space Mono;font-size:0.7rem;color:#333;padding:1rem'>
  CNN · PyTorch · Streamlit · MNIST
</div>
""", unsafe_allow_html=True)
