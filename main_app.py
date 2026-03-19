import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import os

# ======================
# CONFIG
# ======================
MODEL_PATH = "mnist_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CNN MODEL
# ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ======================
# LOAD / TRAIN MODEL
# ======================
@st.cache_resource
def load_or_train_model():
    model = CNN().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model

    st.info("Entrenando modelo (solo la primera vez)... ⏳")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(2):  # puedes subir esto
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx > 200:  # limitar tiempo
                break

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()

    st.success("Modelo entrenado y guardado ✅")
    return model

model = load_or_train_model()

# ======================
# STREAMLIT UI
# ======================
st.title("🧠 MNIST CNN - Dibuja un número")

st.write("Dibuja un número del 0 al 9 👇")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ======================
# PREDICTION
# ======================
if canvas_result.image_data is not None:

    if st.button("Predecir"):
        img = canvas_result.image_data

        # Convertir a escala de grises
        img = Image.fromarray((img[:, :, 0]).astype("uint8"))
        img = img.resize((28, 28))

        img = np.array(img)

        # Invertir colores (fondo negro, número blanco)
        img = 255 - img

        # Normalizar
        img = img / 255.0
        img = (img - 0.1307) / 0.3081

        # Tensor
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(img.to(DEVICE))
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.subheader(f"🔢 Predicción: {pred}")

        st.write("Probabilidades:")
        for i, p in enumerate(probs.cpu().numpy()[0]):
            st.write(f"{i}: {p:.4f}")
