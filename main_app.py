# main_app.py
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ======================
# CNN MODEL (REAL CNN)
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
# LOAD DATA (MNIST REAL)
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# ======================
# TRAIN MODEL
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model... (1 epoch for demo)")
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx > 200:  # limit training for speed
        break

print("Training done")

# ======================
# GUI
# ======================
class App:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label = tk.Label(root, text="Draw a digit")
        self.label.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)

    def preprocess(self):
        img = self.image.resize((28, 28))
        img = np.array(img)
        img = 255 - img  # invert
        img = img / 255.0
        img = (img - 0.1307) / 0.3081
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
        return img

    def predict(self):
        model.eval()
        img = self.preprocess().to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()
        self.label.config(text=f"Prediction: {pred}")

root = tk.Tk()
root.title("MNIST CNN Classifier")
app = App(root)
root.mainloop()


# requirements.txt
# torch
# torchvision
# numpy
# pillow
