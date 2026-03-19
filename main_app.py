import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# NOTE: sklearn does NOT support CNNs directly, so we simulate with MLP

# Load data
digits = load_digits()
X = digits.data
y = digits.target

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

# GUI
class App:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label = tk.Label(root, text="Draw a digit")
        self.label.pack()

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill=255)

    def preprocess(self):
        img = self.image.resize((8, 8))
        img = np.array(img)
        img = 16 - (img / 16)
        return img.flatten().reshape(1, -1)

    def predict(self):
        data = self.preprocess()
        data = scaler.transform(data)
        pred = model.predict(data)[0]
        self.label.config(text=f"Prediction: {pred}")

root = tk.Tk()
root.title("MNIST Digit Classifier")
app = App(root)
root.mainloop()
