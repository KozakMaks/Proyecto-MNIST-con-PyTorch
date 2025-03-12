import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predicción MNIST")
        self.resizable(0,0)
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.button_predict = tk.Button(self, text="Predecir", command=self.predict)
        self.button_predict.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.button_clear = tk.Button(self, text="Limpiar", command=self.clear)
        self.button_clear.grid(row=1, column=2, columnspan=2, sticky="nsew")
        self.label = tk.Label(self, text="Dibuja un dígito", font=("Helvetica", 16))
        self.label.grid(row=2, column=0, columnspan=4)
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Dibuja un dígito")
    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img)/255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
        self.label.config(text="Predicción: " + str(pred))
app = App()
app.mainloop()
