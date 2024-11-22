import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 多感知器分类器模型
class MultiPerceptronClassifier(nn.Module):
    def __init__(self):
        super(MultiPerceptronClassifier, self).__init__()
        # 为每个数字创建一个单层感知器
        self.perceptrons = nn.ModuleList([
            nn.Linear(784, 1) for _ in range(10)
        ])

    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        # 每个感知器的输出
        outputs = [perceptron(x) for perceptron in self.perceptrons]
        return torch.cat(outputs, dim=1)


# 初始化模型、损失函数和优化器
model = MultiPerceptronClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train(model, train_loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # 创建二值化标签
            binary_labels = torch.zeros_like(model(data), dtype=torch.float)
            for i in range(len(target)):
                binary_labels[i, target[i]] = 1.0

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, binary_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'轮次 {epoch + 1}, 损失: {total_loss / len(train_loader):.4f}')


# 评估函数
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            predicted = torch.argmax(outputs, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'测试准确率: {accuracy:.2f}%')


# 手写数字识别应用
class DigitRecognizerApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("手写数字识别")

        # 绘图画布
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()

        # 预测标签
        self.prediction_label = tk.Label(self.window, text="", font=("Arial", 24))
        self.prediction_label.pack()

        # 按钮
        clear_button = tk.Button(self.window, text="清除", command=self.clear_canvas)
        clear_button.pack()
        recognize_button = tk.Button(self.window, text="识别", command=self.recognize_digit)
        recognize_button.pack()

        # 绘图设置
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        # 绘制手写数字
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        # 清除画布
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="")

    def recognize_digit(self):
        # 图像预处理
        resized_image = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(resized_image) / 255.0

        # 转换为张量并标准化
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)

        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            predicted = torch.argmax(outputs, dim=1).item()

        self.prediction_label.config(text=f"预测数字: {predicted}")

    def run(self):
        self.window.mainloop()


# 训练模型
train(model, train_loader, optimizer, criterion, epochs=20)
evaluate(model, test_loader)

# 启动数字识别应用
app = DigitRecognizerApp(model)
app.run()