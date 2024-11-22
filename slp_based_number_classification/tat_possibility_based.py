import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk

# 数据预处理与加载（与之前代码相同）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 多感知器分类器模型
class MultiPerceptronClassifier(nn.Module):
    def __init__(self):
        super(MultiPerceptronClassifier, self).__init__()
        self.perceptrons = nn.ModuleList([
            nn.Linear(784, 1) for _ in range(10)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        outputs = [perceptron(x) for perceptron in self.perceptrons]
        return torch.cat(outputs, dim=1)


# 训练模型（与之前代码相同）
model = MultiPerceptronClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            binary_labels = torch.zeros_like(model(data), dtype=torch.float)
            for i in range(len(target)):
                binary_labels[i, target[i]] = 1.0

            outputs = model(data)
            loss = criterion(outputs, binary_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'轮次 {epoch + 1}, 损失: {total_loss / len(train_loader):.4f}')


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


# 手写数字识别应用（增强版）
class DigitRecognizerApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("手写数字识别")

        # 主容器
        main_frame = tk.Frame(self.window)
        main_frame.pack(padx=10, pady=10)

        # 左侧绘图区
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)

        # 绘图画布
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()

        # 操作按钮
        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)
        clear_button = tk.Button(button_frame, text="清除", command=self.clear_canvas)
        clear_button.pack(side=tk.LEFT, padx=5)

        # 右侧概率区域
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10)

        # 概率柱状图
        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            bar_frame = tk.Frame(right_frame)
            bar_frame.pack(fill=tk.X, pady=2)

            label = tk.Label(bar_frame, text=str(i), width=2)
            label.pack(side=tk.LEFT)

            bar = tk.Canvas(bar_frame, height=20, bg='white')
            bar.pack(side=tk.LEFT, expand=True, fill=tk.X)

            prob_label = tk.Label(bar_frame, text='0.0%', width=6)
            prob_label.pack(side=tk.LEFT)

            self.prob_bars.append(bar)
            self.prob_labels.append(prob_label)

        # 绘图设置
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

        # 实时预测
        self.recognize_digit()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # 重置概率显示
        for bar, label in zip(self.prob_bars, self.prob_labels):
            bar.delete("prob_bar")
            label.config(text='0.0%')

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
            probs = torch.sigmoid(outputs).squeeze().numpy()

        # 更新概率柱状图
        max_width = 200  # 最大柱状图宽度
        for i, (bar, label, prob) in enumerate(zip(self.prob_bars, self.prob_labels, probs)):
            bar.delete("prob_bar")
            bar_width = int(prob * max_width)
            bar.create_rectangle(0, 0, bar_width, 20, fill='blue', tags="prob_bar")
            label.config(text=f'{prob * 100:.1f}%')

    def run(self):
        self.window.mainloop()


# 训练模型
train(model, train_loader, optimizer, criterion, epochs=5)
evaluate(model, test_loader)

# 启动数字识别应用
app = DigitRecognizerApp(model)
app.run()