import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ====== 数据准备 (MNIST) ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ====== 简单分类器 ======
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ====== DualPipe 模拟 ======
class DualPipeTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def pipe_a_forward(self, data, target):
        """Pipe A: 前向传播 + loss"""
        output = self.model(data)
        loss = self.criterion(output, target)
        return output, loss

    def pipe_b_backward(self, loss):
        """Pipe B: 反向传播 + 更新参数"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ====== 训练 & 测试 ======
def train_epoch(pipe, device, train_loader):
    pipe.model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # Pipe A
        output, loss = pipe.pipe_a_forward(data, target)
        # Pipe B
        pipe.pipe_b_backward(loss)
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def test(pipe, device, test_loader):
    pipe.model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = pipe.model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# ====== 主程序 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
pipe = DualPipeTrainer(model, optimizer, criterion)

epochs = 5
for epoch in range(1, epochs+1):
    train_loss = train_epoch(pipe, device, train_loader)
    test_acc = test(pipe, device, test_loader)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Accuracy = {test_acc:.2f}%")
