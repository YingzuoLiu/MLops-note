# =========================================================
# QAT（量化感知训练）最小可运行示例
# - 场景：二维点分类（二分类），不需要下载数据集
# - 流程：FP32 训练 → 插入"假量化"节点 → 小学习率微调
# - 目标：直观理解 QAT 的核心操作（量化 + 反量化 的模拟）
# =========================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(42)

# ---------------------------------------------------------
# 1) 合成一个简单的二维点数据集：两个高斯团，做二分类
# ---------------------------------------------------------
def make_toy_data(n_per_class=1000):
    # 类0：以(0, 0)为中心的高斯分布
    x0 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([0.0, 0.0])
    y0 = torch.zeros(n_per_class, dtype=torch.long)

    # 类1：以(3, 3)为中心的高斯分布
    x1 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([3.0, 3.0])
    y1 = torch.ones(n_per_class, dtype=torch.long)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    # 打乱
    idx = torch.randperm(X.size(0))
    X, y = X[idx], y[idx]

    # 切成 train / val
    n_train = int(0.8 * X.size(0))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:], y[n_train:]
    return (X_train, y_train), (X_val, y_val)

(X_train, y_train), (X_val, y_val) = make_toy_data(n_per_class=1500)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# 2) 一个非常小的MLP模型（FP32）
# ---------------------------------------------------------
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)   # 输入2维 → 隐层32
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)   # 输出2类
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------------------------------------
# 3) 训练与评估工具函数（通用）
# ---------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    return correct / total

def train_epochs(model, loader, epochs=10, lr=1e-2):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % max(1, epochs//5) == 0:
            acc = evaluate(model, val_loader)
            print(f"[FP32 Train] epoch={ep:02d}  val_acc={acc:.4f}")
    return model

# ---------------------------------------------------------
# 4) 先用 FP32 正常训练，得到一个“可用”的起点模型
#    （这一步相当于 QAT 的 Step1：基础训练）
# ---------------------------------------------------------
fp32_model = TinyMLP()
fp32_model = train_epochs(fp32_model, train_loader, epochs=8, lr=5e-3)
fp32_acc = evaluate(fp32_model, val_loader)
print(f"FP32 baseline val acc: {fp32_acc:.4f}")

# ---------------------------------------------------------
# 5) 假量化（Fake Quantization）核心
#    - 思想：在前向里把张量先“量化到 INT8 网格”，再还原回 FP32，
#            让网络在训练时“感受到”低精度的限制。
#    - 数学：dq = (round(x/scale + zp) - zp) * scale
#    - 训练：梯度用 STE（Straight-Through Estimator）穿过round/clamp等非光滑操作。
# ---------------------------------------------------------
def fake_quantize(x, num_bits=8, eps=1e-8, per_tensor=True):
    """
    将张量x做“量化+反量化”的模拟（仍然输出FP32，但值被逼近到INT8网格）
    - num_bits=8：模拟 INT8
    - per_tensor=True：用整张量的min/max算一个scale与zero_point（简单好懂）
    - 要点：用 x + (dq - x).detach() 实现 STE，让反向梯度近似穿过量化操作
    """
    # 求量化范围（简单起见：对称或非对称都行；这里用非对称，按min/max动态确定）
    x_detach = x.detach()
    x_min = x_detach.min()
    x_max = x_detach.max()
    # 避免退化（所有值相同会导致除0）
    if (x_max - x_min).abs() < eps:
        return x  # 这种极端情况下就不量化了

    qmin, qmax = 0, (2 ** num_bits) - 1  # INT8：0~255（这里用无符号网格来演示）
    scale = (x_max - x_min) / (qmax - qmin + eps)
    zero_point = torch.round(qmin - x_min / (scale + eps)).clamp(qmin, qmax)

    # 量化到离散网格，再反量化回FP32
    q = torch.round(x / (scale + eps) + zero_point)
    q = q.clamp(qmin, qmax)
    dq = (q - zero_point) * scale

    # STE：前向用dq，反向对x传梯度
    return x + (dq - x).detach()

# ---------------------------------------------------------
# 6) 把“假量化”插在权重和激活上（最常见的QAT做法）
#    我们写一个“带假量化”的 Linear 层包装器
# ---------------------------------------------------------
class FakeQuantLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, w_bits=8, a_bits=8):
        super().__init__(in_f, out_f, bias=bias)
        self.w_bits = w_bits
        self.a_bits = a_bits

    def forward(self, x):
        # (1) 先对输入激活做假量化（模拟INT8激活）
        x_q = fake_quantize(x, num_bits=self.a_bits)

        # (2) 再对权重做假量化（模拟INT8权重）
        w_q = fake_quantize(self.weight, num_bits=self.w_bits)

        # (3) 用“被假量化”的权重与激活做正常的FP32 matmul（注意这就是QAT的精髓）
        out = F.linear(x_q, w_q, self.bias)

        # (4) 也可以对层输出再来一次假量化（模拟逐层INT8激活）
        out_q = fake_quantize(out, num_bits=self.a_bits)
        return out_q

# ---------------------------------------------------------
# 7) 用带假量化的层，搭一个 QAT 版本的模型
# ---------------------------------------------------------
class TinyMLP_QAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FakeQuantLinear(2, 32, w_bits=8, a_bits=8)
        self.fc2 = FakeQuantLinear(32, 32, w_bits=8, a_bits=8)
        self.fc3 = FakeQuantLinear(32, 2,  w_bits=8, a_bits=8)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------------------------------------
# 8) 将FP32训练好的权重拷到QAT模型里（常见做法：用已收敛模型当QAT起点）
# ---------------------------------------------------------
qat_model = TinyMLP_QAT().to(device)
with torch.no_grad():
    # 名称结构是对应的，所以可以逐层拷
    qat_model.fc1.weight.copy_(fp32_model.fc1.weight)
    qat_model.fc1.bias.copy_(fp32_model.fc1.bias)
    qat_model.fc2.weight.copy_(fp32_model.fc2.weight)
    qat_model.fc2.bias.copy_(fp32_model.fc2.bias)
    qat_model.fc3.weight.copy_(fp32_model.fc3.weight)
    qat_model.fc3.bias.copy_(fp32_model.fc3.bias)

# ---------------------------------------------------------
# 9) 在“假量化环境”下进行小学习率微调（QAT Step3）
#    目标：让参数适应量化误差
# ---------------------------------------------------------
def train_qat(model, loader, epochs=6, lr=2e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # 前向里已经包含“假量化”
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % max(1, epochs//6) == 0:
            acc = evaluate(model, val_loader)
            print(f"[QAT Fine-tune] epoch={ep:02d}  val_acc={acc:.4f}")
    return model

qat_model = train_qat(qat_model, train_loader, epochs=6, lr=2e-3)
qat_acc = evaluate(qat_model, val_loader)
print(f"QAT (fake INT8) val acc: {qat_acc:.4f}")

# ---------------------------------------------------------
# 10)（可选）“导出量化”：真正把权重转成INT8存储
#     - 实际工业落地通常会导出到推理框架（TensorRT、ONNX Runtime、TFLite等）
#     - 这里演示一个“简化版导出”，只展示权重量化后的占用变化
# ---------------------------------------------------------
def export_int8_weights(model):
    """
    简化演示：把每个Linear权重量化到int8（0~255）并保存scale/zero_point。
    实际部署通常交给框架完成（会更全面、含图优化）。
    """
    export = []
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantLinear):
            w = module.weight.detach().cpu()
            w_min, w_max = w.min(), w.max()
            qmin, qmax = 0, 255
            scale = (w_max - w_min) / max(1e-8, (qmax - qmin))
            zp = torch.round(qmin - w_min / max(1e-8, scale)).clamp(qmin, qmax)
            q = torch.round(w / max(1e-8, scale) + zp).clamp(qmin, qmax).to(torch.uint8)
            export.append({
                "name": name,
                "int8_weight": q,
                "scale": float(scale),
                "zero_point": int(zp.item()),
                "orig_bytes": w.numel() * 4,      # FP32 4字节
                "int8_bytes": q.numel() * 1       # INT8 1字节
            })
    return export

exported = export_int8_weights(qat_model)
total_fp32 = sum(x["orig_bytes"] for x in exported)
total_int8 = sum(x["int8_bytes"] for x in exported)
print("\n[Export Int8 Summary]")
for x in exported:
    print(f"- {x['name']}: FP32 {x['orig_bytes']}B → INT8 {x['int8_bytes']}B "
          f"(scale={x['scale']:.6f}, zp={x['zero_point']})")
print(f"Total weights: FP32 {total_fp32}B → INT8 {total_int8}B  (x{total_fp32/max(1,total_int8):.2f} smaller)")
