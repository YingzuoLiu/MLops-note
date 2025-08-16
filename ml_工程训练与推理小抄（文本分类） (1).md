# ML 工程训练与推理小抄（文本分类）

> 目标：快速回忆工程实操要点。涵盖数据规范、动态 batch、内存优化、梯度检查点、计算图优化、分布式训练、推理导出与监控。

---

## 0. 最小可复现骨架

- 固定随机种子（含 DataLoader）

```python
SEED = 2025
import torch, random, numpy as np
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# DataLoader 生成器
g = torch.Generator(); g.manual_seed(SEED)
```

- 评估/推理

```python
model.eval()
with torch.no_grad():
    ...
```

---

## 1. 数据规范（Data Normalization）

- **Tokenizer 统一**：同一 vocab/规则，`truncation=True`，训练/验证一致。
- **动态 padding**：只 pad 到批内最大长度，减少无效计算。

```python
@dataclass
class DynamicPadCollator:
    pad_id: int
    def __call__(self, batch):
        L = max(x['length'] for x in batch)
        def pad(arr, pad_id):
            return arr + [pad_id]*(L-len(arr))
        return {
            'input_ids': torch.tensor([pad(x['input_ids'], self.pad_id)]),
            'attention_mask': torch.tensor([pad(x['attention_mask'], 0)]),
            'labels': torch.tensor([x['label'] for x in batch])
        }
```

- **按长度分桶采样**：同批长度接近，padding 更少。

```python
sampler = BucketSampler(lengths, batch_size=32, buckets=12, shuffle=True)
```

---

## 2. 动态 Batch（训练端）

- **分桶 + 动态 padding**（见上）。
- **服务端推理**：用 TensorRT/ONNX Runtime/TFS 的动态合批（根据流量自动凑批）。

**好处**：吞吐 ↑，延迟 ↓，显存更集中在有效 token。

---

## 3. 显存与稳定性优化

- **AMP 混合精度**（训练）

```python
scaler = torch.cuda.amp.GradScaler(enabled=True)
with torch.cuda.amp.autocast(True):
    loss = criterion(model(x), y)
scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
```

- **梯度累积**：等效大 batch 不涨峰值显存。

```python
loss = loss / grad_acc_steps
loss.backward()
if (step+1)%grad_acc_steps==0:
    clip_grad_norm_(model.parameters(), 1.0)
    optim.step(); optim.zero_grad()
```

- **梯度裁剪**：`clip_grad_norm_` 防爆炸。
- **DataLoader**：`pin_memory=True, num_workers>0, prefetch_factor`。
- **OOM 降级**（示意）

```python
try:
    ... forward/backward ...
except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        torch.cuda.empty_cache()
        USE_AMP=False  # 或提高 grad_acc_steps / 降 seq_len
```

---

## 4. 梯度检查点（Gradient Checkpointing）

- **何时开**：显存吃紧/深模型（BERT/GPT/ViT/U-Net）。
- **收益**：激活显存 ↓30–60%，训练时延 ↑20–40%。
- **Transformers 开启**

```python
hf_model.gradient_checkpointing_enable()
# Decoder 训练建议：hf_model.config.use_cache=False
```

- **验证是否已开**

```python
print(getattr(hf_model, 'is_gradient_checkpointing', False))
```

- **只对重层开**（更细粒度）

```python
from torch.utils.checkpoint import checkpoint
x = checkpoint(module, x)
```

---

## 5. 计算图优化（训练/推理）

- **张量化/去装箱**：避免 `.item()/.tolist()/for tensor:` 热路径出现。

```python
# ❌ 不要
val = 0.0
for t in x: val += float(t.item())
# ✅ 要
val = x.sum()
```

- `torch.compile(model)`（PyTorch 2.x）让 Inductor 做融合/内存复用。
- **常见融合**：MatMul+Bias+GELU、Conv+BN(+ReLU)、Dense+Softmax（推理端）。
- **形状专化**：固定/上限 `max_seq_len`，减少动态形状开销。

---

## 6. 分布式训练：DDP 速记

- **启动**：每卡一进程

```bash
torchrun --nproc_per_node=8 train.py --arg ...
```

- **样板**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
torch.cuda.set_device(local_rank)
model = DDP(model.cuda(), device_ids=[local_rank])
# sampler + set_epoch(epoch)
```

- `** vs **`：前者是底层通信原语；后者是同步数据并行封装（自动梯度 all-reduce、桶化与重叠）。
- **检查一致性**：多 rank 参数/度量应一致（可比较范数）。

> 大模型：FSDP/ZeRO 与 GC/AMP 叠加，显存更省。

---

## 7. 推理导出与加速

- **导出 ONNX（动态轴）**

```python
torch.onnx.export(model.eval(), (input_ids, attn_mask), 'model.onnx',
    input_names=['input_ids','attention_mask'], output_names=['logits'],
    dynamic_axes={'input_ids':{0:'batch',1:'seq'}, 'attention_mask':{0:'batch',1:'seq'}})
```

- **部署**：ONNX Runtime / TensorRT
  - 开启 FP16/BF16
  - 算子融合（LayerNorm/GELU/FusedMatMul）
  - **服务端动态合批**（traffic-aware）
  - KV-Cache（Transformer 推理）

---

## 8. 监控与日志

- **训练**：loss、lr、吞吐（samples/s）、步时、`allocated/reserved MB`。

```python
if step % 100 == 0 and torch.cuda.is_available():
    print('mem(MB)', round(torch.cuda.memory_allocated()/2**20,1),
                     round(torch.cuda.memory_reserved()/2**20,1))
```

- **评估**：acc/F1/AUROC + confusion matrix；早停与 best ckpt。

---

## 9. 常见坑位排查

- DDP 未用 `DistributedSampler` / 未 `set_epoch` → 数据重复/打乱不一致。
- 训练里调用 `.item()` 参与控制流 → 打断图优化，慢。
- Decoder 训练 `use_cache=True` → 与 GC 冲突/显存升高。
- 全局 MAX\_LEN 固定大 → padding 浪费严重（应分桶 + 动态 padding）。
- OOM 后未释放缓存 → `torch.cuda.empty_cache()`。

---

## 10. 快速决策树

- **OOM？** → AMP ✅ → GC ✅ → 分桶+动态 padding ✅ → 累积梯度 ✅ → FSDP/ZeRO ✅ → 降 seq\_len/批次。
- **吞吐低？** → DataLoader 优化 → torch.compile → 融合/精度降级（FP16/BF16）→ 动态合批（推理）。
- **结果不稳？** → 种子固定 → 梯度裁剪 → 扩大等效 batch（累积）。

---




---

## 7.1 ONNX Runtime 动态合批微服务（Python 版）

> 文件：`serve_ort_dynamic_batch.py`（示例：请求在 1–2ms 内聚合到一个 batch；无请求阻塞则立即执行）

```python
import asyncio, time
from typing import List, Dict
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ===== ORT Session（GPU 优先，CPU 兜底）=====
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search":"DEFAULT"})] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
session = ort.InferenceSession("model.onnx", sess_options=sess_opts, providers=providers)

# ===== 输入协议（已分词 & pad_id 交由服务端处理）=====
class Item(BaseModel):
    input_ids: List[int]
    attention_mask: List[int]

app = FastAPI()

# ===== 简易微批中枢 =====
class MicroBatcher:
    def __init__(self, max_batch=32, max_delay_ms=2, pad_id=0):
        self.q: asyncio.Queue = asyncio.Queue()
        self.max_batch = max_batch
        self.max_delay_ms = max_delay_ms
        self.pad_id = pad_id
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._worker())

    async def _worker(self):
        while True:
            item, fut = await self.q.get()
            batch = [(item, fut)]
            t0 = time.time()
            # 在时间窗口内尽可能收集更多请求
            while len(batch) < self.max_batch and (time.time() - t0) * 1000 < self.max_delay_ms:
                try:
                    batch.append(await asyncio.wait_for(self.q.get(), timeout=self.max_delay_ms/1000))
                except asyncio.TimeoutError:
                    break
            # 组装 batch（动态 padding 到批内 max_len）
            inputs = [x[0] for x in batch]
            L = max(len(x.input_ids) for x in inputs)
            def pad(a, v):
                return a + [v]*(L-len(a))
            input_ids = np.asarray([pad(x.input_ids, self.pad_id) for x in inputs], dtype=np.int64)
            attn = np.asarray([pad(x.attention_mask, 0) for x in inputs], dtype=np.int64)
            # 推理
            logits = session.run(["logits"], {"input_ids": input_ids, "attention_mask": attn})[0]
            preds = logits.argmax(axis=-1).tolist()
            # 回填结果
            for (_, fut), y in zip(batch, preds):
                if not fut.done(): fut.set_result({"label": int(y)})

    async def infer(self, item: Item):
        fut: asyncio.Future = self.loop.create_future()
        await self.q.put((item, fut))
        return await fut

mb = MicroBatcher(max_batch=32, max_delay_ms=2, pad_id=0)

@app.post("/predict")
async def predict(item: Item):
    return await mb.infer(item)

# 启动：uvicorn serve_ort_dynamic_batch:app --host 0.0.0.0 --port 8000
```

要点：

- **动态合批**：在 `max_delay_ms` 时间窗内收集请求，批内 **动态 padding**。
- **图优化**：`ORT_ENABLE_ALL` 打开融合/常量折叠；若模型为 FP16 权重，CUDA EP 会走半精度核。
- **吞吐/延迟权衡**：调 `max_batch` 与 `max_delay_ms`。

---

## 7.2 TensorRT + Triton 动态合批（配置）

> 构建 TensorRT 引擎（FP16 + 设定形状 profile）

```bash
trtexec \
  --onnx=model.onnx --saveEngine=model_fp16.plan --fp16 --workspace=4096 \
  --minShapes=input_ids:1x8,attention_mask:1x8 \
  --optShapes=input_ids:16x128,attention_mask:16x128 \
  --maxShapes=input_ids:64x512,attention_mask:64x512
```

> Triton 目录

```
models/
└── textcls_trt/
    ├── 1/model.plan
    └── config.pbtxt
```

> `config.pbtxt`（动态合批 + FP16 打开）

```protobuf
name: "textcls_trt"
platform: "tensorrt_plan"
max_batch_size: 64
input: [
  { name: "input_ids", data_type: TYPE_INT64, dims: [ -1 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1 ] }
]
output: [ { name: "logits", data_type: TYPE_FP32, dims: [ -1 ] } ]
instance_group [{ kind: KIND_GPU, count: 1 }]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 1000  # 1ms
}
optimization { execution_accelerators { gpu_execution_accelerator : [ { name : "tensorrt" } ] } }
```

启动 Triton：

```bash
tritonserver --model-repository=./models
```

---

## 6.1 DDP 启动模板（`train_ddp.py`）

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup():
    # 由 torchrun 注入 LOCAL_RANK/RANK/WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() and torch.distributed.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # 构造数据集与采样器
    dataset = ...  # 自定义
    sampler = DistributedSampler(dataset, shuffle=True)
    dl = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True)

    # 模型
    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False, gradient_as_bucket_view=True)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in dl:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); optim.zero_grad(set_to_none=True)

    cleanup()

if __name__ == "__main__":
    epochs = 3
    main()
```

启动命令：

```bash
# 单机 8 卡
torchrun --nproc_per_node=8 train_ddp.py

# 多机（示例）：在每台机器上分别执行，按需填写
# MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE
torchrun --nnodes=2 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=29500 \
         --nproc_per_node=8 train_ddp.py
```

> 备注：Windows 环境通常无法使用 NCCL（GPU 分布式），建议在 Linux 上使用 NCCL；Windows 如需调试可用 `gloo` 后端（CPU）。

