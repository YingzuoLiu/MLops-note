import math
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Sampler

# ① 模型与分词器：用 BERT 做文本分类（你也可以换成 DistilBERT/roberta）
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

# -----------------------------
# A. 数据规范（Normalization）
# -----------------------------
# 文本场景中的“规范化”重点在：
# - 统一长度：动态 padding 到“本 batch 的最长序列”（减少无意义 PAD）
# - 统一词表/向量空间：使用同一 tokenizer & 同一预训练 embedding
# - 可选特征尺度：若有数值特征，做 z-score 或 min-max；这里用纯文本就略过

class TextClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        # 注意：这里只做“截断”，不做固定 padding，padding 留给 collate_fn 动态处理
        enc = self.tok(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors=None
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": int(self.labels[i]),
            "length": len(enc["input_ids"]),
        }

# -----------------------------
# B. 动态 batch：动态 padding + 按长度分桶
# -----------------------------
# 动态 padding：每个 batch 只 pad 到本 batch 的最长长度，显著减少无效计算与显存浪费
@dataclass
class DynamicPadCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 找到本 batch 的最大长度
        max_len = max(x["length"] for x in batch)
        input_ids, attn_masks, labels = [], [], []
        for x in batch:
            pad_size = max_len - x["length"]
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_size)
            attn_masks.append(x["attention_mask"] + [0] * pad_size)
            labels.append(x["label"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# 按长度“分桶 + 随机打乱”来降低同一 batch 的长度方差，进一步减少 padding
class BucketSampler(Sampler[List[int]]):
    def __init__(self, lengths: List[int], batch_size: int, buckets: int = 10, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # 将样本按长度排序，再切成 buckets 段，每段内再分 batch
        idxs = list(range(len(lengths)))
        idxs.sort(key=lambda i: lengths[i])
        self.bucket_slices = []
        n = len(idxs)
        bucket_size = math.ceil(n / buckets)
        for b in range(buckets):
            s = b * bucket_size
            e = min(n, (b + 1) * bucket_size)
            if s < e:
                self.bucket_slices.append(idxs[s:e])

    def __iter__(self):
        all_batches = []
        for sl in self.bucket_slices:
            cur = sl[:]
            if self.shuffle:
                random.shuffle(cur)
            for i in range(0, len(cur), self.batch_size):
                all_batches.append(cur[i:i + self.batch_size])
        if self.shuffle:
            random.shuffle(all_batches)
        for b in all_batches:
            yield b

    def __len__(self):
        # 近似返回 batch 数量
        total = 0
        for sl in self.bucket_slices:
            total += math.ceil(len(sl) / self.batch_size)
        return total

# -----------------------------
# C. 模型定义 + 内存优化点
# -----------------------------
class BertCLS(nn.Module):
    def __init__(self, backbone="bert-base-uncased", num_labels=2, gradient_checkpointing=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        # 激活重计算（Gradient Checkpointing）：用计算换显存
        if gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] 向量（不同模型字段名不同，BERT 的 pooler 或 last_hidden_state[:,0] 都可）
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        else:
            h = out.last_hidden_state[:, 0, :]
        logits = self.classifier(h)
        return logits

# -----------------------------
# D. 训练循环：AMP混合精度、梯度累积、梯度裁剪、OOM 降级
# -----------------------------
def train(
    model,
    train_loader,
    val_loader,
    epochs=3,
    lr=2e-5,
    max_grad_norm=1.0,
    grad_acc_steps=1,
    use_amp=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader) // grad_acc_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, num_training_steps // 10),
        num_training_steps=num_training_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ce = nn.CrossEntropyLoss()

    model.train()
    global_step = 0

    for ep in range(epochs):
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = ce(logits, batch["labels"]) / grad_acc_steps

            # AMP 缩放梯度，减少溢出风险
            scaler.scale(loss).backward()

            if (step + 1) % grad_acc_steps == 0:
                # 梯度裁剪：防止梯度爆炸
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += loss.item() * grad_acc_steps

            # —— OOM 保护：如果显存溢出，自动降级 batch / 关闭 amp —— #
            # 实际生产可用 try/except 包裹前向后向；示意逻辑如下（不在主循环里捕获会更清晰）
            # try:
            #   ...forward/backward...
            # except RuntimeError as e:
            #   if "out of memory" in str(e).lower():
            #       torch.cuda.empty_cache()
            #       use_amp = False  # 降级：关闭 AMP
            #       grad_acc_steps *= 2  # 或者增大累积步数等
            #       记录日志并继续

            # 可选：训练时打印显存信息（调参期很有用）
            # if step % 100 == 0 and device.startswith("cuda"):
            #     alloc = torch.cuda.memory_allocated() / 1024**2
            #     res = torch.cuda.memory_reserved() / 1024**2
            #     print(f"[mem] allocated={alloc:.1f}MB reserved={res:.1f}MB")

        val_loss, val_acc = evaluate(model, val_loader, ce, device, use_amp)
        print(f"Epoch {ep+1}/{epochs} - train_loss={running_loss/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

def evaluate(model, loader, ce, device, use_amp=True):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = ce(logits, batch["labels"])
            pred = logits.argmax(dim=-1)
            correct += (pred == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            loss_sum += loss.item()
    model.train()
    return loss_sum / len(loader), correct / total

# -----------------------------
# E. 组装：DataLoader + Sampler + Collator
# -----------------------------
def build_loaders(texts_train, labels_train, texts_val, labels_val, backbone="bert-base-uncased",
                  batch_size=32, max_len=256, buckets=12, num_workers=2):
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    ds_tr = TextClsDataset(texts_train, labels_train, tokenizer, max_len=max_len)
    ds_va = TextClsDataset(texts_val, labels_val, tokenizer, max_len=max_len)

    lengths_tr = [len(ds_tr[i]["input_ids"]) for i in range(len(ds_tr))]
    sampler_tr = BucketSampler(lengths_tr, batch_size=batch_size, buckets=buckets, shuffle=True)

    collate = DynamicPadCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0)

    dl_tr = DataLoader(
        ds_tr,
        batch_sampler=sampler_tr,              # 用我们自定义的分桶采样器
        collate_fn=collate,                    # 动态 padding 到当前 batch 的 max_len
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )

    # 验证集可用固定 batch sampler（或同样用分桶）
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return dl_tr, dl_va, tokenizer

# -----------------------------
# F. 示例入口
# -----------------------------
if __name__ == "__main__":
    # 伪数据示例（换成你的数据）
    texts_train = ["i love this movie", "this is terrible", "great acting", "bad script"] * 512
    labels_train = [1, 0, 1, 0] * 512
    texts_val = ["i like it", "awful film"] * 64
    labels_val = [1, 0] * 64

    BATCH = 32
    MAXLEN = 256
    EPOCHS = 3
    GRAD_ACC = 2               # 梯度累积：等效大 batch 而不额外占显存
    USE_AMP = True             # 混合精度：显存更省，速度更快
    GP_CHECKPOINT = True       # 激活重计算：再省显存（训练稍慢）

    dl_tr, dl_va, tok = build_loaders(
        texts_train, labels_train, texts_val, labels_val,
        backbone="bert-base-uncased",
        batch_size=BATCH, max_len=MAXLEN, buckets=12, num_workers=2
    )

    model = BertCLS(backbone="bert-base-uncased", num_labels=2, gradient_checkpointing=GP_CHECKPOINT)

    # 可选：在 A100/RTX40 上提升 matmul 精度性能
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    train(
        model,
        train_loader=dl_tr,
        val_loader=dl_va,
        epochs=EPOCHS,
        lr=2e-5,
        max_grad_norm=1.0,
        grad_acc_steps=GRAD_ACC,
        use_amp=USE_AMP
    )
