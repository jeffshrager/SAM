"""
Minimal decoder-only transformer for the nanoalgebra negation-distribution task.

Usage:
    python train.py                          # generate data, train with defaults
    python train.py --data_dir data/...      # use a saved dataset
    python train.py --n_layer 1 --d_model 64 --max_iters 3000
"""

import argparse
import math
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_gen import (
    ALL_OPS, VOCAB_SIZE, _C2I, _I2C,
    make_dataset, load_dataset, encode,
)

# ── Vocabulary extension ──────────────────────────────────────────────────────
PAD_ID   = VOCAB_SIZE        # one extra token used for padding
SEP_ID   = _C2I['>']        # '>' appears only in the '->' separator
EXT_VOCAB = VOCAB_SIZE + 1  # total vocab size seen by the model


# ── Dataset ───────────────────────────────────────────────────────────────────

class AlgebraDataset(Dataset):
    """
    Each sample is a padded (x, y, mask) triple of length (max_len - 1).
      x    : input token ids
      y    : target token ids (x shifted left by one)
      mask : float, 1.0 only at positions predicting answer chars (after '>')
    """
    def __init__(self, pairs: list[str], max_len: int):
        self.samples = []
        skipped = 0
        for p in pairs:
            ids = encode(p)
            if len(ids) > max_len:
                skipped += 1
                continue
            L   = len(ids)
            pad = max_len - L
            sep = ids.index(SEP_ID)   # index of '>' in ids

            padded = ids + [PAD_ID] * pad
            x = padded[:-1]           # length max_len - 1
            y = padded[1:]            # length max_len - 1
            # mask[i] = 1 where y[i] is an answer char (y[sep] is the first one)
            mask = [0.0] * sep + [1.0] * (L - 1 - sep) + [0.0] * pad
            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float),
            ))
        if skipped:
            print(f"  [dataset] skipped {skipped} sequences longer than max_len={max_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    max_len:   int   = 256
    n_layer:   int   = 2
    n_head:    int   = 4
    d_model:   int   = 128
    dropout:   float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            'mask', torch.tril(torch.ones(cfg.max_len, cfg.max_len))
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        def reshape(t):
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = self.drop(torch.softmax(att, dim=-1))
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(EXT_VOCAB, cfg.d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, EXT_VOCAB, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None, mask=None):
        B, T = x.shape
        pos    = torch.arange(T, device=x.device).unsqueeze(0)
        h      = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        h      = self.blocks(h)
        logits = self.head(self.ln_f(h))   # (B, T, EXT_VOCAB)

        loss = None
        if targets is not None:
            # cross-entropy over all positions, then apply answer mask
            raw = nn.functional.cross_entropy(
                logits.view(-1, EXT_VOCAB),
                targets.view(-1),
                ignore_index=PAD_ID,
                reduction='none',
            ).view(B, T)
            if mask is not None:
                loss = (raw * mask).sum() / mask.sum().clamp(min=1)
            else:
                loss = raw.mean()

        return logits, loss


# ── Greedy generation ─────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(model, prompt_ids: list[int], device) -> list[int]:
    """Extend prompt greedily until PAD or max_len is reached."""
    model.eval()
    ids = list(prompt_ids)
    max_len = model.cfg.max_len
    while len(ids) < max_len:
        x      = torch.tensor([ids], device=device)
        logits, _ = model(x)
        nxt    = logits[0, -1].argmax().item()
        if nxt == PAD_ID:
            break
        ids.append(nxt)
    return ids


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Exact-match accuracy: all answer tokens must be correct."""
    model.eval()
    correct = total = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        preds     = logits.argmax(-1)                     # (B, T)
        # a sample is correct if every masked position matches
        ok = ((preds == y) | (mask == 0)).all(dim=1)
        correct += ok.sum().item()
        total   += x.size(0)
    return correct / total if total else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = (
        'cuda'  if torch.cuda.is_available()  else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Device : {device}")

    # ── Data ──
    if args.data_dir:
        data, _ = load_dataset(args.data_dir)
        print(f"Loaded dataset from {args.data_dir}")
    else:
        gen_cfg = dict(
            n_train=args.n_train, n_test=args.n_test,
            max_depth_train=args.max_depth_train,
            max_depth_test=args.max_depth_test,
            allowed_ops=tuple(ALL_OPS),
            require_neg=True, seed=args.seed,
        )
        data = make_dataset(**gen_cfg)
        print(f"Generated dataset: "
              f"train={len(data['train'])}  "
              f"test_same={len(data['test_same'])}  "
              f"test_deeper={len(data['test_deeper'])}")

    max_len = args.max_len
    train_ds = AlgebraDataset(data['train'],       max_len)
    same_ds  = AlgebraDataset(data['test_same'],   max_len)
    deep_ds  = AlgebraDataset(data['test_deeper'], max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    same_loader  = DataLoader(same_ds,  batch_size=args.batch_size, shuffle=False)
    deep_loader  = DataLoader(deep_ds,  batch_size=args.batch_size, shuffle=False)

    # ── Model ──
    cfg   = GPTConfig(
        max_len=max_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
    )
    model = NanoGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}  ({cfg.n_layer}L {cfg.n_head}H d{cfg.d_model})")

    # ── Optimiser + schedule ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1
    )

    def lr_lambda(step):
        if step < args.warmup_iters:
            return step / max(1, args.warmup_iters)
        t = (step - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loop ──
    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(args.max_iters + 1):
        # fetch next batch, cycling through epochs
        try:
            x, y, mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y, mask = next(train_iter)

        x, y, mask = x.to(device), y.to(device), mask.to(device)
        _, loss = model(x, y, mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"step {step:5d}  loss {loss.item():.4f}  "
                  f"lr {lr_now:.2e}  {time.time()-t0:.0f}s")

        if step > 0 and step % args.eval_every == 0:
            acc_s = evaluate(model, same_loader, device)
            acc_d = evaluate(model, deep_loader, device)
            print(f"         >> test_same={acc_s:.3f}  test_deeper={acc_d:.3f}")
            model.train()

    # ── Final eval + samples ──
    acc_s = evaluate(model, same_loader, device)
    acc_d = evaluate(model, deep_loader, device)
    print(f"\nFinal  test_same={acc_s:.3f}  test_deeper={acc_d:.3f}")

    print("\nSample predictions (greedy):")
    for raw in data['test_same'][:8]:
        src, tgt = raw.split('->')
        prompt   = encode(src + '->')
        out_ids  = greedy_decode(model, prompt, device)
        out_str  = ''.join(_I2C.get(i, '?') for i in out_ids[len(prompt):])
        mark     = '✓' if out_str == tgt else '✗'
        print(f"  {mark}  {src}->  pred={out_str!r}  gold={tgt!r}")

    # ── Save ──
    if args.save:
        save_path = Path(args.save)
        torch.save({'model': model.state_dict(), 'cfg': cfg, 'args': vars(args)},
                   save_path)
        print(f"\nModel saved to {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train nanoGPT on symbolic algebra')
    # data
    p.add_argument('--data_dir',        default=None,  help='load saved dataset')
    p.add_argument('--n_train',         type=int,   default=10_000)
    p.add_argument('--n_test',          type=int,   default=1_000)
    p.add_argument('--max_depth_train', type=int,   default=2)
    p.add_argument('--max_depth_test',  type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    # model
    p.add_argument('--max_len',         type=int,   default=256)
    p.add_argument('--n_layer',         type=int,   default=2)
    p.add_argument('--n_head',          type=int,   default=4)
    p.add_argument('--d_model',         type=int,   default=128)
    p.add_argument('--dropout',         type=float, default=0.1)
    # training
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--max_iters',       type=int,   default=5_000)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--warmup_iters',    type=int,   default=200)
    p.add_argument('--log_every',       type=int,   default=200)
    p.add_argument('--eval_every',      type=int,   default=1_000)
    p.add_argument('--save',            default=None, help='path to save model (.pt)')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
