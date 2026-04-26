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
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_gen import (
    ALL_OPS, VOCAB_SIZE, _C2I, _I2C,
    make_dataset, save_dataset, load_dataset, encode,
)

# ── Logger ───────────────────────────────────────────────────────────────────

class Logger:
    """Writes to stdout and a file simultaneously."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, 'w')

    def __call__(self, msg: str = ''):
        print(msg, flush=True)
        self._f.write(msg + '\n')
        self._f.flush()

    def close(self):
        self._f.close()


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
def greedy_decode(model, prompt_ids: list[int], device, max_new: int = 64) -> list[int]:
    """Extend prompt greedily for at most max_new tokens."""
    model.eval()
    ids = list(prompt_ids)
    limit = min(len(ids) + max_new, model.cfg.max_len)
    while len(ids) < limit:
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
    torch.manual_seed(args.seed)

    # MPS (torch 1.13) is unreliable for this workload; prefer CUDA then CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"{args.n_layer}L{args.n_head}H_d{args.d_model}"
    log_path = Path(args.exp_dir) / f"{ts}_{tag}.txt"
    log = Logger(log_path)

    log(f"Run    : {ts}  {tag}")
    log(f"Device : {device}")

    # ── Data helpers ──
    max_len = args.max_len

    def _make_loaders(depth, label=''):
        """Generate + save a dataset at max_depth_train=depth; return (train, same, data)."""
        cfg = dict(
            n_train=args.n_train, n_test=args.n_test,
            max_depth_train=depth,
            max_depth_test=args.max_depth_test,
            allowed_ops=list(ALL_OPS),
            require_neg=True, seed=args.seed,
        )
        d = make_dataset(**cfg)
        saved = save_dataset(d, cfg, base_dir='data')
        log(f"Data{label}: depth_train={depth}  train={len(d['train'])}  "
            f"test_same={len(d['test_same'])}  saved={saved}")
        tr = DataLoader(AlgebraDataset(d['train'],     max_len),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
        sm = DataLoader(AlgebraDataset(d['test_same'], max_len),
                        batch_size=args.batch_size, shuffle=False)
        return tr, sm, d

    # ── Data ──
    if args.curriculum:
        stages = [int(d) for d in args.stages.split(',')]
        log(f"Curriculum stages: {stages}  threshold={args.stage_threshold}")

        # Fixed test_deeper: generate once from the deepest stage's distribution.
        deep_cfg = dict(n_train=10, n_test=args.n_test,
                        max_depth_train=stages[-1], max_depth_test=args.max_depth_test,
                        allowed_ops=list(ALL_OPS), require_neg=True, seed=args.seed + 1)
        deep_data   = make_dataset(**deep_cfg)
        deep_loader = DataLoader(AlgebraDataset(deep_data['test_deeper'], max_len),
                                 batch_size=args.batch_size, shuffle=False)
        log(f"Data   : fixed test_deeper={len(deep_data['test_deeper'])}  "
            f"depth_test={args.max_depth_test}")

        stage_idx    = 0
        train_loader, same_loader, data = _make_loaders(stages[0], label=f'[stage 1/{len(stages)}]')

    elif args.data_dir:
        data, gen_cfg = load_dataset(args.data_dir)
        log(f"Data   : loaded from {args.data_dir}  train={len(data['train'])}  "
            f"test_same={len(data['test_same'])}  test_deeper={len(data['test_deeper'])}")
        train_loader = DataLoader(AlgebraDataset(data['train'],     max_len),
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        same_loader  = DataLoader(AlgebraDataset(data['test_same'], max_len),
                                  batch_size=args.batch_size, shuffle=False)
        deep_loader  = DataLoader(AlgebraDataset(data['test_deeper'], max_len),
                                  batch_size=args.batch_size, shuffle=False)

    else:
        train_loader, same_loader, data = _make_loaders(args.max_depth_train)
        deep_loader = DataLoader(AlgebraDataset(data['test_deeper'], max_len),
                                 batch_size=args.batch_size, shuffle=False)

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
    log(f"Params : {n_params:,}  ({cfg.n_layer}L {cfg.n_head}H d{cfg.d_model}  "
        f"max_len={max_len}  bs={args.batch_size}  iters={args.max_iters}  lr={args.lr})")

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
    if args.curriculum:
        log(f"\n=== Stage 1/{len(stages)}: max_depth={stages[0]} ===")
    log('')
    model.train()
    train_iter     = iter(train_loader)
    t0             = time.time()
    deeper_history = []
    same_history   = []   # per-stage test_same history (curriculum only)
    stopped_early  = False

    for step in range(args.max_iters + 1):
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
            log(f"step {step:5d}  loss {loss.item():.4f}  "
                f"lr {lr_now:.2e}  {time.time()-t0:.0f}s")

        if step > 0 and step % args.eval_every == 0:
            acc_s = evaluate(model, same_loader, device)
            acc_d = evaluate(model, deep_loader, device)
            log(f"         >> test_same={acc_s:.3f}  test_deeper={acc_d:.3f}")
            model.train()

            deeper_history.append(acc_d)
            same_history.append(acc_s)

            if args.curriculum:
                # Advance stage if test_same hit threshold, or test_same has stalled.
                same_hit  = acc_s >= args.stage_threshold
                same_stalled = (args.patience > 0
                                and len(same_history) >= args.patience
                                and max(same_history[-args.patience:]) <=
                                    (max(same_history[:-args.patience]) if len(same_history) > args.patience else -1.0)
                                    + args.min_delta)
                if same_hit or same_stalled:
                    reason = "threshold reached" if same_hit else "stalled"
                    stage_idx += 1
                    if stage_idx < len(stages):
                        log(f"\n=== Stage {stage_idx+1}/{len(stages)}: "
                            f"max_depth={stages[stage_idx]}  ({reason} at step {step}) ===")
                        train_loader, same_loader, data = _make_loaders(
                            stages[stage_idx], label=f'[stage {stage_idx+1}/{len(stages)}]')
                        train_iter   = iter(train_loader)
                        same_history = []
                        deeper_history = []
                    else:
                        log(f"\n=== All stages complete ({reason} at step {step}) ===")
                        break
            else:
                # Standard early stopping on test_deeper.
                if args.patience > 0 and len(deeper_history) >= args.patience:
                    window_best = max(deeper_history[-args.patience:])
                    prior_best  = max(deeper_history[:-args.patience]) if len(deeper_history) > args.patience else -1.0
                    if window_best <= prior_best + args.min_delta:
                        log(f"  [early stop] test_deeper no improvement > {args.min_delta} "
                            f"in last {args.patience} evals — stopping at step {step}")
                        stopped_early = True
                        break

    # ── Final eval + samples ──
    acc_s  = evaluate(model, same_loader, device)
    acc_d  = evaluate(model, deep_loader, device)
    status = "early-stop" if stopped_early else "full-run"
    log(f"\nFinal  test_same={acc_s:.3f}  test_deeper={acc_d:.3f}  ({status})")

    log("\nSample predictions (greedy):")
    for raw in data['test_same'][:8]:
        src, tgt = raw.split('->')
        prompt   = encode(src + '->')
        out_ids  = greedy_decode(model, prompt, device, max_new=len(tgt) * 2 + 4)
        out_str  = ''.join(_I2C.get(i, '?') for i in out_ids[len(prompt):])
        mark     = 'OK' if out_str == tgt else 'XX'
        log(f"  {mark}  {src}->  pred={out_str!r}  gold={tgt!r}")

    log(f"\nLog    : {log_path}")

    # ── Save ──
    if args.save:
        save_path = Path(args.save)
        torch.save({'model': model.state_dict(), 'cfg': cfg, 'args': vars(args)},
                   save_path)
        log(f"Model  : {save_path}")

    log.close()


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
    p.add_argument('--max_len',         type=int,   default=96)
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
    p.add_argument('--eval_every',      type=int,   default=200)
    p.add_argument('--save',            default=None, help='path to save model (.pt)')
    p.add_argument('--exp_dir',         default='experiments', help='directory for run logs')
    p.add_argument('--patience',        type=int,   default=3,
                   help='early-stop window in evals (0 = disabled)')
    p.add_argument('--min_delta',       type=float, default=0.01,
                   help='min improvement required within patience window')
    # curriculum
    p.add_argument('--curriculum',      action='store_true',
                   help='progressive training through depth stages')
    p.add_argument('--stages',          default='1,2,3,4',
                   help='comma-separated max_depth_train values for each stage')
    p.add_argument('--stage_threshold', type=float, default=0.99,
                   help='test_same accuracy needed to advance to next stage')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
