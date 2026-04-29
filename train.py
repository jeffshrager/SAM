"""
Minimal decoder-only transformer for the nanoalgebra negation-distribution task.

The model is a small GPT-style (causal / autoregressive) transformer.
It reads a string like "-(3+4)->" and must predict the remaining characters
"-3-4" one token at a time.  Loss is only computed on those answer tokens —
the model is not penalised for how well it predicts the input portion.

Usage:
    python train.py                          # generate data, train with defaults
    python train.py --data_dir data/...      # use a saved dataset
    python train.py --n_layer 1 --d_model 64 --max_iters 3000
"""

import argparse
import json
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
    """Writes to stdout and a file simultaneously so nothing is lost."""
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
# The tokenizer in data_gen.py covers 15 characters (VOCAB_SIZE = 15).
# We add one extra token for padding so that sequences can be batched to
# a uniform length.  The model's embedding table therefore has VOCAB_SIZE+1
# rows, where the last row is reserved for PAD.

PAD_ID   = VOCAB_SIZE        # index of the padding pseudo-token
SEP_ID   = _C2I['>']        # index of '>'; marks the start of the answer side
EXT_VOCAB = VOCAB_SIZE + 1  # total vocabulary size seen by the model (incl. PAD)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AlgebraDataset(Dataset):
    """
    Converts raw 'input->target' pair strings into fixed-length tensors
    suitable for batched training.

    Each sample becomes three parallel tensors, all of length (max_len - 1):

      x    : token ids for positions 0 .. max_len-2  (the 'context' fed to the model)
      y    : token ids for positions 1 .. max_len-1  (the 'next token' the model predicts)
      mask : float 1.0 only at positions where y is an answer character

    The x/y pair is the standard language-modelling setup: y[i] is the token
    that should be predicted given x[0..i].  Padding tokens appended after the
    real sequence get mask=0, so they never contribute to the loss.

    Example for max_len=10 and pair "-(3)->-3":
      ids    = encode("-(3)->-3")       length 8
      L      = 8,  pad = 10-8 = 2
      sep    = ids.index(SEP_ID)        index of '>' = 4  (the '>' in '->')
      padded = [ids...] + [PAD, PAD]    length 10
      x      = padded[0:9]              length 9 = max_len-1
      y      = padded[1:10]             length 9
      mask:
        positions 0..3 (sep-1): y contains input chars  -> 0.0
        positions 4..6 (sep..L-2): y contains answer chars -> 1.0
        positions 7..8 (padding): y is PAD              -> 0.0
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
            pad = max_len - L          # number of PAD tokens to append

            # sep is the position of '>' inside ids.
            # In every pair string '->' appears exactly once, so sep is unique.
            sep = ids.index(SEP_ID)

            # Pad to exactly max_len tokens.
            padded = ids + [PAD_ID] * pad   # length max_len

            # x: feed tokens at positions 0..max_len-2 as input to the model.
            x = padded[:-1]                 # slice off last token -> length max_len-1

            # y: the token the model should predict at each position.
            # y[i] = padded[i+1], i.e. the next token after x[i].
            y = padded[1:]                  # slice off first token -> length max_len-1

            # Build the loss mask over the (max_len-1) positions of y.
            #
            # We want mask[i] = 1 exactly when y[i] is part of the answer.
            #
            # y[i] = padded[i+1], so y[i] is an answer char when padded[i+1]
            # is in the range ids[sep+1] .. ids[L-1], i.e. when i+1 is in
            # [sep+1 .. L-1], i.e. when i is in [sep .. L-2].
            #
            # Breaking the mask into three segments:
            #   [0 .. sep-1]  : length sep       — input chars in y -> 0.0
            #   [sep .. L-2]  : length L-1-sep   — answer chars in y -> 1.0
            #   [L-1 .. end]  : length pad        — PAD tokens in y -> 0.0
            #
            # Total length: sep + (L-1-sep) + pad = L-1 + pad = max_len-1 ✓
            #
            # EOS signal: we also unmask the *first* PAD position (y[L-1]) so
            # the model is trained to predict PAD_ID immediately after the last
            # answer character.  greedy_decode already stops on PAD_ID, so this
            # teaches the model when to stop generating.
            # We only do this when pad >= 1 (i.e. the sequence doesn't exactly
            # fill max_len, which is essentially always true in practice).
            if pad >= 1:
                # Answer chars + one EOS position unmasked, rest of PAD masked.
                # Length: sep + (L-sep) + (pad-1) = L + pad - 1 = max_len - 1 ✓
                mask = [0.0] * sep + [1.0] * (L - sep) + [0.0] * (pad - 1)
            else:
                # No room for EOS signal; keep original mask.
                mask = [0.0] * sep + [1.0] * (L - 1 - sep)

            self.samples.append((
                torch.tensor(x,    dtype=torch.long),
                torch.tensor(y,    dtype=torch.long),
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
    """Hyperparameters that define the transformer architecture."""
    max_len:   int   = 256    # maximum sequence length (context window)
    n_layer:   int   = 2      # number of transformer blocks stacked
    n_head:    int   = 4      # number of attention heads per block
    d_model:   int   = 128    # embedding dimension (must be divisible by n_head)
    dropout:   float = 0.1    # dropout probability applied in several places


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    'Causal' means each position can only attend to itself and earlier
    positions — future tokens are masked out.  This is required for
    autoregressive generation where the model predicts one token at a time.

    Key shapes throughout (B=batch, T=seq_len, C=d_model, H=n_head, D=d_head):
      input x:   (B, T, C)
      Q, K, V:   (B, H, T, D)
      att scores:(B, H, T, T)
      output:    (B, T, C)
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head   # size of each head's subspace

        # Single combined projection for Q, K, V — more efficient than three
        # separate Linear layers because it is one matrix multiply instead of three.
        # Output has 3*d_model channels which we split into Q, K, V below.
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)

        # Final linear that mixes the per-head outputs back into d_model dims.
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        # Lower-triangular matrix used as the causal mask.
        # mask[i,j] = 1 means position i is allowed to attend to position j.
        # mask[i,j] = 0 (upper triangle) means j > i, i.e. j is in the future.
        # Registered as a buffer so it is saved with the model but not a parameter.
        self.register_buffer(
            'mask', torch.tril(torch.ones(cfg.max_len, cfg.max_len))
        )

    def forward(self, x):
        # x: (B, T, C)  where C = d_model
        B, T, C = x.shape

        # ── Project to Q, K, V ────────────────────────────────────────────────
        # self.qkv(x) shape: (B, T, 3*C)
        # .split(C, dim=2) divides the last dimension into three equal chunks,
        # each of shape (B, T, C).
        q, k, v = self.qkv(x).split(C, dim=2)

        def reshape(t):
            """
            Re-arrange a (B, T, C) tensor into (B, n_head, T, d_head) so that
            each attention head can operate on its own d_head-dimensional slice.

            Step 1 — .view(B, T, n_head, d_head):
              Splits the last dimension C=n_head*d_head into two dimensions.
              The data isn't moved; we just reinterpret the memory layout.
              Shape: (B, T, n_head, d_head)

            Step 2 — .transpose(1, 2):
              Swaps the T (position 1) and n_head (position 2) axes.
              Shape: (B, n_head, T, d_head)

            After this, dimension 1 indexes the head and dimension 2 indexes
            the sequence position, which is what the attention formula expects.
            """
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # All three: (B, C) -> (B, n_head, T, d_head)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # ── Scaled dot-product attention ──────────────────────────────────────
        # Compute raw attention scores between every pair of positions.
        #
        # k.transpose(-2, -1): (B, n_head, d_head, T)  — flip last two dims of k
        # q @ k.T:             (B, n_head, T, T)
        #   att[b, h, i, j] = dot(q[b,h,i,:], k[b,h,j,:]) / sqrt(d_head)
        #   This is the affinity score of query position i attending to key position j.
        #
        # Dividing by sqrt(d_head) prevents dot products from growing large in
        # high dimensions, which would push softmax into a nearly-one-hot
        # distribution and cause vanishing gradients.
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)

        # ── Apply causal mask ─────────────────────────────────────────────────
        # self.mask is (max_len, max_len); slice to (T, T) for this batch's length.
        # Where mask == 0 (upper triangle, i.e. j > i), overwrite with -inf.
        # After softmax, -inf -> 0 probability, so position i cannot attend to j > i.
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Softmax over the last dim (all key positions) gives attention weights.
        att = self.drop(torch.softmax(att, dim=-1))
        # att: (B, n_head, T, T)  — each row sums to 1.0

        # ── Aggregate values ──────────────────────────────────────────────────
        # att @ v: (B, n_head, T, T) @ (B, n_head, T, d_head) = (B, n_head, T, d_head)
        #   For each position i, compute a weighted average of all value vectors,
        #   weighted by the attention distribution at row i.
        #
        # .transpose(1, 2): (B, T, n_head, d_head)
        #   Puts the sequence dimension back in position 1 (matching the input shape).
        #
        # .contiguous(): after a transpose the tensor's strides are non-contiguous
        #   in memory.  .contiguous() materialises a freshly laid-out copy, which
        #   is required before calling .view() (view needs contiguous memory).
        #
        # .view(B, T, C): (B, T, n_head*d_head) = (B, T, C)
        #   Merges the n_head and d_head dimensions back into a single C-dim vector.
        #   This concatenates each head's output side-by-side in the last dim.
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        # Final linear mixes across heads.  Shape stays (B, T, C).
        return self.proj(out)


class Block(nn.Module):
    """
    One transformer block: LayerNorm -> Attention -> residual,
                           LayerNorm -> MLP        -> residual.

    Pre-norm (normalise before the sub-layer) is used here rather than the
    original post-norm; it is more stable during training.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        # The MLP widens to 4*d_model before projecting back.  GELU is smoother
        # than ReLU and empirically works better in transformers.
        self.mlp  = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        # Residual connections: add the sub-layer output back to the input.
        # This lets gradients flow directly to early layers (skip the sub-layer).
        x = x + self.attn(self.ln1(x))   # attention sub-layer
        x = x + self.mlp(self.ln2(x))    # feed-forward sub-layer
        return x


class NanoGPT(nn.Module):
    """
    Decoder-only transformer (GPT architecture).

    Forward pass:
      token embedding (lookup table) + positional embedding -> dropout
      -> stack of transformer blocks
      -> final LayerNorm
      -> linear projection to vocabulary logits

    Loss (when targets and mask are provided):
      per-token cross-entropy, then masked to answer positions only.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg     = cfg
        # tok_emb maps each token id to a d_model-dimensional vector.
        # padding_idx=PAD_ID ensures the PAD embedding is always zero
        # and receives no gradient — padding should not contribute to the model.
        self.tok_emb = nn.Embedding(EXT_VOCAB, cfg.d_model, padding_idx=PAD_ID)

        # pos_emb maps each position index (0..max_len-1) to a d_model vector.
        # These are learned, not fixed sinusoidal embeddings.
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)   # final layer norm before the head

        # The 'head' projects from d_model to a score for each vocabulary token.
        self.head    = nn.Linear(cfg.d_model, EXT_VOCAB, bias=False)

        # Weight tying: share parameters between the input embedding table and
        # the output projection matrix.  This halves the number of parameters
        # in those two large matrices and is standard practice in language models.
        self.tok_emb.weight = self.head.weight

        self._init_weights()

    def _init_weights(self):
        """Initialise all Linear and Embedding weights with a small normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None, mask=None):
        """
        x:       (B, T)  — input token ids
        targets: (B, T)  — target token ids (x shifted left by 1, i.e. the next token)
        mask:    (B, T)  — float, 1.0 at answer positions, 0.0 elsewhere

        Returns (logits, loss) where loss is None when targets is not provided.
        """
        B, T = x.shape

        # Create a position index tensor [0, 1, ..., T-1] of shape (1, T).
        # The unsqueeze(0) adds a batch dimension so it broadcasts against (B, T).
        pos = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)

        # Sum token and position embeddings element-wise.
        # tok_emb(x):   (B, T) -> (B, T, d_model)
        # pos_emb(pos): (1, T) -> (1, T, d_model), broadcasts to (B, T, d_model)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))    # (B, T, d_model)

        # Pass through all transformer blocks sequentially.
        h = self.blocks(h)                                     # (B, T, d_model)

        # Project each position's d_model vector to a score over all EXT_VOCAB tokens.
        logits = self.head(self.ln_f(h))                       # (B, T, EXT_VOCAB)

        loss = None
        if targets is not None:
            # ── Compute masked cross-entropy loss ─────────────────────────────
            #
            # Step 1: flatten batch and time into one dimension so we can call
            #         F.cross_entropy with a single big batch of predictions.
            #   logits.view(-1, EXT_VOCAB): (B*T, EXT_VOCAB)
            #   targets.view(-1):           (B*T,)
            #
            # reduction='none': return one loss value per token instead of the
            #   mean, so we can apply our own answer-position mask.
            #
            # Note: we do NOT use ignore_index=PAD_ID here.  PyTorch's
            # ignore_index zeroes the loss wherever the TARGET is PAD_ID,
            # which would silently kill the EOS training signal (the one
            # unmasked PAD position we deliberately added to teach the model
            # to stop).  The mask handles all position filtering exclusively:
            # answer positions + EOS position have mask=1; everything else 0.
            #
            # Step 2: reshape back to (B, T) so mask can be applied per-sample.
            raw = nn.functional.cross_entropy(
                logits.view(-1, EXT_VOCAB),   # (B*T, EXT_VOCAB)
                targets.view(-1),             # (B*T,)
                reduction='none',
            ).view(B, T)                      # (B, T) — per-token loss values

            if mask is not None:
                # Zero out losses at non-answer positions by element-wise multiply.
                # mask: (B, T) with 1.0 at answer tokens, 0.0 elsewhere.
                # .sum() over all answer tokens, divide by count of answer tokens.
                # .clamp(min=1) guards against the degenerate case of an empty mask.
                loss = (raw * mask).sum() / mask.sum().clamp(min=1)
            else:
                loss = raw.mean()

        return logits, loss


# ── Greedy generation ─────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(model, prompt_ids: list[int], device, max_new: int = 64) -> list[int]:
    """
    Extend a token-id sequence greedily (always pick the highest-probability
    next token) until either max_new tokens are appended or PAD is predicted.

    PAD is used as an implicit end-of-sequence signal: the model learns to emit
    it once the answer is complete.
    """
    model.eval()
    ids = list(prompt_ids)
    limit = min(len(ids) + max_new, model.cfg.max_len)
    while len(ids) < limit:
        # Build a (1, T) tensor from the current id sequence and run the model.
        x      = torch.tensor([ids], device=device)
        logits, _ = model(x)                  # logits: (1, T, EXT_VOCAB)
        # logits[0, -1] is the distribution over the *next* token (after the last id).
        nxt    = logits[0, -1].argmax().item()
        if nxt == PAD_ID:
            break
        ids.append(nxt)
    return ids


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """
    Compute exact-match accuracy over the given DataLoader.

    A sample counts as correct only if every single answer token is predicted
    correctly — partial credit is not given.  Input tokens (mask==0) are ignored.
    """
    model.eval()
    correct = total = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        preds     = logits.argmax(-1)   # (B, T) — greedy token id at each position

        # A position passes the correctness check if:
        #   (a) the prediction equals the target  (preds == y),  OR
        #   (b) it is a non-answer position       (mask == 0)
        #       because we don't care what the model predicts there.
        #
        # ok[b] = True iff condition (a) OR (b) holds at EVERY position in sample b.
        # .all(dim=1) reduces over the T dimension, giving a boolean vector (B,).
        ok = ((preds == y) | (mask == 0)).all(dim=1)

        correct += ok.sum().item()
        total   += x.size(0)
    return correct / total if total else 0.0


# ── Deep-sample collector ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_correct_deep(model, pairs: list[str], device, n: int) -> list[str]:
    """
    Greedily decode each string in `pairs` and return up to n examples where
    the model's prediction exactly matches the gold target.

    Returns a list of 'src->pred' strings (the '->pred' portion is the model
    output, which equals the gold when the sample is correct).

    We iterate in dataset order and stop as soon as we have n correct samples,
    so this is O(n/accuracy) greedy calls on average.  With a 1 000-sample
    test set and n=100 the worst case is 1 000 calls, which is fast.
    """
    model.eval()
    found = []
    for raw in pairs:
        if len(found) >= n:
            break
        src, tgt = raw.split('->')
        prompt  = encode(src + '->')
        out_ids = greedy_decode(model, prompt, device, max_new=len(tgt) * 2 + 4)
        # out_ids[len(prompt):] are the tokens generated after the prompt.
        # Truncate to len(tgt) before comparing so that any residual tokens
        # generated after the answer (e.g. during early training before the
        # model has fully learned the EOS signal) don't cause false negatives.
        pred_full = ''.join(_I2C.get(i, '?') for i in out_ids[len(prompt):])
        pred = pred_full[:len(tgt)]
        if pred == tgt:
            found.append(f'{src}->{pred}')
    return found


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)

    # MPS (torch 1.13) is unreliable for this workload; prefer CUDA then CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Timestamp of this run (wall-clock start time), used in intparams and
    # also as the auto-generated expdir name when --expdir is not supplied.
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"{args.n_layer}L{args.n_head}H_d{args.d_model}"

    # ── Establish the experiment directory ────────────────────────────────────
    # If the user pre-created a directory and passed --expdir, use it as-is;
    # they may have placed experiment-specific input files there already.
    # Otherwise create a fresh timestamped directory under experiments/.
    if args.expdir:
        expdir = Path(args.expdir)
        expdir_preexisting = expdir.exists()   # note whether we found it or made it
        expdir.mkdir(parents=True, exist_ok=True)
    else:
        expdir = Path('experiments') / ts
        expdir_preexisting = False
        expdir.mkdir(parents=True, exist_ok=True)

    # ── Write extparams immediately ───────────────────────────────────────────
    # extparams contains exactly what was on the command line (or argparse
    # defaults) — nothing the code computed or chose itself.  Writing this
    # first means the record survives even if the run crashes later.
    (expdir / 'extparams.json').write_text(json.dumps(vars(args), indent=2))

    # ── Logger goes into the expdir ───────────────────────────────────────────
    log_path = expdir / 'run.log'
    log = Logger(log_path)

    log(f"Run    : {ts}  {tag}")
    log(f"Expdir : {expdir}  (pre-existing={expdir_preexisting})")
    log(f"Device : {device}")

    # ── intparams: collected throughout the run, written at the end ───────────
    # intparams records everything the code decided or derived — the complement
    # of extparams.  'data_dirs' is a list because curriculum mode generates
    # multiple datasets (one per stage plus the fixed deeper test set).
    intparams = {
        'run_timestamp':       ts,
        'expdir':              str(expdir),
        'expdir_preexisting':  expdir_preexisting,
        'tag':                 tag,
        'device':              device,
        'data_dirs':           [],   # populated below as datasets are generated
    }

    # ── Data helpers ──────────────────────────────────────────────────────────
    max_len = args.max_len

    def _make_loaders(depth, label=''):
        """
        Generate a fresh dataset at the given max training depth, save it
        inside expdir/data/, record its path in intparams, and return
        (train_loader, same_loader, raw_data_dict).

        Saving inside expdir/data/ keeps all inputs and outputs for a run
        co-located.  save_dataset appends its own timestamp subdirectory, so
        successive calls in curriculum mode each get a unique path.
        """
        cfg = dict(
            n_train=args.n_train, n_test=args.n_test,
            max_depth_train=depth,
            max_depth_test=args.max_depth_test,
            allowed_ops=list(ALL_OPS),
            require_neg=True, seed=args.seed,
        )
        d = make_dataset(**cfg)
        # Save to expdir/data/<timestamp>/ rather than a top-level data/ folder
        # so the dataset is part of this experiment's directory tree.
        saved = save_dataset(d, cfg, base_dir=str(expdir / 'data'))
        # Record the path as an internally-derived parameter.
        intparams['data_dirs'].append(str(saved))
        log(f"Data{label}: depth_train={depth}  train={len(d['train'])}  "
            f"test_same={len(d['test_same'])}  saved={saved}")
        tr = DataLoader(AlgebraDataset(d['train'],     max_len),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
        sm = DataLoader(AlgebraDataset(d['test_same'], max_len),
                        batch_size=args.batch_size, shuffle=False)
        return tr, sm, d

    # ── Data ──────────────────────────────────────────────────────────────────
    # Three possible data modes:
    #   curriculum — progressively harder training stages
    #   --data_dir — load a previously saved dataset
    #   default    — generate a single dataset at max_depth_train

    if args.curriculum:
        stages = [int(d) for d in args.stages.split(',')]
        log(f"Curriculum stages: {stages}  threshold={args.stage_threshold}")

        # Build one fixed test_deeper loader using the hardest depth so that the
        # depth-generalisation metric is comparable across all curriculum stages.
        deep_cfg = dict(n_train=10, n_test=args.n_test,
                        max_depth_train=stages[-1], max_depth_test=args.max_depth_test,
                        allowed_ops=list(ALL_OPS), require_neg=True, seed=args.seed + 1)
        deep_data        = make_dataset(**deep_cfg)
        # Save this fixed deeper dataset into expdir like all other generated data.
        deep_saved       = save_dataset(deep_data, deep_cfg, base_dir=str(expdir / 'data'))
        intparams['data_dirs'].append(str(deep_saved))
        deep_loader = DataLoader(AlgebraDataset(deep_data['test_deeper'], max_len),
                                 batch_size=args.batch_size, shuffle=False)
        log(f"Data   : fixed test_deeper={len(deep_data['test_deeper'])}  "
            f"depth_test={args.max_depth_test}  saved={deep_saved}")

        stage_idx  = 0
        train_loader, same_loader, data = _make_loaders(stages[0], label=f'[stage 1/{len(stages)}]')
        # In curriculum mode the deeper test set is fixed for the whole run,
        # so deep_pairs points to the pre-generated deeper split throughout.
        deep_pairs = deep_data['test_deeper']

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
        deep_pairs = data['test_deeper']

    else:
        train_loader, same_loader, data = _make_loaders(args.max_depth_train)
        deep_loader = DataLoader(AlgebraDataset(data['test_deeper'], max_len),
                                 batch_size=args.batch_size, shuffle=False)
        deep_pairs = data['test_deeper']

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg   = GPTConfig(
        max_len=max_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
    )
    model = NanoGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    intparams['n_params'] = n_params   # total trainable parameter count
    log(f"Params : {n_params:,}  ({cfg.n_layer}L {cfg.n_head}H d{cfg.d_model}  "
        f"max_len={max_len}  bs={args.batch_size}  iters={args.max_iters}  lr={args.lr})")

    # ── Optimiser + learning-rate schedule ────────────────────────────────────
    # AdamW is Adam with decoupled weight decay, which prevents L2 decay from
    # interfering with the adaptive learning-rate estimates.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1
    )

    def lr_lambda(step):
        """
        Two-phase LR schedule:
          1. Linear warmup from 0 to 1.0 over the first warmup_iters steps.
             Avoids large updates with a poorly-initialised model.
          2. Cosine decay from 1.0 down to 0.1 over the remaining steps.
             t=0 -> factor 1.0 (peak), t=1 -> factor 0.1 (trough).
             The 0.1 floor keeps learning alive at the end of training.
        The returned value is multiplied by the base lr passed to AdamW.
        """
        if step < args.warmup_iters:
            return step / max(1, args.warmup_iters)
        # Fractional progress through the decay phase (0..1)
        t = (step - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Initialise progress-tracking output files ────────────────────────────
    # metrics.csv: one row per evaluation checkpoint, written throughout the run.
    # Created fresh here (overwriting any prior run's file in a pre-existing
    # expdir) so the header is always at line 1.
    metrics_path      = expdir / 'metrics.csv'
    deep_samples_path = expdir / 'deep_samples.txt'
    metrics_path.write_text('step,loss,acc_same,acc_deeper\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    if args.curriculum:
        log(f"\n=== Stage 1/{len(stages)}: max_depth={stages[0]} ===")
    log('')
    model.train()

    # We iterate over the DataLoader indefinitely (wrapping around when
    # exhausted) rather than tracking epochs, which simplifies the loop.
    train_iter     = iter(train_loader)
    t0             = time.time()
    deeper_history = []    # test_deeper accuracy recorded at each eval step
    same_history   = []    # test_same accuracy, reset at each curriculum stage
    stopped_early  = False
    last_loss      = float('nan')   # most-recent training loss; captured for CSV

    for step in range(args.max_iters + 1):
        # ── Fetch one batch ───────────────────────────────────────────────────
        try:
            x, y, mask = next(train_iter)
        except StopIteration:
            # DataLoader exhausted; restart from the beginning.
            train_iter = iter(train_loader)
            x, y, mask = next(train_iter)

        x, y, mask = x.to(device), y.to(device), mask.to(device)

        # ── Forward + backward ────────────────────────────────────────────────
        _, loss = model(x, y, mask)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm to 1.0 to prevent exploding gradients.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = loss.item()   # snapshot for the next eval checkpoint

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            log(f"step {step:5d}  loss {loss.item():.4f}  "
                f"lr {lr_now:.2e}  {time.time()-t0:.0f}s")

        # ── Periodic evaluation ───────────────────────────────────────────────
        if step > 0 and step % args.eval_every == 0:
            acc_s = evaluate(model, same_loader, device)
            acc_d = evaluate(model, deep_loader, device)
            log(f"         >> test_same={acc_s:.3f}  test_deeper={acc_d:.3f}")
            model.train()   # evaluate() calls model.eval(); restore training mode

            deeper_history.append(acc_d)
            same_history.append(acc_s)

            # ── Append one row to metrics.csv ─────────────────────────────────
            # last_loss is from the most recent training step before this eval.
            # Opening in append mode ('a') means each eval adds exactly one line
            # and the file is safe to `tail -f` from another terminal.
            with open(metrics_path, 'a') as mf:
                mf.write(f'{step},{last_loss:.6f},{acc_s:.6f},{acc_d:.6f}\n')

            # ── Append a block of correct deep examples to deep_samples.txt ──
            # Collect up to args.rndc examples where the model's greedy output
            # matches the gold target.  The header line gives enough context to
            # read the file with `tail -f` without needing to see earlier blocks.
            correct_examples = collect_correct_deep(
                model, deep_pairs, device, args.rndc
            )
            with open(deep_samples_path, 'a') as sf:
                sf.write(
                    f'=== step {step:5d}  loss={last_loss:.4f}  '
                    f'acc_same={acc_s:.3f}  acc_deeper={acc_d:.3f}  '
                    f'[{len(correct_examples)} correct shown of '
                    f'{len(deep_pairs)} sampled] ===\n'
                )
                for ex in correct_examples:
                    sf.write(f'  {ex}\n')
                sf.write('\n')   # blank line between blocks for readability

            if args.curriculum:
                # ── Curriculum stage-advance logic ────────────────────────────
                # Advance to the next (harder) stage if either:
                #   (a) test_same accuracy hit the threshold (model has mastered
                #       the current depth), or
                #   (b) test_same has stalled (no meaningful improvement in
                #       the last `patience` evaluation windows, so we push anyway
                #       rather than spin indefinitely).
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
                        # Replace loaders with new data at the next depth.
                        train_loader, same_loader, data = _make_loaders(
                            stages[stage_idx], label=f'[stage {stage_idx+1}/{len(stages)}]')
                        train_iter   = iter(train_loader)
                        # Reset histories so early-stopping counts from the new stage.
                        same_history = []
                        deeper_history = []
                    else:
                        log(f"\n=== All stages complete ({reason} at step {step}) ===")
                        break
            else:
                # ── Standard early stopping on test_deeper ────────────────────
                # Stop if the best accuracy in the most recent `patience` evals
                # is no better than the best accuracy before that window.
                if args.patience > 0 and len(deeper_history) >= args.patience:
                    window_best = max(deeper_history[-args.patience:])
                    prior_best  = max(deeper_history[:-args.patience]) if len(deeper_history) > args.patience else -1.0
                    if window_best <= prior_best + args.min_delta:
                        log(f"  [early stop] test_deeper no improvement > {args.min_delta} "
                            f"in last {args.patience} evals — stopping at step {step}")
                        stopped_early = True
                        break

    # ── Final evaluation + sample predictions ─────────────────────────────────
    acc_s  = evaluate(model, same_loader, device)
    acc_d  = evaluate(model, deep_loader, device)
    status = "early-stop" if stopped_early else "full-run"
    log(f"\nFinal  test_same={acc_s:.3f}  test_deeper={acc_d:.3f}  ({status})")

    log("\nSample predictions (greedy):")
    for raw in data['test_same'][:8]:
        src, tgt = raw.split('->')
        prompt   = encode(src + '->')          # token ids up to and including '>'
        out_ids  = greedy_decode(model, prompt, device, max_new=len(tgt) * 2 + 4)
        # Decode only the newly generated tokens (everything after the prompt).
        out_str  = ''.join(_I2C.get(i, '?') for i in out_ids[len(prompt):])
        mark     = 'OK' if out_str == tgt else 'XX'
        log(f"  {mark}  {src}->  pred={out_str!r}  gold={tgt!r}")

    log(f"\nLog    : {log_path}")

    # ── Save model checkpoint ─────────────────────────────────────────────────
    if args.save:
        save_path = Path(args.save)
        torch.save({'model': model.state_dict(), 'cfg': cfg, 'args': vars(args)},
                   save_path)
        log(f"Model  : {save_path}")

    # ── Write intparams ───────────────────────────────────────────────────────
    # Now that the run is complete, record the final accuracy values and status.
    # Written last so it reflects the true end state of the run.
    intparams['final_acc_same']   = acc_s
    intparams['final_acc_deeper'] = acc_d
    intparams['run_status']       = status
    (expdir / 'intparams.json').write_text(json.dumps(intparams, indent=2))
    log(f"Params : {expdir / 'intparams.json'}")

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
    p.add_argument('--rndc',            type=int,   default=100,
                   help='max correct deep examples to show per eval checkpoint')
    p.add_argument('--save',            default=None, help='path to save model (.pt)')
    p.add_argument('--expdir',          default=None,
                   help='experiment directory (created if absent; auto-named under '
                        'experiments/ if omitted)')
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
