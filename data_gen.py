"""
Single-digit symbolic algebra: grammar, negation-distribution rewriter,
dataset splits, and character-level tokenizer.

Grammar:   Expr := digit | ('neg',Expr) | ('add',Expr,Expr) | ('sub',Expr,Expr)
           digit := 0..9

Training format:  "-(3+4)->-3-4"
  The model sees the full string; loss is computed on the target side (after '->').
  '>' appears only in the '->' separator, never in expressions or targets.

Rewrite task: distribute all negations to the leaves.
  -(a+b)  ->  -a-b
  -(a-b)  ->  -a+b
  -(-(a)) ->  a
  Applied recursively throughout the expression.

Output is always a flat signed sequence of single digits (e.g. -3+4-5),
so no arithmetic is ever evaluated and all digits remain 0-9.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Union

Expr = Union[int, tuple]

DIGITS = list(range(10))
ALL_OPS = ('neg', 'add', 'sub')

# '>' appears only in the '->' separator; never in expressions or targets.
VOCAB = sorted('0123456789()+->')
VOCAB_SIZE = len(VOCAB)
_C2I = {c: i for i, c in enumerate(VOCAB)}
_I2C = {i: c for i, c in enumerate(VOCAB)}


# ── Grammar ───────────────────────────────────────────────────────────────────

def sample_expr(max_depth: int, allowed_ops=ALL_OPS, rng=None) -> Expr:
    """Sample a random expression. allowed_ops controls curriculum stage."""
    if rng is None:
        rng = random
    if max_depth == 0:
        return rng.choice(DIGITS)
    choice = rng.choice(['digit'] + [op for op in ALL_OPS if op in allowed_ops])
    if choice == 'digit':
        return rng.choice(DIGITS)
    if choice == 'neg':
        return ('neg', sample_expr(max_depth - 1, allowed_ops, rng))
    if choice == 'add':
        return ('add', sample_expr(max_depth - 1, allowed_ops, rng),
                       sample_expr(max_depth - 1, allowed_ops, rng))
    # sub
    return ('sub', sample_expr(max_depth - 1, allowed_ops, rng),
                   sample_expr(max_depth - 1, allowed_ops, rng))


def depth(expr: Expr) -> int:
    if isinstance(expr, int):
        return 0
    return 1 + max(depth(a) for a in expr[1:])


def has_neg(expr: Expr) -> bool:
    if isinstance(expr, int):
        return False
    if expr[0] == 'neg':
        return True
    return any(has_neg(a) for a in expr[1:])


# ── Serializer ────────────────────────────────────────────────────────────────

def expr_to_str(expr: Expr) -> str:
    """Fully-parenthesized string for binary ops; parens around neg arg if compound."""
    if isinstance(expr, int):
        return str(expr)
    op = expr[0]
    if op == 'neg':
        s = expr_to_str(expr[1])
        # Binary ops already supply their own outer parens; only add parens when
        # the inner is also a neg, to distinguish -(- from --.
        needs_parens = not isinstance(expr[1], int) and expr[1][0] == 'neg'
        return f'-({s})' if needs_parens else f'-{s}'
    if op == 'add':
        return f'({expr_to_str(expr[1])}+{expr_to_str(expr[2])})'
    # sub
    return f'({expr_to_str(expr[1])}-{expr_to_str(expr[2])})'


# ── Rewriter: negation distribution ──────────────────────────────────────────

def _terms(expr: Expr, sign: int = 1) -> list[tuple[int, int]]:
    """
    Walk the expression tree, collecting (sign, digit) pairs.
    sign tracks the accumulated sign from enclosing negations:
      neg flips it, add preserves it, sub flips it for the right child.
    """
    if isinstance(expr, int):
        return [(sign, expr)]
    op = expr[0]
    if op == 'add':
        return _terms(expr[1], sign) + _terms(expr[2], sign)
    if op == 'sub':
        return _terms(expr[1], sign) + _terms(expr[2], -sign)
    # neg
    return _terms(expr[1], -sign)


def rewrite(expr: Expr) -> str:
    """Distribute negation; return a flat signed-sum string."""
    terms = _terms(expr)
    out = []
    for i, (sign, digit) in enumerate(terms):
        if i == 0:
            out.append(('' if sign > 0 else '-') + str(digit))
        else:
            out.append(('+' if sign > 0 else '-') + str(digit))
    return ''.join(out)


# ── Pairs and tokenizer ───────────────────────────────────────────────────────

def make_pair(expr: Expr) -> str:
    """'input->target' training string."""
    return expr_to_str(expr) + '->' + rewrite(expr)


def encode(s: str) -> list[int]:
    return [_C2I[c] for c in s]


def decode(ids: list[int]) -> str:
    return ''.join(_I2C[i] for i in ids)


# ── Dataset ───────────────────────────────────────────────────────────────────

def _sample_split(n: int, max_depth: int, allowed_ops, require_neg: bool,
                  rng: random.Random) -> list[str]:
    samples, attempts = [], 0
    while len(samples) < n:
        attempts += 1
        if attempts > n * 50:
            raise RuntimeError(
                f"Could not generate {n} samples with require_neg={require_neg}; "
                "try a higher max_depth or set require_neg=False"
            )
        expr = sample_expr(max_depth, allowed_ops, rng)
        if require_neg and not has_neg(expr):
            continue
        samples.append(make_pair(expr))
    return samples


def make_dataset(
    n_train: int = 10_000,
    n_test: int = 1_000,
    max_depth_train: int = 2,
    max_depth_test: int = 4,
    allowed_ops: tuple = ALL_OPS,
    require_neg: bool = True,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Generate train and test splits.

    Splits returned:
      train        — depth <= max_depth_train, for training
      test_same    — same depth as train, fresh samples (catches memorization)
      test_deeper  — depth <= max_depth_test  (depth generalization probe)
    """
    rng = random.Random(seed)
    return {
        'train':       _sample_split(n_train, max_depth_train, allowed_ops, require_neg, rng),
        'test_same':   _sample_split(n_test,  max_depth_train, allowed_ops, require_neg, rng),
        'test_deeper': _sample_split(n_test,  max_depth_test,  allowed_ops, require_neg, rng),
    }


# ── Persistence ──────────────────────────────────────────────────────────────

def save_dataset(
    data: dict[str, list[str]],
    config: dict,
    base_dir: str = 'data',
) -> Path:
    """
    Write dataset splits to a timestamped directory under base_dir.

    Directory layout:
      data/YYYYMMDD_HHMMSS/
        config.json      — all generation parameters
        train.txt        — one pair per line
        test_same.txt
        test_deeper.txt

    Returns the path to the created directory.
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(base_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    for split, pairs in data.items():
        (out_dir / f'{split}.txt').write_text('\n'.join(pairs) + '\n')

    return out_dir


def load_dataset(path: str | Path) -> tuple[dict[str, list[str]], dict]:
    """
    Load a dataset saved by save_dataset.

    Returns (data, config) where data is a dict of split-name -> list of pair strings.
    """
    p = Path(path)
    config = json.loads((p / 'config.json').read_text())
    data = {}
    for txt in p.glob('*.txt'):
        lines = txt.read_text().splitlines()
        data[txt.stem] = [l for l in lines if l]
    return data, config


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import re

    cfg = dict(
        n_train=10_000, n_test=1_000,
        max_depth_train=2, max_depth_test=4,
        allowed_ops=list(ALL_OPS),
        require_neg=True, seed=42,
    )
    data = make_dataset(**cfg)

    print(f"Vocab ({VOCAB_SIZE}): {''.join(VOCAB)}")
    print(f"Train: {len(data['train'])}  "
          f"Test-same: {len(data['test_same'])}  "
          f"Test-deeper: {len(data['test_deeper'])}")

    print('\nTrain examples:')
    for s in data['train'][:15]:
        print(f'  {s}')
    print('\nDeeper test examples:')
    for s in data['test_deeper'][:8]:
        print(f'  {s}')

    # Verify all chars are in vocab
    all_strings = data['train'] + data['test_same'] + data['test_deeper']
    unknown = {c for s in all_strings for c in s} - set(VOCAB)
    assert not unknown, f"Characters not in vocab: {unknown}"
    print('\nVocab check passed.')

    # Verify no multi-digit numbers in targets
    for s in all_strings:
        target = s.split('->')[1]
        assert all(len(r) == 1 for r in re.findall(r'\d+', target)), \
            f"Multi-digit in target: {s}"
    print('Single-digit check passed.')

    # Save and round-trip
    out_dir = save_dataset(data, cfg)
    print(f'\nDataset saved to: {out_dir}')

    data2, cfg2 = load_dataset(out_dir)
    assert cfg2 == cfg
    assert data2['train'] == data['train']
    print('Save/load round-trip passed.')
