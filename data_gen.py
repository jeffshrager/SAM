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

Design note: keeping digits single-digit ensures there is a 1-to-1 mapping
between characters and tokens — no multi-char numbers to tokenize differently.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Union

# An Expr is either a bare integer (a leaf digit) or a tuple whose first element
# is an operator name and whose remaining elements are sub-expressions.
# Examples:  5                      — leaf digit
#            ('neg', 3)             — negate 3
#            ('add', 2, ('neg', 5)) — 2 + (-(5))
Expr = Union[int, tuple]

DIGITS = list(range(10))           # valid leaf values: 0-9
ALL_OPS = ('neg', 'add', 'sub')    # complete operator set

# The full character vocabulary.  '>' only ever appears as part of the '->'
# separator, so every other character belongs to either an expression or target.
# Using sorted() here makes the mapping deterministic across Python runs.
VOCAB = sorted('0123456789()+->')
VOCAB_SIZE = len(VOCAB)
_C2I = {c: i for i, c in enumerate(VOCAB)}   # char -> integer id
_I2C = {i: c for i, c in enumerate(VOCAB)}   # integer id -> char


# ── Grammar ───────────────────────────────────────────────────────────────────
# Functions for building random expression trees.

def sample_expr(max_depth: int, allowed_ops=ALL_OPS, rng=None) -> Expr:
    """
    Recursively sample a random expression tree up to max_depth levels deep.

    allowed_ops lets callers restrict which operators can appear — this is used
    for curriculum learning where we start with simple expressions and gradually
    add operators.

    Base case (max_depth == 0): always return a leaf digit so we never build
    an infinitely tall tree.
    """
    if rng is None:
        rng = random

    # When at maximum depth, force a leaf to stop recursion.
    if max_depth == 0:
        return rng.choice(DIGITS)

    # Choose uniformly among 'digit' and whichever ops are currently allowed.
    # Including 'digit' in the choice list means even at depth > 0 we sometimes
    # create leaves, giving a realistic mix of shallow and tall sub-trees.
    choice = rng.choice(['digit'] + [op for op in ALL_OPS if op in allowed_ops])

    if choice == 'digit':
        return rng.choice(DIGITS)
    if choice == 'neg':
        # Unary: one child
        return ('neg', sample_expr(max_depth - 1, allowed_ops, rng))
    if choice == 'add':
        # Binary: two independent children, each decreasing depth by one
        return ('add', sample_expr(max_depth - 1, allowed_ops, rng),
                       sample_expr(max_depth - 1, allowed_ops, rng))
    # sub — identical structure to add
    return ('sub', sample_expr(max_depth - 1, allowed_ops, rng),
                   sample_expr(max_depth - 1, allowed_ops, rng))


def depth(expr: Expr) -> int:
    """Return the height of the expression tree (0 for a leaf digit)."""
    if isinstance(expr, int):
        return 0
    # For compound nodes, depth is 1 + the tallest child subtree.
    return 1 + max(depth(a) for a in expr[1:])


def has_neg(expr: Expr) -> bool:
    """Return True if any 'neg' node appears anywhere in the tree."""
    if isinstance(expr, int):
        return False
    if expr[0] == 'neg':
        return True
    # Recurse into all children (expr[1:] skips the operator name at index 0).
    return any(has_neg(a) for a in expr[1:])


# ── Serializer ────────────────────────────────────────────────────────────────
# Converts an expression tree back to the human-readable string that the model
# will see on its input side (left of '->').

def expr_to_str(expr: Expr) -> str:
    """
    Serialize an expression tree to a fully-parenthesized string.

    Binary ops wrap their arguments in parentheses, e.g. (3+4).
    Unary neg is written as a leading '-', e.g. -(3+4) or -3.

    The only tricky case is nested neg: -(-3) must be written -(- 3) not --3.
    The parentheses are added only when the neg's argument is itself a neg,
    because a binary-op argument already supplies its own outer parens.
    """
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
# This is the algebraic transformation the model must learn:
#   push all negations inward until every '-' is directly attached to a digit leaf.

def _terms(expr: Expr, sign: int = 1) -> list[tuple[int, int]]:
    """
    Walk the expression tree, collecting (effective_sign, digit) pairs.

    'sign' is the cumulative sign accumulated by traversing enclosing negations.
    It is always +1 or -1.  The rules are:
      - add node:  both children inherit the current sign unchanged
      - sub node:  left child keeps current sign; right child gets sign flipped
                   (because a-b == a+(-b))
      - neg node:  the single child gets sign flipped
      - leaf digit: emit the current sign paired with the digit value

    By the time we reach a leaf, 'sign' encodes the net sign of all negations
    and subtractions that enclose this digit.
    """
    if isinstance(expr, int):
        return [(sign, expr)]
    op = expr[0]
    if op == 'add':
        # Both branches of an addition share the same outer sign.
        return _terms(expr[1], sign) + _terms(expr[2], sign)
    if op == 'sub':
        # Right-hand side of subtraction is implicitly negated.
        return _terms(expr[1], sign) + _terms(expr[2], -sign)
    # neg: flip the sign for the single child
    return _terms(expr[1], -sign)


def rewrite(expr: Expr) -> str:
    """
    Distribute all negations to the leaves and return a flat signed-sum string.

    Example: -(3+4)  ->  _terms gives [(-1,3), (-1,4)]  ->  "-3-4"
             -(3-4)  ->  _terms gives [(-1,3), (+1,4)]  ->  "-3+4"

    The first term is special: a positive first term has no leading '+', while
    all subsequent terms always print an explicit sign character.
    """
    terms = _terms(expr)
    out = []
    for i, (sign, digit) in enumerate(terms):
        if i == 0:
            # First term: omit '+' for positive, include '-' for negative.
            out.append(('' if sign > 0 else '-') + str(digit))
        else:
            # All subsequent terms: always include an explicit sign character.
            out.append(('+' if sign > 0 else '-') + str(digit))
    return ''.join(out)


# ── Pairs and tokenizer ───────────────────────────────────────────────────────
# A "pair" is the complete training string: "input->target".

def make_pair(expr: Expr) -> str:
    """Build a single 'input->target' training string from an expression tree."""
    return expr_to_str(expr) + '->' + rewrite(expr)


def encode(s: str) -> list[int]:
    """Convert a string to a list of integer token ids using the vocab mapping."""
    return [_C2I[c] for c in s]


def decode(ids: list[int]) -> str:
    """Convert a list of integer token ids back to a string."""
    return ''.join(_I2C[i] for i in ids)


# ── Dataset ───────────────────────────────────────────────────────────────────
# Utilities to generate and organise train/test splits.

def _sample_split(n: int, max_depth: int, allowed_ops, require_neg: bool,
                  rng: random.Random, min_depth: int = 0) -> list[str]:
    """
    Sample exactly n expression pairs satisfying the given constraints.

    require_neg=True discards expressions that contain no negation at all,
    because those would be trivial identity rewrites and would skew the
    difficulty distribution of the dataset.

    min_depth (default 0 = no lower bound) rejects expressions whose tree
    height is below the threshold.  Set min_depth=3 for the deeper test split
    so every example is genuinely challenging and not just a repeat of the
    shallow training distribution.

    A hard upper bound on attempts prevents infinite loops when the constraints
    are very tight (e.g. require_neg=True at depth 0).
    """
    samples, attempts = [], 0
    while len(samples) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError(
                f"Could not generate {n} samples with require_neg={require_neg}, "
                f"min_depth={min_depth}; try relaxing constraints"
            )
        expr = sample_expr(max_depth, allowed_ops, rng)
        if require_neg and not has_neg(expr):
            continue
        if min_depth > 0 and depth(expr) < min_depth:
            continue
        samples.append(make_pair(expr))
    return samples


def make_dataset(
    n_train: int = 10_000,
    n_test: int = 1_000,
    max_depth_train: int = 2,
    max_depth_test: int = 4,
    min_depth_test: int = 0,
    allowed_ops: tuple = ALL_OPS,
    require_neg: bool = True,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Generate three dataset splits using a single seeded random generator so
    the splits are reproducible and non-overlapping in sequence (though
    individual expression trees could theoretically collide).

    Splits returned:
      train        — depth <= max_depth_train, for training
      test_same    — same depth as train, fresh samples (catches memorization)
      test_deeper  — min_depth_test <= depth <= max_depth_test
                     (depth generalization probe; default min=0 gives depth<=4,
                      set min_depth_test=3 to guarantee genuinely deep examples)

    Using a single rng that is advanced in order (train, then test_same, then
    test_deeper) means the three splits will differ even if their parameters
    are identical.
    """
    rng = random.Random(seed)
    return {
        'train':       _sample_split(n_train, max_depth_train, allowed_ops, require_neg, rng),
        'test_same':   _sample_split(n_test,  max_depth_train, allowed_ops, require_neg, rng),
        'test_deeper': _sample_split(n_test,  max_depth_test,  allowed_ops, require_neg, rng,
                                     min_depth=min_depth_test),
    }


# ── Persistence ──────────────────────────────────────────────────────────────
# Save and reload datasets so training runs can reuse the same data.

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

    The timestamp in the directory name makes successive runs easy to tell apart
    without overwriting earlier data.

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
    Empty lines (e.g. the trailing newline added by save_dataset) are filtered out.
    """
    p = Path(path)
    config = json.loads((p / 'config.json').read_text())
    data = {}
    for txt in p.glob('*.txt'):
        lines = txt.read_text().splitlines()
        data[txt.stem] = [l for l in lines if l]
    return data, config


# ── Smoke test ────────────────────────────────────────────────────────────────
# Run this file directly to verify the data generation pipeline end-to-end.

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
