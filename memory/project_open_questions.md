---
name: SAM open experimental questions
description: Deferred experiments and open questions from session 2, to pick up next session
type: project
---

As of 2026-04-28, both curriculum and non-curriculum depth-4 runs complete.
Curriculum (1->2->3->4) beat flat depth-4 training on test_deeper (22% vs 13%),
but neither model mastered depth-4 in-distribution (best test_same ~35%).

**Why:** The model may be at capacity for depth-4 expressions, or early stopping
cut the non-curriculum run short, or both.

**How to apply:** Before starting new runs, check which of these is addressed first.

Recommended cleanup — do before next experiments (needs team sign-off on symbol choice):

0. Replace `->` separator with a single token (TN-005)
   `->` is two tokens (`-` and `>`), both with other meanings in expressions.
   Replace with one unambiguous character (e.g. `|` or `~`).
   Change: VOCAB and make_pair() in data_gen.py, SEP_ID in train.py, regenerate data.
   Low cost, high clarity. Do this before implementing TN-003 (= experiment).

Pending team discussion (do not implement unilaterally):

A. Equivalence understanding experiment (TN-003)
   Add `=` and `_` to vocab; train on relational forms like `3+4+1 = 3+_ -> 5`;
   test whether model replicates children's operational misreading of `=`.
   Low-hanging fruit, testable with existing infrastructure.

B. Arithmetic evaluation (TN-004)
   Add actual computation to training targets (e.g. `3+2 -> 5`).
   Opens the door to the number line hypothesis — does purely symbolic
   training produce magnitude-like internal representations?

C. Training data realism (TN-002)
   Make expression frequency distribution match human exposure: mostly
   2-3 terms, repetition, variation in numbers not syntactic depth.

Deferred experiments in priority order:

1. Non-curriculum depth<=4 with --patience 0
   Fair baseline: let it run 10000 steps uninterrupted.
   Command: python train.py --max_depth_train 4 --max_iters 10000 --patience 0

2. Network depth sweep with curriculum
   Sweep --n_layer in {2, 4, 6} with --curriculum --max_iters 10000.
   Motivation: 2-layer transformer may lack the compositional depth to track
   signs across 4 levels of nesting.  Session 1 sweep was in wrong regime
   (depth<=2 training, too easy) — this would be the first informative sweep.
   Note: ~30 min per run on CPU.

3. Longer curriculum training
   Try --max_iters 20000 or reduce LR floor in cosine schedule to let
   Stage 4 converge further before stalling.

4. Mechanistic probes (longer term)
   Attention visualization + activation patching.  Defer until a model
   reaches test_deeper > 50% so there is something meaningful to probe.
