---
name: SAM open experimental questions
description: Deferred experiments and open questions, updated through 2026-05-03 discussion
type: project
---

As of 2026-04-28, both curriculum and non-curriculum depth-4 runs complete.
Curriculum (1->2->3->4) beat flat depth-4 training on test_deeper (22% vs 13%),
but neither model mastered depth-4 in-distribution (best test_same ~35%).

**Why:** The model may be at capacity for depth-4 expressions, or early stopping
cut the non-curriculum run short, or both.

**How to apply:** Before starting new runs, check which of these is addressed first.

---

## Research scope clarification (2026-05-02 discussion)

Jeff's stated goals, to avoid scope drift:
- NOT modeling real human algebra development or cognitively plausible sequences.
- "Developmental sequence" means ordered curriculum (easy→hard), not a human-like progression.
- Training from scratch only — no pretrained NLP models (confound: can't attribute
  generalization to curriculum if prior knowledge is baked in).
- Roussel's NLP-pretrained LLM suggestion (Deepseek, Qwen) is ruled out.
- Roussel's formal-system pre-training variant (pre-train on broad algebraic instances,
  then fine-tune on the specific task) is NOT ruled out and may be worth a future run.
- Neutral symbolic tags for rule applications (e.g. `a`, `b` not `distribute`) are
  of interest: do they help the model form abstractions faster?

---

## Symbol / vocab change: `->` replaced with `_` (2026-05-03, DONE)

`->` was two tokens (`-` and `>`), making the separator ambiguous to read even if
technically unambiguous (since `>` never appeared elsewhere). Replaced with single
char `_`. VOCAB now has `_` instead of `>` at the same index 14. All saved datasets
generated before this change are incompatible — regenerate data on next run.

Note: TN-003 (equivalence experiment) originally proposed using `_` as the blank
placeholder. That design must change; a different char (e.g. `?`) is needed for the
blank now that `_` is the separator.

---

## Pending team discussion (do not implement unilaterally)

A. Equivalence understanding experiment (TN-003)
   Train on relational forms like `3+4+1 = 3+? -> 5` (use `?` not `_` for blank,
   since `_` is now the separator); test whether model replicates children's
   operational misreading of `=` (treats `=` as `->` rather than as a relation).
   Requires adding `=` and `?` to vocab.

B. Rule-label ablation (from Jeff, 2026-05-02)
   Attach neutral symbolic tags to training examples marking which rule is being
   applied (e.g. `a` for "distribute neg over parens", `b` for "double-neg cancel").
   Hypothesis: labeling an abstraction consistently accelerates or enables its
   formation, even when the tag itself carries no semantic content to the model.
   Requires new data format and careful control condition (same data without tags).

C. Arithmetic evaluation (TN-004)
   Add actual computation to training targets (e.g. `3+2 -> 5`).
   Opens the door to the number line hypothesis — does purely symbolic
   training produce magnitude-like internal representations?

D. Training data realism (TN-002)
   Make expression frequency distribution match human exposure: mostly
   2-3 terms, repetition, variation in numbers not syntactic depth.

---

## Deferred experiments in priority order

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
