---
name: Deep commenting style required
description: All code written or edited in this project must be deeply and verbosely commented
type: feedback
---

All code in this project must be deeply commented. This is a hard requirement, not optional cleanup.

**Why:** The user explicitly requested this style and it reflects the pedagogical/research nature of the project where understanding the code is as important as running it.

**How to apply:**
- Every module gets a thorough docstring explaining its purpose, key design decisions, and the overall flow.
- Every class and function gets a docstring that explains what it does, its parameters, return values, and any non-obvious behaviour.
- Block comments before each logical section explain *what* that section is doing and *why*.
- Inline comments explain anything non-obvious — especially tensor reshaping/slicing operations, which must be annotated with the shape at each step (e.g. `# (B, T, C) -> (B, n_head, T, d_head)`).
- For tensor operations, show: the shape going in, what each operation does to the shape, and the shape coming out.
- For mask or index arithmetic, show a brief proof or worked example that the sizes add up correctly.
- Do not omit comments on the grounds that "the code is self-explanatory."
