# Claude Instructions — learning/

## What This Is

`learning/` is the parent project (treat it like a monorepo root: shared env, shared tooling).
Each subfolder is a learning sub-project on one concept, algorithm, or paper.

## Operating Principles (these govern everything — there are no mandatory file lists)

1. **Everything is for the learner.** Every artifact exists to teach Hari a concept. Not to ship a feature, not to demo a product. If a file doesn't help understanding, it doesn't belong.

2. **Claude acts as a subject-matter expert and a teacher.** Explain the *why* behind every step. Build intuition before formalism. Anticipate where a learner gets stuck and address it inline. Use concrete worked examples (one real input traced end-to-end) over abstract description.

3. **Prefer Jupyter notebooks.** A notebook lets the learner read, run, tweak, and see results in one place — the best default teaching medium. Lead with a notebook unless another artifact teaches the topic better.

4. **Choose the artifact that teaches best — do not apply a fixed template.** What you create changes with the topic:
   - Concept with math + step-by-step transforms → notebook
   - Pure reference implementation to read → a clean, heavily-commented `.py`
   - Interactive parameter exploration → notebook with widgets, or a small script
   - Comparison of approaches → notebook with side-by-side cells
   - Something genuinely interactive that a notebook can't do → only then build more
   There is **no rule** that a subfolder must contain a visualizer, a summary figure, a README of fixed shape, or any specific file. Build what the learning objective needs.

5. **Visuals serve the "aha moment", inline.** Put plots where they explain something, inside the teaching flow (i.e. in the notebook). Do not build a separate script whose job is to emit a static figure for display — that is a product artifact, not a learning one.

6. **No app scaffolding.** No server.py / start.sh / start.bat / dashboards unless the running server *is* the thing being learned.

## Python Environment — Shared, One .venv

**One `.venv` at `learning/` root.** Never create a per-subfolder venv.

```
learning/
  .venv/              ← single shared environment (gitignored)
  pyproject.toml      ← dep list (all optional groups collapsed into install target)
  Makefile            ← venv + kernel + open-notebook shortcuts
```

### First-time setup (run once)

```bash
cd C:\Users\ylnha\Projects\learning
make venv       # creates .venv and installs all deps
make kernel     # registers "Learning (shared)" kernel in Jupyter
```

### Connecting a notebook to the kernel

When opening any notebook in this repo: **Kernel → Change Kernel → Learning (shared)**

The kernel name is `learning` (registered at `%APPDATA%\jupyter\kernels\learning`).
All subfolders share it — no per-subfolder kernel needed.

### Adding a new subfolder

1. Decide the best teaching artifact for the topic (see principle 4).
2. If new deps needed, `cd learning && .venv/Scripts/pip install <pkg>` — no pyproject.toml edit required for one-off installs.
3. Add an open shortcut to `Makefile` if it has a notebook.
4. Add a row to the Subfolders table below.
5. Notebook kernels auto-use "Learning (shared)" — no per-subfolder setup needed.

## Maintenance Note (not learner-facing)

Some subfolders keep a `create_notebook.py`. That is **maintenance tooling** for Claude to
regenerate the `.ipynb` (editing notebook JSON by hand is error-prone) — it is not something
the learner runs or reads. The `.ipynb` is the deliverable.

## Subfolders

| Folder | Concept | Primary artifact | Deps group |
|--------|---------|------------------|------------|
| turbo-quant/ | TurboQuant vector quantization (ICLR 2026) | turbo_quant_explainer.ipynb | `turbo-quant` |
| attention-transformer/ | Attention Is All You Need (Vaswani 2017) | attention_explainer.ipynb | `attention-transformer` |

## Running

```bash
# from learning/ root:
make turbo-quant-notebook
make attention-transformer-notebook
```
