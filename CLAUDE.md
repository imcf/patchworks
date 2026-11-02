# Working on patchworks

## Commits

- Format: `type(scope): gitmoji description`, always with a scope, e.g.
  `fix(viewer): 🐛 channels back on one layer's slider` or
  `feat(serve): ✨ a store for Neuroglancer`. Types and their gitmoji:
  feat ✨, fix 🐛, perf ⚡, refactor ♻️, docs 📝, test ✅, ci 👷, build 🏗️,
  chore 🔧. `cliff.toml` builds the release changelog from them.
- Before every commit, run Ruff on what changed:
  `ruff format <files>` then `ruff check <files>` (the version CI pins,
  see `.github/workflows/lint.yml`), and `mypy` for `src/`.

## Backticks

- Code, names, commands and paths in single backticks (`image.zarr`),
  never double ones (``image.zarr``): in docstrings, comments, commit
  messages, docs and messages to the user alike. The API docs
  (mkdocstrings) render docstrings as Markdown, so single backticks
  show as code there too.
