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

## Writing to the user

- Code, commands and paths in single backticks, inline: never fenced
  blocks with triple backticks.
