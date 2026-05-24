## RTK Command Policy

The workspace root provides a local RTK-compatible wrapper at `..\rtk.cmd`.
Use that wrapper from this repo instead of the global `rtk.exe`.

When running or suggesting shell commands, Codex MUST:

- Prefer `..\rtk.cmd <command>` from this repo.
- Fall back to the raw command when `rtk` is not available.
- Never block progress because `rtk` is missing.
- If a command fails with `CommandNotFoundException: rtk` or `rtk: command not found`, retry the same command without `rtk`.
- If global `rtk.exe` returns a usage error for `--script` / `--out-file`, use `..\rtk.cmd` or the raw command.

Examples:

| Task | Prefer when RTK exists | Fallback |
|---|---|---|
| Git status | `..\rtk.cmd git status` | `git status` |
| Git diff | `..\rtk.cmd git diff` | `git diff` |
| List files | `..\rtk.cmd ls <path>` | `Get-ChildItem <path>` |
| Read file | `..\rtk.cmd read <file>` | `Get-Content <file>` |
| Search text | `..\rtk.cmd grep <pattern>` | `rg <pattern>` |
| Find files | `..\rtk.cmd find <pattern>` | `Get-ChildItem -Recurse -Filter <pattern>` |
| Run tests | `..\rtk.cmd pytest tests/` | `pytest tests/` |
| TypeScript check | `..\rtk.cmd tsc` | `npx tsc --noEmit` |
| Lint | `..\rtk.cmd lint` | project lint command |

---

## Git Branch & PR Workflow

All non-trivial changes (features, fixes, chores, cleanup) MUST go through a feature branch and pull request. Never push directly to `master`.

### Branch naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/<short-desc>` | `feat/7kp-body-end-holdout-eval` |
| Bug fix | `fix/<short-desc>` | `fix/dataset-yaml-val-path` |
| Chore / cleanup | `chore/<short-desc>` | `chore/remove-orphaned-dirs` |
| Training experiment | `exp/<short-desc>` | `exp/7kp-colab-gpu-150ep` |

### Workflow

```
# 1. Create branch before starting any work
git checkout -b chore/my-task

# 2. Make changes, commit
git add -A
git commit -m "chore: describe what changed and why"

# 3. Push and open PR
git push -u origin chore/my-task
# Then open PR targeting master
```

### Rules

- Run `poetry run pytest tests/ -q --ignore=tests/_pytest_cache --ignore=tests/_pytest_tmp` before pushing — all tests must pass.
- PR body must list: what changed, why, and test result line (e.g. `66 passed, 22 skipped`).
- Squash-merge into `master` when approved.
- Delete the feature branch after merge.
