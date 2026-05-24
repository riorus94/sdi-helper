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
