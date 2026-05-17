## RTK Command Policy

Before using `rtk`, first assume it may not be installed.

When suggesting shell commands:

- Prefer `rtk <command>` only if `rtk` is available in the environment.
- If `rtk` is not available, use the raw command directly.
- Do not block progress because `rtk` is missing.
- If a command fails with `CommandNotFoundException: rtk`, retry the same command without `rtk`.

Examples:

Preferred when RTK is available:

```bash
rtk git status
rtk git diff
rtk git log

# Files & Search
rtk ls <path>
rtk read <file>
rtk grep <pattern>
rtk find <pattern>
rtk diff <file>

# Tests
rtk pytest tests/
rtk cargo test
rtk test <cmd>

# Build & Lint
rtk tsc
rtk lint
rtk cargo build
rtk prettier --check
rtk mypy
rtk ruff check

# Analysis
rtk err <cmd>
rtk log <file>
rtk json <file>
rtk summary <cmd>
rtk deps
rtk env

# GitHub
rtk gh pr view <n>
rtk gh run list
rtk gh issue list

# Infrastructure
rtk docker ps
rtk kubectl get
rtk docker logs <container>

# Package Managers
rtk pip list
rtk pnpm install
rtk npm run <script>

rtk git status
rtk ls
rtk find srs
rtk find requirements
rtk find backlog
rtk find priority
rtk find geometry
rtk find frontend