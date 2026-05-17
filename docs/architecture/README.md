# Architecture Diagrams

PlantUML source files documenting the SDI Helper architecture.

## How to render

| Tool | Steps |
|---|---|
| **plantuml.com** | Paste file content at https://www.plantuml.com/plantuml/ |
| **VS Code** | Install "PlantUML" extension by jebbs, then preview .puml files (Alt+D) |
| **CLI** | `java -jar plantuml.jar *.puml` (outputs .png next to each .puml) |
| **Kroki** | `curl -s --data-binary @file.puml https://kroki.io/plantuml/png > file.png` |

## Files

| # | File | View |
|---|------|------|
| 01 | `01-context.puml`           | C4 L1 — system context (external boundaries) |
| 02 | `02-component.puml`         | C4 L3 — hexagonal architecture (layers + adapters) |
| 03 | `03-ports.puml`             | Class diagram — ports + adapters |
| 04 | `04-sequence-process.puml`  | Sequence — per-URL runtime flow |
| 05 | `05-activity-session.puml`  | Activity — scrape session lifecycle |
| 06 | `06-activity-process.puml`  | Activity — per-URL gate-by-gate flow |

## Style conventions

- Colors: domain=cream, application=blue, infrastructure=gray, interfaces=green
- Stereotypes: `<<port>>`, `<<adapter>>`, `<<system>>`, `<<external>>`
- Activity colors use post-fix syntax: `:Label; <<#27AE60>>` (NOT deprecated `#27AE60:Label;`)
- ASCII only in labels — avoid em dash, Unicode bullets, parens-inside-labels for renderer portability
