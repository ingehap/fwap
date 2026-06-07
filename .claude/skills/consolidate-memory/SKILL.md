---
name: consolidate-memory
description: >-
  Tidy this repo's Claude Code memory. Use ONLY when the user explicitly asks
  to consolidate, clean up, prune, or reorganize CLAUDE.md / the project
  memory. Do not invoke automatically.
disable-model-invocation: true
---

# Consolidate project memory

Goal: keep memory high-signal so future sessions start smarter, without bloat.

1. Read `CLAUDE.md` and any `.claude/rules/*.md`.
2. Within `CLAUDE.md`:
   - Merge duplicate or overlapping notes.
   - Delete entries that are stale, contradicted, or now obvious from the code.
   - Promote any durable, generalizable lesson buried in prose into the
     `## Decisions & Lessons` section as a terse bullet.
   - Keep the file focused (aim < ~150 lines); move deep detail into a
     referenced doc and link it with `@path`.
3. Verify every command and file path still exists and is correct — e.g. the
   CI steps in `.github/workflows/ci.yml` and `scripts/check_public_api.py`.
4. Show a diff and a one-line rationale per change. Do not invent project
   facts; only consolidate what is already known or verifiable in the repo.
