# Contributing and coding-assistant handoff

Read [README.md](README.md), [SETUP.md](SETUP.md) and the
[coding-assistant SOP](SOP.md#10-coding-assistant-handoff) before changing code.
This application directory can be exported as a standalone repository. In that
repository all commands and source-map paths are relative to its root.

## Work on the intended path

1. Inspect `git status --short` and the relevant existing diff. Preserve other
   work. Confirm the target environment, issue and requested outcome.
2. Use `falloutassist_next/web.py` for routes/auth, `workflow.py` for the main
   investigation/publisher, `rovo_history.py` and `rovo_a2a.py` for A2A previews,
   and `knowledge_router.py` for downstream jobs. See the README's full source map.
3. Keep a Rovo-only change within that path unless the requested behavior needs
   a broader change. Keep knowledge proposals separate and retain evidence,
   prerequisites, negations, scope and uncertainty.
4. Preserve role checks, CSRF, encrypted secrets, grounded-action gates,
   publication markers, deduplication and audit-before-publish behavior.
5. Update setup/SOP documentation when configuration, cache validity, queue
   behavior, schema, API contracts or publishing behavior changes.

## Validation

Use Python from the project `.venv`. Honor the user's current constraints on
testing and external calls. For an appropriate code change, select existing
isolated tests; they mock models/transports and use temporary storage. For example:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_rovo_a2a.py' -v
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_rovo_history.py' -v
```

On Linux/macOS, use `.venv/bin/python` with the same arguments. Inspect selected
tests before executing them. Documentation-only changes need link/factual checks,
not application tests or a service restart. Do not contact live services, rebuild
production vectors, run a scan that can publish, or migrate a database as an
incidental check.

## Before a commit or push

- Review the staged diff and ensure no `.env`, credentials, OAuth codes, databases,
  uploaded customer files, vectors, logs or virtual environments are staged.
- Include necessary new source modules and documentation; copying only tracked
  files from an older working tree can omit required implementation files.
- State which checks actually ran and which behavior remains unverified. Passing
  syntax/import checks is not the same as passing tests or a production preview.
- Preserve remote history. Fetch and reconcile concurrent changes rather than
  force-pushing. Confirm any deployment target separately from repository upload.

No production secrets, account passwords, private datasets or provisioned external
services are distributed with the source. The repository has no bundled default
password, guaranteed latency SLA or proven production memory budget.
