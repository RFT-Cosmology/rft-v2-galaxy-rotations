---
name: Bug report
about: Report a reproducibility issue or error
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run command '...'
2. See error '...'

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened. Include error messages or unexpected output.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- Installation method: [Conda / Docker / Manual]
- RUNME.sh output: [paste relevant output]

## Additional Context
- Did `./RUNME.sh` complete successfully?
- Did baseline lock pass? `python scripts/audit_baselines.py`
- Did hash lock pass? `python scripts/verify_final_numbers_hash.py`

## Logs
```
Paste relevant logs here
```
