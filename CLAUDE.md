# Coker — Claude Instructions

## Before committing

Always run the linter and ensure it passes before committing any changes:

```
python -m black src
```

The CI pipeline runs `black --check --diff src` on all pull requests to `main`.
