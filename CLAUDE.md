# Coker — Claude Instructions

## Before committing

Always run the linter and ensure it passes before committing any changes:

```
python -m black src
```

The CI pipeline runs `black --check --diff src` on all pull requests to `main`.

## Naming

Do not use shortened or abbreviated variable names. Use clear, descriptive names that convey the meaning of the variable (e.g. `linear_linear_quadratic` not `ll`, `rotation_matrix` not `r`).
