# Contributing

Work happens on feature branches merged via pull request. `main` stays green.

1. Branch: `git checkout -b feat/<short-name>`
2. Keep changes focused; one concern per PR.
3. `make lint && make test` must pass before pushing.
4. Open a PR; CI (ruff + CPU pytest) must be green to merge.

Tests run on CPU with a random tiny model and never download weights, so the suite is fast and
deterministic. GPU- or network-dependent code paths (`load_model`, `load_guard`, `data.load_all`)
are exercised in notebooks, not CI.
