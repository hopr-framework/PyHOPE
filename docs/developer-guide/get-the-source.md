# Environment Setup

Get the source and start developing

---

## Obtaining the Source

PyHOPE uses [git](https://git-scm.com) to sync down source code. Some larger mesh files are stored in [Git LFS](https://git-lfs.com). `git-lfs` must be installed before cloning, otherwise those files will be replaced by pointer stubs.

!!! info
    PyHOPE developers commonly put their virtual environment in `venv` within the git repository. All commands in this document assume that your virtual environment is in `venv`.

```bash
git clone https://github.com/hopr-framework/PyHOPE.git
cd PyHOPE
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade -e .
```

!!! info
    Changes in the source code are immediately reflected within the virtual environment without any explicit sync requirement.

## Linting

PyHOPE enforces code quality through three static analysis tools which are wired up as [pre-commit](https://pre-commit.com/) hooks and run as the first stage of the CI pipeline.

### Tools

**[Ruff](https://docs.astral.sh/ruff/)** is a fast Python linter that consolidates the roles of `flake8`, `black`, `isort`, and many others in a single tool. It checks style and common errors across all `.py` files.  

!!! note
    Some linter errors can be fixed automatically:
    ```bash
    ruff check --fix
    ```
    Note that `--unsafe-fixes` may silently change program behaviour; `--fix` is guaranteed to be behaviour-preserving.

**[ty](https://github.com/astral-sh/ty)** is a static type checker by Astral. It resolves all imports against the full installed dependency set.

**[vulture](https://github.com/jendrikseipp/vulture)** detects dead code such as unused functions, unreachable branches, and variables that are assigned but never read.

### Pre-commit Integration

All three tools are configured as hooks in `.pre-commit-config.yaml`. Install them once after cloning.

```bash
pip install pre-commit
pre-commit install
```
From that point on, every `git commit` runs ruff, vulture, and ty against the staged files automatically. To run all hooks explicitly against the full codebase.

```bash
pre-commit run --all-files
```
To skip the hooks for a specific commit, pass `--no-verify` to `git commit`. Use this sparingly as the CI pipeline enforces the same rules and will catch anything skipped locally.

!!! note
    [ty](https://github.com/astral-sh/ty) does not yet have an official pre-commit hook ([astral-sh/ty#269](https://github.com/astral-sh/ty/issues/269)). The local hook in `.pre-commit-config.yaml` calls the system-installed `ty` binary directly. Make sure `ty` is installed in your virtual environment (`pip install ty`) before committing.

### Ruff Configuration

Ruff's rules, exclusions, and per-file overrides are managed through `pyproject.toml` in the project root. To suppress a specific violation on a single line, use a `# noqa` comment with the rule code:

```python
x = 1        # noqa: F841        # suppress unused-variable for this line only
i = 1        # noqa: E741, F841  # suppress multiple rules
x = 1        # noqa              # suppress all violations on this line
```

!!! note
    Use inline suppressions only when the violation is a known false positive or an intentional deviation. Document the reason in a comment where it is not obvious.

## Continuous Integration

PyHOPE runs a CI pipeline on every pull request. Only one pipeline run executes at a time per branch; a newer push cancels any run still in progress. The pipeline is structured in three stages that run in order.

**1. Lint**

The three tools described above run in parallel. All three must pass before the compatibility stage begins. Running `git commit` locally triggers the same checks automatically via pre-commit.

**2. Compatibility**

A matrix job runs the mesh generation pipeline against five Python versions: **3.10 – 3.14**. All versions run in parallel, so a failure on one version does not abort the others. Two representative tutorials are executed end-to-end:

```bash
pyhope tutorials/1-04-cartbox_multiple_stretch/parameter.ini
pyhope tutorials/2-02-external_mesh_CGNS_mixed/parameter.ini
```

**3. Verification**

| Job             | Description                                                                                                                                                                   |
| ---             | ---                                                                                                                                                                           |
| **examples**    | Runs the full tutorial suite with coverage instrumentation                                                                                                                    |
| **healthcheck** | Runs PyHOPE's internal health checks: `coverage run -m pyhope --verify tutorials`                                                                                             |
| **convergence** | Runs FLEXI solver convergence tests in a dedicated Fedora container (`ghcr.io/hopr-framework/nrg-fedora`). Requires `GITHUB_TOKEN` to pull from the GitHub Container Registry |
| **coverage**    | Merges `coverage.xml` artifacts from `examples` and `healthcheck`. On PRs targeting `main`, a bot posts a summary comment. Thresholds: **85% overall**, **90% for new code**  |
