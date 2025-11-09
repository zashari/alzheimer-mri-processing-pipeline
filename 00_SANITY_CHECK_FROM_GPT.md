
 # Quick audit of your tree 

1. **Packaging & imports (Option B):**
   ✅ `src/data_processing/` (single top-level package) — great for clean imports and packaging.

2. **CLI wiring:**
   ✅ `src/data_processing/cli.py` present.
   🔎 Ensure `pyproject.toml` exposes a console script (e.g., `adp = data_processing.cli:main`) and that your shell/bat scripts call `adp ...`.

3. **Stage interface & registry (extensible):**
   ✅ `src/data_processing/stages/base.py` + per-stage files.
   🔎 Add a small registry in `stages/__init__.py` so new stages register themselves without touching the CLI.

4. **Configs (typed, validated, layered):**
   ✅ `config/{schema.py,loader.py,validators.py}` and `configs/{default.yaml,stages/*.yaml}` — perfect.
   🔎 Confirm loader order: `default.yaml` → `stage.yaml` → user `--config` → env vars.

5. **Module responsibilities:**
   ✅ `environment_setup/`, `data_preparation/`, `nifti_processing/`, `image_processing/`, `visualizations/` map nicely to pipeline domains.

6. **Logging & run safety:**
   ✅ `logging_setup.py` and `utils/{errors.py,timers.py,randomness.py,io.py}`.
   🔎 Pass `--debug`/`--quiet`/`--dry-run` from CLI into stages for consistent behavior.

7. **Scripts:**
   ✅ Unix + Windows launchers under `scripts/*`—keep these calling the CLI once the entry point is wired.

8. **Testing & CI (quick wins):**
   ✅ `tests/` scaffolding and `.github/workflows/ci.yml`.
   🔎 In CI, run: ruff → black check → mypy → pytest (cache enabled). Consider a matrix for py38–py311 if needed.

9. **Documentation:**
   ✅ `README.md`, `CHANGELOG.md` present.
   🔎 Add a short “Config layering” section and a “Stage catalog” table for discoverability.

---

# Go-live checklist (no code—just actions)

* **Packaging**

  * Set the package name (e.g., `data_processing`) and version in `pyproject.toml`.
  * Add `console_scripts` entry (e.g., `adp = data_processing.cli:main`).

* **CLI contract**

  * Subcommands to standardize across stages: `env setup`, `prep split`, `nifti copy`, `image convert`, all supporting `--config`, `--dry-run`, `--debug`.
  * CLI loads configs via `config/loader.py`, initializes logging (`logging_setup.py`), then calls the stage registry.

* **Stage registry**

  * In `stages/__init__.py`, expose `get_stage(name)` and a `register(name, cls)` helper.
  * Each `*_stage.py` registers itself at import time with a unique key (e.g., `environment_setup`, `data_preparation`, `nifti_processing`, `image_processing`).

* **Config validation**

  * `validators.py` checks: required paths exist, CSV columns present, visit set valid, split ratios sum to 1.0, and OS-safe copy modes.

* **Safety & reproducibility**

  * One function to set seeds (`utils/randomness.py`) called by the CLI before invoking stages.
  * Support `ADP_CONFIG` env var as a last-mile override (optional).

* **Pre-commit & CI**

  * Pre-commit: ruff, black, isort (optional), mypy, end-of-file-fixer, trailing-whitespace.
  * CI job order: lint → type check → unit tests; fail fast on lint.

* **Docs**

  * In `README.md`, add:

    * **Quickstart** (install, `adp --help`, where configs live).
    * **Config layering** diagram.
    * **Stage catalog** with required inputs & produced outputs (manifests).
    * **Extending**: how to add a new stage in 3 steps (file, registry entry, YAML).

---

# Sanity checks you can run (manual, no code output here)

* `pip install -e .` then `adp --help` (after you wire the console script).
* Run a **dry run** for each stage with your smallest CSV to verify: config merge → logging → stage dispatch → summary counters.
* Confirm Windows `.bat` and Unix `.sh` scripts successfully call `adp` and pass `--config`.
