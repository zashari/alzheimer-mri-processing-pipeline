# 1) Minimal stage registry map (names only)
Registry key → Stage purpose (one-liner)

* `environment_setup` → GPU/driver/CUDA probing, PyTorch build selection, environment verification.
* `data_preparation` → Load/validate metadata, subject/visit selection, stratified splits, manifest writing.
* `nifti_processing` → NIfTI discovery (priority rules), copy/symlink to standardized layout, (future) 3D preproc/QC.
* `image_processing` → 2D conversion/augmentation/export built from NIfTI outputs.

> Each stage file self-registers at import time under these exact keys.

---

# 2) CLI command layout (args & flags)

Top-level executable: `adp`

## Global options (apply to all subcommands)

* `--config <path>`: path to a user config file to merge on top of defaults and stage config.
* `--stage-config <path>`: explicitly point to a stage YAML (otherwise inferred by stage name).
* `--set key=value` (repeatable): ad-hoc overrides (dot-notation allowed, e.g., `paths.data_root=D:\data`).
* `--dry-run`: perform all checks and planning; no writes.
* `--debug`: verbose logging (DEBUG level).
* `--quiet`: reduce logging (WARN+).
* `--log-file <path>`: write logs to file in addition to console.
* `--seed <int>`: override global random seed.
* `--workers <int>`: worker/thread hint for I/O tasks (stages may ignore if N/A).
* `--version`: print package version.

> Command form: `adp <stage> <action> [stage-specific options] [global options]`

---

## Stage: `environment_setup`

Actions:

* `setup` — run full environment checks and (if enabled) auto-selection of torch build.
* `verify` — run lightweight GPU/CUDA/torch verification and short perf sanity test.

Stage-specific options:

* `--auto-install [true|false]`
* `--cuda-preference <11.8|12.1|auto>`
* `--perf-test [quick|full|off]`

Examples:

* `adp environment_setup setup --cuda-preference auto --perf-test quick --debug`
* `adp environment_setup verify --dry-run`

---

## Stage: `data_preparation`

Actions:

* `analyze` — load CSVs, validate columns, show group/visit statistics.
* `split` — produce subject-level stratified splits (train/val/test) and `metadata_split`.
* `manifests` — (re)write manifests only from current selection/splits.

Stage-specific options:

* `--split-ratios <train,val,test>` (e.g., `0.7,0.15,0.15`)
* `--required-visits <list>` (comma-separated, e.g., `sc,m06,m12`)
* `--stratify-by <col>` (e.g., `Group`)
* `--shuffle [true|false]`

Examples:

* `adp data_preparation analyze --debug`
* `adp data_preparation split --split-ratios 0.7,0.15,0.15 --required-visits sc,m06,m12 --dry-run`
* `adp data_preparation manifests`

---

## Stage: `nifti_processing`

Actions:

* `search` — list the best-match NIfTI per (Subject, Visit) using priority rules (no writes).
* `copy` — materialize the standardized dataset layout (copy/symlink) from search results.
* `qc` (future) — basic header checks / missing-file report.
* `process3d` (future) — entry point for 3D transforms/preprocessing.

Stage-specific options:

* `--priority <Scaled_2,Scaled,Raw>` (ordered list, e.g., `Scaled_2,Scaled,Raw`)
* `--mode <copy|symlink>` (platform-aware)
* `--overwrite [true|false]`
* `--limit <int>` (limit subjects for quick runs)

Examples:

* `adp nifti_processing search --priority Scaled_2,Scaled,Raw --dry-run`
* `adp nifti_processing copy --mode copy --overwrite false --workers 8`

---

## Stage: `image_processing`

Actions:

* `convert2d` — generate 2D representations from NIfTI (placeholders for now).
* `augment` — optional augmentations on converted data.
* `export` — package outputs for training pipelines.

Stage-specific options (examples/placeholders):

* `--plane <axial|sagittal|coronal>`
* `--slice-policy <center|percentile:50|index:80>`
* `--output-format <png|jpeg>`
* `--compression <int>`

Examples:

* `adp image_processing convert2d --plane coronal --slice-policy percentile:50 --output-format png --dry-run`

---

# 3) Config layering spec

## Purpose

Provide deterministic, flexible configuration without code edits. Later stages add options safely.

## Sources (lowest → highest precedence)

1. **Package defaults**: `configs/default.yaml`

   * Global paths (data root, output root), seeds, logging defaults, OS behavior hints.
2. **Stage defaults**: `configs/stages/<stage>.yaml`

   * Only stage-specific keys (e.g., required visits, split ratios, NIfTI priority).
3. **User file** (optional): `--config <path>`

   * Any keys; overrides the two above.
4. **Environment variables** (optional):

   * Prefix `ADP_` with dot-notation, e.g., `ADP_paths.data_root=D:\data`. Parsed and merged after user file.
5. **CLI overrides** (optional): `--set key=value` (repeatable)

   * Highest precedence; applies last.

> All merges are **deep merges**, not replace-object merges.

## Validation

* After merging, the **typed schema** (pydantic/dataclasses) is applied:

  * Required keys present (paths, CSV columns, stage knobs).
  * Types correct (e.g., ratios are floats, lists resolved).
  * Semantic checks (examples):

    * Split ratios sum to 1.0 (±1e-9).
    * `required_visits` is non-empty and a subset of known visits in metadata.
    * `mode` ∈ {copy, symlink} and symlink allowed on current platform.
    * Paths exist or can be created; outputs do not clobber inputs unless `overwrite=true`.

## Introspection & logging

* On every run, log a **compact, redacted** view of the final config (no secrets).
* Optional `--debug` logs the full resolved config to file (`--log-file`), never to console in production mode.

## Example resolution (conceptual)

* Start with `default.yaml`.
* Merge `stages/data_preparation.yaml` when running `adp data_preparation ...`.
* Merge user file from `--config`.
* Apply env overrides (`ADP_*`).
* Apply CLI `--set` overrides.
* Validate; if invalid, print precise error path and abort (non-zero exit).

## Contract between stages (high-level)

* **Manifests** are the handoff artifact:

  * `metadata_split.csv` (per Subject/Visit with splits)
  * `train.csv`, `val.csv`, `test.csv` (stage-agnostic schema: Subject, Visit, Group, FilePath, …)
* Later stages **must not** reinterpret split logic; they **consume** manifests.
