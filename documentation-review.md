# Documentation review — AEIC / trajectories-to-grid

Review date: 2026-04-15. Scope: everything under `docs/`, plus top-level
markdown (`README.md`, `CHANGELOG.md`, `CONDUCT.md`, `CONTRIBUTING.mb`,
`data-dictionary.md`, `grid-map-optimization.md`). "Public interface" =
names exported via `__init__.py` `__all__` plus CLI subcommands registered
in `pyproject.toml`. Non-exported-but-importable items are in Appendix A.

Each finding is tagged **Critical** (users will hit errors or wrong
results), **Major** (misleading / silently outdated / missing public
surface), or **Minor** (cosmetic, duplication, typos).

---

## 1. Top 10 to fix first

| # | Severity | Where | Summary |
|---|----------|-------|---------|
| 1 | Critical | `docs/src/oag.md:21` | `convert-oag-data` example uses `uv run convert-oag-data` — command no longer exists; must be `aeic convert-oag-data`. **DONE** |
| 2 | Critical | `docs/src/performance_models/performance_model_files.md:35-43` | `make-performance-model` example is stale (wrong invocation + missing required `--lto-source` flag). Example will not run. **DONE** |
| 3 | Critical | `docs/src/configuration.md:51` | Example imports `LTOInputMode` from `AEIC.config`; no such name exists. Import raises `ImportError`. **DONE** |
| 4 | Major | `docs/src/developer/tools.md:7` | Claims "AEIC uses Python 3.13"; `pyproject.toml` requires `>=3.12,<3.13`. Actively wrong. **DONE** |
| 5 | Major | CLI coverage | `aeic run`, `aeic merge-stores`, `aeic make-file-bundle` have no documentation pages and no mention in the TOC. **DONE** |
| 6 | Major | `docs/src/parsers.md` | Empty stub page listed in TOC; `AEIC.parsers` has three submodules (`lto_reader`, `opf_reader`, `ptf_reader`) that are entirely undocumented. **DONE** |
| 7 | Major | `docs/src/emission.md:113` | Cross-ref target `AEIC.emission.emission.Emissions` (singular) — correct path is `AEIC.emissions.emission.Emissions`. Broken link. **DONE** |
| 8 | Major | `docs/src/mission_database.md:82, 286` | `AEIC.missions.WritableDatabase` is not exported from `__init__.py`; cross-refs won't resolve. **DONE** |
| 9 | Major | `data-dictionary.md:11` | Malformed table row (only 3 cells for "Airspeed") AND content drifts from the canonical copy in `docs/src/developer/conventions.md`. **DONE** |
| 10 | Major | `docs/src/gridding.md:345, 349` | `autofunction` directives reference `map_phase` / `reduce_phase` in `AEIC.commands.trajectories_to_grid`, which are command-module internals, not public API. **DONE** |

---

## 2. Docs don't match code

### Critical — stale/broken examples, dead imports, wrong CLI invocations

- **Critical** `docs/src/oag.md:21,24` → `src/AEIC/cli.py:32`, `src/AEIC/commands/convert_oag_data.py`
  Example invocation `uv run convert-oag-data -i 2024.csv ...` and the phrase
  "options to the `convert-oag-data` script" refer to a standalone command.
  The only registered script in `pyproject.toml` is `aeic`; OAG conversion
  is a Click subcommand registered as `aeic convert-oag-data`.
  *Fix:* replace `uv run convert-oag-data` with `aeic convert-oag-data` (or
  `uv run aeic convert-oag-data`); same change in the options preamble.
  **DONE** — replaced `uv run convert-oag-data` with `aeic convert-oag-data`
  in the example and preamble prose.

- **Critical** `docs/src/performance_models/performance_model_files.md:35-43` → `src/AEIC/commands/make_performance_model.py:74-145`
  Example uses `uv run make-performance-model ... legacy ...`; correct
  invocation is `aeic make-performance-model ... legacy ...`. The `legacy`
  subcommand also requires `--lto-source edb|custom` (see
  `make_performance_model.py:80`), which the example omits — so even after
  fixing the invocation, the example fails with "Missing option
  '--lto-source'".
  *Fix:* change to `aeic make-performance-model` and add
  `--lto-source edb` in the example (the `--engine-file`/`--engine-uid`
  already imply `edb`).
  **DONE** — replaced `uv run make-performance-model` with
  `aeic make-performance-model` and added `--lto-source edb` to the
  example invocation.

- **Critical** `docs/src/configuration.md:51` → `src/AEIC/config/__init__.py:3`
  Code block does `from AEIC.config import Config, LTOInputMode`. `AEIC.config`
  exports only `Config` and `config`; `LTOInputMode` does not exist anywhere
  in the tree. The rest of the example only uses `Config.load(...)`, so
  `LTOInputMode` is a stale leftover.
  *Fix:* remove `LTOInputMode` from the import.
  **DONE** — removed `LTOInputMode` from the import in the example.

### Major — divergent prose, broken xrefs, module-path typos

- **Major** `docs/src/emission.md:113-114` → `src/AEIC/emissions/emission.py:70`
  Cross-ref written as `<AEIC.emission.emission.Emissions>` (module named
  singular `emission` — does not exist). Autoclass directive at L124 uses
  the correct plural `AEIC.emissions.emission.Emissions`.
  *Fix:* change prose cross-ref to `AEIC.emissions.emission.Emissions` (or
  `AEIC.emissions.Emissions`, since it's re-exported via `__all__`).
  **DONE** — fixed the cross-ref target to `AEIC.emissions.emission.Emissions`.

- **Major** `docs/src/mission_database.md:82, 286` → `src/AEIC/missions/__init__.py` (no export)
  Two references to `{py:class}`AEIC.missions.WritableDatabase``. The class
  lives at `src/AEIC/missions/writable_database.py:99` but is NOT in
  `AEIC.missions.__all__`, so the cross-ref won't resolve. The prose at
  L82 ("There is a derived class called `WritableDatabase` for applications
  that construct flight databases") also implies it's public API; in
  practice it's only used internally by `convert_oag_data`.
  *Fix:* either export it (add to `missions/__init__.py` `__all__`) or
  change cross-refs to `AEIC.missions.writable_database.WritableDatabase`
  and clarify it's internal.
  **DONE** — changed both cross-refs in `mission_database.md` to
  `AEIC.missions.writable_database.WritableDatabase` and updated the
  surrounding prose to clarify that the class is internal (used by
  `aeic convert-oag-data`) and not part of the public API.

- **Major** `docs/src/gridding.md:345, 349` → `src/AEIC/commands/trajectories_to_grid.py:30, 275`
  `autofunction` directives for `map_phase` and `reduce_phase` expose
  internal command-module functions as public API. These are not intended
  for library users (they take specific internal arguments like
  `traj_iter`, `traj_repro`, `traj_comments` and have implicit preconditions).
  *Fix:* either drop these directives from the public doc page or move the
  functions under `AEIC.gridding` and document them there as stable API.
  **DONE** — dropped the `map_phase` / `reduce_phase` `autofunction`
  directives (and the enclosing "Command functions" subsection) from
  `gridding.md`. The map/reduce behaviour is still described in prose
  under "Map mode" / "Reduce mode" earlier in the same page.

- **Major** `docs/src/performance_models/performance_model_api.md:43-44`
  Visible cross-ref text reads "SimplPerformanceModelSelector" (missing
  'e'). The *target* is correct (`SimplePerformanceModelSelector`), but
  the rendered label is wrong.
  *Fix:* change to `{py:class}`SimplePerformanceModelSelector <...>``.
  **DONE** — fixed the visible label to `SimplePerformanceModelSelector`.

- **Major** `docs/src/developer/tools.md:7` → `pyproject.toml:18`
  "AEIC uses Python 3.13." `pyproject.toml` pins
  `requires-python = ">=3.12,<3.13"`, which *excludes* 3.13. Python 3.12
  is the only supported version.
  *Fix:* change to "AEIC uses Python 3.12."
  **DONE** — updated `docs/src/developer/tools.md:7` to say
  "AEIC uses Python 3.12."

- **Major** `docs/src/parsers.md` (entire file) → `src/AEIC/parsers/` (submodules)
  Single heading "# Available parsers" and no body. `AEIC.parsers`
  contains `lto_reader.py`, `opf_reader.py`, `ptf_reader.py`; none are
  documented. `ptf_reader.PTFData` is used by the
  `make-performance-model` command and is effectively public.
  *Fix:* add a short overview and `automodule` directives for each
  submodule; or, if parsers are genuinely internal, remove the page from
  the TOC (`docs/index.rst:38`).
  **DONE** — fleshed out `docs/src/parsers.md` with a short overview and
  `automodule` directives for `ptf_reader`, `opf_reader`, and `lto_reader`.

### Major — stale "uv run" pattern (lower-impact instances)

- **Major** `docs/src/developer/tools.md:58-60` → `pyproject.toml`
  Pre-commit install instructions use `pip install --user pre-commit` or
  `uv run pip install --user pre-commit`. The rest of the page advocates
  `uv sync`-managed environments; `pip install --user` installs into
  `~/.local/bin` outside the uv environment and may surprise uv users.
  *Fix:* prefer `uv tool install pre-commit` (installs into uv's tool bin)
  or document both paths.
  **DONE** — recommend `uv tool install pre-commit` in the primary step
  and keep `pip install --user pre-commit` as a documented fallback.

### Spot-check of code examples (Tier-iii)

- `docs/src/emission.md:23-57` — imports, `LegacyBuilder(options=Options(iterate_mass=False))`, `compute_emissions(perf, fuel, traj)` all resolve against current code.
- `docs/src/performance_models/performance_model_api.md:84-112` — imports and `PerformanceModel.load()` / `evaluate()` signatures match current code.
- `docs/src/mission_database.md:27-49` — uses `AEIC.missions.Database`, `Query`, `Filter`; all exported and current.
- `docs/src/weather.md:21-34` — `Weather(path, mission, track)` and `get_ground_speed(ground_distance, altitude, true_airspeed)` — not verified against source in this pass; flagged for secondary audit.

---

## 3. Undocumented public paths

"Public" = names exported via `__init__.py` `__all__` (or module-level
non-underscore names for plain modules) that are not mentioned in any
`docs/` page.

### Major — missing from public API docs

- **Major** `AEIC.storage.PHASE_FIELDS`, `access_recorder`,
  `track_file_accesses`, `ReproducibilityData`, `Container`,
  `HasFieldSets` — all exported from `src/AEIC/storage/__init__.py`, none
  appear in docs. `storage` has no dedicated page. `FlightPhase`,
  `FieldSet`, `FieldMetadata`, `Dimension`, `Dimensions` are mentioned
  from other pages (`trajectory_data.md`) but via deep `AEIC.storage.X.Y`
  paths rather than the curated top-level exports.
  *Fix:* add a short `docs/src/storage.md` page (or a section in
  `trajectory_stores.md`), with `automodule AEIC.storage`.
  **DONE** — added `docs/src/storage.md` with an overview of the field
  sets / dimensions / phase / reproducibility groupings and an
  `automodule AEIC.storage` block; linked from `docs/index.rst`.

- **Major** `AEIC.trajectories.BASE_FIELDSET_NAME` — exported, not
  referenced anywhere in docs.
  *Fix:* mention under "Field sets" in `docs/src/trajectories/trajectory_data.md`.
  **DONE** — added a paragraph under "Field sets" describing
  `BASE_FIELDSET_NAME` and an `autodata` directive for it.

- **Major** `AEIC.types.FloatOrNDArray` — exported; not documented.
  Alias used throughout the codebase for scalar-or-array numeric
  parameters.
  *Fix:* add a brief entry in `docs/src/utilities.md` "Custom types".
  **DONE** — added a short description of `FloatOrNDArray` and an
  `autodata` directive to the "Custom types" section of
  `utilities.md`.

- **Major** `AEIC.trajectories.builders.ADSBBuilder / ADSBOptions`,
  `DymosBuilder / DymosOptions`, `TASOPTBuilder / TASOPTOptions` —
  exported but not mentioned in `trajectory_builders.md` beyond a
  passing note that only `LegacyBuilder` is fully implemented.
  *Fix:* either add stub sections flagging them as WIP or remove them
  from `__all__` until implemented.
  **DONE** — added a "Work-in-progress builders" section in
  `trajectory_builders.md` with a stub subsection and WIP warning for
  each of `TASOPTBuilder`, `ADSBBuilder`, and `DymosBuilder`.

### CLI (see also §4)

- **Major** `aeic run`, `aeic merge-stores`, `aeic make-file-bundle` — no
  documentation pages; no appearance in `docs/index.rst`.
  **DONE** — added `docs/src/cli.md` covering all three commands and
  linked it from `docs/index.rst`.

---

## 4. CLI documentation gaps

The `aeic` CLI has 6 subcommands (`src/AEIC/cli.py:32-37`):

| Subcommand | Documented? | Issues |
|------------|-------------|--------|
| `trajectories-to-grid` | Yes (`gridding.md`) | Good coverage; options table matches code. |
| `convert-oag-data` | Partial (`oag.md`) | Wrong invocation (Critical #1). Options table matches code. |
| `make-performance-model` | Partial (`performance_model_files.md`) | Wrong invocation + missing `--lto-source` (Critical #2). `tasopt` subcommand undocumented. |
| `run` | **No** | Not mentioned anywhere; 9 options undocumented. |
| `merge-stores` | **No** | Not mentioned anywhere; 2 options + positional `input-stores` undocumented. |
| `make-file-bundle` | **No** | Not mentioned anywhere; 2 options undocumented. |

### Detailed findings

- **Major** `aeic run` → `src/AEIC/commands/run_simulations.py`
  Flags: `--config-file`, `--performance-selector-dir`,
  `--performance-model-file`, `--mission-db-file`, `--output-store`,
  `--sample`, `--seed`, `--slice-count`, `--slice-index`. Semantics
  include "exactly one of selector-dir or model-file must be provided"
  (run_simulations.py:160), parallel slicing similar to
  `trajectories-to-grid`, and SQLite-based `CountQuery` for flight counts.
  *Fix:* add a "Running simulations" page (or section in
  `trajectories/`) with flag table, examples for single/parallel runs,
  and the selector-vs-model-file choice.
  **DONE** — added `aeic run` section in the new `docs/src/cli.md`
  reference page with flag table, selector-vs-model-file note and
  single-process + parallel examples.

- **Major** `aeic merge-stores` → `src/AEIC/commands/merge_stores.py`
  Flags: `--output-store` (required), `--merge/--combine`, plus
  positional `input-stores` (N-ary). `TrajectoryStore.merge` vs
  `.combine` is a real semantic distinction (multi-file vs single-file
  store) but is not explained in docs. `trajectory_stores.md:36-42`
  mentions merging but without the command. `merge-stores` is an
  essential step in the parallel workflow documented in
  `gridding.md:224-259` — the gap is load-bearing.
  *Fix:* document the command and link it from the parallel-simulation
  section of `gridding.md` and from `trajectory_stores.md`.
  **DONE** — added `aeic merge-stores` section in the new `docs/src/cli.md`
  reference page covering `--merge`/`--combine` semantics and a parallel
  example.

- **Major** `aeic make-file-bundle` → `src/AEIC/commands/make_file_bundle.py`
  Flags: `--input-store` (required), `--output-bundle` (required).
  Produces a zip of all files referenced by a store's reproducibility
  data. This is the companion to the "reproducibility provenance"
  machinery described in `gridding.md:292-300` and
  `trajectory_stores.md`.
  *Fix:* document the command in a new "Reproducibility bundles" section.
  **DONE** — added `aeic make-file-bundle` section in the new
  `docs/src/cli.md` reference page.

- **Major** `aeic make-performance-model tasopt` → `make_performance_model.py:184-188`
  `tasopt` is registered as a subcommand but raises
  `NotImplementedError`. Docs say only that "only the legacy performance
  model is implemented" but don't mention the stub subcommand exists.
  *Fix:* either hide the subcommand until implemented, or add a single
  line noting `tasopt` exists and raises `NotImplementedError`.
  **DONE** — added a line inside the existing warning in
  `performance_model_files.md` noting that `aeic make-performance-model
  tasopt` is registered but currently raises `NotImplementedError`.

- **Minor** No top-level "CLI reference" index in `docs/index.rst`.
  Users must hunt across `gridding.md`, `oag.md`, and
  `performance_model_files.md` to discover what the `aeic` command can
  do.
  *Fix:* add a `docs/src/cli.md` page listing all 6 subcommands with
  one-line summaries and links to their detail sections.
  **DONE** — the `docs/src/cli.md` page added for §3/§4 above opens
  with a table of all six subcommands with one-line summaries and
  links, and is listed in `docs/index.rst`.

---

## 5. Cross-file duplication and top-level markdown

- **Major** `data-dictionary.md` (L1-22) vs `docs/src/developer/conventions.md` (L38-57)
  Both contain a "Data dictionary" / units table. They have drifted:
  - `data-dictionary.md:11` is malformed (only 3 cells, missing `|`):
    `| Airspeed | m/s `true_airspeed` | |`. Will render as a broken
    table row.
  - `data-dictionary.md` lacks the `Thrust | N | ..._thrust | kN` row.
  - `data-dictionary.md` lists `Flight level | - | flight_level` where
    `conventions.md` has `Flight level | FL (1) | flight_level` plus a
    footnote defining the unit.
  *Fix:* delete `data-dictionary.md` at the repo root and keep
  `conventions.md` as the single source of truth (Sphinx reference anchor
  `data-dictionary` already exists). Add a redirect note in the root
  file if anything links to it externally.
  **DONE** — deleted `data-dictionary.md` at the repo root; the only
  internal references were from `docs/src/developer/conventions.md`
  (the `(data-dictionary)=` anchor) and this review document itself, so
  no redirect stub was needed.

- **Minor** `README.md:7-62` vs `docs/main_page.md:1-29`
  Installation and Units sections are duplicated. `pyproject.toml:84-87`
  wires `bump-my-version` to update both in lock-step — the version
  string stays in sync, but prose can still drift (e.g. `README.md`
  describes `pip install --editable .` whereas `main_page.md` / Sphinx
  docs describe only the `uv sync` workflow).
  *Fix:* consider promoting `main_page.md` to include a Local
  Development section pulled via `.. include::`, and shortening
  `README.md` to link to the built docs rather than duplicating
  content.
  **DONE** — dropped the duplicated Local Development, `uv`, and
  pre-commit subsections from `README.md` and replaced them with a
  single short paragraph linking to `docs/src/developer/tools.md`.
  The remaining Installation and Units sections still match
  `docs/main_page.md` and are already kept in sync by
  `bump-my-version`.

- **Minor** `docs/conf.py:22` vs `pyproject.toml:4`
  `release = '1.0.0a1'` vs `version = "0.3.0"`. The Sphinx title bar and
  PDF output will show the wrong version.
  *Fix:* read the version from `pyproject.toml` in `conf.py`
  (`tomllib.load(...)`), or update `release` and add it to the
  `bump-my-version` `files` list.
  **DONE** — `docs/conf.py` now reads `release` from
  `pyproject.toml` via `tomllib`, so the Sphinx title bar and PDF
  output automatically track the project version.

- **Minor** `CONTRIBUTING.mb` (file at repo root)
  Filename typo: `.mb` extension is not a standard Markdown extension
  and may not render on GitHub.
  *Fix:* rename to `CONTRIBUTING.md`. Check for any links pointing at
  the old name.
  **DONE** — renamed via `git mv` to `CONTRIBUTING.md`; no other file
  in the repo (outside this review document) referenced the old
  `.mb` name, so no link updates were needed.

- **Minor** `CHANGELOG.md` (0 bytes) and `CONDUCT.md` (0 bytes)
  Both files exist but are empty. `CHANGELOG.md` is especially relevant
  given `releases.md` documents a release workflow.
  *Fix:* either populate or remove these placeholder files.
  **DONE** — removed both 0-byte placeholders (nothing in the repo
  referenced them). They can be re-added with real content when a
  changelog workflow or Code of Conduct is actually adopted.

- **Minor** `grid-map-optimization.md` (repo root, ~100 lines)
  A detailed performance report with benchmarks and git-history context.
  Not referenced in `docs/` TOC. Either it's a PR-adjacent artifact (in
  which case it could move to PR description / wiki) or it's intended
  reference material (in which case it belongs under `docs/src/`).
  *Fix:* decide if this is documentation or a one-off report, and move
  or remove accordingly.

---

## 6. Other finds

- **Minor** `docs/src/configuration.md:56` — `{py.mod}` typo (dot
  instead of colon); renders literally as "{py.mod}". Should be
  `{py:mod}`.
- **Minor** `docs/src/weather.md:19` — `Example::` is reStructuredText
  convention for introducing a code block; in MyST it renders as
  literal text with a trailing `::`.
  *Fix:* change to `Example:` or remove.
- **Minor** `docs/src/emission.md:80, 89` — typo "emissiosn" (two
  occurrences) in the workflow description.
- **Minor** `docs/src/trajectories/trajectory_builders.md` §Legacy —
  coverage builder reports `LegacyBuilder.fly_climb` and
  `LegacyBuilder.fly_descent` have no docstrings. `fly_cruise` does have
  one. (See Appendix B.)
- **Minor** `docs/src/utilities.md:50-52` — `automodule: AEIC.utils.airports`
  uses `:exclude-members: CountriesData, AirportsData`; both classes
  then appear as undocumented in the coverage builder output because
  they're not documented elsewhere. If they're intended to be internal,
  this is fine; if public, they need their own autoclass directives.

---

## Appendix A — Non-exported-but-importable gaps (low priority)

These modules/names are reachable via dotted import paths but are not
exposed through `__init__.py` `__all__`. Per this review's scope (strict
definition), they are treated as internal. Listed here because they're
either (a) used from docs via deep paths, or (b) heavily used by
consumers and might warrant promotion to public API.

- **`AEIC.storage.phase`, `AEIC.storage.field_sets`, `AEIC.storage.dimensions`,
  `AEIC.storage.reproducibility`** — documented from `trajectory_data.md`
  via deep paths. Consider whether these deserve top-level re-exports or
  a standalone `storage.md` doc page (also flagged as Major above).
- **`AEIC.constants`** — top-level module, empty `__init__`, no doc
  page. If genuinely internal, no action needed.
- **`AEIC.verification`** — empty package at `src/AEIC/verification/`,
  no doc page. Looks like a placeholder.
- **`AEIC.BADA.model`, `.aircraft_parameters`, `.fuel_burn_base`,
  `.helper_functions`** — `AEIC.BADA.__init__` is empty, but `bada.md`
  documents these submodules via deep paths. Works, but the missing
  top-level re-export means `from AEIC.BADA import Bada3JetEngineModel`
  fails, surprising users.
- **`AEIC.performance.apu.APU`, `AEIC.performance.model_selector.*`,
  `AEIC.performance.types.*`** — heavily referenced from
  `performance_model_api.md` via deep paths; not surfaced at the
  `AEIC.performance` level.
- **`AEIC.emissions.emission.Emissions`** — IS re-exported as
  `AEIC.emissions.Emissions` via `__all__`, but every doc cross-ref uses
  the deep path. Not broken, just inconsistent.
- **`AEIC.units` (plain module)** — `FL_TO_METERS`, `STATUTE_MILES_TO_KM`,
  `FEET_TO_METERS` etc. live here. `utilities.md` uses
  `automodule AEIC.units :members:` which will pull everything. OK.
- **`AEIC.utils.progress.Progress`** — used by all long-running CLI
  commands; no doc page.
- **`AEIC.utils.airports.AirportsData`, `.CountriesData`** — explicitly
  excluded from autodoc in `utilities.md:51`; coverage builder reports
  them as undocumented classes.
- **`AEIC.utils.models.CIBaseModel`, `CIStrEnum`** — base classes used
  throughout the codebase for case-insensitive TOML parsing. `utilities.md`
  runs `automodule AEIC.utils.models :members:`, but the coverage builder
  reports `CIBaseModel.normalize_keys` as undocumented (no docstring).

---

## Appendix B — Sphinx coverage builder output

Generated with `uv run sphinx-build -b coverage docs docs/_build/coverage`
on 2026-04-15. Full output at `docs/_build/coverage/python.txt`.

Summary: 14 modules inspected, 90.32% coverage, 3 undocumented items (the
coverage builder only inspects modules that are reached by an autodoc
directive — so this is a narrow signal, not a full audit).

Undocumented items reported:

```
AEIC.trajectories.builders.legacy
---------------------------------
 * LegacyBuilder -- missing methods:
   - fly_climb
   - fly_descent

AEIC.utils.airports
-------------------
 * AirportsData
 * CountriesData
 (both intentionally excluded from autodoc via :exclude-members: in
  utilities.md:51 — but not documented elsewhere)

AEIC.utils.models
-----------------
 * CIBaseModel -- missing methods:
   - normalize_keys
```

Modules NOT inspected by the coverage builder because no autodoc
directive reaches them (i.e., genuinely absent from docs): `AEIC.storage`
(except `storage.phase`), `AEIC.constants`, `AEIC.verification`,
`AEIC.utils.progress`, the three `AEIC.parsers.*` readers, and the
`AEIC.commands.*` CLI command modules. These are the real coverage
gaps — the coverage builder can't see them.
