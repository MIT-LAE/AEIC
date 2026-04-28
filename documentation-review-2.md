# Documentation review #2 — AEIC

Review date: 2026-04-28. Scope: everything under `docs/`, plus top-level
markdown (`README.md`, `CLAUDE.md`, `CONTRIBUTING.md`). "Public interface" =
names exported via `__init__.py` `__all__` plus CLI subcommands registered
in `pyproject.toml`. The previous review (saved as `documentation-review.md`,
2026-04-15) was completed — all items there were marked **DONE** and shipped
in PR #137. This review focuses on drift introduced *after* that, primarily
by:

| PR    | Commit    | Title                                                    |
|-------|-----------|----------------------------------------------------------|
| #134  | f354566   | Bring Mission class into line with OAG database          |
| #135  | 3ad3030   | Generate commented, formatted performance model TOML     |
| #138  | c7b650b   | Support multiple ERA5 temporal layouts in Weather class  |
| #139  | 50817c8   | Pre-V1 simulation fixes                                  |

Sphinx coverage was rebuilt for this review (`uv run sphinx-build -b
coverage docs docs/_build/review2-coverage`); see Appendix A.

Each finding is tagged **Critical** (users will hit errors or wrong
results), **Major** (misleading / silently outdated / missing public
surface), or **Minor** (cosmetic, duplication, typos).

---

## 1. Top items to fix first

| # | Severity | Where | Summary |
|---|----------|-------|---------|
| 1 | Critical | `docs/src/weather.md:21-34` | Example uses the pre-#138 `Weather('path', mission, track)` signature. The current constructor takes `(data_dir, file_resolution, data_resolution=None, file_format=None)` — the example will not run. **DONE** |
| 2 | Critical | `docs/src/performance_models/performance_model_files.md:101-114` | Example TOML uses `cas_lo` / `cas_hi`. The current model rejects these — fields are `cas_low` / `cas_high` (see `src/AEIC/data/performance/sample_performance_model.toml:23-36`). **DONE** |
| 3 | Critical | `docs/src/performance_models/performance_model_files.md:124` | Example TOML uses `Foo_kN`. Current field is `rated_thrust`. **DONE** |
| 4 | Critical | `docs/src/performance_models/performance_model_files.md:164-181` | Example TOML shows a single `[flight_performance]` table with 6 columns. PR #135 changed the generated layout to three separate tables (`[climb_flight_performance]`, `[cruise_flight_performance]`, `[descent_flight_performance]`) with 5 columns each. **DONE** |
| 5 | Critical | `docs/src/performance_models/performance_model_files.md:36-47` | The `aeic make-performance-model legacy` example is missing the **required** `--maximum-payload` flag (see `commands/make_performance_model.py:324-329`). The command will exit with "Missing option '--maximum-payload'". **DONE** |
| 6 | Critical | `docs/src/emission.md:94` | Cross-ref target `<AEIC.emissions.types.Emissions>` does not exist. The class lives at `AEIC.emissions.emission.Emissions` (re-exported as `AEIC.emissions.Emissions`). Broken link. **DONE** |
| 7 | Major | `docs/src/weather.md:11-12` | Prose still says `valid_time` is "sliced using `mission.departure.hour` if present". After #138, slicing is by `data_resolution`-aware nearest-time selection, not by hour. **DONE** |
| 8 | Major | `docs/src/configuration.md` (no entry) | PR #138 added a public enum `TemporalResolution` and helper functions `default_file_format` / `resolution_le` in `AEIC.config.weather`. None are mentioned in prose; only `WeatherConfig` is documented. **DONE** |
| 9 | Major | `src/AEIC/trajectories/builders/base.py:157` | `Builder.fly` docstring lists the third argument as `startMass`. The actual parameter is `starting_mass`. The autodoc render in `trajectory_builders.md` therefore documents a non-existent kwarg. **DONE** |
| 10 | Major | `docs/src/mission_database.md` | `TimeRangeQuery` is exported via `AEIC.missions.__all__` (added in #134) but is not mentioned anywhere in `mission_database.md`. **DONE** |

---

## 2. Docs don't match code

### Critical — stale/broken examples, dead imports, dead xrefs

- **Critical** `docs/src/weather.md:21-34` → `src/AEIC/weather.py:46-52`
  The example builds a `Mission` and `GroundTrack`, then constructs
  `Weather('data/weather/sample_weather_subset.nc', mission, track)`. After
  PR #138 the signature is
  `Weather(data_dir, file_resolution, data_resolution=None, file_format=None)`
  — `data_dir` must be a *directory*, not a file, and `file_resolution` is
  required. The example also passes `mission` and `track` positionally
  although the new constructor accepts neither. The companion call
  `weather.get_ground_speed(ground_distance, altitude, true_airspeed)` is
  still valid in isolation, but the surrounding setup is dead.
  *Fix:* rewrite the example to construct
  `weather = Weather('data/weather', TemporalResolution.DAILY)` (importing
  `TemporalResolution` from `AEIC.config.weather`), and either drop the
  mission/track preamble or move it into the `get_ground_speed` call site.
  **DONE** — replaced the example block in `weather.md` with a runnable
  one that imports `TemporalResolution`, constructs
  `Weather('data/weather', file_resolution=TemporalResolution.DAILY)`,
  uses `GroundTrack.great_circle(Location, Location)` (matching
  `ground_track.py:66`), and calls `get_ground_speed(time, gt_point,
  altitude, true_airspeed)` with the current keyword names.

- **Critical** `docs/src/performance_models/performance_model_files.md:101-114`
  → `src/AEIC/performance/types.py` and `src/AEIC/data/performance/sample_performance_model.toml:23-36`
  Example TOML uses `cas_lo` / `cas_hi`. Pydantic validation rejects these;
  the canonical names (and the names emitted by `make-performance-model`)
  are `cas_low` / `cas_high`.
  *Fix:* rename in all three `[speeds.*]` blocks.
  **DONE** — renamed `cas_lo` → `cas_low` and `cas_hi` → `cas_high` in all
  three `[speeds.*]` blocks in `performance_model_files.md`.

- **Critical** `docs/src/performance_models/performance_model_files.md:124`
  → `sample_performance_model.toml:46`
  Under `[LTO_performance]`, the example uses `Foo_kN = 102.695`. The real
  key is `rated_thrust`. (`Foo_kN` is a footing-of-thrust-model curiosity
  carried over from the original BADA OPF format and was never the field
  name on the legacy performance model.)
  *Fix:* `Foo_kN` → `rated_thrust`.
  **DONE** — renamed `Foo_kN` to `rated_thrust` under `[LTO_performance]`
  in the example TOML.

- **Critical** `docs/src/performance_models/performance_model_files.md:164-181`
  → `sample_performance_model.toml:87,179,250` and
  `commands/make_performance_model.py:381-383`
  The example shows a single `[flight_performance]` table with 6 columns
  (`fuel_flow, fl, tas, rocd, mass, ???`). The current generator (#135)
  emits three tables — `[climb_flight_performance]`,
  `[cruise_flight_performance]`, `[descent_flight_performance]` — each
  with 5 columns. The TOML schema in `LegacyPerformanceModel` enforces this
  three-table layout, so the example as written will not load.
  *Fix:* replace the single `[flight_performance]` block with the three
  per-phase tables (5 columns each: `fuel_flow, fl, tas, rocd, mass`).
  Linking to `sample_performance_model.toml` would also work.
  **DONE** — replaced the single `[flight_performance]` block with three
  per-phase tables (`[climb_flight_performance]`,
  `[cruise_flight_performance]`, `[descent_flight_performance]`), each
  with the current 5-column schema and a few representative data rows
  taken from `sample_performance_model.toml`.

- **Critical** `docs/src/performance_models/performance_model_files.md:36-47`
  → `commands/make_performance_model.py:324-329`
  The CLI example invocation:
  ```shell
  aeic make-performance-model \
    --output-file tmp.toml \
    legacy \
    --apu-name 'APU 131-9' \
    --number-of-engines 2 \
    --aircraft-class narrow \
    --ptf-file /home/bada/B738__.PTF \
    --lto-source edb \
    --engine-file engines/sample_edb.xlsx \
    --engine-uid 01P11CM121
  ```
  is missing the **required** `--maximum-payload` option. Click rejects
  the call before any work is done.
  *Fix:* add `--maximum-payload 22422 \` (matching the sample model).
  **DONE** — added `--maximum-payload 22422` to the example invocation.

- **Critical** `docs/src/emission.md:94` → `src/AEIC/emissions/__init__.py:1`
  Step 8 of the workflow says "Collects together all emissions data into a
  single `Emissions` value" with cross-ref
  `<AEIC.emissions.types.Emissions>`. There is no `AEIC.emissions.types`
  module in the tree. `Emissions` is at `AEIC.emissions.emission.Emissions`
  and re-exported as `AEIC.emissions.Emissions`. Sphinx will emit a
  reference warning and the link will not resolve.
  *Fix:* change to `AEIC.emissions.Emissions` (preferred — same as
  `emission.md:18` does) or `AEIC.emissions.emission.Emissions` (matches
  `emission.md:113`).
  **DONE** — fixed the cross-ref target to `AEIC.emissions.Emissions`,
  matching the re-export in `AEIC/emissions/__init__.py:1`.

### Major — divergent prose, broken xrefs, module-path typos

- **Major** `docs/src/weather.md:11-12` → `src/AEIC/weather.py:142-189`
  Prose says `valid_time` is "sliced using `mission.departure.hour` if
  present." Under #138, slicing depends on the relationship between
  `data_resolution` and `file_resolution`: when `data_resolution` is finer
  than `file_resolution` the code uses
  `ds.sel(valid_time=time, method='nearest', tolerance=...)`; otherwise
  `valid_time` is absent or length-1 and there is no slicing at all. The
  hour-based `isel` path is gone.
  *Fix:* replace the bullet with a description tied to `data_resolution`,
  e.g.:
  > a `valid_time` coordinate is required when `data_resolution` is finer
  > than `file_resolution`, in which case it is sliced by nearest-time
  > selection within a one-hour tolerance; otherwise `valid_time` is
  > optional and length-≤1.
  **DONE** — replaced the obsolete `mission.departure.hour` bullet with a
  description tied to `data_resolution` vs `file_resolution`, matching
  the `xarray.Dataset.sel(method='nearest', tolerance=1h)` behaviour
  introduced in #138.

- **Major** `src/AEIC/trajectories/builders/base.py:157` (autodoc target;
  surfaced via `docs/src/trajectories/trajectory_builders.md:64`)
  `Builder.fly`'s docstring `Args` block reads:
  > startMass (float, optional): Starting mass of the aircraft; ...
  but the real parameter name is `starting_mass`. autodoc renders the
  docstring verbatim, so users see a kwarg name that won't work.
  *Fix:* rename `startMass` → `starting_mass` in the docstring at
  `base.py:157`. (Spotted by visual inspection; the docstring was also
  using camelCase `startMass` in the AEIC v1 codebase, suggesting it
  predates the project's snake_case convention.)
  **DONE** — renamed the docstring parameter from `startMass` to
  `starting_mass` so it matches the actual signature.

- **Major** `docs/src/mission_database.md` (entire file, no occurrence)
  → `src/AEIC/missions/__init__.py:13`, `query.py:289-319`
  PR #134 added `TimeRangeQuery` to `AEIC.missions.__all__` and added it
  to `query.py`. It returns `(min_ts, max_ts)` of departure timestamps for
  a given filter — useful for date-range UIs and used internally by
  `aeic run` for sampling progress reports. `mission_database.md` lists
  the "main classes of interest" at lines 53-69 and never mentions
  `TimeRangeQuery`. The "Queries" overview at lines 117-125 explicitly
  says "Database queries come in three flavors" — now wrong, there are
  four.
  *Fix:* add `TimeRangeQuery` to the bullet list at L53-69, update the
  "three flavors" prose, and add a `### Time-range queries` subsection
  with an `autoclass` directive parallel to the existing
  `CountQuery` block.
  **DONE** — added `TimeRangeQuery` to the "main classes of interest"
  bullet list, updated the queries overview to "four flavors", and
  added a `#### Time-range queries` subsection with a usage example
  and an `autoclass` directive parallel to `CountQuery`.

- **Major** `docs/src/mission_database.md:156` → `src/AEIC/missions/query.py:87-110`
  Prose: "These queries return results as a generator of
  `AEIC.missions.query.QueryResult` values, each of which basically
  contains all of the known information about the **flight instances**."
  After #134, `QueryResult` inherits from `Mission` and its `from_row`
  classmethod returns a *plain* `Mission` (`return Mission(...)`, not
  `return cls(...)`). What `db(query)` actually yields is therefore
  `Mission` instances, not `QueryResult`. The `QueryResult` class still
  exists as a `RESULT_CONSTRUCTION_TYPE` marker but never appears in the
  output stream.
  *Fix:* change the prose to "...generator of `AEIC.missions.Mission`
  instances, populated from the corresponding flight instance rows." Then
  drop the trailing `autoclass:: AEIC.missions.query.QueryResult` directive
  at L198-201 (it documents an effectively internal helper) or replace it
  with prose noting it is a database-driver detail.
  **DONE** — updated the prose to say "`AEIC.missions.Mission` instances",
  removed the `QueryResult` bullet from the main-classes list (it is an
  internal database-driver helper, not part of the user-facing API), and
  dropped the `autoclass:: AEIC.missions.query.QueryResult` directive.

- **Major** `docs/src/configuration.md` (entire file, no occurrence)
  → `src/AEIC/config/weather.py:9-15,67-74`
  PR #138 introduced public symbols in `AEIC.config.weather`:
  - `TemporalResolution` (CIStrEnum: `HOURLY`, `DAILY`, `MONTHLY`,
    `ANNUAL`)
  - `default_file_format(file_resolution)` helper
  - `resolution_le(a, b)` helper
  These are imported by `AEIC.weather` (`weather.py:9-13`) and are part of
  the user-facing surface (a user constructing a `Weather` outside of the
  config singleton must pass a `TemporalResolution`). `configuration.md`
  only documents `WeatherConfig` (L132-138). The new fields on
  `WeatherConfig` (`file_resolution`, `data_resolution`, `file_format`)
  are picked up by the existing `automodule` directive but have no prose
  context describing the layout/format coupling rules.
  *Fix:* in `configuration.md`, before or after the `WeatherConfig`
  autoclass, add:
  - `.. autoenum:: AEIC.config.weather.TemporalResolution`
  - a one-paragraph description of the file/data resolution coupling
    (annual files allow literal filenames, monthly files require `%m`,
    daily files require `%d` or `%j`; data resolution must be
    finer-or-equal to file resolution; `HOURLY` is not allowed as a file
    resolution).
  Optionally surface `default_file_format` and `resolution_le` via
  `autofunction` if they are intended as public utilities; otherwise
  prefix them with `_`.
  **DONE** — added a "Weather module configuration" preamble to
  `configuration.md` describing `file_resolution` / `data_resolution` /
  `file_format` and the strftime-token rules, plus `autoenum` and
  `autofunction` directives for `TemporalResolution`,
  `default_file_format`, and `resolution_le`.

- **Major** `docs/src/trajectories/trajectory_builders.md:48-51` → `base.py`
  Prose: "Documentation for the base `Builder` class is currently sparse,
  and the best reference to how to use these things is to look at the
  `LegacyBuilder` implementation." This is no longer accurate: as of the
  prior review's docstring sweep and #139's edits, `Builder`, `Options`,
  and `Context` all carry attribute-level docstrings (see `base.py:18-71`)
  and the autoclass directive at L64-66 renders a usable reference.
  *Fix:* rewrite the paragraph or remove it. Suggested replacement:
  > The base `Builder` class, the `Options` dataclass, and the `Context`
  > dataclass are documented below. `LegacyBuilder` is the most complete
  > reference implementation.

- **Major** `docs/src/trajectories/trajectory_builders.md:53-61` (warning
  block) → `src/AEIC/trajectories/trajectory.py`
  The warning block says "There needs to be a way to incrementally extend
  trajectories instead of having to specify the length of the trajectory
  up-front." This was addressed in commit 6c9f0d5 ("Extensible
  trajectories"); the current `Trajectory` class supports `append` /
  `fix` extensible mode (see also `trajectory_data.md`).
  *Fix:* remove the obsolete warning, or rewrite to flag whatever is
  *currently* unfinished (e.g., extra LTO flight phase helpers, which is
  the second clause and may still be true — verify against
  `trajectories/builders/legacy.py` before keeping that wording).

### Major — undocumented public symbols introduced after the prior review

- **Major** `AEIC.config.weather.TemporalResolution`,
  `default_file_format`, `resolution_le` — see prior bullet.

- **Major** `AEIC.missions.TimeRangeQuery` — see prior bullet.

- **Major** `AEIC.trajectories.builders.LegacyOptions` etc. → already
  exported pre-#137 but the prior review only noted the *Builder* classes
  (TASOPT/ADSB/Dymos). The `Options` companions (`TASOPTOptions`,
  `ADSBOptions`, `DymosOptions`, `LegacyOptions`) are also in
  `__all__` (`trajectories/builders/__init__.py:7-19`). `LegacyOptions`
  has docstrings (`legacy.py:24-36`); the WIP-builder option classes have
  none. They get pulled into the WIP sections of `trajectory_builders.md`
  via the package-level `automodule` only indirectly.
  *Fix:* either (a) add a note in each WIP subsection that the matching
  `*Options` class is unimplemented, or (b) drop the `*Options` classes
  from `__all__` until the corresponding builders are implemented.

### Minor

- **Minor** `src/AEIC/emissions/ei/hcco.py:10-38` (autodoc'd via
  `emission.md:162`)
  PR #139 added an optional `label: str = ''` parameter to
  `EI_HCCO`, used in a warning string at L41. The Numpydoc Parameters
  block was not updated to mention `label`.
  *Fix:* add a `label` entry under Parameters describing it as an
  optional engine/aircraft tag included in calibration warnings.

- **Minor** `docs/src/emission.md:18,113` — inconsistent xref style
  Line 18 uses `<AEIC.emissions.Emissions>` (top-level re-export); line
  113 uses `<AEIC.emissions.emission.Emissions>` (deep path). Both
  resolve, but the inconsistency makes the doc harder to maintain.
  *Fix:* settle on one style across the file. `AEIC.emissions.Emissions`
  is shorter and matches the `__all__` re-export.

- **Minor** `docs/src/oag.md:21-30` — option summary uses a slightly
  awkward double-bullet style ("`-i` / `--in-file` (required)") while
  `docs/src/cli.md` uses tables. Not wrong, but the docs would benefit
  from a consistent style for CLI reference.
  *Fix:* convert the bullet list at L26-30 to a four-column option table
  (Option | Type | Required | Description), matching `cli.md`.

- **Minor** `docs/src/cli.md:8,9,13` — the CLI table sends the user to
  three different pages (`oag.md`, `performance_model_files.md`,
  `gridding.md`) for `convert-oag-data`, `make-performance-model`, and
  `trajectories-to-grid`, but provides inline detail sections (with flag
  tables) for `run`, `merge-stores`, and `make-file-bundle`. The
  asymmetry is by design (see prior review §4) but it surprises readers.
  Consider either (a) inlining flag tables for all six commands in
  `cli.md` and trimming the per-command pages to prose, or (b) adding a
  short "Common options" line to each table-row entry so users skimming
  cli.md can predict whether the linked page covers flags.

- **Minor** `docs/src/storage.md:23-27`
  > "The public surface is re-exported at package level so that
  > everything can be imported from `AEIC.storage` directly."
  True today, but the next sentence (which would point users at the
  authoritative list) is missing. The `__all__` list contains 10 names;
  with no per-name guidance, users may not realize that
  `track_file_accesses` is a *context manager* used at CLI entry points
  and not for general code.
  *Fix:* add a single-line callout for `track_file_accesses` /
  `access_recorder`, noting that they wrap the entire `aeic run` command
  and shouldn't be entered manually inside a normal Python session.

---

## 3. Undocumented public paths

### Major

- **Major** `AEIC.missions.TimeRangeQuery` — covered in §2.
- **Major** `AEIC.config.weather.TemporalResolution`,
  `default_file_format`, `resolution_le` — covered in §2.
- **Major** `*Options` companions for WIP builders — covered in §2.

### Spot-checks (no action required)

- `AEIC.emissions.__all__` contains `compute_emissions`,
  `EMISSIONS_FIELDS`, `Emissions`. All three are referenced by
  `emission.md`. Note that `emission.md:18` already cross-refs the
  re-exported `AEIC.emissions.Emissions` path, which is the right one to
  prefer everywhere.
- `AEIC.storage.__all__` is fully documented via the `automodule`
  directive in `storage.md:32` (added by the prior review).
- `AEIC.trajectories.__all__` (`Trajectory`, `GroundTrack`,
  `BASE_FIELDSET_NAME`) is fully documented across `trajectories/*.md`.
- `AEIC.gridding`, `AEIC.performance.models`, and `AEIC.config` exports
  match their respective doc pages.

---

## 4. CLI documentation gaps

The previous review added `docs/src/cli.md` and adjusted the per-page
flag tables. Re-checking each subcommand against `commands/*.py` post-#134
and #139:

| Subcommand | Documented? | Issues |
|------------|-------------|--------|
| `convert-oag-data` | Yes (`oag.md`) | OK. Flags match `commands/convert_oag_data.py`. Stylistic improvement noted in §2 (Minor). |
| `make-performance-model` | Yes (`performance_model_files.md`) | **Critical issues** with sample TOML and missing `--maximum-payload` (§2). |
| `make-performance-model tasopt` | Yes (warning in `performance_model_files.md:24-31`) | OK. Still a `NotImplementedError` stub. |
| `run` | Yes (`cli.md:15-77`) | Flags table matches `commands/run_simulations.py`. PR #139 did not change the click signature. |
| `merge-stores` | Yes (`cli.md:79-109`) | OK. |
| `make-file-bundle` | Yes (`cli.md:111-137`) | OK. |
| `trajectories-to-grid` | Yes (`gridding.md`) | Not re-audited in this pass; the prior review noted the table matches code. |

No newly-introduced subcommands or top-level flags were detected
between commits 859bf8e and HEAD.

---

## 5. Cross-file duplication / consistency

- **Minor** `docs/src/cli.md:8,9,13` table style asymmetry — see §2.

- **Minor** `docs/src/oag.md:26-30` vs `docs/src/cli.md:26-36` — see §2.

No new duplications were introduced since the prior review. The
`data-dictionary.md` / `CONTRIBUTING.mb` / `CHANGELOG.md` issues from
that review have been resolved.

---

## 6. Other finds

- **Minor** Sphinx coverage builder still reports
  `CIBaseModel.normalize_keys` as undocumented (Appendix A). This was
  already in the prior review's Appendix B and was deferred.

- **Minor** `docs/src/trajectories/trajectory_builders.md:55` — warning
  bullet about needing a way to "incrementally extend trajectories" is
  obsolete (see §2 Major).

- **Minor** `docs/src/configuration.md:135-138` — the `WeatherConfig`
  autoclass directive uses `:exclude-members: model_config`, which is the
  right pattern, but `effective_data_resolution` and
  `effective_file_format` (computed properties added in #138) are
  re-exposed via `automodule`. Worth verifying in the rendered output
  that they show up under the `WeatherConfig` reference. If they don't,
  add `:members:` explicitly listing them.

---

## Appendix A — Sphinx coverage builder output

Generated with `uv run sphinx-build -b coverage docs docs/_build/review2-coverage`
on 2026-04-28. Full output at `docs/_build/review2-coverage/python.txt`.

Summary: 17 modules inspected, 97.06% coverage, 1 undocumented item.

```
AEIC.utils.models
-----------------
 * CIBaseModel -- missing methods:
   - normalize_keys
```

The coverage builder only inspects modules reached by an autodoc
directive, so this is a narrow signal. Modules that have no autodoc
reference (and therefore aren't measured) include
`AEIC.config.weather` helpers (covered above), the `AEIC.commands.*`
modules (intentional — they are CLI entry points), and parts of
`AEIC.utils.progress`. These are the real coverage gaps.

The previous review's Appendix B reported `LegacyBuilder.fly_climb` and
`LegacyBuilder.fly_descent` as undocumented. As of HEAD both methods
have docstrings (`legacy.py:244-248,257-263`). The
`AEIC.trajectories.builders.legacy` module is now at 100% coverage.
`AEIC.utils.airports` is also clean (the `:exclude-members:` directive
that hid `AirportsData` / `CountriesData` was removed in the prior
review).

---

## Appendix B — Verification notes

A few claims surfaced by exploratory passes were investigated and
**ruled out** before being included above:

1. *Did PR #139 remove docstrings from `fly_climb` / `fly_descent`?* —
   No. Both methods carry docstrings at `legacy.py:244-248,257-263`,
   matching what the prior review added. `fly_cruise` is also
   documented at `legacy.py:402-406`. The Sphinx coverage builder
   confirms `AEIC.trajectories.builders.legacy` is at 100%.
2. *Does the `aeic make-performance-model legacy` example pass an
   invalid `--aircraft-name` flag?* — No. The example passes
   `--apu-name 'APU 131-9'`, which is a real (optional) flag at
   `make_performance_model.py:330-334`. There is no `--aircraft-name`
   flag (the aircraft name is derived from the PTF file at
   `make_performance_model.py:372`).
3. *Does `Database.__call__` return `QueryResult` objects?* — Not in
   practice: `QueryResult.from_row` returns `Mission(...)` rather than
   `cls(...)` (see `query.py:91-110`), so callers receive `Mission`
   instances. Documented as a Major finding in §2.
