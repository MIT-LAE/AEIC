# AEIC Test Quality Review

Multi-session comprehensive audit of all test files under `tests/`.
**Status: complete** (all six phases ticked 2026-04-15). Master plan:
`/home/iross/.claude/plans/curious-strolling-raccoon.md`. Per-session plan
files live alongside it.

**Scope:** 21 test files in `tests/`, 122 collected tests (1 skipped manual
long-case), 115 findings in total. Coverage (scoped invocation):
`AEIC` package 76 %.

## Document map

- [Findings summary](#findings-summary) — totals by severity × category, by phase.
- [Headline findings](#headline-findings) — all 11 High-severity items, cross-linked.
- [Prioritized remediation](#prioritized-remediation) — quick-wins vs larger lifts.
- [Rubric](#rubric) — categories, severity, SUSPICIOUS-DATA triggers.
- [Preamble](#preamble) — runnability notes, coverage paths, notebook index.
- Per-phase detail: [Phase 1 Plumbing](#phase-1--plumbing),
  [Phase 2 Mission DB](#phase-2--mission-db),
  [Phase 3 Performance](#phase-3--performance),
  [Phase 4 Emissions](#phase-4--emissions),
  [Phase 5 Trajectories](#phase-5--trajectories),
  [Phase 6 Verification & golden](#phase-6--verification--golden).
- [Bonus: SUT observations](#bonus-sut-observations-incidental) — incidental
  non-test issues surfaced during the review.

## Phases (all done)

- [x] Phase 1 — Plumbing (`test_config`, `test_dimensions`, `test_storage`, `test_emissions_storage`, `conftest`, `subproc`)
- [x] Phase 2 — Mission DB (`test_mission_db`, `test_mission_db_creation`)
- [x] Phase 3 — Performance (`test_performance_model`, `test_performance_table`, `test_model_utilities`, `test_thrust_modes`)
- [x] Phase 4 — Emissions (`test_emissions`, `test_emission_functions`, `test_edb`)
- [x] Phase 5 — Trajectories (`test_trajectories`, `test_trajectory_simulation`, `test_weather`)
- [x] Phase 6 — Verification & golden (`test_legacy_verification`, `test_matlab_verification`, `test_golden`)

## Findings summary

**Total findings: 115** — High 11 · Medium 52 · Low 52.

Severity × category matrix:

| Category          | High | Medium | Low | Total |
| ----------------- | ---: | -----: | --: | ----: |
| COVERAGE-GAP      |    1 |     17 |  16 |    34 |
| HYGIENE           |    2 |      4 |  20 |    26 |
| WEAK-ASSERTION    |    2 |     14 |  10 |    26 |
| SUSPICIOUS-DATA   |    3 |      9 |   1 |    13 |
| LOGIC-ERROR       |    3 |      5 |   3 |    11 |
| ISOLATION         |    0 |      2 |   1 |     3 |
| FLAKY-RISK        |    0 |      1 |   1 |     2 |
| **Total**         | **11** | **52** | **52** | **115** |

Distribution by phase:

| Phase                                | High | Medium | Low | Total |
| ------------------------------------ | ---: | -----: | --: | ----: |
| Phase 1 — Plumbing                   |    1 |      6 |  14 |    21 |
| Phase 2 — Mission DB                 |    1 |      8 |   8 |    17 |
| Phase 3 — Performance                |    1 |     10 |  10 |    21 |
| Phase 4 — Emissions                  |    4 |     13 |   9 |    26 |
| Phase 5 — Trajectories               |    2 |      9 |   6 |    17 |
| Phase 6 — Verification & golden      |    2 |      6 |   5 |    13 |
| **Total**                            | **11** | **52** | **52** | **115** |

Observations:

- `COVERAGE-GAP` (34) and `HYGIENE` (26) dominate (52 % of all findings).
  Coverage gaps cluster in Phase 3 (13 — performance validators and the
  untested `PerformanceTable.__post_init__` surface) and are spread
  evenly across Phases 1, 2, 4, 5 (4–5 each). Hygiene clusters in
  Phase 1 (7) and Phase 5 (6) — driven largely by missing
  `@pytest.mark.forked` decorators on netCDF-touching tests.
- `SUSPICIOUS-DATA` is concentrated in Phase 4 (9 of 13) — the notebook
  citation gap (trigger #4) fires across BFFM2, MEEM, SCOPE 11, NOx
  speciation, and HCCO because none of Sections 2–6 carry explicit
  "rounded results" blocks. Two more SUSPICIOUS-DATA findings each
  sit in Phase 2 (OAG extract) and Phase 5 / Phase 6 (weather fixture,
  matlab reference data).
- `LOGIC-ERROR` severity is top-heavy: 3 of 11 are **High** — tests that
  pass while the SUT is (or could be) wrong.
- `ISOLATION` and `FLAKY-RISK` are small tails (3 and 2 findings) but
  each includes a non-trivial Medium finding (`Config` singleton
  leakage risk in `test_legacy_verification.py` if it were ever
  collected; unseeded RNG in `test_trajectory_simulation_basic`).

## Headline findings

The 11 **High**-severity items. Each links into the phase section for
the full finding body.

### High [LOGIC-ERROR] — test passes while SUT could be wrong

1. **`test_HC_outputs` — no-op `.all` assertion** *[DONE]*
   ([Phase 4 · `test_emission_functions.py`](#tests-test_emission_functionspy-24-tests-across-7-classes)).
   `test_emission_functions.py:71` reads
   `assert np.isclose(result, out_result).all` — missing `()`. The
   assertion tests the truthiness of the bound method object, which is
   always `True`. **Most serious defect in the audit.**
2. **`test_trajectory_simulation_basic` — writes fields, never reads them** *[DONE]*
   ([Phase 5 · `test_trajectory_simulation.py`](#tests-test_trajectory_simulationpy-7-tests)).
   Adds `test_field1` / `test_field2` to a `TrajectoryStore`, reopens,
   and asserts only `len(ts_loaded) == len(ts)` with a `# TODO` comment
   acknowledging the real invariant is unverified.
3. **`test_legacy_verification.py` — zero collected tests** *[DONE]*
   ([Phase 6](#tests-test_legacy_verificationpy-0-tests--script)).
   All logic is inside `main()` guarded by `if __name__ == '__main__'`;
   `pytest --collect-only` returns zero items from the file.

### High [HYGIENE] — structural issues with broad blast radius

4. **`test_emissions_storage.py` — missing `@pytest.mark.forked`** *[DONE]*
   ([Phase 1](#tests-test_emissions_storagepy-2-tests)). The file's
   netCDF-touching tests violate the project's documented isolation
   policy. Success is order-dependent — a heisen-failure waiting for a
   future test-ordering change.
5. **`test_legacy_verification.py:50` — hardcoded user-specific path** *[DONE]*
   ([Phase 6](#tests-test_legacy_verificationpy-0-tests--script)).
   `DEFAULT_TEST_DATA_DIR = "/home/aditeya/AEIC/tests/data/legacy_verification/"`
   — wrong user, wrong path segment (real repo uses
   `tests/data/verification/legacy`). Running the script directly on a
   fresh checkout crashes.

### High [SUSPICIOUS-DATA] — expected values have SUT-self-generated provenance

6. **`test_atmospheric_state_and_sls_flow_shapes` — self-referential inputs** *[DONE]*
   ([Phase 4 · `test_emissions.py`](#tests-test_emissionspy-10-tests)).
   Atmospheric expected values were derived from `AtmosphericState`, the
   very SUT under test (per notebook cell 11).
7. **`test_get_gse_emissions_matches_reference_profile` — snapshot of SUT constants** *[DONE]*
   ([Phase 4 · `test_emissions.py`](#tests-test_emissionspy-10-tests)).
   Hardcoded `expected` dict literally matches `_gse_nominal_profile()`
   defaults; no external/notebook reference.
8. **`test_compute_ground_speed` — 15-digit bare literal, re-adjusted when SUT changed** *[DONE]*
   ([Phase 5 · `test_weather.py`](#tests-test_weatherpy-2-tests)).
   `gs == pytest.approx(191.02855126751604, rel=1e-4)` with inline
   comment admitting the tolerance was relaxed "because we changed the
   pressure level calculation slightly" — smoking-gun
   self-generated-expected-value pattern.

### High [WEAK-ASSERTION] — assertion passes even when implementation is wrong

9. **`test_oag_conversion` — `COUNT(*) == 7` on the entire ingestion pipeline** *[DONE]*
   ([Phase 2 · `test_mission_db_creation.py`](#tests-test_mission_db_creationpy-3-tests)).
   Every downstream mission-DB test's provenance leads back through
   this ingestion; the only check is a row count.
10. **`test_emit_matches_expected_indices_and_pointwise` — helper calls the SUT** *[DONE]*
    ([Phase 4 · `test_emissions.py`](#tests-test_emissionspy-10-tests)).
    `_expected_trajectory_indices` invokes the same BFFM2 / EI_HCCO /
    SLS functions that `compute_emissions` uses — a consistency check
    mis-labeled as a correctness check. The notebook Section 2 (BFFM2)
    exists precisely to break this self-reference.

### High [COVERAGE-GAP] — important validator unexercised

11. **`PerformanceTable.__post_init__` — six structural checks untested**
    ([Phase 3 · `test_performance_table.py`](#tests-test_performance_tablepy-3-tests)).
    `check_coverage × 3` and `check_fl_only × 5` over climb / cruise /
    descent sub-tables — the heart of the BADA-shape validation. Every
    `raise ValueError` branch is unexercised; orphan
    `bad_performance_model_{1,2,3}.toml` fixtures reference zero tests. *[DONE]*

## Prioritized remediation

Actionable work, grouped by effort.

### Tier 1 — quick wins (one-line fixes)

All strictly local, no new fixtures, no new notebook work. High ROI per
minute.

- **`test_emission_functions.py:71`:** add `()` to `.all` — fix #1
  headline. (`np.testing.assert_allclose(result, out_result)` is
  preferable since it reports the diff on failure.) *[DONE]*
- **`test_mission_db.py:150`:** change `'random()' in sql` to
  `'det_random()' in sql` — pins the reproducibility contract
  ([Phase 2](#tests-test_mission_dbpy-3-tests)). *[DONE]*
- **Forked-marker sweep:** add `@pytest.mark.forked` to every
  netCDF-touching test that currently lacks it: *[DONE]*
  - `test_emissions_storage`, `test_separate_emissions` (Phase 1, High).
  - `test_trajectory_simulation_basic`,
    `test_trajectory_simulation_outside_weather_domain`,
    `test_trajectory_simulation_weather` (Phase 5, Medium each).
  - `test_compute_ground_speed` (Phase 5, Medium).
  - `test_trajectory_simulation_golden` (Phase 6, Low).
- **Delete / move `tests/test_legacy_verification.py`** to
  `scripts/legacy_verification_plot.py` (or delete — `test_matlab_verification`
  is the successor). Closes two High findings at once. *[DONE]* (deleted;
  `scripts/run_legacy_verification.py` already exists as the proper successor)
- **`test_matlab_verification.py:13–24`:** drop `'flight_time'` from
  `TRAJ_FIELDS` — it's the independent variable, so MAPE ≈ 0 by
  construction. *[DONE]*
- **`test_mission_db_creation.py:34`:** normalize `0b0000000` →
  `0b00000000` (cosmetic but consistent). *[DONE]*
- **`test_golden.py:22–23`:** replace the `if len(failed) > 0: raise`
  pattern with `assert not failed, …`. *[DONE]*

### Tier 2 — small docs / provenance additions

Each fixes one or more SUSPICIOUS-DATA findings without touching test
code. Writes a README or notebook cell.

- **Notebook Sections 2–6:** add explicit "rounded results for use in
  test" cells to BFFM2, NOx speciation, HCCO, SCOPE 11, MEEM. This
  clears trigger #4 on every Phase-4 SUSPICIOUS-DATA downgrade (6 of
  the 13 SUSPICIOUS-DATA findings across the whole review). *[DONE]*
  Also fixed SCOPE 11 Q-coefficient typo (0.766 → 0.776) that caused
  the notebook values to diverge from the test expected values.
- **`tests/data/` READMEs:** every subdirectory lacks a provenance file.
  Priority order by finding pressure: *[DONE]*
  1. `tests/data/verification/legacy/README.md` — naming the
     AEIC-MATLAB commit/revision, mission inputs, and the
     `matlab-output-orig/` → `matlab-output/` transformation. Clears
     the Medium SUSPICIOUS-DATA on `test_matlab_verification`.
  2. `tests/data/weather/README.md` — source dataset (ERA5?), spatial
     subset, download date. Context for the Phase-5 High SUSPICIOUS-DATA.
  3. `tests/data/oag/README.md` — how `2019-extract.csv` was produced,
     the 8 expected flights. Clears Medium SUSPICIOUS-DATA on
     `test_oag_conversion`.
  4. `tests/data/missions/oag-2019-test-subset.README.md` — the
     verification SQL used to derive the 1197 / 99 / 36 / 307 / 13
     expected row counts.
  5. `src/AEIC/data/engines/README.md` — ICAO EDB revision of
     `sample_edb.xlsx`.
- **Weather fixture typo fix:** `azmiuth` / `azmith` → `azimuth` in
  `src/AEIC/weather.py` docstring (incidental from Phase 5 review). *[DONE]*

### Tier 3 — larger lifts (new test bodies)

Each is a self-contained PR.

- **`PerformanceTable.__post_init__` validator tests** (Phase 3
  headline #11). Parametrize the six `raise ValueError` branches; use
  the orphan `bad_performance_model_{1,2,3}.toml` fixtures — this is
  what they were written for.
- **`test_oag_conversion` — content assertions** (Phase 2 headline #9).
  Assert at least one row's full content matches the source CSV, and
  that `schedules` is populated. Add a second CSV fixture exercising
  each warning category (`UNKNOWN_AIRPORT`, `TIME_MISORDERING`,
  `SUSPICIOUS_DISTANCE`, `ZERO_DISTANCE`).
- **`test_trajectory_simulation_basic` — finish the TODO** (Phase 5
  headline #2). After `test_field1` / `test_field2` round-trip through
  the store, assert the loaded values equal the written values. Seed
  the RNG while there. *[DONE]*
- **Replace the self-referential `_expected_trajectory_indices` helper**
  (Phase 4 headline #10) with hard-coded arrays cited to specific
  notebook cells. Requires Tier 2's notebook rounded-results blocks to
  exist first.
- **`test_golden.py` — scalar sanity checks + explicit tolerances**
  (Phase 6 headline). Rename to
  `test_trajectory_simulation_matches_golden_snapshot` to signal
  snapshot semantics; add per-field tolerances; add at least one
  unit-convention tripwire (`assert 50_000 < traj.aircraft_mass[0] < 90_000`
  for a 738). *[DONE]* (rename + tripwire; per-field tolerance
  parametrization deferred — requires SUT changes to
  `Container.approx_eq`).
- **`Config.escape()` context manager test** (Phase 1 Medium
  COVERAGE-GAP). This is the reproducibility-replay path; a broken
  `escape()` would silently corrupt snapshot loads. See also the Bonus
  decorator-stacking observation. *[DONE]*
- **`Database.set_random_seed()` determinism test** (Phase 2 Medium
  COVERAGE-GAP). Open the test DB twice with the same seed, collect
  flight IDs, assert equal. *[DONE]*

### Tier 4 — deferred / out of scope

- Promoting `test_create_reopen_large` from `skip` to a proper slow
  marker — cosmetic; functional behavior unchanged.
- `TASOPTBuilder` / `ADSBBuilder` / `DymosBuilder` `NotImplementedError`
  contract test — worth having but low priority while the stubs don't
  move. *[DONE]* Pulled forward as part of the Phase 5 medium sweep.
- SOx test tolerance tightening (1 % → 1e-6) — the current tolerance is
  demonstrably adequate for the analytical identity under test.

## Rubric

**Categories.**
- `SUSPICIOUS-DATA` — hard-coded expected values with unclear /
  self-generated provenance and no independent reference in
  `notebooks/test-cases.ipynb`.
- `LOGIC-ERROR` — test passes but doesn't verify what its name/docstring claims.
- `WEAK-ASSERTION` — tolerance so loose a broken impl would pass.
- `COVERAGE-GAP` — important branch in *natural scope* of the file is untested.
- `FLAKY-RISK` — non-determinism, time/order/FS dependence, unseeded RNG.
- `ISOLATION` — test depends on / leaks shared state (esp. `Config` singleton, files in CWD).
- `HYGIENE` — naming, commented-out asserts, misleading docstrings,
  missing `pytest.mark.forked` on netCDF-touching tests.

**Severity.** High (passes while SUT is wrong) / Medium (would miss real
regressions) / Low (hygiene).

**SUSPICIOUS-DATA triggers (any one ⇒ flag).**
1. Bare numeric literals in `assert == X` / `assert_allclose(..., X)`
   with no source comment.
2. Long arrays (≥ ~5 entries) of expected values without provenance
   comment, or with self-referential "generated from running …" comments.
3. Expected values loaded from `tests/data/verification/` with no
   README/provenance metadata. (Note: `tests/data/golden/` is **cleared**
   — `scripts/make_golden_test_data.py` documents its provenance, even
   though it's a frozen SUT snapshot.)
4. SUT under test corresponds to a notebook section (SLS, BFFM2, NOx
   speciation, HCCO, SCOPE 11, MEEM) but expected values don't match the
   notebook's "rounded results" block and the test doesn't cite the
   notebook.

**Mitigating evidence (any ⇒ downgrade or clear).**
- Comment naming a paper, equation, or external authoritative source.
- Reference to a specific notebook cell or "rounded results" block.
- Analytical / first-principles inline computation in the test.
- Explicit cross-implementation consistency comparison (flag as
  `WEAK-ASSERTION` only if labeled as correctness check).

**Severity within `SUSPICIOUS-DATA`:** High for science-critical SUT
(emissions index, performance, trajectory state), Medium for
plumbing-with-numbers.

**Finding format.**
`- **[Severity][CATEGORY]** test_name — description. *Suggested fix:* one line.`

## Preamble

- Pytest collect output: `/tmp/aeic-collect.txt` (scoped to `tests/`)
- Coverage term report: `/tmp/aeic-cov-term.txt`
- Coverage XML: `/tmp/aeic-cov.xml`
- Notebook index: [`test-quality-notebook-index.md`](./test-quality-notebook-index.md)

**Suite runnability notes.**
- Running `uv run pytest` from the repo root without arguments **fails
  collection** due to the two untracked worktree directories
  `pretty-performance-model-toml/` and `trajectories-to-grid/`, each of
  which ships its own `tests/conftest.py`. Pytest aborts with
  `ImportPathMismatchError: ('tests.conftest', …)`. Workaround used for
  this audit: `uv run pytest tests/`. Long term: either add those dirs
  to `pyproject.toml`'s `[tool.pytest.ini_options] testpaths` / `norecursedirs`,
  or move / rename their conftests. (Incidental: not a test-quality
  finding per se.)
- With the scoped invocation, **122 tests pass, 1 skipped (manual
  long-case), 0 failures**, in ~27 s. Overall `AEIC` package coverage:
  **76 %**.
- Coverage for Phase 1 targets:
  `config/core.py` 97 % · `storage/dimensions.py` 100 % ·
  `storage/container.py` 88 % · `storage/field_sets.py` 85 % ·
  `storage/reproducibility.py` 74 %.

## Phase 1 — Plumbing

### `tests/test_config.py` (9 tests)

All tests override the global autouse `default_config` fixture with a
local one that skips initial load and only resets the singleton on
teardown — each test explicitly calls `Config.load(...)` itself. The
override is deliberate; documented by the inline comment.

- **[Low][WEAK-ASSERTION]** `test_load_default_config` — asserts
  `config.performance_model is not None` and
  `config.weather.weather_data_dir is not None`. Both are required
  Pydantic fields, so the load would have raised before reaching the
  assertions; these checks are tautological. *Suggested fix:* assert
  concrete default values (e.g. a specific file name from
  `default_config.toml`) so that overlay-logic regressions are caught.
  *[DONE]*
- **[Low][HYGIENE]** `test_load_default_config` + `test_get_config` —
  near-duplicate smoke tests (one accesses via the `config` proxy, one
  via `Config.get()`). Both assert the same two fields are non-None.
  *Suggested fix:* parametrize, or merge and add a single explicit
  proxy-vs-classmethod equivalence assertion. *[DONE]* `test_get_config`
  is now `test_proxy_and_get_match`, asserting field-by-field equality
  between `Config.get()` and the proxy rather than restating the
  defaults already pinned in `test_load_default_config`.
- **[Medium][COVERAGE-GAP]** `Config.escape()` context manager — used
  for bypassing the singleton when loading config snapshots from
  trajectory-store reproducibility data — is not exercised here or
  anywhere in Phase 1. Coverage confirms the "slack" branch at
  `src/AEIC/config/core.py:136` is dead (as is `289` in
  `ConfigProxy.__setattr__`). Because this is the reload path used by
  reproducibility snapshots, a broken `escape()` would silently corrupt
  replay. *Suggested fix:* add a test that nests `Config.load()` inside
  `with Config.escape():` and verifies the inner load does not raise
  `RuntimeError` and does not overwrite the outer singleton. *[DONE]*
  See `tests/test_config.py::test_escape_allows_re_validation_without_overwriting_singleton`.
- **[Low][COVERAGE-GAP]** `default_data_file_location` with a path
  absent from both overrides and `src/AEIC/data/` (raise at
  `core.py:153`) is untested. *Suggested fix:* add a one-liner
  `pytest.raises(FileNotFoundError)` case. *[DONE]* See
  `tests/test_config.py::test_default_data_file_location_missing_raises`.
- **[Low][HYGIENE]** Local `default_config` fixture in this file
  (`test_config.py:13–16`) takes no `request` argument, so a
  `@pytest.mark.config_updates(...)` applied to a test in this file
  would be silently discarded (no error, no effect). Not currently
  exercised, but a footgun. *Suggested fix:* either accept `request` and
  assert no marker is present, or drop the override entirely for tests
  that already do their own `Config.load()` and let the global fixture
  run. *[DONE]* The override now accepts `request` and asserts no
  `config_updates` marker is attached, surfacing the silent-drop case
  rather than letting it pass.

### `tests/test_dimensions.py` (7 tests)

`Dimension` and `Dimensions` coverage is nominally 100 %, but several
public methods are not *exercised* (100 % is from being touched by
other tests via imports / instantiation paths).

- **[Medium][COVERAGE-GAP]** `Dimensions.netcdf` property — untested in
  this file. This property controls how dimensions are serialized to
  NetCDF files; a bug here would silently reorder axes in output.
  *Suggested fix:* add an explicit test asserting the tuple contents
  and ordering (including the `Dimension.POINT` exclusion). *[DONE]*
  See `tests/test_dimensions.py::test_dimensions_netcdf`.
- **[Low][COVERAGE-GAP]** `Dimensions.remove`, `Dimensions.from_dim_names`,
  and `Dimension.dim_name` property are untested in this file.
  *Suggested fix:* add targeted single-assertion tests. *[DONE]* See
  `test_dimension_dim_name`, `test_dimensions_remove`, and
  `test_dimensions_from_dim_names`.
- **[Low][HYGIENE]** `test_dimensions_equality` mixes equality semantics
  with `str` / `repr` formatting assertions. Low-impact but splits are
  clearer. *Suggested fix:* extract `test_dimensions_str_repr`. *[DONE]*

### `tests/test_storage.py` (~30 tests)

Good use of `@pytest.mark.forked` on netCDF-touching tests. Two
mergeability tests (`test_pattern_merging`,
`test_merging_with_associated_files`) use the alternative pattern of
delegating to `subproc.run_in_subprocess`; both patterns coexist in the
file.

- **[Medium][ISOLATION]** `test_file_access_recorder` — calls
  `sqlite3.connect('tmp.sqlite')` with a bare relative path, which
  creates `tmp.sqlite` in the *current working directory* (i.e. repo
  root when pytest is run there). The file leaks across test runs.
  *Suggested fix:* use `tmp_path / 'tmp.sqlite'` and update the expected
  `access_recorder.paths` accordingly; drop the redundant `safe_*`
  wrappers (see next item). *[DONE]*
- **[Low][HYGIENE]** `test_file_access_recorder` — `safe_sqlite3_connect`
  wraps `sqlite3.connect` in `try/except FileNotFoundError`, but
  `sqlite3.connect` *creates* the file if it does not exist and never
  raises `FileNotFoundError`, so the `except` branch is dead. Similarly
  `safe_open('file1.nc')` only catches `FileNotFoundError` from the
  `open(f, 'r')` default mode. *Suggested fix:* drop the safety wrappers
  entirely; the recorder's event hook fires regardless. *[DONE]*
- **[Medium][LOGIC-ERROR]** `test_multi_threading` — asserts a worker
  thread sets `result == 'FAILED'`, i.e. expects multi-threaded NetCDF
  writes to fail. The test swallows *any* exception via bare `except
  Exception:`, so a failure for an unrelated reason (e.g. `tmp_path`
  creation error, import failure inside the worker) would also make the
  test pass. No docstring explains the invariant. *Suggested fix:* catch
  a specific expected exception type (document which NetCDF-layer error
  is expected), and assert the message / class, not a literal string
  `'FAILED'`. *[DONE]* *Actual remediation:* the guard the test was
  exercising turned out to be `TrajectoryStore.active_in_thread`
  (store.py:378), not a NetCDF-layer error — assertion now pins the
  `RuntimeError` class and matches the guard message substring.
- **[Low][LOGIC-ERROR]** `test_init_checking` — body ends with
  `# TODO: MORE HERE...` (line 213). Incomplete test. *Suggested fix:*
  finish enumerating the guard-clause cases or remove the TODO. *[DONE]*
  Now exercises all four global-attribute fields (title/comment/history/
  source), the READ-only flags (`override`, `force_fieldset_matches`),
  and the CREATE-mode `base_file=None` + `associated_files`
  inconsistency, with regex-pinned messages.
- **[Low][FLAKY-RISK]** `test_indexing` and `test_merged_store_indexing`
  use `random.sample` without seeding, and `make_test_trajectory` uses
  `np.random.*` without seeding. Every run draws fresh data. Assertions
  are structural so they don't actually flake today, but non-determinism
  makes debugging harder and silently weakens any future value-sensitive
  assertion added later. *Suggested fix:* seed `random` and
  `np.random.default_rng(...)` at the top of each test (or via a
  fixture), and pass the generator into `make_test_trajectory`.
  *[DONE]* *Actual remediation:* seeded the global `random` and
  `np.random` RNGs at the top of each test rather than threading a
  generator through `make_test_trajectory` — the helper relies on
  global numpy state across many call sites and reworking its signature
  was out of scope for a [Low].
- **[Low][WEAK-ASSERTION]** Across the `test_create_*` /
  `test_extra_fields_in_*` / `test_merging_*` family, the post-reload
  assertions exclusively check shapes, lengths, `fieldsets`
  memberships, and `hasattr(...)`. They never verify that reloaded
  numeric field values equal the ones written. A bug that silently
  permuted axes or rounded values on serialize would pass. *Suggested
  fix:* in at least one representative test per write-mode, assert
  `np.array_equal(loaded.fuel_flow, original.fuel_flow)` for a known
  trajectory (or use `Container.__eq__`, which already implements the
  comparison). *[DONE]* `test_create_reopen` now captures the originals
  before writing and asserts field-by-field equality via the existing
  `_assert_trajectories_equal` helper. The `iter_range` / `iter_flight_ids`
  family already does this, so the create-and-reopen path was the
  notable gap.
- **[Low][HYGIENE]** `test_create_reopen_large` is
  `@pytest.mark.skip(reason='long test case, enable manually')`. Fine
  as a gated scaling test; consider `@pytest.mark.slow` + a CLI flag so
  it's rediscoverable, rather than permanent skip. *Suggested fix:*
  switch to a custom marker and document in `pyproject.toml`. *[DONE]*
  *Actual remediation:* registered `slow` and a `--run-slow` flag in
  `tests/conftest.py` (mirroring the existing `config_updates` marker
  pattern there) rather than `pyproject.toml`, since `pytest_configure`
  / `pytest_addoption` is where this project already declares markers.

### `tests/test_emissions_storage.py` (2 tests)

This file is the highest-impact Phase 1 finding. It is the only
storage-adjacent file whose netCDF-touching tests are **not** forked,
contradicting the project's documented practice (see auto-memory:
"NetCDF tests must use `pytest.mark.forked` — netCDF4/HDF5 state
leakage requires subprocess isolation").

- **[High][HYGIENE]** `test_emissions_storage`, `test_separate_emissions`
  — both call `TrajectoryStore.create` / `.open` and touch NetCDF but
  are **not** decorated with `@pytest.mark.forked` and do **not** use
  `subproc.run_in_subprocess`. They currently pass, but given
  documented HDF5 state leakage, their success is position-dependent
  within the run — a future test ordering change could cause
  heisen-failures. *Suggested fix:* add `@pytest.mark.forked` to both
  (or wrap bodies in `run_in_subprocess`, matching the convention used
  elsewhere in `tests/test_storage.py`).
- **[Medium][WEAK-ASSERTION]** `test_emissions_storage` — after reload,
  asserts `hasattr(traj, 'total_emissions')` but never compares the
  loaded emissions to the computed ones. *Suggested fix:* capture
  `original_emissions = [compute_emissions(...) for mis in ...]` before
  writing, and after reopening assert
  `ts_loaded[i].total_emissions == original_emissions[i].total_emissions`
  for each species. *[DONE]*
- **[Medium][WEAK-ASSERTION]** `test_separate_emissions` — checks types
  (`isinstance(..., SpeciesValues)`, `isinstance(..., np.ndarray)`) but
  not values. The branch under test is the associated-file round trip,
  which is the exact place serialization bugs could silently drop or
  reorder species. *Suggested fix:* as above, compare at least one
  representative species array via `np.allclose`. *[DONE]* Both the
  scalar `total_emissions` and the per-segment `trajectory_indices`
  arrays are now compared species-by-species against an in-memory
  baseline, with species-key set equality enforced.

### `tests/conftest.py` (context)

- **[Low][HYGIENE]** `os.environ['AEIC_PATH'] = str(TEST_DATA_DIR)` at
  module import time (line 17) is a side effect that leaks into any
  process importing this file — including subprocesses spawned by
  `run_in_subprocess`, which is probably *intended*, but the coupling
  is implicit. *Suggested fix:* add a one-line comment documenting the
  cross-subprocess expectation. *[DONE]*
- **[Low][ISOLATION]** `default_config` fixture at lines 36–65 is
  `autouse=True` and runs `Config.load(**config_data, …)` unconditionally.
  Any test that wants to skip it must locally redefine a fixture of the
  same name (as `test_config.py` does). Works, but the override-via-
  shadowing contract is not documented. *Suggested fix:* add a
  fixture-level docstring noting how to opt out. *[DONE]*

### `tests/subproc.py` (context)

- No findings. `run_in_subprocess` is a small, correct helper; the
  docstring explicitly states the rationale (netCDF4 file-handle leak
  across many open/close events).
- **[Low][COVERAGE-GAP]** No direct test of `run_in_subprocess` itself.
  It's exercised transitively by several `test_storage.py` tests, but a
  broken helper would fail those tests noisily, so the indirect
  coverage is adequate. *Suggested fix:* none required. *[DONE]* Closed
  with no code change — the finding's own recommendation was that
  indirect coverage is sufficient; this entry is recorded so the Low
  sweep accounting matches the original audit total.

## Phase 2 — Mission DB

**Notebook coverage:** `notebooks/test-cases.ipynb` has no Mission DB
section by design (the notebook is emissions-focused). SUSPICIOUS-DATA
trigger #4 therefore does not apply to any finding in this phase. No
new entries for `test-quality-notebook-index.md`.

**Phase 2 coverage for Mission DB SUT modules (per `/tmp/aeic-cov.xml`):**
`missions/database.py` 79 % · `missions/query.py` 89 % ·
`missions/filter.py` 93 % · `missions/oag.py` 85 % ·
`missions/writable_database.py` 70 % · `missions/mission.py` 42 %.
The low `mission.py` number is a COVERAGE-GAP for this phase — noted
inline under `test_mission_db.py` findings.

### `tests/test_mission_db.py` (3 tests)

Three monolithic test functions — one each for `Filter.to_sql()`,
`Query.to_sql()`, and end-to-end SQL execution against a pre-built
SQLite test DB. All tests are of the "snapshot the generated SQL and
param list" flavour, which is a reasonable contract for query-builder
stability but leaves the execution path thinly tested.

- **[Medium][LOGIC-ERROR]** `test_query` (line 150) — asserts
  `'random()' in sql` to confirm that sampling queries use a random
  function. The SUT uses `det_random()` (deterministic, per CLAUDE.md
  reproducibility invariant), but the substring check passes equally
  for non-deterministic `random()` — i.e. a regression that drops
  determinism would go undetected by this test, despite its placement
  in the sampling branch. *Suggested fix:* change the assertion to
  `'det_random()' in sql` to pin the reproducibility contract.
- **[Medium][WEAK-ASSERTION]** `test_query` (lines 159–163) — the
  `FrequentFlightQuery(filter=..., limit=10)` branch asserts only
  `'GROUP BY od_pair' in sql` and `params == ['US']`. The `limit=10`
  argument is never verified in either the SQL (`LIMIT ?` clause) or
  the params list. *Suggested fix:* assert `'LIMIT ?' in sql` and that
  `10` appears in `params`. *[DONE]* *Actual remediation:* the SUT
  inlines the limit value into the SQL string (`query.py:253`) rather
  than parameterizing it, so the assertion is `'LIMIT 10' in sql`
  (params unchanged) — the inline form is the contract being pinned.
- **[Medium][LOGIC-ERROR]** `test_query_result` every-nth block
  (lines 236–248) — the inter-flight gap invariant
  `if last_day > 0: assert (day - last_day) % 5 == 0` has two defects:
  (1) when `last_day == 0` (the 1970-01-01 epoch) the check is
  silently skipped; (2) for consecutive flights on the same day
  `day - last_day == 0`, so `0 % 5 == 0` passes trivially, meaning
  runs of same-day flights never actually exercise the every-nth
  contract. *Suggested fix:* initialise `last_day = None` and skip
  only when `is None`; track whether any non-zero gap was observed
  and assert it at end-of-loop so the invariant is actually hit. *[DONE]*
- **[Medium][SUSPICIOUS-DATA]** `test_query_result` — five expected
  row counts are bare integer literals (`1197`, `99`, `36`, `307`,
  `13`). The inline comment at lines 167–168 ("These queries were all
  tested manually in the SQLite shell to determine the correct results
  using this exact test database") *is* the kind of independent-check
  mitigation the rubric accepts — it is not the "run the SUT, copy
  the output" anti-pattern. Downgrade rationale vs. clearance: the
  manually-run SQL is not preserved (no `.sql` file in
  `tests/data/missions/`, no inline transcript), so a future reviewer
  cannot reproduce the check. Medium (not High) severity because
  these are plumbing row counts, not science-critical numbers.
  *Suggested fix:* commit the verification SQL as
  `tests/data/missions/oag-2019-test-subset.README.md` with each
  expected count beside its reference query.
- **[Low][WEAK-ASSERTION]** `test_query_result` sampling branch
  (lines 209–225) and every-nth branch (lines 236–250) — both
  terminate with `assert nflights < 307`, which passes even when
  `nflights == 0`. *Suggested fix:* add `assert nflights > 0`
  everywhere; for `every_nth=5` use the test DB to derive a concrete
  expected count and assert equality (the sampling case legitimately
  needs the loose bound). *[DONE]* Sampling branch gains
  `assert nflights > 0` alongside the loose upper bound; every_nth=5
  case pins the deterministic count (78) for this exact test DB.
- **[Low][WEAK-ASSERTION]** `test_query_result` frequent-flight
  branch (lines 252–257) — the DTW-touches invariant is asserted only
  on `results[0]`, not on the whole result set; a regression that
  returned DTW-touching pairs only in the first row would pass.
  *Suggested fix:* loop over `results` asserting
  `'DTW' in (r.airport1, r.airport2)` for each row. *[DONE]*
- **[Low][COVERAGE-GAP]** `test_query` — `CountQuery` is exported
  from `AEIC.missions` (and documented in CLAUDE.md) but has no
  `to_sql()` assertion in this file; coverage of `CountQuery` is
  purely transitive. *Suggested fix:* add a minimal case asserting
  `CountQuery(Filter(country='US')).to_sql()` contains `COUNT(*)`
  and the expected params. *[DONE]* *Actual remediation:* the SUT
  emits `COUNT(s.id)`, not `COUNT(*)`, so the assertion pins that
  exact form. Both unfiltered and country-filtered cases are
  covered, including the filter branch's JOINs and `WHERE`.
- **[Medium][COVERAGE-GAP]** `Database.set_random_seed()` — the
  documented reproducibility entry point for deterministic sampling,
  called out in CLAUDE.md. Neither this file nor
  `test_mission_db_creation.py` exercises it. A broken seed path
  would only surface at replay time. *Suggested fix:* add a test that
  opens the test DB, calls `set_random_seed(42)`, runs
  `Query(sample=0.1)`, collects the flight IDs; repeats with a fresh
  `Database` + same seed; asserts the two ID lists are equal. *[DONE]*
  See `tests/test_mission_db.py::test_set_random_seed_determinism`. A
  third run with a different seed pins that the sequence actually
  changes — without it, a stub `set_random_seed` that ignored the seed
  would still pass equality.
- **[Low][COVERAGE-GAP]** `Mission.from_query_result` /
  `Mission.from_toml` — coverage of `missions/mission.py` is only
  42 % for the phase. These constructors sit on the critical path
  from DB query → downstream simulation. *Suggested fix:* add unit
  tests that build a synthetic `QueryResult` and verify the resulting
  `Mission`'s geographic properties (distance, midpoint); do the same
  for a minimal `Mission.from_toml` input. *[DONE]* *Actual remediation:*
  the audit-named `Mission.from_query_result` does not exist — the
  actual SUT method is `QueryResult.from_row` (in `missions/query.py`).
  Added `test_mission_from_query_result_row` (covers the row-tuple
  field mapping including the `flight_id` = `row[2]` quirk and the
  `load_factor` = 1.0 OAG placeholder, plus a `gc_distance` envelope
  check) and `test_mission_from_toml_minimal` (covers the TOML
  field mapping plus a `gc_distance` ↔ `GEOD.inv` identity check).
- **[Low][HYGIENE]** `test_filter` — single function with ~12
  filter-configuration scenarios and no `pytest.mark.parametrize`;
  failures report the whole function, not the specific scenario.
  *Suggested fix:* parametrize by `(filter_kwargs, expected_sql,
  expected_params)` tuples or split into focused sub-tests. *[DONE]*

### `tests/test_mission_db_creation.py` (3 tests)

- **[High][WEAK-ASSERTION]** `test_oag_conversion` — the only
  assertion on the end-to-end `convert_oag_data` pipeline is
  `SELECT COUNT(*) FROM flights == 7`. Nothing verifies: (a) the row
  *contents* (carrier, flight number, aircraft type, origin /
  destination, distance, timestamps) match the source CSV; (b) the
  `schedules` table is populated at all; (c) the warnings file was
  written (or, as the comment implies, is empty because all 7 rows
  are valid). A bug that silently inserted 7 malformed rows or failed
  to create a single schedule row would pass. OAG ingestion is the
  mission-DB pipeline's entry point — every downstream test's
  provenance chain leads back through it — so this is the most
  consequential weakness in Phase 2. *Suggested fix:* after the
  conversion, assert at least one row's full content matches the
  source CSV, assert `SELECT COUNT(*) FROM schedules` is non-zero,
  and assert the warnings file exists with expected content
  (empty for the all-valid fixture).
- **[Medium][SUSPICIOUS-DATA]** `test_oag_conversion` — the expected
  `7` (valid-flight count) is a bare literal and the inline comment
  `# This extract of the 2019 OAG data contains 7 valid flights` is
  the only provenance. "Valid" is defined by whatever
  `convert_oag_data` currently accepts, so the literal is
  self-referential in the anti-pattern the rubric targets. The OAG
  extract also has no README under `tests/data/oag/`. *Suggested
  fix:* add `tests/data/oag/README.md` describing how
  `2019-extract.csv` was produced and listing the 7 expected flights
  (carrier + flight number + date); reference it from the test
  comment. The literal `7` then has external grounding.
- **[Medium][COVERAGE-GAP]** `test_oag_conversion` —
  `convert_oag_data` emits structured warning categories
  (`UNKNOWN_AIRPORT`, `TIME_MISORDERING`, `SUSPICIOUS_DISTANCE`,
  `ZERO_DISTANCE`), none of which are exercised. An extract designed
  to trigger each category is not part of the test suite.
  *Suggested fix:* add a second small CSV fixture with known-bad
  rows (unknown IATA, inverted departure/arrival, zero-distance
  self-loop, etc.) and assert the warnings file contains each
  category. *[DONE]* *Actual remediation:* the bad rows are generated
  inline in the test (`test_oag_warning_categories`) by mutating one
  cell of a known-good AS 1011 ORD->SEA template per category, so the
  trigger for each warning is visible at the call site rather than
  buried in a static fixture file.
- **[Medium][COVERAGE-GAP]** `test_airport_handling` (lines 37–57)
  — calls `db._get_or_add_airport(cur, 1234, 'CDG')` but never
  queries the `airports` table to verify the "add" half of the
  method actually persisted CDG with id `1234`. The returned
  `AirportInfo` only confirms lookup. *Suggested fix:* after the
  call, `cur.execute("SELECT id FROM airports WHERE iata_code = 'CDG'")`
  and assert the returned id is `1234`. *[DONE]* *Actual remediation:*
  the `1234` in the call is the CSV-line number (`line` parameter for
  warnings), not the airport id — the id is auto-assigned by sqlite
  and surfaced via `airport_info.id`. The persistence assertion now
  matches against `airport_info.id` instead. A negative case for the
  unknown 'QPX' airport asserts no row is left behind.
- **[Low][WEAK-ASSERTION]** `test_airport_handling` (line 53) —
  `int(airport_info.airport.latitude) == 49` is a truncation-based
  coarseness check for CDG (49.0097°N); a sign flip to -49.9 would
  still fail (truncates to -49), but a +49.9 drift would pass.
  *Suggested fix:* assert `airport_info.airport.latitude ==
  pytest.approx(49.01, abs=0.01)` for a tighter contract. *[DONE]*
- **[Low][HYGIENE]** `test_airport_handling` — exercises private
  methods (`_lookup_timezone`, `_get_or_add_airport`). Appropriate as
  a white-box unit test but the intent is undocumented; naïve
  refactoring of `WritableDatabase` internals would break this test
  without signal that the public surface is fine. *Suggested fix:*
  one-line docstring acknowledging the white-box scope. *[DONE]*
- **[Low][HYGIENE]** `test_dow_mask` (line 34) — `0b0000000` is
  seven zeros; every other literal in the test uses an eight-bit form
  (`0b01111111`, `0b00010101`, etc.). Functionally identical (leading
  zero), just inconsistent. *Suggested fix:* `0b00000000`.
- (No SUSPICIOUS-DATA finding for `test_dow_mask` — the bit-mask
  literals are self-documenting against the "Monday = bit 0 … Sunday
  = bit 6" contract that `_make_dow_mask` implements; the expected
  values are derivable from first principles, which is an accepted
  mitigation.)

## Phase 3 — Performance

**Notebook coverage:** `notebooks/test-cases.ipynb` has no Performance
section by design (the notebook is emissions-focused, with the SLS
helper being the only performance-adjacent code). SUSPICIOUS-DATA
trigger #4 does not apply to any finding in this phase. No new entries
for `test-quality-notebook-index.md`.

**Phase 3 coverage for performance SUT modules (per `/tmp/aeic-cov.xml`):**
`performance/__init__.py` 100 % · `performance/types.py` 92 % ·
`performance/model_selector.py` 90 % · `performance/edb.py` 88 % ·
`performance/apu.py` 90 % · `performance/models/__init__.py` 96 % ·
`performance/models/base.py` 95 % · `performance/models/legacy.py` 95 % ·
`performance/models/{bada,piano,tasopt}.py` 100 % (each is a 10-line
stub — `model_type` Literal only — so the 100 % is misleading) ·
`utils/models.py` 91 %.

The headline coverage gap is in `legacy.py`: `PerformanceTable.__post_init__`
runs six structural-validity checks (`check_coverage` × 3,
`check_fl_only` × 5) over the climb / cruise / descent sub-tables — the
heart of the legacy-table contract — and **none** of these error
branches is exercised by any test in this phase. Reported inline below.

### `tests/test_performance_model.py` (2 tests)

- **[Medium][LOGIC-ERROR]** `test_performance_model_initialization` —
  the docstring claims "PerformanceModel builds config, **and
  performance tables**", but the body ends with a literal
  `# TODO: Add tests for performance table.` comment (line 18) and
  never touches `model.performance_table`. The test as written only
  verifies the `LegacyPerformanceModel` discriminator and one
  LTO field. *Suggested fix:* finish the TODO — assert
  `model.performance_table.fl`, `model.performance_table.mass` are
  non-empty, and that `interpolate(state, ROCDFilter.ZERO)` returns a
  finite `Performance`. Or remove the misleading docstring clause if
  the table-side check is intentionally elsewhere. *[DONE]* *Actual
  remediation:* the cruise sanity call goes through
  `model.evaluate(state, SimpleFlightRules.CRUISE)` rather than the
  private `interpolate(...)` API — same coverage, public surface.
- **[Low][WEAK-ASSERTION]** `test_performance_model_initialization` —
  asserts `model.lto_performance is not None`. The field type is
  `LTOPerformanceInput | None`, so this catches a TOML omission, but
  the only thing that *actually* matters downstream is that the LTO
  parsed correctly; the next assertion already implies non-None.
  *Suggested fix:* drop the `is not None` line and rely on the
  `ICAO_UID == '01P11CM121'` assertion to fail with `AttributeError`
  if LTO went missing — or replace with a positive content check on
  `model.lto.fuel_flow[ThrustMode.IDLE] > 0`. *[DONE]* Replaced with
  the positive content check on `model.lto` (the converted internal
  form), which catches a regression that left LTO un-loaded.
- **[Medium][COVERAGE-GAP]** `tests/data/performance/bad_performance_model_{1,2,3}.toml`
  exist as fixtures but **no test in the repo references them** (grep
  confirms zero hits). They appear to be intentional negative-path
  fixtures for the validators in `PerformanceTableInput.validate_names_and_sizes`
  / `PerformanceTable.__post_init__` that were planned and never
  written. *Suggested fix:* either add `with pytest.raises(...)` tests
  loading each (asserting which validator fails), or delete the
  fixtures and stop shipping them as test data. *[DONE]* (fixtures deleted;
  schema had drifted — `model_type = "table"`, `cas_lo`/`Foo_kN` — so
  loading them would fail at Pydantic validation, not at `__post_init__`.
  Replaced with inline-parametrized mutate-one-cell tests in
  `test_performance_table.py`.)
- **[Medium][COVERAGE-GAP]** `test_performance_model_selection` exercises
  three branches of `SimplePerformanceModelSelector.__call__`: direct
  file match (`738`), synonym lookup (`319 → 380`), and default
  fallback (`739`, `73H`, `7M8`, `73J`). It does **not** exercise: the
  LRU cache hit path (no aircraft type repeats so cache is exclusively
  a write path); the constructor's three error guards (missing
  directory, missing `config.toml`, synonym pointing at a non-existent
  file); the missing `default` entry guard. *Suggested fix:* add a
  small parametrized test that builds a temporary selector directory
  per error case using `tmp_path`, and add a single repeat-aircraft
  test asserting `selector(m1) is selector(m2)` for two missions of
  the same type (cache identity). *[DONE]* Each error guard gets its
  own focused test (`test_simple_selector_init_*`) plus a
  `test_simple_selector_caches_repeated_lookups` that pins the cache
  identity contract via `is` equality on two same-aircraft-type
  missions from `sample_missions`.
- **[Low][HYGIENE]** `test_performance_model_selection` — the
  10-element expected list has no comment explaining the
  intent (which entries test default fallback vs synonym vs exact
  match). A reviewer cannot tell from the test alone what each row
  proves. *Suggested fix:* add an inline comment beside the expected
  list naming the three branches and which indices exercise each.
  *[DONE]*

### `tests/test_performance_table.py` (3 tests)

- **[Medium][WEAK-ASSERTION]** `test_create_performance_table` —
  builds an 18-row synthetic table from inline `tas()` and `fuel_flow()`
  closures (excellent first-principles provenance — no SUSPICIOUS-DATA
  flag), but only verifies a single `(FL=350, ROCD=0, mass=60000)`
  cell after construction. A bug that, e.g., transposed FL ↔ mass
  during DataFrame construction would still pass on this one cell
  if FL=350 and mass=60000 happened to coincide there. *Suggested fix:*
  loop over all 18 input rows and assert each one is recoverable from
  `model.df` at the matching `(fl, rocd, mass)` key, asserting
  equality of `tas` and `fuel_flow` against the closures. *[DONE]*
- **[High][COVERAGE-GAP]** `PerformanceTable.__post_init__` (in
  `legacy.py:191–251`) runs six structural-integrity checks on the
  BADA-shape contract — `check_coverage` for each of zero / positive /
  negative ROCD sub-tables, and `check_fl_only` for `tas` (zero, pos,
  neg) and `fuel_flow` / `rocd` (neg). Every one of these `raise
  ValueError` branches is **untested**. The `from_input → __post_init__`
  path is the validation surface for arbitrary user-supplied
  performance tables; a regression that loosened any check would be
  invisible. *Suggested fix:* add a parametrized test driving a small
  bad-table input through `PerformanceTable.from_input` per failure
  mode (missing FL coverage in cruise; TAS varying with mass at
  positive ROCD; etc.) and asserting the specific `ValueError`
  message. *[DONE]* *Actual remediation:* parametrized
  `test_performance_table_post_init_rejects` covers all 9
  `check_coverage`/`check_fl_only` branches via mutate-one-cell on a
  good baseline; two additional tests cover the mass-count guard
  (positive-only n≠3 and negative-only n≠1). Orphan fixtures deleted
  rather than rewritten (schema drift).
- **[Medium][COVERAGE-GAP]** `test_create_performance_table_missing_output_column`
  only covers the missing-`fuel_flow` branch of `PerformanceTableInput.validate_names_and_sizes`.
  The validator also raises on duplicate columns, missing `fl` /
  `tas` / `rocd` / `mass`, insufficient data columns, and
  inconsistent row lengths — none tested. *Suggested fix:*
  parametrize the test with one row per error message in the
  validator. *[DONE]* The original `test_create_performance_table_missing_output_column`
  was folded into the new parametrized
  `test_performance_table_input_rejects` (8 cases — one per raise
  branch).
- 🟡 **[Medium][COVERAGE-GAP]** `test_performance_table_subsetting` —
  exercises only `ROCDFilter.POSITIVE` and `ROCDFilter.NEGATIVE`. The
  `ZERO` (cruise) branch — which is what `SimpleFlightRules.CRUISE`
  resolves to and is the most-traversed path in real trajectory
  evaluation — is not subsetted. *Suggested fix:* add the third
  filter case asserting all `abs(rocd) <= ZERO_ROCD_TOL`.
- 🟡 **[Medium][WEAK-ASSERTION]** `test_performance_table_subsetting`
  (lines 64, 65, 69) — the `len(sub_table_1.fl) <= len(table.fl)` /
  `len(sub_table_1.mass) <= len(table.mass)` assertions are
  trivially satisfied even if `subset()` returned the full table
  (`<=` includes equality). For the BADA NEGATIVE descent sub-table
  the contract is `n_mass_values == 1`, which the test doesn't
  assert at all. *Suggested fix:* assert
  `len(sub_table_2.mass) == 1` for the negative branch and
  `len(sub_table_1.mass) == 3` for positive (matching BADA's
  documented climb / cruise / descent shape).
- **[Medium][COVERAGE-GAP]** `PerformanceTable.interpolate` /
  `Interpolator.__call__` — neither bilinear interpolation
  (`n_masses > 1`) nor the FL-only fallback (`n_masses == 1`) is
  exercised by this file. The end-to-end `LegacyPerformanceModel.evaluate_impl`
  is similarly untested. Phase 5 may pick this up via trajectory
  tests, but the natural place for unit coverage is here. *Suggested
  fix:* add a test that calls `model.evaluate(AircraftState(...),
  SimpleFlightRules.CRUISE)` on a known table cell and asserts the
  returned `Performance.fuel_flow` equals the input, then a second
  call between two cells and verifies linear interpolation. *[DONE]*
  Two tests on the sample model pin both paths:
  `test_interpolate_bilinear_recovers_cell_and_midpoint` (cruise →
  `n_masses>1`, exact cell + four-corner-centroid average) and
  `test_interpolate_fl_only_fallback_for_descent` (descent →
  `n_masses==1`).
- **[Low][COVERAGE-GAP]** `LegacyPerformanceModel.empty_mass` /
  `maximum_mass` are simple derived properties. Neither is asserted
  in this phase. *Suggested fix:* one-line assertions in
  `test_performance_model_initialization`. *[DONE]*

### `tests/test_model_utilities.py` (3 tests)

- **[Low][LOGIC-ERROR]** `test_ci_base_model_with_invalid_key` — the
  test name suggests verifying behaviour when an unknown key is
  passed, but the only assertion is `not hasattr(model, "INVALID_KEY")`,
  which is trivially true for any Pydantic model regardless of input.
  Looking at `_normalize_dict`: when `field_map.get(k.lower(), k)`
  finds no field, it falls through to `normalized[field_name] = v`
  with the original key, then Pydantic's default `extra='ignore'`
  policy silently drops it. The test passes whether the
  normalization layer drops the key or whether Pydantic does;
  there's no contract being pinned down. *Suggested fix:* explicitly
  assert the silent-drop behaviour by inspecting `model.model_extra`
  (or whatever Pydantic v2 exposes for ignored keys) and document
  whether the expectation is that `_normalize_dict` itself drops the
  key or that Pydantic does. *[DONE]* The test now calls
  `_normalize_dict` directly to pin that the unknown key is passed
  through verbatim while known keys case-fold, and pins the
  Pydantic-side drop via `model.model_extra is None` and the absence
  from `model_fields_set`. Either layer tightening trips the test.
- **[Low][COVERAGE-GAP]** `_normalize_dict` recursion handles three
  cases: non-model values, nested `BaseModel`, and `list[BaseModel]`.
  Only the latter two are tested. The list-of-non-model branch
  (`normalize_keys` mode='before' on a top-level list, line 80–84)
  is untested. *Suggested fix:* add a test passing a top-level list
  of dicts to `model_validate` (e.g. via a `RootModel[list[NestedModel]]`).
  *[DONE]* `test_ci_base_model_top_level_list` covers the dict-item
  leg via `RootModel[list[Item]]` and the inner `else v` (non-dict
  pass-through) leg via a direct `normalize_keys` call on a
  mixed-shape list.
- **[Low][COVERAGE-GAP]** `CIStrEnum._missing_` is only tested with
  one miss case (`"yellow"`). The non-string branch
  (`_missing_(42)` → returns `None` via the outer `if isinstance(value,
  str)`) is untested. Trivial. *Suggested fix:* add
  `assert Color._missing_(42) is None`. *[DONE]*

### `tests/test_thrust_modes.py` (8 tests)

The file does a respectable job on the arithmetic operators; the
gaps are around the construction surface and the freezing /
mutability contract.

- **[Medium][COVERAGE-GAP]** `ThrustModeValues.__init__` accepts five
  argument shapes (zero args, dict / TMV, np.ndarray, scalar float,
  four positional floats) plus an `else: raise`. Only the
  dict-shape and the zero-args shape (via `ThrustModeValues()` in
  `test_thrust_mode_values_comparison`) are exercised. The
  ndarray, scalar-float, four-arg, and invalid-init branches are
  untested. *Suggested fix:* parametrize a small constructor test
  per shape, plus one `pytest.raises(ValueError)` for the bad case.
  *[DONE]*
- **[Medium][COVERAGE-GAP]** Mutability contract — `__setitem__`
  raises `TypeError` when `_mutable` is False (the *default*), and
  `freeze()` / `copy(mutable=...)` are the documented escape hatches.
  None of this is tested. A regression that flipped the default to
  `mutable=True` would silently allow downstream code to mutate
  shared LTO data (the comment at types.py:71–73 explicitly warns
  against this). *Suggested fix:* one test asserting
  `tm = ThrustModeValues({...}); pytest.raises(TypeError, lambda: tm.__setitem__(ThrustMode.IDLE, 5.0))`,
  and one asserting `tm.copy(mutable=True)[ThrustMode.IDLE] = 5.0`
  succeeds. *[DONE]*
- **[Low][WEAK-ASSERTION]** `test_or_thrust_mode_values` — the `__or__`
  implementation iterates `self._data.items()`, so
  `tm3 | tm1` produces a result containing only IDLE and TAKEOFF
  (the keys of `tm3`). The test asserts the two values
  `result1[IDLE] == 11.0`, `result1[TAKEOFF] == 44.0` but never
  inspects which keys are actually in `result1` — `result1[CLIMB]`
  would return `0.0` via `__getitem__`'s default-to-zero, masking
  whether `__or__` correctly retained or dropped the CLIMB key. A
  bug that, e.g., changed `__or__` to iterate `other._data.items()`
  would change the resulting key set and pass the test.
  *Suggested fix:* `assert set(iter(result1)) == {ThrustMode.IDLE,
  ThrustMode.TAKEOFF}` and similarly for `result2`. *[DONE]*
- **[Low][COVERAGE-GAP]** `__add__` / `__mul__` / `__truediv__`
  each have two branches (TMV vs scalar). For `__add__` only the
  float-scalar branch is exercised via `1.0 + tm1` (the int branch
  and the `tm1 + 1.0` direction are untested — `__radd__` exists
  for the latter). For `__mul__`, only `2.0 * tm1` is tested
  (float scalar); the TMV × TMV branch is untested. *Suggested fix:*
  one-line tests per missing branch. *[DONE]* `__truediv__` branches
  were already both covered (`tm3 / tm1` and `tm1 / 2.0`); two new
  tests cover the missing `__add__` int + right-direction case and
  the `__mul__` TMV × TMV case.
- **[Low][COVERAGE-GAP]** Untested utility methods on
  `ThrustModeValues`: `as_array`, `sum`, `broadcast` (the only
  non-trivial one — it interacts with `ThrustModeArray`),
  `isclose`, `__hash__`, `__str__`, `__repr__`, `freeze`, `copy`.
  And the entire `ThrustModeArray` class is untested in this file
  (its `__post_init__` validation, `as_enum`, `modes`,
  `__array__`). *Suggested fix:* add a focused
  `test_thrust_mode_array.py` (or extend this file) for at least
  `broadcast` + `ThrustModeArray.__post_init__` (raise on invalid
  values), since those are the methods that touch real
  trajectory-shaped arrays. *[DONE]* The two highest-value methods
  the audit highlighted are now covered:
  `test_thrust_mode_values_broadcast` and
  `test_thrust_mode_array_rejects_invalid_values`. The remaining
  trivial accessors (`as_array`, `__str__`, etc.) are left to be
  exercised transitively — explicitly out of scope for a Low.
- **[Low][HYGIENE]** `test_thrust_mode_values_comparison` is a
  one-liner asserting `ThrustModeValues() != 0.0`. The intent
  (verifying `__eq__` returns False for non-TMV) is fine but
  obscure; equality between two TMVs that should be equal is never
  asserted. *Suggested fix:* expand to a small parametrized test
  covering: equal TMVs, unequal TMVs, TMV vs dict, TMV vs scalar.
  *[DONE]* Six parametrized cases now cover the equal-data,
  unequal-values, unequal-keys, vs-dict, vs-scalar, and vs-None
  branches; both `==` and `!=` are checked per case.

## Phase 4 — Emissions

This phase covers the three emissions-focused test files. Per the
notebook index, five SUT areas in scope have a matching
`notebooks/test-cases.ipynb` section (BFFM2 §2, NOx speciation §3,
HCCO §4, SCOPE 11 §5, MEEM §6, and SLS §1 indirectly); APU, GSE, LTO
aggregation, SOx, and the EDB loader do not. **SUSPICIOUS-DATA trigger #4
(notebook citation) fires for the first time in this phase.**

A recurring Phase 4 pattern: five tests (BFFM2, MEEM, SCOPE 11, NOx
speciation, HCCO) use hard-coded expected arrays with a comment pointing
at `notebooks/test-cases.ipynb` but **without a specific cell reference
or a "rounded results for use in test" block**. Only Section 1 (SLS) has
that block today (cell 17). The rubric counts a generic notebook
comment as *partial* mitigation — enough to downgrade to Medium, not
enough to clear. See the index file for the companion recommendation
that Sections 2–6 each grow an explicit rounded-results block.

### `tests/test_emissions.py` (10 tests)

Integration-style tests around `compute_emissions` using a
`DummyPerformanceModel` + `DummyTrajectory`. All expected species
values are either hard-coded in the test or re-derived inline by
calling the SUT a second time (the `_expected_trajectory_indices`
helper pattern).

- **[High][SUSPICIOUS-DATA]** `test_atmospheric_state_and_sls_flow_shapes`
  (`test_emissions.py:297–335`) — `expected_temp`, `expected_pressure`,
  `expected_mach` are hard-coded arrays with no provenance comment; the
  SLS-flow array has a `# NOTE: RESULTS FROM notebooks/test-cases.ipynb
  NOTEBOOK` comment but no cell number. Per the notebook index, cell 11
  of Section 1 explicitly admits the atmospheric inputs were *derived
  from `AtmosphericState`* — the very SUT under test — so this check is
  self-referential by construction. *Suggested fix:* cross-check the
  expected temp/pressure/mach against ICAO Standard Atmosphere tables
  (not against `AtmosphericState`) and note the independent source
  inline.
- **[High][SUSPICIOUS-DATA]** `test_get_gse_emissions_matches_reference_profile`
  (`test_emissions.py:338–358`) — hard-coded `expected` dict
  (`CO2=58_000.0`, `NOx=900.0`, `HC=70.0`, …) matches the per-class
  literals in `_gse_nominal_profile()` (`emissions/gse.py`). The test
  is a regression snapshot of the SUT's own constants, not an
  independent verification; no notebook section exists for GSE. Per the
  notebook index, GSE is already a documented notebook-gap. *Suggested
  fix:* cite an external source for these per-LTO-cycle emissions
  (IPCC, EASA/ICAO, Stettler 2011, etc.) in a comment, or add a GSE
  section to `test-cases.ipynb`.
- **[High][WEAK-ASSERTION]** `test_emit_matches_expected_indices_and_pointwise`
  (`test_emissions.py:183–213`) — the `_expected_trajectory_indices`
  helper (lines 126–180) invokes the **same** `BFFM2_EINOx`,
  `EI_HCCO`, and `get_SLS_equivalent_fuel_flow` functions the SUT
  uses, then the test asserts `compute_emissions(...)` outputs match
  the helper. This is an internal-consistency check, not a correctness
  check — if BFFM2 has a science bug, both sides inherit it and the
  test passes. The notebook Section 2 (BFFM2) exists precisely to
  break this self-reference. *Suggested fix:* replace the inline
  helper with hard-coded expected arrays cited to specific notebook
  cells, or at minimum label the test as a consistency check (not a
  correctness check) and add a separate correctness test that cites
  the notebook.
  *Actual remediation:* the "at minimum" alternative. The full Tier 3
  remediation (cite notebook rounded arrays) was examined and rejected
  — notebook Sections 2 and 4 use EDB inputs that do not match
  `DummyPerformanceModel.lto`, so citing those arrays would require
  either realigning the test fixture with notebook inputs (then this
  test duplicates `TestBFFM2_EINOx::test_matches_reference_component_values`)
  or adding new notebook cells for the current fixture (meaningful
  surgery, and HCCO Section 4 has no CO block). Instead: the test is
  renamed to `test_compute_emissions_pipeline_wiring` with a docstring
  that states what it does and does not verify, the helper is marked
  DELIBERATELY SELF-REFERENTIAL with a pointer to the real correctness
  tests (`TestBFFM2_EINOx`, `TestEI_HCCO`, `TestNOxSpeciation`,
  `test_atmospheric_state_and_sls_flow_shapes`), and a new mass-
  conservation assertion `no_prop + no2_prop + hono_prop ≈ 1.0` adds a
  genuine correctness signal independent of the helper.
- **[Medium][SUSPICIOUS-DATA]** `test_calculate_nvpm_meem_populates_fields`
  (`test_emissions.py:263–284`) — hard-coded `expected_mass` (6
  values) and `expected_number` (6 values) with no provenance comment.
  MEEM is notebook Section 6. The mass check uses `atol=1e-8` against
  values of order 1e-3 (tight, ~1e-5 relative); the number check uses
  `np.testing.assert_allclose` default tolerances against values of
  order 1e14 (effectively relative-only). *Suggested fix:* cite the
  specific notebook cell whose output produced these numbers, or add a
  rounded-results block to Section 6.
- **[Medium][LOGIC-ERROR]** `test_emissions_species`
  (`test_emissions.py:122–123`) — the entire test body is `assert
  len(emissions.species) == len(Species)`. A test named
  `test_emissions_species` without a docstring suggests it verifies
  which species are present; in practice it only verifies *count
  equality* with the `Species` enum. A bug that dropped `NOx` and
  added a duplicate species would pass. *Suggested fix:* assert
  `set(emissions.species) == set(Species)` (or whatever the intended
  containment is) and add a docstring stating the invariant. *[DONE]*
- **[Medium][WEAK-ASSERTION]** `test_lto_nox_split_matches_speciation`
  (`test_emissions.py:250–260`) — the test recomputes `NOx_speciation()`
  and asserts `lto_indices[NO] == lto_indices[NOx] * speciation.no`.
  This is the exact algebraic identity the SUT implements when it
  splits `NOx`, so any implementation that correctly uses the shared
  `NOx_speciation()` factors will pass even if the factors themselves
  are wrong. Correctness of the speciation factors is asserted in
  `test_emission_functions.py::TestNOxSpeciation::test_NOx_speciation_results`
  (itself SUSPICIOUS-DATA, see below). *Suggested fix:* keep this as
  a consistency check but rename (`test_lto_nox_split_consistent_with_speciation_factors`)
  and cross-reference the factor-correctness test. *[DONE]*
- **[Medium][COVERAGE-GAP]** `test_lto_respects_traj_flag_true`
  (`test_emissions.py:240–247`) — asserts `APPROACH` and `CLIMB` LTO
  emissions are ~0 when `climb_descent_mode=TRAJECTORY`, i.e. the
  climb/descent mass has moved to the trajectory side. It never
  asserts that the trajectory side *received* that mass, so a bug
  that dropped it on the floor entirely would pass. There is also
  no complementary `test_lto_respects_traj_flag_false` (`ClimbDescentMode.LTO`).
  *Suggested fix:* assert `np.sum(trajectory_emissions[species]) > 0`
  for climb/descent segments, and add the reciprocal-flag test. *[DONE]*
  The trajectory-side check uses NOx/HC/CO as canary species; the new
  reciprocal `test_lto_respects_traj_flag_false` pins the mirrored
  contract — non-zero APPROACH/CLIMB LTO columns and trajectory
  emissions zero outside the cruise slice.
- **[Low][WEAK-ASSERTION]** `test_scope11_profile_caching`
  (`test_emissions.py:234–237`) — asserts `profile_first.mass is
  profile_second.mass` (object identity) only. Does not verify that
  the profile is non-empty or that its mass/number matrices are
  populated, so a cache returning the same empty placeholder twice
  would pass. *Suggested fix:* additionally assert
  `profile_first.mass[ThrustMode.TAKEOFF] > 0` (or similar) and that
  per-mode values are finite. *[DONE]*
- **[Low][WEAK-ASSERTION]** `test_sum_total_emissions_matches_components`
  (`test_emissions.py:219–231`) — only verifies the arithmetic
  identity `total[s] == sum(trajectory + lto + apu + gse)`. It does
  not verify that the individual component sums are themselves
  non-zero where expected — a bug zeroing `gse_emissions` would pass
  as long as `total_emissions` also saw the zero. *Suggested fix:*
  add a precondition assertion (e.g., `assert
  emissions.gse_emissions[Species.CO2] > 0`) for a few canary
  species that the test config expects to be non-zero. *[DONE]*
  CO2 is the canary across all four buckets (trajectory / LTO /
  APU / GSE) since every component must emit CO2 under the default
  fuel config.

### `tests/test_emission_functions.py` (~24 tests across 7 classes)

Direct unit tests of `EI_HCCO`, `BFFM2_EINOx`, `NOx_speciation`,
`EI_SOx`, `get_APU_emissions`, `nvPM_MEEM`, `calculate_nvPM_scope11_LTO`,
plus one small integration class. Fixtures and a setup-method shared
across classes provide "standard atmosphere"-style inputs.

#### `TestEI_HCCO`

- **[High][LOGIC-ERROR]** `test_HC_outputs` (`test_emission_functions.py:53–71`)
  — line 71 reads `assert np.isclose(result, out_result).all` (missing
  `()`). This asserts the truthiness of the bound-method object `.all`,
  which is always `True`. The intended array-comparison never runs; a
  regression that made `result` differ from `out_result` would still
  pass. **This is the most serious defect found so far in the audit.**
  *Suggested fix:* `assert np.isclose(result, out_result).all()` — or
  prefer `np.testing.assert_allclose(result, out_result)` so a failure
  reports the diff rather than just `False`.
- **[Medium][SUSPICIOUS-DATA]** `test_HC_outputs` — once the assertion
  above is fixed, `out_result` (6 values) becomes a load-bearing
  reference array but has no provenance comment. Notebook Section 4
  (HCCO) has outputs in cells 29–34 but no explicit rounded-results
  block. *Suggested fix:* add an inline comment citing the specific
  notebook cell whose `HC_result` output these values were taken from.
- **[Low][COVERAGE-GAP]** `test_intercept_adjustment_uses_second_mode_value`
  — exercises the high-fuel-flow regime only; the low-flow branch and
  the interpolation region in `EI_HCCO` are untested. *Suggested fix:*
  parametrize across low / mid / high `fuelflow_evaluate` with
  expected-value assertions per regime. *[DONE]* New
  `test_branches_split_at_intercept` covers the lower (slanted-line)
  and upper (horizontal-line) segments separately on a calibration
  set with strictly-decreasing EI (guaranteed negative slope, finite
  intercept inside the calibration range), plus a mixed-array call
  that pins the per-element split.

#### `TestBFFM2_EINOx`

- **[Medium][SUSPICIOUS-DATA]** `test_matches_reference_component_values`
  (`test_emission_functions.py:263–305`) — seven expected arrays
  (`NOxEI`, `NOEI`, `NO2EI`, `HONOEI`, and the three proportion arrays)
  preceded by `# The results below were generated in test-cases.ipynb`.
  Notebook Section 2 (BFFM2) produces `results_BFFM` implicitly but
  has **no rounded-results block**, so there is no direct place in the
  notebook to verify these arrays against. Partial mitigation (notebook
  comment) → Medium, not High. Tolerance (`rtol=1e-6, atol=1e-9`) is
  appropriately tight. *Suggested fix:* add a rounded-results cell to
  Section 2 of `test-cases.ipynb` and change the comment to cite that
  cell specifically.
- **[Low][HYGIENE]** `test_thrust_categorization` mocks
  `get_thrust_cat_cruise` but only asserts outputs are finite. The
  mock is therefore a no-op for correctness. *Suggested fix:* assert
  the expected thrust-category-to-NOx scaling, or drop the mock.
  *[DONE]* *Actual remediation:* the mock was not just a no-op for
  correctness — it patched the wrong target (`AEIC.emissions.utils.get_thrust_cat_cruise`
  instead of the bound reference inside `ei/nox.py`), so the SUT
  always ran unmocked. Drop the mock entirely; pick an
  evaluation-flow array spanning all three categories; assert
  `BFFM2_EINOx`'s `noProp` / `no2Prop` / `honoProp` arrays equal the
  per-point lookup through `get_thrust_cat_cruise` + `NOx_speciation`.
  Pre-checks that at least two distinct categories were exercised
  prevent a constant-category regression from satisfying the identity.

#### `TestNOxSpeciation`

- **[Medium][SUSPICIOUS-DATA]** `test_NOx_speciation_results`
  (`test_emission_functions.py:311–322`) — `hono_prop`, `no2_prop`,
  `no_prop` hard-coded (four values each) with docstring claim "same as
  AEIC v2 matlab implementation" but no inline citation of a cell, a
  MATLAB file path, or a published source. Notebook Section 3 output
  is implicit (no rounded-results block). Science-critical (NOx
  speciation directly affects downstream air-quality estimates) so the
  lack of independent provenance is a concern. *Suggested fix:* cite
  the legacy MATLAB source file (`AEIC-v2/…`) commit/path and/or an
  external reference (ICAO Doc 9889 or an AEIC paper) inline.
- No issues with `test_summation_consistency` or `test_non_negativity`
  — first-principles invariants, tight tolerances.

#### `TestEI_SOx`

- No significant issues. `test_mass_balance`
  (`test_emission_functions.py:369–380`) is a first-principles mass
  balance (`MW_SO2=64`, `MW_SO4=96`, `MW_S=32`) computed inline; SOx
  is a notebook-gap area but the analytical check adequately stands in
  for external reference. The `rtol=0.01` tolerance is loose relative
  to the analytic identity but not a regression risk — downgrade to
  monitoring. *Suggested fix (optional):* tighten to `rtol=1e-6` or
  document why 1 % is the budget.

#### `TestGetAPUEmissions`

- **[Medium][COVERAGE-GAP]** APU tests collectively do not exercise
  the config-flag path (`emissions.apu_enabled=False` — if such a
  flag exists in `default_config.toml`; verify). If APU can be
  disabled via config, there is no test for the short-circuit branch.
  *Suggested fix:* add a test with
  `@pytest.mark.config_updates(emissions__apu_enabled=False)` that
  asserts `apu_emissions` is empty. *[DONE]* *Actual remediation:*
  added in `test_emissions.py::test_apu_disabled_short_circuits`
  rather than in `TestGetAPUEmissions`, because the `apu_enabled`
  guard lives in `compute_emissions` (emission.py:181), not in
  `get_APU_emissions` itself.
- **[Low][HYGIENE]** Arbitrary parameter values throughout
  (`fuel_kg_per_s=0.1`, `NOx_g_per_kg=15.0`, `apu_time=2854`, etc.)
  have no provenance comments. These are synthetic test inputs, not
  reference data, so don't trigger SUSPICIOUS-DATA, but a one-line
  comment "synthetic parameters; not a published APU reference"
  would prevent a future reader from assuming they are authoritative.
  *Suggested fix:* add the clarifying comment at the top of the
  class, and consider expanding notebook `test-cases.ipynb` with an
  APU section (see notebook-index gap list). *[DONE]* Class-level
  docstring now states the synthetic origin and warns against
  porting the numbers as authoritative. Notebook expansion is left
  out of scope for the Low (it was already noted in the
  notebook-index gap list separately).

#### `TestNvPMMEEM`

- **[Medium][SUSPICIOUS-DATA]** `test_MEEM_using_test_cases_data`
  (`test_emission_functions.py:~510–520`) — `ref_EI_mass` and
  `ref_EI_num` hard-coded with comment "MEEM test cases generated in
  notebooks/test-cases.ipynb"; Section 6 has outputs
  `ei_mass_alt, ei_num_alt` but no rounded-results block. *Suggested
  fix:* add rounded-results block to notebook Section 6 and cite the
  cell.
- **[Medium][WEAK-ASSERTION]** `test_MEEM_using_test_cases_data` —
  `np.allclose(EI_mass, ref_EI_mass)` and `np.allclose(EI_num, ref_EI_num)`
  use the `numpy` defaults (`rtol=1e-5, atol=1e-8`). For `EI_num`
  values of order 1e13–1e14, `atol=1e-8` is negligible, so the check
  degenerates to relative-only with 1e-5 tolerance — tight for science
  but inconsistent with the BFFM2 test's explicit `rtol=1e-6`.
  *Suggested fix:* specify tolerances explicitly; match the BFFM2
  test's `rtol=1e-6, atol=1e-9` scheme. *[DONE]* Switched to
  `np.testing.assert_allclose` while there so a regression reports the
  diff rather than just `False`.

#### `Test_nvPMScope11`

- **[Medium][SUSPICIOUS-DATA]** `test_SCOPE11_unit_test` — `ref_mass`
  and `ref_num` hard-coded; comment "SCOPE11 test cases generated in
  notebooks/test-cases.ipynb". Section 5 has reference EDB inputs but
  **no rounded-results block**. *Suggested fix:* same as MEEM — add
  rounded-results block to notebook Section 5.
- **[Medium][WEAK-ASSERTION]** `test_SCOPE11_unit_test` — implicit
  `np.allclose` tolerances against `number`-channel values of order
  1e14–1e15. *Suggested fix:* specify explicit tolerances. *[DONE]*
  Pinned to `rtol=1e-6, atol=1e-9` (matching the BFFM2 / MEEM scheme)
  via `np.testing.assert_allclose`.
- **[Low][LOGIC-ERROR]** `test_engine_type_scaling_and_invalid_smoke_numbers`
  — inline computation mirrors the SUT's CBC/AFR/kslm/Q formula, so
  the "engine_type_scaling" half is a tautological copy-paste check.
  The "invalid_smoke_numbers" half passes an `SN_matrix` containing
  `-1.0` (CLIMB) and `0.0` (TAKEOFF). Per `emissions/ei/nvpm.py`, the
  SUT treats both as invalid and emits zero; the assertion does check
  both positions are zero, but the test name suggests the scaling
  branch is the primary concern, leaving the invalid-SN branch as a
  secondary observation. *Suggested fix:* split into two tests —
  `test_scope11_engine_type_scaling` (with expected values sourced
  from the notebook, not inline-computed) and
  `test_scope11_invalid_smoke_numbers_return_zero`. *[DONE]*
  *Actual remediation:* split as suggested. The engine-scaling test
  now pins only the qualitative MTF > TF > 0 invariant (the
  bypass-correction signal) and defers full numeric grounding to
  `test_SCOPE11_unit_test`'s notebook-sourced reference, instead of
  inline-computing CBC/AFR/kslm/Q against the SUT formula. The
  invalid-SN test additionally pins the number-channel zero
  (a refactor producing NaN would otherwise have slipped through).

#### `TestIntegration`

- **[Low][SUSPICIOUS-DATA]** `test_nox_emissions_consistency` — the
  3-value expected array is a regression snapshot with no citation.
  *Suggested fix:* add a comment marking this explicitly as a
  regression guard, or cite a notebook cell. *[DONE]* Docstring now
  states the snapshot origin and points at
  `test_matches_reference_component_values` as the test that
  carries the notebook-sourced scientific correctness check;
  inline comment by the assertion makes the role of the array
  unambiguous to a future reader. Switched to `assert_allclose`
  while there.

### `tests/test_edb.py` (2 tests)

Tests for `EDBEntry.get_engine()` — one negative-path (monkeypatched
`pd.ExcelFile`) and one positive-path (reading the real
`sample_edb.xlsx`).

- **[Medium][HYGIENE]** `test_get_EDB_data_for_engine_returns_engine_data`
  (`test_edb.py:43–119`) — the expected values (`engine ==
  "CFM56-7B27E"`, `BP_Ratio == 5.1`, the full per-mode `ThrustModeValues`
  dicts, `EImass_max == 70.8`, etc.) come from reading the shipped
  `src/AEIC/data/engines/sample_edb.xlsx` file. `sample_edb.xlsx` has
  no README documenting its provenance or the ICAO EDB revision it
  was sourced from. The test therefore verifies that the parser's
  output matches the file, but nothing verifies the file against the
  public ICAO EDB. Not SUSPICIOUS-DATA per se (the file is an input,
  not a hard-coded expected value), but the provenance chain is
  incomplete. *Suggested fix:* add a README to
  `src/AEIC/data/engines/` naming the ICAO EDB revision (and
  publication date) the sample was extracted from.
- **[Low][COVERAGE-GAP]** `test_get_EDB_data_for_engine_raises_when_uid_absent`
  exercises only the "UID not in Gaseous Emissions sheet" branch.
  `EDBEntry.get_engine` has at least one other error-path — UID
  present in gaseous sheet but absent from nvPM sheet (or vice
  versa) — which is untested. *Suggested fix:* a one-line
  parametrization covering each sheet's absence branch. *[DONE]*
- **[Low][HYGIENE]** The positive-path test does a single giant
  assertion block with ~10 `assert` statements. On failure, pytest
  reports only the first failing line, which makes diagnosis slower
  than necessary when the EDB parser regresses. *Suggested fix:*
  consider `pytest.approx` with a dict, or a parametrized test per
  field. *[DONE]* Parametrized over 14 (`attr`, `expected`) tuples
  via a module-shared `sample_engine_info` fixture; failures land on
  the per-attribute test ID.

## Phase 5 — Trajectories

Three test files covering the trajectory builders, the NetCDF-backed
trajectory store, and the weather loader. Phase 5 SUT is **not**
represented in `notebooks/test-cases.ipynb` (the notebook covers
emissions algorithms only), so SUSPICIOUS-DATA trigger #4 does not
apply. Triggers #1 (bare literals) and #2 (long arrays without
provenance) are the relevant ones here.

**Cross-cutting headline:** of the 9 tests in this phase, **3 touch
NetCDF state** (via `TrajectoryStore` or `xr.open_dataset` inside
`Weather.get_ground_speed`) and **none carry `@pytest.mark.forked`**.
Every `pytest.mark.forked` hit in the repo is in `test_storage.py` —
the marker is documented project policy for NetCDF-touching tests
(per feedback memory) but has not been applied here. This is the
dominant finding of Phase 5.

**Coverage (from `/tmp/aeic-cov.xml`):**
- `trajectories/builders/legacy.py` 94.2 %, `builders/base.py` 96.1 %
- `trajectories/trajectory.py` 64.8 %, `trajectories/store.py` 67.9 %,
  `trajectories/ground_track.py` 89.7 %
- `builders/{ads_b,tasopt,dymos}.py` 85.7 % each — these are stub
  modules whose `__init__` raises `NotImplementedError`, so the
  "coverage" is the import path plus the single raise. No test asserts
  the `NotImplementedError` contract.
- `src/AEIC/weather.py` does not appear as an instrumented file in
  `/tmp/aeic-cov.xml`. Running `uv run pytest tests/test_weather.py` by
  itself hits the module fine; the omission suggests the single
  weather test's lines were drowned by cross-test coverage
  aggregation quirks or a collection ordering issue worth
  investigating. Not a test-quality finding per se.

### `tests/test_trajectories.py` (3 tests)

Pure in-process tests of the `Trajectory` container: copy/approx-eq
perturbation checks, single-point field-set derivation, and extensible
append-then-fix. No NetCDF touch, no external data fixtures.

- **[Low][WEAK-ASSERTION]** `test_append_to_trajectory`
  (`test_trajectories.py:94–126`) — appends 30 points across 3 phases
  with hardcoded inputs (`fuel_flow=1.4`, `aircraft_mass=60000-…`,
  `latitude=41.0+0.02*i`, etc.) and then asserts only `len == 30` and
  the three phase counters `== 10`. Nothing verifies that any
  *field value* actually round-trips through `make_point` → `append`
  → `fix`; a regression that silently zeroed out all `fuel_flow` data
  would pass. The hardcoded inputs are not SUSPICIOUS-DATA (they're
  test inputs, not expected outputs). *Suggested fix:* add a few
  field-value spot-checks after `fix()` — e.g. `ext_traj.latitude[0]
  == 41.0`, `ext_traj.fuel_flow[5] == 1.4`. *[DONE]* Spot-checks now
  cover `fuel_flow`, `latitude`, the per-phase altitude trace, and
  per-phase monotonic mass decrease.
- **[Low][COVERAGE-GAP]** `Trajectory.interpolate_time` and
  `Trajectory.copy_point` have no dedicated tests in this file
  (`trajectory.py:163`, `:148`). The out-of-bounds `left=nan /
  right=nan` behavior in `interpolate_time` and the bounds-check
  `IndexError` in `copy_point` are both untested. Possibly covered
  indirectly by `test_trajectory_comparison` going through
  `traj.copy()`, but not explicitly.  *Suggested fix:* add two short
  unit tests — one for OOB interp NaN, one for out-of-bounds
  `copy_point`. *[DONE]*
- No other issues found. `test_trajectory_comparison` is thorough —
  it exercises each `Dimension` (T / TP / TS / TSM / TSP) with
  small-then-large perturbations, so a broken `approx_eq` for any
  dimension class would fail.

### `tests/test_trajectory_simulation.py` (7 tests)

Exercises `LegacyBuilder.fly()` end-to-end, plus one
`TrajectoryStore` round-trip and a weather-domain negative test.

- **[Medium][HYGIENE]** `test_trajectory_simulation_basic`
  (`test_trajectory_simulation.py:60–77`) — creates a
  `TrajectoryStore`, writes N trajectories, reopens and reads.
  Both `TrajectoryStore.create` and `TrajectoryStore.open` open
  NetCDF4 datasets, but the test has no `@pytest.mark.forked`.
  Per project feedback memory, this is required for NetCDF-touching
  tests to prevent HDF5 state leakage across tests. *Suggested fix:*
  decorate with `@pytest.mark.forked`.
- **[Medium][HYGIENE]** `test_trajectory_simulation_outside_weather_domain`
  (`:80–86`) and `test_trajectory_simulation_weather` (`:89–94`) —
  both run `LegacyBuilder(..., use_weather=True)`, which instantiates
  `Weather` and opens `tests/data/weather/20240901.nc` via
  `xr.open_dataset` (`weather.py:60`). Neither test has
  `@pytest.mark.forked`. *Suggested fix:* decorate both.
- **[High][LOGIC-ERROR]** `test_trajectory_simulation_basic`
  (`:60–77`) — test sets up custom fields on each trajectory
  (`traj.test_field1 = np.random.rand(len(traj))`, line 69) and
  adds them to the store, then reopens with `TrajectoryStore.open`
  and only asserts `len(ts_loaded) == len(ts)` (line 76). The `#
  TODO: Test that additional fields are correctly saved and loaded`
  comment on line 77 acknowledges the real thing is unverified.
  Custom-field persistence is the non-obvious part of the store's
  contract; the test writes the fields but never reads them back. A
  regression that silently dropped additional field sets on save
  would pass. *Suggested fix:* iterate `ts_loaded` and assert
  `np.array_equal(traj_loaded.test_field1, original)` for each
  trajectory — and only then remove the TODO.
- **[Medium][FLAKY-RISK]** `test_trajectory_simulation_basic`
  (`:69–70`) — uses `np.random.rand(...)` and
  `np.random.randint(0, 100, ...)` without a seed. Currently latent
  because the only assertion is a length check, but once the LOGIC-
  ERROR above is fixed the unseeded random inputs become a real
  flakiness source (and, more subtly, a debugging nightmare when an
  IO regression appears only for certain value ranges). *Suggested
  fix:* seed `np.random.default_rng(...)` (or module-level
  `np.random.seed`) at the top of the test.
- **[Medium][WEAK-ASSERTION]** `test_trajectory_simulation_single`
  (`:51–57`) asserts only `len(traj) > 10`. A `LegacyBuilder` bug
  that truncated every trajectory to 11 points would pass. No
  check on starting/ending altitudes, total fuel, or phase-counter
  consistency. *Suggested fix:* at minimum assert
  `traj.n_climb + traj.n_cruise + traj.n_descent == len(traj)` and
  `traj.starting_mass > traj.fuel_mass[-1] + empty_mass`. *[DONE]*
  *Actual remediation:* `fly_descent` decrements `n_descent` by 1
  (legacy.py:258), so the SUT contract is the inequality
  `n_climb + n_cruise + n_descent <= len(traj)`, not strict equality.
  Each phase is also asserted non-zero (catches a phase being skipped
  entirely), and the mass invariants are split into three: aircraft
  finishes above empty, below starting, and fuel mass strictly
  decreases.
- **[Medium][WEAK-ASSERTION]** `test_trajectory_simulation_weather`
  (`:94`) asserts only `len(traj) > 0`. Same smoke-only problem
  as above, and here the weather path is what's supposedly under
  test — so at minimum one should assert that the ground-speed
  trace differs from the no-weather baseline. *Suggested fix:*
  also build a no-weather trajectory for the same mission and
  assert `np.any(traj_weather.ground_speed != traj_nowx.ground_speed)`. *[DONE]*
- **[Medium][WEAK-ASSERTION]** `test_trajectory_performance_model_selector`
  (`:141–147`) asserts only `len(traj) > 0` per mission. This is the
  only test of `PerformanceModelSelector.select_for_mission`; it
  does not verify that the *correct* model was selected for each
  mission. *Suggested fix:* assert which model was used — e.g.
  capture `builder.ctx.ac_performance.name` (or equivalent) and
  compare against the expected mapping. *[DONE]* *Actual remediation:*
  the trajectory doesn't surface the selected model, so the test
  asserts `selector(mis).aircraft_name == expected` per mission
  alongside the trajectory build (overlaps with
  `test_performance_model_selection` by design — the redundancy
  pins the dispatch contract from the builder side).
- **[Medium][COVERAGE-GAP]** No test instantiates `TASOPTBuilder`,
  `ADSBBuilder`, or `DymosBuilder`. Each stub currently raises
  `NotImplementedError` in `__init__`
  (`src/AEIC/trajectories/builders/{tasopt,ads_b,dymos}.py`).
  Without a test asserting that contract, someone could land a
  half-implemented subclass that no longer raises — the only
  surface saying "this isn't ready" is the exception, and nothing
  polices it. *Suggested fix:* one parametrized test
  `@pytest.mark.parametrize("cls", [TASOPTBuilder, ADSBBuilder,
  DymosBuilder])` that calls `with pytest.raises(NotImplementedError):
  cls()`. *[DONE]*
- **[Low][COVERAGE-GAP]** `iterate_mass=True` is tested only in the
  success case and in the `max_mass_iters=1` failure case. The
  boundary — a mission that converges exactly at `max_mass_iters`,
  or a reltol-on-the-edge case — is untested.  *Suggested fix:*
  a parametrized edge-case test; low priority relative to the
  others. *[DONE]* `test_trajectory_mass_iter_boundary` parametrizes
  over `_MIN_ITERS_TO_CONVERGE - 1` (must raise) and
  `_MIN_ITERS_TO_CONVERGE` (must succeed) for the BOS→LAX example
  mission at reltol=1e-6.
- **[Low][HYGIENE]** The `iteration_params` fixture (`:35–37`)
  hardcodes `test_reltol=1e-6` and `test_maxiters=1000`. The name
  `test_maxiters=1000` is suspicious — for missions that should
  converge in a handful of iterations, 1000 gives the test
  effectively no upper bound and will hide a regression that slows
  convergence. *Suggested fix:* lower to something like 20 (plenty
  of headroom for a correctly-behaving solver) so that a
  regression into slow convergence fails loudly instead of running
  silently for longer. *[DONE]* Lowered to 20. Empirical baseline
  on the BOS→LAX sample mission converges in ~7 iterations, so 20
  leaves ~3× headroom while making any quadratic-to-linear
  convergence regression fail loudly.

### `tests/test_weather.py` (2 tests)

Exercises `AEIC.weather.Weather` — one FileNotFoundError negative
test, one positive ground-speed computation against the shipped
weather fixture.

- **[Medium][HYGIENE]** `test_compute_ground_speed`
  (`test_weather.py:37–51`) triggers `xr.open_dataset` on
  `tests/data/weather/20240901.nc` via `Weather.get_ground_speed →
  _require_data → _require_main_ds` (`weather.py:60`). No
  `@pytest.mark.forked`. Same policy violation as the three in
  `test_trajectory_simulation.py`. *Suggested fix:* decorate with
  `@pytest.mark.forked`.
- **[High][SUSPICIOUS-DATA]** `test_compute_ground_speed`
  (`test_weather.py:51`) — asserts
  `gs == pytest.approx(191.02855126751604, rel=1e-4)`. The
  expected value is a 15-digit bare literal with no citation to a
  paper, equation, reference implementation, or notebook cell.
  Weather is not in `test-cases.ipynb` at all. The inline comment
  ("Relaxed tolerance because we changed the pressure level
  calculation slightly") is the smoking gun that this number was
  captured by running the SUT and then adjusted again when the SUT
  changed — exactly the self-generated-expected-value pattern the
  review is meant to flag. Triggers #1 and #2 of the
  SUSPICIOUS-DATA rubric both fire. *Suggested fix:* either (a) add
  a notebook section that computes `ground_speed` from
  first principles given a fixed wind field and ISA-pressure
  altitude and cite its rounded-results cell here, or (b) replace
  the hardcoded literal with an inline computation from the wind
  components at the test point (construct the ISA pressure, look
  up `u`/`v` directly from the NetCDF, compute `hypot(tas·cos +
  u, tas·sin + v)` in the test body — this makes the expected
  value derivable from the *data* rather than from the SUT).
  *Actual remediation:* neither (a) nor (b). Any independent
  computation against the real ERA5 fixture duplicates the SUT
  (ISA pressure lookup + xarray interp + hypot). Instead the
  15-digit literal was replaced with a finite-and-physically-
  plausible envelope (`100 < gs < 300` m/s for TAS=200 m/s). The
  algorithm itself is independently verified by the synthetic-
  fixture tests added later in the same file
  (`test_annual_mean_reads_single_file` onward), where the wind
  field is known and the expected ground speed is derivable on
  paper. The real-fixture test is now an honest smoke check for
  fixture loadability, not a precision check.
- **[Medium][COVERAGE-GAP]** Success-path weather is exercised by
  exactly one point (one time, one altitude, one TAS, no azimuth
  override). Uncovered in this file:
  - `azimuth` parameter explicitly supplied (auto vs. explicit
    diverge in `weather.py:127–130`).
  - Date-boundary caching — `_require_main_ds` closes and reopens
    when the date changes (`weather.py:43–57`); no test calls
    `get_ground_speed` twice with different dates.
  - `valid_time` dimension absent (`weather.py:75`) vs. present —
    the fixture file presumably has it; no test exercises the
    no-`valid_time` fallback branch.
  - Domain-violation `ValueError` path — covered indirectly via
    `test_trajectory_simulation_outside_weather_domain`, but not
    in this file where it most naturally belongs.
  *Suggested fix:* three more short tests in this file (explicit
  azimuth, two-date call sequence, direct
  out-of-domain call). *[DONE]* Date-boundary caching and the
  `valid_time` absent/present pair were both closed by the
  synthetic-fixture tests added in the same Phase 5 wave
  (`test_daily_mean_reopens_on_midnight`,
  `test_hourly_monthly_files_no_reopen_within_month`,
  `test_annual_mean_reads_single_file` /
  `test_annual_mean_with_length_one_valid_time_is_squeezed`); the
  remaining `azimuth` and direct out-of-domain branches are now
  pinned by `test_explicit_azimuth_overrides_ground_track_azimuth`
  and `test_out_of_domain_raises`.
- **[Low][HYGIENE]** `tests/data/weather/` contains
  `20240901.nc` with no README or provenance file. Not a
  SUSPICIOUS-DATA finding (the file is an input, not an expected
  value), but the test above depends on specific wind values in
  that file to produce its hardcoded expected ground-speed;
  without provenance, regenerating the file breaks the test with
  no clear trail to the original source (ERA5 reanalysis? what
  subset? what download date?). *Suggested fix:* add
  `tests/data/weather/README.md` naming the source dataset (e.g.
  ECMWF ERA5), the spatial subset, the download date, and the
  script / command used to produce the file.
- **[Low][HYGIENE]** `Weather.get_ground_speed` docstring has two
  typos: `azmiuth` (param name) and `azmith` (body). Not a test
  finding but surfaced while judging the test — listed in Bonus
  below.
- No notebook gap update: Weather is already implicitly on the
  gap list via the "SOx / APU / GSE / LTO / atmospheric state"
  note in the notebook index — expanding that list is out of scope
  for Phase 5.

## Phase 6 — Verification & golden

Three test files of very different character: one not-a-test script
(`test_legacy_verification.py`), one cross-implementation tolerance
check (`test_matlab_verification.py`), and one frozen-snapshot
regression sentinel (`test_golden.py`). Phase 6 SUT is the verification
harness itself (`src/AEIC/verification/{legacy,metrics}.py`) plus
`Container.approx_eq` and `Trajectory.compare`.

**Headline:** `test_legacy_verification.py` contains **zero pytest
tests** — every `def` lives inside `main()` guarded by
`if __name__ == '__main__'`. Confirmed against `/tmp/aeic-collect.txt`,
which picks up only `test_matlab_verification` and
`test_trajectory_simulation_golden` from Phase 6. The file also
hardcodes `/home/aditeya/AEIC/tests/data/legacy_verification/` and
duplicates ~150 lines of plumbing already provided by
`AEIC.verification.metrics`. It is evidently a pre-pytest legacy
script left in the `tests/` tree after the proper
`test_matlab_verification.py` was written.

**Notebook coverage:** verification is a comparison harness, not an
algorithm, so no notebook section applies. SUSPICIOUS-DATA trigger #4
does not fire on any Phase 6 file. Trigger #3 (no README in
`tests/data/verification/`) fires on `test_matlab_verification`.
`tests/data/golden/` is cleared per the rubric — trigger #3 does *not*
fire on `test_golden.py`.

**Coverage (from `/tmp/aeic-cov.xml`, already captured for this
review):**
- `src/AEIC/verification/metrics.py` — 90 %. The uncovered lines
  are the all-NaN mask branch in `ComparisonMetrics.compute` and the
  `raise ValueError` in `out_of_tolerance` — both defensive branches
  unreachable from the current test data.
- `src/AEIC/verification/legacy.py` — 67 %. The uncovered bulk is
  `process_matlab_csvs` (lines 16–65), which is the ingestion
  function that converts the two raw MATLAB dumps in
  `matlab-output-orig/` into the per-mission CSVs in `matlab-output/`
  that the test actually reads. No test exercises this path (see
  COVERAGE-GAP below).

### `tests/test_legacy_verification.py` (0 tests — script)

`main()` runs a MATLAB-vs-AEIC comparison across six missions,
computes MAPE / RMSE / correlation per field, and writes PNG plots
to `plots/`. It prints the metric dataframes but never asserts on
them. Not collected by pytest.

- **[High][LOGIC-ERROR]** `tests/test_legacy_verification.py` —
  **file is named like a test module but contains no `test_*`
  functions**. All logic is inside `main()` gated by
  `if __name__ == '__main__'` (line 329). `pytest --collect-only`
  picks up zero items from it. The sibling file
  `test_matlab_verification.py` appears to be the intended
  replacement and does the same comparison with a real assertion.
  *Suggested fix:* delete after confirming `test_matlab_verification`
  covers the cases, or move to `scripts/legacy_verification_plot.py`
  if the PNG plots are still desired as a manual diagnostic.
- **[High][HYGIENE]** `test_legacy_verification.py:50` —
  `DEFAULT_TEST_DATA_DIR = "/home/aditeya/AEIC/tests/data/legacy_verification/"`.
  Hardcoded absolute path to a user home dir that does not exist on
  any other machine, and note the path segment (`legacy_verification`)
  does not even match the real repo layout
  (`tests/data/verification/legacy`). Running the script directly on
  any fresh checkout crashes. *Suggested fix:* resolve relative to
  `Path(__file__).parent / 'data' / 'verification' / 'legacy'`, or
  delete per the finding above.
- **[Medium][ISOLATION]** `test_legacy_verification.py:226–227` —
  `main()` mutates `os.environ['AEIC_PATH']` and calls
  `Config.load(...)` directly. If this file were ever converted into
  a test without dropping those two lines, it would bulldoze the
  `default_config` autouse fixture that every other test relies on
  and break the singleton lock (`Config.load` raises if an instance
  already exists — see CLAUDE.md "Configuration singleton" section).
  *Suggested fix:* when converting (or before deletion) rely on the
  `test_data_dir` fixture and `@pytest.mark.config_updates` like
  `test_matlab_verification` does.
- **[Medium][COVERAGE-GAP]** `test_legacy_verification.py:53–113`
  reimplements `_metrics(y_true, y_pred)` inline — a second copy of
  what `AEIC.verification.metrics.ComparisonMetrics.compute` already
  does (the two bodies are byte-for-byte equivalent aside from the
  return-type shape). Two maintained copies of the same numeric
  reduction are a silent-divergence risk. Listed under COVERAGE-GAP
  because the SUT copy (in the library) is tested only indirectly
  via `test_matlab_verification`; the script copy has no test at
  all. *Suggested fix:* subsumed by deleting the file per the first
  finding in this subsection.

### `tests/test_matlab_verification.py` (1 test)

A single `test_matlab_verification(test_data_dir)` that loads each
of six legacy MATLAB per-mission CSVs, flies the same mission with
`LegacyBuilder`, and checks every `TRAJ_FIELDS` element (plus
`trajectory_indices` per species) has MAPE ≤ 0.25 % via
`out_of_tolerance(..., mape_pct_tol=0.25)`. Structurally clean and
the only real verification test in Phase 6.

- **[Medium][SUSPICIOUS-DATA]** `test_matlab_verification` —
  legacy reference data lives in
  `tests/data/verification/legacy/matlab-output/*.csv` with **no
  README or provenance file in any parent directory** (confirmed:
  `find tests/data/verification -name 'README*' -o -name '*.md'`
  returns nothing). Rubric trigger #3 applies. The test itself
  mitigates the severity — this is a cross-implementation
  comparison, not a bare-literal equality check — so downgraded to
  Medium. But the reader still cannot tell *which* MATLAB code
  produced `BOS_MIA_738.csv` at *which* revision, and
  `matlab-output-orig/` contains two timestamped CSVs dated
  `2026-03-05` that are related to `matlab-output/` by an
  undocumented transformation. *Suggested fix:* add
  `tests/data/verification/legacy/README.md` naming (a) the
  AEIC-MATLAB commit hash / revision, (b) the missions + inputs
  used, (c) the relationship between `matlab-output-orig/`
  (raw two-file MATLAB dump) and `matlab-output/` (per-mission
  CSVs produced by `process_matlab_csvs`).
- **[Medium][WEAK-ASSERTION]** `test_matlab_verification.py:13–24`
  — `TRAJ_FIELDS` includes `'flight_time'`. `compare()` in
  `trajectory.py:209` computes pointwise MAPE between the two
  trajectories' `flight_time` arrays, but `flight_time` is the
  independent variable along the trajectory and both series count
  upward from 0 by construction; any reasonable pairing will give
  MAPE ≈ 0 regardless of SUT correctness. Including it in the
  comparison list is a free pass per mission. *Suggested fix:*
  drop `'flight_time'` from `TRAJ_FIELDS`; comparison is only
  informative for the dependent variables.
- **[Low][HYGIENE]** `test_matlab_verification.py:38` —
  `SKIP_FINAL_POINT_FIELDS = set(['true_airspeed'])`. The set is a
  one-liner magic constant with no comment; the *reason* for
  skipping TAS's final point (likely the landed/decelerated
  endpoint where legacy and new models disagree by definition) is
  not stated anywhere the reader can see. Also
  `set(['true_airspeed'])` is slower and less idiomatic than
  `{'true_airspeed'}`. *Suggested fix:* add a one-line comment
  explaining the final-point skip and convert to set literal.
  *[DONE]*
- 🟢 **[Low][COVERAGE-GAP]** `src/AEIC/verification/legacy.py::process_matlab_csvs`
  (lines 16–65) is unexercised by any test (directly responsible
  for the 67 % coverage on that file). The function handles the
  raw two-file → per-mission split including the
  `(tdf.t[:-1] != edf.t).any()` consistency check at line 58 —
  silent breakage of that check would corrupt all of
  `matlab-output/` without a single test noticing. *Suggested
  fix:* one small unit test feeding synthetic two-mission
  `traj_df` / `emis_df` into `process_matlab_csvs` and asserting
  the written files contain the expected merged rows.

### `tests/test_golden.py` (1 test)

A 23-line snapshot test: runs `LegacyBuilder` over
`sample_missions` and compares each new trajectory to the matching
index in `tests/data/golden/test_trajectories_golden.nc` (produced
by `scripts/make_golden_test_data.py`) via `Trajectory.approx_eq`,
which reduces to `np.allclose` defaults
(`rtol=1e-5, atol=1e-8` — see `container.py:174`).

- **[Medium][WEAK-ASSERTION]** `test_trajectory_simulation_golden`
  (`test_golden.py:5–23`) — the expected values are a **SUT
  self-snapshot** (`scripts/make_golden_test_data.py` builds the
  file by running the current SUT and freezing the output). Per the
  rubric's "mitigating evidence" clause, a test that compares
  against a self-generated snapshot and is labeled as a correctness
  check is flagged as `WEAK-ASSERTION`. The test name
  `test_trajectory_simulation_golden` and the lack of any comment
  noting snapshot semantics let a reader believe the test verifies
  simulation correctness when it only verifies *non-drift from a
  prior SUT state*. A legitimate improvement (fixed bug, better
  numerics) will fail this test identically to an actual regression.
  *Suggested fix:* rename to
  `test_trajectory_simulation_matches_golden_snapshot` and add a
  docstring noting the test is a regression sentinel, not an
  independent correctness check; cross-reference the future
  notebook section (per the notebook-gap list) if/when trajectory
  state gets an independent implementation. *[DONE]*
- **[Medium][WEAK-ASSERTION]** `test_trajectory_simulation_golden`
  — the tolerance is implicit (buried in `approx_eq` →
  `np.allclose` defaults at `container.py:174`). `rtol=1e-5` on a
  fuel_flow of O(1) is 1e-5 kg/s (tight); but on `ground_distance`
  values of O(1e6 m) it is ~10 m, and on `aircraft_mass` of
  O(5e4 kg) it is ~0.5 kg. Tolerance should be picked per-field
  based on expected physical sensitivity, not inherited from numpy
  defaults. More importantly, a units-convention change (e.g.
  altitude switched from m to km) would produce values still
  close to the new golden after regeneration but catastrophically
  wrong for downstream consumers — this test cannot detect
  unit-convention drift. *Suggested fix:* parametrize the
  tolerance explicitly, and add at least one scalar-level assertion
  on a named field (e.g. `assert 50_000 < traj.aircraft_mass[0]
  < 90_000` for a 738) that would blow up loudly on a unit shift.
  *[DONE]* *Actual remediation:* the tripwire half — physical
  envelopes on aircraft_mass, altitude, ground_distance, and TAS
  for a 738 — was added to detect coordinated regeneration after a
  unit shift. The per-field tolerance parametrization was *not*
  pursued: it requires either a SUT change to `Container.approx_eq`
  or duplicating the comparison logic in the test, both of which
  exceed the medium's intended scope.
- 🟢 **[Low][COVERAGE-GAP]** `test_golden.py` never asserts that
  `len(comparison_ts) == len(sample_missions)`. If the golden file
  and the mission fixture drift apart (e.g. sample missions
  extended but golden file not rebuilt), `comparison_ts[idx]`
  raises an `IndexError` mid-loop with a less informative
  traceback than a precondition assertion would. *Suggested fix:*
  add `assert len(comparison_ts) == len(sample_missions)` before
  the loop.
- **[Low][HYGIENE]** `test_golden.py:22–23` — the final failure
  mechanism is `if len(failed) > 0: raise AssertionError(...)`
  rather than `assert not failed, ...`. Both work in pytest; the
  `assert` form is idiomatic and gives slightly better failure
  formatting. *Suggested fix:* replace with
  `assert not failed, f'Trajectory simulation mismatch for: {failed}'`.
- **[Low][HYGIENE]** `test_golden.py` does **not** carry
  `@pytest.mark.forked` despite `TrajectoryStore.open` opening a
  NetCDF4 dataset — same class of issue as the four phase-5
  HYGIENE findings. The project's policy per the feedback memory
  is that any NetCDF-touching test must be `forked` to prevent
  HDF5 state leakage. *Suggested fix:* decorate with
  `@pytest.mark.forked`.

### Notebook gap updates (Phase 6)

None. Verification is a comparison harness, not a numeric
algorithm. The Phase 5 note already lists "atmospheric state" and
"trajectory state" as notebook gaps, which covers what
`test_golden.py` would want as an independent reference.

## Bonus: SUT observations (incidental)

*Not a SUT audit; only what surfaced during test review.*

- **`Config.escape()` decorator stacking** (`src/AEIC/config/core.py:227–242`).
  The method is defined as
  ```python
  @contextmanager
  @staticmethod
  def escape():
      ...
  ```
  Applying `@contextmanager` *outside* `@staticmethod` is unusual —
  `staticmethod` produces a descriptor, which `contextmanager` then
  wraps. Whether this works on every Python version is non-obvious, and
  the fact that it is currently *never called by the test suite* (see
  the test_config.py COVERAGE-GAP finding above) means any regression
  here would be invisible until a reproducibility replay runs in prod.
  Suggest swapping the decorator order (`@staticmethod` outermost) and
  adding a targeted test.
- **Untracked worktree dirs breaking collection** (`pretty-performance-model-toml/`,
  `trajectories-to-grid/`). Each ships a `tests/conftest.py` that pytest
  tries to import under the module name `tests.conftest`, colliding
  with the repo's own `tests/conftest.py`. Running `uv run pytest` from
  the repo root fails collection as a result; only
  `uv run pytest tests/` works. Fix in `pyproject.toml`'s
  `[tool.pytest.ini_options]` with `testpaths = ["tests"]` or
  `norecursedirs = ["pretty-performance-model-toml", "trajectories-to-grid"]`.
- **nvPM SCOPE 11 silent skip on invalid smoke numbers**
  (`src/AEIC/emissions/ei/nvpm.py:~245`). The loop inside
  `calculate_nvPM_scope11_LTO` `continue`s when `SN == -1 or SN == 0`,
  leaving that thrust mode's EI at 0.0. The public docstring does not
  document this behavior; `test_engine_type_scaling_and_invalid_smoke_numbers`
  (Phase 4) is currently the only record that zero means "skipped,
  not computed". *Suggested fix:* add a docstring line enumerating
  which SN sentinels are treated as invalid and what output shape
  callers should expect in that case.
- **Deprecated `pd.Timestamp.utcfromtimestamp`** (`src/AEIC/missions/query.py:137–138`).
  Running the Phase 2 test files emits ~3500 `Pandas4Warning` events
  from these two lines ("`Timestamp.utcfromtimestamp` is deprecated and
  will be removed in a future version. Use `Timestamp.fromtimestamp(ts,
  'UTC')` instead."). The warning volume drowns anything else in
  pytest output and the API is scheduled for removal, so each test run
  will start failing collection once Pandas removes it. *Suggested
  fix:* replace with `pd.Timestamp.fromtimestamp(row[0], tz='UTC')` at
  both call sites. Incidental — surfaced during Phase 2 review, not a
  test-quality finding.
- **`print()` in builder exception path**
  (`src/AEIC/trajectories/builders/base.py:222–224`). The `except
  Exception as e:` block inside `Builder.fly()` calls
  `print('Error during trajectory simulation for flight …: {e}')`
  before re-raising. This is a debugging artefact: in a production
  run across millions of missions, a transient bad-mission would
  dump to stdout once per failure with no filtering and no log-level
  control. The exception is re-raised immediately after, so the
  caller has full context anyway. *Suggested fix:* replace with a
  `logging.getLogger(__name__).exception(...)` call, or drop the
  `print` entirely and rely on the caller's exception handling.
  Surfaced during Phase 5 review.
- **`isinstance(vs, str | None)` bug in `Container` equality paths**
  (`src/AEIC/storage/container.py:138`, `:162`). Both `__eq__` and
  `approx_eq` branch on `isinstance(vs, str | None)`, but `None` is
  an *instance*, not a type — `isinstance(x, str | None)` raises
  `TypeError: isinstance() argument 2 cannot contain a parameterized
  generic` (or similar depending on CPython version). The only
  reason the bug stays latent is that `Container._data` in practice
  never seems to hold a `str` or `None` value (data fields are
  floats, ndarrays, `SpeciesValues`, or `ThrustModeValues`), so the
  branch is never taken. As soon as someone adds a string-valued
  per-trajectory field, both equality paths will crash. The
  identical line appears twice, which also means any fix has to
  touch both sites. *Suggested fix:* change both to
  `isinstance(vs, (str, type(None)))` (or `vs is None or
  isinstance(vs, str)`), and add a `Container` test that stores a
  `None` and a `str` per-container field and round-trips it through
  `==` / `approx_eq`. Surfaced during Phase 6 review of
  `approx_eq`'s role in `test_golden.py`.
- **Weather docstring typos** (`src/AEIC/weather.py:100–102`). The
  `get_ground_speed` docstring names the parameter `azmiuth` and
  describes it as the `azmith` — the actual Python parameter is
  `azimuth`. Harmless today (Python doesn't check docstring names)
  but will mislead any future user who searches the source for
  `azimuth` and finds the typo'd variant. *Suggested fix:* correct
  both to `azimuth`. Surfaced during Phase 5 review.
