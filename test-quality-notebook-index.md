# Notebook Index — `notebooks/test-cases.ipynb`

Purpose: map each notebook section to (a) the AEIC SUT function(s) it
provides an independent implementation for, (b) whether the section ends
with a *"Rounded results for use in test"* block that tests can cite as
their source of expected values, and (c) which tests currently reference
it.

> This index is consulted by every phase of the test-quality review when
> evaluating `SUSPICIOUS-DATA` findings. If a test in Phase 2–6 lacks a
> provenance citation but a notebook section exists that *could* back it,
> flag as `SUSPICIOUS-DATA` with suggestion "cite notebook section N".

## Sections

| # | Section title | SUT target (in `src/AEIC/`) | "Rounded results" block? | Known tests that cite / should cite |
|---|---|---|---|---|
| 1 | Sea-level-static fuel flow | `emissions/ei/*` (SLS conversion used by BFFM2 / nvPM pipelines); function is notebook-local `fuel_flow_to_sls` | **Yes** (cell 16 → cell 17: `[round(r, 6) for r in results]`) | `tests/test_emissions.py::test_atmospheric_state_and_sls_flow_shapes` (cites notebook in comment but generically; self-referential atmospheric inputs per cell 11) |
| 2 | BFFM2 — NOx emissions index (MATLAB-equivalent) | `src/AEIC/emissions/ei/nox.py::BFFM2_EINOx` | Implicit via `results_BFFM` output cells; **no explicit rounded-results block** | `tests/test_emission_functions.py::TestBFFM2_EINOx::test_matches_reference_component_values` cites notebook generically (no cell); **Phase 4 recommends adding a rounded-results block**. |
| 3 | NOx speciation (MATLAB-equivalent) | `src/AEIC/emissions/ei/nox.py::NOx_speciation` | Output of `print(NOx_speciation_from_matlab())`; **no rounded-results block** | `tests/test_emission_functions.py::TestNOxSpeciation::test_NOx_speciation_results` claims "AEIC v2 matlab implementation" in docstring but cites no notebook cell. **Phase 4 recommends adding a rounded-results block** and/or MATLAB-source citation. |
| 4 | HCCO — HC and CO emissions indices | `src/AEIC/emissions/ei/hcco.py::EI_HCCO` | Implicit via `print(HC_result)`; **no rounded-results block** | `tests/test_emission_functions.py::TestEI_HCCO::test_HC_outputs` — ⚠ Phase 4 found the assertion is a no-op (missing `()` on `.all`). Once fixed, the expected array needs a cell-specific citation. |
| 5 | SCOPE 11 nvPM (mass + number from smoke number) | `src/AEIC/emissions/ei/nvpm.py::calculate_nvPM_scope11_LTO` | **No explicit rounded-results block found**; has reference EDB inputs | `tests/test_emission_functions.py::Test_nvPMScope11::test_SCOPE11_unit_test` cites notebook generically. **Phase 4 recommends adding a rounded-results block**. |
| 6 | MEEM steps 1–4 (nvPM EI mass/number scaling) | `src/AEIC/emissions/ei/nvpm.py::nvPM_MEEM` | Outputs of `ei_mass_alt, ei_num_alt`; **no rounded-results block** | `tests/test_emission_functions.py::TestNvPMMEEM::test_MEEM_using_test_cases_data` cites notebook generically; also `tests/test_emissions.py::test_calculate_nvpm_meem_populates_fields` has no notebook comment at all. **Phase 4 recommends adding a rounded-results block**. |

## Notebook gaps — SUT areas that arguably should have an independent implementation here but don't

These are candidates for the user to decide whether to expand
`test-cases.ipynb`. Populated by Phase 1 observations; later phases will
append.

- **APU emissions** — `emissions/apu.py`. Tested via
  `tests/test_emission_functions.py::TestGetAPUEmissions::*`. No notebook
  section; expected values currently come from wherever the tests compute
  them.
- **GSE emissions** — `emissions/gse.py`. Tested via
  `tests/test_emissions.py::test_get_gse_emissions_matches_reference_profile`.
  No notebook section.
- **SOx (fuel-sulfur mass balance)** — `emissions/ei/` SOx path. Tested
  via `TestEI_SOx::*`. Mostly trivial math but a notebook section would
  still lock provenance.
- **P3T3 NOx method** — alternative to BFFM2. Not obviously represented
  in the notebook.
- **LTO cycle aggregation** — `emissions/lto.py`. Not represented.
- **Atmospheric state / standard-atmosphere conversions** —
  `utils/standard_atmosphere.py`. Section 1 of the notebook relies on
  atmospheric state values but notes they are "generated from the
  `AtmosphericState` class in the AEIC code" (cell 11). That is the
  self-referential pattern the review is meant to flag; a genuinely
  independent implementation in a notebook section would be an
  improvement.

## Housekeeping notes

- The guidelines cell (cell 0) states the notebook is an *independent*
  source and code must not import AEIC. Cell 11 markdown admits Section 1
  test-data **is** derived from `AtmosphericState`; review Section 1's
  atmospheric inputs more carefully in Phase 4.
- Section numbering in the notebook uses both `###` (Section 1) and `#`
  (Sections 2–6). Cosmetic only.

## Phase 4 recommendation: add rounded-results blocks to Sections 2–6

Phase 4 confirmed that only Section 1 (SLS) has a
`[round(r, 6) for r in results]` block usable as a citable provenance
anchor. Sections 2, 3, 4, 5, and 6 all have tests that cite
`test-cases.ipynb` in a generic inline comment, but no specific
cell/output they can point at. The recommended remediation pattern is:

1. At the end of each section, add a markdown cell titled
   **"Rounded results for use in test"**.
2. Below it, add a code cell that prints the rounded expected arrays
   the tests consume (matching the tolerances the tests use).
3. Update each test's inline comment to cite the specific cell
   number (e.g., `# From notebooks/test-cases.ipynb §3 cell NN`).

This turns "generated in test-cases.ipynb" (partial mitigation) into
a verifiable cell reference (full mitigation per the rubric), and
lets a future maintainer re-derive the expected values without
running the SUT.
