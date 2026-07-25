from __future__ import annotations

import pandas as pd
import pytest

from AEIC.commands.make_piano_performance_model import write_piano_performance_toml
from AEIC.config import config
from AEIC.parsers.piano_reader import LB_TO_KG, PianoData
from AEIC.performance.edb import EDBEntry
from AEIC.performance.models import LegacyPerformanceModel, PerformanceModel
from AEIC.performance.models.base import LTOPerformanceInput
from AEIC.performance.models.piano import PianoPerformanceModel
from AEIC.units import KNOTS_TO_MPS

LBHR_TO_KGS = LB_TO_KG / 3600

# Synthetic PIANO-format fixtures generated (via scripts/make_piano_test_fixture.py)
# from the public sample_performance_model.toml. Real PIANO output is
# proprietary and cannot be committed, so these are used instead to validate
# the parser against the exact file format it targets.
SOURCE_MASSES_KG = [51434.0, 68534.0, 81371.0]


@pytest.fixture
def piano_files(test_data_dir):
    base = test_data_dir / 'performance' / 'piano'
    return {
        'cruise': base / 'cruise.txt',
        'climb': base / 'climb.txt',
        'descent': base / 'descent.txt',
    }


@pytest.fixture
def source_table():
    """The original climb/cruise/descent flight_performance tables that the
    PIANO fixtures were generated from (see scripts/make_piano_test_fixture.py),
    concatenated into a single DataFrame (phases distinguished by ROCD sign,
    same as PianoData.flight_performance)."""
    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, LegacyPerformanceModel)
    frames = []
    for table in (
        model.climb_flight_performance,
        model.cruise_flight_performance,
        model.descent_flight_performance,
    ):
        frames.append(
            pd.DataFrame(
                [row[: len(table.cols)] for row in table.data],
                columns=[c.lower() for c in table.cols],
            )
        )
    return pd.concat(frames, ignore_index=True)


def _flight_performance_df(piano_data: PianoData) -> pd.DataFrame:
    cols = ['fl', 'mass', 'tas', 'rocd', 'fuel_flow']
    frames = [
        pd.DataFrame(piano_data.flight_performance.climb, columns=cols),
        pd.DataFrame(piano_data.flight_performance.cruise, columns=cols),
        pd.DataFrame(piano_data.flight_performance.descent, columns=cols),
    ]
    return pd.concat(frames, ignore_index=True)


def test_piano_data_load_recovers_source_table(piano_files, source_table):
    climb_masses_lb = [m / LB_TO_KG for m in SOURCE_MASSES_KG]

    piano_data = PianoData.load(
        piano_files['cruise'],
        piano_files['climb'],
        piano_files['descent'],
        climb_masses_lb=climb_masses_lb,
    )

    assert piano_data.maximum_altitude_ft == 41000
    assert piano_data.design_mach == pytest.approx(0.8, abs=1e-3)
    assert piano_data.climb_mach == pytest.approx(0.8, abs=1e-3)
    assert piano_data.descent_mach == pytest.approx(0.8, abs=1e-3)

    recon = _flight_performance_df(piano_data)
    orig = source_table

    # Every mass present in the source table shows up in each phase.
    for rocd_filter, label in [
        (lambda d: d.rocd > 1e-6, 'climb'),
        (lambda d: d.rocd < -1e-6, 'descent'),
    ]:
        recon_masses = set(recon[rocd_filter(recon)].mass.round().unique())
        orig_masses = set(orig[rocd_filter(orig)].mass.round().unique())
        assert orig_masses <= recon_masses, label

    # ROCD and fuel_flow are reconstructed from the synthetic PIANO text
    # (via finite-difference burn/time integration), so check them against
    # the source table's values at matching (fl, mass) points, not just that
    # rows exist.
    for rocd_filter, label, tol in [
        (lambda d: d.rocd.abs() <= 1e-6, 'cruise', 0.01),
        (lambda d: d.rocd > 1e-6, 'climb', 0.05),
        (lambda d: d.rocd < -1e-6, 'descent', 0.05),
    ]:
        o = orig[rocd_filter(orig)]
        r = recon[rocd_filter(recon)]
        merged = o.merge(r, on=['fl', 'mass'], suffixes=('_orig', '_recon'))
        assert len(merged) > 0, f'no matching {label} rows to compare'
        rel_err = (
            (merged.fuel_flow_recon - merged.fuel_flow_orig) / merged.fuel_flow_orig
        ).abs()
        assert rel_err.max() < tol, f'{label} fuel_flow mismatch: {rel_err.max()}'


def test_piano_command_produces_loadable_model(piano_files, tmp_path):
    """End-to-end: PianoData -> piano TOML -> PerformanceModel.load(), using
    the same helpers the `make-piano-performance-model` CLI command uses."""
    climb_masses_lb = [m / LB_TO_KG for m in SOURCE_MASSES_KG]

    piano_data = PianoData.load(
        piano_files['cruise'],
        piano_files['climb'],
        piano_files['descent'],
        climb_masses_lb=climb_masses_lb,
    )

    edb_data = EDBEntry.get_engine(
        config.file_location('engines/sample_edb.xlsx'), '01P11CM121'
    )
    lto = edb_data.make_lto_performance((0.07, 0.30, 0.85, 1.0))
    lto_dump = LTOPerformanceInput.from_internal(lto).model_dump()

    cols = ['fl', 'mass', 'tas', 'rocd', 'fuel_flow']
    output_file = tmp_path / 'piano_model.toml'
    write_piano_performance_toml(
        str(output_file),
        aircraft_name='B738',
        aircraft_class='narrow',
        isa_offset=0,
        maximum_altitude_ft=piano_data.maximum_altitude_ft,
        maximum_payload_kg=22422,
        maximum_payload_source='test fixture value',
        number_of_engines=2,
        apu_name=None,
        lto_dump=lto_dump,
        speeds_dump=piano_data.speeds.model_dump(),
        climb_flight_performance=dict(
            cols=cols, data=piano_data.flight_performance.climb
        ),
        cruise_flight_performance=dict(
            cols=cols, data=piano_data.flight_performance.cruise
        ),
        descent_flight_performance=dict(
            cols=cols, data=piano_data.flight_performance.descent
        ),
    )

    toml_text = output_file.read_text()
    assert 'maximum_payload_kg = 22422' in toml_text
    assert '# test fixture value' in toml_text

    model = PerformanceModel.load(output_file)
    assert isinstance(model, PianoPerformanceModel)
    assert model.maximum_mass == max(SOURCE_MASSES_KG)
    assert model.lto_performance is not None
    assert model.lto_performance.ICAO_UID == '01P11CM121'


def _cruise_row(mass_lb: float, alt_ft: float, mach: float) -> str:
    # Distinct, easily-identifiable TAS/fuel_flow per Mach -- not physically
    # derived, just unique markers so the test can tell which row was picked.
    tas_kts = mach * 700.0
    ff_lbhr = mach * 10000.0
    return (
        f'  {mass_lb:9.1f}  {alt_ft:8.1f}  {mach:.3f}    |    '
        f'{tas_kts:6.1f}  {tas_kts:6.1f}  {0.0:8.1f}  {0.0:8.1f}  '
        f'{0.0:6.2f}  {ff_lbhr:8.1f}'
    )


def test_piano_data_selects_cas_equivalent_mach_below_crossover(tmp_path):
    """With a multi-Mach cruise sweep, PianoData.load must select, at each
    FL, the Mach matching how the aircraft is actually flown there: the
    design Mach above the CAS/Mach crossover altitude, and the CAS-equivalent
    Mach below it -- not always the design Mach.

    design_mach=0.8, cas_low=250kts, cas_high=300kts gives a crossover
    altitude of ~30,600 ft (FL306). So FL100/FL200 are below crossover and
    should select the CAS-equivalent Mach (computed independently here);
    FL350 is above crossover and should select the design Mach exactly.
    """
    mass_lb = 150000.0

    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(
        '\n'.join(
            [
                '  Cruise table (synthetic, multi-Mach sweep)',
                '',
                '  Mass  Altitude  Mach  |  TAS   CAS   Drag  MCR.%  L/D  FuelFlow',
                '',
                _cruise_row(mass_lb, 10000.0, 0.40),
                _cruise_row(mass_lb, 10000.0, 0.55),
                _cruise_row(mass_lb, 10000.0, 0.70),
                _cruise_row(mass_lb, 20000.0, 0.50),
                _cruise_row(mass_lb, 20000.0, 0.65),
                _cruise_row(mass_lb, 20000.0, 0.80),
                _cruise_row(mass_lb, 35000.0, 0.60),
                _cruise_row(mass_lb, 35000.0, 0.75),
                _cruise_row(mass_lb, 35000.0, 0.80),
                '',
            ]
        )
    )

    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(
        '\n'.join(
            [
                ' Airspeed schedule   250./ 300.kcas/ mach 0.800 above 29000.feet',
                '',
                ' Climb details',
                '',
                '     0.0    0.0    0.0    0.0    0.0  3000.0    0.0    0.0',
                '  5000.0   50.0    0.0  100.0    0.0  2800.0    0.0    0.0',
                ' 10000.0  100.0    0.0  200.0    0.0  2600.0    0.0    0.0',
                ' 20000.0  200.0    0.0  400.0    0.0  2200.0    0.0    0.0',
                ' 35000.0  400.0    0.0  800.0    0.0  1500.0    0.0    0.0',
                '',
            ]
        )
    )

    descent_file = tmp_path / 'descent.txt'
    descent_file.write_text(
        '\n'.join(
            [
                ' Descent from FL350',
                '',
                f' Mass {mass_lb:.1f}',
                ' mach 0.780 above 29000.feet/ 300./ 250.kcas',
                '',
                ' Descent details',
                '',
                ' 35000.0    0.0    0.0    0.0  1500.0    0.0',
                ' 20000.0  100.0    0.0  200.0  1600.0    0.0',
                ' 10000.0  200.0    0.0  400.0  1700.0    0.0',
                '     0.0  300.0    0.0  600.0  1800.0    0.0',
                '',
            ]
        )
    )

    piano_data = PianoData.load(
        cruise_file,
        climb_file,
        descent_file,
        climb_masses_lb=[mass_lb],
        design_mach=0.8,
        cas_low_kts=250.0,
        cas_high_kts=300.0,
    )

    cruise = pd.DataFrame(
        piano_data.flight_performance.cruise,
        columns=['fl', 'mass', 'tas', 'rocd', 'fuel_flow'],
    )

    expected_mach_by_fl = {100.0: 0.55, 200.0: 0.65, 350.0: 0.80}
    for fl, expected_mach in expected_mach_by_fl.items():
        row = cruise[cruise.fl == fl]
        assert len(row) == 1, f'expected exactly one cruise row at FL{fl}'
        expected_tas = expected_mach * 700.0 * KNOTS_TO_MPS
        expected_ff = expected_mach * 10000.0 * LBHR_TO_KGS
        assert row.tas.iloc[0] == pytest.approx(expected_tas), fl
        assert row.fuel_flow.iloc[0] == pytest.approx(expected_ff), fl


def test_piano_data_load_rejects_climb_mass_count_mismatch(piano_files):
    """climb_masses_lb must have exactly one mass per climb block -- a
    mismatch must raise, not silently truncate to the shorter of the two
    (see the module docstring's note on data loss)."""
    # The fixture climb file has 3 blocks; supply only 2 masses.
    with pytest.raises(
        ValueError,
        match=r'climb_masses_lb has 2 masses but the climb file has 3 blocks',
    ):
        PianoData.load(
            piano_files['cruise'],
            piano_files['climb'],
            piano_files['descent'],
            climb_masses_lb=[113392.4, 151091.4],
        )


def test_piano_data_load_climb_masses_cannot_be_inferred(tmp_path):
    """If climb_masses_lb isn't supplied, the auto-inference fallback proxies
    from cruise masses -- but if there aren't enough distinct cruise masses
    to cover every climb block, that must raise rather than silently proxy
    with fewer masses than blocks."""
    mass_lb = 150000.0

    # Cruise file: only 1 mass. Climb file: 2 blocks -- more than the cruise
    # proxy can cover.
    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(
        '\n'.join(
            [
                '  Cruise table (single mass)',
                '',
                '  Mass  Altitude  Mach  |  TAS   CAS   Drag  MCR.%  L/D  FuelFlow',
                '',
                _cruise_row(mass_lb, 35000.0, 0.80),
                '',
            ]
        )
    )

    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(
        '\n'.join(
            [
                ' Airspeed schedule   250./ 300.kcas/ mach 0.800 above 29000.feet',
                '',
                ' Climb details',
                '',
                '     0.0    0.0    0.0    0.0    0.0  3000.0    0.0    0.0',
                ' 20000.0  200.0    0.0  400.0    0.0  2200.0    0.0    0.0',
                '',
                ' Climb details',
                '',
                '     0.0    0.0    0.0    0.0    0.0  2900.0    0.0    0.0',
                ' 20000.0  200.0    0.0  400.0    0.0  2100.0    0.0    0.0',
                '',
            ]
        )
    )

    descent_file = tmp_path / 'descent.txt'
    descent_file.write_text(
        '\n'.join(
            [
                ' Descent from FL350',
                '',
                f' Mass {mass_lb:.1f}',
                ' mach 0.780 above 29000.feet/ 300./ 250.kcas',
                '',
                ' Descent details',
                '',
                ' 35000.0    0.0    0.0    0.0  1500.0    0.0',
                '     0.0  300.0    0.0  600.0  1800.0    0.0',
                '',
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match=r'climb_masses_lb not supplied and cannot be inferred',
    ):
        PianoData.load(cruise_file, climb_file, descent_file)


def test_piano_data_load_requires_explicit_schedule_when_header_missing(tmp_path):
    """Many real PIANO climb exports have no Airspeed schedule header line
    at all (depends on export settings, not aircraft type) -- if it's
    missing, loading must require cas_low_kts/cas_high_kts/climb_mach
    explicitly rather than silently defaulting."""
    mass_lb = 150000.0

    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(
        '\n'.join(
            [
                '  Cruise table',
                '',
                '  Mass  Altitude  Mach  |  TAS   CAS   Drag  MCR.%  L/D  FuelFlow',
                '',
                _cruise_row(mass_lb, 35000.0, 0.80),
                '',
            ]
        )
    )

    # No "Airspeed schedule" header line -- matches real PIANO exports that
    # jump straight from setup/loading messages to "Climb details".
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(
        '\n'.join(
            [
                ' Loading "Test Aircraft"....checking..',
                ' Climb details',
                '',
                '     0.0    0.0    0.0    0.0    0.0  3000.0    0.0    0.0',
                ' 20000.0  200.0    0.0  400.0    0.0  2200.0    0.0    0.0',
                '',
            ]
        )
    )

    descent_file = tmp_path / 'descent.txt'
    descent_file.write_text(
        '\n'.join(
            [
                ' Descent from FL350',
                '',
                f' Mass {mass_lb:.1f}',
                ' mach 0.780 above 29000.feet/ 300./ 250.kcas',
                '',
                ' Descent details',
                '',
                ' 35000.0    0.0    0.0    0.0  1500.0    0.0',
                '     0.0  300.0    0.0  600.0  1800.0    0.0',
                '',
            ]
        )
    )

    # Nothing supplied -- must raise, not silently default.
    with pytest.raises(
        ValueError,
        match=r'Climb speed schedule not found in file header.*'
        r'cas_low_kts, cas_high_kts, climb_mach',
    ):
        PianoData.load(cruise_file, climb_file, descent_file, climb_masses_lb=[mass_lb])

    # Partially supplied -- still must raise, listing only what's missing.
    with pytest.raises(ValueError, match=r'Pass climb_mach explicitly'):
        PianoData.load(
            cruise_file,
            climb_file,
            descent_file,
            climb_masses_lb=[mass_lb],
            cas_low_kts=250.0,
            cas_high_kts=300.0,
        )

    # All three supplied explicitly -- succeeds.
    piano_data = PianoData.load(
        cruise_file,
        climb_file,
        descent_file,
        climb_masses_lb=[mass_lb],
        cas_low_kts=250.0,
        cas_high_kts=300.0,
        climb_mach=0.78,
    )
    assert piano_data.climb_mach == 0.78
