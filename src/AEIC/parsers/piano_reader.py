"""Parser for PIANO performance data.

PIANO exports climb, cruise and descent performance as separate text files.
This module parses the *formatted* "Text" export (the variant with column
headers and row labels like ``maxSAR``) and converts it into the
``[fl, mass, tas, rocd, fuel_flow]`` flight-performance table format used by
the legacy performance model (see :mod:`AEIC.performance.models.legacy` and
the ``piano`` subcommand of :mod:`AEIC.commands.make_performance_model`).

Unformatted (bare-numbers) PIANO exports lack the ``|`` marker and Mach-row
labels needed for reliable parsing and are not supported.

The cruise file may report either a single design Mach per (mass, altitude)
or a full Mach sweep (several rows per altitude). A single fixed Mach is
only physically realistic above the CAS/Mach crossover altitude -- below it,
flying at that Mach would exceed the aircraft's CAS operating limit. If the
cruise file has a Mach sweep, ``_select_cruise_rows`` picks, at each
altitude, the Mach that matches how the aircraft is actually flown there
(CAS-equivalent Mach below crossover, design Mach above it). For a
single-Mach cruise file this selection is a no-op.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from AEIC.performance.types import SpeedData, Speeds
from AEIC.units import FEET_TO_METERS, FPM_TO_MPS, KNOTS_TO_MPS

logger = logging.getLogger(__name__)

LB_TO_KG = 1 / 2.20462
"""Unit conversion factor for pounds to kilograms. (Not in AEIC.units, since
BADA's PTF files -- the only other performance-data source -- report masses
in kg already and never needed this.)"""

LBHR_TO_KGS = LB_TO_KG / 3600
"""Unit conversion factor for pounds per hour to kilograms per second."""


# ---------------------------------------------------------------------------
# Standard atmosphere (ISA) — for CAS <-> TAS conversion
# ---------------------------------------------------------------------------
def _isa_temperature(alt_m: float) -> float:
    return max(288.15 - 0.0065 * alt_m, 216.65)


def _isa_pressure(alt_m: float) -> float:
    if alt_m <= 11000.0:
        return 101325.0 * (_isa_temperature(alt_m) / 288.15) ** 5.2561
    return 22632.1 * np.exp(-0.0001577 * (alt_m - 11000.0))


def _isa_density(alt_m: float) -> float:
    return _isa_pressure(alt_m) / (287.05 * _isa_temperature(alt_m))


def _speed_of_sound(alt_m: float) -> float:
    return (1.4 * 287.05 * _isa_temperature(alt_m)) ** 0.5


def _cas_to_tas_ms(cas_kts: float, alt_ft: float) -> float:
    """CAS [kts] -> TAS [m/s] using the compressible ISA formula (ICAO Doc 8643).

    Derivation: CAS is defined so that qc/P0 = f(CAS/a0). At altitude, the
    same qc gives a different Mach via qc/P = f(M), and TAS = M*a. The TAS
    relation is TAS^2 = gamma*(2/(gamma-1))*(P/rho)*g where g = f^-1(qc/P).
    The gamma factor (=1.4) is often mishandled — at sea level this collapses
    to CAS = TAS.
    """
    cas_ms = cas_kts * KNOTS_TO_MPS
    alt_m = alt_ft * FEET_TO_METERS
    pressure = _isa_pressure(alt_m)
    density = _isa_density(alt_m)
    a0 = 340.294  # speed of sound at sea level [m/s]
    gamma = 1.4
    qc_p0 = (1.0 + 0.2 * (cas_ms / a0) ** 2) ** 3.5 - 1.0
    f = (qc_p0 * 101325.0 / pressure + 1.0) ** (0.4 / 1.4) - 1.0
    return (gamma * (2.0 / 0.4) * (pressure / density) * f) ** 0.5


def _mach_to_tas_ms(mach: float, alt_ft: float) -> float:
    return mach * _speed_of_sound(alt_ft * FEET_TO_METERS)


def _crossover_altitude_ft(mach: float, cas_kts: float) -> float:
    """Altitude [ft] at which constant-CAS flight first reaches the Mach limit.

    During climb at constant CAS, TAS rises and Mach increases with altitude.
    At the crossover, Mach(CAS, alt) = design_mach and the pilot switches to
    the Mach schedule.
    """
    for alt_ft in range(0, 50001, 100):
        tas_ms = _cas_to_tas_ms(cas_kts, alt_ft)
        a_ms = _speed_of_sound(alt_ft * FEET_TO_METERS)
        if tas_ms / a_ms >= mach:
            return float(alt_ft)
    return 29000.0  # fallback — typical narrow-body transition


# ---------------------------------------------------------------------------
# Build TAS lookup function from cruise table + speed schedule
# ---------------------------------------------------------------------------
def _build_tas_fn(
    cruise_df: pd.DataFrame,
    design_mach: float,
    cas_high_kts: float = 280.0,
    cas_low_kts: float = 250.0,
    climb_mach: float | None = None,
):
    """Return a callable ``tas_at(alt_ft) -> TAS [m/s]`` covering 0-50,000 ft.

    Priority:
      1. Cruise table TAS (at design Mach) — exact PIANO value.
      2. Above the CAS/Mach crossover altitude -> climb_mach TAS (defaults to
         design_mach if not supplied separately).
      3. Below crossover -> CAS schedule: cas_high above FL100, cas_low below.

    climb_mach: Mach at which the CAS/Mach transition occurs in climb/descent.
                May differ from design_mach (e.g. an airline uses an M0.762
                climb schedule while the aircraft's design cruise Mach is
                M0.82).
    """
    _climb_mach = climb_mach if climb_mach is not None else design_mach
    cruise_lookup = dict(
        zip(cruise_df['alt_ft'].round(0).astype(int), cruise_df['tas_ms'])
    )
    xover_ft = _crossover_altitude_ft(_climb_mach, cas_high_kts)

    def tas_at(alt_ft: float) -> float:
        key = int(round(alt_ft))
        if key in cruise_lookup:
            return cruise_lookup[key]  # 1: exact PIANO value
        if alt_ft >= xover_ft:
            return _mach_to_tas_ms(_climb_mach, alt_ft)  # 2: Mach above crossover
        cas = cas_low_kts if alt_ft < 10000.0 else cas_high_kts
        return _cas_to_tas_ms(cas, alt_ft)  # 3: CAS schedule

    # Validate: no TAS discontinuity > 30 m/s within the pure CAS-schedule
    # region (i.e. below the first cruise-table entry and outside the
    # 10,000 ft band). The jump AT the first cruise-table entry is
    # intentional — it reflects the real CAS-to-Mach transition when the
    # aircraft levels off at cruise altitude.
    if cruise_lookup:
        min_cruise_alt = min(cruise_lookup.keys())
        check_alts = np.arange(0, min_cruise_alt, 500, dtype=float)
        if len(check_alts) > 1:
            check_tas = np.array([tas_at(a) for a in check_alts])
            jumps = np.abs(np.diff(check_tas))
            mask = ~((check_alts[:-1] >= 9000) & (check_alts[:-1] <= 11000))
            if mask.any():
                assert jumps[mask].max() < 30.0, (
                    f'Unexpected TAS discontinuity {jumps[mask].max():.1f} m/s '
                    f'below cruise-table minimum ({min_cruise_alt} ft) and outside '
                    f'the 10,000 ft acceleration band (design_mach={design_mach}, '
                    f'xover_ft={xover_ft:.0f})'
                )

    return tas_at, xover_ft


# ---------------------------------------------------------------------------
# Parse PIANO cruise table
# ---------------------------------------------------------------------------
def _parse_cruise(path: str) -> pd.DataFrame:
    """Parse a formatted PIANO cruise table.

    Rows are identified by the ``|`` separator between the three left columns
    (mass, altitude, Mach) and the performance data columns. Rows with
    additional Mach labels (maxSAR, 99%SAR, maxLim) are skipped. A cruise
    file may contain either a single Mach per (mass, altitude) -- the
    "single design-Mach" export style -- or a full Mach sweep with several
    rows per (mass, altitude); both are retained here, one DataFrame row
    each. Callers that need one row per (mass, altitude) select among them
    (see ``_select_cruise_rows``).

    Returns a DataFrame with columns: mass_kg, alt_ft, fl, mach, tas_ms,
    fuel_flow_kgs.
    """
    rows = []
    with open(path, errors='ignore') as f:
        for line in f:
            if '|' not in line:
                continue
            left, right = line.split('|', 1)
            lp = left.split()
            rp = right.split()
            # Expect exactly 3 tokens on the left: mass, alt, mach.
            if len(lp) != 3 or len(rp) < 6:
                continue
            try:
                mass_lb = float(lp[0])
                alt_ft = float(lp[1])
                mach = float(lp[2])
                # rp columns: TAS, CAS, Drag, MCR%, L/D, FuelFlow, ...
                tas_kts = float(rp[0])
                ff_lbhr = float(rp[5])
            except ValueError:
                continue
            rows.append(
                {
                    'mass_kg': mass_lb * LB_TO_KG,
                    'alt_ft': alt_ft,
                    'fl': alt_ft / 100.0,
                    'mach': mach,
                    'tas_ms': tas_kts * KNOTS_TO_MPS,
                    'fuel_flow_kgs': ff_lbhr * LBHR_TO_KGS,
                }
            )
    if not rows:
        raise ValueError(f'No cruise rows found in cruise file: {path}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Select the physically appropriate cruise Mach at each flight level
# ---------------------------------------------------------------------------
def _select_cruise_rows(
    cruise_df: pd.DataFrame,
    design_mach: float,
    cas_low_kts: float,
    cas_high_kts: float,
) -> pd.DataFrame:
    """Reduce a (possibly multi-Mach) cruise DataFrame to one row per (mass,
    altitude), selecting the row whose Mach is nearest the physically
    appropriate cruise Mach at that altitude: the design Mach above the
    CAS/Mach crossover altitude, or the CAS-equivalent Mach below it.

    A fixed-design-Mach PIANO cruise export reports TAS/fuel_flow for flying
    at ``design_mach`` at every altitude, including altitudes where that
    would exceed the aircraft's CAS operating limit -- an operationally
    unrealistic condition. If the cruise file instead contains a Mach sweep
    (multiple rows per altitude), this selects the row that corresponds to
    how the aircraft is actually flown at each altitude. For a single-Mach
    cruise file, this is a no-op (only one row exists per altitude).
    """
    xover_ft = _crossover_altitude_ft(design_mach, cas_high_kts)

    def target_mach(alt_ft: float) -> float:
        if alt_ft >= xover_ft:
            return design_mach
        cas = cas_low_kts if alt_ft < 10000.0 else cas_high_kts
        tas_ms = _cas_to_tas_ms(cas, alt_ft)
        return tas_ms / _speed_of_sound(alt_ft * FEET_TO_METERS)

    selected_rows = []
    for (_mass_kg, alt_ft), group in cruise_df.groupby(['mass_kg', 'alt_ft']):
        target = target_mach(alt_ft)
        # On an exact tie between two Mach rows equidistant from the target,
        # prefer the lower one: it's the more conservative choice (the
        # higher one risks being on the wrong side of the CAS/Mach limit at
        # this altitude), and sorting on (distance, mach) makes the choice
        # deterministic regardless of file row order.
        distances = (group['mach'] - target).abs()
        ordered = pd.DataFrame(
            {'distance': distances, 'mach': group['mach']}
        ).sort_values(['distance', 'mach'])
        idx = ordered.index[0]
        selected_rows.append(group.loc[idx])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parse speed schedule from PIANO climb file header
# ---------------------------------------------------------------------------
def _parse_climb_schedule(path: str) -> tuple[float, float, float] | None:
    """Extract (cas_low_kts, cas_high_kts, mach) from the PIANO climb file header.

    PIANO formats seen:
      "Airspeed schedule   250./ 268.kcas/ mach 0.762 above 33327.feet"
      "Climb schedule: 250./ 297. kcas / mach 0.769 above 29,068 feet"

    Returns None if the line is not found (caller falls back to defaults).
    """
    pattern = re.compile(
        r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*kcas.*?mach\s*([\d.]+)',
        re.IGNORECASE,
    )
    with open(path, errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1)), float(m.group(2)), float(m.group(3))
            # Stop once data blocks begin — the schedule is in the header section.
            if 'Climb details' in line:
                break
    return None


# ---------------------------------------------------------------------------
# Parse PIANO climb file
# ---------------------------------------------------------------------------
def _parse_climb(path: str) -> list[pd.DataFrame]:
    """Parse a formatted PIANO climb file (one block per starting mass).

    Returns a list of DataFrames (lightest mass first), each with columns:
    alt_ft, fl, time_s, burn_lb, roc_fpm.
    """
    blocks: list[pd.DataFrame] = []
    current_lines: list[str] = []
    in_block = False  # True once we've seen "Climb details"

    with open(path, errors='ignore') as f:
        for raw in f:
            line = raw.rstrip()
            stripped = line.strip()

            if 'Climb details' in line:
                if current_lines:
                    blocks.append(_parse_climb_block(current_lines))
                current_lines = []
                in_block = True
                continue

            if not in_block:
                continue
            if not stripped:
                continue
            # Data row: starts with a digit (altitude value). Any
            # non-numeric, non-blank line inside a block (e.g. column
            # headers, "Rate of Climb < 50" messages) is silently skipped.
            if stripped[0].isdigit():
                current_lines.append(stripped)

    if current_lines:
        blocks.append(_parse_climb_block(current_lines))

    if not blocks:
        raise ValueError(f'No climb blocks found in: {path}')
    return blocks


def _parse_climb_block(lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        p = line.split()
        # Alt, Time, Dist, Burn, FN/eng, R.o.C., Drag, NOx
        if len(p) < 6:
            continue
        try:
            rows.append(
                {
                    'alt_ft': float(p[0]),
                    'fl': float(p[0]) / 100.0,
                    'time_s': float(p[1]),
                    'burn_lb': float(p[3]),
                    'roc_fpm': float(p[5]),
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parse PIANO descent file
# ---------------------------------------------------------------------------
def _parse_descent(path: str) -> list[tuple]:
    """Parse a formatted PIANO descent file (one block per starting mass).

    Returns a list of (mass_kg, mach_descent, xover_ft, cas_high_kts, DataFrame)
    tuples. The DataFrame has columns: alt_ft, fl, time_s, burn_lb, rod_fpm.

    Each block has a metadata header (Mass, Airspeed schedule) followed by
    "Descent details" and then altitude data rows.
    """
    results: list[tuple] = []
    cur_mass = None
    cur_mach = None
    cur_xover = None
    cur_cas = None
    cur_lines: list[str] = []
    in_block = False  # True after "Descent details" line

    def save():
        if cur_lines and cur_mass is not None:
            df = _parse_descent_block(cur_lines)
            results.append((cur_mass * LB_TO_KG, cur_mach, cur_xover, cur_cas, df))

    with open(path, errors='ignore') as f:
        for raw in f:
            line = raw.rstrip()
            stripped = line.strip()

            # New block header: save previous block and reset metadata.
            if line.lstrip().startswith('Descent from'):
                save()
                cur_lines = []
                in_block = False
                continue

            m = re.search(r'\bMass\s+([\d.]+)', line)
            if m:
                cur_mass = float(m.group(1))

            # "mach 0.743 above 29877.feet/ 281./ 250.kcas"
            m2 = re.search(
                r'mach\s+([\d.]+)\s+above\s+([\d.]+).*?/\s*([\d.]+).*?/\s*([\d.]+)',
                line,
            )
            if m2:
                cur_mach = float(m2.group(1))
                cur_xover = float(m2.group(2))
                cur_cas = float(m2.group(3))  # high CAS (e.g. 281 kts)

            if 'Descent details' in line:
                in_block = True
                cur_lines = []
                continue

            if not in_block:
                continue
            if not stripped or not stripped[0].isdigit():
                continue
            cur_lines.append(stripped)

    save()  # save the last block

    if not results:
        raise ValueError(f'No descent blocks found in: {path}')
    return results


def _parse_descent_block(lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        p = line.split()
        # Alt, Time, Dist, Burn, R.o.D., FN/eng
        if len(p) < 5:
            continue
        try:
            rows.append(
                {
                    'alt_ft': float(p[0]),
                    'fl': float(p[0]) / 100.0,
                    'time_s': float(p[1]),
                    'burn_lb': float(p[3]),
                    'rod_fpm': float(p[4]),  # rate of descent (positive number)
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Derive instantaneous fuel flow from cumulative burn
# ---------------------------------------------------------------------------
def _derive_fuel_flow_kgs(df: pd.DataFrame) -> np.ndarray:
    """Compute instantaneous fuel flow [kg/s] from cumulative burn [lb] and
    elapsed time [s] using central finite differences."""
    burn_lb = df['burn_lb'].values
    time_s = df['time_s'].values
    ff_lbs = np.gradient(burn_lb, time_s)  # lb/s at each sample point
    return np.maximum(ff_lbs * LB_TO_KG, 0.0)  # kg/s, no negative values


# ---------------------------------------------------------------------------
# Build flight_performance rows
# ---------------------------------------------------------------------------
@dataclass
class FlightPerformanceByPhase:
    """Flight-performance rows, one dense (FL x mass) grid per phase.

    Each list holds rows of ``[fl, mass_kg, tas_ms, rocd_ms, fuel_flow_kgs]``,
    matching the ``climb_flight_performance``/``cruise_flight_performance``/
    ``descent_flight_performance`` sections of the legacy performance model
    TOML format."""

    climb: list[list[float]]
    cruise: list[list[float]]
    descent: list[list[float]]


def _build_flight_performance(
    cruise_df: pd.DataFrame,
    climb_blocks: list[pd.DataFrame],
    climb_masses_lb: list[float],
    descent_results: list[tuple],
    tas_at,
    tas_at_descent=None,
) -> FlightPerformanceByPhase:
    """Assemble [fl, mass_kg, tas_ms, rocd_ms, fuel_flow_kgs] rows for each of
    the climb, cruise and descent phases.

    All available PIANO masses are used for every phase. Each phase forms a
    dense (FL x mass) grid for bilinear interpolation. TAS is FL-only (from
    the speed schedule). ROCD and fuel_flow vary with both FL and mass (fuel
    flow is FL-only in climb, and ROCD/fuel_flow are both FL-only in
    descent, matching the legacy performance model's requirements).

    ``cruise_df`` must already have at most one row per (mass, altitude) --
    see ``_select_cruise_rows`` for reducing a multi-Mach cruise sweep to
    that form before calling this function.
    """
    # ---- CLIMB ----
    if len(climb_masses_lb) != len(climb_blocks):
        raise ValueError(
            f'climb_masses_lb has {len(climb_masses_lb)} masses but the climb '
            f'file has {len(climb_blocks)} blocks -- pass exactly one mass per '
            f'block, in the same (ascending) order as the file.'
        )
    masses_kg_climb = [m * LB_TO_KG for m in climb_masses_lb]
    blocks = climb_blocks

    # Dense rectangular grid: use the highest-reaching mass (usually the
    # lightest) as the altitude reference. Heavier masses that halt early are
    # padded by np.interp's right-side behaviour (repeating the last valid
    # value) so the grid stays rectangular without truncating the lighter
    # masses' high-FL data.
    ref_block = max(blocks, key=lambda b: b['alt_ft'].max())
    alts_proto = ref_block[ref_block['alt_ft'] > 0]['alt_ft'].values

    # ROCD and fuel_flow per mass, interpolated onto the full altitude grid.
    # np.interp returns fp[-1] for x > xp.max(), so heavy masses that halt
    # before the grid ceiling are extended with their last-valid-row
    # performance.
    roc_by_mass: dict[float, np.ndarray] = {}
    ff_by_mass: dict[float, np.ndarray] = {}
    for m_kg, blk in zip(masses_kg_climb, blocks):
        bs = blk.sort_values('alt_ft')
        ff_raw = _derive_fuel_flow_kgs(bs)
        roc_by_mass[m_kg] = np.interp(
            alts_proto, bs['alt_ft'].values, bs['roc_fpm'].values
        )
        ff_by_mass[m_kg] = np.interp(alts_proto, bs['alt_ft'].values, ff_raw)

    # Keep only altitudes where every mass has strictly positive ROCD. Padded
    # rows (above a heavy mass's halt altitude) carry that mass's last
    # positive ROCD value, so they pass this filter correctly.
    valid = np.array(
        [
            all(roc_by_mass[m_kg][i] > 0 for m_kg in masses_kg_climb)
            for i in range(len(alts_proto))
        ]
    )
    alts_climb = alts_proto[valid]
    for m_kg in masses_kg_climb:
        roc_by_mass[m_kg] = roc_by_mass[m_kg][valid]
        ff_by_mass[m_kg] = ff_by_mass[m_kg][valid]

    climb_rows: list[list[float]] = []

    # CLIMB rows
    for i, alt_ft in enumerate(alts_climb):
        fl = alt_ft / 100.0
        tas_ms = tas_at(alt_ft)
        for m_kg in masses_kg_climb:
            climb_rows.append(
                [
                    fl,
                    round(m_kg),
                    tas_ms,
                    roc_by_mass[m_kg][i] * FPM_TO_MPS,
                    float(ff_by_mass[m_kg][i]),
                ]
            )

    # ---- CRUISE ----
    # cruise_df is expected to already have one row per (mass, altitude) --
    # see _select_cruise_rows, called by the caller before this function.
    # Keep only FLs present for all cruise masses (dense grid requirement).
    n_masses_cruise = cruise_df['mass_kg'].nunique()
    fl_counts = cruise_df.groupby('fl')['mass_kg'].nunique()
    common_fls = fl_counts[fl_counts == n_masses_cruise].index
    cruise_df = cruise_df[cruise_df['fl'].isin(common_fls)].copy()

    cruise_rows: list[list[float]] = []
    for _, crow in cruise_df.iterrows():
        cruise_rows.append(
            [
                crow['fl'],
                round(crow['mass_kg']),
                crow['tas_ms'],
                0.0,
                crow['fuel_flow_kgs'],
            ]
        )

    # ---- DESCENT ----
    # All descent blocks, using starting mass as the mass coordinate.
    # Interpolate every block onto the first block's altitude grid so all
    # masses share identical FL values (dense grid requirement).
    _, _, _, _, ref_desc_df = descent_results[0]
    ref_desc_asc = ref_desc_df.sort_values('alt_ft')
    desc_alts = ref_desc_asc['alt_ft'].values  # ascending; used as interp target

    roc_by_mass_desc: dict[float, np.ndarray] = {}
    ff_by_mass_desc: dict[float, np.ndarray] = {}
    masses_kg_desc: list[float] = []

    for mass_kg, _, _, _, desc_df in descent_results:
        ds = desc_df.sort_values('alt_ft')  # ascending for np.interp xp
        ff_raw = _derive_fuel_flow_kgs(ds)
        roc_by_mass_desc[mass_kg] = np.interp(
            desc_alts, ds['alt_ft'].values, ds['rod_fpm'].values
        )
        ff_by_mass_desc[mass_kg] = np.interp(desc_alts, ds['alt_ft'].values, ff_raw)
        masses_kg_desc.append(mass_kg)

    tas_desc = tas_at_descent if tas_at_descent is not None else tas_at

    descent_rows: list[list[float]] = []
    # Emit rows in descending altitude order (high FL -> FL 0).
    for alt_ft in desc_alts[::-1]:
        fl = alt_ft / 100.0
        tas_ms = tas_desc(alt_ft)
        idx = np.searchsorted(desc_alts, alt_ft)
        for m_kg in masses_kg_desc:
            rod_ms = -abs(roc_by_mass_desc[m_kg][idx] * FPM_TO_MPS)
            ff_kgs = float(ff_by_mass_desc[m_kg][idx])
            descent_rows.append([fl, round(m_kg), tas_ms, rod_ms, ff_kgs])

    return FlightPerformanceByPhase(
        climb=climb_rows, cruise=cruise_rows, descent=descent_rows
    )


@dataclass
class PianoData:
    """Internal representation of PIANO climb/cruise/descent performance
    data, converted to the ``[fl, mass, tas, rocd, fuel_flow]``
    flight-performance table format used by the legacy performance model."""

    maximum_altitude_ft: int
    """Aircraft ceiling [ft], from the highest climb-block altitude."""

    design_mach: float
    """Cruise design Mach number."""

    climb_mach: float
    """Operational climb Mach number (may differ from ``design_mach``)."""

    descent_mach: float
    """Operational descent Mach number."""

    speeds: Speeds
    """Speed schedule (CAS/Mach) for climb, cruise and descent phases."""

    flight_performance: FlightPerformanceByPhase
    """Rows of ``[fl, mass_kg, tas_ms, rocd_ms, fuel_flow_kgs]``, one dense
    (FL x mass) grid per phase, suitable for the ``climb_flight_performance``/
    ``cruise_flight_performance``/``descent_flight_performance`` TOML
    sections."""

    @classmethod
    def load(
        cls,
        cruise_file: str,
        climb_file: str,
        descent_file: str,
        *,
        climb_masses_lb: list[float] | None = None,
        design_mach: float | None = None,
        max_alt_ft: int | None = None,
        cas_high_kts: float | None = None,
        cas_low_kts: float | None = None,
        climb_mach: float | None = None,
        descent_mach: float | None = None,
        descent_cas_kts: float | None = None,
    ) -> Self:
        """Parse PIANO cruise/climb/descent text files and build a dense
        flight-performance table.

        Args:
            cruise_file: PIANO cruise table file (formatted text export).
            climb_file: PIANO climb file (formatted text export).
            descent_file: PIANO descent file (formatted text export).
            climb_masses_lb: Starting masses [lb], matching climb block
                order. If not supplied, the heaviest cruise masses are used
                as a rough proxy (PIANO climb blocks start near MTOW); for
                accurate mass-ROCD mapping, supply this explicitly.
            design_mach: Cruise design Mach. If not supplied, it is inferred
                from the highest-altitude cruise-table row.
            max_alt_ft: Maximum operating altitude [ft]. If not supplied, the
                maximum climb-block altitude is used.
            cas_high_kts: High-speed CAS for climb [kts]. Read from the climb
                file header if present. Many real PIANO climb exports don't
                include this header line at all (it depends on export
                settings, not aircraft type) -- if it's missing, this must be
                supplied explicitly, or loading raises.
            cas_low_kts: Low-speed CAS (below FL100) [kts]. Same as
                cas_high_kts: read from the header if present, otherwise
                required.
            climb_mach: Operational climb Mach. Read from the header if
                present (overrides it if also supplied explicitly); required
                if the header is missing.
            descent_mach: Operational descent Mach; overrides the file header.
            descent_cas_kts: Operational descent high CAS [kts]; overrides
                the file header.

        Raises:
            ValueError: If the climb file has no parseable speed-schedule
                header and cas_low_kts/cas_high_kts/climb_mach aren't all
                supplied explicitly.
        """
        cruise_df = _parse_cruise(cruise_file)
        climb_blocks = _parse_climb(climb_file)

        climb_schedule = _parse_climb_schedule(climb_file)
        if climb_schedule is not None:
            hdr_cas_low, hdr_cas_high, hdr_mach = climb_schedule
        else:
            hdr_cas_low, hdr_cas_high, hdr_mach = None, None, None
            # No schedule in the file to fall back to -- in practice, most
            # PIANO climb exports don't include this header line at all (it
            # depends on export settings, not aircraft type), so silently
            # defaulting here would be wrong far more often than not. Require
            # the caller to supply the real values instead of guessing.
            missing = [
                name
                for name, val in (
                    ('cas_low_kts', cas_low_kts),
                    ('cas_high_kts', cas_high_kts),
                    ('climb_mach', climb_mach),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f'Climb speed schedule not found in file header: {climb_file}. '
                    f'Pass {", ".join(missing)} explicitly.'
                )

        resolved_cas_high = cas_high_kts if cas_high_kts is not None else hdr_cas_high
        resolved_cas_low = cas_low_kts if cas_low_kts is not None else hdr_cas_low

        descent_results = _parse_descent(descent_file)

        if climb_masses_lb is None:
            # Use unique cruise masses as a rough proxy for the climb mass
            # range. For accurate results, pass climb_masses_lb explicitly.
            cruise_masses_lb = sorted(cruise_df['mass_kg'].unique() / LB_TO_KG)
            n_needed = len(climb_blocks)
            if len(cruise_masses_lb) < n_needed:
                raise ValueError(
                    f'climb_masses_lb not supplied and cannot be inferred: the '
                    f'climb file has {n_needed} blocks but the cruise file only '
                    f'has {len(cruise_masses_lb)} distinct masses to use as a '
                    f'proxy. Pass climb_masses_lb explicitly.'
                )
            climb_masses_lb = cruise_masses_lb[-n_needed:]
            logger.warning(
                'climb_masses_lb not supplied; using heaviest cruise masses as '
                'proxy: %s lb',
                [int(m) for m in climb_masses_lb],
            )

        resolved_design_mach = design_mach
        if resolved_design_mach is None:
            # Infer from the cruise table: the design Mach is taken as the
            # highest Mach flown at the highest cruise altitude in the file.
            # For a single-design-Mach cruise export there is exactly one row
            # there, so this is unambiguous. For a multi-Mach sweep, this is
            # a heuristic -- pass design_mach explicitly for a sweep file to
            # avoid relying on it.
            highest_alt = cruise_df['alt_ft'].max()
            candidates = cruise_df[cruise_df['alt_ft'] == highest_alt]
            if len(candidates) > 1:
                logger.warning(
                    'Multiple Mach rows at the highest cruise altitude '
                    '(%.0f ft) in %s and no design_mach given; using the '
                    'highest Mach among them (%.3f). Pass design_mach '
                    'explicitly for a multi-Mach cruise sweep.',
                    highest_alt,
                    cruise_file,
                    candidates['mach'].max(),
                )
            resolved_design_mach = float(
                candidates.loc[candidates['mach'].idxmax(), 'mach']
            )

        # climb_mach: from the file header if available, falling back to
        # design_mach; explicit argument always wins.
        resolved_climb_mach = hdr_mach if hdr_mach is not None else resolved_design_mach
        if climb_mach is not None:
            resolved_climb_mach = climb_mach

        # Reduce the (possibly multi-Mach) cruise table to one row per
        # (mass, altitude) -- the row matching how the aircraft is actually
        # flown at each altitude -- before it's used for either the climb/
        # descent TAS lookup or the cruise flight_performance rows.
        selected_cruise_df = _select_cruise_rows(
            cruise_df, resolved_design_mach, resolved_cas_low, resolved_cas_high
        )

        tas_at, _ = _build_tas_fn(
            selected_cruise_df,
            resolved_design_mach,
            cas_high_kts=resolved_cas_high,
            cas_low_kts=resolved_cas_low,
            climb_mach=resolved_climb_mach,
        )

        # Descent speed schedule — resolved here so tas_at_descent is
        # available before the flight_performance rows are built.
        resolved_descent_mach = descent_results[0][1] or resolved_design_mach
        resolved_descent_cas = descent_results[0][3] or resolved_cas_high
        if descent_mach is not None:
            resolved_descent_mach = descent_mach
        if descent_cas_kts is not None:
            resolved_descent_cas = descent_cas_kts

        tas_at_descent, _ = _build_tas_fn(
            selected_cruise_df,
            resolved_design_mach,
            cas_high_kts=resolved_descent_cas,
            cas_low_kts=resolved_cas_low,
            climb_mach=resolved_descent_mach,
        )

        flight_performance = _build_flight_performance(
            cruise_df=selected_cruise_df,
            climb_blocks=climb_blocks,
            climb_masses_lb=climb_masses_lb,
            descent_results=descent_results,
            tas_at=tas_at,
            tas_at_descent=tas_at_descent,
        )

        resolved_max_alt_ft = (
            max_alt_ft
            if max_alt_ft is not None
            else int(max(b['alt_ft'].max() for b in climb_blocks))
        )

        speeds = Speeds(
            climb=SpeedData(
                cas_low=resolved_cas_low * KNOTS_TO_MPS,
                cas_high=resolved_cas_high * KNOTS_TO_MPS,
                mach=resolved_climb_mach,
            ),
            cruise=SpeedData(
                cas_low=resolved_cas_low * KNOTS_TO_MPS,
                cas_high=resolved_cas_high * KNOTS_TO_MPS,
                mach=resolved_design_mach,
            ),
            descent=SpeedData(
                cas_low=resolved_cas_low * KNOTS_TO_MPS,
                cas_high=resolved_descent_cas * KNOTS_TO_MPS,
                mach=resolved_descent_mach,
            ),
        )

        return cls(
            maximum_altitude_ft=resolved_max_alt_ft,
            design_mach=resolved_design_mach,
            climb_mach=resolved_climb_mach,
            descent_mach=resolved_descent_mach,
            speeds=speeds,
            flight_performance=flight_performance,
        )
