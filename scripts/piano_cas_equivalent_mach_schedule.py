"""Compute the CAS-equivalent Mach schedule for low-altitude PIANO cruise runs.

A single-design-Mach PIANO cruise export only reports performance at one
fixed Mach number for every altitude. Below the CAS/Mach crossover altitude,
flying at that fixed Mach would exceed the aircraft's CAS operating limit, so
that TAS/fuel_flow doesn't represent an achievable cruise condition (see
AEIC.parsers.piano_reader's crossover-altitude filtering).

To get *physically self-consistent* cruise data below crossover, PIANO needs
to be re-run at each altitude using the Mach number that corresponds to
flying at the aircraft's constant-CAS speed limit at that altitude --
``M_equiv(h)`` -- rather than the design Mach. This script computes that
schedule so it can be handed to PIANO as run parameters (one cruise-table
generation per altitude/Mach pair below crossover).
"""

import argparse

from AEIC.parsers.piano_reader import (
    _cas_to_tas_ms,
    _crossover_altitude_ft,
    _speed_of_sound,
)
from AEIC.units import FEET_TO_METERS


def mach_equivalent_schedule(
    design_mach: float,
    cas_low_kts: float,
    cas_high_kts: float,
    fl_step: int = 20,
    cas_transition_ft: float = 10000.0,
) -> list[tuple[int, float, float]]:
    """Return [(fl, alt_ft, mach_equiv), ...] for every FL below the CAS/Mach
    crossover altitude, at ``fl_step``-FL increments."""

    xover_ft = _crossover_altitude_ft(design_mach, cas_high_kts)
    rows = []
    alt_ft = 0.0
    while alt_ft < xover_ft:
        cas = cas_low_kts if alt_ft < cas_transition_ft else cas_high_kts
        tas_ms = _cas_to_tas_ms(cas, alt_ft)
        mach_equiv = tas_ms / _speed_of_sound(alt_ft * FEET_TO_METERS)
        rows.append((int(round(alt_ft / 100.0)), alt_ft, mach_equiv))
        alt_ft += fl_step * 100.0
    return rows, xover_ft


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--design-mach', type=float, required=True)
    ap.add_argument(
        '--cas-low-kts', type=float, required=True, help='CAS below 10,000 ft.'
    )
    ap.add_argument(
        '--cas-high-kts', type=float, required=True, help='CAS above 10,000 ft.'
    )
    ap.add_argument(
        '--fl-step', type=int, default=20, help='FL increment (default: 20).'
    )
    args = ap.parse_args()

    rows, xover_ft = mach_equivalent_schedule(
        args.design_mach, args.cas_low_kts, args.cas_high_kts, args.fl_step
    )

    print(f'CAS/Mach crossover altitude: {xover_ft:.0f} ft (FL{xover_ft / 100:.0f})')
    print(f'At and above this altitude, use the design Mach ({args.design_mach}).')
    print()
    print('Below crossover, run PIANO cruise at these (FL, Mach) pairs instead:')
    print(f'{"FL":>6}  {"Alt [ft]":>10}  {"Mach":>8}')
    for fl, alt_ft, mach_equiv in rows:
        print(f'{fl:6d}  {alt_ft:10.0f}  {mach_equiv:8.4f}')


if __name__ == '__main__':
    main()
