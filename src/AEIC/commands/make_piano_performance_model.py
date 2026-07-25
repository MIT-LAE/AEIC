"""Generate a ``piano``-type performance model TOML file from PIANO
cruise/climb/descent text exports.

This is the PIANO-format counterpart to ``make-performance-model legacy``
(:mod:`AEIC.commands.make_performance_model`), which builds a
``legacy``-type performance model from BADA PTF files. It's a separate
command (rather than a `piano` subcommand of `make-performance-model`)
writing a separate ``piano`` model type
(:mod:`AEIC.performance.models.piano`), because PIANO can offer more mass
resolution and per-mass fuel_flow/ROCD variation than the ``legacy`` model's
BADA-shaped validation allows for.
"""

from typing import Any

import click
import tomlkit
from tomlkit import comment, document, nl, table

from AEIC.commands._performance_model_toml import (
    LTO_MODE_KEY_ORDER,
    LTO_MODE_ORDER,
    MODEL_TYPE_COMMENT,
    SPEED_KEY_ORDER,
    SPEED_PHASE_ORDER,
    add_sub_banner,
    add_top_banner,
    fix_empty_comments,
    format_flight_performance,
    set_with_inline,
)
from AEIC.commands.make_performance_model import lto_from_edb, lto_from_toml
from AEIC.config import Config, config
from AEIC.parsers.piano_reader import PianoData
from AEIC.performance.apu import lookup_apu
from AEIC.performance.models.base import LTOPerformanceInput


def write_piano_performance_toml(
    path: str,
    *,
    aircraft_name: str,
    aircraft_class: str,
    isa_offset: int,
    maximum_altitude_ft: int,
    maximum_payload_kg: int,
    maximum_payload_source: str | None,
    number_of_engines: int,
    apu_name: str | None,
    lto_dump: dict[str, Any],
    speeds_dump: dict[str, Any],
    climb_flight_performance: dict[str, Any],
    cruise_flight_performance: dict[str, Any],
    descent_flight_performance: dict[str, Any],
) -> None:
    doc = document()
    doc.add(comment(MODEL_TYPE_COMMENT))
    doc['model_type'] = 'piano'

    doc.add(nl())
    add_top_banner(
        doc, 'COMMON FIELDS', 'Fields common to all performance model types.'
    )
    doc.add(nl())

    set_with_inline(doc, 'aircraft_name', aircraft_name)
    set_with_inline(doc, 'aircraft_class', aircraft_class)
    set_with_inline(doc, 'ISA_offset', isa_offset)
    set_with_inline(doc, 'maximum_altitude_ft', maximum_altitude_ft)
    set_with_inline(doc, 'maximum_payload_kg', maximum_payload_kg)
    if maximum_payload_source is not None:
        doc['maximum_payload_kg'].comment(maximum_payload_source)
    set_with_inline(doc, 'number_of_engines', number_of_engines)
    if apu_name is not None:
        set_with_inline(doc, 'APU_name', apu_name)

    doc.add(nl())
    add_sub_banner(doc, 'Speed data')

    speeds_super = table(True)
    for phase in SPEED_PHASE_ORDER:
        if phase not in speeds_dump:
            continue
        phase_tbl = table()
        phase_data = speeds_dump[phase]
        for key in SPEED_KEY_ORDER:
            if key in phase_data:
                phase_tbl[key] = phase_data[key]
        speeds_super.append(phase, phase_tbl)
    doc['speeds'] = speeds_super

    doc.add(nl())
    add_sub_banner(doc, 'LTO data')

    lto_tbl = table()
    lto_tbl['source'] = lto_dump['source']
    lto_tbl['ICAO_UID'] = lto_dump['ICAO_UID']
    if lto_dump['source'] == 'EDB':
        lto_tbl['ICAO_UID'].comment('Add UID for EDB data')
    lto_tbl['rated_thrust'] = lto_dump['rated_thrust']

    mode_data = lto_dump.get('mode_data', {})
    mode_super = table(True)
    for mode in LTO_MODE_ORDER:
        if mode not in mode_data:
            continue
        mode_tbl = table()
        md = mode_data[mode]
        for key in LTO_MODE_KEY_ORDER:
            if key in md:
                mode_tbl[key] = md[key]
        mode_super.append(mode, mode_tbl)
    lto_tbl.append('mode_data', mode_super)
    doc['LTO_performance'] = lto_tbl

    body = fix_empty_comments(tomlkit.dumps(doc))
    if not body.endswith('\n'):
        body += '\n'

    trailer_doc = document()
    trailer_doc.add(nl())
    trailer_doc.add(nl())
    add_top_banner(trailer_doc, 'MODEL-TYPE SPECIFIC FIELDS')
    trailer_doc.add(nl())
    add_sub_banner(trailer_doc, 'Performance table data.')
    trailer_doc.add(nl())
    trailer = fix_empty_comments(tomlkit.dumps(trailer_doc))

    climb_fp_section = format_flight_performance(
        'climb', climb_flight_performance['cols'], climb_flight_performance['data']
    )
    cruise_fp_section = format_flight_performance(
        'cruise', cruise_flight_performance['cols'], cruise_flight_performance['data']
    )
    descent_fp_section = format_flight_performance(
        'descent',
        descent_flight_performance['cols'],
        descent_flight_performance['data'],
    )

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(body)
        fp.write(trailer)
        fp.write(climb_fp_section)
        fp.write('\n')
        fp.write(cruise_fp_section)
        fp.write('\n')
        fp.write(descent_fp_section)


@click.command(
    short_help='Create a piano-type performance model from PIANO text exports.',
    help="""Generate a performance model from PIANO cruise/climb/descent text
    exports. The LTO data can come either from the EDB or from a user-provided
    TOML file, same as `make-performance-model legacy`. Unlike BADA, PIANO can
    provide more than three masses per phase, with genuine per-mass
    fuel_flow/ROCD variation in climb/descent; all of it is kept.""",
)
@click.option(
    '--output-file',
    type=click.Path(),
    required=True,
    help='Output TOML file to write extracted data.',
)
@click.option(
    '--lto-source',
    type=click.Choice(['edb', 'custom']),
    required=True,
    help='Source of LTO performance data.',
)
@click.option(
    '--engine-file',
    help='Input engine database file.',
)
@click.option(
    '--engine-uid',
    type=str,
    help='UID of the engine to extract data for.',
)
@click.option(
    '--thrust-fractions',
    nargs=4,
    type=float,
    default=(0.07, 0.30, 0.85, 1.0),
    help='Thrust fractions for LTO modes: idle, approach, climb, takeoff.',
)
@click.option(
    '--lto-file',
    type=click.Path(exists=True),
    help='Input LTO TOML file.',
)
@click.option(
    '--cruise-file',
    type=click.Path(exists=True),
    required=True,
    help='PIANO cruise table file (formatted text export).',
)
@click.option(
    '--climb-file',
    type=click.Path(exists=True),
    required=True,
    help='PIANO climb file (formatted text export).',
)
@click.option(
    '--descent-file',
    type=click.Path(exists=True),
    required=True,
    help='PIANO descent file (formatted text export).',
)
@click.option(
    '--aircraft-name',
    required=True,
    help='Aircraft name (for documentation only).',
)
@click.option(
    '--aircraft-class',
    required=True,
    type=click.Choice(['wide', 'narrow', 'small', 'freight']),
)
@click.option(
    '--number-of-engines',
    type=click.IntRange(1, 8),
    required=True,
    help='Number of engines on the aircraft (1-8).',
)
@click.option(
    '--maximum-payload',
    type=click.IntRange(min=1),
    required=True,
    help='Maximum payload in kg. Not present in PIANO climb/cruise/descent '
    'files, so this must be supplied explicitly (e.g. from MZFW - OEW).',
)
@click.option(
    '--maximum-payload-source',
    type=str,
    default=None,
    help='Optional note on where --maximum-payload came from (e.g. '
    '"MZFW - OEW from piano_step8_instructions.csv"). Written as a comment '
    'next to maximum_payload_kg in the output TOML, purely for future '
    'readers -- there is otherwise no record of how this number was chosen.',
)
@click.option(
    '--apu-name',
    required=False,
    help='Name of the APU used on the aircraft.',
)
@click.option(
    '--climb-masses-lb',
    type=str,
    default=None,
    help='Comma-separated starting masses [lb], matching climb block order. '
    'Defaults to the heaviest cruise masses as a rough proxy; for accurate '
    'mass-ROCD mapping, pass this explicitly.',
)
@click.option(
    '--design-mach',
    type=float,
    default=None,
    help='Cruise design Mach (default: inferred from cruise table).',
)
@click.option(
    '--max-alt-ft',
    type=int,
    default=None,
    help='Maximum operating altitude [ft] (default: from max climb altitude).',
)
@click.option(
    '--cas-high-kts',
    type=float,
    default=None,
    help='High-speed CAS for climb [kts] (default: read from climb file header).',
)
@click.option(
    '--cas-low-kts',
    type=float,
    default=None,
    help='Low-speed CAS below FL100 [kts] (default: read from climb file '
    'header, else 250).',
)
@click.option(
    '--climb-mach',
    type=float,
    default=None,
    help='Operational climb Mach; overrides climb file header.',
)
@click.option(
    '--descent-mach',
    type=float,
    default=None,
    help='Operational descent Mach; overrides descent file header.',
)
@click.option(
    '--descent-cas-kts',
    type=float,
    default=None,
    help='Operational descent high CAS [kts]; overrides descent file header.',
)
def make_piano_performance_model(
    output_file,
    lto_source,
    engine_file,
    engine_uid,
    thrust_fractions,
    lto_file,
    cruise_file,
    climb_file,
    descent_file,
    aircraft_name,
    aircraft_class,
    number_of_engines,
    maximum_payload,
    maximum_payload_source,
    apu_name,
    climb_masses_lb,
    design_mach,
    max_alt_ft,
    cas_high_kts,
    cas_low_kts,
    climb_mach,
    descent_mach,
    descent_cas_kts,
):
    Config.load()

    if apu_name is not None and lookup_apu(apu_name) is None:
        raise click.UsageError(f'APU "{apu_name}" not found in APU database.')
    if engine_file is not None:
        engine_file = config.file_location(engine_file)

    # LTO data comes either from the Emissions Databank (EDB) or user provided TOML file
    match lto_source:
        case 'edb':
            lto = lto_from_edb(engine_file, engine_uid, thrust_fractions)
        case 'custom':
            lto = lto_from_toml(lto_file)
        case _:
            raise click.UsageError(f'Unsupported LTO source: {lto_source}')
    lto_dump = LTOPerformanceInput.from_internal(lto).model_dump()

    # Parse PIANO performance files.
    climb_masses = (
        [float(x) for x in climb_masses_lb.split(',')] if climb_masses_lb else None
    )
    piano_data = PianoData.load(
        cruise_file,
        climb_file,
        descent_file,
        climb_masses_lb=climb_masses,
        design_mach=design_mach,
        max_alt_ft=max_alt_ft,
        cas_high_kts=cas_high_kts,
        cas_low_kts=cas_low_kts,
        climb_mach=climb_mach,
        descent_mach=descent_mach,
        descent_cas_kts=descent_cas_kts,
    )

    cols = ['fl', 'mass', 'tas', 'rocd', 'fuel_flow']
    write_piano_performance_toml(
        output_file,
        aircraft_name=aircraft_name,
        aircraft_class=aircraft_class,
        isa_offset=0,
        maximum_altitude_ft=piano_data.maximum_altitude_ft,
        maximum_payload_kg=maximum_payload,
        maximum_payload_source=maximum_payload_source,
        number_of_engines=number_of_engines,
        apu_name=apu_name,
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
