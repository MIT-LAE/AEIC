import tomllib
from typing import Any

import click
import tomlkit
from tomlkit import comment, document, nl, table

from AEIC.config import Config, config

# from AEIC.parsers.lto_reader import parseLTO
from AEIC.parsers.piano_reader import PianoData, PianoOverrides
from AEIC.parsers.ptf_reader import PTFData
from AEIC.performance.apu import lookup_apu
from AEIC.performance.edb import EDBEntry
from AEIC.performance.models.base import LTOPerformanceInput
from AEIC.performance.types import LTOPerformance, SpeedData, Speeds

###########################################
######   AEIC output .toml writer    ######
###########################################

_BANNER_WIDTH = 78

_INLINE_COMMENTS = {
    'aircraft_class': 'wide, narrow, small, freight',
    'number_of_engines': 'Number of engines',
    'APU_name': 'None: APU emissions not calculated',
    'operating_empty_mass_kg': 'kg',
}

_COL_COMMENTS = {
    'fuel_flow': 'kg/s - REQUIRED; OUTPUT COLUMN',
    'fl': 'Flight levels',
    'tas': 'm/s',
    'rocd': 'm/s',
    'mass': 'kg',
    'cas': 'm/s',
    'mach': 'Mach number',
    'time': 's - cumulative from start of phase',
    'distance': 'm - cumulative from start of phase',
    'burn': 'kg - cumulative from start of phase',
    'drag': 'N',
    'fn_per_engine': 'N - net thrust per engine',
    'mcr_pct': 'percent of maximum cruise thrust',
    'lift_to_drag': 'Lift-to-drag ratio',
    'sfc': 'kg/(N.s) - specific fuel consumption',
    'sar': 'm/kg - specific air range',
    'mcl_avail_per_engine': 'N - maximum climb thrust available per engine',
    'rocd_mcl_fix_mach': 'm/s - at maximum climb thrust, fixed Mach',
    'rocd_mcl_fix_cas': 'm/s - at maximum climb thrust, fixed CAS',
    'max_sar': 'Mach at maximum specific air range',
    'sar_99': 'Mach at 99% of maximum specific air range',
    'max_lim': 'Maximum limiting Mach',
    'idle_thrust_altitude': 'm - altitude below which idle thrust is used',
}

_LTO_MODE_ORDER = ('idle', 'approach', 'climb', 'takeoff')
_LTO_MODE_KEY_ORDER = ('thrust_frac', 'fuel_kgs', 'EI_NOx', 'EI_HC', 'EI_CO')
_SPEED_PHASE_ORDER = ('climb', 'cruise', 'descent')
_SPEED_KEY_ORDER = ('cas_low', 'cas_high', 'mach', 'crossover_altitude_m')


def _add_top_banner(doc, title: str, description: str | None = None) -> None:
    doc.add(comment('=' * _BANNER_WIDTH))
    doc.add(comment(''))
    doc.add(comment(f' {title}'))
    doc.add(comment(''))
    if description is not None:
        doc.add(comment(description))


def _add_sub_banner(doc, title: str) -> None:
    doc.add(comment('-' * _BANNER_WIDTH))
    doc.add(comment(''))
    doc.add(comment(title))
    doc.add(comment(''))


def _set_with_inline(tbl, key: str, value: Any) -> None:
    tbl[key] = value
    if key in _INLINE_COMMENTS:
        tbl[key].comment(_INLINE_COMMENTS[key])


def _format_table_section(
    section: str, cols: list[str], data: list[list[float]]
) -> str:
    """Render a tabular section as a string with the right-aligned numeric
    column layout used by the sample performance model file."""
    col_lines = []
    for i, name in enumerate(cols):
        sep = ',' if i < len(cols) - 1 else ''
        quoted = f'"{name}"{sep}'
        if name in _COL_COMMENTS:
            col_lines.append(f'  {quoted}  # {_COL_COMMENTS[name]}')
        else:
            col_lines.append(f'  {quoted}')
    cols_block = 'cols = [\n' + '\n'.join(col_lines) + '\n]'

    # An empty table is valid: it can happen for the descent idle thrust
    # table which may be empty if there is never an idle thrust for that
    # descent profile.
    if not data:
        data_block = 'data = []'
    else:
        cells = [[repr(float(v)) for v in row] for row in data]
        widths = [max(len(row[c]) for row in cells) for c in range(len(cols))]
        data_lines = []
        for row in cells:
            padded = ', '.join(row[c].rjust(widths[c]) for c in range(len(cols)))
            data_lines.append(f'  [ {padded}],')
        data_block = 'data = [\n' + '\n'.join(data_lines) + '\n]'

    return f'[{section}]\n' + cols_block + '\n\n' + data_block + '\n'


def _fix_empty_comments(text: str) -> str:
    """tomlkit renders `comment('')` as `# ` (with a trailing space);
    strip the trailing space so empty banner lines are plain `#`."""
    return text.replace('# \n', '#\n')


def write_performance_toml(
    path: str,
    *,
    model_type: str,
    aircraft_name: str,
    aircraft_class: str,
    isa_offset: int,
    maximum_altitude_ft: int,
    maximum_payload_kg: int,
    number_of_engines: int,
    apu_name: str | None,
    lto_dump: dict[str, Any],
    speeds_dump: dict[str, Any],
    tables: dict[str, dict[str, Any]],
    extra_common: dict[str, Any] | None = None,
) -> None:
    """Write a performance model TOML file.

    Args:
        path: Output file path.
        model_type: Performance model type discriminator.
        aircraft_name: Aircraft name (e.g., "A320").
        aircraft_class: One of wide, narrow, small, freight.
        isa_offset: ISA temperature offset in degrees Celsius.
        maximum_altitude_ft: Aircraft maximum altitude in feet.
        maximum_payload_kg: Aircraft maximum payload in kilograms.
        number_of_engines: Number of engines on the aircraft.
        apu_name: APU name, or None to omit the field.
        lto_dump: Dumped `LTOPerformanceInput` data.
        speeds_dump: Dumped `Speeds` data. Keys with a `None` value are
            omitted.
        tables: Ordered mapping of section name to `{'cols': ..., 'data': ...}`.
            Sections are emitted in mapping order.
        extra_common: Model-specific common-block fields, written after
            `number_of_engines` and before `APU_name`.
    """
    doc = document()
    doc.add(comment('Performance model type (one of: legacy, bada, tasopt, piano).'))
    doc['model_type'] = model_type

    doc.add(nl())
    _add_top_banner(
        doc, 'COMMON FIELDS', 'Fields common to all performance model types.'
    )
    doc.add(nl())

    _set_with_inline(doc, 'aircraft_name', aircraft_name)
    _set_with_inline(doc, 'aircraft_class', aircraft_class)
    _set_with_inline(doc, 'ISA_offset', isa_offset)
    _set_with_inline(doc, 'maximum_altitude_ft', maximum_altitude_ft)
    _set_with_inline(doc, 'maximum_payload_kg', maximum_payload_kg)
    _set_with_inline(doc, 'number_of_engines', number_of_engines)
    for key, value in (extra_common or {}).items():
        if value is not None:
            _set_with_inline(doc, key, value)
    if apu_name is not None:
        _set_with_inline(doc, 'APU_name', apu_name)

    doc.add(nl())
    _add_sub_banner(doc, 'Speed data')

    speeds_super = table(True)
    for phase in _SPEED_PHASE_ORDER:
        if phase not in speeds_dump or speeds_dump[phase] is None:
            continue
        phase_tbl = table()
        phase_data = speeds_dump[phase]
        for key in _SPEED_KEY_ORDER:
            if key in phase_data and phase_data[key] is not None:
                phase_tbl[key] = phase_data[key]
        speeds_super.append(phase, phase_tbl)
    doc['speeds'] = speeds_super

    doc.add(nl())
    _add_sub_banner(doc, 'LTO data')

    lto_tbl = table()
    lto_tbl['source'] = lto_dump['source']
    lto_tbl['ICAO_UID'] = lto_dump['ICAO_UID']
    if lto_dump['source'] == 'EDB':
        lto_tbl['ICAO_UID'].comment('Add UID for EDB data')
    lto_tbl['rated_thrust'] = lto_dump['rated_thrust']

    mode_data = lto_dump.get('mode_data', {})
    mode_super = table(True)
    for mode in _LTO_MODE_ORDER:
        if mode not in mode_data:
            continue
        mode_tbl = table()
        md = mode_data[mode]
        for key in _LTO_MODE_KEY_ORDER:
            if key in md:
                mode_tbl[key] = md[key]
        mode_super.append(mode, mode_tbl)
    lto_tbl.append('mode_data', mode_super)
    doc['LTO_performance'] = lto_tbl

    body = _fix_empty_comments(tomlkit.dumps(doc))
    if not body.endswith('\n'):
        body += '\n'

    trailer_doc = document()
    trailer_doc.add(nl())
    trailer_doc.add(nl())
    _add_top_banner(trailer_doc, 'MODEL-TYPE SPECIFIC FIELDS')
    trailer_doc.add(nl())
    _add_sub_banner(trailer_doc, 'Performance table data.')
    trailer_doc.add(nl())
    trailer = _fix_empty_comments(tomlkit.dumps(trailer_doc))

    sections = [
        _format_table_section(section, tbl['cols'], tbl['data'])
        for section, tbl in tables.items()
    ]

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(body)
        fp.write(trailer)
        fp.write('\n'.join(sections))


###########################################
######  Shared CLI argument parsing  ######
###########################################


def lto_from_edb(engine_file, engine_uid, thrust_fractions) -> LTOPerformance:
    if engine_file is None or engine_uid is None:
        raise click.UsageError(
            'Both --engine-file and --engine-uid must be provided when '
            'using "edb" as the LTO source.'
        )
    edb_data = EDBEntry.get_engine(engine_file, engine_uid)
    return edb_data.make_lto_performance(thrust_fractions)


def lto_from_toml(lto_file) -> LTOPerformance:
    if lto_file is None:
        raise click.UsageError(
            '--lto-file must be provided when using "lto-file" as the LTO source.'
        )

    with open(lto_file, 'rb') as fp:
        toml_data = tomllib.load(fp)

    lto_dict = toml_data.get('LTO_performance', toml_data)
    lto_input = LTOPerformanceInput.model_validate(lto_dict)
    return lto_input.convert()


def resolve_lto(
    lto_source, engine_file, engine_uid, thrust_fractions, lto_file
) -> dict[str, Any]:
    """Resolve LTO data from the shared options and dump it for the writer."""
    if engine_file is not None:
        engine_file = config.file_location(engine_file)

    # LTO data comes either from the Emissions Databank (EDB) or user provided
    # TOML file
    match lto_source:
        case 'edb':
            lto = lto_from_edb(engine_file, engine_uid, thrust_fractions)
        case 'custom':
            lto = lto_from_toml(lto_file)
        case _:
            raise click.UsageError(f'Unsupported LTO source: {lto_source}')
    return LTOPerformanceInput.from_internal(lto).model_dump()


def check_apu(apu_name: str | None) -> None:
    """Check that an APU name, if given, is in the APU database."""
    if apu_name is not None and lookup_apu(apu_name) is None:
        raise click.UsageError(f'APU "{apu_name}" not found in APU database.')


_SHARED_OPTIONS = [
    click.option(
        '--lto-source',
        type=click.Choice(['edb', 'custom']),
        required=True,
        help='Source of LTO performance data.',
    ),
    click.option(
        '--engine-file',
        help='Input engine database file.',
    ),
    click.option(
        '--engine-uid',
        type=str,
        help='UID of the engine to extract data for.',
    ),
    click.option(
        '--thrust-fractions',
        nargs=4,
        type=float,
        default=(0.07, 0.30, 0.85, 1.0),
        help='Thrust fractions for LTO modes: idle, approach, climb, takeoff.',
    ),
    click.option(
        '--lto-file',
        type=click.Path(exists=True),
        help='Input LTO TOML file.',
    ),
    click.option(
        '--aircraft-class',
        required=True,
        type=click.Choice(['wide', 'narrow', 'small', 'freight']),
    ),
    click.option(
        '--number-of-engines',
        type=click.IntRange(1, 8),
        required=True,
        help='Number of engines on the aircraft (1-8).',
    ),
    click.option(
        '--maximum-payload',
        type=int,
        required=True,
        help='Maximum payload in kg.',
    ),
    click.option(
        '--apu-name',
        required=False,
        help='Name of the APU used on the aircraft.',
    ),
]
"""Options every performance model subcommand takes."""


def shared_options(f):
    """Apply the options every performance model subcommand takes.

    The options are applied in reverse so that they list in declaration
    order in the subcommand's help."""
    for option in reversed(_SHARED_OPTIONS):
        f = option(f)
    return f


@click.group(
    short_help='Create a performance model from legacy data sources.',
    help="""Generate a performance model from legacy data sources.
    This command extracts performance data from the Emissions Databank (EDB) and
    BADA performance tables, and compiles it into a TOML file for use in the
    legacy performance model. The LTO data can come either from the EDB or from
    a user-provided TOML file. The BADA performance data is extracted from a
    PTF file. The resulting TOML file contains all necessary data to define the
    performance model for a given aircraft class and engine configuration.""",
)
@click.option(
    '--output-file',
    type=click.Path(),
    required=True,
    help='Output TOML file to write extracted data.',
)
@click.pass_context
def make_performance_model(ctx, output_file):
    ctx.ensure_object(dict)
    ctx.obj['output_file'] = output_file


###########################################
######   Legacy performance model    ######
###########################################


def build_performance_table(ptf: PTFData, phase: str) -> dict[str, Any]:
    cols = ['fl', 'mass', 'tas', 'rocd', 'fuel_flow']
    include_high = True
    if ptf.high_mass == ptf.nominal_mass:
        include_high = False
    data = []
    match phase:
        case 'climb':
            for r in ptf.climb:
                data.append([r.fl, ptf.low_mass, r.tas, r.rocd_low, r.fuel_flow_nom])
                data.append(
                    [r.fl, ptf.nominal_mass, r.tas, r.rocd_nom, r.fuel_flow_nom]
                )
                if include_high:
                    data.append(
                        [r.fl, ptf.high_mass, r.tas, r.rocd_high, r.fuel_flow_nom]
                    )
        case 'cruise':
            for r in ptf.cruise:
                data.append([r.fl, ptf.low_mass, r.tas, 0.0, r.fuel_flow_low])
                data.append([r.fl, ptf.nominal_mass, r.tas, 0.0, r.fuel_flow_nom])
                if include_high:
                    data.append([r.fl, ptf.high_mass, r.tas, 0.0, r.fuel_flow_high])
        case 'descent':
            for r in ptf.descent:
                data.append(
                    [r.fl, ptf.nominal_mass, r.tas, r.rocd_nom, r.fuel_flow_nom]
                )
    return dict(cols=cols, data=sorted(data, key=lambda x: (x[1], x[0], -x[3])))


@make_performance_model.command()
@shared_options
@click.option(
    '--ptf-file',
    type=click.Path(exists=True),
    required=True,
    help='Input PTF file for other performance data.',
)
@click.pass_context
def legacy(
    ctx,
    lto_source,
    engine_file,
    engine_uid,
    thrust_fractions,
    lto_file,
    aircraft_class,
    number_of_engines,
    maximum_payload,
    apu_name,
    ptf_file,
):
    Config.load()

    check_apu(apu_name)
    lto_dump = resolve_lto(
        lto_source, engine_file, engine_uid, thrust_fractions, lto_file
    )

    # Parse BADA performance file.
    ptf_data = PTFData.load(ptf_file)

    write_performance_toml(
        ctx.obj['output_file'],
        model_type='legacy',
        aircraft_name=ptf_data.aircraft_type,
        aircraft_class=aircraft_class,
        isa_offset=ptf_data.isa_offset,
        maximum_altitude_ft=ptf_data.maximum_altitude_ft,
        maximum_payload_kg=maximum_payload,
        number_of_engines=number_of_engines,
        apu_name=apu_name,
        lto_dump=lto_dump,
        speeds_dump=ptf_data.speeds.model_dump(),
        tables={
            'climb_flight_performance': build_performance_table(ptf_data, 'climb'),
            'cruise_flight_performance': build_performance_table(ptf_data, 'cruise'),
            'descent_flight_performance': build_performance_table(ptf_data, 'descent'),
        },
    )


###########################################
######    PIANO performance model    ######
###########################################


def parse_climb_masses(value: str | None) -> list[float] | None:
    """Parse the comma-separated `--climb-masses` value."""
    if value is None:
        return None
    try:
        return [float(mass) for mass in value.split(',')]
    except ValueError as e:
        raise click.UsageError(
            '--climb-masses must be a comma-separated list of masses in kg.'
        ) from e


@make_performance_model.command()
@shared_options
@click.option(
    '--cruise-file',
    type=click.Path(exists=True),
    required=True,
    help='Input PIANO cruise table file.',
)
@click.option(
    '--climb-file',
    type=click.Path(exists=True),
    required=True,
    help='Input PIANO climb details file.',
)
@click.option(
    '--descent-file',
    type=click.Path(exists=True),
    required=True,
    help='Input PIANO descent details file.',
)
@click.option(
    '--climb-masses',
    type=str,
    help='Comma-separated initial mass in kg of each climb block, in file order.',
)
@click.option(
    '--climb-cas-low-kts',
    type=float,
    help='Climb calibrated airspeed below FL100 in knots, overriding the file.',
)
@click.option(
    '--climb-cas-high-kts',
    type=float,
    help='Climb calibrated airspeed above FL100 in knots, overriding the file.',
)
@click.option(
    '--climb-mach',
    type=float,
    help='Climb Mach number above the crossover altitude, overriding the file.',
)
@click.option(
    '--climb-crossover-altitude-ft',
    type=float,
    help='Altitude in feet above which the climb flies its Mach number, '
    'overriding the file.',
)
@click.option(
    '--cruise-mach',
    type=float,
    help='Cruise Mach number. Omit to write no cruise speed data.',
)
@click.option(
    '--operating-empty-mass',
    type=float,
    help='Operating empty mass in kg. PIANO exports do not contain one.',
)
@click.option(
    '--aircraft-name',
    type=str,
    help='Aircraft name, overriding the cruise file title.',
)
@click.option(
    '--maximum-altitude-ft',
    type=int,
    help='Maximum altitude in feet, overriding the highest altitude climbed.',
)
@click.pass_context
def piano(
    ctx,
    lto_source,
    engine_file,
    engine_uid,
    thrust_fractions,
    lto_file,
    aircraft_class,
    number_of_engines,
    maximum_payload,
    apu_name,
    cruise_file,
    climb_file,
    descent_file,
    climb_masses,
    climb_cas_low_kts,
    climb_cas_high_kts,
    climb_mach,
    climb_crossover_altitude_ft,
    cruise_mach,
    operating_empty_mass,
    aircraft_name,
    maximum_altitude_ft,
):
    Config.load()

    check_apu(apu_name)
    lto_dump = resolve_lto(
        lto_source, engine_file, engine_uid, thrust_fractions, lto_file
    )

    # Parse PIANO performance files. The parser raises ValueError only when it
    # needs a value the files do not contain, so report those as usage errors.
    try:
        piano_data = PianoData.load(
            cruise_file,
            climb_file,
            descent_file,
            overrides=PianoOverrides(
                climb_masses_kg=parse_climb_masses(climb_masses),
                climb_cas_low_kts=climb_cas_low_kts,
                climb_cas_high_kts=climb_cas_high_kts,
                climb_mach=climb_mach,
                climb_crossover_altitude_ft=climb_crossover_altitude_ft,
            ),
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    cruise_speeds = None
    if cruise_mach is not None:
        # TODO: cruise cas_low is expected to equal climb cas_low, and cruise
        # cas_high is the cruise maximum Mach converted to CAS. Both are left
        # unset because PIANO's cruise table states no CAS schedule, and the
        # sweep already carries per-row `cas` at every (mass, fl, mach).
        cruise_speeds = SpeedData(mach=cruise_mach)
    speeds = Speeds(
        climb=piano_data.climb_speeds,
        cruise=cruise_speeds,
        descent=piano_data.descent_speeds,
    )

    write_performance_toml(
        ctx.obj['output_file'],
        model_type='piano',
        aircraft_name=aircraft_name or piano_data.aircraft_name,
        aircraft_class=aircraft_class,
        isa_offset=piano_data.isa_offset,
        maximum_altitude_ft=maximum_altitude_ft or piano_data.maximum_altitude_ft,
        maximum_payload_kg=maximum_payload,
        number_of_engines=number_of_engines,
        apu_name=apu_name,
        lto_dump=lto_dump,
        speeds_dump=speeds.model_dump(),
        tables={
            'climb_flight_performance': piano_data.climb.model_dump(),
            'cruise_flight_performance': piano_data.cruise.model_dump(),
            'descent_flight_performance': piano_data.descent.model_dump(),
            'cruise_reference_mach': piano_data.cruise_reference_mach.model_dump(),
            'descent_idle_thrust': piano_data.descent_idle_thrust.model_dump(),
        },
        extra_common={'operating_empty_mass_kg': operating_empty_mass},
    )


###########################################
######   TASOPT performance model    ######
###########################################


@make_performance_model.command()
def tasopt():
    raise NotImplementedError(
        'TASOPT performance model generation not yet implemented.'
    )
