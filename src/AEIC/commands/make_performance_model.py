import tomllib
from typing import Any

import click
import tomlkit
from tomlkit import comment, document, nl, table

from AEIC.config import Config, config

# from AEIC.parsers.lto_reader import parseLTO
from AEIC.parsers.ptf_reader import PTFData
from AEIC.performance.apu import lookup_apu
from AEIC.performance.edb import EDBEntry
from AEIC.performance.models.base import LTOPerformanceInput
from AEIC.performance.types import LTOPerformance

_BANNER_WIDTH = 78

_INLINE_COMMENTS = {
    'aircraft_class': 'wide, narrow, small, freight',
    'number_of_engines': 'Number of engines',
    'APU_name': 'None: APU emissions not calculated',
}

_COL_COMMENTS = {
    'fuel_flow': 'kg/s - REQUIRED; OUTPUT COLUMN',
    'fl': 'Flight levels',
    'tas': 'm/s',
    'rocd': 'm/s',
    'mass': 'kg',
}

_LTO_MODE_ORDER = ('idle', 'approach', 'climb', 'takeoff')
_LTO_MODE_KEY_ORDER = ('thrust_frac', 'fuel_kgs', 'EI_NOx', 'EI_HC', 'EI_CO')
_SPEED_PHASE_ORDER = ('climb', 'cruise', 'descent')
_SPEED_KEY_ORDER = ('cas_low', 'cas_high', 'mach')


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


def _format_flight_performance(cols: list[str], data: list[list[float]]) -> str:
    """Render the [flight_performance] section as a string with the
    right-aligned numeric column layout used by the sample performance
    model file."""
    col_lines = []
    for i, name in enumerate(cols):
        sep = ',' if i < len(cols) - 1 else ''
        quoted = f'"{name}"{sep}'
        if name in _COL_COMMENTS:
            col_lines.append(f'  {quoted}  # {_COL_COMMENTS[name]}')
        else:
            col_lines.append(f'  {quoted}')
    cols_block = 'cols = [\n' + '\n'.join(col_lines) + '\n]'

    cells = [[repr(float(v)) for v in row] for row in data]
    widths = [max(len(row[c]) for row in cells) for c in range(len(cols))]
    data_lines = []
    for row in cells:
        padded = ', '.join(row[c].rjust(widths[c]) for c in range(len(cols)))
        data_lines.append(f'  [ {padded}],')
    data_block = 'data = [\n' + '\n'.join(data_lines) + '\n]'

    return '[flight_performance]\n' + cols_block + '\n\n' + data_block + '\n'


def _fix_empty_comments(text: str) -> str:
    """tomlkit renders `comment('')` as `# ` (with a trailing space);
    strip the trailing space so empty banner lines are plain `#`."""
    return text.replace('# \n', '#\n')


def write_legacy_performance_toml(
    path: str,
    *,
    aircraft_name: str,
    aircraft_class: str,
    isa_offset: int,
    maximum_altitude_ft: int,
    maximum_payload_kg: int,
    number_of_engines: int,
    apu_name: str | None,
    lto_dump: dict[str, Any],
    speeds_dump: dict[str, Any],
    flight_performance: dict[str, Any],
) -> None:
    doc = document()
    doc.add(comment('Performance model type (one of: legacy, bada, tasopt, piano).'))
    doc['model_type'] = 'legacy'

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
    if apu_name is not None:
        _set_with_inline(doc, 'APU_name', apu_name)

    doc.add(nl())
    _add_sub_banner(doc, 'Speed data')

    speeds_super = table(True)
    for phase in _SPEED_PHASE_ORDER:
        if phase not in speeds_dump:
            continue
        phase_tbl = table()
        phase_data = speeds_dump[phase]
        for key in _SPEED_KEY_ORDER:
            if key in phase_data:
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

    fp_section = _format_flight_performance(
        flight_performance['cols'], flight_performance['data']
    )

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(body)
        fp.write(trailer)
        fp.write(fp_section)


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


def build_performance_table(ptf: PTFData) -> dict[str, Any]:
    cols = ['fl', 'mass', 'tas', 'rocd', 'fuel_flow']
    data = []
    for r in ptf.climb:
        data.append([r.fl, ptf.low_mass, r.tas, r.rocd_low, r.fuel_flow_nom])
        data.append([r.fl, ptf.nominal_mass, r.tas, r.rocd_nom, r.fuel_flow_nom])
        data.append([r.fl, ptf.high_mass, r.tas, r.rocd_high, r.fuel_flow_nom])
    for r in ptf.cruise:
        data.append([r.fl, ptf.low_mass, r.tas, 0.0, r.fuel_flow_low])
        data.append([r.fl, ptf.nominal_mass, r.tas, 0.0, r.fuel_flow_nom])
        data.append([r.fl, ptf.high_mass, r.tas, 0.0, r.fuel_flow_high])
    for r in ptf.descent:
        data.append([r.fl, ptf.nominal_mass, r.tas, r.rocd_nom, r.fuel_flow_nom])
    return dict(cols=cols, data=sorted(data, key=lambda x: (x[1], x[0], -x[3])))


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
    '--ptf-file',
    type=click.Path(exists=True),
    required=True,
    help='Input PTF file for other performance data.',
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
    '--apu-name',
    required=False,
    help='Name of the APU used on the aircraft.',
)
@make_performance_model.command()
@click.pass_context
def legacy(
    ctx,
    lto_source,
    engine_file,
    engine_uid,
    thrust_fractions,
    lto_file,
    ptf_file,
    aircraft_class,
    number_of_engines,
    apu_name,
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

    # Parse BADA performance file.
    ptf_data = PTFData.load(ptf_file)

    write_legacy_performance_toml(
        ctx.obj['output_file'],
        aircraft_name=ptf_data.aircraft_type,
        aircraft_class=aircraft_class,
        isa_offset=ptf_data.isa_offset,
        maximum_altitude_ft=ptf_data.maximum_altitude_ft,
        maximum_payload_kg=ptf_data.maximum_payload,
        number_of_engines=number_of_engines,
        apu_name=apu_name,
        lto_dump=lto_dump,
        speeds_dump=ptf_data.speeds.model_dump(),
        flight_performance=build_performance_table(ptf_data),
    )


@make_performance_model.command()
def tasopt():
    raise NotImplementedError(
        'TASOPT performance model generation not yet implemented.'
    )
