"""Shared TOML writer for the ``make-performance-model`` subcommands.

Every performance model type writes the same common fields, speed data and
LTO data, and then a model-specific sequence of tabular sections. This module
holds that machinery so each subcommand only has to supply the values.
"""

from typing import Any

import tomlkit
from tomlkit import comment, document, nl, table

BANNER_WIDTH = 78

INLINE_COMMENTS = {
    'aircraft_class': 'wide, narrow, small, freight',
    'number_of_engines': 'Number of engines',
    'APU_name': 'None: APU emissions not calculated',
    'operating_empty_mass_kg': 'kg',
}

COL_COMMENTS = {
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

LTO_MODE_ORDER = ('idle', 'approach', 'climb', 'takeoff')
LTO_MODE_KEY_ORDER = ('thrust_frac', 'fuel_kgs', 'EI_NOx', 'EI_HC', 'EI_CO')
SPEED_PHASE_ORDER = ('climb', 'cruise', 'descent')
SPEED_KEY_ORDER = ('cas_low', 'cas_high', 'mach', 'crossover_altitude_m')


def add_top_banner(doc, title: str, description: str | None = None) -> None:
    doc.add(comment('=' * BANNER_WIDTH))
    doc.add(comment(''))
    doc.add(comment(f' {title}'))
    doc.add(comment(''))
    if description is not None:
        doc.add(comment(description))


def add_sub_banner(doc, title: str) -> None:
    doc.add(comment('-' * BANNER_WIDTH))
    doc.add(comment(''))
    doc.add(comment(title))
    doc.add(comment(''))


def set_with_inline(tbl, key: str, value: Any) -> None:
    tbl[key] = value
    if key in INLINE_COMMENTS:
        tbl[key].comment(INLINE_COMMENTS[key])


def format_table_section(section: str, cols: list[str], data: list[list[float]]) -> str:
    """Render a tabular section as a string with the right-aligned numeric
    column layout used by the sample performance model file."""
    col_lines = []
    for i, name in enumerate(cols):
        sep = ',' if i < len(cols) - 1 else ''
        quoted = f'"{name}"{sep}'
        if name in COL_COMMENTS:
            col_lines.append(f'  {quoted}  # {COL_COMMENTS[name]}')
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


def fix_empty_comments(text: str) -> str:
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
    add_top_banner(
        doc, 'COMMON FIELDS', 'Fields common to all performance model types.'
    )
    doc.add(nl())

    set_with_inline(doc, 'aircraft_name', aircraft_name)
    set_with_inline(doc, 'aircraft_class', aircraft_class)
    set_with_inline(doc, 'ISA_offset', isa_offset)
    set_with_inline(doc, 'maximum_altitude_ft', maximum_altitude_ft)
    set_with_inline(doc, 'maximum_payload_kg', maximum_payload_kg)
    set_with_inline(doc, 'number_of_engines', number_of_engines)
    for key, value in (extra_common or {}).items():
        if value is not None:
            set_with_inline(doc, key, value)
    if apu_name is not None:
        set_with_inline(doc, 'APU_name', apu_name)

    doc.add(nl())
    add_sub_banner(doc, 'Speed data')

    speeds_super = table(True)
    for phase in SPEED_PHASE_ORDER:
        if phase not in speeds_dump or speeds_dump[phase] is None:
            continue
        phase_tbl = table()
        phase_data = speeds_dump[phase]
        for key in SPEED_KEY_ORDER:
            if key in phase_data and phase_data[key] is not None:
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

    sections = [
        format_table_section(section, tbl['cols'], tbl['data'])
        for section, tbl in tables.items()
    ]

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(body)
        fp.write(trailer)
        fp.write('\n'.join(sections))
