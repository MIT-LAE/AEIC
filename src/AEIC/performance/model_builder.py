"""Build performance model instances from parsed source data, and write them
back out as commented TOML.

This module implements the logic behind the ``make-performance-model`` command.
The builders take data a parser has already produced (a
:class:`~AEIC.parsers.ptf_reader.PTFData` or
:class:`~AEIC.parsers.piano_reader.PianoData`, plus LTO data) and return a
performance model instance.

:func:`write_performance_model` writes a model as an AEIC performance .toml file.
The layout of that file is described a by per model type ordered sequence of
``(field name, output key)`` pairs.
"""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomlkit
from tomlkit import comment, document, nl, table

from AEIC.performance.models import (
    BasePerformanceModel,
    LegacyPerformanceModel,
    PianoPerformanceModel,
)
from AEIC.performance.models.base import LTOPerformanceInput
from AEIC.performance.types import SpeedData, Speeds, TableInput

if TYPE_CHECKING:
    from AEIC.parsers.piano_reader import PianoData
    from AEIC.parsers.ptf_reader import PTFData

###########################################
######        File layout            ######
###########################################

_BANNER_WIDTH = 78

_MODEL_TYPE_COMMENT = 'Performance model type (one of: legacy, bada, tasopt, piano).'

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

_COMMON_HEAD_SPEC = (
    ('model_type', 'model_type'),
    ('aircraft_name', 'aircraft_name'),
    ('aircraft_class', 'aircraft_class'),
    ('isa_offset', 'ISA_offset'),
    ('maximum_altitude_ft', 'maximum_altitude_ft'),
    ('maximum_payload_kg', 'maximum_payload_kg'),
    ('number_of_engines', 'number_of_engines'),
)
"""Common-block fields every model type writes first, in file order."""

_COMMON_TAIL_SPEC = (
    ('apu_name', 'APU_name'),
    ('speeds', 'speeds'),
    ('lto_performance', 'LTO_performance'),
)
"""Common-block fields every model type writes last, in file order."""

_PHASE_TABLE_SPEC = (
    ('climb_flight_performance', 'climb_flight_performance'),
    ('cruise_flight_performance', 'cruise_flight_performance'),
    ('descent_flight_performance', 'descent_flight_performance'),
)
"""Per-phase performance tables every model type writes."""

LEGACY_WRITE_SPEC = (*_COMMON_HEAD_SPEC, *_COMMON_TAIL_SPEC, *_PHASE_TABLE_SPEC)
"""Field order and output keys for a `LegacyPerformanceModel` file."""

PIANO_WRITE_SPEC = (
    *_COMMON_HEAD_SPEC,
    ('operating_empty_mass_kg', 'operating_empty_mass_kg'),
    *_COMMON_TAIL_SPEC,
    *_PHASE_TABLE_SPEC,
    ('cruise_reference_mach', 'cruise_reference_mach'),
    ('descent_idle_thrust', 'descent_idle_thrust'),
)
"""Field order and output keys for a `PianoPerformanceModel` file."""

WRITE_SPECS: dict[type[BasePerformanceModel], tuple[tuple[str, str], ...]] = {
    LegacyPerformanceModel: LEGACY_WRITE_SPEC,
    PianoPerformanceModel: PIANO_WRITE_SPEC,
}
"""Write spec for each model type this module can write."""


###########################################
######   AEIC output .toml writer    ######
###########################################


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
    tbl[key] = str(value) if isinstance(value, Enum) else value
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


def _speeds_table(speeds: Speeds):
    """Render speed data as a nested TOML table, one sub-table per phase.

    Phases with no data are omitted, as are unset keys within a phase."""
    speeds_dump = speeds.model_dump()
    speeds_super = table(True)
    for phase in _SPEED_PHASE_ORDER:
        if speeds_dump.get(phase) is None:
            continue
        phase_tbl = table()
        phase_data = speeds_dump[phase]
        for key in _SPEED_KEY_ORDER:
            if phase_data.get(key) is not None:
                phase_tbl[key] = phase_data[key]
        speeds_super.append(phase, phase_tbl)
    return speeds_super


def _lto_table(lto: LTOPerformanceInput):
    """Render LTO data as a nested TOML table, one sub-table per thrust mode."""
    lto_dump = lto.model_dump()
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
    return lto_tbl


def write_performance_model(path: str | Path, model: BasePerformanceModel) -> None:
    """Write a performance model as a commented TOML file.

    Fields are written in the order given by the model type's entry in
    `WRITE_SPECS`, under the output key that entry names.
    - Tabular data is written as a trailing ``[section]`` block with
      aligned numeric layout
    - Speed and LTO data are written as nested sub-tables
    - Anything else is written inline.

    A field whose value is ``None`` is not written.

    Args:
        path: Output file path.
        model: The performance model to write.

    Raises:
        ValueError: If no write spec is defined for this model type.
    """
    spec = WRITE_SPECS.get(type(model))
    if spec is None:
        raise ValueError(
            f'No write spec defined for performance model type '
            f'"{type(model).__name__}".'
        )

    (head_field, head_key), *rest = spec
    doc = document()
    doc.add(comment(_MODEL_TYPE_COMMENT))
    doc[head_key] = getattr(model, head_field)

    doc.add(nl())
    _add_top_banner(
        doc, 'COMMON FIELDS', 'Fields common to all performance model types.'
    )
    doc.add(nl())

    table_sections: list[tuple[str, TableInput]] = []
    for field, key in rest:
        value = getattr(model, field)
        if value is None:
            continue
        if isinstance(value, TableInput):
            table_sections.append((key, value))
        elif isinstance(value, Speeds):
            doc.add(nl())
            _add_sub_banner(doc, 'Speed data')
            doc[key] = _speeds_table(value)
        elif isinstance(value, LTOPerformanceInput):
            doc.add(nl())
            _add_sub_banner(doc, 'LTO data')
            doc[key] = _lto_table(value)
        else:
            _set_with_inline(doc, key, value)

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
        _format_table_section(key, tbl.cols, tbl.data) for key, tbl in table_sections
    ]

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(body)
        fp.write(trailer)
        fp.write('\n'.join(sections))


###########################################
######   Legacy performance model    ######
###########################################


def build_performance_table(ptf: PTFData, phase: str) -> dict[str, Any]:
    """Build a phase's performance table from BADA PTF data."""
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


def build_legacy_model(
    ptf_data: PTFData,
    lto: LTOPerformanceInput | None,
    *,
    aircraft_class: str,
    number_of_engines: int,
    maximum_payload: int,
    apu_name: str | None = None,
) -> LegacyPerformanceModel:
    """Build a legacy performance model from BADA PTF data.

    Args:
        ptf_data: Parsed BADA PTF file contents.
        lto: LTO performance data, or None to write no LTO block.
        aircraft_class: One of wide, narrow, small, freight.
        number_of_engines: Number of engines on the aircraft.
        maximum_payload: Maximum payload [kg].
        apu_name: APU name, or None if the aircraft has no APU data.

    Returns:
        Built model. Aircraft name, ISA offset, maximum altitude and speed
        schedules all come from the PTF file.

    Raises:
        ValidationError: If the resulting model is not valid, for example
            because the APU name is not in the APU database.
    """
    return LegacyPerformanceModel(
        model_type='legacy',
        aircraft_name=ptf_data.aircraft_type,
        aircraft_class=aircraft_class,
        isa_offset=ptf_data.isa_offset,
        maximum_altitude_ft=ptf_data.maximum_altitude_ft,
        maximum_payload_kg=maximum_payload,
        number_of_engines=number_of_engines,
        apu_name=apu_name,
        speeds=ptf_data.speeds,
        lto_performance=lto,
        climb_flight_performance=build_performance_table(ptf_data, 'climb'),
        cruise_flight_performance=build_performance_table(ptf_data, 'cruise'),
        descent_flight_performance=build_performance_table(ptf_data, 'descent'),
    )


###########################################
######    PIANO performance model    ######
###########################################


def build_piano_model(
    piano_data: PianoData,
    lto: LTOPerformanceInput | None,
    *,
    aircraft_class: str,
    number_of_engines: int,
    maximum_payload: int,
    apu_name: str | None = None,
    cruise_mach: float | None = None,
    operating_empty_mass: float | None = None,
    aircraft_name: str | None = None,
    maximum_altitude_ft: int | None = None,
) -> PianoPerformanceModel:
    """Build a PIANO performance model from parsed PIANO outputs.

    Args:
        piano_data: Parsed PIANO cruise, climb and descent outputs.
        lto: LTO performance data, or None to write no LTO block.
        aircraft_class: One of wide, narrow, small, freight.
        number_of_engines: Number of engines on the aircraft.
        maximum_payload: Maximum payload [kg].
        apu_name: APU name, or None if the aircraft has no APU data.
        cruise_mach: Cruise Mach number. PIANO sweeps many, so the model states
            one only when the caller picks it.
        operating_empty_mass: Operating empty mass [kg]. PIANO exports do not
            contain one.
        aircraft_name: Aircraft name, overriding the cruise file title.
        maximum_altitude_ft: Maximum altitude [feet], overriding the highest
            altitude any climb block reaches.

    Returns:
        Built model.

    Raises:
        ValidationError: If the resulting model is not valid, for example
            because the APU name is not in the APU database.
    """
    cruise_speeds = None
    if cruise_mach is not None:
        # TODO: cruise cas_low is expected to equal climb cas_low, and cruise
        # cas_high is the cruise maximum Mach converted to CAS. Both are left
        # unset because PIANO's cruise table states no CAS schedule, and the
        # sweep already carries per-row `cas` at every (mass, fl, mach).
        cruise_speeds = SpeedData(mach=cruise_mach)

    return PianoPerformanceModel(
        model_type='piano',
        aircraft_name=aircraft_name or piano_data.aircraft_name,
        aircraft_class=aircraft_class,
        isa_offset=piano_data.isa_offset,
        maximum_altitude_ft=maximum_altitude_ft or piano_data.maximum_altitude_ft,
        maximum_payload_kg=maximum_payload,
        number_of_engines=number_of_engines,
        operating_empty_mass_kg=operating_empty_mass,
        apu_name=apu_name,
        speeds=Speeds(
            climb=piano_data.climb_speeds,
            cruise=cruise_speeds,
            descent=piano_data.descent_speeds,
        ),
        lto_performance=lto,
        climb_flight_performance=piano_data.climb.model_dump(),
        cruise_flight_performance=piano_data.cruise.model_dump(),
        descent_flight_performance=piano_data.descent.model_dump(),
        cruise_reference_mach=piano_data.cruise_reference_mach,
        descent_idle_thrust=piano_data.descent_idle_thrust,
    )
