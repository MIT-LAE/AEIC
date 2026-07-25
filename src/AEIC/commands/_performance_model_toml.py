"""Shared TOML-formatting helpers for writing performance model files.

Used by :mod:`AEIC.commands.make_piano_performance_model`. Also duplicated
(independently) in :mod:`AEIC.commands.make_performance_model`, which
predates this module -- consolidating that file onto this one, so there is
a single copy instead of two that can drift, is a follow-up for whoever
owns that file to weigh in on, not done here.
"""

from typing import Any

from tomlkit import comment

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

LTO_MODE_ORDER = ('idle', 'approach', 'climb', 'takeoff')
LTO_MODE_KEY_ORDER = ('thrust_frac', 'fuel_kgs', 'EI_NOx', 'EI_HC', 'EI_CO')
SPEED_PHASE_ORDER = ('climb', 'cruise', 'descent')
SPEED_KEY_ORDER = ('cas_low', 'cas_high', 'mach')


def add_top_banner(doc, title: str, description: str | None = None) -> None:
    doc.add(comment('=' * _BANNER_WIDTH))
    doc.add(comment(''))
    doc.add(comment(f' {title}'))
    doc.add(comment(''))
    if description is not None:
        doc.add(comment(description))


def add_sub_banner(doc, title: str) -> None:
    doc.add(comment('-' * _BANNER_WIDTH))
    doc.add(comment(''))
    doc.add(comment(title))
    doc.add(comment(''))


def set_with_inline(tbl, key: str, value: Any) -> None:
    tbl[key] = value
    if key in _INLINE_COMMENTS:
        tbl[key].comment(_INLINE_COMMENTS[key])


def format_flight_performance(
    phase: str, cols: list[str], data: list[list[float]]
) -> str:
    """Render a ``[{phase}_flight_performance]`` section as a string with
    the right-aligned numeric column layout used by the sample performance
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

    return f'[{phase}_flight_performance]\n' + cols_block + '\n\n' + data_block + '\n'


def fix_empty_comments(text: str) -> str:
    """tomlkit renders `comment('')` as `# ` (with a trailing space);
    strip the trailing space so empty banner lines are plain `#`."""
    return text.replace('# \n', '#\n')


MODEL_TYPE_COMMENT = 'Performance model type (one of: legacy, bada, tasopt, piano).'
