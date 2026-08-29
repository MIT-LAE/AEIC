import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from AEIC.cli import cli
from AEIC.commands.make_performance_model import _format_table_section
from AEIC.config import Config

TEST_DATA_DIR = (Path(__file__).parent / 'data').resolve()

GOLDEN_FILE = TEST_DATA_DIR / 'performance' / 'legacy_golden.toml'
PTF_FILE = TEST_DATA_DIR / 'verification' / 'legacy' / 'legacy_performance.PTF'


# Disable default configuration loading fixture just for this file. The
# `make-performance-model` subcommands call `Config.load()` themselves, which
# fails if the global fixture has already initialized the singleton.
@pytest.fixture(autouse=True)
def default_config(request):
    # Accept `request` and assert no `config_updates` marker is present —
    # without this argument, such a marker on a test in this file would be
    # silently dropped instead of running the global fixture's overlay logic.
    assert request.node.get_closest_marker('config_updates') is None, (
        'tests/test_make_performance_model.py shadows the global '
        'default_config fixture, so the @pytest.mark.config_updates marker '
        'has no effect here'
    )
    yield None
    Config.reset()


def test_legacy_output_byte_identical(tmp_path):
    """The legacy writer's output is frozen. Any diff here is a regression."""
    out_file = tmp_path / 'legacy_performance.toml'

    result = CliRunner().invoke(
        cli,
        [
            'make-performance-model',
            '--output-file',
            str(out_file),
            'legacy',
            '--lto-source',
            'edb',
            '--engine-file',
            'engines/sample_edb.xlsx',
            '--engine-uid',
            '01P11CM121',
            '--ptf-file',
            str(PTF_FILE),
            '--aircraft-class',
            'narrow',
            '--number-of-engines',
            '2',
            '--maximum-payload',
            '22422',
            '--apu-name',
            'APU 131-9',
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_file.read_bytes() == GOLDEN_FILE.read_bytes()


def test_empty_table_section():
    """`TableInput` accepts an empty table, and the PIANO reader produces one
    when no row qualifies, so the writer must render it instead of crashing."""
    section = _format_table_section('descent_idle_thrust', ['mass', 'x'], [])
    assert 'data = []' in section
    assert tomllib.loads(section) == {
        'descent_idle_thrust': {'cols': ['mass', 'x'], 'data': []}
    }
