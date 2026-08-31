"""Tests for the `make-performance-model`."""

import pytest

from AEIC.commands.make_performance_model import (
    check_apu,
    lto_from_toml,
    parse_climb_masses,
    resolve_lto,
)
from AEIC.config import config
from AEIC.performance.model_builder import build_legacy_model, write_performance_model
from AEIC.performance.models.base import LTOPerformanceInput

ENGINE_FILE = 'engines/sample_edb.xlsx'
ENGINE_UID = '01P11CM121'
THRUST_FRACTIONS = (0.07, 0.30, 0.85, 1.0)


def test_legacy_output_byte_identical(tmp_path, ptf_data, lto):
    """The legacy writer's output is frozen. Any diff here is a regression."""

    reference_toml = config.file_location('performance/legacy_golden.toml')
    out_file = tmp_path / 'legacy_performance.toml'

    model = build_legacy_model(
        ptf_data,
        lto,
        aircraft_class='narrow',
        number_of_engines=2,
        maximum_payload=22422,
        apu_name='APU 131-9',
    )
    write_performance_model(out_file, model)

    assert out_file.read_bytes() == reference_toml.read_bytes()


###########################################
######      Option resolution        ######
###########################################


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (None, None),
        ('54000', [54000.0]),
        ('54000,50000.5', [54000.0, 50000.5]),
    ],
)
def test_parse_climb_masses(value, expected):
    assert parse_climb_masses(value) == expected


def test_parse_climb_masses_rejects_a_non_numeric_value():
    with pytest.raises(ValueError, match='comma-separated list of masses'):
        parse_climb_masses('54000,heavy')


@pytest.mark.parametrize('apu_name', [None, 'APU 131-9'])
def test_check_apu_accepts_no_name_and_a_known_name(apu_name):
    assert check_apu(apu_name) is None


def test_check_apu_rejects_an_unknown_name():
    with pytest.raises(ValueError, match='not found in APU database'):
        check_apu('APU 000-0')


def test_resolve_lto_from_edb():
    lto = resolve_lto('edb', ENGINE_FILE, ENGINE_UID, THRUST_FRACTIONS, None)
    assert isinstance(lto, LTOPerformanceInput)
    assert lto.source == 'EDB'
    assert lto.ICAO_UID == ENGINE_UID


@pytest.mark.parametrize(
    ('engine_file', 'engine_uid'),
    [(None, ENGINE_UID), (ENGINE_FILE, None)],
)
def test_resolve_lto_from_edb_requires_both_engine_options(engine_file, engine_uid):
    with pytest.raises(ValueError, match='--engine-file and --engine-uid'):
        resolve_lto('edb', engine_file, engine_uid, THRUST_FRACTIONS, None)


def test_lto_from_toml_reads_an_lto_performance_table():
    """The reader accepts either a whole model file or a bare LTO table."""
    reference_toml = config.file_location('performance/legacy_golden.toml')
    lto = lto_from_toml(reference_toml)
    assert lto.ICAO_UID == ENGINE_UID
