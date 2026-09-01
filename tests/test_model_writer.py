"""Tests for the performance model TOML writer.

These cover how a model is rendered: which fields are written, in what order,
under what key, and how tabular data is laid out.
"""

import tomllib
from pathlib import Path

import pytest

from AEIC.performance.model_builder import (
    WRITE_SPECS,
    build_legacy_model,
    build_piano_model,
    write_performance_model,
)
from AEIC.performance.models import (
    LegacyPerformanceModel,
    PerformanceModel,
    PianoPerformanceModel,
)
from AEIC.performance.types import TableInput

MODEL_TYPES = list(WRITE_SPECS)


class _UnregisteredModel(LegacyPerformanceModel):
    """A model type with no entry in `WRITE_SPECS`."""


@pytest.fixture
def piano_model(piano_data, lto) -> PianoPerformanceModel:
    return build_piano_model(
        piano_data,
        lto,
        aircraft_class='narrow',
        number_of_engines=2,
        maximum_payload=18000,
        operating_empty_mass=37100,
        apu_name='APU 131-9',
    )


@pytest.fixture
def legacy_model(ptf_data, lto) -> LegacyPerformanceModel:
    return build_legacy_model(
        ptf_data,
        lto,
        aircraft_class='narrow',
        number_of_engines=2,
        maximum_payload=22422,
        apu_name='APU 131-9',
    )


def write_toml(tmp_path, model, name: str = 'model.toml') -> Path:
    """Write a model and return the file it was written to."""
    out_file = tmp_path / name
    write_performance_model(out_file, model)
    return out_file


def read_toml(tmp_path, model, name: str = 'model.toml') -> dict:
    """Write a model and read it back as raw TOML data."""
    with open(write_toml(tmp_path, model, name), 'rb') as fp:
        return tomllib.load(fp)


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_write_spec_covers_every_model_field(model_type):
    """Ensure the WRITER_SPECS dict is consistent with the fields of
    each performance model.
    Every field in model must have an entry in WRITE_SPEC,
    and all WRITE_SPEC entries must be a valid model field."""
    spec_fields = {field for field, _ in WRITE_SPECS[model_type]}
    assert spec_fields == set(model_type.model_fields)


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_write_spec_output_keys_are_unique(model_type):
    keys = [key for _, key in WRITE_SPECS[model_type]]
    assert len(keys) == len(set(keys))


def test_table_sections_are_written_in_spec_order(tmp_path, piano_model, legacy_model):
    for name, model in (('piano.toml', piano_model), ('legacy.toml', legacy_model)):
        # Expected order of tables from reading the WRITE_SPECS
        expected = [
            key
            for field, key in WRITE_SPECS[type(model)]
            if isinstance(getattr(model, field), TableInput)
        ]
        assert expected, f'{name} states no tables, so this asserts nothing'

        data = read_toml(tmp_path, model, name)
        written = [
            key
            for key, value in data.items()
            if isinstance(value, dict) and 'cols' in value
        ]
        assert written == expected


def test_empty_table_writes_its_columns_and_no_rows(tmp_path, piano_model):
    """The PIANO reader produces an empty descent idle thrust table when no
    descent row qualifies, so the writer must render it rather than crash."""
    cols = ['mass', 'idle_thrust_altitude']
    empty = piano_model.model_copy(
        update={'descent_idle_thrust': TableInput(cols=cols, data=[])}
    )
    out_file = write_toml(tmp_path, empty)

    assert 'data = []' in out_file.read_text()
    with open(out_file, 'rb') as fp:
        assert tomllib.load(fp)['descent_idle_thrust'] == {'cols': cols, 'data': []}


def test_none_table_omits_its_whole_section(tmp_path, piano_model):
    """`None` states the model has no such table. That is distinct from a
    table that exists but holds no rows, which keeps its column labels."""
    without = piano_model.model_copy(update={'cruise_reference_mach': None})
    data = read_toml(tmp_path, without)

    assert 'cruise_reference_mach' not in data
    assert 'descent_idle_thrust' in data


def test_apu_name_omitted_when_unset(tmp_path, piano_model):
    assert 'APU_name' in read_toml(tmp_path, piano_model, 'with_apu.toml')

    without = piano_model.model_copy(update={'apu_name': None})
    assert 'APU_name' not in read_toml(tmp_path, without, 'without_apu.toml')


def test_unset_speed_phase_is_omitted(tmp_path, piano_model):
    """The fixture model states no cruise Mach number, so it has no cruise
    speed data to write."""
    assert piano_model.speeds.cruise is None
    assert set(read_toml(tmp_path, piano_model)['speeds']) == {'climb', 'descent'}


def test_isa_offset_round_trips(tmp_path, piano_model, legacy_model):
    """`ISA_offset` used to be written and then dropped on load, because no
    model declared the field."""
    for name, model in (('piano.toml', piano_model), ('legacy.toml', legacy_model)):
        out_file = write_toml(tmp_path, model, name)
        with open(out_file, 'rb') as fp:
            assert tomllib.load(fp)['ISA_offset'] == model.isa_offset
        assert PerformanceModel.load(out_file).isa_offset == model.isa_offset


def test_model_type_without_a_write_spec_raises(tmp_path, legacy_model):
    """The spec is looked up by exact type, so a new model type has to
    register one rather than inheriting a parent's layout."""
    model = _UnregisteredModel(**legacy_model.model_dump())
    with pytest.raises(ValueError, match='No write spec'):
        write_performance_model(tmp_path / 'unregistered.toml', model)
