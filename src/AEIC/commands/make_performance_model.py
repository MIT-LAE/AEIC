import tomllib
from typing import Any

import click

from AEIC.commands._performance_model_toml import write_performance_toml
from AEIC.config import Config, config

# from AEIC.parsers.lto_reader import parseLTO
from AEIC.parsers.ptf_reader import PTFData
from AEIC.performance.apu import lookup_apu
from AEIC.performance.edb import EDBEntry
from AEIC.performance.models.base import LTOPerformanceInput
from AEIC.performance.types import LTOPerformance


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
    '--maximum-payload',
    type=int,
    required=True,
    help='Maximum payload in kg.',
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
    maximum_payload,
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


@make_performance_model.command()
def tasopt():
    raise NotImplementedError(
        'TASOPT performance model generation not yet implemented.'
    )
