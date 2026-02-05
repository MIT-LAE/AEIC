# WARNING: THIS IS NOT REALLY TESTED! IT'S MORE A PLACEHOLDER FOR SOMEONE ELSE
# (ADI? WYATT?) TO FIX.

from typing import Any

import click
import tomli_w

from AEIC.config import Config, config

# from AEIC.parsers.lto_reader import parseLTO
from AEIC.parsers.ptf_reader import PTFData
from AEIC.performance.models.base import LTOPerformanceInput
from AEIC.performance.utils.apu import lookup_apu
from AEIC.performance.utils.edb import EDBEntry
from AEIC.types import LTOPerformance

Config.load()


def lto_from_edb(engine_file, engine_uid, thrust_fractions) -> LTOPerformance:
    if engine_file is None or engine_uid is None:
        raise click.UsageError(
            'Both --engine-file and --engine-uid must be provided when '
            'using "edb" as the LTO source.'
        )
    edb_data = EDBEntry.get_engine(engine_file, engine_uid)
    return edb_data.make_lto_performance(thrust_fractions)


def lto_from_lto_file(lto_file) -> LTOPerformance:
    if lto_file is None:
        raise click.UsageError(
            '--lto-file must be provided when using "lto-file" as the LTO source.'
        )

    raise NotImplementedError('LTO file parsing not yet implemented.')

    # lto_data = parseLTO(lto_file)
    # # TODO: Somehow make an LTOPerformance value from this.
    # result = LTOPerformance()
    # return result


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


@click.group()
@click.option(
    '--output-file',
    type=click.Path(),
    required=True,
    help='Output TOML file to write extracted data.',
)
@click.pass_context
def cli(ctx, output_file):
    ctx.ensure_object(dict)
    ctx.obj['output_file'] = output_file


@click.option(
    '--lto-source',
    type=click.Choice(['edb', 'lto-file']),
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
    help='Input BADA LTO file.',
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
    # TODO: Limit to reasonable values.
    '--number-of-engines',
    type=int,
    required=True,
    help='Number of engines on the aircraft.',
)
@click.option(
    '--apu-name',
    required=False,
    help='Name of the APU used on the aircraft.',
)
@cli.command()
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
    if apu_name is not None and lookup_apu(apu_name) is None:
        raise click.UsageError(f'APU "{apu_name}" not found in APU database.')
    if engine_file is not None:
        engine_file = config.file_location(engine_file)

    toml_data: dict[str, Any] = {'model_type': 'legacy'}

    # LTO data comes either from the engine database (plus explicitly provided
    # Foo value) or from a BADA LTO file.
    match lto_source:
        case 'edb':
            lto = lto_from_edb(engine_file, engine_uid, thrust_fractions)
        case 'lto-file':
            lto = lto_from_lto_file(lto_file)
        case _:
            raise click.UsageError(f'Unsupported LTO source: {lto_source}')
    toml_data['LTO_performance'] = LTOPerformanceInput.from_internal(lto).model_dump()

    # Parse BADA performance file.
    ptf_data = PTFData.load(ptf_file)

    # Extract top-level data.
    toml_data['aircraft_name'] = ptf_data.aircraft_type
    toml_data['aircraft_class'] = aircraft_class
    toml_data['ISA_offset'] = ptf_data.isa_offset
    toml_data['maximum_altitude_ft'] = ptf_data.maximum_altitude_ft
    toml_data['maximum_payload_kg'] = ptf_data.maximum_payload
    toml_data['number_of_engines'] = number_of_engines
    if apu_name is not None:
        toml_data['APU_name'] = apu_name

    # Extract speed data.
    toml_data['speeds'] = ptf_data.speeds.model_dump()

    # Extract performance table data.
    toml_data['flight_performance'] = build_performance_table(ptf_data)

    with open(ctx.obj['output_file'], 'wb') as fp:
        tomli_w.dump(toml_data, fp)


@cli.command()
def bada():
    raise NotImplementedError('BADA performance model generation not yet implemented.')


@cli.command()
def tasopt():
    raise NotImplementedError(
        'TASOPT performance model generation not yet implemented.'
    )


if __name__ == '__main__':
    cli()
