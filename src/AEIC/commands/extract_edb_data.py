import click
import tomli_w

from AEIC.performance.utils.edb import EDBEntry


@click.option(
    '--engine-file',
    '-e',
    type=click.Path(exists=True),
    required=True,
    help='Input engine database file.',
)
@click.option(
    '--output-file',
    '-o',
    type=click.Path(),
    required=True,
    help='Output TOML file to write extracted data.',
)
@click.option(
    '--engine-uid',
    '-u',
    type=str,
    required=True,
    help='UID of the engine to extract data for.',
)
@click.option(
    '--thrust-fractions',
    '-t',
    nargs=4,
    type=float,
    default=(0.07, 0.30, 0.85, 1.0),
    help='Thrust fractions for LTO modes: idle, approach, climb, takeoff.',
)
@click.option(
    '--foo-kn',
    '-f',
    type=float,
    required=True,
    help='Sea-level static thrust of the engine in kN.',
)
@click.command()
def run(engine_file, output_file, engine_uid, thrust_fractions, foo_kn):
    edb_data = EDBEntry.get_engine(engine_file, engine_uid)
    lto = edb_data.make_lto_performance(foo_kn, thrust_fractions)

    with open(output_file, 'wb') as fp:
        tomli_w.dump({'LTO_performance': lto.model_dump()}, fp)


if __name__ == '__main__':
    run()
