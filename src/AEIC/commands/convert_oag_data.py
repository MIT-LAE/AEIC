import logging
import os

import click

from AEIC.missions.oag import convert_oag_data

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '-w',
    '--warnings-file',
    type=click.Path(),
    default=None,
    help='File to write warnings to (default: stdout).',
)
@click.option(
    '-y', '--year', type=int, required=True, help='Year of the OAG data (e.g., 2023).'
)
@click.option(
    '-i',
    '--in-file',
    type=click.Path(exists=True),
    required=True,
    help='Input CSV file.',
)
@click.option(
    '-d',
    '--db-file',
    type=click.Path(),
    required=True,
    help='Output SQLite database file.',
)
def run(warnings_file, year, in_file, db_file):
    if os.environ.get('AEIC_DATA_DIR') is None:
        raise RuntimeError('AEIC_DATA_DIR environment variable is not set.')
    if os.path.exists(db_file):
        raise RuntimeError(f'Database file {db_file} already exists.')

    logging.basicConfig(level=logging.INFO)
    convert_oag_data(in_file, year, db_file, warnings_file)


if __name__ == '__main__':
    run()
