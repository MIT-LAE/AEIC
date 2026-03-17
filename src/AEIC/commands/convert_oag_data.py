import logging
import os

import click

import AEIC.missions.oag as oag

logger = logging.getLogger(__name__)


@click.command(
    short_help='Convert OAG data to SQLite database.',
    help="""Convert OAG data from a CSV file to an SQLite database format used
    by AEIC. The input CSV file should have the same format as the original OAG
    data files, with columns for origin. The output SQLite database will
    contain tables for airports, flights, and individual scheduled flight
    instances, which can be used for trajectory generation and analysis in
    AEIC. """,
)
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
def convert_oag_data(warnings_file, year, in_file, db_file):
    if os.environ.get('AEIC_PATH') is None:
        raise RuntimeError('AEIC_PATH environment variable is not set.')
    if os.path.exists(db_file):
        raise RuntimeError(f'Database file {db_file} already exists.')

    logging.basicConfig(level=logging.INFO)
    oag.convert_oag_data(in_file, year, db_file, warnings_file)
