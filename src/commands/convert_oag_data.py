import logging
import os

import click
from tqdm import tqdm

import missions.oag as oag
from missions.oag.data import CSVEntry


@click.command()
@click.argument('in-file')
@click.argument('db-file')
def run(in_file, db_file):
    if os.environ.get('AEIC_DATA_DIR') is None:
        raise RuntimeError('AEIC_DATA_DIR environment variable is not set.')
    if os.path.exists(db_file):
        raise RuntimeError(f'Database file {db_file} already exists.')

    logging.basicConfig(level=logging.INFO)

    db = oag.Database(db_file, write_mode=True)

    nlines = -1  # (Skip header line.)
    with open(in_file) as fp:
        for _ in fp:
            nlines += 1

    n = 0
    for entry in tqdm(CSVEntry.read(in_file), total=nlines):
        db.add(entry, commit=False)
        n += 1
        if n % 10000 == 0:
            db.commit()

    db.commit()
    db.index()


if __name__ == '__main__':
    run()
