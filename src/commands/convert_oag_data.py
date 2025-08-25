import os

import click
from tqdm import tqdm

from missions.OAG_db import OAGDatabase
from parsers.OAG_reader import read_oag_file


@click.command()
@click.argument('in-file')
@click.argument('db-file')
def run(in_file, db_file):
    if os.path.exists(db_file):
        raise RuntimeError(f'Database file {db_file} already exists.')
    db = OAGDatabase(db_file, write_mode=True)

    nlines = -1  # (Skip header line.)
    with open(in_file) as fp:
        for _ in fp:
            nlines += 1

    n = 0
    for entry in tqdm(read_oag_file(in_file), total=nlines):
        db.add(entry, commit=False)
        n += 1
        if n % 10000 == 0:
            db.conn.commit()


if __name__ == '__main__':
    run()
