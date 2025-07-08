import click
from tqdm import tqdm

from parsers.OAG_reader import read_oag_file
from missions.OAG_db import OAGDatabase


@click.command()
@click.option('--in-file', '-i', help='OAG data CSV file')
@click.option('--db-file', '-d', help='Output database file')
def run(in_file, db_file):
    db = OAGDatabase(db_file, write_mode=True)

    with open(in_file) as fp:
        nlines = len(fp.readlines())

    n = 0
    for entry in tqdm(read_oag_file(in_file), total=nlines-1):
        db.add(entry, commit=False)
        n += 1
        if n % 10000 == 0:
            db.conn.commit()


if __name__ == '__main__':
    run()
