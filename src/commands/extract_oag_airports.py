import logging
import os

import click
from tqdm import tqdm

from missions.oag.data import CSVEntry


@click.command()
@click.argument('in-file')
@click.argument('out-file')
def run(in_file, out_file):
    if os.path.exists(out_file):
        raise RuntimeError(f'Output file {out_file} already exists.')

    logging.basicConfig(level=logging.INFO)

    nlines = -1  # (Skip header line.)
    with open(in_file) as fp:
        for _ in fp:
            nlines += 1

    airports = set()
    for entry in tqdm(CSVEntry.read(in_file), total=nlines):
        if entry is None:
            continue
        airports.add(entry.depapt)
        airports.add(entry.arrapt)

    with open(out_file, 'w') as fp:
        for airport in sorted(airports):
            print(airport, file=fp)


if __name__ == '__main__':
    run()
