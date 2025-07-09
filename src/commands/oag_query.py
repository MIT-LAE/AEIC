import os
import re

import click

from missions.OAG_db import Comparison, Condition, OAGDatabase

CONDITION_RE = re.compile(r'^([a-zA-Z0-9_]+)([=<>!]+)(.*)$')


@click.command()
@click.argument('db-file')
@click.option('--limit', '-l', default=10, type=int,
              help='Limit the number of results to display')
@click.argument('conditions', nargs=-1)
def run(db_file, limit, conditions: tuple[str, ...]):
    if not os.path.exists(db_file):
        raise RuntimeError(f'Database file {db_file} does not exist.')

    db = OAGDatabase(db_file, write_mode=False)
    if not conditions:
        for index, entry in enumerate(db()):
            if index >= limit:
                break
            print(entry)
            print('')
    else:
        query_conditions = []
        for condition in conditions:
            result = CONDITION_RE.match(condition)
            if not result:
                raise ValueError(f'Invalid condition: {condition}')
            field, cond, value = result.groups()
            query_conditions.append(
                Condition(
                    field=field,
                    value=value.strip(),
                    comp=Comparison(cond)
                )
            )

        for index, entry in enumerate(db(*query_conditions)):
            if index >= limit:
                break
            print(entry)
            print('')


if __name__ == '__main__':
    run()
