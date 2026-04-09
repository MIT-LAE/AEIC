#!/usr/bin/env python3

import logging

import click

from AEIC.commands.convert_oag_data import convert_oag_data
from AEIC.commands.make_file_bundle import make_file_bundle
from AEIC.commands.make_performance_model import make_performance_model
from AEIC.commands.merge_stores import merge_stores
from AEIC.commands.run_simulations import run_simulations
from AEIC.commands.trajectories_to_grid import trajectories_to_grid

logger = logging.getLogger(__name__)


def cli_safe():
    """Run the CLI, catching and logging any exceptions."""
    try:
        cli()
    except Exception as e:
        logger.exception('An error occurred: %s', e)
        raise click.ClickException(str(e)) from e


@click.group()
def cli():
    """AEIC CLI - tools for working with AEIC data and simulations."""
    pass


cli.add_command(convert_oag_data, name='convert-oag-data')
cli.add_command(make_file_bundle, name='make-file-bundle')
cli.add_command(make_performance_model, name='make-performance-model')
cli.add_command(merge_stores, name='merge-stores')
cli.add_command(run_simulations, name='run')
cli.add_command(trajectories_to_grid, name='trajectories-to-grid')
