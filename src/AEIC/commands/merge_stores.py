import logging

import click

from AEIC.trajectories import TrajectoryStore

logger = logging.getLogger(__name__)


@click.command(
    short_help='Merge trajectory stores from multiple slices.',
    help="""Merge trajectory stores from multiple slices into a single store. This
    is used to combine the results from parallel simulation runs that were split
    into slices. The input stores should have been generated with the same
    configuration and performance model, and should contain the same missions. The
    output store will contain the combined trajectories from all input stores.""",
)
@click.option(
    '--output-store',
    type=click.Path(),
    required=True,
    help='Path to the output trajectory store to create.',
)
@click.option(
    '--merge/--combine',
    default=True,
    help="""Whether to create a merged (multi-file) store or a combined
    (single-file) store.""",
)
@click.argument('input-stores', type=click.Path(exists=True), nargs=-1)
def merge_stores(output_store, merge, input_stores):
    if merge:
        logger.info(f'Merging trajectory stores: {input_stores}')
        TrajectoryStore.merge(output_store, input_stores)
    else:
        logger.info(f'Combining trajectory stores: {input_stores}')
        TrajectoryStore.combine(output_store, input_stores)
