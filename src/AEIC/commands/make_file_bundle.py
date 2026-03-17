import logging
import zipfile

import click

from AEIC.trajectories import TrajectoryStore


@click.command(
    short_help='Create a file bundle for reproducibility.',
    help="""Create a file bundle containing all files
    required to reproduce the results in a trajectory store.""",
)
@click.option(
    '--input-store',
    type=click.Path(exists=True),
    required=True,
    help='Path to the input store referencing the files to bundle.',
)
@click.option(
    '--output-bundle',
    type=click.Path(),
    required=True,
    help='Path to the output bundle file to create.',
)
def make_file_bundle(input_store, output_bundle):
    with TrajectoryStore.open(base_file=input_store) as store:
        if store.reproducibility_data is None:
            logging.error(
                'The input store does not contain reproducibility data. '
                'Cannot create file bundle.'
            )
            return
        required_files = [
            str(p)
            for p in store.reproducibility_data.files
            if not str(p).startswith('...')
        ]
        print('Bundling files:')
        with zipfile.ZipFile(
            output_bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zipped:
            for f in required_files:
                print(f'  {f}')
                zipped.write(f)
