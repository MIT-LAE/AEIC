import math
import os
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import AEIC.trajectories.builders as tb
from AEIC.config import Config, config
from AEIC.emissions import compute_emissions
from AEIC.missions import Mission
from AEIC.performance.models import PerformanceModel
from AEIC.trajectories import Trajectory
from AEIC.types import Fuel, Species, SpeciesValues
from AEIC.verification.legacy import LegacyTrajectory
from AEIC.verification.metrics import (
    ComparisonMetrics,
    ComparisonMetricsCollection,
    out_of_tolerance,
)

TRAJ_FIELDS = [
    'fuel_flow',
    'aircraft_mass',
    'ground_distance',
    'latitude',
    'longitude',
    'azimuth',
    'true_airspeed',
    'altitude',
    'rate_of_climb',
]

COMPARISON_FIELDS = TRAJ_FIELDS + ['trajectory_indices']


def metrics_page(
    pdf: PdfPages,
    mission_id: str,
    legacy_traj: Trajectory,
    new_traj: Trajectory,
    metrics: ComparisonMetricsCollection,
) -> None:
    # This is a slightly hacky way of doing this, but it lets us display the
    # metrics in a table without needing any external dependencies apart from
    # matplotlib.

    # Find the species to include in the table, which is the intersection of
    # species present in the legacy and new trajectories.
    species = species_to_plot(legacy_traj, new_traj)

    fig = plt.figure(figsize=(8.5, 11))

    # Spacing parameters.
    lmargin = 0.1
    tmargin = 0.8
    line_height = 0.025

    # Title: mission ID.
    fig.text(0.5, 0.9, mission_id, ha='center', va='center', fontsize=24, weight='bold')

    def t(x, line, txt):
        fig.text(
            lmargin + x,
            tmargin - line * line_height,
            txt,
            ha='left',
            va='center',
            fontsize=14,
        )

    # Display flight times.
    t(0.0, 0, 'Legacy flight time (s):')
    t(0.0, 1, 'New flight time (s):')
    t(0.26, 0, f'{legacy_traj.flight_time[-1]:.1f}')
    t(0.26, 1, f'{new_traj.flight_time[-1]:.1f}')
    t(0.5, 0, 'Legacy point count:')
    t(0.5, 1, 'New point count:')
    t(0.76, 0, f'{len(legacy_traj)}')
    t(0.76, 1, f'{len(new_traj)}')

    col_width = 0.1

    def cell(col, row, txt, bold=False, left=False):
        fig.text(
            lmargin + col * col_width,
            tmargin - (row + 3) * line_height,
            txt,
            ha='left' if left else 'center',
            va='center',
            fontsize=11,
            weight='bold' if bold else 'normal',
        )

    # Table column headers.
    for idx, label in enumerate(['RMSE', 'MAE', 'MAPE%', 'Max Err', 'Corr', 'R²']):
        cell(idx + 2.5, 0, label, bold=True)

    # Table row headers.
    for idx, field in enumerate(TRAJ_FIELDS):
        cell(0, idx + 1, field, bold=True, left=True)
    for idx, sp in enumerate(species):
        cell(0, idx + len(TRAJ_FIELDS) + 1, f'EI_{sp.name}', bold=True, left=True)

    # Table values.
    for idx, field in enumerate(TRAJ_FIELDS):
        ms = metrics[field]
        assert isinstance(ms, ComparisonMetrics)
        for col_idx, val in enumerate(
            [ms.rmse, ms.mae, ms.mape_pct, ms.max_error, ms.corr, ms.r2]
        ):
            cell(col_idx + 2.5, idx + 1, f'{val:.2f}')
    for idx, sp in enumerate(species):
        mss = metrics['trajectory_indices']
        assert isinstance(mss, SpeciesValues)
        ms = mss[sp]
        assert isinstance(ms, ComparisonMetrics)
        for col_idx, val in enumerate(
            [ms.rmse, ms.mae, ms.mape_pct, ms.max_error, ms.corr, ms.r2]
        ):
            cell(col_idx + 2.5, idx + len(TRAJ_FIELDS) + 1, f'{val:.2f}')

    pdf.savefig(fig)
    plt.close(fig)


def plot_trajectory_fields(
    pdf: PdfPages, mission_id: str, legacy_traj: Trajectory, new_traj: Trajectory
) -> None:
    plot_fields(
        pdf,
        mission_id,
        legacy_traj,
        new_traj,
        TRAJ_FIELDS,
        lambda t, f: getattr(t, f),
        lambda f: f,
    )


def species_to_plot(legacy_traj: Trajectory, new_traj: Trajectory) -> list[Species]:
    legacy_emissions = legacy_traj.trajectory_indices
    assert isinstance(legacy_emissions, SpeciesValues)
    new_emissions = new_traj.trajectory_indices
    assert isinstance(new_emissions, SpeciesValues)
    return sorted(set(legacy_emissions.keys()) & set(new_emissions.keys()))


def plot_emissions_fields(
    pdf: PdfPages,
    mission_id: str,
    legacy_traj: Trajectory,
    new_traj: Trajectory,
) -> None:
    def get_field(t: Trajectory, species: Species) -> np.ndarray:
        return t.trajectory_indices[species]

    def make_title(species: Species) -> str:
        return f'EI_{species.name}'

    plot_fields(
        pdf,
        mission_id,
        legacy_traj,
        new_traj,
        species_to_plot(legacy_traj, new_traj),
        get_field,
        make_title,
    )


def plot_fields[T](
    pdf: PdfPages,
    mission_id: str,
    legacy_traj: Trajectory,
    new_traj: Trajectory,
    fields: list[T],
    get_field: Callable[[Trajectory, T], np.ndarray],
    make_title: Callable[[T], str],
) -> None:
    base_fontsize = 16
    line_width = 3.2
    rc_params = {
        'font.size': base_fontsize,
        'axes.titlesize': base_fontsize + 2,
        'axes.labelsize': base_fontsize + 1,
        'xtick.labelsize': base_fontsize,
        'ytick.labelsize': base_fontsize,
        'legend.fontsize': base_fontsize,
        'figure.titlesize': base_fontsize + 3,
        'lines.linewidth': line_width,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
    }

    ncols = 2 if len(fields) > 1 else 1
    nrows = math.ceil(len(fields) / ncols)
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.5 * ncols, 3.2 * nrows),
            sharex=True,
        )
        axes = np.atleast_1d(axes).ravel()

        def p(ax, t, field, label):
            ax.plot(
                t.flight_time / 3600,
                get_field(t, field),
                label=label,
                linewidth=line_width,
            )

        for idx, field in enumerate(fields):
            ax = axes[idx]
            p(ax, legacy_traj, field, 'legacy')
            p(ax, new_traj, field, 'new')
            ax.set_title(make_title(field))
            ax.grid(True, alpha=0.3)

        for ax in axes[len(fields) :]:
            ax.set_visible(False)

        axes[0].legend(loc='best')
        fig.suptitle(mission_id)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    pdf.savefig()
    plt.close(fig)


@click.command()
@click.option(
    '--report-file',
    type=click.Path(exists=False, dir_okay=False),
    help='Path to output report file (CSV).',
)
def run(report_file) -> None:
    # Set up paths to test data.
    data_dir = Path(__file__).parent.parent / 'tests/data/verification/legacy'
    legacy_dir = data_dir / 'matlab-output'
    missions_file = data_dir / 'legacy_verf_missions.toml'
    perf_path = data_dir / 'legacy_verification.toml'

    # Set up AEIC configuration and set the AEIC_PATH to include the test data
    # directory.
    os.environ['AEIC_PATH'] = str(data_dir)
    Config.load(weather={'use_weather': False})

    # Load test data: performance model, missions, fuel file.
    pm = PerformanceModel.load(perf_path)
    with open(missions_file, 'rb') as fp:
        mission_dict = tomllib.load(fp)
    missions = Mission.from_toml(mission_dict)
    with open(config.emissions.fuel_file, 'rb') as fp:
        fuel = Fuel.model_validate(tomllib.load(fp))

    # Create a single trajectory builder to fly all missions.
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    failed = []

    with PdfPages(report_file) as pdf:
        for mission in missions:
            # Load legacy data for mission and convert to "new AEIC" format. (This
            # includes unit conversions.)
            legacy_traj_in = LegacyTrajectory(legacy_dir / f'{mission.label}.csv')
            legacy_traj = legacy_traj_in.trajectory()

            # Simulate mission and compute emissions with new AEIC.
            new_traj = builder.fly(pm, mission)
            new_traj.add_fields(compute_emissions(pm, fuel, new_traj))

            # For comparison, we need to interpolate the new AEIC trajectory onto
            # the same time points as the legacy trajectory, since they may not
            # match.
            interp_traj = new_traj.interpolate_time(legacy_traj.flight_time)

            # Compute comparison metrics.
            # dict[str, ComparisonMetrics | SpeciesValues[ComparisonMetrics]]
            metrics = legacy_traj.compare(interp_traj, COMPARISON_FIELDS)

            # Record any metrics that are outside tolerance.
            bad_metrics = out_of_tolerance(metrics, rtol=0.05, atol=1.0e-3)
            if len(bad_metrics) > 0:
                failed.append((mission.label, bad_metrics))

            # Add pages to report.
            metrics_page(pdf, mission.label, legacy_traj, new_traj, metrics)
            plot_trajectory_fields(pdf, mission.label, legacy_traj, new_traj)
            plot_emissions_fields(pdf, mission.label, legacy_traj, new_traj)

    if len(failed) > 0:
        print('Missions with metrics outside tolerance:')
        for mission_id, bad_metrics in failed:
            print(f'  {mission_id}:')
            for m in bad_metrics:
                print(f'    {m}')
        sys.exit(1)


if __name__ == '__main__':
    run()
