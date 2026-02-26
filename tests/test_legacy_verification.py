import math
import os
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import AEIC.trajectories.builders as tb

# load AEIC modules
from AEIC.config import Config, config
from AEIC.emissions import compute_emissions
from AEIC.missions import Mission
from AEIC.performance.models import PerformanceModel
from AEIC.types import Fuel, Species
from AEIC.units import (
    FPM_TO_MPS,
    METERS_TO_FEET,
    MINUTES_TO_SECONDS,
    NAUTICAL_MILES_TO_METERS,
)

MPS_TO_FPM = 1.0 / FPM_TO_MPS
METERS_TO_NM = 1.0 / NAUTICAL_MILES_TO_METERS

TRAJ_FIELDS = [
    'fuelFlow',
    'acMass',
    'horDist',
    'lat',
    'long',
    'az',
    'TAS',
    'alt',
    'roc_fpm',
]

EMISSIONS_FIELDS = [
    'EI_CO2',
    'EI_H2O',
    'EI_HC',
    'EI_CO',
    'EI_NOx',
    'EI_PMnvol',
    'EI_PMvol',
    'EI_SOx',
]

DEFAULT_TEST_DATA_DIR = "/home/aditeya/AEIC/tests/data/legacy_verification/"


def _interp_to_legacy(t_new: np.ndarray, y_new: np.ndarray, t_old: np.ndarray):
    t_new = np.asarray(t_new, dtype=float)
    y_new = np.asarray(y_new, dtype=float)
    t_old = np.asarray(t_old, dtype=float)
    if t_new.size == 0 or t_old.size == 0:
        return np.full_like(t_old, np.nan, dtype=float)
    if not np.all(np.diff(t_new) >= 0):
        order = np.argsort(t_new)
        t_new = t_new[order]
        y_new = y_new[order]
    return np.interp(t_old, t_new, y_new, left=np.nan, right=np.nan)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'mape_pct': float('nan'),
            'max_error': float('nan'),
            'corr': float('nan'),
            'r2': float('nan'),
            'n': 0,
        }

    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    diff = y_pred_m - y_true_m
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_error = float(np.max(np.abs(diff)))

    denom_mask = np.abs(y_true_m) > 1e-12
    if np.any(denom_mask):
        mape_pct = float(
            np.mean(np.abs(diff[denom_mask] / y_true_m[denom_mask])) * 100.0
        )
    else:
        mape_pct = float('nan')

    if y_true_m.size < 2 or np.std(y_true_m) == 0 or np.std(y_pred_m) == 0:
        corr = float('nan')
    else:
        corr = float(np.corrcoef(y_true_m, y_pred_m)[0, 1])

    ss_res = float(np.sum((y_true_m - y_pred_m) ** 2))
    ss_tot = float(np.sum((y_true_m - np.mean(y_true_m)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    return {
        'rmse': rmse,
        'mae': mae,
        'mape_pct': mape_pct,
        'max_error': max_error,
        'corr': corr,
        'r2': r2,
        'n': int(y_true_m.size),
    }


def fix_trajectory_legacy_fields(traj) -> dict[str, np.ndarray]:
    # OLD MATLAB HAS DIFFERENT UNITS
    return {
        'fuelFlow': traj.fuel_flow * MINUTES_TO_SECONDS,
        'acMass': traj.aircraft_mass,
        'horDist': traj.ground_distance * METERS_TO_NM,
        'lat': traj.latitude,
        'long': traj.longitude,
        'az': traj.azimuth,
        'TAS': traj.true_airspeed,
        'alt': traj.altitude * METERS_TO_FEET,
        'roc_fpm': traj.rate_of_climb * MPS_TO_FPM,
    }


def _emission_array(indices, species: Species, npoints: int) -> np.ndarray:
    if species in indices:
        return np.asarray(indices[species], dtype=float)
    return np.zeros(npoints, dtype=float)


def make_emissions_legacy_fields(emissions, npoints: int) -> dict[str, np.ndarray]:
    indices = emissions.trajectory_indices
    so2 = _emission_array(indices, Species.SO2, npoints)
    so4 = _emission_array(indices, Species.SO4, npoints)
    return {
        'EI_CO2': _emission_array(indices, Species.CO2, npoints),
        'EI_H2O': _emission_array(indices, Species.H2O, npoints),
        'EI_HC': _emission_array(indices, Species.HC, npoints),
        'EI_CO': _emission_array(indices, Species.CO, npoints),
        'EI_NOx': _emission_array(indices, Species.NOx, npoints),
        'EI_PMnvol': _emission_array(indices, Species.PMnvol, npoints),
        'EI_PMvol': _emission_array(indices, Species.PMvol, npoints),
        'EI_SOx': so2 + so4,
    }


def plot_fields(
    plot_dir: Path,
    mission_id: str,
    t_old: np.ndarray,
    legacy_df: pd.DataFrame,
    t_new: np.ndarray,
    new_fields: dict[str, np.ndarray],
    fields: list[str],
    title_prefix: str,
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

        for idx, field in enumerate(fields):
            ax = axes[idx]
            ax.plot(
                t_old,
                legacy_df[field].to_numpy(),
                label='legacy',
                linewidth=line_width,
            )
            ax.plot(
                t_new,
                new_fields[field],
                label='new',
                linewidth=line_width,
            )
            ax.set_title(field)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(fields) :]:
            ax.set_visible(False)

        axes[0].legend(loc='best')
        fig.suptitle(f'{title_prefix} {mission_id}')
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    plot_path = plot_dir + f'{title_prefix}_{mission_id}.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    data_dir = DEFAULT_TEST_DATA_DIR
    legacy_dir = data_dir
    traj_path = data_dir + 'AEIC_OUTPUT_MATLAB_TRAJ.csv'
    emis_path = data_dir + 'AEIC_OUTPUT_MATLAB_EMISSIONS.csv'
    missions_path = data_dir + 'legacy_verf_missions.toml'
    perf_path = data_dir + 'legacy_verification.toml'
    plot_dir = legacy_dir + 'plots/'

    os.environ['AEIC_PATH'] = str(data_dir)
    Config.load(weather={'use_weather': False}, data_path_overrides=[data_dir])

    pm = PerformanceModel.load(str(perf_path))

    with open(missions_path, 'rb') as f:
        mission_dict = tomllib.load(f)
    sample_missions = Mission.from_toml(mission_dict)

    legacy_traj = pd.read_csv(traj_path)
    legacy_emis = pd.read_csv(emis_path)

    traj_groups = {
        key: df.sort_values('t').reset_index(drop=True)
        for key, df in legacy_traj.groupby(['airportDepart', 'airportArrive'])
    }
    emis_groups = {
        key: df.sort_values('t').reset_index(drop=True)
        for key, df in legacy_emis.groupby(['airportDepart', 'airportArrive'])
    }

    with open(config.emissions.fuel_file, 'rb') as fp:
        fuel = Fuel.model_validate(tomllib.load(fp))

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    traj_metrics = []
    emis_metrics = []

    for mission in sample_missions:
        key = (mission.origin, mission.destination)
        legacy_traj_df = traj_groups[key]
        legacy_emis_df = emis_groups[key]

        trajectory = builder.fly(pm, mission)
        emissions = compute_emissions(pm, fuel, trajectory)

        t_old_traj = legacy_traj_df['t'].to_numpy()
        t_old_emis = legacy_emis_df['t'].to_numpy()
        t_new = trajectory.flight_time

        new_traj_fields = fix_trajectory_legacy_fields(trajectory)
        new_emis_fields = make_emissions_legacy_fields(emissions, len(trajectory))

        traj_interp = {}
        # NEED TO INTERPOLATE TO GET METRICS.. not sure if this is the right way
        for field in TRAJ_FIELDS:
            new_interp = _interp_to_legacy(t_new, new_traj_fields[field], t_old_traj)
            traj_interp[field] = new_interp
            metrics = _metrics(legacy_traj_df[field].to_numpy(), new_interp)
            traj_metrics.append(
                {
                    'mission': f'{mission.origin}_{mission.destination}',
                    'field': field,
                    **metrics,
                }
            )

        emis_interp = {}
        for field in EMISSIONS_FIELDS:
            new_interp = _interp_to_legacy(t_new, new_emis_fields[field], t_old_emis)
            emis_interp[field] = new_interp
            metrics = _metrics(legacy_emis_df[field].to_numpy(), new_interp)
            emis_metrics.append(
                {
                    'mission': f'{mission.origin}_{mission.destination}',
                    'field': field,
                    **metrics,
                }
            )

        mission_id = f'{mission.origin}_{mission.destination}_{mission.aircraft_type}'
        plot_fields(
            plot_dir,
            mission_id,
            t_old_traj,
            legacy_traj_df,
            t_new,
            new_traj_fields,
            TRAJ_FIELDS,
            'trajectory',
        )
        plot_fields(
            plot_dir,
            mission_id,
            t_old_emis,
            legacy_emis_df,
            t_new,
            new_emis_fields,
            EMISSIONS_FIELDS,
            'emissions',
        )

    traj_metrics_df = pd.DataFrame(traj_metrics)
    emis_metrics_df = pd.DataFrame(emis_metrics)

    print('\nTRAJECTORY:')
    print(traj_metrics_df.to_string(index=False))
    print('\nEMISSIONS EI')
    print(emis_metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
