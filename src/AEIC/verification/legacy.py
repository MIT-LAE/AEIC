from pathlib import Path

import numpy as np
import pandas as pd

from AEIC.trajectories import Trajectory
from AEIC.types.species import Species, SpeciesValues
from AEIC.units import (
    FEET_TO_METERS,
    FPM_TO_MPS,
    MINUTES_TO_SECONDS,
    NAUTICAL_MILES_TO_METERS,
)


def process_matlab_csvs(traj_csv: Path, emissions_csv: Path, out_dir: Path):
    """Generate per-mission combined CSV files from MATLAB output.

    The MATLAB output comes in two CSV files: one for trajectories and one for
    emissions. This function combines the two files on common columns and
    generates separate CSV files, one per unique combination of departure
    airport, arrival airport, and aircraft type."""

    traj_df = pd.read_csv(traj_csv)
    emissions_df = pd.read_csv(emissions_csv)

    if not out_dir.exists():
        raise FileNotFoundError(f'Output directory {out_dir} does not exist.')

    # Iterate over missions:
    key_cols = ['airportDepart', 'airportArrive', 'AC']
    for key, df in traj_df.groupby(key_cols):
        assert isinstance(key, tuple)
        airportDepart, airportArrive, AC = key

        # Trajectory data for mission.
        tdf = (
            df.drop(columns=['airportDepart', 'airportArrive', 'AC'])
            .sort_values('t')
            .reset_index(drop=True)
        )

        # Emissions data for mission.
        edf = emissions_df[
            (emissions_df['airportDepart'] == airportDepart)
            & (emissions_df['airportArrive'] == airportArrive)
            & (emissions_df['AC'] == AC)
        ]
        assert isinstance(edf, pd.DataFrame)
        edf = (
            edf.drop(columns=['airportDepart', 'airportArrive', 'AC'])
            .sort_values('t')
            .reset_index(drop=True)
        )

        # Check for consistency. (There is one extra post-landing time point
        # for the trajectory data.)
        if (tdf.t[:-1] != edf.t).any():
            raise ValueError(f'Time columns do not match for mission {key}')

        # Combine trajectory and emissions data along columns and write to CSV.
        combined_df = pd.concat([tdf, edf.drop(columns='t')], axis='columns')
        fname = f'{airportDepart}_{airportArrive}_{AC}.csv'
        with open(out_dir / fname, 'w') as f:
            combined_df.to_csv(f, index=False)


class LegacyTrajectory:
    def __init__(self, csv_file: Path):
        self.df = pd.read_csv(csv_file)

    def trajectory(self) -> Trajectory:
        retval = Trajectory(npoints=len(self.df), fieldsets=['base', 'emissions'])
        retval.flight_time = self.df.t.values
        retval.latitude = self.df.lat.values
        retval.longitude = self.df.long.values
        retval.altitude = self.df.alt.values * FEET_TO_METERS
        retval.ground_distance = self.df.horDist.values * NAUTICAL_MILES_TO_METERS
        retval.azimuth = self.df.az.values
        retval.true_airspeed = self.df.TAS.values
        retval.rate_of_climb = self.df.roc_fpm.values * FPM_TO_MPS
        retval.aircraft_mass = self.df.acMass.values
        retval.fuel_flow = self.df.fuelFlow.values / MINUTES_TO_SECONDS
        emissions_indices = SpeciesValues[np.ndarray]()
        emissions_indices[Species.CO2] = self.df.EI_CO2.values
        emissions_indices[Species.H2O] = self.df.EI_H2O.values
        emissions_indices[Species.HC] = self.df.EI_HC.values
        emissions_indices[Species.CO] = self.df.EI_CO.values
        emissions_indices[Species.NOx] = self.df.EI_NOx.values
        emissions_indices[Species.PMnvol] = self.df.EI_PMnvol.values
        emissions_indices[Species.PMvol] = self.df.EI_PMvol.values
        emissions_indices[Species.SOx] = self.df.EI_SOx.values
        retval.trajectory_indices = emissions_indices

        return retval
