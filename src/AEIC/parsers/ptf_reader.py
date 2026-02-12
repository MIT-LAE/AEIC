import re
from dataclasses import dataclass

from AEIC.trajectories.phase import FlightPhase
from AEIC.types import SpeedData, Speeds
from AEIC.units import FPM_TO_MPS, KNOTS_TO_MPS, MINUTES_TO_SECONDS


@dataclass
class ClimbPhaseData:
    """One row of PTF file performance data for climb phase."""

    fl: int
    """Flight level."""

    tas: float
    """True airspeed at nominal mass [m/s]."""

    rocd_low: float
    """Rate of climb at low mass [m/s]."""

    rocd_nom: float
    """Rate of climb at nominal mass [m/s]."""

    rocd_high: float
    """Rate of climb at high mass [m/s]."""

    fuel_flow_nom: float
    """Fuel flow at nominal mass [kg/s]."""


@dataclass
class CruisePhaseData:
    """One row of PTF file performance data for cruise phase."""

    fl: int
    """Flight level."""

    tas: float
    """True airspeed at nominal mass [m/s]."""

    fuel_flow_low: float
    """Fuel flow at low mass [kg/s]."""

    fuel_flow_nom: float
    """Fuel flow at nominal mass [kg/s]."""

    fuel_flow_high: float
    """Fuel flow at high mass [kg/s]."""


@dataclass
class DescentPhaseData:
    """One row of PTF file performance data for descent phase."""

    fl: int
    """Flight level."""

    tas: float
    """True airspeed at nominal mass [m/s]."""

    rocd_nom: float
    """Rate of descent at nominal mass [m/s]."""

    fuel_flow_nom: float
    """Fuel flow at nominal mass [kg/s]."""


@dataclass
class PTFData:
    """Internal representation of the contents of a BADA PTF file."""

    aircraft_type: str
    temperature_reference: str
    maximum_altitude_ft: int
    maximum_payload: int
    low_mass: int
    nominal_mass: int
    high_mass: int
    climb: list[ClimbPhaseData]
    cruise: list[CruisePhaseData]
    descent: list[DescentPhaseData]
    speeds: Speeds

    @property
    def isa_offset(self) -> int:
        """Estimate ISA offset from temperature reference string."""
        match = re.search(r'ISA\s*([+-]\d+)', self.temperature_reference)
        if match:
            return int(match.group(1))
        else:
            return 0

    @classmethod
    def load(cls, file_path):
        """Reads a BADA-format PTF file in a single pass."""

        climb_data: list[ClimbPhaseData] = []
        cruise_data: list[CruisePhaseData] = []
        descent_data: list[DescentPhaseData] = []
        speed_data: dict[FlightPhase, SpeedData] = {}

        aircraft_type = 'unknown'
        temperature_reference = 'ISA'

        # All the following are initialized to zero to ensure types are int.
        maximum_altitude_ft = 0
        maximum_payload = 0
        low_mass = 0
        nominal_mass = 0
        high_mass = 0

        capture = False

        with open(file_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 1) Parse top-level lines before/after the table for
                # alt/payload/mass and speeds:
                if not capture:
                    # Aircraft type
                    match_ac_type = re.search(r"AC/Type\s*:\s*([^_]+)_*\s*$", line)
                    if match_ac_type:
                        aircraft_type = match_ac_type.group(1)

                    # Temperature reference
                    match_temp = re.search(r"Temperature\s*:\s*(.+)\s*$", line)
                    if match_temp:
                        temperature_reference = match_temp.group(1)

                    # Max altitude
                    match_alt = re.search(r"Max Alt\.\s*\[ft\]:\s*([\d,]+)", line)
                    if match_alt:
                        maximum_altitude_ft = int(match_alt.group(1).replace(',', ''))

                    # Max payload
                    match_payload = re.search(r"Max Payload\s*\[kg\]:\s*([\d,]+)", line)
                    if match_payload:
                        maximum_payload = int(match_payload.group(1).replace(',', ''))

                    # Mass levels
                    # low mass
                    if 'low' in line and 'climb' in line:
                        match = re.search(r'low\s*-\s*(\d+)', line)
                        if match:
                            low_mass = int(match.group(1))
                    # nominal mass
                    elif 'nominal' in line and 'cruise' in line:
                        match = re.search(r'nominal\s*-\s*(\d+)', line)
                        if match:
                            nominal_mass = int(match.group(1))
                    # high mass
                    elif 'high' in line and 'descent' in line:
                        match = re.search(r'high\s*-\s*(\d+)', line)
                        if match:
                            high_mass = int(match.group(1))

                    # Parse speeds for each phase if the line starts with
                    # climb, cruise, or descent
                    def process(phase, stripped):
                        tokens = stripped.split()
                        if len(tokens) >= 4:
                            try:
                                # Expecting a format like "250/300" for CAS, and,
                                # e.g. "0.80" for Mach number.
                                cas_low, cas_high = tokens[2].split('/')
                                speed_data[phase] = SpeedData(
                                    cas_low=int(cas_low) * KNOTS_TO_MPS,
                                    cas_high=int(cas_high) * KNOTS_TO_MPS,
                                    mach=float(tokens[3]),
                                )
                            except Exception:
                                pass

                    stripped = line.lstrip()
                    if stripped == '':
                        continue
                    match stripped.split()[0]:
                        case 'climb':
                            process(FlightPhase.CLIMB, stripped)
                        case 'cruise':
                            process(FlightPhase.CRUISE, stripped)
                        case 'descent':
                            process(FlightPhase.DESCENT, stripped)

                # 2) Detect the start of the flight-level table
                if 'FL |' in line:
                    capture = True
                    continue

                # 3) Once capturing, parse columns for FL, climb, cruise, descent
                if capture:
                    parts = line.split('|')
                    # We expect 4 columns: FL, CRUISE, CLIMB, DESCENT
                    if len(parts) < 4:
                        continue

                    # Flight level in parts[0]
                    try:
                        fl = int(parts[0].strip())
                    except ValueError:
                        # If we can't parse a valid FL, skip
                        continue

                    # ============ CRUISE ============
                    cruise_str = parts[1]
                    # Typically we expect [TAS, fuel_low, fuel_nom, fuel_high]
                    c_vals = re.findall(r"\d+\.?\d*", cruise_str)
                    if len(c_vals) >= 4:
                        cruise_data.append(
                            CruisePhaseData(
                                fl=fl,
                                tas=float(c_vals[0]) * KNOTS_TO_MPS,
                                fuel_flow_low=float(c_vals[1]) / MINUTES_TO_SECONDS,
                                fuel_flow_nom=float(c_vals[2]) / MINUTES_TO_SECONDS,
                                fuel_flow_high=float(c_vals[3]) / MINUTES_TO_SECONDS,
                            )
                        )

                    # ============ CLIMB ============
                    climb_str = parts[2]
                    # Typically we expect [TAS, rocd_low, rocd_nom, rocd_high, fuel_nom]
                    cl_vals = re.findall(r"\d+\.?\d*", climb_str)
                    if len(cl_vals) >= 5:
                        climb_data.append(
                            ClimbPhaseData(
                                fl=fl,
                                tas=float(cl_vals[0]) * KNOTS_TO_MPS,
                                rocd_low=float(cl_vals[1]) * FPM_TO_MPS,
                                rocd_nom=float(cl_vals[2]) * FPM_TO_MPS,
                                rocd_high=float(cl_vals[3]) * FPM_TO_MPS,
                                fuel_flow_nom=float(cl_vals[4]) / MINUTES_TO_SECONDS,
                            )
                        )

                    # ============ DESCENT ============
                    descent_str = parts[3]
                    # Typically we expect [TAS_nom, rocd_nom, fuel_nom]
                    d_vals = re.findall(r"\d+\.?\d*", descent_str)
                    if len(d_vals) >= 3:
                        descent_data.append(
                            DescentPhaseData(
                                fl=fl,
                                tas=float(d_vals[0]) * KNOTS_TO_MPS,
                                # Descent ROCD is negative
                                rocd_nom=-float(d_vals[1]) * FPM_TO_MPS,
                                fuel_flow_nom=float(d_vals[2]) / MINUTES_TO_SECONDS,
                            )
                        )

        return cls(
            aircraft_type=aircraft_type,
            temperature_reference=temperature_reference,
            maximum_altitude_ft=maximum_altitude_ft,
            maximum_payload=maximum_payload,
            low_mass=low_mass,
            nominal_mass=nominal_mass,
            high_mass=high_mass,
            climb=climb_data,
            cruise=cruise_data,
            descent=descent_data,
            speeds=Speeds(
                climb=speed_data[FlightPhase.CLIMB],
                cruise=speed_data[FlightPhase.CRUISE],
                descent=speed_data[FlightPhase.DESCENT],
            ),
        )
