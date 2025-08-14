import pandas as pd
import toml


def generate_engines_toml(excel_path: str, toml_path: str):
    """
    Reads the EDB Excel workbook and writes out a TOML file with
    one [[engine]] table per engine,
    combining data from the "Gaseous Emissions and Smoke" and "nvPM Emissions" sheets.
    """
    # Load both sheets
    xls = pd.ExcelFile(excel_path)
    gaseous = xls.parse('Gaseous Emissions and Smoke')
    nvpm = xls.parse('nvPM Emissions')

    # Define the four LTO modes in the desired order
    modes = ['Idle', 'App', 'C/O', 'T/O']

    # Determine the set of engine UIDs present in both sheets
    uids = set(gaseous['UID No']).intersection(nvpm['UID No'])

    engines = []
    for uid in sorted(uids):
        # Select the first matching row in each sheet (wide-format)
        g = gaseous[gaseous['UID No'] == uid].iloc[0]
        n = nvpm[nvpm['UID No'] == uid].iloc[0]

        # Build a dict for this engine
        entry = {
            'ENGINE': g['Engine Identification'],
            'UID': str(uid),
            'ENGINE_TYPE': g['Eng Type'],
            'BP_Ratio': float(g['B/P Ratio']),
            'fuelflow_KGperS': [
                float(g[f'Fuel Flow {mode} (kg/sec)']) for mode in modes
            ],
            'CO_EI_matrix': [float(g[f'CO EI {mode} (g/kg)']) for mode in modes],
            'HC_EI_matrix': [float(g[f'HC EI {mode} (g/kg)']) for mode in modes],
            'NOX_EI_matrix': [float(g[f'NOx EI {mode} (g/kg)']) for mode in modes],
            'SN_matrix': [float(g[f'SN {mode}']) for mode in modes],
            'nvPM_mass_matrix': [
                float(n[f'nvPM EImass {mode} (mg/kg)']) for mode in modes
            ],
            'nvPM_num_matrix': [
                float(n[f'nvPM EInum {mode} (#/kg)']) for mode in modes
            ],
            'PR': [float(g['Pressure Ratio'])] * len(modes),
            'SNmax': [float(g['SN Max'])] * len(modes),
            'EImass_max': float(n['nvPM EImass Max (mg/kg)']),
            'EImass_max_thrust': -1.0,
            'EInum_max': float(n['nvPM EInum Max (#/kg)']),
            'EInum_max_thrust': -1.0,
        }
        engines.append(entry)

    # Write out the TOML
    with open(toml_path, 'w') as f:
        toml.dump({'engine': engines}, f)


if __name__ == '__main__':
    excel_file = 'edb-emissions-databank_v30__web_ (1).xlsx'  # adjust
    output_toml = 'engines.toml'
    generate_engines_toml(excel_file, output_toml)
    print(f'Generated TOML file: {output_toml}')
