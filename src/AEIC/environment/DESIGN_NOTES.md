# Environment Module

## Purpose

The goal of the environment module is to get detailed and all possible climate and air-quality impact metrics from the emissions output for a choice of configuration parameters, background scenarios, discount rates, etc.

The aim is to get all climate and AQ metrics from individual components available for different kinds of messaging. Certain users/audiences would prefer GWP or deltaT over Net Present Value/monetized damages.

While the module has all the pipeline to calculate all the metrics needed (list later in the doc), it also has the ability to switch out with external modules. For example, there is a way to quickly calculate RF from contrails using an estimate of RF/km contrail. But if the user wants they can also choose to calculate contrail impacts using PyContrails.

---

## High-Level Flow

### Climate

```
Emissions
   ↓
Radiative Forcing (RF)
   ↓
Temperature Change (ΔT)
   ↓
Climate Metrics (GWP, TP, ATR, CO2e)
   ↓
Damages ($)
   ↓
Discounting
   ↓
Net Present Value (NPV)
```

### Air Quality

```
Emissions
   ↓
Pollutant Concentration
   ↓
Concentration Response Functions
   ↓
Mortalities
   ↓
Value of Statistical Life
   ↓
Discounting
   ↓
Net Present Value (NPV)
```

Climate and air-quality paths can be thought of seperately and are merged only at the monetization/NPV level.

---

## Simplest Usage

```python
env_config = config.environment()

env = AEIC.environment.EnvironmentClass(config=env_config)

# Radiative forcing only
rf = env.climate.emit_RF(emissions=em)

# Full pipeline
out = env.emit(emissions=em)

print(f"""
RF CO2 (year 1): {rf.CO2[0]} W/m^2
ΔT CO2 (year 1): {out.climate.deltaT.CO2[0]} K
NPV NOx damages: {out.climate.NPV.NOx}
NPV total climate: {out.climate.NPV.total}
""")
```

---

## Supported Use Cases

### 1. Configuration Sensitivity

```python

RE_CO2_options = [1.0, 1.1, 0.9]

results = []
for RE_CO2_i in RE_CO2_options:
    env_config = config.environment(RE_CO2 = RE_CO2_i)
    env = EnvironmentClass(config=env_config)
    results.append(env.emit(emissions=em))

for i, r in enumerate(results):
    print(f"Config {i}: NPV = {r.climate.NPV.total}")
```

---

### 2. Swapping Physical Models

```python
# Contrails
env_config = config.environment(
    contrail_model="pycontrails"  # default: "simple")
)
env = EnvironmentClass(
    config=env_config,
)

# AQ adjoint sensitivities
env_config = config.environment(
    adjoint_sens_file="custom_adjoints.nc"
)
env = EnvironmentClass(
    config=env_config
)
```

---

### 3. Partial Pipelines

```python
climate_results = env.emit_climate(emissions=em)       # climate only
AQ_results = env.emit_AQ(emissions=em)   # AQ only
GWP_results = env.climate.get_GWP(emissions=em, time_horizon=100)
```

---

## Core Data Model

### Dimensional Convention

| Dimension   | Meaning                                       |
| ----------- | --------------------------------------------- |
| `forcer`.   | Forcing agent (CO2, contrails, O3, PM, etc.)  |
| `time`      | Years since emission (annual resolution)      |

**DIMENSIONS:**
All time-resolved outputs are shaped as:

```
(forcer × time)
```

---

### Emissions Input

**Class:** `EmissionsOutput`

| Field         | Units | Shape               |
| ------------- | ----- | ------------------- |
| fuel_burn     | kg    | (t_emit,)           |
| CO2           | kg    | (t_emit,)           |
| NOx           | kg    | (t_emit,)           |
| PM            | kg    | (t_emit,)           |
| H2O           | kg    | (t_emit,)           |
| flight_km     | km    | (t_emit,)           |
| CO2_lifecycle | kg    | (t_emit,)           |

Optional (for pycontrails):

**Class:** `Trajectory`

---

## Configuration (`EnvironmentConfig`)

### ClimateConfig

| Parameter              | Default   |
| ---------------------- | --------- |
| radiative_efficacy_CO2 | 1.0       |
| CO2_IRF_source         | Joos2013  |
| climate_sensitivity    | 3.0 K     |
| contrail_model         | simple    |
| time_horizon           | 100 years |
| time_step              | 1 year    |

### MonetizationConfig

| Parameter       | Default  |
| --------------- | -------- |
| discount_rate   | 3%       |
| discount_type   | constant |
| damage_function | DICE     |
| background_scenario    | SSP2-4.5  |

### Air Quality

| Parameter           | Default     |
| ------------------- | ----------- |
| VSL                 | $10M        |
| CRF_source          | Burnett2018 |
| adjoint_sens_source | GEOS-Chem   |

---

## Outputs (Unified Structure)

```python
EnvironmentOutput(
    climate=ClimateOutput(...),
    air_quality=AirQualityOutput(...)
)
```

---

## Climate Outputs

### Radiative Forcing — `RFOutput`

**Shape:** `(component, time)`

**Components (canonical):**

| Index | Component       |
| ----- | --------------- |
| 1     | CO2             |
| 2     | CO2_background  |
| 3     | CO2_lifecycle   |
| 4     | O3_short        |
| 5     | H2O_short       |
| 6     | contrails_short |
| 7     | sulfates_short  |
| 8     | soot_short      |
| 9     | nitrate_short   |
| 10    | O3_long         |
| 11    | CH4_long        |

Access:

```python
out.climate.RF.CO2
out.climate.RF.total
out.climate.RF.components
```

---

### Temperature Change — `TemperatureOutput`

Same structure as RF.

```python
out.climate.deltaT.CO2
out.climate.deltaT.total
```

---

### Damages — `DamageOutput`

* Climate damages: driven by ΔT
* AQ damages: driven by mortality

```python
out.climate.damage_costs.CO2
out.air_quality.damage_costs.PM25
out.damage_costs.total
```

---

### Discounted Damages — `DiscountedDamageOutput`

Same `(component, time)` shape.

---

### Net Present Value — `NPVOutput`

**Shape:** `(component,)`

```python
out.climate.NPV.CO2
out.climate.NPV.NOx
out.NPV.total
```

---

## Climate Metrics

### GWP

Stored per horizon:

```python
out.climate.GWP_20
out.climate.GWP_100
out.climate.GWP_500
```

Derived from GWP

```python
out.climate.get_CO2e(100)
out.climate.get_AGWP(100)
```

---

### TP and ATR

Derived from deltaT

```python
out.climate.get_TP
out.climate.get_ATR
```

---
