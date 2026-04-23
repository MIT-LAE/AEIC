# Storage module

The {py:mod}`AEIC.storage` package provides the low-level building blocks
used by trajectories, trajectory stores, and the rest of AEIC for:

- representing per-point data with named field sets
  ({py:class}`FieldSet <AEIC.storage.FieldSet>`,
  {py:class}`FieldMetadata <AEIC.storage.FieldMetadata>`,
  {py:class}`HasFieldSets <AEIC.storage.HasFieldSets>`,
  {py:class}`Container <AEIC.storage.Container>`);
- describing the dimensions of those fields
  ({py:class}`Dimension <AEIC.storage.Dimension>`,
  {py:class}`Dimensions <AEIC.storage.Dimensions>`);
- tagging points with their flight phase
  ({py:class}`FlightPhase <AEIC.storage.FlightPhase>`, together with the
  {py:data}`PHASE_FIELDS <AEIC.storage.PHASE_FIELDS>` mapping of per-phase
  counter field names); and
- tracking provenance of the files that were read during a run
  ({py:class}`ReproducibilityData <AEIC.storage.ReproducibilityData>`,
  {py:func}`access_recorder <AEIC.storage.access_recorder>`,
  {py:func}`track_file_accesses <AEIC.storage.track_file_accesses>`).

Most user-facing code will reach these types indirectly, via
{py:class}`Trajectory <AEIC.trajectories.trajectory.Trajectory>` and
{py:class}`TrajectoryStore <AEIC.trajectories.store.TrajectoryStore>`. The
public surface is re-exported at package level so that everything can be
imported from `AEIC.storage` directly.

## API reference

```{eval-rst}
.. automodule:: AEIC.storage
   :members:
```
