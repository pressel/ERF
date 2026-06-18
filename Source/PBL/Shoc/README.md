# ERF-native SHOC

This directory contains the ERF-native implementation of the Simplified
Higher-Order Closure (SHOC).

SHOC represents unresolved turbulence, shallow convection, and subgrid-scale
cloud macrophysics. It uses prognostic turbulent kinetic energy, diagnostic
higher-order moments, and an assumed probability density function (PDF) for
subgrid thermodynamic variability.

## Provenance

This implementation is based on the E3SM/EAMxx SHOC algorithms and source
structure. It is not a bit-for-bit port of EAMxx SHOC.

The native ERF implementation has been adapted to ERF data structures, AMReX
loops, ERF surface-flux coupling, diagnostics, and build systems.

Primary reference:

```text
Bogenschutz, P. A., and S. K. Krueger, 2013:
A simplified PDF parameterization of subgrid-scale clouds and turbulence
for cloud-resolving models.
Journal of Advances in Modeling Earth Systems, 5, 195-211.
https://doi.org/10.1002/jame.20018
```

Upstream project reference:

```text
E3SM Project
https://github.com/E3SM-Project/E3SM
```

Reference E3SM commit used during this port:

```text
0053a696ce40fc5ffa0b07a007426940b8af09a1
```

Keep the required license notices when moving, copying, or modifying code derived
from E3SM.

## Runtime selection

Native SHOC is always built in tree. It does not require EAMxx, EKAT, or Kokkos.

Select it at runtime:

```text
zlo.type = "surface_layer"

erf.pbl_type = NATIVE_SHOC
```

ERF also supports the optional EAMxx SHOC interface:

```text
erf.pbl_type = EAMXX_SHOC
```

The legacy runtime name:

```text
erf.pbl_type = SHOC
```

is a deprecated alias for `EAMXX_SHOC`. It does not select native SHOC.

Do not add new build workflows that use `ERF_ENABLE_NATIVE_SHOC` or
`USE_NATIVE_SHOC`. Native SHOC is an in-tree ERF implementation. Runtime
selection controls whether it runs.

## Surface fluxes

SHOC needs lower-boundary heat, moisture, and momentum fluxes.

Native SHOC does not compute those exchanges directly from a land surface model.
ERF computes the lower-boundary fluxes first and passes the resulting flux arrays
to SHOC.

Those fluxes come through ERF's surface-layer infrastructure. They may come from
Monin-Obukhov similarity theory (MOST), prescribed surface-layer inputs, or an
active land or ocean surface model when that model provides fluxes.

Native SHOC consumes the ERF flux arrays and converts ERF's host
density-weighted fluxes to kinematic surface fluxes.

## Microphysics coupling

SHOC diagnoses subgrid non-precipitating cloud partitioning with its assumed PDF.
This includes cloud fraction and non-precipitating liquid water.

Native SHOC remains a liquid-cloud macrophysics closure under the interim ice
contract. It may use pre-existing cloud ice for phase-aware thermodynamics and
buoyancy, but it does not create or repartition cloud ice.

To avoid double counting, ERF disables the saturation-adjustment or condensation
step in the microphysics package when a SHOC-family PBL scheme is active.

This does not disable microphysics. Microphysics still handles precipitating
processes outside SHOC's cloud macrophysics role. Number-aware microphysics
layouts with cloud-droplet or ice number concentrations still need an explicit
number closure in their own microphysics pathways if they are coupled to SHOC.
Native SHOC `state_update` rejects those number-aware layouts until that
number closure exists.

## Transport modes

Native SHOC has two transport modes.

The default mode is:

```text
erf.shoc.transport_mode = state_update
```

In this mode, SHOC applies its coupled column increment before the dycore sees
the state. This avoids splitting theta and momentum from moisture across ERF's
fast and slow update channels.

The host-diffusion mode is:

```text
erf.shoc.transport_mode = host_diffusion
```

In this mode, SHOC exports eddy diffusivities to ERF's host diffusion path.
SHOC does not apply the pre-dycore state update. At present this mode is only
supported for dry/no-moisture configurations.

## ERF time-step integration

Native SHOC runs inside `ERF::Advance()` once per ERF time step on each
SHOC-active level. At the start of the step, ERF swaps old and new state
pointers, fill-patches level data when needed, and converts old velocities to
old momenta. If the lower boundary uses the surface-layer boundary condition,
ERF updates the surface-layer state and computes the lower-boundary heat,
moisture, and momentum flux arrays before SHOC runs. Those fluxes may come from
Monin-Obukhov similarity theory (MOST), prescribed surface-layer inputs, or an
active land or ocean surface model when that model provides fluxes. Radiation
sources are also updated on the old state before SHOC runs.

When `erf.pbl_type = NATIVE_SHOC`, ERF calls `compute_native_shoc_tendencies()`
before `advance_dycore()`. The coupling call passes the old conserved state,
face velocities, vertical velocity, surface stress and flux arrays,
eddy-diffusivity storage, terrain metrics, optional subsidence data, and `dt` to
`ShocDriver::advance()`. The driver preprocesses each full-height AMReX box into
SHOC column data. It reads density, potential temperature, water species,
pressure and temperature diagnostics, velocities, geometry, carried turbulence
state, and lower-boundary fluxes. It converts ERF's density-weighted surface
fluxes and stresses to the kinematic quantities used by the SHOC column solve.

The coupling then depends on `erf.shoc.transport_mode`.

In the default `state_update` mode, SHOC computes one coupled column increment
for potential temperature, water vapor, non-precipitating cloud liquid,
pre-existing cloud ice, turbulent kinetic energy, and horizontal velocity. It
applies that increment directly to the old ERF state before the dycore sees the
state. This update does not change density, vertical velocity, passive scalars,
or precipitating species. After SHOC consumes the lower-boundary fluxes and
stresses, it clears the corresponding ERF arrays so the host diffusion path does
not apply the same surface terms a second time. The native SHOC fast and slow
right-hand-side hooks remain no-ops in this mode. This prevents double
application and avoids splitting SHOC's coupled thermodynamic and momentum
update across ERF's fast and slow integrator channels.

After the `state_update` increment, `ERF::Advance()` immediately refreshes
boundary and halo data before any pre-dycore checks or dycore precomputations.
On level 0, `FillPatchCrseLevel()` refills `S_old`, `U_old`, `V_old`, and
`W_old`, and `VelocityToMomentum()` rebuilds `rU_old`, `rV_old`, and `rW_old`
from the synchronized velocities and density. On refined levels,
`FillPatchFineLevel()` refills conserved state, velocities, and momenta through
the coarse/fine fill-patch machinery. This synchronization happens before the
pre-dycore NaN and temperature checks and before `advance_dycore()` computes
strain, eddy viscosity, primitive variables, Exner pressure, fast coefficients,
and the multirate time integrator.

During the dycore step, native SHOC in `state_update` mode does not add fast or
slow RHS source terms. ERF still builds host turbulence and diffusion terms as
needed. `ShocDriver::set_eddy_diffs()` clears the SHOC-owned vertical
diffusivity components from the host eddy-diffusivity arrays, except for
non-transport diagnostics such as the SHOC length scale, so the host does not
reapply SHOC vertical transport. `ShocDriver::set_diff_stresses()` clears the
SHOC-consumed lower-boundary surface terms when `owns_surface_fluxes()` is true.

In `host_diffusion` mode, SHOC does not apply the pre-dycore state update and
does not own the surface fluxes. Instead, SHOC exports vertical eddy-diffusivity
coefficients to ERF's host diffusion path. This mode is currently limited to
dry/no-moisture configurations because SHOC-family microphysics suppresses
saturation-adjustment or condensation while host-diffusion SHOC does not own the
cloud-macrophysics mass update.

Microphysics runs after the dycore unless `erf.moisture_tight_coupling = true`.
With the default loose coupling, `ERF::Advance()` calls `advance_microphysics()`
after `advance_dycore()` produces `S_new`. With tight coupling, ERF calls
microphysics inside the slow post-RHS path. In both cases, SHOC-family PBL
selection disables the microphysics saturation-adjustment or condensation step to
avoid double-counting SHOC's non-precipitating liquid-cloud macrophysics.
Microphysics still handles precipitating processes outside SHOC's cloud
macrophysics role. After loose-coupled microphysics, ERF advances the land-surface
model. That land-surface update affects fluxes used by later surface-layer
updates; it is not a second SHOC surface-flux application in the same pre-dycore
state update.

## Runtime options

Native SHOC reads options from the `erf.shoc` namespace.

The source of truth for option names, defaults, and validation rules is:

```text
ERF_ShocTypes.H
```

The user-facing documentation is:

```text
Docs/sphinx_doc/theory/PBLschemes.rst
```

Do not duplicate the full options table here. Keep the Sphinx documentation as
the user manual.

## Code organization

The main source files are:

```text
ERF_ShocDriver.cpp       Driver and ERF-facing orchestration
ERF_ShocPreprocess.cpp   ERF state and flux arrays to SHOC column data
ERF_ShocStructure.cpp    SHOC structure diagnostics
ERF_ShocTKE.cpp          TKE and eddy diffusivity diagnostics
ERF_ShocMoments.cpp      Second- and third-moment diagnostics
ERF_ShocPDF.cpp          Assumed-PDF cloud diagnostics
ERF_ShocEnergyFixer.cpp  Energy and consistency fixes
ERF_ShocImplicit.cpp     Implicit state-update reconstruction
ERF_ShocDiagnostics.cpp  Diagnostic sequencing
ERF_ShocCoupling.cpp     ERF coupling helpers
```

The main headers are:

```text
ERF_ShocTypes.H
ERF_ShocColumnData.H
ERF_ShocDriver.H
```

## Developer notes

Like ERF's other column physics, it requires each AMReX box on a SHOC-active
level to span the full vertical domain. Do not use a grid decomposition that
splits boxes in the vertical direction. If an input file sets vector-valued
grid sizing controls, choose a vertical size at least as large as the level's
vertical cell count. With AMR, SHOC-active refined grids must also cover full
vertical columns.

Keep compile-time availability separate from runtime behavior. Native SHOC may be
built, but it must not change a non-SHOC simulation unless the runtime PBL type
selects native SHOC.

Guard native-SHOC behavior with runtime checks such as:

```cpp
uses_native_shoc()
uses_shoc_family()
```

Do not use compile-time SHOC macros as a substitute for runtime selection.

Keep AMReX portability in mind. Avoid host-only objects in device paths. Avoid
capturing `this` in GPU lambdas. Use AMReX math utilities and `amrex::Real`
literals in device-callable code.

## Testing

Run the native SHOC unit tests after changing this directory:

```bash
ctest --test-dir build -L shoc --output-on-failure
```

Also run representative non-SHOC regression tests after changing shared coupling,
microphysics, plotfile, or time-integration paths. Native SHOC must not alter
non-SHOC results when it is not selected at runtime.

## See also

User documentation:

```text
Docs/sphinx_doc/theory/PBLschemes.rst
```

Build documentation:

```text
Docs/sphinx_doc/buildingLibraryConfig.rst
Docs/sphinx_doc/buildingSystems.rst
```
