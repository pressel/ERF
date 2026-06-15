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
layouts with cloud-droplet or ice number concentrations need an explicit number
closure before SHOC tendencies mode can couple to them.

## Transport modes

Native SHOC has two transport modes.

The default mode is:

```text
erf.shoc.transport_mode = tendencies
```

In this mode, SHOC computes tendencies and ERF applies them to momentum,
thermodynamic, moisture, and TKE fields.

The advanced mode is:

```text
erf.shoc.transport_mode = host_diffusion
```

In this mode, SHOC exports eddy diffusivities to ERF's host diffusion path. SHOC
does not add overlapping explicit tendencies.

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
ERF_ShocImplicit.cpp     Implicit tendency-mode updates
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
