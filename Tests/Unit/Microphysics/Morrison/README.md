# Morrison microphysics contract tests

This directory documents the Morrison refactor-for-testing surface used by the
unit tests. The production changes are intentionally narrow: extracted helpers
mirror scalar formulas that already existed in the C++ Morrison path, and public
tests now cover copy-in/copy-out mapping plus selected C++ `Advance` contracts
without changing legacy microphysics numerics.

## Implementation inventory

| Area | Production location | Verified role | Current contract coverage |
| --- | --- | --- | --- |
| Public class and state ownership | `Source/Microphysics/Morrison/ERF_Morrison.H` | Defines `MicVar_Morr`, surface accumulators, qstate sizes, `Init`, `Advance`, and copy wrappers. | Public copy, public warm advance, rain-only sedimentation, and MPI copy tests cover the class entry points. |
| Initialization | `Source/Microphysics/Morrison/ERF_InitMorrison.cpp` | Allocates all `mic_fab_vars`; maps restart accumulators to `rain_accum`, `snow_accum`, `graup_accum`; initializes Fortran Morrison only when requested. | Public tests exercise allocation for copy and C++ advance with `use_morr_cpp_answer=true`. |
| State copy-in | `Source/Microphysics/Morrison/ERF_InitMorrison.cpp` | Converts ERF conserved `rho*q` to Morrison mixing ratios and number concentrations; clamps moist inputs to nonnegative; computes pressure in Pa. | `MorrisonPublic.CopyStateRoundTripPreservesMappedStateAndNonnegativity`. |
| State copy-out | `Source/Microphysics/Morrison/ERF_UpdateMorrison.cpp` | Writes Morrison `qv,qc,qi,qr,qs,qg,nc,ni,nr,ns,ng` back to conserved state with nonnegative clamps; preserves `rho` and writes `rho*theta`. | Same public round-trip test. |
| C++ advance control | `Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp` | Selects C++ vs Fortran path via `erf.use_morr_cpp_answer`; computes Exner and local temperature; owns scalar source, PSD, sedimentation, and surface accumulation formulas. | Public warm no-ice `Advance` checks finite nonnegative copy-out and active surface precipitation; rain-only `Advance` checks column loss against surface accumulation. |
| Cleanup and warm tiny-ice melt | `Source/Microphysics/Morrison/ERF_MorrisonUtils.H` called from `ERF_AdvanceMorrison.cpp` | qsmall mass-number cleanup, subsaturated tiny-hydrometeor cleanup, warm small snow/graupel melt. | Scalar, property, and host/device tests. |
| Warm-rain source family | `ERF_MorrisonUtils.H` called from warm and cold branches in `ERF_AdvanceMorrison.cpp` | Khairoutdinov-Kogan autoconversion/accretion plus warm QC sink limiter. | Formula, threshold, number-limit, and local liquid conservation tests. |
| Exponential PSD bounds | `ERF_MorrisonUtils.H` called from `ERF_AdvanceMorrison.cpp` | Rain/snow/graupel exponential lambda bounds; clamped lambda readjusts number to preserve mass. | Scalar, property, host/device, and MPI helper tests. |
| Sedimentation budget | `ERF_MorrisonUtils.H`; public loop in `ERF_AdvanceMorrison.cpp` | Flux-divergence helper for column budget reasoning; public loop computes per-column split substeps from local max fall speed. | Exact helper telescoping/property tests, MPI helper reduction, and public rain-only column loss tracking. |
| Surface precipitation increment | `ERF_MorrisonUtils.H` called from `ERF_AdvanceMorrison.cpp` | Bottom fallout partition into total precip, snow, snow-plus-ice, and graupel accumulation increments. | Exact surface partition property test, MPI helper reduction, and public surface-accumulator smoke/budget tests. |
| Optional Fortran path | `ERF_Morrison_Advance_F.H`, `ERF_module_mp_morr_two_moment.F90` | Legacy/optional Morrison implementation and provenance for many constants. | Not refactored here; used as provenance for thresholds and formulas. |

## Extracted seams

Production helpers live in `Source/Microphysics/Morrison/ERF_MorrisonUtils.H`.
All helpers are `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE`, allocate no memory,
use no virtual dispatch, and avoid host-only utilities in device-reachable code.

- `MorrisonCellState`: scalar thermodynamic, mass, and number state for local helper contracts.
- `MorrisonEffectiveRadii`: effective-radius values touched by qsmall cleanup.
- `MorrisonQSmallCleanupDiagnostics`: branch diagnostics for mass-number cleanup.
- `MorrisonDistributionParameters`: bounded exponential PSD result and clamp diagnostics.
- `MorrisonAutoconversionRates`: `prc`, `nprc`, `nprc1`, and number-limit diagnostics.
- `MorrisonAccretionRates`: `pra`, `npra`, and active-branch diagnostic.
- `MorrisonCloudWaterLimiterDiagnostics`: warm QC limiter ratio and active flag.
- `MorrisonSedimentationBudget`: flux divergence, mixing-ratio tendency, and density-content delta.
- `MorrisonSurfacePrecipitationIncrement`: bottom-flux partition for surface accumulators.

## Scalar contracts

Scalar tests cover threshold equality, formula values, and branch diagnostics for
qsmall cleanup, subsaturated tiny cleanup, warm small snow/graupel melt,
warm-rain autoconversion, cloud-rain accretion, warm QC limiting, and
rain/snow/graupel exponential PSD bounds.

Physical-property tests cover full vs no-ice local total water definitions,
mass-number consistency, nonnegativity, PSD mass reconstruction under slope
limits, warm-rain local liquid conservation, exact sedimentation helper column
telescoping, and exact surface precipitation partitioning.

Kernel tests launch the helper surface through `amrex::ParallelFor` and compare
device results with host reference results for every device-callable helper in
`ERF_MorrisonUtils.H`: total-water helpers, cleanup/melt helpers, warm-rain
sources and limiter, sedimentation budget, surface increment, and PSD bounds.

## Public Advance contracts

Public tests exercise the real Morrison `Init -> Copy_State_to_Micro ->
Copy_Micro_to_State` MultiFab path. They verify the ERF state mapping, the
nonnegative copy-in/copy-out policy for all six mass species and five number
concentrations, and preservation of `rho` and `rho*theta` across the copy path.
The copy tolerances are scaled by each initialized conserved component; mass
species are not compared with number-concentration scales.

`MorrisonPublic.WarmNoIceAdvanceProducesFiniteStateAndSurfacePrecipitation`
calls the actual public C++ `Advance` path on a warm no-ice state. It checks
finite nonnegative conserved output, active surface precipitation accumulation,
and cloud-water depletion through the production warm-rain path.

## Sedimentation and surface accumulation contracts

`MorrisonPhysicalProperties.SedimentationBudgetTelescopesAcrossColumn` protects
the exact extracted flux-divergence convention. The density-content delta is a
per-volume update; column-integrated checks multiply it by `dz`.

`MorrisonPublic.RainOnlyAdvanceTracksColumnWaterLossWithSurfacePrecipitation`
drives rain sedimentation through the public C++ `Advance` path. It verifies
that rain leaves the column, the total-precipitation surface accumulator grows,
and public-path column water loss tracks that accumulator within the current
production-path tolerance. The exact algebra remains covered by the helper test;
the public path is intentionally not used to silently tighten legacy Morrison
sedimentation numerics.

In `ERF_AdvanceMorrison.cpp`, sedimentation substeps are column-local inside the
`ParallelFor(boxD)` loop: `nstep` is reset for each `(i,j)` column and then takes
the max of `rgvm * dt / dzq + 1` over vertical levels in that column. There is no
MPI-wide max fall-speed reduction in this path.

## MPI/decomposition contracts

MPI tests partition deterministic helper cases over ranks and reduce checked
case counts and residuals.

`MorrisonParallel.PublicCopyRoundTripIsDecompositionInvariant` uses a split
`BoxArray` and the real Morrison public copy path, then reduces component errors
globally. This is the public-path decomposition contract in this slice.

CTest now defaults parallel GoogleTest registration to `1;2;4` ranks for new
unit-test build trees. Existing build trees with a cached
`ERF_PARALLEL_TEST_NRANKS` should be reconfigured explicitly, for example:

```text
cmake -S . -B build-gtest-unitonly-ci -DERF_PARALLEL_TEST_NRANKS="1;2;4"
```

## Derivation notes

Warm tiny-ice melt:

```text
qs' = 0
qg' = 0
qr' = qr + qs + qg
nr' = nr + ns + ng
ns' = 0
ng' = 0
T'  = T - (qs + qg) Lf / cpm
```

Subsaturated tiny cleanup:

```text
qv' = qv + qc_removed + qr_removed + qi_removed + qs_removed + qg_removed
T'  = T - (qc_removed + qr_removed) Lv / cpm
        - (qi_removed + qs_removed + qg_removed) Ls / cpm
```

Warm-rain autoconversion and accretion:

```text
prc  = 1350 qc^2.47 (nc / 1e6 * rho)^(-1.79)
nprc = min(prc / (qc / nc), nc / dt)
nprc1 = min(prc / cons29, nprc)

pra  = 67 (qc qr)^1.15
npra = pra / (qc / nc)

qc_tendency = -prc - pra
qr_tendency =  prc + pra
qc_tendency + qr_tendency = 0
```

Warm QC limiter:

```text
requested = (prc + pra) dt
if requested > qc and qc >= qsmall:
    ratio = qc / requested
    prc' = ratio prc
    pra' = ratio pra
```

Rain/snow/graupel exponential PSD relationship:

```text
lambda = (coefficient * number / mass)^(1/3)
n0 = number * lambda
```

When `lambda` is clamped, Morrison preserves mass by changing number:

```text
n0' = lambda_bounded^4 * mass / coefficient
number' = n0' / lambda_bounded = lambda_bounded^3 * mass / coefficient
mass_reconstructed = number' * coefficient / lambda_bounded^3 = mass
```

Sedimentation helper convention:

```text
flux_divergence = (fallout_from_above - fallout_to_below) / dz
mixing_ratio_tendency = flux_divergence / (nstep rho)
density_content_delta = flux_divergence dt / nstep
```

Column-integrated telescoping over layers gives:

```text
sum_k density_content_delta_k dz_k = (top_inflow - bottom_outflow) dt / nstep
```

Morrison's tested top inflow is zero, so the column loss is the bottom fallout
used by the surface increment helper.

Surface precipitation increment:

```text
precrt_increment  = (faloutr + faloutc + falouts + falouti + faloutg) dt / nstep
snowrt_increment  = (falouts + falouti + faloutg) dt / nstep
snowprt_increment = (falouti + falouts) dt / nstep
grplprt_increment = faloutg dt / nstep
```

Public copy mapping:

```text
copy-in:  q = max(0, rhoq / rho)
copy-out: rhoq = rho * max(0, q)
```

This applies to `qv,qc,qi,qr,qs,qg,nc,ni,nr,ns,ng`.

## Tolerance strategy

- `formula_abs_tol`: machine-epsilon-scaled checks for branch and algebraic identities.
- `backend_math_abs_tol`: wider tolerance for `pow` and host/device backend math variation.
- `exact_zero_tol`: machine-epsilon-scaled tolerance for exact-zero limiter outputs.
- `property_tol`: accumulation tolerance for multi-term conservation checks.
- `latent_proxy_tol`: scales with both actual latent-energy terms and the reconstructed temperature-difference energy scale. This avoids hiding behind temperature scale alone while still accounting for subtracting two O(280 K) temperatures to recover a tiny `dT`.
- Public copy tolerances use the initialized conserved component scale for each mapped quantity, so small mass species stay on mass scales while number concentrations use number scales.
- Public rain-only sedimentation uses a 5% production-path surface-budget tolerance because the current monolithic C++ path does not match the extracted telescoping helper to roundoff. This documents current behavior without changing Morrison numerics in a test-focused PR.

## Remaining limitations

- The complete Morrison source matrix is not decomposed into helper families yet. Cold-phase deposition/sublimation, freezing, riming, aggregation, melting, and their number tendencies remain inside the large C++ kernel.
- The full mass and number limiter allocation policy after all warm/cold source terms is not fully extracted.
- The public rain-only sedimentation test tracks column loss and surface accumulation through `Morrison::Advance`, but exact conservation remains helper-level because the current monolithic production loop shows an O(5%) public-path residual for this fixture.
- MPI public-path coverage currently exercises copy decomposition. A future public decomposition test should run `Morrison::Advance` on the same physical state at multiple rank counts and compare conserved state plus accumulators.
- GPU-specific validation was not run in this macOS CPU/MPI build.

## Reviewer checklist

- [x] Refactor is limited to testability seams.
- [x] Existing production formulas and threshold comparisons are preserved in extracted helpers.
- [x] Device-reachable helpers are GPU-safe and allocation-free.
- [x] No broad style cleanup is mixed into the Morrison refactor.
- [x] Scalar state and diagnostics are minimal and physically named.
- [x] Duplicated constants in test code include provenance comments.
- [x] Threshold tests cover below/equal/above behavior where applicable.
- [x] Local total-water definitions distinguish full and no-ice Morrison helper contracts.
- [x] Local water conservation is tested for extracted cleanup, melt, warm-rain, and sedimentation contracts.
- [x] Number concentration contracts are tested for qsmall cleanup, warm-rain autoconversion limits, accretion, and PSD slope limiting.
- [x] Tests do not assert number conservation for PSD slope limiting.
- [x] Latent-heating signs and residuals are tested for extracted cleanup/melt branches.
- [x] Device helper parity is tested through AMReX kernels for all extracted device-callable helpers.
- [x] MPI helper contracts are tested at 1, 2, and 4 ranks.
- [x] Public copy-in/copy-out mapping is tested through the real Morrison class.
- [x] Public warm no-ice `Advance` is tested through the real Morrison C++ path.
- [x] Public rain-only sedimentation and total-precipitation accumulation are tested through the real Morrison C++ path.
- [x] Public copy-path decomposition is tested through the real Morrison class on a split `BoxArray`.
- [ ] Full public `Advance` MPI decomposition remains future work.