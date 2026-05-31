# Morrison microphysics contract tests

This directory documents the Morrison refactor-for-testing surface used by the
unit tests. The production changes are intentionally narrow: extracted helpers
mirror scalar formulas that already existed in the C++ Morrison path, and public
tests cover the copy-in/copy-out state mapping without changing legacy
microphysics numerics.

## Implementation inventory

| Area | Production location | Verified role | Current contract coverage |
| --- | --- | --- | --- |
| Public class and state ownership | `Source/Microphysics/Morrison/ERF_Morrison.H` | Defines `MicVar_Morr`, surface accumulators, qstate sizes, `Init`, `Advance`, and copy wrappers. | Public copy round-trip test covers mapped state and nonnegative copy-out. |
| Initialization | `Source/Microphysics/Morrison/ERF_InitMorrison.cpp` | Allocates all `mic_fab_vars`; maps restart accumulators to `rain_accum`, `snow_accum`, `graup_accum`; initializes Fortran Morrison only when requested. | Public test exercises allocation and copy path with `use_morr_cpp_answer=true`. |
| State copy-in | `Source/Microphysics/Morrison/ERF_InitMorrison.cpp` | Converts ERF conserved `rho*q` to Morrison mixing ratios and number concentrations; clamps moist inputs to nonnegative; computes pressure in Pa. | `MorrisonPublic.CopyStateRoundTripPreservesMappedStateAndNonnegativity`. |
| State copy-out | `Source/Microphysics/Morrison/ERF_UpdateMorrison.cpp` | Writes Morrison `qv,qc,qi,qr,qs,qg,nc,ni,nr,ns,ng` back to conserved state with nonnegative clamps; preserves `rho` and writes `rho*theta`. | Same public round-trip test. |
| C++ advance control | `Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp` | Selects C++ vs Fortran path via `erf.use_morr_cpp_answer`; computes Exner and local temperature; owns scalar source, PSD, sedimentation, and surface accumulation formulas. | Helper tests cover extracted scalar contracts; full public `Advance` conservation/decomposition remains a larger seam. |
| Cleanup and warm tiny-ice melt | `Source/Microphysics/Morrison/ERF_MorrisonUtils.H` called from `ERF_AdvanceMorrison.cpp` | qsmall mass-number cleanup, subsaturated tiny-hydrometeor cleanup, warm small snow/graupel melt. | Scalar, property, and host/device tests. |
| Warm-rain source family | `ERF_MorrisonUtils.H` called from warm and cold branches in `ERF_AdvanceMorrison.cpp` | Khairoutdinov-Kogan autoconversion/accretion plus warm QC sink limiter. | Formula, threshold, number-limit, and local liquid conservation tests. |
| Exponential PSD bounds | `ERF_MorrisonUtils.H` called from `ERF_AdvanceMorrison.cpp` | Rain/snow/graupel exponential lambda bounds; clamped lambda readjusts number to preserve mass. | Scalar, property, host/device, and MPI helper tests. |
| Sedimentation budget | `ERF_MorrisonUtils.H` | Flux-divergence helper for column budget reasoning. | Column telescoping property test and MPI helper reduction test. |
| Surface precipitation increment | `ERF_MorrisonUtils.H` called from `ERF_AdvanceMorrison.cpp` | Bottom fallout partition into total precip, snow, snow-plus-ice, and graupel accumulation increments. | Surface partition property test and MPI helper reduction test. |
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

## Contracts tested

Scalar tests cover threshold equality, formula values, and branch diagnostics for
qsmall cleanup, subsaturated tiny cleanup, warm small snow/graupel melt,
warm-rain autoconversion, cloud-rain accretion, warm QC limiting, and
rain/snow/graupel exponential PSD bounds.

Physical-property tests cover full vs no-ice local total water definitions,
mass-number consistency, nonnegativity, PSD mass reconstruction under slope
limits, warm-rain local liquid conservation, sedimentation column telescoping,
and surface precipitation partitioning.

Kernel tests launch the helper surface through `amrex::ParallelFor` and compare
device results with host reference results for the extracted cleanup, melt, and
PSD branches.

Public tests exercise the real Morrison `Init -> Copy_State_to_Micro ->
Copy_Micro_to_State` MultiFab path. They verify the ERF state mapping, the
nonnegative copy-in/copy-out policy for all six mass species and five number
concentrations, and preservation of `rho` and `rho*theta` across the copy path.

MPI tests partition deterministic helper cases over ranks and reduce checked
case counts and residuals. They run cleanly at 1, 2, and 4 ranks, but they are
helper-level MPI tests; they do not claim full public `Advance` decomposition
invariance.

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

## Tolerances

- `formula_abs_tol`: machine-epsilon-scaled checks for branch and algebraic identities.
- `backend_math_abs_tol`: wider tolerance for `pow` and host/device backend math variation.
- `exact_zero_tol`: machine-epsilon-scaled tolerance for exact-zero limiter outputs.
- `property_tol`: accumulation tolerance for multi-term conservation checks.
- `latent_proxy_tol`: scales with both actual latent-energy terms and the reconstructed temperature-difference energy scale. This avoids hiding behind temperature scale alone while still accounting for subtracting two O(280 K) temperatures to recover a tiny `dT`.

## Remaining limitations

- The complete Morrison source matrix is not decomposed into helper families yet. Cold-phase deposition/sublimation, freezing, riming, aggregation, melting, and their number tendencies remain inside the large C++ kernel.
- The full mass and number limiter allocation policy after all warm/cold source terms is not fully extracted.
- The sedimentation helper tests cover flux-divergence algebra and bottom-flux accumulation partitioning, but the full public sedimentation loop is not yet driven end to end in a MultiFab test.
- The MPI tests are helper-level reductions, not public `Advance` decomposition tests. A future public decomposition test should run `Morrison::Advance` on the same physical state at multiple rank counts and compare conserved state plus accumulators.
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
- [x] Device helper parity is tested through AMReX kernels.
- [x] MPI helper contracts are tested at 1, 2, and 4 ranks.
- [x] Public copy-in/copy-out mapping is tested through the real Morrison class.
- [ ] Full public `Advance` conservation and MPI decomposition tests remain future work.