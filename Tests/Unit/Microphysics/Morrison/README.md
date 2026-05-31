# Morrison microphysics contract tests

This directory records the first contract-first Morrison testability slice. The
production refactor is deliberately narrow: it extracts local scalar helpers from
device-reachable Morrison C++ blocks without changing the legacy formulas.

## Inventory Summary

Verified facts from the current ERF implementation:

- Main public class: `Source/Microphysics/Morrison/ERF_Morrison.H`.
- Main C++ advance path: `Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp`.
- Optional Fortran path: `Source/Microphysics/Morrison/ERF_Morrison_Advance_F.H` and `ERF_module_mp_morr_two_moment.F90`.
- ERF prognostic hydrometeor mass variables are `qv`, `qcl`, `qci`, `qpr`, `qps`, and `qpg`.
- ERF number variables are `nc`, `ni`, `nr`, `ns`, and `ng`.
- Surface accumulation variables are `rain_accum`, `snow_accum`, and `graup_accum`.
- Full Morrison total water for local helper tests is `qv + qc + qi + qr + qs + qg`.
- `Morrison_NoIce` local total water starts from `qv + qc + qr`; public-path no-ice behavior still needs a dedicated test seam.
- C++ Morrison uses pressure in Pa in the translated Morrison block.
- The C++ path computes temperature from `theta * pii` and applies temperature changes directly in local Morrison variables before writing back through `theta`.
- Device-reachable code is implemented inside AMReX `ParallelFor` lambdas.

Verified local branch and limiter facts covered by the current tests:

- qsmall mass-number cleanup uses strict `< qsmall` with `qsmall = 1.0e-14`.
- qsmall cleanup zeros the matching number concentration and effective radius for cloud water, rain, cloud ice, snow, and graupel.
- subsaturated tiny-hydrometeor cleanup uses strict saturation-ratio threshold `< 0.9` and mass threshold `< 1.0e-8`.
- warm-branch tiny snow/graupel melt uses strict mass threshold `< 1.0e-6`.
- warm tiny snow/graupel melt transfers mass to rain, transfers `ns`/`ng` to `nr`, zeros donor number, and applies fusion cooling.
- rain, snow, and graupel exponential PSD lambda bounds adjust number concentration to preserve the mass-lambda relationship; they do not conserve number.

Hypotheses and open inventory items:

- The full source-term limiter policy after autoconversion, accretion, riming, deposition, sublimation, melting, and freezing still needs extraction or diagnostics.
- Morrison sedimentation appears to include mass and number fall-speed state, but the current PR slice does not yet expose a scalar sedimentation flux seam.
- Surface accumulation must be checked against the same limited flux used by column updates; this is not yet tested for Morrison.
- MPI rank-count invariance for the public Morrison path still needs a MultiFab-level test once sedimentation/global controls are inventoried.

## Extracted Seams

Production helpers live in `Source/Microphysics/Morrison/ERF_MorrisonUtils.H`.

- `MorrisonCellState`: minimal scalar mass, number, thermodynamic, pressure, and density state used by extracted helpers.
- `MorrisonConfig`: POD placeholder for Morrison scalar mode/config tests; only the fields needed by current helper contracts are included.
- `morrison_total_water_full` and `morrison_total_water_no_ice`: local total-water definitions.
- `morrison_apply_subsaturation_small_hydrometeor_cleanup`: exposes the tiny hydrometeor evaporation/sublimation cleanup branch.
- `morrison_apply_qsmall_mass_number_cleanup`: exposes qsmall mass-number cleanup and diagnostics.
- `morrison_apply_warm_small_ice_melt_to_rain`: exposes warm tiny snow/graupel melt semantics.
- `morrison_exponential_distribution_parameters`: exposes rain/snow/graupel exponential PSD lambda bounds and number readjustment.

All helpers are `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE`, allocate no memory,
use no virtual dispatch, and are callable from AMReX kernels.

## Contracts Tested

Implemented scalar tests:

- qsmall below/equal/above threshold behavior.
- subsaturation cleanup water conservation and latent cooling sign.
- subsaturation threshold equality no-op behavior.
- warm tiny snow/graupel melt mass transfer, number transfer, and fusion cooling.
- rain/snow/graupel PSD in-range, lower-clamp, and upper-clamp behavior.

Implemented physical-property tests:

- full and no-ice total-water definitions.
- qsmall mass-number consistency policy.
- mass and number nonnegativity for extracted helpers.
- PSD mass-number relationship under slope limiting.

Implemented host/device tests:

- device helper sweep compared against the same host helper surface for qsmall cleanup, threshold equality, subsaturation cleanup, warm melt, and PSD branches.

Implemented MPI tests:

- distributed PSD helper branch sweep with MPI reductions of checked-case count and maximum normalized error.

Known gaps:

- Public Morrison local source update conservation tests are not implemented yet.
- Broad latent-heating phase-change matrix is not implemented yet.
- Full mass/number limiter allocation policy is not implemented yet.
- Sedimentation and surface accumulation budgets are not implemented yet.
- 1/2/4-rank public-path decomposition invariance is not implemented yet; the build tree controls registered rank counts through `ERF_PARALLEL_TEST_NRANKS`.

## Derivation Notes

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

Total-water residual:

```text
(qv + qc + qi + qr' + qs' + qg') - (qv + qc + qi + qr + qs + qg) = 0
```

Latent proxy residual for melting cooling:

```text
cpm (T' - T) + (qs + qg) Lf = 0
```

Subsaturated tiny cleanup for liquid and ice:

```text
qv' = qv + qc_removed + qr_removed + qi_removed + qs_removed + qg_removed
qhydro_removed' = 0
T' = T - (qc_removed + qr_removed) Lv / cpm
       - (qi_removed + qs_removed + qg_removed) Ls / cpm
```

Total-water residual is zero. Latent proxy residual:

```text
cpm (T' - T) + liquid_removed Lv + ice_removed Ls = 0
```

Exponential PSD relationship for rain/snow/graupel:

```text
lambda = (coefficient * number / mass)^(1 / shape_exponent)
n0 = number * lambda
```

When `lambda` is clamped to `lambda_bounded`, Morrison preserves mass by
readjusting number:

```text
n0' = lambda_bounded^4 * mass / coefficient
number' = n0' / lambda_bounded = lambda_bounded^3 * mass / coefficient
mass_reconstructed = number' * coefficient / lambda_bounded^3 = mass
```

## Tolerances

- `formula_abs_tol`: machine-epsilon-scaled checks for simple branch and algebraic identities.
- `backend_math_abs_tol`: wider tolerance for `pow` and host/device backend math variation.
- `exact_zero_tol`: machine-epsilon-scaled tolerance for exact-zero limiter outputs.
- `property_tol`: accumulation tolerance for multi-term conservation checks.
- `latent_proxy_tol`: scales with `cpm * temperature` because tests reconstruct small `dT` values by subtracting two O(280 K) temperatures.

Tolerances are tied to arithmetic scale and backend behavior, not chosen from a
failing residual.

## Reviewer Checklist

- [ ] Refactor is limited to testability seams.
- [ ] Production numerics are preserved unless a contract test exposed a bug.
- [ ] Device-reachable helpers are GPU-safe.
- [ ] No runtime allocation was added in hot paths.
- [ ] No broad style cleanup was mixed into the PR.
- [ ] Scalar cell-state/config structures are minimal.
- [ ] Diagnostics expose physical source families, not arbitrary temporaries.
- [ ] Total water conservation is defined and tested for extracted local helpers.
- [ ] `Morrison_NoIce` total-water definition is documented.
- [ ] Latent-heating signs for extracted cleanup/melt branches are tested.
- [ ] Number concentration contracts are defined for extracted mass-number cleanup and PSD limiting.
- [ ] Tests do not assert number conservation for PSD slope limiting.
- [ ] Major extracted branch thresholds are tested below, at, and above equality.
- [ ] Duplicated thresholds are marked `MUST MATCH` in production helpers.
- [ ] Device-callable helpers have host/device equivalence tests.
- [ ] MPI helper contracts are reduced across ranks.
- [ ] Known gaps for public source, sedimentation, surface accumulation, and decomposition tests are documented.