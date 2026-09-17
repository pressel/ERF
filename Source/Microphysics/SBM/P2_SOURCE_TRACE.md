# ERF SBM P2 source trace

This document is the implementation trace for the final P2 correction pass.
The historical P0/P1 trace remains in `P0_P1_SOURCE_TRACE.md`. The symbols
and paths below are the source of truth for the qualified P2 envelope.

## G0 — locked contracts and source locations

| Contract | Production source | Implemented meaning |
|---|---|---|
| Authoritative state | `Source/AuxiliaryState/ERF_AuxiliaryStateManager.{H,cpp}` (`old`, `evaluation`, `output`, `scratch`) and `Source/Microphysics/SBM/ERF_SBMErfIntegration.cpp` (`sbm_auxiliary`) | The spectral state is provider-owned and authoritative. ERF compact `qc/qr` are projections, not an independent transport state. |
| Physical carrier flux | `ERF::advance_sbm_stage` in `Source/Microphysics/SBM/ERF_SBMErfIntegration.cpp`; ERF face fields in `Source/ERF.H` | DonorCell and GroupedFCT_WENOZ3 use the existing ERF face-centered dry-air mass flux. They do not reconstruct independent velocity fluxes. |
| Physical storage | `SBMLayout` / `SBMBulkProjection` in `Source/Microphysics/SBM/` | One-moment storage is `M`; two-moment storage is physical `(M,C)`. Endpoint variables are bounded transport scratch only. |
| Complete constraints | `ERF_SBMConstraintGroups.{H,cpp}` and `ERF_SBMTransportPrototype.cpp` | Every population/bin/property group is limited with one common face limiter and complete linear-form constraints. |
| Accepted transfer | `AuxiliaryFaceTransferLedger::record_stage` in `Source/AuxiliaryState/ERF_AuxiliaryFaceTransfer.{H,cpp}` | The accepted physical face transfer is the single quantity used for divergence, projection, diagnostics, boundary budgets, and AMR register accounting. |
| Diffusion | `ERF_SBMDiffusion.{H,cpp}` and `ERF_SBMTransportPrototype.cpp` | Explicit scalar density-weighted diffusion acts on `X/rho` with the declared SBM coefficient and is included once in the low-order flux. |
| Schema identity | `ERF_SBMRestart.{H,cpp}` | Restart compares exact layout, projection, transport, AMR-transfer, numerical, boundary, and grid identities before restoring state. |

## G1 — production WENO-Z3 face convention

The production helper is `sbm_weno_z3_face` in
`Source/Microphysics/SBM/ERF_SBMTransportPrototype.cpp` (near line 166),
called by `ERF_SBMTransportPrototype::advance_stage` (near line 990). It uses
the ERF local finite-volume WENO-Z3 candidate and canonical fixed-precision
epsilon and weight algebra on the prepared two-ghost intensive `X/rho`
stencil. It has no beta-dependent epsilon floor, `abs(q)`/amplitude branch,
or offset-dependent scale.

The face index is the finite-volume face between cells `i-1` and `i` (and the
analogous index in `y`/`z`). A positive carrier uses the left/upwind donor side
and a negative carrier uses the right/upwind donor side. The independent tests
`WENOZ3TranslationCovarianceForSmoothAndDiscontinuousStencils`,
`WENOZ3CanonicalERFEquivalenceRandomized`,
`WENOZ3CanonicalERFDiscontinuityAndConstantStencils`,
`CoarseFineWENOInterfaceOracleUsesBothUpwindSigns`, and
`FiniteVolumeWENOQuadraticOptimalCandidateIsExact` exercise additive offsets,
randomized algebraic equivalence, discontinuous `[0,1,1,1]` and mirror
stencils, both signs, finite constant data, and exact cell-average quadratic
data. The production/helper equivalence tests compare the adapter against a
separately written canonical scalar reference. Retained smooth constant-density
convergence data is an approximately third-order spatial operator result; it
is not a claim about nonlinear ERF time-integration order.

The variable-density convergence fixture uses positive nonconstant density
and a smooth mixing ratio and records the bounded approximately second-order
behavior required for the simple `U/rho` stencil.

## G2 — temporal source views and target density

`ERF::advance_sbm_stage` in
`Source/Microphysics/SBM/ERF_SBMErfIntegration.cpp` explicitly prepares every
view before donor, WENO, support, or diffusion reads:

* `AuxiliaryStateManager::fill_stage_from_coarse(..., Old, old_step_time)`
  fills the fine old view for all stages that read the full-step old source.
* For later stages,
  `AuxiliaryStateManager::fill_stage_from_coarse(..., Evaluation,
  old_stage_time)` fills the fine evaluation view at the actual predictor
  evaluation time. Stage 0 does not substitute the evaluation time for the
  old view.
* The same preparation is used by both production transport methods. The
  source is selected from the prepared old or evaluation view according to
  the temporal contract.
* The transport boundary receives explicit `rho_anchor`, `rho_input`, and
  `rho_target` views. `ProductionTargetDensityPreservesConstantRatioForBothTemporalContracts`
  checks all three compressible stages and both anelastic Heun stages with
  variable density and no order-one tolerance floor.

The compressible ledger is `dt*F_stage2_accepted`; the anelastic ledger is
`0.5*dt*(F_stage0_accepted+F_stage1_accepted)`. Source selection and ledger
weights are implemented in `make_compressible_stage`, `make_anelastic_stage`,
and `StageContext::accepted_ledger_weight`.

## G3 — carrier-weighted AMR transfer

`AuxiliaryStateManager::carrier_weighted_fill` in
`Source/AuxiliaryState/ERF_AuxiliaryStateManager.cpp` implements the generic,
provider-neutral transfer. For every density-weighted component it forms

```text
z_c(t) = U_c(t) / rho_c(t)
z_f    = supported piecewise-constant interpolation/fill of z_c
U_f    = z_f * rho_f(target time)
```

The coarse `U` and coarse host density are time-interpolated to the same time
before division. The fine host density is passed as an explicit target view;
it is never reconstructed by the SBM manager. A finite, strictly positive
density is required wherever a supported value is formed, and the same
multiplier is applied to every component in an atomic group. Tiny exact zeros
remain exact zeros.

The carrier-weighted stage transaction is ordered explicitly: preserve fine
valid cells, write coarse-derived values into the grown ghost region, then
perform the final fine-level `FillBoundary(fine_geometry.periodicity())`.
That final operation restores same-level and periodic fine authority; a true
uncovered coarse/fine ghost has no same-level owner and therefore retains the
coarse-derived value.

Callsites are `ERF::MakeNewLevelFromCoarse` in
`Source/ERF_MakeNewLevel.cpp` (near line 516), after host `FillCoarsePatch`
has built the target fine density; `ERF::RemakeLevel` in the same file (near
line 1109), after `synchronize_sbm_level_companions`, with current coarse
old/output and fine target density views; and
`ERF::advance_sbm_stage`, for old and evaluation coarse/fine stage fills.
Restart restores the authoritative per-level spectrum and reprojects compact
state; it does not convert or reconstruct the spectrum from bulk.

`AuxiliaryStateManager::prolong_from_coarse`,
`fill_stage_from_coarse`, `remake_level_from_coarse`, and `average_down_to`
cover creation, stage FillPatch, remake/regrid, and restriction. The manager
remains generic: cloud/bin meanings exist only in the SBM provider and
projection services. Host density sources are the actual ERF
`vars_old[lev][Vars::cons]` `Rho_comp` and `vars_new[lev][Vars::cons]`
`Rho_comp` aliases constructed at the callsites; stage target density is the
current host state passed through the production adapter.

The manager unit coverage verifies the carrier identity and
`CarrierWeightedStageFillPreservesFineFABAuthority` independently checks two
fine FABs, nonuniform target density, same-level overlap, and a true
coarse/fine ghost. `AttachedPropertySupportUsesFineDonorAcrossInternalFABBoundary`
then forces transport across the internal fine-FAB boundary and checks the
analytic donor envelope `[3,7]` and fine donor ratio 7, rather than deriving
the expected value through a production support helper. The production
variable-density periodic two-level fixture continues to verify creation,
stage transport, and dynamic regrid/remake.

## G4 — compact ghost projection

After every supported spectral prolongation/fill, the production callsite
projects the authoritative spectrum into core `qc/qr` over the common
valid-plus-available-grow region. The helper is
`SBMBulkProjection::apply_to_core`; callsites are in `ERF_MakeNewLevel.cpp`
after creation/remake and in `ERF_SBMErfIntegration.cpp` after stage fill,
reflux, and matched auxiliary average-down. The operation is ordered after
the spectral transfer and never uses a host compact FillPatch value as a
source. The compact ghost unit test injects a discrepant stale compact value
and requires machine-scale agreement with the spectral ghost projection.

## G5 — DonorCell and grouped transport

`TransportMethod::DonorCell` and `TransportMethod::GroupedFCT_WENOZ3` enter
the same prepared-state, carrier-weighted density, accepted-ledger, compact
projection, flux-register, reflux, and fail-closed validation path. The
shared `prepare_transport_coordinates` helper evaluates the complete
`fabbox()` (not only `validbox()`), uses already prepared fine and true
coarse/fine source ghosts, and converts 1M `(M,C)` to `(L,H)` with
`std::fma`, a scale-aware cancellation tolerance, and no order-one floor.
Donor support is derived from the low-order source, actual upwind donors,
active diffusion neighbors, and the Heun old state; no universal carrier
floor is introduced. `DonorCellTwoMomentUsesPreparedCoarseFineEndpointDonors`
is an independent two-level oracle with both carrier signs, accepted-flux,
endpoint-realizability, compact-projection, and rank-equivalence checks. The
real fixtures `SBM_P2_AMR_DONOR_2M` and `SBM_P2_AMR_DONOR_SUBCYCLE_2M`
qualify DonorCell on one periodic refinement level at one and two MPI ranks.

## G6 — timestep and capability policy

`ERF::sbm_admissible_timestep` in
`Source/TimeIntegration/ERF_ComputeTimestep.cpp` is called from ERF's normal
`ComputeDt` path. For static Cartesian geometry it computes, with one fixed
MPI/global reduction, the actual bin-independent low-order demand

```text
tau/(rho_anchor V) * [sum_d max(sigma_d mdot_d, 0)
                      + sum_faces A_f rho_face K/d_if] <= 1,
mdot_d = u_face,d rho_face.
```

It returns the mathematical bound `1/(advective_rate+diffusive_rate)` to the
host admissibility helper, which retains the explicit `0.5` safety factor;
both values and the selected timestep are logged. The production stage check
remains an exact stage-local defensive guard, with no hidden subcycling or
velocity-only replacement. Invalid density/velocity/coefficient, terrain,
EB, or missing host ghosts fail closed. `SBM_P2_HOST_CFL_VARIABLE_RHO_1` and
`_2` exercise the variable-density counterexample, combined diffusion, and
1/2-rank reductions; their evidence also records that native stepping was
used and that the old velocity-only estimate would violate the mathematical
bound.

`CapabilityInput` parsing and `evaluate_capability` in
`ERF_SBMContracts.{H,cpp}` expose and validate `max_level`, every active
spatial ratio, the native time factor, `TwoWay`, periodicity, transport and
moment modes, explicit diffusion, chunk size, boundaries, precision, and
unsupported P3/terrain/EB/implicit/tensor policies. Defaults are
conservative: an unpopulated native-subcycling field cannot grant AMR
qualification. The negative matrix rejects max level 2, non-factor-2
spatial/time refinement, OneWay/nonperiodic AMR, prescribed inflow, implicit
or tensor diffusion, terrain/EB, and P3 physics.

## G7 — AMR synchronization, reflux, and qualified envelope

`sbm_flux_reg` is allocated only for active SBM two-way coupling. The
integration hook registers accepted per-area face fluxes with the
YAFluxRegister `dt/dx` convention exactly once. ERF's native
`erf.dt_ref_ratio=2` advances the fine level twice per coarse step. At
synchronization, ERF refluxes the authoritative spectrum, calls
`validate_sbm_post_reflux`, projects compact fields, then performs normal ERF
average-down and matched auxiliary average-down with a second validation.
`post_reflux_validation_count` and `post_reflux_material_rejection_count` are
emitted in the composite diagnostic; successful qualification fixtures report
zero material rejections. The negative unit path injects a materially
inadmissible correction and verifies fail-closed behavior before any clipping
or repair.

The qualified multilevel envelope is deliberately bounded:

```text
static Cartesian, double precision, fully periodic,
max_level=1, spatial refinement ratio=(2,2,2),
native time refinement factor=2, TwoWay coupling,
runtime 1M/2M bins, DonorCell or GroupedFCT_WENOZ3.
```

Single-level impermeable-wall and outward-only advective-outflow cases are
also qualified. Nonperiodic AMR, max level greater than one, other spatial or
time ratios, OneWay multilevel coupling, prescribed production spectral
inflow, terrain, EB, moving/dynamic geometry, native/implicit moisture
diffusion, tensor/cross diffusion, dynamic spectral grids, schema conversion,
and all P3 physics fail closed.

## G8 — strict restart identities

`ERF_SBMRestart.H` defines the current stable identities:

```text
constraint: complete-groups-donor-support-v2
transport: WENO_Z3-group-FCT-v4
AMR transfer: carrier-weighted-mixing-ratio-AMR-v1
boundary: periodic+wall+outflow-no-inflow-v1
schema: ERF-SBM-P2-4
```

The transport and AMR-transfer identities are part of the exact checkpoint
schema comparison. Old WENO v2/v3 and old direct-extensive AMR-transfer
payloads reject; mismatched grid, layout, or checkpointed compact projection
also reject. New-versus-restarted two-level trajectories are compared using
the authoritative `SBMAux_*`, compact cell fields, and schema payloads, with
repeated runs covering deterministic endpoint-ghost initialization.

## G9 — evidence locations

Focused unit and production tests are registered in `Tests/CTestList.cmake`.
The real P2 runners are `Tests/RunSBMP2AMR.cmake`,
`Tests/RunSBMP2AMRSubcycle.cmake`, `Tests/RunSBMP2Timestep.cmake`, and
`Tests/RunSBMP2VariableHostCFL.cmake`.
The focused tests include the fine-FAB authority, attached-property donor,
prepared 2M coarse/fine endpoint, canonical WENO randomized/reference, and
host-CFL counterexample checks. Temporary negative controls were run for the
F01 final-sync bypass, F02 validbox-only scratch, F03 velocity-only host
bound, and F04 adaptive WENO epsilon, and were restored before qualification.
Independent convergence and scratch-memory evidence is written under
`/private/tmp/erf_sbm_p2_*.csv`; final machine-generated qualification values
and exact command/rank matrix belong in
`Source/Microphysics/SBM/P2_QUALIFICATION_REPORT.md` after implementation
freeze. GPU status is reported separately because this environment has no
qualified GPU runtime.
