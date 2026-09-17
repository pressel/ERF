# ERF SBM P2 closeout source trace

This is the source-level trace for the closeout branch
`sbm-p2-final-closeout`, based on
`13ff28958a2286b9969d77dd4d6396a9d6475e4c` and normally merged with
`origin/development` at `805bdd156cd1213200d839c3945a023c3d9bd7e6`.
The design authority is
`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`.
This trace distinguishes the host estimate from the actual SBM carrier; it
does not claim the missing host-to-actual proof is closed.

## G0 — locked architecture and authoritative state

| Contract | Production source | Meaning |
|---|---|---|
| Authoritative spectral state | `Source/AuxiliaryState/ERF_AuxiliaryStateManager.{H,cpp}` and `Source/Microphysics/SBM/ERF_SBMErfIntegration.cpp` | Provider-owned `SBMAux` is authoritative; compact `qc/qr` are projections. |
| Actual carrier | `ERF::advance_sbm_stage` and ERF face fields | SBM receives ERF's `avg_xmom`, `avg_ymom`, and `avg_zmom`; they are not reconstructed velocity fields. |
| Physical storage | `ERF_SBMLayout`, `ERF_SBMBulkProjection` | One-moment is `M`; two-moment is physical `(M,C)`. |
| Complete constraints | `ERF_SBMConstraintGroups.{H,cpp}` and transport prototype | Every population/bin/property group is limited by complete linear-form constraints and one face limiter. |
| Accepted transfer | `ERF_AuxiliaryFaceTransfer.{H,cpp}` | The accepted physical face transfer drives divergence, projection, boundary budgets, and AMR accounting. |
| Diffusion | `ERF_SBMDiffusion.{H,cpp}` and transport prototype | Explicit scalar density-weighted diffusion on `X/rho`, included once in the low-order flux. |
| Restart identity | `ERF_SBMRestart.{H,cpp}` | Layout, projection, transport, AMR, boundary, grid, and schema identities are compared before restore. |

P3 condensation, activation, regeneration, collision/coalescence,
sedimentation, aerosol lifecycle, ice, and related warm-cloud physics remain
outside this branch.

## G1 — WENO-Z3 and donor conventions

`sbm_weno_z3_face` in
`Source/Microphysics/SBM/ERF_SBMTransportPrototype.cpp` is the ERF
finite-volume WENO-Z3 adapter over prepared intensive `X/rho` ghosts. A
positive face carrier uses the left/upwind value; a negative carrier uses the
right/upwind value. The canonical/reference tests cover additive offsets,
randomized algebraic equivalence, discontinuities, constant data, quadratic
cell averages, and both signs. The production identity remains
`WENO_Z3-group-FCT-v4`.

DonorCell and grouped WENO/FCT use the same prepared-state, accepted-ledger,
compact-projection, flux-register, reflux, and fail-closed validation path.
Runtime-sized one- and two-moment groups remain atomic. No `MAX_BINS` or
bulk-independent compact transport was introduced.

## G2 — temporal source and density views

`ERF::advance_sbm_stage` prepares source views before donor, WENO, support, or
diffusion reads. Stage 0 uses the prepared old view; later stages use the
prepared evaluation view when the contract requires it. The explicit density
views are `rho_anchor`, `rho_input`, and `rho_target`; they are not replaced
with a neighboring or constant density.

`StageContext::rhs_interval()` is the transport interval. Compressible RK3
uses `h/3`, `h/2`, and `h`; anelastic Heun uses `h` and `h`. These are distinct
from accepted-ledger weights: compressible accepts stage 2, while anelastic
uses `0.5*(stage 0 + stage 1)`.

## G3 — carrier-weighted AMR transfer

`AuxiliaryStateManager::carrier_weighted_fill` transfers each density-weighted
component as

```text
z_c = U_c/rho_c
z_f = supported fill/interpolation of z_c
U_f = z_f*rho_f(target time)
```

The coarse numerator and denominator are time-aligned before division. The
fine target density is passed by the ERF callsite. A finite positive density is
required; exact zeros remain exact zeros. Fine valid cells retain authority,
coarse data fills the grown ghost region, and final fine periodic/same-level
fill restores fine ownership.

Callsites are `ERF::MakeNewLevelFromCoarse`, `ERF::RemakeLevel`, and
`ERF::advance_sbm_stage`. The manager remains provider-neutral. Tests cover
creation, stage FillPatch, two-FAB authority, internal-FAB attached-property
donors, restriction, remake, variable density, and one/two-rank equivalence.

## G4 — compact projection

After supported spectral prolongation, stage fill, reflux, and matched
average-down, `SBMBulkProjection::apply_to_core` projects the authoritative
spectrum into compact fields over the valid-plus-available-grow region. A
stale compact ghost is never used as a spectral source. Projection is ordered
after spectral transfer and before the compact state is consumed.

## G5 — boundary policy

`Source/Microphysics/SBM/ERF_SBMBoundary.H` contains the shared GPU-safe
predicates used by production transport and the actual-stage demand oracle:

```text
suppress normal advective transfer: ImpermeableWall
suppress normal diffusion:          ImpermeableWall or AdvectiveOutflow
```

Periodic faces retain periodic transport. Outward advective outflow is
accepted; inward carrier is rejected and diffusion is not charged. A physical
wall has no required normal ghost read. The four-boundary production matrix
and direct unit tests cover x/y/z signs, wall closure, outward-only outflow,
compact/state transfer, and nonzero wall diffusion.

## G6 — host timestep path

`ERF::sbm_admissible_timestep` in
`Source/TimeIntegration/ERF_ComputeTimestep.cpp` is called by normal ERF
`ComputeDt`. On its supported static-Cartesian envelope it computes one
bin-independent reduction over host `vars_new`:

```text
R_host = max_i { [sum_d outgoing(u_face,d*rho_face)]/(rho_i V_i)
                 + [sum_faces A_f*rho_face*K/d_if]/(rho_i V_i) }
dt_host = 0.5/R_host
```

The diagnostic labels this carrier as `host_carrier=reconstructed_u_rho`.
The locked safety factor is unchanged. Invalid density, velocity, coefficient,
terrain, EB, or missing host ghost input fails closed. The implementation has
no spectral-bin loop, full host spectral copy, or private retry/subcycle.

## G7 — actual-stage carrier oracle and assertion

`ERF::advance_sbm_stage` passes the exact `avg_*mom` arrays to
`ERF_SBMTransportPrototype::advance_stage`. Before transport,
`measure_actual_stage_low_order_demand` computes the same low-order outgoing
advective and density-weighted diffusive demand from those arrays, exact
stage density, geometry, and the shared G5 boundary policy. It uses the exact
`StageContext::rhs_interval()`, reports advective, diffusive, combined, and
`tau*rate`, and feeds the result into the existing fail-closed stage
assertion.

The reduction is fixed-size and independent of the number of bins. It does
not change the `0.5` host safety factor, clip, retry, private-subcycle, or
narrow the accepted carrier speed.

The important qualification boundary is explicit:

```text
host carrier:   reconstructed u_face*rho_face from current host state
actual carrier: ERF avg_xmom/avg_ymom/avg_zmom after fast/source/AMR updates
```

`ERF_Substep_NS.cpp`, `ERF_Substep_MT.cpp`, and `ERF_Substep_T.cpp` update
`avg_*mom` with fast pressure/source terms and vertical flux terms. The
current host path does not provide a rigorous finite pre-step bound on all
such later values. The actual oracle is therefore a runtime defensive check,
not a proof that the host timestep always suffices. This is the unresolved
`NEEDS_EXPERIMENT` / `REVISE` item that forces the final partial gate.

## G8 — acoustic and AMR policy

`ERF_SBMContracts.cpp` rejects any active ERF acoustic substepping with the
stable reason:

```text
P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping
```

This rejection is tested for DonorCell and GroupedFCT_WENOZ3, one and two
moments, and one and two MPI ranks. It is separate from qualified native AMR
factor-2 subcycling. The supported AMR envelope is static Cartesian,
fully-periodic, `max_level=1`, spatial ratio `(2,2,2)`, native time factor 2,
and `TwoWay` coupling. Unsupported AMR, terrain, EB, moving geometry,
prescribed production spectral inflow, implicit/tensor diffusion, and schema
conversion fail closed.

## G9 — AMR synchronization and restart

`sbm_flux_reg` is active only for SBM two-way coupling. Accepted area-weighted
face fluxes are registered once with the YAFluxRegister `dt/dx` convention.
ERF native factor-2 fine stepping, reflux, post-reflux validation, compact
projection, average-down, and matched auxiliary average-down preserve the
authoritative spectral state. Material post-reflux rejection remains
fail-closed; clipping is not used.

The current stable identities are:

```text
constraint: complete-groups-donor-support-v2
transport: WENO_Z3-group-FCT-v4
AMR transfer: carrier-weighted-mixing-ratio-AMR-v1
boundary: periodic+wall+outflow-no-inflow-v1
schema: ERF-SBM-P2-4
```

Restart tests compare exact schema/layout/projection identities and
authoritative `SBMAux_*` plus compact fields. They do not reconstruct the
spectrum from bulk `qc/qr`.

## G10 — evidence locations and limitations

CTest registration is in `Tests/CTestList.cmake`. The production runners are:

* `Tests/RunSBMPrototype.cmake` for P1 stage timing;
* `Tests/RunSBMP2AMR.cmake` and `RunSBMP2AMRSubcycle.cmake`;
* `Tests/RunSBMP2Restart.cmake`;
* `Tests/RunSBMP2Boundaries.cmake`;
* `Tests/RunSBMP2Timestep.cmake`;
* `Tests/RunSBMP2VariableHostCFL.cmake`; and
* `Tests/RunSBMP2AcousticSubsteppingRejection.cmake`.

The direct unit coverage is in
`Tests/Unit/Microphysics/SBM/ERF_GTestSBMP0P1.cpp` and
`ERF_GTestSBMP2.cpp`. The closeout wall regression is
`ActualStageDemandSkipsWallDiffusionWithoutPhysicalGhostValues`; its physical
wall ghost values are deliberately non-finite and its coefficient is `K=1`.

The current local evidence is recorded in
`Source/Microphysics/SBM/P2_QUALIFICATION_REPORT.md`: merged SBM/P1/P2
22/22, full CTest 908/908, CMake DOUBLE and SINGLE/SINGLE builds, and the
captured per-stage wall rates. GitHub Actions and GPU runtime qualification
remain external/pending. No P3 work is authorized by this trace.
