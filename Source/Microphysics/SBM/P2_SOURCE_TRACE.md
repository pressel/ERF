# ERF SBM P2 archaeology-driven source trace

This document records the source-level contract used to close out
`sbm-p2-final-qualification` and `sbm-p2-final-closeout`. The current
implementation is on `sbm-p2-final-closeout` at
`c3eed779c36c2d24f95c317c371ad75a01a3549c`, starting from the required
`d006aef20410703e40bb7c19d6de1804588551a5`. The design authority is
`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`.

This trace separates the current-state host estimate from the actual ERF
carrier and identifies the exact no-acoustic production path. It is a P2
closeout record; P3 physics is not implemented here.

## 1. Authoritative state and transport architecture

| Contract | Source | Closeout interpretation |
|---|---|---|
| Authoritative spectral state | `Source/AuxiliaryState/ERF_AuxiliaryStateManager.{H,cpp}`, `ERF_SBMErfIntegration.cpp` | Provider-owned `SBMAux` is authoritative. Compact `qc/qr` are projections, never an independent transport state. |
| Physical layout | `ERF_SBMLayout`, `ERF_SBMBulkProjection` | One-moment populations carry `M`; two-moment populations carry physical `(M,C)`. |
| Constraints | `ERF_SBMConstraintGroups.{H,cpp}`, transport prototype | Complete population/bin/property groups are constrained atomically, with one face limiter. |
| Accepted transfer | `ERF_AuxiliaryFaceTransfer.{H,cpp}` | Accepted physical face transfer is used for divergence, projection, boundary budgets, reflux, and AMR accounting. |
| Diffusion | `ERF_SBMDiffusion.{H,cpp}`, `ERF_SBMTransportPrototype.cpp` | Explicit density-weighted diffusion acts on `X/rho` and is included once in the low-order flux. |
| Restart | `ERF_SBMRestart.{H,cpp}` | Layout, projection, transport, AMR, boundary, grid, and schema identities are checked before restore. |

P3 condensation/evaporation, activation/regeneration, collision/coalescence,
sedimentation, aerosol lifecycle, ice, and related warm-cloud physics remain
outside this branch.

## 2. Archaeology: the supported no-acoustic path

The supported production path is the following call chain:

```text
ERF::Evolve
  -> ComputeDt
  -> ERF::estTimeStep / sbm_admissible_timestep
  -> timeStep
  -> Advance
  -> swap
  -> FillPatch
  -> VelocityToMomentum
  -> advance_dycore
  -> MRISplitIntegrator
  -> slow_rhs_pre
       -> AdvectionSrcForRho
          -> avg_xmom / avg_ymom / avg_zmom
  -> no_substep
       -> predictor only; no avg-momentum modification
  -> slow_rhs_post
  -> apply_bcs
  -> ERF::advance_sbm_stage
```

`ERF_Substep_NS.cpp`, `ERF_Substep_MT.cpp`, and `ERF_Substep_T.cpp` modify or
time-average the `avg_*mom` arrays when acoustic substepping is active. P2
therefore rejects acoustic substepping rather than claiming that the host
estimate bounds those intermediate carriers. Native ERF AMR time subcycling
is a separate supported path under the envelope in Section 8.

`AdvectionSrcForRho` uses the current-state host carrier. On the static
Cartesian interior, `VelocityToMomentum` establishes the identity

```text
avg_xmom = rho*u,  avg_ymom = rho*v,  avg_zmom = rho*w
```

at cell centers through face mass carriers. For an ordinary face the
production convention is

```text
momentum_face = velocity_face * (rho_cell + rho_neighbor) / 2.
```

The direct `StageZeroAdvectionCarrierMatchesArithmeticFaceMomentum` test
executes the real `AdvectionSrcForRho` routine and independently verifies
that identity for nonconstant face velocities.

## 3. WENO-Z3, donor, and temporal contracts

`sbm_weno_z3_face` in `ERF_SBMTransportPrototype.cpp` is the ERF finite-volume
WENO-Z3 adapter over prepared intensive `X/rho` ghosts. A positive carrier
uses the left/upwind value and a negative carrier uses the right/upwind value.
The canonical transport identity remains `WENO_Z3-group-FCT-v4`.

DonorCell and grouped WENO/FCT share the prepared-state, complete-group,
accepted-ledger, compact-projection, reflux, and fail-closed validation path.
Runtime-sized one- and two-moment groups remain atomic; no `MAX_BINS` or
bulk-independent compact transport was introduced.

`StageContext::rhs_interval()` is the physical transport interval:

| Integrator | Stage intervals | Accepted ledger |
|---|---|---|
| Compressible RK3 | `h/3`, `h/2`, `h` | Stage-specific RK acceptance, including the stage-2 accepted state |
| Anelastic Heun | `h`, `h` | `0.5*(stage 0 + stage 1)` |

The intervals are not ledger weights and are not interchangeable in the
actual-stage check.

## 4. AMR and compact projections

`AuxiliaryStateManager::carrier_weighted_fill` transfers each density-weighted
component as

```text
z_c = U_c/rho_c
z_f = supported fill/interpolation of z_c
U_f = z_f*rho_f(target time)
```

The coarse numerator and denominator are time-aligned before division. Fine
valid cells retain authority, coarse data fills only the grown ghost region,
and final fine periodic/same-level fill restores fine ownership. The ERF
call sites are `MakeNewLevelFromCoarse`, `RemakeLevel`, and
`advance_sbm_stage`; the manager remains provider-neutral.

After supported spectral prolongation, stage fill, reflux, and matched
average-down, `SBMBulkProjection::apply_to_core` projects the authoritative
spectrum into compact fields over the valid-plus-available-grow region. A
stale compact ghost is never used as a spectral source.

The qualified AMR case is static Cartesian, one refined level, spatial ratio
`(2,2,2)`, native time factor 2, and `TwoWay` coupling. Accepted area-weighted
face fluxes are registered once using the YAFluxRegister `dt/dx` convention.
Post-reflux validation is fail-closed; clipping is not used.

## 5. One shared ERF boundary policy

`ERF_SBMErfBoundary.H` is the single ERF-facing policy constructor used by
the production integration and the host/actual demand paths. It maps ERF
physical boundary enums to the provider-neutral `TransportBoundaryPolicy`:

| ERF boundary | SBM policy |
|---|---|
| `symmetry`, `no_slip_wall`, `slip_wall` | `ImpermeableWall` |
| `outflow`, `ho_outflow`, `open` | `AdvectiveOutflow` |
| `periodic` in a periodic direction | `Periodic` |
| other prescribed physical boundary | `PrescribedSpectralInflow` |

Periodicity is authoritative by direction. The policy is stored in the
interleaved face order `[xlow,xhigh,ylow,yhigh,zlow,zhigh]`, matching the
AMReX orientation convention.

The boundary semantics are:

```text
impermeable wall:    suppress normal advection and normal diffusion
outward outflow:     retain outward advection, suppress normal diffusion
inward outflow:      reject/fail closed (no unresolved inflow state)
periodic:            use periodic transport
prescribed inflow:   explicit service exists, but production P2 rejects it
```

Physical wall faces are skipped before a physical-side ghost is read. This
is important for the positive-diffusion wall case whose physical wall ghost
values are deliberately non-finite. The direct boundary-policy and host-CFL
tests cover mapping, wall poisoning, outward-only outflow, and periodicity.

## 6. Host estimate versus actual stage carrier

`ERF::sbm_admissible_timestep` in `ERF_ComputeTimestep.cpp` is called by the
normal ERF `ComputeDt` path. On the supported static-Cartesian envelope it
performs one fixed-size, bin-independent GPU reduction over the current host
state. Its low-order demand is

```text
A_i = outgoing advective demand in cell i
D_i = density-weighted diffusive demand in cell i
A_max = max_i A_i
D_max = max_i D_i
R_cell_max = max_i (A_i + D_i)
selected_dt_host = 0.5 / (A_max + D_max)
diagnostic_bound = 1 / R_cell_max
```

The selected host value is conservative because `A_max + D_max` is at least
`R_cell_max`. The diagnostic explicitly says
`host_carrier=reconstructed_u_rho`. The path has no per-bin reduction, full
host spectral copy, host-only device capture, private retry, private
subcycle, clipping, or narrow empirical carrier multiplier. Invalid density,
velocity, coefficient, terrain, EB, or missing host ghost input fails closed
and reports a recommended timestep when one can be computed.

The actual-stage path is `ERF::advance_sbm_stage` into
`ERF_SBMTransportPrototype::advance_stage`. Before spectral transport,
`measure_actual_stage_low_order_demand` uses the exact `avg_xmom`,
`avg_ymom`, and `avg_zmom` arrays, exact stage density, geometry, shared
boundary policy, and `rhs_interval()`. Its fixed-size reduction reports
advective rate, diffusive rate, total rate, stage interval, `tau*rate`, worst
cell, and `recommended_max_host_dt`.

The stage guard is unconditional and diagnostic-only pointer state cannot
disable it. It rejects non-finite demand or `tau*rate > 1` (within the
scale-aware floating-point comparison), before stage flux mutation or
spectral transport. For a finite positive rate, with
`rhs_interval = alpha_s * current_full_step`, it recommends

```text
recommended_max_host_dt = current_full_step / tau_rate
                         = 1 / (alpha_s * actual_total_rate)
```

This is a full-step cap derived from the measured stage contract, not
`full_step / total_rate`.

The host estimate and actual check are intentionally distinct. Fast
pressure/source and AMR updates can change `avg_*mom` after the host estimate.
The runtime actual-stage check is a defensive guard and evidence of the
declared supported behavior; it is not a theorem that every nonlinear
intermediate carrier is bounded by the current-state host estimate. The
qualification envelope closes this gap operationally by rejecting acoustic
substepping and unsupported configurations.

## 7. Capability gates and rejected configurations

`ERF_SBMContracts.cpp` rejects active ERF acoustic substepping with the stable
reason:

```text
P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping
```

The rejection is tested for DonorCell and GroupedFCT_WENOZ3, one and two
moments, and one and two MPI ranks. Terrain, embedded boundaries, moving or
dynamic geometry, unsupported host ghosts, prescribed production spectral
inflow, implicit/tensor/cross diffusion, dynamic spectral grids, and schema
conversion fail closed. P3 physics remains out of scope.

The manufactured carrier mismatch `0.098125 -> 0.125` is deliberately
retained as a test-only negative control. It demonstrates that a velocity-
only or reconstructed host estimate must not be relabeled as the actual
`avg_*mom` carrier; it is not a production failure.

## 8. Evidence locations

The closeout test registration is in `Tests/CTestList.cmake`.

* `RunSBMPrototype.cmake` covers P1 stage timing.
* `RunSBMP2AMR.cmake` and `RunSBMP2AMRSubcycle.cmake` cover AMR and native
  subcycling.
* `RunSBMP2Restart.cmake` covers restart identity.
* `RunSBMP2Boundaries.cmake` covers wall/outflow policy and positive wall
  diffusion.
* `RunSBMP2Timestep.cmake` and `RunSBMP2VariableHostCFL.cmake` cover host
  demand, variable density, and fail-closed behavior.
* `RunSBMP2DynamicRK.cmake` runs real compressible RK3 and anelastic Heun
  carriers at one and two ranks and checks stage count, stage intervals,
  separate `actual_rate` and `tau*rate` vectors, and rank-equivalent evidence.
* `RunSBMP2AcousticSubsteppingRejection.cmake` covers the explicit rejection.
* `ERF_GTestSBMP0P1.cpp` and `ERF_GTestSBMP2.cpp` contain the direct unit
  and negative controls.

The local dynamic-RK evidence recorded by that script is:

```text
compressible_rk3 actual_rate (1r, 2r):
  302.73955166118299;302.93466123274578;302.90599276820632
compressible_rk3 tau_actual_rate (1r, 2r):
  0.100913183887061;0.15146733061637288;0.30290599276820634
anelastic_heun actual_rate (1r, 2r): 1;1
anelastic_heun tau_actual_rate (1r, 2r): 0.0001;0.0001
no_acoustic_real_carrier=verified
```

The final numerical disposition and CI ledger are maintained in
`P2_QUALIFICATION_REPORT.md`. No P3 work is authorized by this trace.
