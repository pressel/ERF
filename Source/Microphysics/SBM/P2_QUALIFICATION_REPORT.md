# ERF SBM P2 archaeology-driven closeout and qualification report

This report records the implementation of
`ERF_SBM_P2_Final_Archaeology_Driven_Closeout_Codex_Prompt.md`. The attached
prompt is treated as the implementation specification; the user's request to
implement it is the authorization to change this repository. The warm-aerosol
design specification at
`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`
remains the architectural authority. P3 physics was not started.

## Disposition

The code and local qualification evidence are complete for the declared P2
configuration. The required GitHub Actions matrix for implementation SHA
`c5e1b4bda` is still in progress; several jobs have passed, but the matrix is
not yet final. GPU runtime qualification is separate and was not run on this
host. Until the remote matrix is complete, the exact permitted disposition is:

```text
P2 PARTIAL — DO NOT BEGIN P3
```

This is a qualification-state disposition, not a claim of a local code
failure. After all required workflows complete, update this report in a
docs-only commit with the exact ledger and promote the disposition only if
every required gate is green.

## Repository, branch, and provenance

| Field | Value |
|---|---|
| Remote | `https://github.com/pressel/ERF` |
| Branch | `sbm-p2-final-closeout` |
| Required starting SHA | `f57d5b5e0b1da862adf45b853a9cd17f09e99cdb` |
| Implementation SHA | `c5e1b4bda9e76137ef07bb1baf13d9c115b29641` |
| `origin/development` checked | `805bdd156cd1213200d839c3945a023c3d9bd7e6` |
| Development relation | Already an ancestor of the required starting SHA; no additional merge was needed |
| Implementation commit | `c5e1b4bda` — `SBM P2: close archaeology-driven qualification gaps` |
| Evidence commit | This docs-only snapshot; final CI ledger update may add a later docs-only commit |
| Spack compiler wrappers | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc`, `mpicxx`, `mpifort` |
| Spack MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| Build parallelism | `-j8` / `--parallel 8` |

No history was rewritten. The pre-existing untracked build artifacts were
preserved and are not part of the implementation commit.

## Implemented changes

The implementation commit changes the following production and qualification
files:

* `Source/Microphysics/SBM/ERF_SBMErfBoundary.H` — one ERF-to-SBM boundary
  policy constructor with periodicity precedence.
* `Source/Microphysics/SBM/ERF_SBMHostCFL.H` — fixed-size GPU-safe host face
  and cell demand oracle.
* `Source/Microphysics/SBM/ERF_SBMErfIntegration.cpp` — shared boundary
  policy in production integration and portable device captures.
* `Source/Microphysics/SBM/ERF_SBMTransportPrototype.{H,cpp}` — exact-stage
  carrier measurement, unconditional guard, and recommended full-step cap.
* `Source/Microphysics/SBM/Make.package` — build registration for the new
  provider helpers.
* `Source/TimeIntegration/ERF_ComputeTimestep.cpp` — current-state host
  estimate using the shared oracle and fixed-size reduction.
* `Tests/CTestList.cmake` — real dynamic carrier fixture registration.
* `Tests/RunSBMP2DynamicRK.cmake` — one/two-rank RK3 and Heun evidence.
* `Tests/Unit/Microphysics/SBM/ERF_GTestSBMP2.cpp` — direct carrier,
  boundary, host-CFL, stage-guard, AMR, restart, and negative-control tests.
* `Tests/Unit/Microphysics/SBM/inputs_sbm_p2_dynamic_rk` — compressible
  static-Cartesian real-carrier fixture.
* `Tests/Unit/Microphysics/SBM/inputs_sbm_p2_anelastic_heun` — anelastic
  real-carrier Heun fixture.

Earlier commits in the current branch lineage retain the P0/P1/P2 carrier,
AMR, precision, warning, and portable-capture fixes. The current closeout
commit starts exactly at the required `f57d5b5e0` state.

## Carrier and CFL contract

The no-acoustic production path is

```text
Evolve -> ComputeDt -> estTimeStep/sbm_admissible_timestep -> timeStep
  -> Advance -> swap -> FillPatch -> VelocityToMomentum -> advance_dycore
  -> MRISplitIntegrator -> slow_rhs_pre -> AdvectionSrcForRho
  -> no_substep -> slow_rhs_post -> apply_bcs -> advance_sbm_stage
```

`AdvectionSrcForRho` consumes the current-state host carrier. On the static
Cartesian interior, ordinary face momentum is independently verified as

```text
momentum_face = velocity_face * (rho_cell + rho_neighbor) / 2.
```

The host estimator in `ERF_ComputeTimestep.cpp` is bin-independent and uses
the current host state. Its diagnostic is explicitly
`host_carrier=reconstructed_u_rho`; it retains the locked `0.5` safety factor.
It includes outgoing advection and density-weighted diffusion, rejects invalid
state/coefficient/geometry data, and fails closed. It does not copy the host
spectrum, reduce per bin, allocate per bin, retry, clip, or privately
subcycle.

The actual-stage oracle in `ERF_SBMTransportPrototype.cpp` consumes the exact
`avg_xmom`, `avg_ymom`, and `avg_zmom` arrays handed to SBM after the ERF
updates. It uses exact stage density, geometry, shared boundary policy, and
`StageContext::rhs_interval()`. The unconditional guard reports
`advective_rate`, `diffusive_rate`, `total_rate`, `tau_rate`, worst cell, and
`recommended_max_host_dt`, and rejects non-finite demand or `tau_rate > 1`
within a scale-aware roundoff tolerance before stage flux mutation.

For a finite positive actual rate:

```text
recommended_max_host_dt = full_step / total_rate.
```

The guard is diagnostic-only-pointer independent. There is no retry, clipping,
private subcycle, empirical speed multiplier, or narrowed carrier definition.

The stage intervals are explicit and are not ledger weights:

| Integrator | Stage interval |
|---|---|
| Compressible RK3 | `h/3`, `h/2`, `h` |
| Anelastic Heun | `h`, `h` |

## Boundary fix

`ERF_SBMErfBoundary.H` maps symmetry/no-slip/slip walls to
`ImpermeableWall`, outflow/HO-outflow/open to `AdvectiveOutflow`, and honors
periodicity before physical-face classification. The shared policy is stored
as `[xlow,xhigh,ylow,yhigh,zlow,zhigh]`.

* Wall faces suppress normal advection and normal diffusion and are skipped
  before physical-side ghost reads.
* Outward advective outflow retains outward advection and suppresses normal
  diffusion.
* Inward outflow is rejected because no resolved inflow state exists.
* Prescribed spectral inflow remains an explicit service but is rejected by
  the declared production P2 envelope.

The positive-diffusion wall regression uses deliberately non-finite wall
ghost values. The wall path remains valid because it does not read those
values. The direct host-CFL regression also verifies outward outflow and
rejects inward outflow.

## Acoustic, AMR, restart, and support envelope

Active ERF acoustic substepping is rejected with the stable reason:

```text
P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping
```

This rejection is tested for DonorCell and GroupedFCT_WENOZ3, one and two
moments, and one and two ranks. Native ERF AMR factor-2 subcycling is separate
and supported for static Cartesian one-level `(2,2,2)` `TwoWay` AMR. Accepted
area-weighted face fluxes are registered once, reflux/average-down occurs
before compact projection, and authoritative spectral state owns restart.

Supported local envelope:

* static Cartesian geometry, double-precision no-provider CMake path;
* runtime-sized complete one- and two-moment groups;
* DonorCell and GroupedFCT_WENOZ3;
* explicit scalar density-weighted diffusion;
* periodic, impermeable-wall, and outward-only advective outflow;
* one level of native AMR with spatial ratio `(2,2,2)`, time factor 2, and
  `TwoWay` coupling;
* dynamic regrid/remake, matched auxiliary average-down, and strict restart;
* variable dry density; and
* compressible RK3 and anelastic Heun with acoustic substepping disabled.

Rejected or fail-closed: acoustic substepping, unsupported AMR ratios or
couplings, terrain, embedded boundaries, moving geometry, unsupported host
ghosts, prescribed production spectral inflow, implicit/tensor/cross
diffusion, dynamic spectral grids, schema conversion, and all P3 physics.

## Real-carrier fixture matrix

`SBM_P2_DYNAMIC_REAL_CARRIERS` runs both fixtures at one and two ranks and
parses the production actual-stage diagnostics. The observed local evidence
is:

```text
compressible RK3, 1 rank: 3 stages
tau*rate: 0.100913183887061; 0.15146733061637288; 0.30290599276820634
compressible RK3, 2 ranks: 3 stages, exact same values
anelastic Heun, 1 rank: 2 stages
tau*rate: 0.0001; 0.0001
anelastic Heun, 2 ranks: 2 stages, exact same values
no_acoustic_real_carrier=verified
```

The manufactured mismatch `reconstructed_u_rho=0.098125` versus
`actual_avg_xmom=0.125` is retained as a deliberate test-only negative
control. It prevents the host reconstruction from being mislabeled as the
actual carrier.

## Adversarial controls

The qualification suite contains independent controls for the required
failure modes. Their expected failure signatures are:

| Control | Failure that must be observable |
|---|---|
| Wall ghost poisoning | Reading the physical wall ghost makes the K>0 wall test non-finite; the corrected path skips it. |
| Inward outflow | Host and production boundary checks reject the unresolved inflow sign. |
| Null diagnostic pointer | The unconditional stage guard still rejects an unsafe actual stage. |
| Omitted face | The independent host/oracle demand no longer matches the all-face reference. |
| Omitted diffusion | Combined advection-plus-diffusion demand loses its diffusion contribution. |
| Wrong stage interval | `tau*rate` no longer matches the RK3/Heun interval contract. |
| Unsafe manufactured stage | The stage guard fails closed and reports the recommended maximum full-step dt. |

The corresponding direct tests are in `ERF_GTestSBMP2.cpp`, including
`HostCFLBoundarySemanticsSkipWallPoisonAndRejectInwardOutflow`,
`ProductionCombinedAdvectionDiffusionFailsClosed`,
`ProductionVariableDensityHostCFLBypassFailsClosed`,
`ProductionCombinedDemandUsesPreStageBaselineForAllStageContracts`, and
`StageZeroAdvectionCarrierMatchesArithmeticFaceMomentum`.

## Local build and test ledger

All current CMake and GNU Make compilation used the pinned Spack wrappers and
`-j8`/`--parallel 8`.

| Check | Result |
|---|---|
| `cmake --build BuildTests --parallel 8` | Passed |
| `ctest --test-dir BuildTests -L sbm --output-on-failure -j8` | Passed, 23/23 |
| `ctest --test-dir BuildTests --output-on-failure -j8` | Passed, 912/912 |
| Production `cmake --build Build --parallel 8` | Passed, DOUBLE, MPI, particles OFF |
| `cmake --build BuildTestsSingle --parallel 8` | Passed, SINGLE/SINGLE, MPI and particles ON, all warnings |
| SINGLE runtime SBM label | Not a qualification gate: legacy DOUBLE-oriented tolerances produce known runtime failures; compile gate passes |
| GNU Make DOUBLE MPI, NetCDF OFF | Passed; produced `Exec/ERF3d.gnu.TEST.MPI.ex` |
| GNU Make NetCDF ON | Not available in the pinned Spack view; no host provider substituted |
| Dynamic real-carrier fixtures | Passed at 1 and 2 ranks for RK3 and Heun |
| GPU runtime | Not run on this host |

The full CTest matrix includes P0/P1 regressions, AMR, restart, boundary,
host-CFL, acoustic rejection, MPI, and the new real-carrier fixture. The
scale-aware WENO ordering check only admits the measured CI rounding spread
(`~7e-15` in DOUBLE); it does not change the convergence requirement.

## CI ledger

The exact pre-closeout failures were inspected from the f57 artifacts. The
shared serial failure was a one-ULP WENO ordering difference, CUDA/SYCL
failures were device-capture/extended-lambda portability issues, and the
Windows/Linux warning and precision reports were branch-introduced. The
implementation fixes those concrete causes without disabling tests.

| Workflow | Implementation SHA `c5e1b4bda` result |
|---|---|
| Style | Passed |
| codespell | Passed |
| draft PDF | Passed |
| DocHTML | Passed |
| Linux GCC | Pending final run completion at report update |
| Linux GCC NetCDF/RRTMGP | Pending final run completion at report update |
| ERF CI | Pending final run completion at report update |
| macOS | Pending final run completion at report update |
| Windows / Windows MPI | Pending final run completion at report update |
| CUDA | Pending final run completion at report update |
| HIP | Pending final run completion at report update |
| SYCL | Pending final run completion at report update |

The final evidence commit must replace the pending entries with the exact
conclusion/job result or an exact, reproducible external-infrastructure
limitation. A local pass must never be substituted for a required CI result.

## Classification and stop conditions

| Claim | Classification |
|---|---|
| ERF boundary mapping, wall closure, and outward-only outflow | `SURVIVES_REVIEW` locally |
| Host/actual carrier identity and exact stage diagnostics | `SURVIVES_REVIEW` locally |
| Static-Cartesian no-acoustic P2 envelope | `SURVIVES_REVIEW` locally, pending final CI ledger |
| Host reconstructed carrier is a theorem-level bound for every later actual stage | `NEEDS_EXPERIMENT` / `REVISE`; not claimed |
| GPU runtime qualification | Separate; not run |
| P3 readiness | Must remain closed until the final disposition is green |

No P3 work is authorized by this report. The final disposition must use only
one of the two exact strings required by the archaeology prompt:

```text
P0 COMPLETE
P1 QUALIFIED
P2 v1.0 FULLY QUALIFIED FOR THE DECLARED SUPPORTED CONFIGURATION
READY FOR P3 REVIEW
```

or

```text
P2 PARTIAL — DO NOT BEGIN P3
```
