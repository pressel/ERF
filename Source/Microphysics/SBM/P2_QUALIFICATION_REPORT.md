# ERF SBM P2 final closeout and qualification report

This report records the implementation of
`ERF_SBM_P2_Final_Closeout_and_Qualification_Codex_Prompt.md`. The attached
prompt is treated as the implementation specification; the user's request to
implement it is the authorization to change this repository. The warm-aerosol
design specification at
`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`
remains the architectural authority. P3 physics was not started.

## Final disposition

The closeout implementation fixes the production boundary-semantic mismatch,
adds a real positive-diffusion impermeable-wall regression, removes the
branch-introduced SINGLE-precision narrowing failures, and fixes the
branch-introduced warning failures found in the pre-merge CI archaeology.
The merged branch builds with the pinned Spack MPI toolchain and passes the
full local CTest suite (908/908).

The host timestep estimate is still a reconstructed `u_face*rho_face`
estimate, while the production SBM carrier is the actual ERF
`avg_xmom/avg_ymom/avg_zmom` state. ERF fast pressure/source and AMR updates
can change those arrays after the host estimate is formed. A rigorous finite
pre-step upper bound from the host state to every later actual carrier stage
has not been established without a further integrator redesign. The runtime
actual-stage assertion and the explicit acoustic-substepping rejection remain
in force, but they do not turn that missing proof into a proof.

The exact closeout gate is therefore:

```text
P2 PARTIAL — DO NOT BEGIN P3
```

## Repository and provenance

| Field | Value |
|---|---|
| Remote | `https://github.com/pressel/ERF` |
| Closeout branch | `sbm-p2-final-closeout` |
| Required starting SHA | `13ff28958a2286b9969d77dd4d6396a9d6475e4c` |
| Current closeout HEAD before this report commit | `6cfa577e56fc4b5e277415eb688b23926fdbe97a` |
| Implementation commit | `53ae6e1ea` — `Close P2 boundary semantics and precision gates` |
| Development merge commit | `6cfa577e5` — `Merge current development into P2 closeout` |
| `origin/development` at archaeology | `805bdd156cd1213200d839c3945a023c3d9bd7e6` |
| Merge base | `7ed1a98f91bac26b557e2f3bc0071a8af91935f7` |
| Compiler wrappers | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc`, `mpicxx` |
| MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| MPI runtime | Spack Open MPI 5.0.9 |
| Parallel build setting | `-j8` / `--parallel 8` |

The branch was created from the required SHA, then updated by a normal
non-fast-forward merge of `origin/development`; no history was rewritten and
the previously qualified `sbm-p2-final-carrier-cfl-closure` branch was not
modified. The merge had no conflicts and brought in the current development
surface-layer, planar-boundary, scalar-profile, CTest, and parallel-test
changes. Submodule revisions were not changed.

Pre-existing untracked build artifacts were preserved. Current untracked build
directories include `Build/`, `BuildDocs/`, `BuildTests/`,
`BuildTestsASAN/`, `BuildTestsDevelopmentMerge/`, and `BuildTestsSingle/`;
they are not part of the implementation commits.

## Implemented closeout changes

### Issue A — production boundary semantics

`ERF_SBMBoundary.H` now provides shared GPU-safe boundary-policy predicates.
The production transport path and its independent actual-stage demand oracle
use the same policy:

* impermeable walls suppress normal advective transfer and normal diffusion;
* advective outflow suppresses diffusion, while outward accepted transport is
  retained and inward transport is rejected;
* periodic and prescribed/inflow handling retain their existing explicit
  behavior.

The old actual-stage oracle could read a physical wall diffusion ghost even
though production transport charged no wall diffusion. That read is removed.
The wall fixture now uses `erf.sbm_diffusion_coeff=1.0`, so the regression is a
real positive-`K` numerical case with deliberately non-finite physical wall
ghost values rather than a zero-coefficient shortcut.

The new unit test
`SBMP2.ActualStageDemandSkipsWallDiffusionWithoutPhysicalGhostValues` verifies
for a one-cell x-normal wall case that the exact production semantics are
`advective_rate=0`, `diffusive_rate=4`, `maximum_rate=4`, and
`tau_actual_rate=4/3` for its `h/3` stage interval. It does not invent a
physical ghost value. The production 2M wall run also reports all four
boundary-policy faces and positive diffusion without a wall sink.

### Issue B — host versus actual carrier

The actual-stage oracle in
`ERF_SBMTransportPrototype.cpp` consumes the exact carrier arrays handed to
SBM: `avg_xmom`, `avg_ymom`, and `avg_zmom`. It uses the exact stage density,
face-density convention, geometry, boundary policy, and
`StageContext::rhs_interval()`. It performs fixed-size reductions and retains
the existing runtime admissibility assertion. It does not add private SBM
subcycling, retries, clipping, an empirical multiplier, or an `O(Nbin)` MPI
reduction.

The normal host path in `ERF_ComputeTimestep.cpp` computes a bin-independent
bound from the current host state and reconstructed `u_face*rho_face`, with
the locked `0.5` safety factor. This is now explicitly diagnosed as
`host_carrier=reconstructed_u_rho`; it is not mislabeled as the actual carrier.

Source archaeology shows why a proof is still missing. During ERF fast
substeps, `ERF_Substep_NS.cpp` and its terrain variants accumulate momentum
changes into the `avg_*mom` arrays. The terms include fast pressure gradients,
slow/source contributions, and vertical flux updates. AMR stage and
subcycling paths can also alter the state before a later SBM stage. No finite
pre-step bound currently relates all of those changes to the host estimate
without redesigning the timestep/integrator contract. This is classified as
`NEEDS_EXPERIMENT` / `REVISE`, not as a completed P2 proof.

Acoustic substepping remains fail-closed with the stable reason
`P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping`.
This is a scope restriction, not evidence that the host bound covers acoustic
intermediate carriers. Native AMR factor-2 subcycling is a separate qualified
ERF path.

### Issue C — SINGLE precision

The exact pre-merge macOS SINGLE failure was narrowing from raw `double`
literals into `amrex::Real` (`float`) in `ERF_SBMConstraintGroups.cpp` and
`ERF_SBMFCT.cpp`. The literals and selected defaults are now explicitly
constructed as `amrex::Real`. The actual-stage reduction buffer is also
`amrex::Real`, and the unused `restrict_constraint` helper was removed after
the local all-warnings build exposed it.

The matching local build uses MPI and particles enabled, SINGLE state
precision, SINGLE particle precision, Debug, and all warnings enabled. It
completes cleanly with the pinned Spack wrappers. The local numerical SBM
unit executable is not claimed as a SINGLE qualification suite: several
pre-existing tests use DOUBLE-oriented tolerances, whereas the upstream
SINGLE CI gate is a compile/report gate.

### Issue D — warnings

The exact pre-merge Linux GCC warning report identified six branch-introduced
warnings: an unused diffusion face variable, a copied range-loop item in the
P0/P1 test, an unused P2 lambda variable, an ambiguous `else`, and two copied
P2 contract loop items. All six are fixed. The local merged CMake builds emit
no branch warning from these sites. CI remains the authoritative warning gate
after push.

### Issue E — NetCDF/RRTMGP serial unit failure

The pre-merge workflow contained a NetCDF/RRTMGP job failure in
`Run Serial Unit Tests`. The exact GitHub log was unavailable during local
closeout because the GitHub API intermittently returned a connection error;
the failure is retained in the ledger below as `NEEDS_EXPERIMENT`, not
silently relabeled. Current `origin/development` had a successful recent
NetCDF/RRTMGP workflow at its then-current SHA, so this remains an upstream
CI comparison item for the pushed closeout branch. No local NetCDF provider is
present in the pinned Spack view.

## Actual-stage production evidence

The merged wall runner (`SBM_P2_BOUNDARIES_2M`) uses positive diffusion and
prints the following exact production values from the captured 2-rank run:

```text
host stability bound = 0.0008650519031
host advective_rate = 2
host diffusive_rate = 576
stage 0: tau=0.00028835063437139563 actual_rate=578
         actual_advective_rate=2 actual_diffusive_rate=576
         tau_actual_rate=0.16666666666666669
stage 1: tau=0.00043252595155709344 actual_rate=578.000000...
         actual_advective_rate=2 actual_diffusive_rate=576
         tau_actual_rate=0.25
stage 2: tau=0.00086505190311418688 actual_rate=578.00000072064734
         actual_advective_rate=2 actual_diffusive_rate=576
         tau_actual_rate=0.50000000062339733
```

The qualified compressible stage intervals are `h/3`, `h/2`, and `h`; the
qualified anelastic Heun intervals are `h` and `h`. Existing manufactured
carrier evidence remains in the unit and production fixtures:

```text
reconstructed_u_rho=0.098125
actual_avg_xmom=0.125
actual_over_reconstructed=1.2748091603...
```

The variable-density fixture retains the intentionally non-identical host and
actual values and passes at one and two ranks:

```text
host_cfl_diffusion_rate=1154.513514
host_cfl_diffusion_advective_rate=4.243243243
host_cfl_diff_actual_stage_count=3
host_cfl_diff_actual_stage_max_rate=1155.6756756756756
host_cfl_diff_actual_stage_max_tau_rate=0.50050331249853697
host_cfl_old_velocity_only_bound=0.25
host_cfl_old_velocity_only_product=1.06081081075
```

The last two values are a negative control against the old velocity-only
estimate. They do not establish a host-to-actual proof.

## Adversarial and negative controls

The following controls were run as temporary mutations and restored before
the implementation commit. The new wall test also supplies the requested
wall-diffusion adversary: restoring the old wall diffusion read makes the
NaN physical ghost observable and fails the test.

| Control | Mutation | Expected/observed result |
|---|---|---|
| NC-1 | Bypass the corrected host carrier/CFL bound | Variable-density production check reaches actual-stage inadmissibility (`tau_rate=1.35135...`). |
| NC-2 | Remove acoustic-substepping rejection | Acoustic startup test fails because the stable capability reason is absent. |
| NC-3 | Omit an outgoing y-face | Direct oracle fails (`0.1` rather than `0.4` advective rate). |
| NC-4 | Omit diffusion from combined demand | Direct oracle fails (`0` rather than `0.45` diffusive and `0.85` total rate). |
| NC-5 | Replace anchor density with constant `1` | Variable-density oracle fails (`0.4` rather than `0.2`). |
| NC-6 | Restore wall diffusion charging/physical-ghost read | K>0 wall unit fails on the deliberately non-finite wall ghost. |
| NC-7 | Remove actual-stage assertion | Production stage safety test no longer fails closed; control rejected. |
| NC-8 | Use an empirical host factor to hide carrier growth | Not accepted; the implementation retains `0.5` and reports host/actual separately. |
| NC-9 | Use SINGLE-incompatible reduction/literals | Matching SINGLE all-warnings build fails; corrected build is clean. |
| NC-10 | Allow acoustic configuration | Four acoustic rejection tests fail; accepted configuration is rejected before stepping. |

## Local build and test ledger

All CMake builds and MPI tests used the pinned Spack wrappers. MPI CTest runs
were executed with the required `--parallel 8` and with MPI socket access
enabled by the environment.

| Check | Result |
|---|---|
| CMake Release, MPI ON, DOUBLE, no optional providers | Passed: `cmake --build BuildTests --parallel 8` |
| CMake Debug, MPI/particles ON, SINGLE/SINGLE, all warnings | Passed: `cmake --build BuildTestsSingle --parallel 8` |
| Merged SBM/P1/P2 label suite | Passed: 22/22 |
| P1 compressible and anelastic smoke | Passed: 6/6 production cases; actual stage counts 6 and 4 |
| P2 AMR, donor, variable density, restart, active MPI, native subcycling | Passed |
| Four boundary cases, including K>0 wall and outward-only outflow | Passed |
| Variable-density host-CFL at 1 and 2 ranks | Passed: 2/2 |
| Acoustic rejection Donor/Grouped × 1M/2M | Passed: 4/4 |
| Full merged repository CTest | Passed: 908/908 |
| GNU Make DOUBLE, MPI, no NetCDF optional provider | Passed with pinned Spack MPI: `ERF3d.gnu.TEST.MPI.ex` |
| GNU Make DOUBLE, NetCDF enabled | Compile succeeded but link failed because the pinned Spack view has no NetCDF library/provider; no host NetCDF was substituted |
| Documentation script | Python/catalog and Doxygen passed; Sphinx unavailable (`sphinx-build` missing) |
| GPU runtime | Not run; no qualified GPU runtime is available on this host |

The CMake `BuildTests` cache is the no-provider baseline: MPI ON, DOUBLE,
NetCDF OFF, HDF5 OFF, particles OFF, RRTMGP OFF, and EAMxx SHOC OFF. It is
the configuration used for the 908/908 full CTest run, so the SBM source and
the normal no-provider ERF path are exercised together.

## Pre-merge GitHub Actions archaeology and closeout ledger

The required workflow families were inspected before editing: Linux GCC,
NetCDF/RRTMGP, ERF CI, macOS, Windows, Windows MPI, CUDA, HIP, SYCL, style,
codespell, documentation HTML, and draft PDF. At the required base SHA
`13ff289...`, Style, codespell, draft PDF, and DocHTML passed. The failures
below were recorded from the exact workflow/job reports:

| Workflow/job | Base result | Root cause and closeout status |
|---|---|---|
| Linux GCC / `GNU@9.3 C++17 Release - Mesh DOUBLE Particles OFF DOUBLE` | Failed Report | Six branch warnings; fixed in `53ae6e1ea`. |
| macOS / `Apple Clang@11.0 - Mesh SINGLE Particles ON SINGLE` | Failed Report | Three raw-double-to-`amrex::Real` narrowing diagnostics; fixed. |
| ERF CI / particles-off job | Failed Report | Same branch warning family; fixed. |
| ERF CI / particles-on matrix job | Canceled after matrix failure | Re-run required on pushed closeout branch. |
| Windows / CMake Unit Tests | Failed | Exact upstream log was not available locally; re-run required. |
| Windows MPI / SINGLE and DOUBLE/SINGLE build jobs | Failed Build | Exact upstream log was not available locally; re-run required. |
| SYCL / SINGLE/SINGLE Report | Failed | Exact upstream log was not available locally; re-run required. |
| CUDA / DOUBLE/particles-off Report | Failed | Exact upstream log was not available locally; re-run required. |
| HIP / SINGLE/SINGLE Report | Failed | Exact upstream log was not available locally; re-run required. |
| NetCDF/RRTMGP / GNU, NetCDF ON, particles ON, RRTMGP OFF serial tests | Failed `Run Serial Unit Tests` | Exact log retrieval hit GitHub API connection failure; current development had a recent successful corresponding workflow. |
| Style, codespell, draft PDF, DocHTML | Passed | No SBM-specific action required. |

The ledger is intentionally not converted into a green CI claim by local
results. The authoritative closeout status will be updated after the branch
is pushed and GitHub Actions completes.

## Supported and rejected envelope

Supported and locally qualified:

* static Cartesian, double-precision CMake/no-provider path;
* runtime-sized complete one- and two-moment population/bin groups;
* `DonorCell` and `GroupedFCT_WENOZ3`;
* explicit scalar density-weighted diffusion;
* periodic, impermeable-wall, and outward-only advective-outflow boundaries;
* one periodic AMR level with spatial ratio `(2,2,2)`;
* native ERF AMR time-refinement factor 2;
* `TwoWay` AMR reflux/feedback and dynamic regrid/remake;
* strict same-decomposition restart equivalence;
* variable dry density; and
* compressible RK3 and anelastic Heun with acoustic substepping disabled.

Rejected or fail-closed:

* ERF acoustic substepping for SBM P2;
* `max_level > 1`, unsupported spatial/time ratios, nonperiodic AMR, or
  unsupported multilevel `OneWay` coupling;
* prescribed production spectral inflow;
* terrain, embedded boundaries, moving/dynamic geometry, and unsupported host
  ghosts;
* implicit/native moisture diffusion and tensor/cross diffusion;
* dynamic spectral grids or schema conversion; and
* all P3 condensation/evaporation, activation/regeneration,
  collision/coalescence, sedimentation, aerosol lifecycle, ice, and other
  warm-cloud/macrophysics physics.

No `MAX_BINS`, bulk independent `qc/qr` transport, per-stage `O(Nbin)` MPI
reduction, unbounded diagnostic scratch, full host spectral copy, clipping,
private SBM retry, or private SBM subcycle was introduced.

## Changed artifacts

Production and test changes are in:

* `Source/Microphysics/SBM/ERF_SBMBoundary.H`;
* `Source/Microphysics/SBM/ERF_SBMConstraintGroups.H/.cpp`;
* `Source/Microphysics/SBM/ERF_SBMDiffusion.H`;
* `Source/Microphysics/SBM/ERF_SBMFCT.H/.cpp`;
* `Source/Microphysics/SBM/ERF_SBMTransportPrototype.cpp`;
* `Source/TimeIntegration/ERF_ComputeTimestep.cpp`;
* `Tests/RunSBMP2Boundaries.cmake`;
* `Tests/Unit/Microphysics/SBM/ERF_GTestSBMP0P1.cpp`;
* `Tests/Unit/Microphysics/SBM/ERF_GTestSBMP2.cpp`; and
* `Tests/Unit/Microphysics/SBM/inputs_sbm_p2_wall`.

The development merge additionally contains the current upstream changes to
surface-layer/planar-boundary production code and their tests. The source
trace is maintained separately in
`Source/Microphysics/SBM/P2_SOURCE_TRACE.md`.

## Qualification classifications

| Claim | Classification |
|---|---|
| Production wall/outflow semantics and K>0 wall oracle match | `SURVIVES_REVIEW` locally |
| CMake DOUBLE no-provider path and full local CTest | `SURVIVES_REVIEW` locally |
| SINGLE/SINGLE compile compatibility | `SURVIVES_REVIEW` locally; upstream CI pending |
| Branch warning fixes | `SURVIVES_REVIEW` locally; upstream CI pending |
| Host `u_face*rho_face` estimate is a rigorous bound for actual `avg_*mom` at all stages | `NEEDS_EXPERIMENT` / `REVISE` |
| NetCDF/RRTMGP serial failure classification | `NEEDS_EXPERIMENT` pending exact rerun/log |
| P3 readiness | `REFUTED` by the unresolved host-carrier proof and explicit final gate |

The fallback status is deliberate. P2 is not declared fully qualified, and
P3 must not begin until the host-versus-actual carrier proof (or a permitted
architecture-preserving redesign) and the pushed CI ledger are closed.
