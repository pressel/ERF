# ERF SBM P2 final AMR/CFL correction and P3-readiness qualification report

This report records the evidence for the implementation requested by
`ERF_SBM_P2_Final_AMR_CFL_Correction_and_P3_Readiness_Prompt.md`. The warm-
aerosol design specification at
`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`
remains the architectural authority. The attached prompt is treated as the
implementation request; its embedded instructions are recorded here as
qualification criteria, not as additional user requests. P3 physics is not
implemented or claimed by this pass.

## Disposition

The four final P2 correction areas are implemented and verified in the bounded
supported envelope below. The focused positive tests, 1/2-rank MPI checks, the
real P2 matrix, the full CTest audit, the production CMake build, and the
GNUmake build passed. The documentation script completed its available Python
and Doxygen checks but could not complete Sphinx/Graphviz steps because those
executables are not installed. No qualified GPU runtime is available in this
environment.

## Provenance and toolchain

| Field | Value |
|---|---|
| Branch | `sbm-p2-final-amr-cfl-fixes` |
| Required starting SHA | `ad74ec081bc7a8295e7e0b40fbe18822a92cf44c` |
| Implementation Commit A | `2451376b6` |
| Evidence Commit B | This report-only commit; resolve with `git rev-parse HEAD` |
| Remote | `https://github.com/pressel/ERF` |
| C compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc` |
| C++ compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicxx` |
| MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| MPI runtime | Pinned Spack Open MPI 5.0.9 |
| Precision | Double |
| Build parallelism | `-j8` / `--parallel 8` where supported |

Both CMake caches confirm MPI enabled, double precision, and the pinned Spack
compiler wrappers. Submodules were not updated. The pre-existing untracked
build artifacts were preserved and are intentionally outside the two commits.

Submodule revisions at qualification were unchanged:

| Submodule | Revision |
|---|---|
| AMReX | `e60cdc18711ccf7fc0616d7a2fdf062021376976` |
| Noah-MP | `e03243d2c5eac19b1c6a02a2c3ca12f03f92cceb` |
| RRTMGP | `71693e7490db0908a625eadb6ac3f23967de2c7d` |
| WW3 | `f5fa5822849e33976cdc935a91b304c60bc2a976` |
| ekat | `10ab9d34ceba4eead542702a29676adf77ad91af` |
| googletest | `52eb8108c5bdec04579160ae17225d66034bd723` |

## Implemented correction closure

### F-P2-FINAL-01 — carrier-weighted AMR stage authority

`AuxiliaryStateManager::fill_stage_from_coarse` now performs the final
fine-level periodic/same-level `FillBoundary` after coarse-derived carrier-
weighted ghost writes. Fine valid and same-level values therefore remain
authoritative, while a true uncovered coarse/fine ghost retains the
coarse-interpolated value. The transfer remains generic and uses the explicit
target host density to form and restore `U/rho`.

Coverage includes two fine FABs, nonuniform positive target density, an
internal same-level interface, a true coarse/fine ghost, and an adversarial
attached-property donor across the internal FAB boundary. The temporary
negative control that removed the final sync failed with large fine-authority
errors and was restored before qualification.

### F-P2-FINAL-02 — 2M endpoint donor preparation

DonorCell and grouped transport now share full `fabbox()` coordinate
preparation. Prepared source ghosts include true coarse/fine donors. The 2M
conversion uses the physical `(M,C) -> (L,H) -> (L/rho,H/rho)` representation,
FMA-based cancellation handling, scale-aware tolerances, and fail-closed
nonphysical input behavior without an order-one floor. The temporary
`validbox()`-only control failed the independent 2M oracle and was restored.

The independent two-level test checks both carrier signs, accepted flux,
final-state update, endpoint realizability, compact projection, and rank
equivalence.

### F-P2-FINAL-03 — variable-density host timestep demand

The normal ERF `ComputeDt` path now bounds the actual bin-independent low-order
demand for static Cartesian geometry:

```text
tau/(rho_anchor V) * [sum_d max(sigma_d mdot_d, 0)
                      + sum_faces A_f rho_face K/d_if] <= 1,
mdot_d = u_face,d rho_face.
```

The implementation uses one fixed three-component MPI reduction, returns the
mathematical bound to the existing host admissibility helper, retains its
documented `0.5` safety factor, and logs both bounds plus the selected
timestep. The exact stage-local guard remains in place; there is no hidden
subcycling or velocity-only replacement. Unsupported geometry and invalid
host data fail closed.

The 1/2-rank host fixture includes a variable-density counterexample, a
combined advection/diffusion case, and a zero-diffusion comparison. The
temporary host-bound bypass failed at the exact stage-local rejection and was
restored.

Rank-independent evidence from the fixture:

```text
host_cfl_zero_rate=4.243243243
host_cfl_zero_mathematical_bound=0.2356687898
host_cfl_zero_selected_dt=0.1178343949
host_cfl_zero_bound=0.1178343949
host_cfl_diffusion_rate=1154.513514
host_cfl_diffusion_advective_rate=4.243243243
host_cfl_diffusion_selected_dt=0.0004330828476
host_cfl_diffusion_bound=0.0004330828476
host_cfl_old_velocity_only_bound=0.25
host_cfl_old_velocity_only_product=1.06081081075
host_cfl_no_substepping=verified
host_cfl_combined_advection_diffusion=verified
```

### F-P2-FINAL-04 — canonical WENO-Z3 and transport identity

The production SBM WENO-Z3 helper now uses the canonical fixed precision
epsilon (`1e-40` in double precision and `1e-12` in single precision) and no
adaptive beta-dependent epsilon floor. An independently written randomized
reference test, a discontinuous `[0,1,1,1]` stencil and mirror, finite
constant stencils, retained smooth convergence, and both carrier signs pass.
The temporary adaptive-epsilon control failed the randomized and discontinuity
checks and was restored.

The transport identity is `WENO_Z3-group-FCT-v4` and the restart schema is
`ERF-SBM-P2-4`. The prior v3 identity is explicitly rejected by the restart
unit test; no silent schema conversion is provided.

## Supported and rejected envelope

Supported:

- static Cartesian, double precision;
- runtime-sized complete one-moment and two-moment liquid population/bin
  groups, with physical 2M storage `(M,C)`;
- `DonorCell` and `GroupedFCT_WENOZ3` transport;
- explicit scalar density-weighted diffusion;
- single-level periodic, wall, and outward-only advective-outflow boundaries;
- fully periodic `max_level=1` AMR with spatial ratio `(2,2,2)`;
- native time refinement factor `2`, with two fine substeps per coarse step;
- `TwoWay` AMR reflux/feedback;
- dynamic regrid/remake;
- strict same-decomposition restart equivalence; and
- variable dry density.

Rejected or fail-closed:

- `max_level > 1`, other spatial/time refinement ratios, nonperiodic AMR, or
  multilevel `OneWay` coupling;
- prescribed production spectral inflow;
- terrain, embedded boundaries, moving/dynamic geometry, or unsupported host
  ghost state;
- implicit/native moisture diffusion and tensor/cross diffusion;
- dynamic spectral grids or schema conversion; and
- P3 condensation/evaporation, activation/regeneration,
  collision/coalescence, sedimentation, aerosol lifecycle, ice, and other
  warm-cloud/macrophysics physics.

## Test and build evidence

All compilation and MPI execution used the pinned Spack wrappers above.

| Check | Exact command/result |
|---|---|
| Test build | `cmake --build BuildTestsDevelopmentMerge --parallel 8 --target erf_exec erf_unit_tests` — passed |
| Focused unit tests, 1 rank | `mpiexec -n 1 BuildTestsDevelopmentMerge/Tests/Unit/erf_unit_tests --gtest_filter='SBMP2.CarrierWeightedStageFillPreservesFineFABAuthority:SBMP2.AttachedPropertySupportUsesFineDonorAcrossInternalFABBoundary:SBMP2.DonorCellTwoMomentUsesPreparedCoarseFineEndpointDonors:SBMP2.ProductionVariableDensityHostCFLBypassFailsClosed:SBMP2.WENOZ3CanonicalERF*'` — 6/6 passed |
| Focused unit tests, 2 ranks | Same pinned launcher with `-n 2` — 6/6 passed |
| Real P2 matrix | `ctest --test-dir BuildTestsDevelopmentMerge -L sbm-p2 --output-on-failure` — 12/12 passed |
| Full repository CTest | `ctest --test-dir BuildTestsDevelopmentMerge --output-on-failure` — 899/899 passed in 81.77 s |
| Production CMake build | `cmake --build Build --parallel 8` — passed; MPI ON, DOUBLE, pinned wrappers confirmed |
| GNUmake build | `make -j8 COMP=gnu USE_MPI=TRUE PRECISION=DOUBLE CXX=<pinned mpicxx> CC=<pinned mpicc>` from `Exec` — passed; `ERF3d.gnu.TEST.MPI.NC.ex` linked |

The full CTest audit retained ordinary ERF, unit, MPI, SBM, P1, P2, AMR,
restart, subcycling, memory, and qualification coverage. The only expected
build warnings were nonfatal Apple-clang warnings about unsupported
`-finline-limit`, ignored `-rdynamic`, and duplicate MPI linkage in the
GNUmake configuration.

Independent numerical and scratch-memory evidence remains in:

```text
/private/tmp/erf_sbm_p2_weno_convergence.csv
/private/tmp/erf_sbm_p2_full_transport_convergence.csv
/private/tmp/erf_sbm_p2_variable_density_convergence.csv
/private/tmp/erf_sbm_p2_chunk_equivalence_memory.csv
```

The canonical direct WENO convergence orders are approximately `1.9833`,
`1.9958`, and `1.9990` after the first refinement pair for the reported
finite-volume diagnostic; the independent grouped operator and variable-
density fixtures remain within their declared bounded-order envelopes. The
chunk-memory evidence shows fixed temporary working storage for fixed chunk
size and complete-group storage scaling with the number of bins.

Documentation checks: `./BuildDocs.sh` from `Docs` passed 15 Python tests,
passed 65 fixed plus 3 dynamic catalog checks, and ran Doxygen. It exited with
status 2 because `sphinx-build` and Graphviz `dot` are unavailable. This is
recorded as an environment limitation, not a documentation-pass claim.

GPU runtime: NOT RUN - no qualified GPU runtime available in this environment

## Restart identities

```text
layout constraint: complete-groups-donor-support-v2
transport: WENO_Z3-group-FCT-v4
AMR transfer: carrier-weighted-mixing-ratio-AMR-v1
boundary policy: periodic+wall+outflow-no-inflow-v1
schema: ERF-SBM-P2-4
```

## Commit separation

Commit A contains only the production implementation, tests, fixtures, CTest
registration, and source trace. Commit B contains only this evidence report.
The four negative controls were temporary source mutations used to demonstrate
that the corresponding regressions are observable; every mutation was
restored before Commit A was created.

## Final disposition

P0 COMPLETE

P1 QUALIFIED

P2 FINAL AMR/CFL CORRECTIONS QUALIFIED FOR THE DECLARED ENVELOPE
P3 READINESS: HOST/TRANSPORT GUARDS AND EVIDENCE COMPLETE; P3 PHYSICS REMAINS
UNIMPLEMENTED

The two commits are ready to be pushed to the feature branch
`sbm-p2-final-amr-cfl-fixes`.
