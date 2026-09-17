# ERF SBM P2 final correction and P3-readiness qualification report

This report records the evidence for the implementation requested by
`ERF_SBM_P2_Final_Correction_and_P3_Readiness_Codex_Prompt.md`. The warm-aerosol
design specification remains the architectural authority. P3 physics is not
implemented or claimed here.

## Qualification disposition

The Commit A implementation is qualified for the bounded supported envelope
below. Focused SBM tests, the real P2 matrix, the full CTest suite, the CMake
production build, and the GNUmake build passed. The complete documentation
pipeline could not run because `sphinx-build` and Graphviz `dot` are absent in
this environment; its available Python/catalog checks passed. No qualified GPU
runtime is available here.

## Git, provenance, and toolchain

| Field | Value |
|---|---|
| Branch | `sbm-p2-final-qualification` |
| Required starting SHA | `955c96cbbbc608caf6c6537db30859f107cc5231` |
| Implementation Commit A | `0afa5bc319094db24d6d1d05908af0b7d061bf08` |
| Final evidence Commit B | This evidence-only commit; resolve with `git rev-parse HEAD` after commit |
| Commits added | 2: Commit A implementation, Commit B evidence report |
| Remote | `https://github.com/pressel/ERF` |
| C compiler wrapper | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc` |
| C++ compiler wrapper | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicxx` |
| MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| Build parallelism | `-j8` / `--parallel 8` where supported |
| MPI/toolchain context | Pinned Spack Open MPI 5.0.9 wrappers/runtime (`mpicc --showme` resolves to Open MPI include/lib paths) |

The production CMake cache confirms MPI is enabled, double precision is
selected, and both compiler wrappers are the pinned Spack paths. The GNUmake
link also used the pinned `mpicc`/`mpicxx` wrappers. Existing untracked build
artifacts were preserved; after Commit B the tracked worktree is clean.

Submodules were left at the required revisions:

| Submodule | Revision |
|---|---|
| AMReX | `e60cdc18711ccf7fc0616d7a2fdf062021376976` |
| Noah-MP | `e03243d2c5eac19b1c6a02a2c3ca12f03f92cceb` |
| RRTMGP | `71693e7490db0908a625eadb6ac3f23967de2c7d` |
| WW3 | `f5fa5822849e33976cdc935a91b304c60bc2a976` |
| ekat | `10ab9d34ceba4eead542702a29676adf77ad91af` |
| googletest | `52eb8108c5bdec04579160ae17225d66034bd723` |

## Declared supported P2 envelope

Supported:

- static Cartesian, double precision;
- runtime-sized one-moment and two-moment liquid state, with physical
  two-moment storage `(M,C)`;
- complete atomic population/bin groups;
- `DonorCell` and `GroupedFCT_WENOZ3` transport;
- explicit scalar density-weighted diffusion;
- single-level periodic, wall, and outward-only advective-outflow boundaries;
- fully periodic `max_level=1` AMR;
- spatial refinement ratio exactly `(2,2,2)`;
- native time refinement factor exactly `2`, with two fine substeps per coarse
  step;
- `TwoWay` AMR reflux/feedback;
- dynamic regrid/remake;
- strict same-decomposition restart equivalence; and
- variable dry density.

Rejected as unsupported, with fail-closed capability checks where applicable:

- `max_level > 1`;
- spatial refinement ratios other than `(2,2,2)`;
- native time factors other than `2`;
- multilevel `OneWay` AMR;
- non-periodic AMR;
- prescribed production spectral inflow;
- implicit/native moisture diffusion;
- tensor or cross diffusion;
- terrain or embedded boundaries;
- dynamic spectral grids;
- schema conversion; and
- P3 condensation/evaporation, activation/regeneration,
  collision/coalescence, sedimentation, aerosol lifecycle, ice, and other
  warm-cloud/macrophysics physics.

## Commit A implementation corrections

- Fixed native-subcycling capability validation so every AMR level must use
  exactly two native substeps.
- Corrected AMR auxiliary-state view sequencing. Prolong/remake fills valid
  cells, while stage fills update only grown ghost cells and preserve the
  provider-owned valid data used for WENO interface transfer.
- Added per-component physical interface scales so reflux/interface checks do
  not compare cancelling integrated totals against an order-one tolerance.
- Kept WENO-Z3 epsilon translation-invariant with a precision-specific
  absolute floor and stencil smoothness scaling; no solution-amplitude branch
  changes the operator.
- Made `rho_target` explicit through the production transport API and corrected
  density selection for compressible RK3 and anelastic Heun temporal contracts.
  The production target-density test verifies `U = k*rho_target` for all
  compressible and anelastic stages without an order-one tolerance floor.
- Added a variable-density production AMR fixture. It preserves the constant
  mixing ratio under nonuniform dry density and checks the ratio at machine
  scale.
- Added real two-rank DonorCell AMR and native-subcycle fixtures, plus the host
  timestep fixture with combined advection/diffusion demand.
- Kept restart identities strict and updated them to the final P2 identities.
- Added the required implementation source trace in
  `Source/Microphysics/SBM/P2_SOURCE_TRACE.md`.

Commit A changed these tracked files:

```
Source/AuxiliaryState/ERF_AuxiliaryStateManager.H
Source/AuxiliaryState/ERF_AuxiliaryStateManager.cpp
Source/DataStructs/ERF_DataStruct.H
Source/ERF.H
Source/ERF.cpp
Source/ERF_MakeNewLevel.cpp
Source/Microphysics/SBM/ERF_SBMContracts.H
Source/Microphysics/SBM/ERF_SBMContracts.cpp
Source/Microphysics/SBM/ERF_SBMErfIntegration.cpp
Source/Microphysics/SBM/ERF_SBMRestart.H
Source/Microphysics/SBM/ERF_SBMRestart.cpp
Source/Microphysics/SBM/ERF_SBMTransportPrototype.H
Source/Microphysics/SBM/ERF_SBMTransportPrototype.cpp
Source/Microphysics/SBM/P2_SOURCE_TRACE.md
Source/TimeIntegration/ERF_ComputeTimestep.cpp
Tests/CTestList.cmake
Tests/RunSBMP2AMR.cmake
Tests/RunSBMP2AMRSubcycle.cmake
Tests/RunSBMP2Timestep.cmake
Tests/Unit/Microphysics/SBM/ERF_GTestSBMP2.cpp
Tests/Unit/Microphysics/SBM/inputs_sbm_p2_timestep
```

Commit B changes only this report.

## Claim ledger

| Claim | Status | Evidence |
|---|---|---|
| `P2F-WENO-01` | `SURVIVES_REVIEW / PASS` | `SBMP2.WENOZ3TranslationCovarianceForSmoothAndDiscontinuousStencils` passes the `64*epsilon` assertion for smooth/jump data and both carrier signs. The independent smooth FV oracle is third-order for WENO and first-order for DonorCell; see numerical evidence below. The unit test does not emit a separate aggregate maximum. |
| `P2F-WENO-02` | `SURVIVES_REVIEW / PASS` | `SBMP2.CoarseFineWENOInterfaceOracleUsesBothUpwindSigns` and `SBMP2.FiniteVolumeWENOQuadraticOptimalCandidateIsExact` pass; all 39 focused `SBMP2.*` tests pass at one and two MPI ranks. |
| `P2F-AMRVIEW-01` | `REVISED_AND_VERIFIED` | Production Old/evaluation FillPatch sequencing is explicit in the source trace; coarse/fine interface tests and real grouped AMR tests pass. |
| `P2F-AMRRHO-01` | `REVISED_AND_VERIFIED` | Carrier-weighted volume/refinement transfer and the variable-density real AMR fixture pass. The maximum reported ratio error is `8.4703294725430034e-22`, below tolerance `9.094947017729282e-19`. |
| `P2F-BULKGHOST-01` | `REVISED_AND_VERIFIED` | Compact ghost projection and bulk ghost tests pass. The variable-density composite reports `compact_projection_error=9.3173624197973037e-21`. |
| `P2F-DONORAMR-01` | `REVISED_AND_VERIFIED` | `SBM_P2_AMR_DONOR_2M` and `SBM_P2_AMR_DONOR_SUBCYCLE_2M` pass in the real P2 matrix. |
| `P2F-CAP-01` | `REVISED_AND_VERIFIED` | Runtime negative capability matrix rejects unsupported refinement, boundary, diffusion, and P3 configurations; the complete real P2 matrix passes. |
| `P2F-DT-01` | `REVISED_AND_VERIFIED` | Two-rank host fixture reports diffusion bound/selected `0.001724137931`, zero-diffusion bound `0.25`, selected `0.005`, and verifies combined advection/diffusion demand. |
| `P2F-RHOTARGET-01` | `REVISED_AND_VERIFIED` | `SBMP2.ProductionTargetDensityPreservesConstantRatioForBothTemporalContracts` passes at one and two MPI ranks for all compressible RK3 and anelastic Heun stages. |
| `P2F-REFLUX-01` | `REVISED_AND_VERIFIED` | Real AMR reports `post_reflux_validation_count=4` and `post_reflux_material_rejection_count=0`; `SBMP2.PostRefluxFailsClosedOnMaterialMismatch` passes. |
| `P2F-SCHEMA-01` | `REVISED_AND_VERIFIED` | Strict restart equivalence passes, and old/mismatched identities reject. Final IDs are recorded below. |
| `P2F-REG-01` | `SURVIVES_REVIEW / PASS` | Full CTest retains ordinary ERF and non-SBM coverage: 891/891 tests pass on Commit A. |
| `P2F-QUAL-01` | `REVISED_AND_VERIFIED` | This report states the exact bounded envelope, records all available numerical/build evidence, and explicitly records unavailable documentation/GPU validation. |

## Numerical and memory evidence

The independent FV convergence file is
`/private/tmp/erf_sbm_p2_weno_convergence.csv`. WENO-Z3 uses the exact face
value oracle for the cell-average primitive `2+sin(2*pi*x)`; positive and
negative carrier signs agree:

| N | WENO + error | WENO - error | Donor error | WENO + order | WENO - order | Donor order |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4.9183763132e-3 | 4.9183763132e-3 | 1.9383917874e-1 | -- | -- | -- |
| 32 | 6.2678263759e-4 | 6.2678263770e-4 | 9.7859763326e-2 | 2.9721450010 | 2.9721450010 | 0.9860724988 |
| 64 | 7.8726462753e-5 | 7.8726462754e-5 | 4.9047971357e-2 | 2.9930446528 | 2.9930446528 | 0.9965223264 |
| 128 | 9.8526722456e-6 | 9.8526722458e-6 | 2.4538767037e-2 | 2.9982616871 | 2.9982616871 | 0.9991308436 |

The independent production semidiscrete/operator file is
`/private/tmp/erf_sbm_p2_full_transport_convergence.csv`. WENO orders are
`2.9771699548, 2.9922082879, 2.9897310309`; DonorCell orders are
`0.9860724988, 0.9965223264, 0.9991308436`. These are smooth spatial/operator
results and do not claim nonlinear ERF time-integration order.

The variable-density WENO file is
`/private/tmp/erf_sbm_p2_variable_density_convergence.csv`. Its observed orders
are `2.5372071374, 2.4113951096, 2.2630051710`, bounded approximately
second-order on the finest pairs.

The chunk-equivalence file is
`/private/tmp/erf_sbm_p2_chunk_equivalence_memory.csv`. Temporary working bytes
are independent of total bin count for a fixed chunk size, while all-group
storage scales with complete groups:

| Moment | Bins | chunk 1 | chunk 2 | chunk 4 | all groups |
|---|---:|---:|---:|---:|---:|
| 1M | 4 | 15,808 | 31,616 | 63,232 | 63,232 |
| 1M | 16 | 15,808 | 31,616 | 63,232 | 252,928 |
| 1M | 64 | 15,808 | 31,616 | 63,232 | 1,011,712 |
| 2M | 4 | 34,176 | 68,352 | 136,704 | 136,704 |
| 2M | 16 | 34,176 | 68,352 | 136,704 | 546,816 |
| 2M | 64 | 34,176 | 68,352 | 136,704 | 2,187,264 |

The latest variable-density composite diagnostic reports:

```
composite_error=0
composite_tolerance_factor=9.9999999999999998e-13
compact_projection_error=9.3173624197973037e-21
accepted_face_projection_error=0
variable_density_ratio_error=8.4703294725430034e-22
variable_density_ratio_tolerance=9.094947017729282e-19
interface_oracle_passed=1
post_reflux_validation_count=4
post_reflux_material_rejection_count=0
```

## Validation commands and results

All commands below were run from the ERF repository and used the pinned Spack
wrappers where compilation or MPI execution was involved.

| Check | Exact command/result |
|---|---|
| CMake test configure | `cmake -S . -B BuildTestsDevelopmentMerge -DCMAKE_C_COMPILER=/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc -DCMAKE_CXX_COMPILER=/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicxx -DERF_ENABLE_MPI=ON -DERF_ENABLE_TESTS=ON -DERF_PRECISION=DOUBLE` — passed |
| CMake test build | `cmake --build BuildTestsDevelopmentMerge --parallel 8 --target erf_exec erf_unit_tests` — passed |
| Focused SBM tests, 1 rank | `ctest --test-dir BuildTestsDevelopmentMerge -R '^SBMP2\.' --output-on-failure` — 39/39 passed |
| Focused SBM tests, 2 ranks | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec -n 2 ./Tests/Unit/erf_unit_tests '--gtest_filter=SBMP2.*'` — 39/39 passed |
| Real P2 matrix | `ctest --test-dir BuildTestsDevelopmentMerge -L sbm-p2 --output-on-failure` — 10/10 passed |
| Full CTest | `ctest --test-dir BuildTestsDevelopmentMerge --output-on-failure` — 891/891 passed |
| Production CMake build | `cmake --build Build --parallel 8` — passed; MPI ON, DOUBLE, pinned `mpicc`/`mpicxx` confirmed in `Build/CMakeCache.txt` |
| GNUmake build | `make -j8 COMP=gnu USE_MPI=TRUE PRECISION=DOUBLE CXX=/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicxx CC=/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc` from `Exec` — passed; `ERF3d.gnu.TEST.MPI.NC.ex` linked |
| Documentation checks | `./BuildDocs.sh` from `Docs` — 15 Python unit tests passed; 65 fixed and 3 dynamic catalog checks passed; Doxygen ran, then script exited 2 because `sphinx-build` is absent and Graphviz `dot` is unavailable |

The GNUmake build emitted nonfatal Apple-clang warnings for unsupported
`-finline-limit`, ignored `-rdynamic`, and duplicate `-lmpi`; the target linked
successfully. The documentation limitation is environmental and is not being
represented as a completed full Sphinx/Doxygen/Graphviz build.

GPU runtime: NOT RUN - no qualified GPU runtime available in this environment

## Final P2 identities

```
layout constraint: complete-groups-donor-support-v2
transport: WENO_Z3-group-FCT-v3
AMR transfer: carrier-weighted-mixing-ratio-AMR-v1
boundary policy: periodic+wall+outflow-no-inflow-v1
schema: ERF-SBM-P2-3
```

## Residual risk and disposition

The remaining risk is intentionally bounded to the rejected P3 and unsupported
runtime/configuration cases listed above. This work does not expand P2 into
warm-cloud microphysics, and it does not qualify GPU execution.

P0 COMPLETE
P1 QUALIFIED
P2 FULLY QUALIFIED FOR THE DECLARED SUPPORTED CONFIGURATION
READY TO BEGIN P3

## Review link

`https://github.com/pressel/ERF/tree/sbm-p2-final-qualification`
