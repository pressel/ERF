# SBM migration provenance: `sbm-native-transport-v1`

## Immutable source identities

| Item | Identity |
|---|---|
| Development branch | `sbm-native-transport-v1` |
| Exact ERF base | `78707d33f32212c83f612d62c2f86723a0651d62` |
| M0/M1 foundation commit | `2d64a058159e61b061881c601b7fa7dd0e70ad72` (audited parent; unchanged) |
| PR #4096 merge | `966c6f54aae9b83a818147b98ce93b3b90519b15` (verified ancestor of base) |
| Historical qualified Cartesian SBM oracle | `7f5742d9ac1edb3e6eca1be4f8c0f2d357b0781a` (object verified locally; not checked out or modified) |
| AMReX submodule at base | `53fb957f5e136ed8317d584b75edd321c142d1c7` |
| ERF remote observed before branch creation | `origin https://github.com/pressel/ERF` (fetch and push) |

Fetching `origin` was attempted before branch creation, but the environment could not resolve `github.com`. The exact base, #4096 ancestry, and historical oracle objects were already present locally and verified. The Noah-MP submodule worktree was observed clean but checked out at `e03243d2c5eac19b1c6a02a2c3ca12f03f92cceb`, while the base expects gitlink `ca03320921afb319315f8764b174f473ca3d77cc`. This pre-existing gitlink difference is outside this migration; Noah-MP is treated as read-only and must not be staged in an SBM commit.

## Design and mathematics artifacts

SHA-256 hashes were computed from the supplied files in `/Users/pres026/Research/ERF_SBM_Public` before implementation.

| File | SHA-256 |
|---|---|
| `ERF_SBM_Warm_Rain_Design_and_Completion_Plan_v2.1.md` | `8908144cf9746ba7d540e60164af0ef8db661dcaa8aed97594b388f34a55b9ba` |
| `ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md` | `a0fc579a65c66b18acbf98761eda0796f52e181a2ca57eb52fe2804f48fa6272` |
| `ERF_SBM_Warm_Rain_Design_Math_Checks_v2.1.py` | `4c7ad7a940cabb56022ae3206fb2967dcc32c52f4ea30f8b60cc6c2e3d360730` |
| `ERF_SBM_Warm_Rain_Math_Audit_v2.1.json` | `fc5010fc637346e7ac0f950c7a0ecd79afe3f5bb779ece3538bd0bb4e031ceab` |

The supplied prompt names the warm-aerosol Markdown as `...v1.0(3).md`; the available local Markdown copy used here is `ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`. The design markdown is authoritative for architecture and milestone scope. The math script and JSON are audit evidence, not ERF runtime qualification.

## Migration ledger

| Disposition | Material |
|---|---|
| Port nearly unchanged | `ERF_SpectralGrid.*`, `ERF_SBMLayout.*`, `ERF_SBMConstraintGroups.*`, `ERF_SBMBulkProjection.*`, `ERF_AuxiliaryStateLayout.H`, `ERF_AuxiliaryProjection.*`, `ERF_SBMCanonicalIdentity.H`, and the semantic no-op compact adapter. Compatibility changes are limited to the current ERF APIs. |
| Port conceptually / redesign | State-only auxiliary allocation and level lifetime; focused `qc/qr` write ownership; exact-schema restart validation; SBM selection and zero-transport fixture integration. No stage views, flux ledgers, mapped AMR, or transport services are included. |
| Oracle / test only | Historical commit `7f5742d9ac1edb3e6eca1be4f8c0f2d357b0781a`; focused layout, constraint, projection, ownership, and restart cases may be adapted as tests. Historical qualification reports remain evidence about that commit only. |
| Retired / not ported | Historical Cartesian transport prototype, private diffusion, Cartesian FCT, WENO transport, stage/face-transfer machinery, production AMR/reflux, boundary transport, CFL helpers, and all warm-process physics. These are outside M0/M1 or tied to superseded transport assumptions. |

The historical private transport, WENO, FCT, diffusion, Cartesian divergence, stage/face-transfer logic, and old AMR machinery are **not** the production foundation of this branch. No historical SBM branch was merged, rebased, or rewritten.

## Baseline at the exact base

The baseline was configured in a fresh temporary build, using DOUBLE precision, 3D CPU (`AMReX_GPU_BACKEND=NONE`), and MPI. The existing `BuildTests*` directories were not used because their generated CTest inventory retained obsolete SBMP0/P1/P2 tests.

Command used for the focused semantic host baseline:

```text
/private/tmp/erf_sbm_m0m1_baseline/Tests/Unit/erf_unit_tests --gtest_filter=ScalarAdvectionPrimitives.*:ScalarDiffusionPrimitives.*
```

Result: 22/22 tests passed (4 scalar-advection primitive tests and 18 scalar-diffusion primitive tests). No baseline regression references were changed.

## M0/M1 hardening evidence at this commit

The hardening was built and exercised in `/private/tmp/erf_sbm_m0m1_baseline`, configured for ERF `DOUBLE`, 3D, CPU (`AMReX_GPU_BACKEND=NONE`), with MPI enabled, on macOS using AppleClang 21.0.0.21000334 and CMake 4.2.1.

| Evidence | Result |
|---|---|
| `cmake --build /private/tmp/erf_sbm_m0m1_baseline --target erf_exec erf_unit_tests -j 6` | Passed |
| `Tests/Unit/erf_unit_tests --gtest_filter=SBMFoundation.*` | 10/10 passed, including finite/nonfinite restart corruption, full-range ownership, 2M runtime constraints, and mass-only 2M projection |
| `Tests/Unit/erf_unit_tests --gtest_filter=ScalarAdvectionPrimitives.*:ScalarDiffusionPrimitives.*` | 22/22 passed (4 advection, 18 diffusion) |
| `ctest --test-dir /private/tmp/erf_sbm_m0m1_baseline --output-on-failure -R '^SBM_'` | 9/9 passed: 1M and 2M checkpoint/restart plus seven diagnostic-checked expected failures |
| `git diff --check` | Passed before commit |

The ERF integration cases run the local CPU executable with one MPI process. The build has MPI enabled; no multi-rank M1 runtime case was executed. No CUDA, HIP, or SYCL build/runtime was run, and no accelerator runtime evidence is claimed. Hosted CI was not queried or instantiated for this local hardening revision.

The required pre-write checkpoint projection comparison remains deferred: the restart-side finite-safe check is reused and covered, but no new checkpoint-side stage/synchronization semantics were added.
