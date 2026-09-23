# ERF SBM P0–P2 remediation and corrective-pass qualification report

This report records the corrective pass for the ERF SBM P0–P2 implementation.
The warm-aerosol design specification remains the normative architectural
reference:

`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`

P3 physics was not started.

## Revision record

| Field | Value |
|---|---|
| Starting HEAD | `8aea39e633932909453c504e1763fa291192bec0` |
| Implementation/test commit | `54feca77060b100bd842d3142ce627a14c92ba8e` |
| Branch | `sbm-p2-final-closeout` |
| `origin/development` | `b4eda429ed3c47804666c75c79fae18c94d22c0d` |
| Development relation | Development baseline is already in branch history; no additional merge was required for this pass |
| AMReX submodule | `53fb957f5e136ed8317d584b75edd321c142d1c7` |
| C compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc` |
| C++ compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicxx` |
| MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| Build directory | `BuildTestsDevelopmentMerge` |
| Configuration | Release, DOUBLE, MPI, tests/unit tests ON, CPU AMReX backend |
| Parallel build | `cmake --build ... --parallel 8` |
| Documentation commit | The following docs-only commit records this report and the source trace |

## Final disposition

All in-scope local CPU SBM qualification evidence is green at the
implementation commit. The branch is not claiming GPU qualification: this
macOS host has no CUDA, HIP, or SYCL runtime/device available. A full 982-test
repository run was not repeated in this corrective pass; no current
full-repository pass/fail result is claimed. The earlier MYNNEDMF `SIGILL`
finding was not re-run here and remains an external follow-up, not evidence
against this SBM pass.

```text
SBM unit qualification: 53/53 passed
SBM CTest label:        24/24 passed
GPU runtime:            NOT RUN
GPU memory safety:      NOT RUN
Full repository:        NOT RUN in this pass
Remote CI:              pending push/CI execution
```

## Claim disposition

| Claim | Disposition | Decisive evidence |
|---|---|---|
| F01 predictor spectrum/density pairing | `FIXED_AND_VERIFIED` | `SBM_P2_DYNAMIC_REAL_CARRIERS`: real ERF Advecting Isentropic Vortex carrier, native nonuniform host density, 1/2 ranks, RK3 and Heun; state-evaluation-density mutant fails with the required ratio diagnostic |
| F02 absolute-time stage duration | `FIXED_AND_VERIFIED` | Existing P0/P1 stage-interval and translated-time unit controls pass |
| F03 active limiter/decomposition and duplicate-face accounting | `FIXED_AND_VERIFIED` | `SBM_P2_ACTIVE_MPI`: one-FAB, split-FAB, and two-rank layouts at diffusion 0 and 1e-4; limiter, per-component transfers, budgets, projections, totals, and qc/qr diagnostics agree |
| F04 carrier/property support transactions | `FIXED_AND_VERIFIED` for the supported unit/runtime envelope | Unit transaction test validates attached-property state before and after `remake_level`, then rejects an orphan; runtime attached-property input is not exposed by the current qualification fixtures |
| F05 hidden `qc/qr` writes and mutation surface | `FIXED_AND_VERIFIED` | Ownership fault matrix catches `qc_advection`, `qr_diffusion`, and `bulk_clip`; test-only controls are compile-guarded and absent from the tests-disabled executable |
| F06 indistinguishable spectral edges | `FIXED_AND_VERIFIED` | Existing conditioning-scale, tiny-width, repeated-edge, and nonfinite unit controls pass |
| F07 schema identity aliasing | `FIXED_AND_VERIFIED` | Actual schema write/read round trip passes; support maxima `1.0000001` and `1.0000002` compare unequal with `property_identity` mismatch |

## Corrective-pass implementation

F01 now preserves the native ERF density in the positive dynamic-RK fixture
and hands the provider the predictor density captured from the matching ERF
temporal view. The fixture no longer relies on a per-stage manufactured
carrier overwrite. It records predictor/target/evaluation density differences,
density ranges, ratio residual, and the actual carrier/demand used by the
production callback. The negative state-evaluation-density selector is
available only in test builds and is required to fail closed.

F03 now qualifies three layouts: one rank/one FAB, one rank/split FAB, and two
ranks/split FAB. It repeats the matrix with explicit positive diffusion. The
accepted face-transfer diagnostics now canonicalize face ownership: a
face-centered `MultiFab` legitimately stores both copies of an internal face
when cell FABs are split, but qualification sums and norms count each physical
face once and omit the duplicate periodic high endpoint. This fixes the
diagnostic layout dependence without changing the conservative production
face storage or update.

F04 adds a transaction-level attached-property validation test across a
`remake_level` layout change, including the explicit universal-support
diagnostic for a zero-carrier/property-positive orphan.

F05 places dynamic regrid, active-limiter, manufactured-carrier, density-view,
and fault-injection selectors behind `ERF_SBM_QUALIFICATION_TEST_HOOKS`, which
is defined only when ERF tests or unit tests are enabled. The tests-disabled
production build contains no `sbm_test_*` strings or hook macro.

F07 exercises the real `write_checkpoint_schema`, `read_checkpoint_schema`,
and `compare_checkpoint_schema` integration rather than only comparing
in-memory objects.

## Validation record

| Command | Result |
|---|---|
| `cmake --build BuildTestsDevelopmentMerge --parallel 8` | Passed with the pinned Spack MPI compilers |
| `./BuildTestsDevelopmentMerge/Tests/Unit/erf_unit_tests --gtest_filter='SBMP2.*' --gtest_color=no` | 53/53 passed |
| `ctest --test-dir BuildTestsDevelopmentMerge -L sbm --output-on-failure` | 24/24 passed |
| `ctest --test-dir BuildTestsDevelopmentMerge -R '^(SBM_P2_DYNAMIC_REAL_CARRIERS|SBM_P2_ACTIVE_MPI)$' --output-on-failure` | Both passed; the full label subsequently passed |
| Tests-disabled CMake configure with Spack `mpicc`/`mpicxx` | Passed in `/private/tmp/ERF_sbm_no_tests` |
| `cmake --build /private/tmp/ERF_sbm_no_tests --parallel 8` | Passed; production `erf_exec` has no qualification-hook strings |
| `git diff --check` | Passed before implementation commit and before docs commit |

The final dynamic-RK summary is:

```text
compressible RK3 actual carrier demand:
  63.56478522503771;63.152441507123655;62.871602268658961
compressible RK3 tau*rate:
  0.16666666666666669;0.2483782541054208;0.49454742941453605
anelastic Heun actual carrier demand:
  1.1608765066384732;1.3059860699682824
anelastic Heun tau*rate:
  0.00011608765066384732;0.00013059860699682823
1-rank and 2-rank vectors: identical
F01 density view: verified
F01 negative ratio mutant: verified
```

The active-limiter matrix reports all six cases as verified:

```text
diffusion=0.0:   A(1 rank, max_grid_size 16), B(1, 8), C(2, 8)
diffusion=1e-4:  A(1 rank, max_grid_size 16), B(1, 8), C(2, 8)
```

## Negative-control record

The complete SBM label also passes the acoustic-substepping rejection matrix
for DonorCell and GroupedFCT_WENOZ3, one and two moments. The ownership fault
matrix catches all three retained mutations. Direct unit controls continue to
cover inward outflow, poisoned wall ghosts with positive diffusion, invalid
carrier demand, omitted diffusion, wrong stage interval, reconstructed-carrier
substitution, zero-carrier attached-property orphans, hard support violations,
and invalid spectral edges.

## Scope and outstanding evidence

The qualified envelope remains double-precision CPU ERF with MPI, grouped
FCT/WENO-Z3 or DonorCell, explicit scalar density-weighted diffusion,
periodic/impermeable-wall/outward-only-outflow boundaries, static Cartesian
AMR, restart/regrid, native factor-2 time subcycling, and acoustic
substepping disabled. Terrain, embedded boundaries, moving geometry,
prescribed production spectral inflow, implicit/tensor/cross diffusion,
dynamic spectral grids, and P3 physics remain rejected or out of scope.

GPU runtime and memory qualification are `NOT RUN`, not inferred from the CPU
build. MYNNEDMF baseline reproduction and the full repository matrix are
`NOT RUN` in this corrective pass. Remote CI is to be inspected after the
implementation and docs commits are pushed.

Pre-existing untracked build/test artifacts were preserved and are not part of
the commits. The design specification was not modified.
