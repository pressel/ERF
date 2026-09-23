# ERF SBM P0–P2 remediation and corrective-pass qualification report

This report records the corrective pass for the ERF SBM P0–P2 implementation.
The warm-aerosol design specification remains the normative architectural
reference. It is the project-level file
`ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`
at the `ERF_SBM_Public` project root; the local checkout copy is
`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`.

P3 physics was not started.

## Revision record

| Field | Value |
|---|---|
| Starting HEAD for this corrective pass | `5eb23dd04648688ffdf27a0f507bcc669cb61ad8` |
| Implementation/test commit | `224d822d8c85f589dae97c8a115bfdd9f3ff84b6` (`Close out SBM P0-P2 CI and seam qualification`) |
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
| Working-tree state before final docs refresh | Tracked files clean after implementation; pre-existing untracked build/test artifacts preserved |

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
F07 MPI stress:         2 ranks x 20 repetitions passed
F03 active matrix:      12/12 cases passed (internal/seam x 2 diffusion x A/B/C)
Acoustic matrix:         4/4 passed in single-config and Ninja Multi-Config Debug
GPU runtime:            NOT RUN
GPU memory safety:      NOT RUN
Full repository:        NOT RUN in this pass
Remote CI:              source implementation/docs SHA `5a168c5572b75d73ca9e8445c0ebfbb20383e802`: Style, codespell, draft-PDF, and DocHTML passed; substantive workflows were in progress when inspected, with no completed branch-specific failure observed. Subsequent documentation-only follow-ups launched fresh CI sets, still in progress at final inspection
```

## Claim disposition

| Claim | Disposition | Decisive evidence |
|---|---|---|
| F01 predictor spectrum/density pairing | `FIXED_AND_VERIFIED` | `SBM_P2_DYNAMIC_REAL_CARRIERS`: real ERF Advecting Isentropic Vortex carrier, native nonuniform host density, 1/2 ranks, RK3 and Heun; state-evaluation-density mutant fails with the required ratio diagnostic |
| F02 absolute-time stage duration | `FIXED_AND_VERIFIED` | Existing P0/P1 stage-interval and translated-time unit controls pass |
| F03 decomposition, duplicate-face accounting, and periodic seam | `FIXED_AND_VERIFIED` | `SBM_P2_ACTIVE_MPI`: internal-FAB and periodic-seam locations, one-FAB/split-FAB/two-rank layouts, diffusion 0 and 1e-4; all limiter, transfer, budget, projection, total, and qc/qr comparisons agree |
| F04 carrier/property support transactions | `FIXED_AND_VERIFIED` for the supported unit/runtime envelope | Unit transaction test validates attached-property state before and after `remake_level`, then rejects an orphan; runtime attached-property input is not exposed by the current qualification fixtures |
| F05 hidden `qc/qr` writes and mutation surface | `FIXED_AND_VERIFIED` | Ownership fault matrix catches `qc_advection`, `qr_diffusion`, and `bulk_clip`; the public test-hook definition is propagated to consumers, while the tests-disabled executable has no hook macro or `sbm_test_*` strings |
| F06 indistinguishable spectral edges | `FIXED_AND_VERIFIED` | Existing conditioning-scale, tiny-width, repeated-edge, and nonfinite unit controls pass |
| F07 schema identity aliasing | `FIXED_AND_VERIFIED` | Actual schema write/read round trip passes on both ranks for 20 repetitions; invocation-local filenames include `NProcs()` and `MyProc()`, and support maxima `1.0000001` and `1.0000002` compare unequal with `property_identity` mismatch |

## Corrective-pass implementation

F01 now preserves the native ERF density in the positive dynamic-RK fixture
and hands the provider the predictor density captured from the matching ERF
temporal view. The fixture no longer relies on a per-stage manufactured
carrier overwrite. It records predictor/target/evaluation density differences,
density ranges, ratio residual, and the actual carrier/demand used by the
production callback. The negative state-evaluation-density selector is
available only in test builds and is required to fail closed.

F03 now qualifies three layouts: one rank/one FAB, one rank/split FAB, and two
ranks/split FAB. It repeats the matrix with explicit positive diffusion for
both the original internal-FAB face and a distinct periodic-seam fixture. In
the seam mode, only the low and high stored copies of the periodic x face carry
the manufactured positive carrier, and the high-domain donor cell is the
limiting cell. At zero diffusion, the accepted seam face-transfer L1 is
`6.8149061898245816e-09` in all A/B/C layouts. The
accepted face-transfer diagnostics now canonicalize face ownership: a
face-centered `MultiFab` legitimately stores both copies of an internal face
when cell FABs are split, but qualification sums and norms count each physical
face once and omit the duplicate periodic high endpoint. This fixes the
diagnostic layout dependence without changing the conservative production
face storage or update.

F04 adds a transaction-level attached-property validation test across a
`remake_level` layout change, including the explicit universal-support
diagnostic for a zero-carrier/property-positive orphan.

F05 places dynamic regrid, active-limiter, periodic-seam, manufactured-carrier,
density-view, and fault-injection selectors behind
`ERF_SBM_QUALIFICATION_TEST_HOOKS`, which is a `PUBLIC` usage requirement only
when ERF tests or unit tests are enabled. This keeps inline
`SolverChoice::init_params` definitions ODR-consistent in consumers such as
`erf_exec` and `erf_unit_tests`. The tests-disabled production build contains
no `sbm_test_*` strings or hook macro.

F07 exercises the real `write_checkpoint_schema`, `read_checkpoint_schema`,
and `compare_checkpoint_schema` integration rather than only comparing
in-memory objects. The test now gives each MPI invocation a filename containing
the process count and rank, so every rank still performs the complete
write/read/compare/remove sequence without a shared-file race.

CI01 removes only the live unused `heun_corrector` declaration in the stage
entry point; the independently used Heun variables and historical `#if 0`
declaration remain. CI03 registers the checker through
`$<TARGET_FILE:erf_sbm_acoustic_substepping_check>`, which resolves to the
selected configuration directory without a Windows-specific path rule.

## Validation record

| Command | Result |
|---|---|
| `cmake -S . -B BuildTestsDevelopmentMerge && cmake --build BuildTestsDevelopmentMerge --parallel 8` | Passed with the pinned Spack MPI compilers |
| `./BuildTestsDevelopmentMerge/Tests/Unit/erf_unit_tests --gtest_filter='SBMP2.*' --gtest_color=no` | 53/53 passed |
| `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec -n 2 ./BuildTestsDevelopmentMerge/Tests/Unit/erf_unit_tests --gtest_filter=SBMP2.RestartSchemaRoundTripRejectsDistinctFinitePropertySupport --gtest_repeat=20 --gtest_break_on_failure` | 40 rank-level executions passed; no schema-header race |
| `ctest --test-dir BuildTestsDevelopmentMerge -R '^(SBM_P2_ACTIVE_MPI|erf_sbm_p2_mpi|SBM_P2_REJECT_ACOUSTIC_)' --output-on-failure` | 6/6 passed |
| `ctest --test-dir BuildTestsDevelopmentMerge -L sbm --output-on-failure` | 24/24 passed |
| `ctest --test-dir /private/tmp/ERF_sbm_multiconfig -C Debug -R '^SBM_P2_REJECT_ACOUSTIC_' --output-on-failure` | 4/4 passed under Ninja Multi-Config; generated commands resolve `Tests/Debug/erf_sbm_acoustic_substepping_check` |
| Warning check: Spack `mpicxx` compile of `ERF_SBMTransportPrototype.cpp` with `-Wall -Wextra -Werror=unused-variable` and `-Wno-unused-variable` removed | Passed; no `heun_corrector` unused-variable diagnostic |
| Tests-enabled compile flags: `BuildTestsDevelopmentMerge/.../erf_srclib.dir/flags.make`, `erf_exec.dir/flags.make`, `erf_unit_tests.dir/flags.make` | All contain `-DERF_SBM_QUALIFICATION_TEST_HOOKS=1`, demonstrating PUBLIC propagation |
| Tests-disabled CMake configure with Spack `mpicc`/`mpicxx` | Passed in `/private/tmp/ERF_sbm_no_tests` |
| `cmake --build /private/tmp/ERF_sbm_no_tests --target erf_exec --parallel 8` | Passed; production flags and binary contain no qualification hook or `sbm_test_*` strings |
| `ctest --test-dir BuildTestsDevelopmentMerge -R '^(IsentropicVortexAdvecting|DensityCurrent|ScalarAdvectionUniformU)$' --output-on-failure` | 3/3 ordinary ERF regressions passed |
| `git diff --check` | Passed before implementation commit and before docs commit |
| `gh run list --repo pressel/ERF --branch sbm-p2-final-closeout` | Source implementation/docs SHA `5a168c5572b75d73ca9e8445c0ebfbb20383e802`: ancillary checks passed and substantive workflows were in progress when inspected, with no completed branch-specific failure observed; subsequent documentation-only follow-ups launched fresh sets that remained in progress at final inspection |

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

The active-limiter matrix reports all twelve cases as verified:

```text
internal_fab_face, diffusion=0.0:   A(1 rank, max_grid_size 16), B(1, 8), C(2, 8)
internal_fab_face, diffusion=1e-4:  A(1 rank, max_grid_size 16), B(1, 8), C(2, 8)
periodic_seam, diffusion=0.0:       A(1 rank, max_grid_size 16), B(1, 8), C(2, 8)
periodic_seam, diffusion=1e-4:       A(1 rank, max_grid_size 16), B(1, 8), C(2, 8)
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
build. HIP/SYCL compilation and CUDA/HIP/SYCL runtime were not available on
this host. The branch-specific remote CUDA, HIP, and SYCL workflows for the
source-bearing revision were observed but had not completed when inspected;
the documentation-only follow-up also had not completed at final inspection.
No result is claimed for those backends. MYNNEDMF baseline reproduction and
the full repository matrix are `NOT RUN` in this corrective pass. The known
development-side Ubuntu particles-off failure and the earlier un-reproduced
MYNNEDMF report were not modified.

Pre-existing untracked build/test artifacts were preserved and are not part of
the commits. The design specification was not modified.
