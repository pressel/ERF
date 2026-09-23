# ERF SBM P0–P2 remediation and pre-PR qualification report

This report records the implementation and qualification of the attached
`ERF SBM P0–P2 Remediation and Pre-PR Qualification` prompt. The warm-aerosol
design specification remains the normative architectural reference:

`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`

P3 physics was not started. The report describes the current working tree,
not a commit or a pushed branch.

## Revision record

| Field | Value |
|---|---|
| Starting HEAD | `d42970517c612f69a525987ece232488940381ca` |
| Final working-tree revision | `d42970517c612f69a525987ece232488940381ca` plus the uncommitted changes described below |
| Branch | `sbm-p2-final-closeout` |
| `origin/development` | `b4eda429ed3c47804666c75c79fae18c94d22c0d` |
| Development relation | Starting HEAD is at the development merge base; no merge was required |
| AMReX submodule | `53fb957f5e136ed8317d584b75edd321c142d1c7` |
| C compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc` |
| C++ compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicxx` |
| MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| Build directory | `BuildTestsDevelopmentMerge` |
| Configuration | Release, DOUBLE, MPI/tests/unit tests ON, AMReX GPU backend NONE |
| Parallel build | `--parallel 8` / `-j8` |
| Commit/push | Not performed |

## Final disposition

The SBM remediation is locally implemented and the complete SBM-labeled suite
passes. The branch is nevertheless not fully qualified for PR review because
the required GPU runtime and memory-safety evidence was not available on this
macOS host, and the full repository matrix has one unrelated pre-existing
MYNNEDMF `SIGILL` failure.

```text
GPU runtime qualification: NOT RUN
Memory-safety qualification: NOT RUN
Full-repository result: 979/980 passed; Closure_BoxParity_MYNNEDMF failed
```

## Claim disposition

| Claim | Disposition | Changed files/symbols | Decisive evidence |
|---|---|---|---|
| F01 predictor spectrum/density pairing | `FIXED_AND_VERIFIED` | `ERF_SBMErfIntegration.cpp` (`sbm_evaluation_density`, stage handoff); `ERF_SBMTransportPrototype.cpp` (`uses_predictor_view`) | `SBM_P2_DYNAMIC_REAL_CARRIERS`; compressible RK3 real-host path passes at 1 and 2 ranks with stage-varying carrier and variable density |
| F02 absolute-time stage duration | `FIXED_AND_VERIFIED` | `ERF_AuxiliaryStageContext.H`; P0/P1 unit tests | `SBMP1.StageIntervalsAreTimeOriginInvariantAndFinite`; translated-origin and invalid-time controls pass |
| F03 duplicate physical-face demand | `FIXED_AND_VERIFIED` | `ERF_SBMTransportPrototype.cpp` grouped demand gathers | `SBM_P2_ACTIVE_MPI`, `SBM_P2_BOUNDARIES_2M`, and full SBM suite pass at 1/2 ranks |
| F04 carrier/property support | `FIXED_AND_VERIFIED` | `ERF_SBMTransportPrototype.cpp` universal support validation and donor-local support accounting | attached-property, zero-carrier/orphan, AMR, restart, and boundary controls pass |
| F05 hidden `qc/qr` writes | `FIXED_AND_VERIFIED` | `ERF_SBMErfIntegration.cpp`, `ERF_DataStruct.H`, `RunSBMP2OwnershipFault.cmake` | `SBM_P2_OWNERSHIP_FAULTS`; `qc_advection`, `qr_diffusion`, and `bulk_clip` all fail before projection |
| F06 indistinguishable spectral edges | `FIXED_AND_VERIFIED` | `ERF_SpectralGrid.cpp`; P0 unit tests | conditioning-scale, zero-origin tiny-width, repeated-edge, and nonfinite controls pass |
| F07 schema identity aliasing | `FIXED_AND_VERIFIED` | `ERF_SBMCanonicalIdentity.H`, `ERF_SBMLayout.cpp`, `ERF_SpectralGrid.cpp` | distinct finite bounds produce distinct identities; NaN is canonicalized as `unbounded`; restart/schema tests pass |

## Root causes and implementation summary

F01 was a temporal-view mismatch: compressible RK stages could consume the
auxiliary predictor spectrum with the step-anchor density. The integration
layer now captures the host density at the matching predictor handoff, and
the transport helper selects predictor spectrum and predictor density
together for every nonzero stage. Coarse/fine stage preparation follows the
same view identity.

F02 accepted stage times through absolute-time arithmetic and allowed an
`output-old` shortcut to substitute for the canonical RK/Heun intervals. The
stage context now validates finite times and full-step values, uses canonical
`{h/3,h/2,h}` and `{h,h}` intervals, and applies scale-aware timestamp checks.

F03 used face-FAB deposits plus boundary accumulation in a way that could
count a shared physical face more than once. Low-order, high-minus-low, and
attached-property demand are now accumulated by a unique cell-centered gather
from the two incident canonical faces; directional contributions accumulate
with `+=`.

F04 now validates finite/nonnegative carrier and property values, the
zero-carrier/orphan rule, and configured hard support bounds at accepted-state
boundaries. Donor-derived support envelopes are checked while donor state is
still available; no clipping, flooring, or orphan deletion was added.

F05 adds a pre-projection ownership residual check and qualification-only fault
injection for compact cloud/rain writes. The normal production ownership
contract is unchanged; projection cannot erase the evidence before the check.

F06 replaces an absolute-width rejection floor with a scale-relative
conditioning rule. A valid interval near zero remains valid, while intervals
smaller than the representable conditioning scale at their own magnitude are
rejected.

F07 uses locale-independent scientific canonicalization with
`max_digits10`, including explicit support kind, transport/remap policy,
support bounds, and canonical unbounded values. Restart mismatch rejection is
not weakened and no migration path was introduced.

## Validation record

### Tests run

| Command | Configuration/rank/backend | Result and important observable |
|---|---|---|
| `/opt/homebrew/bin/cmake --build BuildTestsDevelopmentMerge --parallel 8` | Spack `mpicc`/`mpicxx`, CPU AMReX, Release | Passed |
| `ctest --test-dir BuildTestsDevelopmentMerge -L sbm --output-on-failure --parallel 8` | Spack Open MPI, CPU, SBM labels | **24/24 passed** |
| `ctest --test-dir BuildTestsDevelopmentMerge -R '^SBM_P2_DYNAMIC_REAL_CARRIERS$' --output-on-failure` | RK3 + Heun, 1 and 2 ranks | Passed; rank-equivalent stage-rate vectors |
| `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec -n 2 ./BuildTestsDevelopmentMerge/Tests/Unit/erf_unit_tests --gtest_filter=SBMP2.DonorCellTwoMomentUsesPreparedCoarseFineEndpointDonors --gtest_color=no` | 2 MPI ranks, CPU | Passed |
| `ctest --test-dir BuildTestsDevelopmentMerge --output-on-failure --parallel 8` | Full repository, Spack Open MPI, CPU | **979/980 passed**; only `Closure_BoxParity_MYNNEDMF` failed with macOS `SIGILL` in `ComputeDiffusivityMYNNEDMF` |

The dynamic real-carrier evidence from the final passing run is:

```text
compressible RK3 actual_rate: 0.14510956332980915;0.1632482587460353;0.18138695416226144
compressible RK3 tau*rate:    4.8369854443269716e-05;8.1624129373017646e-05;0.00018138695416226145
anelastic Heun actual_rate:   1.1608765066384732;1.3059860699682824
anelastic Heun tau*rate:      0.00011608765066384732;0.00013059860699682823
1-rank and 2-rank vectors:    identical
```

AMR, restart, dynamic regrid/remake, native factor-2 subcycling, boundaries,
host-CFL, acoustic rejection, donor/grouped transport, and ownership controls
are included in the 24-test SBM result. The CPU unit matrix also passed the
F01–F07 direct and negative-control tests.

### Tests not run or not sufficient

* GPU runtime qualification: **NOT RUN**; this host has no CUDA, HIP, or SYCL
  device/runtime available.
* GPU memory-safety qualification: **NOT RUN**.
* Remote CI status for the current uncommitted tree: not available and not
  substituted with local CPU evidence.
* The attempted GNU Make production build with the pinned Spack wrappers and
  `-j8` remains blocked by the existing missing prerequisite path
  `..//Source/PhysicsInterfaces/Radiation/Simple/ERF_RadiationSimple.cpp`;
  the CMake production build passed.
* Scientific warm-aerosol validation and P3 physics were not run or added.

The full-repository failure is outside the SBM change set: its backtrace is
`ComputeDiffusivityMYNNEDMF -> ComputeTurbulentViscosity ->
ERF::advance_dycore`, and no MYNNEDMF source was modified by this remediation.

## Negative-control record

The following faulty or unsupported variants were demonstrated to fail closed:

* DonorCell and GroupedFCT acoustic-substepping configurations for one/two
  moments and one/two ranks: rejected by the stable P2 capability gate.
* `qc_advection`, `qr_diffusion`, and `bulk_clip` test-only mutations: caught
  by the pre-projection ownership invariant.
* inward advective outflow: rejected because no production spectral inflow
  state exists.
* poisoned physical wall ghosts with positive diffusion: the corrected wall
  path passes without reading them; the direct control distinguishes the
  forbidden read.
* unsafe actual carrier demand, omitted diffusion, wrong stage interval, and
  reconstructed-carrier substitution: direct SBM controls fail closed or
  report the changed demand as required.
* zero-carrier orphan attached property and hard-support violations: accepted
  state validation rejects them.
* repeated/indistinguishable/nonfinite spectral edges: initialization rejects
  them; a scale-valid tiny interval at zero remains accepted.

## Compatibility and support envelope

Non-SBM regression coverage remains green except for the unrelated
`Closure_BoxParity_MYNNEDMF` SIGILL described above. MPI CPU qualification
passed at one and two ranks. AMR and restart qualification passed for the
declared static-Cartesian factor-2 envelope. The build is double precision
with no GPU backend on this host.

The schema identity is intentionally stricter: checkpoints whose layout or
support identity does not match the current canonical schema are rejected.
No schema migration or silent compatibility conversion was added. Supported
SBM configurations are runtime-sized one-/two-moment DonorCell and grouped
FCT/WENO-Z3 with explicit scalar density-weighted diffusion, periodic,
impermeable-wall, or outward-only outflow boundaries, static Cartesian AMR,
restart/regrid, and acoustic-substepping-disabled RK3/Heun. Unsupported
terrain, embedded boundaries, moving geometry, prescribed production
spectral inflow, implicit/tensor/cross diffusion, dynamic spectral grids,
acoustic substepping, and P3 physics remain rejected or out of scope.

## Diff summary

The working tree contains the following tracked implementation/test changes:

```text
Source/AuxiliaryState/ERF_AuxiliaryStageContext.H
Source/DataStructs/ERF_DataStruct.H
Source/ERF.H
Source/Microphysics/SBM/ERF_SBMErfIntegration.cpp
Source/Microphysics/SBM/ERF_SBMLayout.cpp
Source/Microphysics/SBM/ERF_SBMTransportPrototype.cpp
Source/Microphysics/SBM/ERF_SpectralGrid.cpp
Tests/CTestList.cmake
Tests/RunSBMP2AMR.cmake
Tests/Unit/Microphysics/SBM/ERF_GTestSBMP0P1.cpp
Tests/Unit/Microphysics/SBM/ERF_GTestSBMP2.cpp
Tests/Unit/Microphysics/SBM/inputs_sbm_p2_anelastic_heun
Tests/Unit/Microphysics/SBM/inputs_sbm_p2_dynamic_rk
```

New files are `Source/Microphysics/SBM/ERF_SBMCanonicalIdentity.H` and
`Tests/RunSBMP2OwnershipFault.cmake`. Pre-existing untracked build/test
artifacts were preserved and are not part of the implementation.

Current status summary:

```text
git status --short: tracked files above are modified; the pre-existing
untracked Build*/Exec/tmp_* artifacts remain untouched.
git diff --check: passed
git diff --stat: 15 tracked files changed, 688 insertions, 394 deletions
```

No commit, merge, push, design-baseline edit, or destructive cleanup was
performed.

```text
BLOCKED_BEFORE_PR
```
