# ERF SBM P2 carrier-CFL closure and P3-readiness qualification report

This report records the implementation and qualification evidence requested by
`ERF_SBM_P2_Final_Carrier_CFL_Closure_Codex_Prompt.md`. The warm-aerosol design
specification at
`/Users/pres026/Research/ERF_SBM_Public/ERF_SBM_Warm_Aerosol_Design_and_Implementation_Specification_v1.0.md`
remains the architectural authority. The attached prompt is treated as the
implementation request; its embedded instructions are recorded here as
qualification criteria, not as additional user requests. P3 physics is not
implemented or claimed by this pass.

## Disposition

The remaining host carrier/CFL closure is implemented with Path B from the
prompt. The production path now measures the actual low-order stage demand
from the exact SBM carrier arrays and exact stage density, checks the selected
host timestep before the SBM update, and emits fixed-cost diagnostics. ERF
acoustic substepping is rejected at startup because a general proof tying its
intermediate carrier states to the host estimate is not available. Native
AMR factor-2 subcycling is separate and remains supported.

Focused qualification, 1/2-rank MPI checks, the full P2 matrix, the complete
CTest audit, both production build systems, and the required negative controls
passed. The documentation script completed its available Python/catalog and
Doxygen checks but could not complete Sphinx because `sphinx-build` is not
installed. No qualified GPU runtime is available in this environment.

## Provenance and toolchain

| Field | Value |
|---|---|
| Branch | `sbm-p2-final-carrier-cfl-closure` |
| Required starting SHA | `be4fab25e80abbd225e30200568d904e80d7cee7` |
| Implementation Commit A | `cb5144e60` — `Close P2 carrier CFL qualification` |
| Evidence Commit B | This report-only commit; resolve with `git rev-parse HEAD` |
| Remote | `https://github.com/pressel/ERF` |
| C compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc` |
| C++ compiler | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicxx` |
| MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| MPI runtime | Pinned Spack Open MPI 5.0.9 |
| Precision | Double |
| Build parallelism | `-j8` / `--parallel 8` where supported |

The CMake caches confirm MPI enabled, double precision, and the pinned Spack
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

## Carrier/CFL closure

### Path B: explicit acoustic-substepping rejection

Capability evaluation now records whether any ERF `SubsteppingType` other than
`None` is active. P2 rejects that configuration before time integration with:

```text
P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping
```

This applies to DonorCell and grouped WENO-Z3/FCT transport and to one- and
two-moment modes. It is distinct from the supported native AMR factor-2
subcycling path. There is no private SBM retry or subcycle controller.

### Actual-stage demand oracle

`measure_actual_stage_low_order_demand` consumes the exact `avg_xmom`,
`avg_ymom`, and `avg_zmom` carrier arrays handed to SBM, the exact density
view used by the low-order update, the active geometry, and the production
face-density convention for diffusion. It computes the outgoing advective
demand in all directions and the combined advection-plus-diffusion demand,
including both carrier signs and impermeable-wall handling.

The diagnostic records the maximum advective rate, diffusive rate, combined
rate, `tau*rate`, level, stage, temporal mode, MPI rank count, and worst-cell
metadata. Each stage uses a fixed number of global reductions (two) and no
`O(Nbin)` reduction or scratch allocation. The production handoff passes the
diagnostic result back into the actual-stage admissibility assertion, so the
selected host step is checked against the actual carrier before transport.

Stage durations come directly from `StageContext`: compressible RK3 uses
`h/3`, `h/2`, and `h`; anelastic Heun uses `h` and `h`. These durations are not
the RK ledger weights.

### Host estimate and static-geometry bound

The normal ERF host timestep path remains the conservative static-Cartesian
estimate based on reconstructed `u*rho`, with the existing safety factor of
`0.5`. It is explicitly labeled `host_carrier=reconstructed_u_rho` in the
host diagnostic. The actual-stage oracle is the authoritative closure for the
carrier that SBM truly consumes; if the actual stage is inadmissible, the
production run fails closed before the SBM update.

For the supported static Cartesian geometry, the measured low-order rate is

```text
tau/(rho_i V) * [sum_d outgoing(mdot_d) + sum_faces A_f rho_face K/d_if]
mdot_d = carrier_face,d * rho_face
rho_face = 0.5 * (rho_i + rho_neighbor)
```

Invalid density/carrier data and unsupported geometry fail closed. No identity
bump was needed: the existing transport identity remains
`WENO_Z3-group-FCT-v4` and the restart schema remains `ERF-SBM-P2-4`.

## Focused carrier evidence

The manufactured host initialization records the required discrepancy between
the reconstructed host estimate and the actual SBM carrier:

```text
reconstructed_u_rho=0.098125
actual_avg_xmom=0.125
actual_over_reconstructed=1.2748091603...
```

The direct actual-stage unit oracle covers multiple directions and both signs:

```text
advective_rate=0.4
diffusive_rate=0.45
maximum_rate=0.85
maximum_tau_rate=0.2833333333...
worst_cell=(0,0,0)
```

The variable-density anchor test uses densities `1` and `2` with a negative
x-carrier and verifies that the exact cell density, not a neighboring or
constant density, anchors the update:

```text
advective_rate=0.2
maximum_rate=0.2
maximum_tau_rate=0.2
worst_cell=(1,0,0)
```

Temporal smoke evidence uses the production stage diagnostics for both
qualified modes. Two compressible steps produce six stages with mode
`compressible_rk3`, durations `h/3,h/2,h`, and maximum `tau*rate=0.0001`.
Two anelastic steps produce four stages with mode `anelastic_heun`, durations
`h,h`, and maximum `tau*rate=0.0001`. Acoustic substepping is disabled in both
qualified runs.

The variable-density host-CFL fixture passed identically at one and two MPI
ranks. Its combined advection/diffusion and zero-diffusion evidence is:

```text
host_cfl_zero_rate=4.243243243
host_cfl_zero_mathematical_bound=0.2356687898
host_cfl_zero_selected_dt=0.1178343949
host_cfl_zero_bound=0.1178343949
host_cfl_diffusion_rate=1154.513514
host_cfl_diffusion_advective_rate=4.243243243
host_cfl_diffusion_selected_dt=0.0004330828476
host_cfl_diffusion_bound=0.0004330828476
host_cfl_diff_actual_stage_count=3
host_cfl_diff_actual_stage_max_rate=1155.6756756756756
host_cfl_diff_actual_stage_max_tau_rate=0.50050331249853697
host_cfl_zero_actual_stage_count=3
host_cfl_zero_actual_stage_max_rate=5.4054054054054053
host_cfl_zero_actual_stage_max_tau_rate=0.63694267515923564
host_cfl_old_velocity_only_bound=0.25
host_cfl_old_velocity_only_product=1.06081081075
```

## Required negative controls

All controls were temporary source mutations, each was observed to fail, and
each was restored before Commit A:

| Control | Mutation | Observed failure |
|---|---|---|
| NC-1 | Bypass the corrected host carrier/CFL bound | Variable-density production run failed at the actual-stage admissibility assertion with `tau_rate=1.35135...` |
| NC-2 | Disable the acoustic-substepping capability rejection | Startup rejection test failed because the required capability reason was absent |
| NC-3 | Omit the outgoing y-direction contribution | Direct actual-stage oracle failed: advective rate `0.1` instead of `0.4` |
| NC-4 | Omit diffusion from the combined rate | Direct actual-stage oracle failed: diffusive rate `0` and total `0.4` instead of `0.45` and `0.85` |
| NC-5 | Replace the anchor density with a constant `1` | Variable-density anchor oracle failed: `0.4` instead of `0.2` |

## Test, build, and documentation evidence

All compilation and MPI execution used the pinned Spack wrappers above.

| Check | Result |
|---|---|
| Focused SBM unit tests, 1 rank | 47/47 passed |
| Focused SBM unit tests, 2 ranks | 47/47 passed |
| P1 temporal smoke | Compressible and anelastic passed; actual stage counts 6 and 4 |
| Variable-density host CFL, 1/2 ranks | 2/2 CTest cases passed |
| Acoustic rejection matrix | 4/4 CTest cases passed for Donor/Grouped and 1M/2M |
| Real P2 matrix | `ctest --test-dir BuildTestsDevelopmentMerge -L sbm-p2 --output-on-failure` — 16/16 passed |
| Full repository CTest | `ctest --test-dir BuildTestsDevelopmentMerge --output-on-failure` — 905/905 passed in 83.48 s |
| Production CMake build | `cmake --build Build --parallel 8` — passed; MPI ON, DOUBLE, pinned wrappers |
| GNUmake build | `make -j8 COMP=gnu USE_MPI=TRUE PRECISION=DOUBLE` with pinned `mpicc/mpicxx` — passed; `ERF3d.gnu.TEST.MPI.NC.ex` linked |

The P2 re-audit includes AMR 2M and DonorCell, variable-density AMR, restart,
active MPI, native AMR subcycling, boundaries, host timestep, host CFL,
acoustic rejection, and the direct SBM MPI qualification target. Existing
WENO/FCT, 2M endpoint, restart, and P1 coverage remains in the full 905-test
audit.

Documentation: `./BuildDocs.sh` from `Docs` passed 15 Python tests, passed the
65 fixed and 3 dynamic catalog checks, and ran Doxygen successfully. The script
then exited with status 2 because `sphinx-build` is unavailable. This is an
environment limitation, not a documentation-pass claim.

GPU runtime: NOT RUN — no qualified GPU runtime is available in this
environment.

## Supported and rejected envelope

Supported and qualified:

- static Cartesian, double precision;
- runtime-sized complete one-moment and two-moment liquid population/bin
  groups with physical 2M storage `(M,C)`;
- `DonorCell` and `GroupedFCT_WENOZ3` transport;
- explicit scalar density-weighted diffusion;
- periodic, wall, and outward-only advective-outflow boundaries;
- fully periodic `max_level=1` AMR with spatial ratio `(2,2,2)`;
- native AMR time refinement factor `2`, with two fine substeps per coarse
  step;
- `TwoWay` AMR reflux/feedback and dynamic regrid/remake;
- strict same-decomposition restart equivalence;
- variable dry density; and
- compressible RK3 and anelastic Heun stage modes with acoustic substepping
  disabled.

Rejected or fail-closed:

- ERF acoustic substepping (`SubsteppingType != None`) for SBM P2;
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

## Commit separation and changed artifacts

Commit A contains the production capability gate, actual-stage diagnostic and
assertion, host diagnostic labeling, tests, fixtures, CTest registration, and
source trace. Commit B contains only this evidence report. The five negative
controls were temporary mutations and none remains in the branch. Existing
untracked build artifacts were preserved; no submodule contents or revisions
were changed.

The source trace is maintained in
`Source/Microphysics/SBM/P2_SOURCE_TRACE.md` and records the exact handoff,
density, stage-duration, reduction, and rejection paths.

## Final disposition

```text
P0 COMPLETE
P1 QUALIFIED
P2 FULLY QUALIFIED FOR THE DECLARED SUPPORTED CONFIGURATION
READY TO BEGIN P3
```

This disposition means P2 infrastructure and qualification are complete for
the declared envelope. P3 physics remains unimplemented and is the next
development phase.
