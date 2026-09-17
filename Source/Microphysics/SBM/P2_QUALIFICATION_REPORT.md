# ERF SBM P2 qualification report

This is the evidence ledger for the P2 transport/lifecycle implementation. The
warm-aerosol design specification is the architectural authority. This report
distinguishes implemented and qualified infrastructure from P3 physics, which
is not added here.

## Qualification disposition

Every G0--G6 gate, the full CTest matrix, the required CMake/GNUmake builds,
and the documentation build passed on the current source. This is a frozen
P2 transport/lifecycle qualification for the declared supported configuration;
it is not a claim of production warm-cloud microphysics or GPU performance.

## Git and toolchain

| Field | Value |
|---|---|
| Branch | `sbm-p2-final-qualification` |
| Required starting SHA | `ff96ba02c2a67d1aa7b8ab01f842e0ead87071e3` |
| Final implementation SHA | `9c5e60925` |
| Remote | `https://github.com/pressel/ERF.git` |
| Compiler wrappers | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpicc` and `mpicxx` |
| MPI launcher | `/Users/pres026/Spack/var/spack/environments/erf-fresh/.spack-env/view/bin/mpiexec` |
| Parallel build policy | `-j8` / `--parallel 8` |

All compilation and MPI execution evidence in this report uses the pinned
Spack environment. Existing untracked build artifacts are preserved.

## Declared supported P2 matrix

The intended qualification envelope is:

* static Cartesian, double precision;
* runtime-sized one- or two-moment liquid state, with physical two-moment
  storage `(M,C)`;
* `DonorCell` and production `GroupedFCT_WENOZ3` transport;
* complete atomic population/bin groups, including attached-property and
  donor-derived support constraints;
* compressible RK3 and anelastic Heun stage contracts;
* explicit orthogonal density-weighted diffusion;
* fully periodic AMR with ERF native refinement-ratio-two subcycling;
* single-level impermeable-wall and outward-only advective-outflow boundaries;
* strict same-decomposition restart equivalence.

The P2 boundary policy rejects prescribed production spectral inflow. The
following remain outside the supported envelope and fail closed: non-periodic
AMR, implicit/native moisture diffusion, tensor or non-orthogonal diffusion,
moving terrain, embedded boundaries, dynamic spectral grids, schema conversion,
SHOC/macrophysics coupling, and all condensation/evaporation,
activation/regeneration, collision/coalescence, sedimentation, aerosol
lifecycle, ice, and other P3 physics.

## Implemented corrections

* Authoritative spectral state is separate from ERF compact `cons`; `qc` and
  `qr` are projections only.
* The production grouped path uses complete atomic groups and one common face
  limiter. It evaluates the exact FCT algebra
  `A = high_adv - low_adv`, `low_total = low_adv + low_diff`, and
  `accepted = low_total + lambda*A`.
* Low-order admissibility is evaluated against the correct pre-transfer
  temporal baseline for compressible RK3 and both Heun stage contracts.
* High-order transport is local finite-volume WENO-Z3 on `X/rho`, with two
  ghost layers and actual ERF carrier mass fluxes. The boundary-adjacent
  candidate falls back to the low-order advection candidate.
* Donor-derived support bounds use low sources, actual upwind donors, active
  diffusion neighbors, the Heun old state, and the boundary donor. Hard
  metadata is intersected, chunk-locally; no universal upper bound is invented.
  Zero carrier requires zero extensive state, and invalid density fails closed.
* Physical boundaries are mapped internally from ERF configuration. Walls have
  zero advection/diffusion; outflow uses an interior donor, zero diffusion, and
  rejects inward carrier flux without explicit spectral inflow.
* Auxiliary old/evaluation/output views are filled at the requested stage time;
  coarse/fine WENO interface checks cover both upwind signs.
* ERF's native `erf.dt_ref_ratio=2` is exercised with two fine substeps per
  coarse step. Accepted spectral transfers feed divergence, compact projection,
  diagnostics, and the provider-owned AMR flux register.
* Restart schema identities are strict and include
  `complete-groups-donor-support-v2`, `WENO_Z3-group-FCT-v2`, and
  `periodic+wall+outflow-no-inflow-v1`.
* WP9 residual checks use per-component
  `1e-12 * max(abs(initial), abs(final))`, with exact zero tolerance for exact
  zero scale; there is no fixed `1e-10` checker floor.

## Numerical evidence

The finite-volume smooth periodic oracle compares WENO-Z3 reconstruction of
the exact face value of the cell-average primitive `2+sin(2*pi*x)` for both
positive and negative carrier signs against the donor reconstruction. The
generated evidence file is
`/private/tmp/erf_sbm_p2_weno_convergence.csv`.

| N | WENO + error | WENO - error | Donor error | WENO + order | WENO - order | Donor order |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2.34696e-2 | 2.34696e-2 | 1.93839e-1 | -- | -- | -- |
| 32 | 6.26796e-4 | 6.26796e-4 | 9.78598e-2 | 5.2267 | 5.2267 | 0.9861 |
| 64 | 7.87265e-5 | 7.87265e-5 | 4.90480e-2 | 2.9931 | 2.9931 | 0.9965 |
| 128 | 9.85267e-6 | 9.85267e-6 | 2.45388e-2 | 2.9983 | 2.9983 | 0.9991 |

The production semidiscrete manufactured transport evidence is
`/private/tmp/erf_sbm_p2_full_transport_convergence.csv`; its WENO orders are
`2.9777, 2.9944, 2.9986` and donor orders are `0.9861, 0.9965, 0.9991`.
These are measured smooth finite-volume/operator results, not a universal
third-order claim for the nonlinear ERF time integrator.

The independent coarse/fine interface oracle checks both signs of the
upwind-facing WENO stencil and both `AuxiliaryTimeView::Old` and
`AuxiliaryTimeView::Evaluation` at the stage time. The production interface
diagnostic compares accepted fine-minus-coarse transfer with actual spectral
reflux correction using a relative machine-epsilon tolerance.

## Memory evidence

The production diagnostic reports logical grown-FAB payloads separately for
persistent state, accepted/stage ledger, and peak chunk-local working data.
`/private/tmp/erf_sbm_p2_chunk_equivalence_memory.csv` records the actual
chunk-policy comparison for 1M/2M layouts and 4/16/64 bins. For fixed chunk
size, temporary bytes are independent of total bin count; all-group bytes grow
with the number of complete groups. The test compares authoritative spectral
state, compact projection, and accepted face-transfer results against the
all-groups reference and rejects zero chunk size.

The measured temporary working-byte rows are:

| Moment mode | Bins | chunk 1 | chunk 2 | chunk 4 | all groups |
|---|---:|---:|---:|---:|---:|
| 1M | 4 | 15,808 | 31,616 | 63,232 | 63,232 |
| 1M | 16 | 15,808 | 31,616 | 63,232 | 252,928 |
| 1M | 64 | 15,808 | 31,616 | 63,232 | 1,011,712 |
| 2M | 4 | 34,176 | 68,352 | 136,704 | 136,704 |
| 2M | 16 | 34,176 | 68,352 | 136,704 | 546,816 |
| 2M | 64 | 34,176 | 68,352 | 136,704 | 2,187,264 |

For fixed chunk size, temporary bytes are invariant across 4/16/64 bins;
the all-groups column is intentionally proportional to the number of complete
groups.

## Runtime and lifecycle evidence

The qualification fixtures and focused tests cover:

* 1M and 2M layouts, runtime bin counts, complete groups, endpoint transforms,
  attached properties, donor support, orphan zero-carrier states, and no
  clipping;
* exact FCT signs, combined advection/diffusion demand, both RK/Heun stage
  contracts, WENO versus donor convergence, and chunk-policy equivalence;
* two-level AMR, volume-weighted restriction, stage-time coarse/fine fill,
  dynamic regrid/remake, accepted-transfer reflux accounting, native factor-two
  subcycling, and same-decomposition restart;
* one- and two-rank active-limiter decomposition, wall/outflow diagnostics,
  physical boundary policy translation, and inward-carrier rejection;
* capability rejection for prescribed inflow, non-periodic AMR, native
  subcycling disabled, tensor diffusion, unsupported moisture paths, and P3
  processes.

Generated production diagnostics are under the configured test build's
`Tests/test_files/` directory and include per-component transfer, projection,
boundary inventory/closure, scale/tolerance, interface-oracle, and subcycling
fields.

## Final validation matrix

| Required check | Result |
|---|---|
| Focused P0/P1/P2, 1 MPI rank | 45/45 passed |
| Focused P0/P1/P2, 2 MPI ranks | 45/45 passed |
| P0/P1 regression | Retained in both focused runs; 13/13 passed at each rank count |
| SBM CTest | 12/12 passed (`-L sbm`, including 6 P1 and 6 P2 tests) |
| Full CTest | 880/880 passed, `--parallel 8` |
| CMake BuildTests | Passed with pinned Spack `mpicc`/`mpicxx`, `--parallel 8` |
| CMake production build | Passed in `Build`, `--parallel 8` |
| GNUmake | Passed with pinned Spack wrappers, `make -C Exec -j8`; `Exec/ERF3d.gnu.TEST.MPI.NC.ex` linked |
| AMR, no subcycle | Passed; two-level 2M grouped FCT, dynamic regrid, reflux/interface oracle |
| AMR, native subcycle | Passed at 1/2 ranks; `coarse_steps=2`, `fine_steps=4`, `fine_substeps_per_coarse=2` |
| Open boundary | Wall and outflow fixtures passed at 1/2 ranks; wall outward inventory exactly 0; outflow component 0 inventory `1.249375e-9` |
| Restart equivalence | Passed by `SBM_P2_AMR_RESTART_2M`; altered/old policy identities reject |
| Documentation | Passed: 15 unit checks, 65 fixed + 3 dynamic catalog checks, Doxygen/Sphinx generated |
| GPU runtime | Not run; no GPU runtime is advertised or required for P2 |

The strongest no-subcycle AMR interface-oracle error was
`7.3438532655006961e-25`; the largest no-subcycle component residual was
`1.6940658945086007e-21` against a largest physical tolerance of
`3.9992466485459458e-18`. Native-subcycle interface-oracle maxima were
`1.6233847159845438e-24` at both rank counts. Boundary closure maxima were
`2.541098841762901e-21` for the wall fixture and
`4.8580317375239315e-21` for the outflow fixture, each below its
per-component physical tolerance. The outflow fixture used zero explicit
diffusion and the inward-flow negative control rejected the exact prescribed
diagnostic message.

## Gate table

| Gate | Required evidence | Status |
|---|---|---|
| G0 source archaeology | ERF carrier flux, stage timing, FillPatch, AMR register, regrid, restart | PASS |
| G1 constraint groups | Complete 1M/2M/property/support groups and atomic chunks | PASS |
| G2 grouped FCT | Exact algebra, pre-stage baseline, active MPI, no clipping | PASS |
| G3 diffusion/boundaries | Density-weighted diffusion and wall/outflow matrix | PASS |
| G4 AMR/reflux | Composite totals, independent interface oracle, native subcycling | PASS |
| G5 regrid/restart | Changed footprint and strict exact restart comparison | PASS |
| G6 qualification/memory/docs | Full CTest, CMake/GNUmake, WENO, memory, docs | PASS |

## Final disposition

`P2 FULLY QUALIFIED FOR THE DECLARED SUPPORTED CONFIGURATION — READY TO BEGIN P3`

This milestone does not expand the support boundary or claim any unimplemented
warm-cloud physics.

## Review link

`https://github.com/pressel/ERF/tree/sbm-p2-final-qualification`
