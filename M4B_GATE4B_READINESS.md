# M4b Gate 4B readiness record

**Gate 4A local evidence:** the timestamp, halo, split ownership, signed interface, direct-rate, and fixed-grid conservation cases listed below pass on the CPU/MPI qualification build. This is local source/build/test evidence, not CI or independent review.

**Gate 4B entry disposition: CONDITIONAL.** Gate 4B implementation remains stopped pending user/reviewer decision about proceeding on CPU-only evidence. The bounded conditions are: GPU execution and device-memory high-water are not qualified; optimized-build communication/memory measurements are not yet available; and dynamic regrid/remake/derefinement transaction tests and their peak allocation measurement do not exist. The source-grounded transaction below is a readiness design only. It does not enable regridding or change `RemakeLevel`.

The tranche status is local to Gate 4A; it does not declare Gate 4B or full M4b complete. No commit, push, PR, merge, or AMReX submodule change was made.

## Current ownership and callback order

1. `ERF::timeStep` schedules `regrid(lev, time)` before advancing that level when the regrid interval is due (`Source/TimeIntegration/ERF_TimeStep.cpp`). The callback may create a finer level or remake an existing one. The top-level `ERF::Evolve` calls `post_timestep` only after the recursive timestep returns and the fine levels have completed their substeps.
2. `MakeNewLevelFromScratch` creates host level storage and the SBM manager/transport views. It defines the level's static mapped measure after the host density and geometry exist, and initializes compact `qc`/`qr` as a projection of the authoritative spectrum. Restart defers the measure rebuild until checkpoint geometry is restored (`Source/ERF_MakeNewLevel.cpp`, `Source/IO/ERF_Checkpoint.cpp`).
3. `MakeNewLevelFromCoarse` installs the requested `BoxArray`/`DistributionMapping`, constructs host arrays and metrics, then defines the SBM level and static mapped measure. It uses the accepted coarse `new_state` and same-time coarse `vars_new` density to construct a private carrier-relative fine candidate. `ProlongCarrierRelativeSpectrumPrivate` and strict admission complete before `accept_synchronized_state` publishes that candidate; compact fields are projected afterward (`Source/ERF_MakeNewLevel.cpp`). This path initializes a genuinely new level. It does not preserve overlap data for a remade level.
4. `RemakeLevel` currently rejects an active `sbm_state_manager` immediately with the message that conservative hierarchy remap is not qualified (`Source/ERF_MakeNewLevel.cpp`). Dynamic regrid and derefinement therefore remain unsupported for SBM.
5. `post_timestep` runs after recursive level advance and all fine substeps. It first performs ERF's host reflux/average-down work. Those host mutations cannot be rolled back by the SBM owner. The SBM coarse candidate is then private through mapped restriction, chunked YA corrections, full canonical admission, manager preflight, and pending-register acceptance; only then is the spectral state published and `qc`/`qr` regenerated (`Source/ERF.cpp`). A failure after the host callback must remain fail-stop. The spectral/compact pair is failure-atomic; the whole ERF host callback is not.
6. `make_flux_register` recreates the native host and SBM interface registers from the current neighboring layouts. `bind_sbm_amr_flux_registers` reconnects coarse/fine transport owners. `ClearLevel` destroys the level's SBM manager and transport views, resets both adjacent interface registers, then rebinds surviving neighbors (`Source/ERF_MakeNewArrays.cpp`, `Source/ERF_MakeNewLevel.cpp`). A future regrid transaction must not discard an outstanding interval while doing this.

## Required Gate 4B transaction design

### Derefinement and new-cell initialization

Before any old fine level storage is cleared, restrict the accepted fine authoritative spectrum to the coarse cells that are about to lose fine coverage. Pair the fine spectrum with the same accepted fine `rho_d` time and its existing `omega`; form the mapped amount `H = omega U`, average children, and divide by the coarse `omega` only after checking the mapped carrier inventory identity. Keep retained fine-cell values by overlap copy, preserving their bits. Apply carrier-relative piecewise-constant prolongation only to newly introduced cells. A full fine candidate may be private, but no existing value may be overwritten by a coarse reconstruction before overlap classification.

### Time views and host density

At a synchronized regrid instant, create remapped state views whose semantic timestamps equal the host density timestamps they use. If old and new views represent distinct times, remap each with its matching host density endpoint. Collapse old/new to one state only when ERF's regrid entry is proven to be a single accepted physical instant and both host views have that same semantic identity. Never pair an old spectrum with new `rho_d`, or rebuild a time bracket from an unrelated `S_old`/`S_new` copy.

### Validation and publication

Keep the old hierarchy and all pending amounts alive while new layout metadata, host carriers, static measures, new-level/retained-cell candidates, and replacement interface registers are prepared. Collectively validate every full-spectrum candidate using existing canonical admission and mapped carrier-inventory checks. Confirm all levels and interfaces are ready and no old YA interval remains pending. Only then switch the manager/transport/register owners and project compact fields. A failure before publication destroys only private candidates; after ERF mutates host state in its callback, failure must terminate the run rather than promise unsupported rollback.

### Register and scratch recreation

After the new `BoxArray`/`DistributionMapping` is accepted, rebuild each affected interface register from its final pair of layouts, rebind both adjacent transport levels, redefine the two-ghost coarse-only masks and bounded complete-group workspaces, then rebuild `omega` from final `detJ` and map factors. Reset or consume an old register only as part of the admitted layout switch; never silently drop a pending coarse/fine interval.

## Memory peak model and known costs

For `N` spectral components, local valid/ghost cell counts `C` (coarse) and `F` (fine), and complete-group scratch width `K`, count FAB payload in bytes using the configured `sizeof(Real)`:

| Owner / phase | Payload model | Status |
|---|---:|---|
| Authoritative old/new spectrum, two levels | `2 N (C + F) sizeof(Real)` | Persistent time views required by ERF stage ownership |
| Fine transport | Two-ghost `N`-component input/base/anchor views plus the physical input view; bounded rate and ratio workspaces | Existing M3 scheme; no new persistent FillPatch snapshot is planned |
| Coarse synchronization | One private `N C` candidate, then reuse it as restricted `U`, mapped `H`, refluxed `H`, and final `U`; chunk correction `K C` plus scalar masks | Successive representations must share the candidate where safe |
| Interface YA | Dense coarse `N C` storage in pinned `YAFluxRegister`, plus interface patch state | Known structural cost; remains after completed-ledger removal |
| Parent-fill scratch, production | `(2K+2) * (coarsened local fine cells + ghost cells) * sizeof(Real)` | Bounded destination scratch; measured componentwise source copies are the production default |
| Parent-fill scratch, optional pack comparator | `(2K+2) * local coarse cells * sizeof(Real)` | Native coarse BoxArray/DM; allocated only when the test/benchmark explicitly selects packed mode |
| Regrid overlap/remap | Old fine `N F_old` + new private fine candidate `N F_new` + host old/new views + YA + bounded transfer scratch | Planned peak; actual high-water has not yet been measured |

The source-shaped unit census ran on one CPU rank with 32,768 local coarse cells and 32,768 local fine cells, 3D double precision (`sizeof(Real)=8`), `K=16`, and separately constructed YA/rate/transport views. Its byte totals are FAB payload, not process RSS. `YA_plus_face_chunks` includes the pinned dense `YA` and bounded rate/face storage; `YA_dense_coarse` is the YA all-component coarse portion called out separately. The componentwise production-stage peak excludes the optional pack. The packed-comparison peak adds the one source pack.

| Components | Dense coarse YA bytes | YA + face chunks bytes | Componentwise stage peak bytes | Packed comparison stage peak bytes | Full completed-integral reference ledger bytes |
|---:|---:|---:|---:|---:|---:|
| 32 | 8,388,608 | 97,719,488 | 321,565,888 | 330,478,784 | 53,477,376 |
| 128 | 33,554,432 | 124,550,336 | 942,939,328 | 951,852,224 | 213,909,504 |
| 256 | 67,108,864 | 160,324,800 | 1,771,437,248 | 1,780,350,144 | 427,819,008 |

The census measured `old_new_transport_and_ghost_bytes` of 220,086,272 / 814,628,864 / 1,607,352,320 and two-ghost region payload of 63,963,136 / 255,852,544 / 511,705,088 bytes for 32 / 128 / 256 components. Parent destination scratch was 3,760,128 bytes. The optional 34-component source pack was 8,912,896 bytes on this one-rank layout. The private coarse synchronization candidate and mutually exclusive chunk workspace had a measured peak of 26,476,544 / 51,642,368 / 85,196,800 bytes; adding that to YA/rate and existing state payload gives sync peaks of 344,282,304 / 990,821,568 / 1,852,873,920 bytes. These source-shaped peaks coexist only with the fields indicated in the test; they are not a live ERF-wide or regrid peak.

### Remaining whole-domain all-component storage

Production no longer allocates the completed full-spectrum reference ledger: the ERF constructor selects direct accepted stage-rate contributions, with the full ledger retained only by the small-grid oracle. The one-rank 32,768-cell, 256-component census shows the pinned YA register's dense coarse portion is 67,108,864 bytes (64 MiB), compared with 427,819,008 bytes (about 408 MiB) for the reference ledger. No configured memory budget or measured run currently shows the remaining YA allocation to be unacceptable. Its per-rank size follows each rank's local coarse BoxArray; device allocation was not measured.

The remaining whole-domain component views have distinct roles: ERF's required old/new authoritative time states; the existing M3 transport inputs (`input`, `physical_input`, `base`, and `anchor`, including their required two-ghost regions); the single private coarse synchronization candidate; and the pinned YA dense coarse register. The candidate is reused successively for restricted `U`, mapped `H`, refluxed `H`, and final `U`. There is no production completed-ledger duplicate, no persistent all-component FillPatch pack, and no regrid staging copy added by this work. The measured source-shaped census above accounts for these payload classes and clearly distinguishes the reference ledger and optional packed-fill comparator from production storage.

If future per-rank measurements exceed an explicit YA memory budget, the bounded architecture alternative is an interface-cell-only sparse register with `O(N * interface coarse cells)` storage, while preserving pending stage-rate accumulation, interval scaling, component mapping, and failure-atomic acceptance. That would replace the pinned dense YA implementation and therefore needs its own equivalence and MPI qualification; the present evidence does not justify changing register architecture. Dynamic regrid remains disabled, so the old-state-plus-new-candidate regrid peak in the model is not a measurement.

The live parent-fill profile is separate: inside the actual ERF helper it recorded `TotalBytesAllocatedInFabs()` by rank and the local destination and optional source-pack FABs. Componentwise live FAB MiB per rank at 1 / 2 / 4 ranks was 47.639 / 23.819 / 12.922 for N=32, 119.557 / 59.778 / 32.328 for N=128, and 215.448 / 107.724 / 58.203 for N=256. The local destination scratch was 0.398 / 0.199 / 0.149 MiB per rank at 1 / 2 / 4 ranks. Packed mode added 1.062 / 0.531 / 0.266 MiB source-pack payload per rank at 1 / 2 / 4 ranks for each N; measured rank live-FAB values rose by the same payload. This is an actual FAB counter, not RSS or device allocator high-water.

### Measured communication choice

The profile used the same CPU Debug build, split-capable periodic fixture, three fine substeps, 3 host stages per fine step, and one, two, or four MPI ranks. Each parent role was called nine times per fine substep. The table shows mean elapsed time in milliseconds per complete fine step, averaged over emitted rank records. `Calls/role` is the actual instrumented `ParallelCopy` call count inside one parent fill, not an MPI message count.

| N | Componentwise → packed calls per parent role | Componentwise fine-step ms (1/2/4 ranks) | Packed fine-step ms (1/2/4 ranks) |
|---:|---:|---:|---:|
| 32 | 68 → 2 | 460 / 258 / 189 | 547 / 299 / 202 |
| 128 | 272 → 8 | 1,590 / 882 / 656 | 1,919 / 1,034 / 751 |
| 256 | 544 → 16 | 3,108 / 1,731 / 1,275 | 3,743 / 2,017 / 1,466 |

Packing reduced the number of `ParallelCopy` calls per role by about 34x, and reduced measured copy-only time by roughly 5–14%; however, complete fine-step time increased by about 7–21% across this matrix. The helper total also increased, and packing allocated the bounded 34-component source FAB. The production default therefore remains componentwise; the packed path is retained only as a bounded exact-value comparator and profile option. There is no claim that either mode is optimized for production-sized science cases. The simple path should be reprofiled in optimized CPU builds before Gate 4B decides whether a different batched design is justified.

No GPU was available: GPU execution and device memory are **NOT QUALIFIED**. No process RSS or regrid high-water was collected. Gate 4B must measure live per-rank FAB plus device allocator high-water during each transaction phase before any production regrid is enabled. Keep the old fine state plus a single private candidate as the planned unavoidable peak. Do not add full-component copies for overlap masks, FillPatch, or regrid routing.

## Minimum deterministic Gate 4B tests

1. Retained overlap is bitwise identical after a layout change.
2. Newly refined cells use same-time carrier-relative prolongation and conserve mapped carrier inventory.
3. Derefinement restricts accepted fine state before clear and preserves each component's composite leaf inventory.
4. A rejected candidate leaves authoritative spectral and compact publication unchanged and leaves no consumable pending YA amount.
5. Create, clear, and recreate each level and adjacent interface; verify no old handle or measure survives.
6. Split MPI layout with empty and remote-owner ranks follows one collective path and preserves inventory.
7. Record live per-rank/device allocation high-water during prepare, validation, publish, and old-owner release; compare with the model above.

**Open measurement:** current fixed-grid tests cannot establish regrid peak bytes or callback rollback behavior. Resolve the byte budget with an instrumented create/remake/derefinement test before implementing production publication. Source already establishes that host-side callbacks mutate ERF state before return, so the transaction boundary must be fail-stop past that point.
