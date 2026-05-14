#ifndef ERF_SHOC_GPU_UTILS_H_
#define ERF_SHOC_GPU_UTILS_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

namespace shoc {

// SHOC GPU portability rules:
// 1. ShocColumnData, FArrayBox, and MultiFab are host-side ownership objects.
// 2. Device-callable functions must take Array4 views, POD structs, scalars, and options only.
// 3. Device-callable functions must not call .array(), .const_array(), .box(), .nComp(), resize(), or setVal().
// 4. Do not capture this in AMReX GPU lambdas.
// 5. Do not reference anonymous-namespace constexpr Real constants inside GPU lambdas.
// 6. Use SHOC constants through AMREX_GPU_HOST_DEVICE inline accessors.
// 7. Use amrex::min, amrex::max, amrex::Math::abs, and explicit arithmetic in device paths.
// 8. Use a central SHOC FArrayBox initialization helper, not direct FArrayBox::setVal calls.
// 9. Unit tests must allocate column data in backend-safe memory and synchronize before host assertions.

inline void
set_fab_val (amrex::FArrayBox& fab, amrex::Real value)
{
#ifdef AMREX_USE_GPU
    fab.setVal<amrex::RunOn::Device>(value, fab.box(), 0, fab.nComp());
#else
    fab.setVal<amrex::RunOn::Host>(value, fab.box(), 0, fab.nComp());
#endif
}

inline void
sync_if_needed ()
{
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif
}

} // namespace shoc

#endif