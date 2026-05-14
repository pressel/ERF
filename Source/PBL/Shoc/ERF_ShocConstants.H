#ifndef ERF_SHOC_CONSTANTS_H_
#define ERF_SHOC_CONSTANTS_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace shoc::constants {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real min_tke () noexcept { return amrex::Real(4.0e-4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real max_tke () noexcept { return amrex::Real(50.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real base_temp () noexcept { return amrex::Real(300.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real latent_ice () noexcept { return amrex::Real(3.34e5); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real min_temp () noexcept { return amrex::Real(180.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real freezing_temp () noexcept { return amrex::Real(273.15); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tropopause_pressure () noexcept { return amrex::Real(8.0e4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real stable_ckh () noexcept { return amrex::Real(0.1); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real stable_ckm () noexcept { return amrex::Real(0.1); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real shear_ck () noexcept { return amrex::Real(0.1); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tke_cs () noexcept { return amrex::Real(0.15); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pbl_transition_depth () noexcept { return amrex::Real(200.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real max_isotropy () noexcept { return amrex::Real(2.0e4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real zvir () noexcept { return amrex::Real(0.61); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real u_star_min () noexcept { return amrex::Real(0.01); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real u_ws_min () noexcept { return amrex::Real(1.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ksrf_min () noexcept { return amrex::Real(1.0e-4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real min_qw () noexcept { return amrex::Real(0.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real min_length () noexcept { return amrex::Real(20.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real max_length () noexcept { return amrex::Real(2.0e4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pbl_max_pressure () noexcept { return amrex::Real(4.0e4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pbl_ri_crit () noexcept { return amrex::Real(0.3); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pbl_fac () noexcept { return amrex::Real(100.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pbl_fak () noexcept { return amrex::Real(8.5); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pbl_betam () noexcept { return amrex::Real(15.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pbl_sffrac () noexcept { return amrex::Real(0.1); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real kbfs_eps () noexcept { return amrex::Real(1.0e-10); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real large_negative () noexcept { return amrex::Real(-99999999.99); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real w3_clip () noexcept { return amrex::Real(1.2); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real w3_clip_default () noexcept { return amrex::Real(0.02); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real uf_min () noexcept { return amrex::Real(0.01); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real thl_tol () noexcept { return amrex::Real(1.0e-2); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rt_tol () noexcept { return amrex::Real(1.0e-4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real w_tol_squared () noexcept { return amrex::Real(4.0e-4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real w_threshold () noexcept { return amrex::Real(0.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tl_min () noexcept { return amrex::Real(100.0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pdf_tmp () noexcept { return amrex::Real(0.4); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sqrt_pdf_tmp () noexcept { return amrex::Real(0.77459666924148337704); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool do_thetal_skew () noexcept { return false; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real inv_sqrt_two_pi () noexcept { return amrex::Real(0.39894228040143267794); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tabs_crit () noexcept { return amrex::Real(182.0); }

} // namespace shoc::constants

#endif