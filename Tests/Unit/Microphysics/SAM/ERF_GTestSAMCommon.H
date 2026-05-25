#ifndef ERF_GTEST_SAM_COMMON_H_
#define ERF_GTEST_SAM_COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParReduce.H>
#include <AMReX_Reduce.H>
#include <AMReX_TableData.H>

#include <ERF_Constants.H>
#include <ERF_DataStruct.H>
#include <ERF_EOS.H>
#include <ERF_IndexDefines.H>
#include <ERF_MicrophysicsUtils.H>
#include <ERF_PlaneAverage.H>
#include <ERF_SAM.H>
#include <ERF_TileNoZ.H>

namespace sam_test {

constexpr amrex::Real kRdOcp = R_d / Cp_d;
constexpr amrex::Real kDefaultDt = amrex::Real(1.0);

struct PrimitiveState {
    amrex::Real rho;
    amrex::Real theta;
    amrex::Real tabs;
    amrex::Real pres_mbar;
    amrex::Real qv;
    amrex::Real qc;
    amrex::Real qi;
    amrex::Real qr;
    amrex::Real qs;
    amrex::Real qg;
};

inline void sync ()
{
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif
}

template <typename F>
inline void run_and_sync (F&& fn)
{
    std::forward<F>(fn)();
    sync();
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PrimitiveState make_primitive_state (const amrex::Real tabs,
                                     const amrex::Real pres_mbar,
                                     const amrex::Real qv,
                                     const amrex::Real qc,
                                     const amrex::Real qi,
                                     const amrex::Real qr,
                                     const amrex::Real qs,
                                     const amrex::Real qg)
{
    const amrex::Real pres_pa = amrex::Real(100.0) * pres_mbar;
    const amrex::Real rho = getRhogivenTandPress(tabs, pres_pa, qv);
    const amrex::Real theta = getThgivenTandP(tabs, pres_pa, kRdOcp);
    return PrimitiveState{rho, theta, tabs, pres_mbar, qv, qc, qi, qr, qs, qg};
}

inline amrex::Geometry make_geometry (const int nx,
                                      const int ny,
                                      const int nz)
{
    amrex::Box domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
                      amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(static_cast<amrex::Real>(nx),
                                          static_cast<amrex::Real>(ny),
                                          static_cast<amrex::Real>(nz))});
    amrex::Array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 0)};
    return amrex::Geometry(domain, &real_box, amrex::CoordSys::cartesian, periodicity.data());
}

inline SolverChoice make_solver_choice (const MoistureType moisture_type,
                                        const bool use_shoc)
{
    SolverChoice sc;
    sc.c_p = Cp_d;
    sc.rdOcp = kRdOcp;
    sc.ave_plane = 2;
    sc.moisture_type = moisture_type;
    sc.use_shoc = use_shoc;

    if (moisture_type == MoistureType::SAM) {
        sc.moisture_indices = MoistureComponentIndices(
            RhoQ1_comp, RhoQ2_comp, RhoQ3_comp, RhoQ4_comp, RhoQ5_comp, RhoQ6_comp);
    } else if (moisture_type == MoistureType::SAM_NoIce) {
        sc.moisture_indices = MoistureComponentIndices(
            RhoQ1_comp, RhoQ2_comp, -1, RhoQ4_comp);
    } else if (moisture_type == MoistureType::SAM_NoPrecip_NoIce) {
        sc.moisture_indices = MoistureComponentIndices(RhoQ1_comp, RhoQ2_comp);
    }

    return sc;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void set_conserved_cell (const amrex::Array4<amrex::Real>& arr,
                         const int i,
                         const int j,
                         const int k,
                         const PrimitiveState& state)
{
    arr(i,j,k,Rho_comp) = state.rho;
    arr(i,j,k,RhoTheta_comp) = state.rho * state.theta;
    arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
    arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);
    arr(i,j,k,RhoQ1_comp) = state.rho * state.qv;
    arr(i,j,k,RhoQ2_comp) = state.rho * state.qc;
    arr(i,j,k,RhoQ3_comp) = state.rho * state.qi;
    arr(i,j,k,RhoQ4_comp) = state.rho * state.qr;
    arr(i,j,k,RhoQ5_comp) = state.rho * state.qs;
    arr(i,j,k,RhoQ6_comp) = state.rho * state.qg;
}

inline void copy_multifab (const amrex::MultiFab& src,
                           amrex::MultiFab& dst)
{
    amrex::MultiFab::Copy(dst, src, 0, 0, src.nComp(), src.nGrowVect());
}

inline void fill_detj_portable (amrex::MultiFab& detj)
{
    for (amrex::MFIter mfi(detj, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = detj.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                arr(i,j,k) = amrex::Real(1.0) + amrex::Real(0.05) * (i + 2 * j + 3 * k);
            });
        });
    }
}

inline void fill_compressed_detj_portable (amrex::MultiFab& detj)
{
    for (amrex::MFIter mfi(detj, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = detj.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                arr(i,j,k) = amrex::Real(0.35) + amrex::Real(0.15) * k;
            });
        });
    }
}

inline void fill_shoc_no_precip_no_ice_state_portable (amrex::MultiFab& cons)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                PrimitiveState state;
                const int selector = (i + 2 * j + 3 * k) % 4;
                if (selector == 0) {
                    state = make_primitive_state(amrex::Real(289.0), amrex::Real(900.0),
                                                 amrex::Real(1.10e-2), amrex::Real(4.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 1) {
                    state = make_primitive_state(amrex::Real(286.0), amrex::Real(875.0),
                                                 amrex::Real(8.4e-3), amrex::Real(9.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 2) {
                    state = make_primitive_state(amrex::Real(284.0), amrex::Real(860.0),
                                                 amrex::Real(7.5e-3), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else {
                    state = make_primitive_state(amrex::Real(291.0), amrex::Real(930.0),
                                                 amrex::Real(1.20e-2), amrex::Real(1.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                }
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }
}

inline void fill_no_ice_rain_matrix_state_portable (amrex::MultiFab& cons)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                PrimitiveState state;
                const int selector = (i + 3 * j + 5 * k) % 6;
                if (selector == 0) {
                    state = make_primitive_state(amrex::Real(289.0), amrex::Real(900.0),
                                                 amrex::Real(1.05e-2), qcw0 + amrex::Real(4.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 1) {
                    state = make_primitive_state(amrex::Real(288.0), amrex::Real(890.0),
                                                 amrex::Real(9.8e-3), qcw0 - amrex::Real(1.5e-4),
                                                 amrex::Real(0.0), amrex::Real(3.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 2) {
                    state = make_primitive_state(amrex::Real(287.0), amrex::Real(880.0),
                                                 amrex::Real(8.0e-3), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(1.2e-3),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 3) {
                    state = make_primitive_state(amrex::Real(290.0), amrex::Real(910.0),
                                                 amrex::Real(1.25e-2), qcw0 + amrex::Real(2.0e-4),
                                                 amrex::Real(0.0), amrex::Real(8.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 4) {
                    state = make_primitive_state(amrex::Real(286.0), amrex::Real(870.0),
                                                 amrex::Real(7.2e-3), amrex::Real(2.0e-4),
                                                 amrex::Real(0.0), amrex::Real(1.8e-3),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else {
                    state = make_primitive_state(amrex::Real(284.0), amrex::Real(860.0),
                                                 amrex::Real(7.8e-3), qcw0 + amrex::Real(5.0e-4),
                                                 amrex::Real(0.0), amrex::Real(2.2e-3),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                }
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }
}

inline void fill_full_sam_cloud_matrix_state_portable (amrex::MultiFab& cons)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                PrimitiveState state;
                const int selector = (i + 2 * j + 5 * k) % 8;
                if (selector == 0) {
                    state = make_primitive_state(tbgmin - amrex::Real(15.0), amrex::Real(800.0),
                                                 amrex::Real(9.0e-4), amrex::Real(5.0e-4),
                                                 amrex::Real(6.0e-4), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 1) {
                    state = make_primitive_state((tbgmin + tbgmax) * amrex::Real(0.5), amrex::Real(820.0),
                                                 amrex::Real(2.0e-3), amrex::Real(7.0e-4),
                                                 amrex::Real(5.0e-4), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 2) {
                    state = make_primitive_state(tbgmax + amrex::Real(8.0), amrex::Real(840.0),
                                                 amrex::Real(8.0e-3), amrex::Real(6.0e-4),
                                                 amrex::Real(4.0e-4), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 3) {
                    state = make_primitive_state((tbgmin + tbgmax) * amrex::Real(0.5), amrex::Real(780.0),
                                                 amrex::Real(5.5e-3), amrex::Real(1.5e-4),
                                                 amrex::Real(2.5e-4), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 4) {
                    state = make_primitive_state(tbgmax + amrex::Real(4.0), amrex::Real(860.0),
                                                 amrex::Real(7.2e-3), amrex::Real(2.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 5) {
                    state = make_primitive_state(tbgmin + amrex::Real(3.0), amrex::Real(790.0),
                                                 amrex::Real(1.4e-3), amrex::Real(7.5e-4),
                                                 amrex::Real(9.0e-4), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 6) {
                    state = make_primitive_state((tbgmin + tbgmax) * amrex::Real(0.5), amrex::Real(805.0),
                                                 amrex::Real(2.3e-3), amrex::Real(1.0e-4),
                                                 amrex::Real(1.2e-3), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else {
                    state = make_primitive_state(tbgmax - amrex::Real(1.0), amrex::Real(815.0),
                                                 amrex::Real(3.2e-3), amrex::Real(2.0e-5),
                                                 amrex::Real(1.0e-5), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                }
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }
}

inline void fill_full_sam_precip_matrix_state_portable (amrex::MultiFab& cons)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                PrimitiveState state;
                const int selector = (i + 2 * j + 4 * k) % 8;
                if (selector == 0) {
                    state = make_primitive_state(tbgmax + amrex::Real(6.0), amrex::Real(900.0),
                                                 amrex::Real(1.05e-2), qcw0 + amrex::Real(4.0e-4),
                                                 amrex::Real(0.0), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 1) {
                    state = make_primitive_state(tbgmin + amrex::Real(5.0), amrex::Real(820.0),
                                                 amrex::Real(1.8e-3), amrex::Real(2.0e-4),
                                                 qci0 + amrex::Real(8.0e-4), amrex::Real(0.0),
                                                 amrex::Real(8.0e-4), amrex::Real(0.0));
                } else if (selector == 2) {
                    state = make_primitive_state(tgrmin - amrex::Real(2.0), amrex::Real(800.0),
                                                 amrex::Real(1.3e-3), amrex::Real(5.0e-4),
                                                 qci0 + amrex::Real(6.0e-4), amrex::Real(0.0),
                                                 amrex::Real(1.1e-3), amrex::Real(9.0e-4));
                } else if (selector == 3) {
                    state = make_primitive_state(tprmin + amrex::Real(2.0), amrex::Real(860.0),
                                                 amrex::Real(7.0e-3), qcw0 + amrex::Real(3.0e-4),
                                                 amrex::Real(0.0), amrex::Real(1.5e-3),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                } else if (selector == 4) {
                    state = make_primitive_state(tgrmin + amrex::Real(2.0), amrex::Real(810.0),
                                                 amrex::Real(1.5e-3), amrex::Real(7.0e-4),
                                                 qci0 + amrex::Real(7.0e-4), amrex::Real(0.0),
                                                 amrex::Real(5.0e-4), amrex::Real(1.2e-3));
                } else if (selector == 5) {
                    state = make_primitive_state(tprmin - amrex::Real(4.0), amrex::Real(780.0),
                                                 amrex::Real(1.2e-3), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(2.5e-4),
                                                 amrex::Real(2.0e-3), amrex::Real(5.0e-4));
                } else if (selector == 6) {
                    state = make_primitive_state(tgrmin + amrex::Real(6.0), amrex::Real(790.0),
                                                 amrex::Real(8.0e-4), amrex::Real(1.0e-4),
                                                 amrex::Real(2.0e-4), amrex::Real(1.0e-4),
                                                 amrex::Real(4.0e-4), amrex::Real(2.5e-3));
                } else {
                    state = make_primitive_state(tbgmax - amrex::Real(4.0), amrex::Real(830.0),
                                                 amrex::Real(1.0e-3), qcw0 + amrex::Real(8.0e-4),
                                                 qci0 + amrex::Real(9.0e-4), amrex::Real(9.0e-4),
                                                 amrex::Real(8.0e-4), amrex::Real(7.0e-4));
                }
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }
}

inline void fill_full_sam_sedimentation_column_state_portable (amrex::MultiFab& cons)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                PrimitiveState state;
                if (k == 0) {
                    state = make_primitive_state(tprmin + amrex::Real(6.0), amrex::Real(900.0),
                                                 amrex::Real(9.0e-3), amrex::Real(0.0),
                                                 amrex::Real(4.0e-4), amrex::Real(2.0e-3),
                                                 amrex::Real(8.0e-4), amrex::Real(0.0));
                } else if (k == 1) {
                    state = make_primitive_state(tprmin - amrex::Real(3.0), amrex::Real(860.0),
                                                 amrex::Real(5.0e-3), amrex::Real(1.0e-4),
                                                 amrex::Real(7.0e-4), amrex::Real(7.0e-4),
                                                 amrex::Real(1.6e-3), amrex::Real(3.0e-4));
                } else if (k == 2) {
                    state = make_primitive_state(tgrmin + amrex::Real(2.0), amrex::Real(820.0),
                                                 amrex::Real(2.0e-3), amrex::Real(0.0),
                                                 amrex::Real(9.0e-4), amrex::Real(2.0e-4),
                                                 amrex::Real(7.0e-4), amrex::Real(1.4e-3));
                } else if (k == 3) {
                    state = make_primitive_state(tbgmin + amrex::Real(2.0), amrex::Real(790.0),
                                                 amrex::Real(1.1e-3), amrex::Real(0.0),
                                                 amrex::Real(1.1e-3), amrex::Real(0.0),
                                                 amrex::Real(8.0e-4), amrex::Real(9.0e-4));
                } else {
                    state = make_primitive_state(tbgmin - amrex::Real(8.0), amrex::Real(760.0),
                                                 amrex::Real(6.0e-4), amrex::Real(0.0),
                                                 amrex::Real(4.0e-4), amrex::Real(0.0),
                                                 amrex::Real(0.0), amrex::Real(0.0));
                }
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }
}

#define SAM_BFB_BRANCH_HIT_FIELDS(X) \
    X(cloud_noop) \
    X(cloud_t_bgmin_or_less) \
    X(cloud_t_between_bgmin_bgmax) \
    X(cloud_t_bgmax_or_greater) \
    X(cloud_qt_gt_qsat) \
    X(cloud_qt_le_qsat) \
    X(cloud_evap_all_then_recheck) \
    X(cloud_newton) \
    X(cloud_condensate_limiter) \
    X(icefall_noop) \
    X(icefall_bottom_face) \
    X(icefall_interior_face) \
    X(icefall_top_face) \
    X(icefall_detj_present) \
    X(precip_noop) \
    X(precip_cloud_only) \
    X(precip_only) \
    X(precip_cloud_and_precip) \
    X(precip_water_autoconversion_off) \
    X(precip_water_autoconversion_on) \
    X(precip_ice_autoconversion_off) \
    X(precip_ice_autoconversion_on) \
    X(precip_rain_accretion_off) \
    X(precip_rain_accretion_on) \
    X(precip_snow_accretion_off) \
    X(precip_snow_accretion_on) \
    X(precip_graupel_accretion_off) \
    X(precip_graupel_accretion_on) \
    X(precip_sink_limited_cloud_water) \
    X(precip_sink_limited_cloud_ice) \
    X(precip_evaporation_off) \
    X(precip_evaporation_on) \
    X(precip_evaporation_species_limited) \
    X(precipfall_noop) \
    X(precipfall_bottom_face) \
    X(precipfall_interior_face) \
    X(precipfall_top_face) \
    X(precipfall_detj_present) \
    X(precipfall_zero_flux_below_threshold) \
    X(precipfall_nonzero_rain_flux) \
    X(precipfall_nonzero_snow_flux) \
    X(precipfall_nonzero_graupel_flux) \
    X(precipfall_surface_rain_accum) \
    X(precipfall_surface_snow_accum) \
    X(precipfall_surface_graupel_accum) \
    X(precipfall_nonnegative_clipping)

struct SAMBFBBranchHits {
#define X(name) amrex::Long name{0};
    SAM_BFB_BRANCH_HIT_FIELDS(X)
#undef X
};

enum SAMBFBBranchHitIndex : int {
#define X(name) k_##name,
    SAM_BFB_BRANCH_HIT_FIELDS(X)
#undef X
    k_num_branch_hits
};

constexpr int kSAMBFBBranchHitCount = k_num_branch_hits;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void sam_bfb_record_hit (amrex::Long* hits,
                         const int index) noexcept
{
    amrex::Gpu::Atomic::AddNoRet(&hits[index], amrex::Long(1));
}

inline void sam_bfb_accumulate_hit (SAMBFBBranchHits& hits,
                                    const int index,
                                    const amrex::Long count)
{
    switch (index) {
#define X(name) case k_##name: hits.name += count; break;
        SAM_BFB_BRANCH_HIT_FIELDS(X)
#undef X
    default:
        AMREX_ALWAYS_ASSERT(false);
    }
}

#undef SAM_BFB_BRANCH_HIT_FIELDS

inline amrex::Long component_mismatch_count (const amrex::MultiFab& lhs,
                                             const amrex::MultiFab& rhs,
                                             const int comp,
                                             const amrex::IntVect& ngrow = amrex::IntVect(0))
{
    auto lhs_arrays = lhs.const_arrays();
    auto rhs_arrays = rhs.const_arrays();
    amrex::GpuTuple<amrex::Long> reduced = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Long>{},
        lhs, ngrow,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Long> {
            return {lhs_arrays[box_no](i,j,k,comp) == rhs_arrays[box_no](i,j,k,comp) ? 0L : 1L};
        });
    return amrex::get<0>(reduced);
}

inline amrex::Long scalar_mismatch_count (const amrex::MultiFab& lhs,
                                          const amrex::MultiFab& rhs,
                                          const amrex::IntVect& ngrow = amrex::IntVect(0))
{
    auto lhs_arrays = lhs.const_arrays();
    auto rhs_arrays = rhs.const_arrays();
    amrex::GpuTuple<amrex::Long> reduced = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Long>{},
        lhs, ngrow,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Long> {
            return {lhs_arrays[box_no](i,j,k) == rhs_arrays[box_no](i,j,k) ? 0L : 1L};
        });
    return amrex::get<0>(reduced);
}

// Frozen reference copied from development @ b9247103a856e0552cf47f050e9a7c8b7846a96a
// for the Phase 1 SAM BFB guard. Rebase this copy deliberately, not incidentally.
class SAMCurrentBehaviorReference {
public:
    using FabPtr = std::shared_ptr<amrex::MultiFab>;

    void Define (SolverChoice& sc)
    {
        m_fac_cond = lcond / sc.c_p;
        m_fac_fus = lfus / sc.c_p;
        m_fac_sub = lsub / sc.c_p;
        m_axis = sc.ave_plane;
        m_rdOcp = sc.rdOcp;
        m_do_cond = (!sc.use_shoc);
    }

    void Set_dzmin (const amrex::Real dz_min) { m_dzmin = dz_min; }

    void Set_RealWidth (const int real_width) { m_real_width = real_width; }

    void Init (const amrex::MultiFab& cons_in,
               const amrex::BoxArray&,
               const amrex::Geometry& geom,
               const amrex::Real& dt_advance,
               std::unique_ptr<amrex::MultiFab>& z_phys_nd,
               std::unique_ptr<amrex::MultiFab>& detJ_cc)
    {
        dt = dt_advance;
        m_geom = geom;

        m_z_phys_nd = z_phys_nd.get();
        m_detJ_cc = detJ_cc.get();

        MicVarMap.resize(m_qmoist_size);
        MicVarMap = {MicVar::rain_accum, MicVar::snow_accum, MicVar::graup_accum};

        for (int ivar = 0; ivar < MicVar::NumVars; ++ivar) {
            mic_fab_vars[ivar] = std::make_shared<amrex::MultiFab>(cons_in.boxArray(), cons_in.DistributionMap(),
                                                                   1, cons_in.nGrowVect());
            mic_fab_vars[ivar]->setVal(amrex::Real(0.0));
        }

        nlev = m_geom.Domain().length(2);
        zlo = m_geom.Domain().smallEnd(2);
        zhi = m_geom.Domain().bigEnd(2);

        accrrc.resize({zlo}, {zhi});
        accrsi.resize({zlo}, {zhi});
        accrsc.resize({zlo}, {zhi});
        coefice.resize({zlo}, {zhi});
        evaps1.resize({zlo}, {zhi});
        evaps2.resize({zlo}, {zhi});
        accrgi.resize({zlo}, {zhi});
        accrgc.resize({zlo}, {zhi});
        evapg1.resize({zlo}, {zhi});
        evapg2.resize({zlo}, {zhi});
        evapr1.resize({zlo}, {zhi});
        evapr2.resize({zlo}, {zhi});

        rho1d.resize({zlo}, {zhi});
        pres1d.resize({zlo}, {zhi});
        tabs1d.resize({zlo}, {zhi});

        ResetBranchHits();
    }

    void Copy_State_to_Micro (const amrex::MultiFab& cons_in)
    {
        for (amrex::MFIter mfi(cons_in); mfi.isValid(); ++mfi) {
            const auto& box3d = mfi.growntilebox();

            auto states_array = cons_in.array(mfi);

            auto qv_array = mic_fab_vars[MicVar::qv]->array(mfi);
            auto qc_array = mic_fab_vars[MicVar::qcl]->array(mfi);
            auto qi_array = mic_fab_vars[MicVar::qci]->array(mfi);
            auto qn_array = mic_fab_vars[MicVar::qn]->array(mfi);
            auto qt_array = mic_fab_vars[MicVar::qt]->array(mfi);

            auto qpr_array = mic_fab_vars[MicVar::qpr]->array(mfi);
            auto qps_array = mic_fab_vars[MicVar::qps]->array(mfi);
            auto qpg_array = mic_fab_vars[MicVar::qpg]->array(mfi);
            auto qp_array = mic_fab_vars[MicVar::qp]->array(mfi);

            auto rho_array = mic_fab_vars[MicVar::rho]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar::theta]->array(mfi);
            auto tabs_array = mic_fab_vars[MicVar::tabs]->array(mfi);
            auto pres_array = mic_fab_vars[MicVar::pres]->array(mfi);

            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                rho_array(i,j,k) = states_array(i,j,k,Rho_comp);
                theta_array(i,j,k) = states_array(i,j,k,RhoTheta_comp) / states_array(i,j,k,Rho_comp);

                qv_array(i,j,k) = std::max(amrex::Real(0.0), states_array(i,j,k,RhoQ1_comp) / states_array(i,j,k,Rho_comp));
                qc_array(i,j,k) = std::max(amrex::Real(0.0), states_array(i,j,k,RhoQ2_comp) / states_array(i,j,k,Rho_comp));
                qi_array(i,j,k) = std::max(amrex::Real(0.0), states_array(i,j,k,RhoQ3_comp) / states_array(i,j,k,Rho_comp));
                qn_array(i,j,k) = qc_array(i,j,k) + qi_array(i,j,k);
                qt_array(i,j,k) = qv_array(i,j,k) + qn_array(i,j,k);

                qpr_array(i,j,k) = std::max(amrex::Real(0.0), states_array(i,j,k,RhoQ4_comp) / states_array(i,j,k,Rho_comp));
                qps_array(i,j,k) = std::max(amrex::Real(0.0), states_array(i,j,k,RhoQ5_comp) / states_array(i,j,k,Rho_comp));
                qpg_array(i,j,k) = std::max(amrex::Real(0.0), states_array(i,j,k,RhoQ6_comp) / states_array(i,j,k,Rho_comp));
                qp_array(i,j,k) = qpr_array(i,j,k) + qps_array(i,j,k) + qpg_array(i,j,k);

                tabs_array(i,j,k) = getTgivenRandRTh(states_array(i,j,k,Rho_comp),
                                                     states_array(i,j,k,RhoTheta_comp),
                                                     qv_array(i,j,k));
                pres_array(i,j,k) = getPgivenRTh(states_array(i,j,k,RhoTheta_comp), qv_array(i,j,k)) * amrex::Real(0.01);
            });
        }
    }

    void Copy_Micro_to_State (amrex::MultiFab& cons)
    {
        for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto& box3d = mfi.tilebox();

            auto states_arr = cons.array(mfi);
            auto rho_arr = mic_fab_vars[MicVar::rho]->array(mfi);
            auto theta_arr = mic_fab_vars[MicVar::theta]->array(mfi);
            auto qv_arr = mic_fab_vars[MicVar::qv]->array(mfi);
            auto qc_arr = mic_fab_vars[MicVar::qcl]->array(mfi);
            auto qi_arr = mic_fab_vars[MicVar::qci]->array(mfi);
            auto qpr_arr = mic_fab_vars[MicVar::qpr]->array(mfi);
            auto qps_arr = mic_fab_vars[MicVar::qps]->array(mfi);
            auto qpg_arr = mic_fab_vars[MicVar::qpg]->array(mfi);

            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                states_arr(i,j,k,RhoTheta_comp) = rho_arr(i,j,k) * theta_arr(i,j,k);

                states_arr(i,j,k,RhoQ1_comp) = rho_arr(i,j,k) * std::max(amrex::Real(0.0), qv_arr(i,j,k));
                states_arr(i,j,k,RhoQ2_comp) = rho_arr(i,j,k) * std::max(amrex::Real(0.0), qc_arr(i,j,k));
                states_arr(i,j,k,RhoQ3_comp) = rho_arr(i,j,k) * std::max(amrex::Real(0.0), qi_arr(i,j,k));
                states_arr(i,j,k,RhoQ4_comp) = rho_arr(i,j,k) * std::max(amrex::Real(0.0), qpr_arr(i,j,k));
                states_arr(i,j,k,RhoQ5_comp) = rho_arr(i,j,k) * std::max(amrex::Real(0.0), qps_arr(i,j,k));
                states_arr(i,j,k,RhoQ6_comp) = rho_arr(i,j,k) * std::max(amrex::Real(0.0), qpg_arr(i,j,k));
            });
        }

        cons.FillBoundary(m_geom.periodicity());
    }

    void Advance (const amrex::Real& dt_advance,
                  const SolverChoice& sc)
    {
        dt = dt_advance;
        Cloud(sc);
        IceFall(sc);
        Precip(sc);
        PrecipFall(sc);
    }

    amrex::MultiFab* Qmoist_Ptr (const int varIdx)
    {
        AMREX_ALWAYS_ASSERT(varIdx < m_qmoist_size);
        return mic_fab_vars[MicVarMap[varIdx]].get();
    }

    SAMBFBBranchHits BranchHits ()
    {
        amrex::Gpu::HostVector<amrex::Long> branch_hits_host(kSAMBFBBranchHitCount, 0);
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                              m_branch_hits_device.begin(), m_branch_hits_device.end(),
                              branch_hits_host.begin());
        amrex::Gpu::streamSynchronize();

        SAMBFBBranchHits branch_hits = m_branch_hits_host;
        for (int index = 0; index < kSAMBFBBranchHitCount; ++index) {
            sam_bfb_accumulate_hit(branch_hits, index, branch_hits_host[index]);
        }
        return branch_hits;
    }

private:
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NewtonIterSat (int& i, int& j, int& k,
                               const int& SAM_moisture_type,
                               const amrex::Real& fac_cond,
                               const amrex::Real& fac_fus,
                               const amrex::Real& fac_sub,
                               const amrex::Real& an,
                               const amrex::Real& bn,
                               const amrex::Array4<amrex::Real>& tabs_array,
                               const amrex::Array4<amrex::Real>& pres_array,
                               const amrex::Array4<amrex::Real>& qv_array,
                               const amrex::Array4<amrex::Real>& qc_array,
                               const amrex::Array4<amrex::Real>& qi_array,
                               const amrex::Array4<amrex::Real>& qn_array,
                               const amrex::Array4<amrex::Real>& qt_array,
                               amrex::Long* branch_hits)
    {
        amrex::Real tol = amrex::Real(1.0e-4);
        amrex::Real omn, domn;
        amrex::Real qsat = amrex::Real(0.0);
        amrex::Real dqsat = amrex::Real(0.0);
        amrex::Real qsatw, dqsatw;
        amrex::Real qsati, dqsati;
        int niter;
        amrex::Real fff, dfff, dtabs;
        amrex::Real lstar, dlstar;
        amrex::Real lstarw, lstari;
        amrex::Real delta_qv, delta_qc, delta_qi;
        amrex::Real tabs = tabs_array(i,j,k);
        amrex::Real pres = pres_array(i,j,k);

        sam_bfb_record_hit(branch_hits, k_cloud_newton);

        niter = 0;
        dtabs = 1;
        do {
            lstarw = fac_cond;
            lstari = fac_sub;
            domn = zero;

            erf_qsatw(tabs, pres, qsatw);
            erf_qsati(tabs, pres, qsati);
            erf_dtqsatw(tabs, pres, dqsatw);
            erf_dtqsati(tabs, pres, dqsati);

            if (SAM_moisture_type == 1) {
                if (tabs >= tbgmax) {
                    omn = one;
                } else if (tabs <= tbgmin) {
                    omn = zero;
                } else {
                    lstari = fac_fus;
                    omn = an * tabs - bn;
                    domn = an;
                }
            } else if (SAM_moisture_type == 2) {
                omn = one;
                domn = zero;
            } else {
                omn = one;
                domn = zero;
            }

            qsat = omn * qsatw + (one - omn) * qsati;
            dqsat = omn * dqsatw + (one - omn) * dqsati + domn * qsatw - domn * qsati;
            lstar = omn * lstarw + (one - omn) * lstari;
            dlstar = domn * lstarw - domn * lstari;

            fff = -tabs + tabs_array(i,j,k) + lstar * (qv_array(i,j,k) - qsat);
            dfff = -one + dlstar * (qv_array(i,j,k) - qsat) - lstar * dqsat;

            dtabs = -fff / dfff;
            tabs += dtabs;
            niter = niter + 1;
        } while (std::abs(dtabs) > tol && niter < 20);

        qsat += dqsat * dtabs;

        sam_bfb_record_hit(branch_hits, k_cloud_condensate_limiter);
        delta_qv = qv_array(i,j,k) - qsat;
        const amrex::Real unclamped_delta_qc = delta_qv * omn;
        const amrex::Real unclamped_delta_qi = delta_qv * (one - omn);
        delta_qc = std::max(-qc_array(i,j,k), unclamped_delta_qc);
        delta_qi = std::max(-qi_array(i,j,k), unclamped_delta_qi);

        qv_array(i,j,k) = qsat;
        qc_array(i,j,k) += delta_qc;
        qi_array(i,j,k) += delta_qi;
        qn_array(i,j,k) = qc_array(i,j,k) + qi_array(i,j,k);
        qt_array(i,j,k) = qv_array(i,j,k) + qn_array(i,j,k);

        return tabs;
    }

    void ResetBranchHits ()
    {
        m_branch_hits_host = SAMBFBBranchHits{};
        m_branch_hits_device.resize(kSAMBFBBranchHitCount);

        amrex::Gpu::HostVector<amrex::Long> zero_hits(kSAMBFBBranchHitCount, 0);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              zero_hits.begin(), zero_hits.end(),
                              m_branch_hits_device.begin());
        amrex::Gpu::streamSynchronize();
    }

public:
    void Compute_Coefficients ()
    {
        auto accrrc_t = accrrc.table();
        auto accrsi_t = accrsi.table();
        auto accrsc_t = accrsc.table();
        auto coefice_t = coefice.table();
        auto evaps1_t = evaps1.table();
        auto evaps2_t = evaps2.table();
        auto accrgi_t = accrgi.table();
        auto accrgc_t = accrgc.table();
        auto evapg1_t = evapg1.table();
        auto evapg2_t = evapg2.table();
        auto evapr1_t = evapr1.table();
        auto evapr2_t = evapr2.table();

        auto rho1d_t = rho1d.table();
        auto pres1d_t = pres1d.table();
        auto tabs1d_t = tabs1d.table();

        amrex::Real gam3 = erf_gammafff(three);
        amrex::Real gamr1 = erf_gammafff(three + b_rain);
        amrex::Real gamr2 = erf_gammafff((amrex::Real(5.0) + b_rain) / two);
        amrex::Real gams1 = erf_gammafff(three + b_snow);
        amrex::Real gams2 = erf_gammafff((amrex::Real(5.0) + b_snow) / two);
        amrex::Real gamg1 = erf_gammafff(three + b_grau);
        amrex::Real gamg2 = erf_gammafff((amrex::Real(5.0) + b_grau) / two);

        PlaneAverage rho_ave(mic_fab_vars[MicVar::rho].get(), m_geom, m_axis);
        PlaneAverage theta_ave(mic_fab_vars[MicVar::theta].get(), m_geom, m_axis);
        PlaneAverage qv_ave(mic_fab_vars[MicVar::qv].get(), m_geom, m_axis);
        rho_ave.compute_averages(ZDir(), rho_ave.field());
        theta_ave.compute_averages(ZDir(), theta_ave.field());
        qv_ave.compute_averages(ZDir(), qv_ave.field());

        const int ncell = rho_ave.ncell_line();
        amrex::Gpu::HostVector<amrex::Real> rho_h(ncell), theta_h(ncell), qv_h(ncell);
        rho_ave.line_average(0, rho_h);
        theta_ave.line_average(0, theta_h);
        qv_ave.line_average(0, qv_h);

        amrex::Gpu::DeviceVector<amrex::Real> rho_d(ncell), theta_d(ncell), qv_d(ncell);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, rho_h.begin(), rho_h.end(), rho_d.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, theta_h.begin(), theta_h.end(), theta_d.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, qv_h.begin(), qv_h.end(), qv_d.begin());
        amrex::Gpu::streamSynchronize();

        amrex::Real* rho_dptr = rho_d.data();
        amrex::Real* theta_dptr = theta_d.data();
        amrex::Real* qv_dptr = qv_d.data();

        amrex::ParallelFor(nlev, [=] AMREX_GPU_DEVICE (int k) noexcept
        {
            const amrex::Real RhoTheta = rho_dptr[k] * theta_dptr[k];
            const amrex::Real pressure = getPgivenRTh(RhoTheta, qv_dptr[k]);
            rho1d_t(k) = rho_dptr[k];
            pres1d_t(k) = pressure * amrex::Real(0.01);
            tabs1d_t(k) = std::min(getTgivenRandRTh(rho_dptr[k], RhoTheta, qv_dptr[k]), amrex::Real(273.16));
        });

        if (std::round(gam3) != 2) {
            std::cout << "cannot compute gamma-function in SAMCurrentBehaviorReference::Init" << std::endl;
            std::exit(-1);
        }

        amrex::ParallelFor(nlev, [=] AMREX_GPU_DEVICE (int k) noexcept
        {
            amrex::Real Prefactor;
            const amrex::Real pratio = std::sqrt(amrex::Real(1.29) / rho1d_t(k));
            const amrex::Real estw = amrex::Real(100.0) * erf_esatw(tabs1d_t(k));
            const amrex::Real esti = amrex::Real(100.0) * erf_esati(tabs1d_t(k));

            amrex::Real coef1 = fourth * PI * nzeros * a_snow * gams1 * pratio /
                std::pow((PI * rhos * nzeros / rho1d_t(k)), ((three + b_snow) / amrex::Real(4.0)));
            amrex::Real coef2 = std::exp(amrex::Real(0.025) * (tabs1d_t(k) - amrex::Real(273.15)));
            accrsi_t(k) = coef1 * coef2 * esicoef;
            accrsc_t(k) = coef1 * esccoef;
            coefice_t(k) = coef2;

            coef1 = (lsub / (tabs1d_t(k) * R_v) - one) * lsub / (therco * tabs1d_t(k));
            coef2 = R_v * R_d / (diffelq * esti);
            Prefactor = two * PI * nzeros / (rho1d_t(k) * (coef1 + coef2));
            Prefactor *= (two / PI);
            evaps1_t(k) = Prefactor * amrex::Real(0.65) * std::sqrt(rho1d_t(k) / (PI * rhos * nzeros));
            evaps2_t(k) = Prefactor * amrex::Real(0.44) * std::sqrt(a_snow * rho1d_t(k) / muelq) * gams2 *
                std::sqrt(pratio) * std::pow(rho1d_t(k) / (PI * rhos * nzeros), ((amrex::Real(5.0) + b_snow) / amrex::Real(8.0)));

            coef1 = fourth * PI * nzerog * a_grau * gamg1 * pratio /
                std::pow((PI * rhog * nzerog / rho1d_t(k)), ((three + b_grau) / amrex::Real(4.0)));
            coef2 = std::exp(amrex::Real(0.025) * (tabs1d_t(k) - amrex::Real(273.15)));
            accrgi_t(k) = coef1 * coef2 * egicoef;
            accrgc_t(k) = coef1 * egccoef;

            coef1 = (lsub / (tabs1d_t(k) * R_v) - one) * lsub / (therco * tabs1d_t(k));
            coef2 = R_v * R_d / (diffelq * esti);
            Prefactor = two * PI * nzerog / (rho1d_t(k) * (coef1 + coef2));
            evapg1_t(k) = Prefactor * amrex::Real(0.78) * std::sqrt(rho1d_t(k) / (PI * rhog * nzerog));
            evapg2_t(k) = Prefactor * amrex::Real(0.31) * std::sqrt(a_grau * rho1d_t(k) / muelq) * gamg2 *
                std::sqrt(pratio) * std::pow(rho1d_t(k) / (PI * rhog * nzerog), ((amrex::Real(5.0) + b_grau) / amrex::Real(8.0)));

            accrrc_t(k) = fourth * PI * nzeror * a_rain * gamr1 * pratio /
                std::pow((PI * rhor * nzeror / rho1d_t(k)), ((three + b_rain) / amrex::Real(4.0))) * erccoef;

            coef1 = (lcond / (tabs1d_t(k) * R_v) - one) * lcond / (therco * tabs1d_t(k));
            coef2 = R_v * R_d / (diffelq * estw);
            Prefactor = two * PI * nzeror / (rho1d_t(k) * (coef1 + coef2));
            evapr1_t(k) = Prefactor * amrex::Real(0.78) * std::sqrt(rho1d_t(k) / (PI * rhor * nzeror));
            evapr2_t(k) = Prefactor * amrex::Real(0.31) * std::sqrt(a_rain * rho1d_t(k) / muelq) * gamr2 *
                std::sqrt(pratio) * std::pow(rho1d_t(k) / (PI * rhor * nzeror), ((amrex::Real(5.0) + b_rain) / amrex::Real(8.0)));
        });
    }

private:
    void Cloud (const SolverChoice& sc)
    {
        if (!m_do_cond) {
            sam_bfb_accumulate_hit(m_branch_hits_host, k_cloud_noop, 1);
            return;
        }

        constexpr amrex::Real an = one / (tbgmax - tbgmin);
        constexpr amrex::Real bn = tbgmin * an;

        const amrex::Real fac_cond = m_fac_cond;
        const amrex::Real fac_sub = m_fac_sub;
        const amrex::Real fac_fus = m_fac_fus;
        const amrex::Real rdOcp = m_rdOcp;

        int SAM_moisture_type = 1;
        if (sc.moisture_type == MoistureType::SAM_NoIce ||
            sc.moisture_type == MoistureType::SAM_NoPrecip_NoIce) {
            SAM_moisture_type = 2;
        }

        const auto domain = m_geom.Domain();
        const int i_lo = domain.smallEnd(0);
        const int i_hi = domain.bigEnd(0);
        const int j_lo = domain.smallEnd(1);
        const int j_hi = domain.bigEnd(1);
        amrex::Long* branch_hits = m_branch_hits_device.data();

        for (amrex::MFIter mfi(*(mic_fab_vars[MicVar::tabs]), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qt_array = mic_fab_vars[MicVar::qt]->array(mfi);
            auto qn_array = mic_fab_vars[MicVar::qn]->array(mfi);
            auto qv_array = mic_fab_vars[MicVar::qv]->array(mfi);
            auto qcl_array = mic_fab_vars[MicVar::qcl]->array(mfi);
            auto qci_array = mic_fab_vars[MicVar::qci]->array(mfi);

            auto tabs_array = mic_fab_vars[MicVar::tabs]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar::theta]->array(mfi);
            auto pres_array = mic_fab_vars[MicVar::pres]->array(mfi);

            auto tbx = mfi.tilebox();
            if (tbx.smallEnd(0) == i_lo) { tbx.growLo(0, -m_real_width); }
            if (tbx.bigEnd(0) == i_hi) { tbx.growHi(0, -m_real_width); }
            if (tbx.smallEnd(1) == j_lo) { tbx.growLo(1, -m_real_width); }
            if (tbx.bigEnd(1) == j_hi) { tbx.growHi(1, -m_real_width); }

            amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real omn;
                amrex::Real qsat;
                amrex::Real qsatw;
                amrex::Real qsati;
                amrex::Real delta_qv;
                amrex::Real delta_qc = amrex::Real(0.0);
                amrex::Real delta_qi = amrex::Real(0.0);

                omn = one;
                if (SAM_moisture_type == 1) {
                    if (tabs_array(i,j,k) >= tbgmax) {
                        sam_bfb_record_hit(branch_hits, k_cloud_t_bgmax_or_greater);
                        omn = one;
                        delta_qi = qci_array(i,j,k);
                        qci_array(i,j,k) = zero;
                        qcl_array(i,j,k) += delta_qi;
                        tabs_array(i,j,k) -= fac_fus * delta_qi;
                        theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), pres_array(i,j,k), rdOcp);
                    } else if (tabs_array(i,j,k) <= tbgmin) {
                        sam_bfb_record_hit(branch_hits, k_cloud_t_bgmin_or_less);
                        omn = zero;
                        delta_qc = qcl_array(i,j,k);
                        qcl_array(i,j,k) = zero;
                        qci_array(i,j,k) += delta_qc;
                        tabs_array(i,j,k) += fac_fus * delta_qc;
                        theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), pres_array(i,j,k), rdOcp);
                    } else {
                        sam_bfb_record_hit(branch_hits, k_cloud_t_between_bgmin_bgmax);
                        omn = an * tabs_array(i,j,k) - bn;
                        delta_qc = qcl_array(i,j,k) - qn_array(i,j,k) * omn;
                        delta_qi = qci_array(i,j,k) - qn_array(i,j,k) * (one - omn);
                        qcl_array(i,j,k) = qn_array(i,j,k) * omn;
                        qci_array(i,j,k) = qn_array(i,j,k) * (one - omn);
                        tabs_array(i,j,k) += fac_fus * delta_qc;
                        theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), pres_array(i,j,k), rdOcp);
                    }
                } else if (SAM_moisture_type == 2) {
                    delta_qc = qcl_array(i,j,k) - qn_array(i,j,k);
                    delta_qi = zero;
                    qcl_array(i,j,k) = qn_array(i,j,k);
                    qci_array(i,j,k) = zero;
                    tabs_array(i,j,k) += fac_cond * delta_qc;
                    theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), pres_array(i,j,k), rdOcp);
                }

                erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsatw);
                erf_qsati(tabs_array(i,j,k), pres_array(i,j,k), qsati);
                qsat = omn * qsatw + (one - omn) * qsati;

                if (qt_array(i,j,k) > qsat) {
                    sam_bfb_record_hit(branch_hits, k_cloud_qt_gt_qsat);
                    tabs_array(i,j,k) = NewtonIterSat(i, j, k, SAM_moisture_type,
                                                      fac_cond, fac_fus, fac_sub,
                                                      an, bn,
                                                      tabs_array, pres_array,
                                                      qv_array, qcl_array, qci_array,
                                                      qn_array, qt_array,
                                                      branch_hits);
                    theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), amrex::Real(100.0) * pres_array(i,j,k), rdOcp);
                } else {
                    sam_bfb_record_hit(branch_hits, k_cloud_qt_le_qsat);
                    sam_bfb_record_hit(branch_hits, k_cloud_evap_all_then_recheck);
                    delta_qv = qcl_array(i,j,k) + qci_array(i,j,k);
                    delta_qc = qcl_array(i,j,k);
                    delta_qi = qci_array(i,j,k);

                    qv_array(i,j,k) += delta_qv;
                    qcl_array(i,j,k) = zero;
                    qci_array(i,j,k) = zero;
                    qn_array(i,j,k) = zero;
                    qt_array(i,j,k) = qv_array(i,j,k);

                    tabs_array(i,j,k) -= fac_cond * delta_qc + fac_sub * delta_qi;
                    theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), amrex::Real(100.0) * pres_array(i,j,k), rdOcp);

                    erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsatw);
                    erf_qsati(tabs_array(i,j,k), pres_array(i,j,k), qsati);
                    qsat = omn * qsatw + (one - omn) * qsati;
                    if (qt_array(i,j,k) > qsat) {
                        tabs_array(i,j,k) = NewtonIterSat(i, j, k, SAM_moisture_type,
                                                          fac_cond, fac_fus, fac_sub,
                                                          an, bn,
                                                          tabs_array, pres_array,
                                                          qv_array, qcl_array, qci_array,
                                                          qn_array, qt_array,
                                                          branch_hits);
                        theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), amrex::Real(100.0) * pres_array(i,j,k), rdOcp);
                    }
                }
            });
        }
    }

    void IceFall (const SolverChoice& sc)
    {
        if (sc.moisture_type == MoistureType::SAM_NoIce ||
            sc.moisture_type == MoistureType::SAM_NoPrecip_NoIce) {
            sam_bfb_accumulate_hit(m_branch_hits_host, k_icefall_noop, 1);
            return;
        }

        amrex::Real dtn = dt;
        amrex::Real coef = dtn / m_dzmin;

        const auto domain = m_geom.Domain();
        const int k_lo = domain.smallEnd(2);
        const int k_hi = domain.bigEnd(2);

        auto qcl = mic_fab_vars[MicVar::qcl];
        auto qci = mic_fab_vars[MicVar::qci];
        auto qn = mic_fab_vars[MicVar::qn];
        auto qt = mic_fab_vars[MicVar::qt];
        auto rho = mic_fab_vars[MicVar::rho];
        amrex::Long* branch_hits = m_branch_hits_device.data();

        amrex::MultiFab fz;
        const amrex::IntVect ng = qcl->nGrowVect();
        const amrex::BoxArray ba = qcl->boxArray();
        const amrex::DistributionMapping dm = qcl->DistributionMap();
        fz.define(convert(ba, amrex::IntVect(0,0,1)), dm, 1, ng);
        fz.setVal(amrex::Real(0.0));

        for (amrex::MFIter mfi(fz, TileNoZ()); mfi.isValid(); ++mfi) {
            auto qci_array = qci->array(mfi);
            auto rho_array = rho->array(mfi);
            auto fz_array = fz.array(mfi);

            const auto& box3d = mfi.tilebox();
            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real rho_avg;
                amrex::Real qci_avg;
                if (k == k_lo) {
                    sam_bfb_record_hit(branch_hits, k_icefall_bottom_face);
                    rho_avg = rho_array(i,j,k);
                    qci_avg = qci_array(i,j,k);
                } else if (k == k_hi + 1) {
                    sam_bfb_record_hit(branch_hits, k_icefall_top_face);
                    rho_avg = rho_array(i,j,k-1);
                    qci_avg = qci_array(i,j,k-1);
                } else {
                    sam_bfb_record_hit(branch_hits, k_icefall_interior_face);
                    rho_avg = myhalf * (rho_array(i,j,k-1) + rho_array(i,j,k));
                    qci_avg = myhalf * (qci_array(i,j,k-1) + qci_array(i,j,k));
                }
                const amrex::Real vt_ice = amrex::min(amrex::Real(0.4),
                                                      amrex::Real(8.66) * std::pow((amrex::max(amrex::Real(0.0), qci_avg) + amrex::Real(1.e-10)), amrex::Real(0.24)));
                fz_array(i,j,k) = rho_avg * vt_ice * qci_avg;
            });
        }

        const auto& ma_fz_arr = fz.const_arrays();
        amrex::GpuTuple<amrex::Real> max = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpMax>{},
            amrex::TypeList<amrex::Real>{},
            fz, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
                return {ma_fz_arr[box_no](i,j,k)};
            });
        amrex::Real wt_max = amrex::get<0>(max) + std::numeric_limits<amrex::Real>::epsilon();
        int n_substep = int(std::ceil(wt_max * coef / CFL_MAX));
        AMREX_ALWAYS_ASSERT(n_substep >= 1);
        coef /= amrex::Real(n_substep);
        dtn /= amrex::Real(n_substep);

        for (int nsub = 0; nsub < n_substep; ++nsub) {
            for (amrex::MFIter mfi(*qci, TileNoZ()); mfi.isValid(); ++mfi) {
                auto qci_array = qci->array(mfi);
                auto qn_array = qn->array(mfi);
                auto qt_array = qt->array(mfi);
                auto rho_array = rho->array(mfi);
                auto fz_array = fz.array(mfi);
                const auto dJ_array = (m_detJ_cc) ? m_detJ_cc->const_array(mfi) : amrex::Array4<const amrex::Real>{};

                const auto& tbx = mfi.tilebox();
                const auto& tbz = mfi.tilebox(amrex::IntVect(0,0,1), amrex::IntVect(0));

                amrex::ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    amrex::Real rho_avg;
                    amrex::Real qci_avg;
                    if (k == k_lo) {
                        rho_avg = rho_array(i,j,k);
                        qci_avg = qci_array(i,j,k);
                    } else if (k == k_hi + 1) {
                        rho_avg = rho_array(i,j,k-1);
                        qci_avg = qci_array(i,j,k-1);
                    } else {
                        rho_avg = myhalf * (rho_array(i,j,k-1) + rho_array(i,j,k));
                        qci_avg = myhalf * (qci_array(i,j,k-1) + qci_array(i,j,k));
                    }
                    const amrex::Real vt_ice = amrex::min(amrex::Real(0.4),
                                                          amrex::Real(8.66) * std::pow((amrex::max(amrex::Real(0.0), qci_avg) + amrex::Real(1.e-10)), amrex::Real(0.24)));
                    fz_array(i,j,k) = rho_avg * vt_ice * qci_avg;
                });

                amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const amrex::Real dJinv = (dJ_array) ? one / dJ_array(i,j,k) : one;
                    if (dJ_array) {
                        sam_bfb_record_hit(branch_hits, k_icefall_detj_present);
                    }
                    amrex::Real dqi = dJinv * (one / rho_array(i,j,k)) * (fz_array(i,j,k+1) - fz_array(i,j,k)) * coef;
                    dqi = std::max(-qci_array(i,j,k), dqi);
                    qci_array(i,j,k) += dqi;
                    qn_array(i,j,k) += dqi;
                    qt_array(i,j,k) += dqi;
                });
            }
        }
    }

    void Precip (const SolverChoice& sc)
    {
        if (sc.moisture_type == MoistureType::SAM_NoPrecip_NoIce) {
            sam_bfb_accumulate_hit(m_branch_hits_host, k_precip_noop, 1);
            return;
        }

        const amrex::Real powr1 = (three + b_rain) / amrex::Real(4.0);
        const amrex::Real powr2 = (amrex::Real(5.0) + b_rain) / amrex::Real(8.0);
        const amrex::Real pows1 = (three + b_snow) / amrex::Real(4.0);
        const amrex::Real pows2 = (amrex::Real(5.0) + b_snow) / amrex::Real(8.0);
        const amrex::Real powg1 = (three + b_grau) / amrex::Real(4.0);
        const amrex::Real powg2 = (amrex::Real(5.0) + b_grau) / amrex::Real(8.0);

        auto accrrc_t = accrrc.table();
        auto accrsc_t = accrsc.table();
        auto accrsi_t = accrsi.table();
        auto accrgc_t = accrgc.table();
        auto accrgi_t = accrgi.table();
        auto coefice_t = coefice.table();
        auto evapr1_t = evapr1.table();
        auto evapr2_t = evapr2.table();
        auto evaps1_t = evaps1.table();
        auto evaps2_t = evaps2.table();
        auto evapg1_t = evapg1.table();
        auto evapg2_t = evapg2.table();

        const amrex::Real fac_cond = m_fac_cond;
        const amrex::Real fac_sub = m_fac_sub;
        const amrex::Real fac_fus = m_fac_fus;
        const amrex::Real rdOcp = m_rdOcp;
        const amrex::Real eps = std::numeric_limits<amrex::Real>::epsilon();
        const amrex::Real dtn = dt;
        amrex::Long* branch_hits = m_branch_hits_device.data();

        int SAM_moisture_type = 1;
        if (sc.moisture_type == MoistureType::SAM_NoIce) {
            SAM_moisture_type = 2;
        }

        const auto domain = m_geom.Domain();
        const int i_lo = domain.smallEnd(0);
        const int i_hi = domain.bigEnd(0);
        const int j_lo = domain.smallEnd(1);
        const int j_hi = domain.bigEnd(1);

        for (amrex::MFIter mfi(*(mic_fab_vars[MicVar::tabs]), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto theta_array = mic_fab_vars[MicVar::theta]->array(mfi);
            auto tabs_array = mic_fab_vars[MicVar::tabs]->array(mfi);
            auto pres_array = mic_fab_vars[MicVar::pres]->array(mfi);

            auto qv_array = mic_fab_vars[MicVar::qv]->array(mfi);
            auto qcl_array = mic_fab_vars[MicVar::qcl]->array(mfi);
            auto qci_array = mic_fab_vars[MicVar::qci]->array(mfi);
            auto qn_array = mic_fab_vars[MicVar::qn]->array(mfi);
            auto qt_array = mic_fab_vars[MicVar::qt]->array(mfi);

            auto qpr_array = mic_fab_vars[MicVar::qpr]->array(mfi);
            auto qps_array = mic_fab_vars[MicVar::qps]->array(mfi);
            auto qpg_array = mic_fab_vars[MicVar::qpg]->array(mfi);
            auto qp_array = mic_fab_vars[MicVar::qp]->array(mfi);

            auto tbx = mfi.tilebox();
            if (tbx.smallEnd(0) == i_lo) { tbx.growLo(0, -m_real_width); }
            if (tbx.bigEnd(0) == i_hi) { tbx.growHi(0, -m_real_width); }
            if (tbx.smallEnd(1) == j_lo) { tbx.growLo(1, -m_real_width); }
            if (tbx.bigEnd(1) == j_hi) { tbx.growHi(1, -m_real_width); }

            amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real omn, omp, omg;
                amrex::Real qsat, qsatw, qsati;
                amrex::Real qcc, qii, qpr, qps, qpg;
                amrex::Real dprc, dpsc, dpgc;
                amrex::Real dpsi, dpgi;
                amrex::Real dqc, dqca, dqi, dqia, dqp;
                amrex::Real dqpr, dqps, dqpg;
                amrex::Real auto_r, autos;
                amrex::Real accrcr, accrcs, accris, accrcg, accrig;

                if (qn_array(i,j,k) + qp_array(i,j,k) > zero) {
                    const bool has_cloud = qn_array(i,j,k) > zero;
                    const bool has_precip = qp_array(i,j,k) > zero;
                    if (has_cloud && !has_precip) {
                        sam_bfb_record_hit(branch_hits, k_precip_cloud_only);
                    } else if (!has_cloud && has_precip) {
                        sam_bfb_record_hit(branch_hits, k_precip_only);
                    } else if (has_cloud && has_precip) {
                        sam_bfb_record_hit(branch_hits, k_precip_cloud_and_precip);
                    }

                    if (SAM_moisture_type == 2) {
                        omn = amrex::Real(1.0);
                        omp = amrex::Real(1.0);
                        omg = amrex::Real(0.0);
                    } else {
                        omn = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tabs_array(i,j,k) - tbgmin) * a_bg));
                        omp = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tabs_array(i,j,k) - tprmin) * a_pr));
                        omg = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tabs_array(i,j,k) - tgrmin) * a_gr));
                    }

                    qcc = qcl_array(i,j,k);
                    qii = qci_array(i,j,k);
                    qpr = qpr_array(i,j,k);
                    qps = qps_array(i,j,k);
                    qpg = qpg_array(i,j,k);

                    if (qn_array(i,j,k) > zero) {
                        accrcr = zero;
                        accrcs = zero;
                        accris = zero;
                        accrcg = zero;
                        accrig = zero;

                        auto_r = (qcc > qcw0) ? alphaelq : zero;
                        autos = (qii > qci0) ? betaelq * coefice_t(k) : zero;
                        if (auto_r > zero) {
                            sam_bfb_record_hit(branch_hits, k_precip_water_autoconversion_on);
                        } else {
                            sam_bfb_record_hit(branch_hits, k_precip_water_autoconversion_off);
                        }
                        if (SAM_moisture_type == 1) {
                            if (autos > zero) {
                                sam_bfb_record_hit(branch_hits, k_precip_ice_autoconversion_on);
                            } else {
                                sam_bfb_record_hit(branch_hits, k_precip_ice_autoconversion_off);
                            }
                        }
                        if (omp > amrex::Real(0.001)) {
                            accrcr = accrrc_t(k);
                            sam_bfb_record_hit(branch_hits, k_precip_rain_accretion_on);
                        } else {
                            sam_bfb_record_hit(branch_hits, k_precip_rain_accretion_off);
                        }
                        if (omp < amrex::Real(0.999) && omg < amrex::Real(0.999)) {
                            accrcs = accrsc_t(k);
                            accris = accrsi_t(k);
                            sam_bfb_record_hit(branch_hits, k_precip_snow_accretion_on);
                        } else {
                            sam_bfb_record_hit(branch_hits, k_precip_snow_accretion_off);
                        }
                        if (omp < amrex::Real(0.999) && omg > amrex::Real(0.001)) {
                            accrcg = accrgc_t(k);
                            accrig = accrgi_t(k);
                            sam_bfb_record_hit(branch_hits, k_precip_graupel_accretion_on);
                        } else {
                            sam_bfb_record_hit(branch_hits, k_precip_graupel_accretion_off);
                        }

                        dqca = dtn * auto_r * (qcc - qcw0);
                        dprc = dtn * accrcr * qcc * std::pow(qpr, powr1);
                        dpsc = dtn * accrcs * qcc * std::pow(qps, pows1);
                        dpgc = dtn * accrcg * qcc * std::pow(qpg, powg1);
                        dqia = dtn * autos * (qii - qci0);
                        dpsi = dtn * accris * qii * std::pow(qps, pows1);
                        dpgi = dtn * accrig * qii * std::pow(qpg, powg1);

                        dqc = dqca + dprc + dpsc + dpgc;
                        dqi = dqia + dpsi + dpgi;
                        const amrex::Real scalec = std::min(qcl_array(i,j,k), dqc) / (dqc + eps);
                        const amrex::Real scalei = std::min(qci_array(i,j,k), dqi) / (dqi + eps);
                        if (scalec < one) {
                            sam_bfb_record_hit(branch_hits, k_precip_sink_limited_cloud_water);
                        }
                        if (scalei < one) {
                            sam_bfb_record_hit(branch_hits, k_precip_sink_limited_cloud_ice);
                        }
                        dqca *= scalec;
                        dprc *= scalec;
                        dpsc *= scalec;
                        dpgc *= scalec;
                        dqia *= scalei;
                        dpsi *= scalei;
                        dpgi *= scalei;
                        dqc = dqca + dprc + dpsc + dpgc;
                        dqi = dqia + dpsi + dpgi;

                        dqpr = (dqca + dqia) * omp + dprc;
                        dqps = (dqca + dqia) * (one - omp) * (one - omg) + dpsc + dpsi;
                        dqpg = (dqca + dqia) * (one - omp) * omg + dpgc + dpgi;

                        qcl_array(i,j,k) -= dqc;
                        qci_array(i,j,k) -= dqi;
                        qpr_array(i,j,k) += dqpr;
                        qps_array(i,j,k) += dqps;
                        qpg_array(i,j,k) += dqpg;

                        qn_array(i,j,k) = qcl_array(i,j,k) + qci_array(i,j,k);
                        qt_array(i,j,k) = qv_array(i,j,k) + qn_array(i,j,k);
                        qp_array(i,j,k) = qpr_array(i,j,k) + qps_array(i,j,k) + qpg_array(i,j,k);

                        tabs_array(i,j,k) += fac_fus * (dqca * (one - omp) - dqia * omp);
                        theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), amrex::Real(100.0) * pres_array(i,j,k), rdOcp);
                    }

                    erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsatw);
                    erf_qsati(tabs_array(i,j,k), pres_array(i,j,k), qsati);
                    qsat = qsatw * omn + qsati * (one - omn);
                    if ((qp_array(i,j,k) > zero) && (qv_array(i,j,k) < qsat)) {
                        sam_bfb_record_hit(branch_hits, k_precip_evaporation_on);
                        dqpr = evapr1_t(k) * std::sqrt(qpr) + evapr2_t(k) * std::pow(qpr, powr2);
                        dqps = evaps1_t(k) * std::sqrt(qps) + evaps2_t(k) * std::pow(qps, pows2);
                        dqpg = evapg1_t(k) * std::sqrt(qpg) + evapg2_t(k) * std::pow(qpg, powg2);

                        dqpr *= dtn * (one - qv_array(i,j,k) / qsat);
                        dqps *= dtn * (one - qv_array(i,j,k) / qsat);
                        dqpg *= dtn * (one - qv_array(i,j,k) / qsat);
                        if (dqpr > qpr_array(i,j,k) || dqps > qps_array(i,j,k) || dqpg > qpg_array(i,j,k)) {
                            sam_bfb_record_hit(branch_hits, k_precip_evaporation_species_limited);
                        }

                        dqpr = std::min(qpr_array(i,j,k), dqpr);
                        dqps = std::min(qps_array(i,j,k), dqps);
                        dqpg = std::min(qpg_array(i,j,k), dqpg);
                        dqp = dqpr + dqps + dqpg;

                        qv_array(i,j,k) += dqp;
                        qpr_array(i,j,k) -= dqpr;
                        qps_array(i,j,k) -= dqps;
                        qpg_array(i,j,k) -= dqpg;

                        qt_array(i,j,k) = qv_array(i,j,k) + qn_array(i,j,k);
                        qp_array(i,j,k) = qpr_array(i,j,k) + qps_array(i,j,k) + qpg_array(i,j,k);

                        tabs_array(i,j,k) -= fac_cond * dqpr + fac_sub * (dqps + dqpg);
                        theta_array(i,j,k) = getThgivenTandP(tabs_array(i,j,k), amrex::Real(100.0) * pres_array(i,j,k), rdOcp);
                    } else if (qp_array(i,j,k) > zero) {
                        sam_bfb_record_hit(branch_hits, k_precip_evaporation_off);
                    }
                }
            });
        }
    }

    void PrecipFall (const SolverChoice& sc)
    {
        if (sc.moisture_type == MoistureType::SAM_NoPrecip_NoIce) {
            sam_bfb_accumulate_hit(m_branch_hits_host, k_precipfall_noop, 1);
            return;
        }

        const amrex::Real rho_0 = amrex::Real(1.29);
        const amrex::Real gamr3 = erf_gammafff(amrex::Real(4.0) + b_rain);
        const amrex::Real gams3 = erf_gammafff(amrex::Real(4.0) + b_snow);
        const amrex::Real gamg3 = erf_gammafff(amrex::Real(4.0) + b_grau);

        const amrex::Real vrain = (a_rain * gamr3 / amrex::Real(6.0)) * std::pow((PI * rhor * nzeror), -crain);
        const amrex::Real vsnow = (a_snow * gams3 / amrex::Real(6.0)) * std::pow((PI * rhos * nzeros), -csnow);
        const amrex::Real vgrau = (a_grau * gamg3 / amrex::Real(6.0)) * std::pow((PI * rhog * nzerog), -cgrau);

        amrex::Real dtn = dt;
        amrex::Real coef = dtn / m_dzmin;

        const auto domain = m_geom.Domain();
        const int k_lo = domain.smallEnd(2);
        const int k_hi = domain.bigEnd(2);

        auto qpr = mic_fab_vars[MicVar::qpr];
        auto qps = mic_fab_vars[MicVar::qps];
        auto qpg = mic_fab_vars[MicVar::qpg];
        auto qp = mic_fab_vars[MicVar::qp];
        auto rho = mic_fab_vars[MicVar::rho];
        auto tabs = mic_fab_vars[MicVar::tabs];
        auto rain_accum = mic_fab_vars[MicVar::rain_accum];
        auto snow_accum = mic_fab_vars[MicVar::snow_accum];
        auto graup_accum = mic_fab_vars[MicVar::graup_accum];
        amrex::Long* branch_hits = m_branch_hits_device.data();

        amrex::MultiFab fz;
        fz.define(convert(tabs->boxArray(), amrex::IntVect(0,0,1)), tabs->DistributionMap(), 1, tabs->nGrowVect());

        int SAM_moisture_type = 1;
        if (sc.moisture_type == MoistureType::SAM_NoIce) {
            SAM_moisture_type = 2;
        }

        for (amrex::MFIter mfi(fz, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qp_array = qp->array(mfi);
            auto rho_array = rho->array(mfi);
            auto tabs_array = tabs->array(mfi);
            auto fz_array = fz.array(mfi);

            const auto& box3d = mfi.tilebox();
            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real rho_avg;
                amrex::Real tab_avg;
                amrex::Real qp_avg;
                if (k == k_lo) {
                    sam_bfb_record_hit(branch_hits, k_precipfall_bottom_face);
                    rho_avg = rho_array(i,j,k);
                    tab_avg = tabs_array(i,j,k);
                    qp_avg = qp_array(i,j,k);
                } else if (k == k_hi + 1) {
                    sam_bfb_record_hit(branch_hits, k_precipfall_top_face);
                    rho_avg = rho_array(i,j,k-1);
                    tab_avg = tabs_array(i,j,k-1);
                    qp_avg = qp_array(i,j,k-1);
                } else {
                    sam_bfb_record_hit(branch_hits, k_precipfall_interior_face);
                    rho_avg = myhalf * (rho_array(i,j,k-1) + rho_array(i,j,k));
                    tab_avg = myhalf * (tabs_array(i,j,k-1) + tabs_array(i,j,k));
                    qp_avg = myhalf * (qp_array(i,j,k-1) + qp_array(i,j,k));
                }

                amrex::Real Pprecip = zero;
                if (qp_avg > qp_threshold) {
                    amrex::Real omp;
                    amrex::Real omg;
                    if (SAM_moisture_type == 2) {
                        omp = one;
                        omg = zero;
                    } else {
                        omp = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tab_avg - tprmin) * a_pr));
                        omg = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tab_avg - tgrmin) * a_gr));
                    }
                    const amrex::Real qrr = omp * qp_avg;
                    const amrex::Real qss = (one - omp) * (one - omg) * qp_avg;
                    const amrex::Real qgg = (one - omp) * omg * qp_avg;
                    const amrex::Real rain_flux = omp * vrain * std::pow(rho_avg * qrr, one + crain);
                    const amrex::Real snow_flux = (one - omp) * (one - omg) * vsnow * std::pow(rho_avg * qss, one + csnow);
                    const amrex::Real graupel_flux = (one - omp) * omg * vgrau * std::pow(rho_avg * qgg, one + cgrau);
                    if (rain_flux > zero) {
                        sam_bfb_record_hit(branch_hits, k_precipfall_nonzero_rain_flux);
                    }
                    if (snow_flux > zero) {
                        sam_bfb_record_hit(branch_hits, k_precipfall_nonzero_snow_flux);
                    }
                    if (graupel_flux > zero) {
                        sam_bfb_record_hit(branch_hits, k_precipfall_nonzero_graupel_flux);
                    }
                    Pprecip = omp * vrain * std::pow(rho_avg * qrr, one + crain)
                            + (one - omp) * ((one - omg) * vsnow * std::pow(rho_avg * qss, one + csnow)
                                           + omg * vgrau * std::pow(rho_avg * qgg, one + cgrau));
                } else {
                    sam_bfb_record_hit(branch_hits, k_precipfall_zero_flux_below_threshold);
                }
                fz_array(i,j,k) = Pprecip * std::sqrt(rho_0 / rho_avg);
            });
        }

        const auto& ma_fz_arr = fz.const_arrays();
        amrex::GpuTuple<amrex::Real> max = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpMax>{},
            amrex::TypeList<amrex::Real>{},
            fz, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
                return {ma_fz_arr[box_no](i,j,k)};
            });
        amrex::Real wt_max = amrex::get<0>(max) + std::numeric_limits<amrex::Real>::epsilon();
        int n_substep = int(std::ceil(wt_max * coef / CFL_MAX));
        AMREX_ALWAYS_ASSERT(n_substep >= 1);
        coef /= amrex::Real(n_substep);
        dtn /= amrex::Real(n_substep);

        for (int nsub = 0; nsub < n_substep; ++nsub) {
            for (amrex::MFIter mfi(*qp, TileNoZ()); mfi.isValid(); ++mfi) {
                auto qpr_array = qpr->array(mfi);
                auto qps_array = qps->array(mfi);
                auto qpg_array = qpg->array(mfi);
                auto qp_array = qp->array(mfi);
                auto rho_array = rho->array(mfi);
                auto tabs_array = tabs->array(mfi);
                auto fz_array = fz.array(mfi);
                auto rain_accum_array = rain_accum->array(mfi);
                auto snow_accum_array = snow_accum->array(mfi);
                auto graup_accum_array = graup_accum->array(mfi);
                const auto dJ_array = (m_detJ_cc) ? m_detJ_cc->const_array(mfi) : amrex::Array4<const amrex::Real>{};

                const auto& tbx = mfi.tilebox();
                const auto& tbz = mfi.tilebox(amrex::IntVect(0,0,1), amrex::IntVect(0));

                amrex::ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    amrex::Real rho_avg;
                    amrex::Real tab_avg;
                    amrex::Real qp_avg;
                    if (k == k_lo) {
                        rho_avg = rho_array(i,j,k);
                        tab_avg = tabs_array(i,j,k);
                        qp_avg = qp_array(i,j,k);
                    } else if (k == k_hi + 1) {
                        rho_avg = rho_array(i,j,k-1);
                        tab_avg = tabs_array(i,j,k-1);
                        qp_avg = qp_array(i,j,k-1);
                    } else {
                        rho_avg = myhalf * (rho_array(i,j,k-1) + rho_array(i,j,k));
                        tab_avg = myhalf * (tabs_array(i,j,k-1) + tabs_array(i,j,k));
                        qp_avg = myhalf * (qp_array(i,j,k-1) + qp_array(i,j,k));
                    }

                    amrex::Real Pprecip = zero;
                    amrex::Real omp;
                    amrex::Real omg;
                    if (SAM_moisture_type == 2) {
                        omp = one;
                        omg = zero;
                    } else {
                        omp = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tab_avg - tprmin) * a_pr));
                        omg = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tab_avg - tgrmin) * a_gr));
                    }

                    if (qp_avg > qp_threshold) {
                        const amrex::Real qrr = omp * qp_avg;
                        const amrex::Real qss = (one - omp) * (one - omg) * qp_avg;
                        const amrex::Real qgg = (one - omp) * omg * qp_avg;
                        Pprecip = omp * vrain * std::pow(rho_avg * qrr, one + crain)
                                + (one - omp) * ((one - omg) * vsnow * std::pow(rho_avg * qss, one + csnow)
                                               + omg * vgrau * std::pow(rho_avg * qgg, one + cgrau));
                    }
                    fz_array(i,j,k) = Pprecip * std::sqrt(rho_0 / rho_avg);

                    if (k == k_lo) {
                        if (rho_avg * (omp * qp_avg) * vrain * dtn / rhor > zero) {
                            sam_bfb_record_hit(branch_hits, k_precipfall_surface_rain_accum);
                        }
                        if (rho_avg * (one - omp) * (one - omg) * qp_avg * vsnow * dtn / rhos > zero) {
                            sam_bfb_record_hit(branch_hits, k_precipfall_surface_snow_accum);
                        }
                        if (rho_avg * (one - omp) * omg * qp_avg * vgrau * dtn / rhog > zero) {
                            sam_bfb_record_hit(branch_hits, k_precipfall_surface_graupel_accum);
                        }
                        rain_accum_array(i,j,k) = rain_accum_array(i,j,k) + rho_avg * (omp * qp_avg) * vrain * dtn / rhor * amrex::Real(1000.0);
                        snow_accum_array(i,j,k) = snow_accum_array(i,j,k) + rho_avg * (one - omp) * (one - omg) * qp_avg * vsnow * dtn / rhos * amrex::Real(1000.0);
                        graup_accum_array(i,j,k) = graup_accum_array(i,j,k) + rho_avg * (one - omp) * omg * qp_avg * vgrau * dtn / rhog * amrex::Real(1000.0);
                    }
                });

                amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const amrex::Real dJinv = (dJ_array) ? one / dJ_array(i,j,k) : one;
                    if (dJ_array) {
                        sam_bfb_record_hit(branch_hits, k_precipfall_detj_present);
                    }
                    const amrex::Real dqp = dJinv * (one / rho_array(i,j,k)) * (fz_array(i,j,k+1) - fz_array(i,j,k)) * coef;
                    amrex::Real omp;
                    amrex::Real omg;
                    if (SAM_moisture_type == 2) {
                        omp = one;
                        omg = zero;
                    } else {
                        omp = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tabs_array(i,j,k) - tprmin) * a_pr));
                        omg = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), (tabs_array(i,j,k) - tgrmin) * a_gr));
                    }

                    if (qpr_array(i,j,k) + dqp * omp < zero ||
                        qps_array(i,j,k) + dqp * (one - omp) * (one - omg) < zero ||
                        qpg_array(i,j,k) + dqp * (one - omp) * omg < zero) {
                        sam_bfb_record_hit(branch_hits, k_precipfall_nonnegative_clipping);
                    }

                    qpr_array(i,j,k) = std::max(amrex::Real(0.0), qpr_array(i,j,k) + dqp * omp);
                    qps_array(i,j,k) = std::max(amrex::Real(0.0), qps_array(i,j,k) + dqp * (one - omp) * (one - omg));
                    qpg_array(i,j,k) = std::max(amrex::Real(0.0), qpg_array(i,j,k) + dqp * (one - omp) * omg);
                    qp_array(i,j,k) = qpr_array(i,j,k) + qps_array(i,j,k) + qpg_array(i,j,k);
                });
            }
        }
    }

    int m_qmoist_size = 3;
    static constexpr amrex::Real CFL_MAX = myhalf;

    amrex::Vector<int> MicVarMap;
    amrex::Geometry m_geom;
    int m_real_width{0};
    amrex::Real dt{amrex::Real(0.0)};
    int nlev{0};
    int zlo{0};
    int zhi{0};
    int m_axis{2};
    amrex::Real m_fac_cond{amrex::Real(0.0)};
    amrex::Real m_fac_fus{amrex::Real(0.0)};
    amrex::Real m_fac_sub{amrex::Real(0.0)};
    amrex::Real m_rdOcp{amrex::Real(0.0)};
    bool m_do_cond{true};
    amrex::Real m_dzmin{amrex::Real(1.0)};
    amrex::MultiFab* m_z_phys_nd{nullptr};
    amrex::MultiFab* m_detJ_cc{nullptr};
    amrex::Array<FabPtr, MicVar::NumVars> mic_fab_vars;
    amrex::TableData<amrex::Real, 1> accrrc;
    amrex::TableData<amrex::Real, 1> accrsi;
    amrex::TableData<amrex::Real, 1> accrsc;
    amrex::TableData<amrex::Real, 1> coefice;
    amrex::TableData<amrex::Real, 1> evaps1;
    amrex::TableData<amrex::Real, 1> evaps2;
    amrex::TableData<amrex::Real, 1> accrgi;
    amrex::TableData<amrex::Real, 1> accrgc;
    amrex::TableData<amrex::Real, 1> evapg1;
    amrex::TableData<amrex::Real, 1> evapg2;
    amrex::TableData<amrex::Real, 1> evapr1;
    amrex::TableData<amrex::Real, 1> evapr2;
    amrex::TableData<amrex::Real, 1> rho1d;
    amrex::TableData<amrex::Real, 1> pres1d;
    amrex::TableData<amrex::Real, 1> tabs1d;
    SAMBFBBranchHits m_branch_hits_host;
    amrex::Gpu::DeviceVector<amrex::Long> m_branch_hits_device;
};

template <typename SamType>
inline void run_public_flow (SamType& sam,
                             SolverChoice& sc,
                             const amrex::Geometry& geom,
                             amrex::MultiFab& cons,
                             std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                             std::unique_ptr<amrex::MultiFab>& detJ_cc,
                             const int real_width,
                             const amrex::Real dt = kDefaultDt)
{
    run_and_sync([&] {
        sam.Define(sc);
        sam.Set_RealWidth(real_width);
        sam.Set_dzmin(geom.CellSize(2));
        sam.Init(cons, cons.boxArray(), geom, dt, z_phys_nd, detJ_cc);
        sam.Copy_State_to_Micro(cons);
        sam.Compute_Coefficients();
        sam.Advance(dt, sc);
        sam.Copy_Micro_to_State(cons);
    });
}

template <typename FillStateFn>
inline void initialize_case (const amrex::Geometry& geom,
                             const FillStateFn& fill_state,
                             const int ngrow,
                             amrex::MultiFab& cons_reference,
                             amrex::MultiFab& cons_production)
{
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    cons_reference.define(ba, dm, RhoQ6_comp + 1, ngrow);
    cons_production.define(ba, dm, RhoQ6_comp + 1, ngrow);
    cons_reference.setVal(amrex::Real(0.0));
    cons_production.setVal(amrex::Real(0.0));
    fill_state(cons_reference);
    copy_multifab(cons_reference, cons_production);
}

inline void initialize_optional_detj (const amrex::MultiFab& template_mf,
                                      const bool use_detj,
                                      const bool compressed_detj,
                                      std::unique_ptr<amrex::MultiFab>& detj)
{
    if (!use_detj) {
        detj.reset();
        return;
    }

    detj = std::make_unique<amrex::MultiFab>(template_mf.boxArray(), template_mf.DistributionMap(),
                                             1, template_mf.nGrowVect());
    detj->setVal(amrex::Real(1.0));
    if (compressed_detj) {
        fill_compressed_detj_portable(*detj);
    } else {
        fill_detj_portable(*detj);
    }
}

} // namespace sam_test

#endif