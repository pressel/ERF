#include "ERF_ShocDriver.H"
#include "ERF_ShocImplicit.H"

#include "ERF_IndexDefines.H"

#include <AMReX_BLProfiler.H>

#include <iomanip>

using namespace amrex;

bool
shoc_boxarray_spans_full_height (const BoxArray& ba, const Box& domain)
{
    const int dom_klo = domain.smallEnd(2);
    const int dom_khi = domain.bigEnd(2);
    for (int ibox = 0; ibox < ba.size(); ++ibox) {
        const Box& bx = ba[ibox];
        if (bx.smallEnd(2) != dom_klo || bx.bigEnd(2) != dom_khi) {
            return false;
        }
    }
    return true;
}

namespace
{
    constexpr int k_shoc_vertical_diff_comp = EddyDiff::Mom_v;
    constexpr int k_shoc_vertical_diff_count = EddyDiff::Q_v - EddyDiff::Mom_v + 1;

    void require_full_height_shoc_boxes (const BoxArray& ba, const Box& domain)
    {
        // Tile-local SHOC packing and writeback still assume each MFIter tile
        // spans a complete vertical column.
        if (!shoc_boxarray_spans_full_height(ba, domain)) {
            amrex::Abort(
                "Native SHOC requires each BoxArray box on a SHOC-active level to span the full vertical domain. Z-split boxes are not supported. Increase the vertical max_grid_size/blocking size, or use full-column refined grids.");
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real weighted_linear_interp (Real x0, Real x1, Real y0, Real y1, Real x)
    {
        const Real denom = x1 - x0;
        if (amrex::Math::abs(denom) <= 1.0e-12_rt) {
            return 0.5_rt * (y0 + y1);
        }
        return y0 + (y1 - y0) * (x - x0) / denom;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int face_neighbor_cell (int idx, int domlo, int domhi, bool periodic)
    {
        return periodic ? idx : shoc_clamp(idx, domlo, domhi);
    }

    void seed_carried_turbulence_impl (
        Box const& xy_box,
        ShocColumnLayout layout,
        Array4<const Real> const& rho_host,
        Array4<const Real> const& carried,
        Array4<const Real> const& host_diff,
        Array4<Real> const& tk,
        Array4<Real> const& tkh,
        bool prev_turb_valid)
    {
        ParallelFor(xy_box, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            const int ic = shoc_column_index(layout, i, j);
            for (int kk = 0; kk < layout.nlev; ++kk) {
                const int k = layout.kmin + kk;
                if (prev_turb_valid) {
                    tk(ic,kk,0) = carried(i,j,k,0);
                    tkh(ic,kk,0) = carried(i,j,k,1);
                } else {
                    const Real rho = amrex::max(rho_host(i,j,k,Rho_comp), 1.0e-12_rt);
                    tk(ic,kk,0) = amrex::max(0.0_rt, host_diff(i,j,k,EddyDiff::Mom_v) / rho);
                    tkh(ic,kk,0) = amrex::max(0.0_rt, host_diff(i,j,k,EddyDiff::Theta_v) / rho);
                }
            }
        });
    }

    void store_carried_turbulence_impl (
        Box const& xy_box,
        ShocColumnLayout layout,
        Array4<Real> const& carried,
        Array4<const Real> const& tk,
        Array4<const Real> const& tkh)
    {
        ParallelFor(xy_box, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            const int ic = shoc_column_index(layout, i, j);
            for (int kk = 0; kk < layout.nlev; ++kk) {
                const int k = layout.kmin + kk;
                carried(i,j,k,0) = tk(ic,kk,0);
                carried(i,j,k,1) = tkh(ic,kk,0);
            }
        });
    }

    void seed_carried_buoyancy_flux_impl (
        Box const& xy_box,
        ShocColumnLayout layout,
        Array4<const Real> const& carried,
        Array4<Real> const& wthv_sec)
    {
        ParallelFor(xy_box, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            const int ic = shoc_column_index(layout, i, j);
            for (int kk = 0; kk < layout.nlev; ++kk) {
                const int k = layout.kmin + kk;
                wthv_sec(ic,kk,0) = carried(i,j,k,0);
            }
        });
    }

    void store_carried_buoyancy_flux_impl (
        Box const& xy_box,
        ShocColumnLayout layout,
        Array4<Real> const& carried,
        Array4<const Real> const& wthv_sec)
    {
        ParallelFor(xy_box, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            const int ic = shoc_column_index(layout, i, j);
            for (int kk = 0; kk < layout.nlev; ++kk) {
                const int k = layout.kmin + kk;
                carried(i,j,k,0) = wthv_sec(ic,kk,0);
            }
        });
    }

    void sync_face_multifab_impl (MultiFab& mf, const Geometry& geom)
    {
        // OverrideSync reconciles duplicate values on overlapping face-centered
        // tiles before SHOC reads or writes them. It is a no-op for cell-centered
        // MultiFabs, so the helper can be reused for the face-centered state and
        // tendency fields without special casing the index type.
        mf.OverrideSync(geom.periodicity());
    }

    void apply_state_update_cons_impl (MultiFab& cons,
                                       const MultiFab& theta_tend_cc,
                                       const MultiFab& qv_tend_cc,
                                       const MultiFab& qc_tend_cc,
                                       const MultiFab& qi_tend_cc,
                                       const MultiFab& tke_tend_cc,
                                       int qv_comp,
                                       int qc_comp,
                                       int qi_comp,
                                       Real dt)
    {
        const bool has_qv = shoc_valid_comp(qv_comp, cons.nComp());
        const bool has_qc = shoc_valid_comp(qc_comp, cons.nComp());
        const bool has_qi = shoc_valid_comp(qi_comp, cons.nComp());

        for (MFIter mfi(cons, false); mfi.isValid(); ++mfi) {
            const auto rho = cons.const_array(mfi);
            auto cc = cons.array(mfi);
            const auto theta_tend = theta_tend_cc.const_array(mfi);
            const auto qv_tend = qv_tend_cc.const_array(mfi);
            const auto qc_tend = qc_tend_cc.const_array(mfi);
            const auto qi_tend = qi_tend_cc.const_array(mfi);
            const auto tke_tend = tke_tend_cc.const_array(mfi);

            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real rho_val = rho(i,j,k,Rho_comp);
                cc(i,j,k,RhoTheta_comp) += rho_val * dt * theta_tend(i,j,k);
                if (has_qv) {
                    cc(i,j,k,qv_comp) = amrex::max(
                        cc(i,j,k,qv_comp) + rho_val * dt * qv_tend(i,j,k), 0.0_rt);
                }
                if (has_qc) {
                    cc(i,j,k,qc_comp) = amrex::max(
                        cc(i,j,k,qc_comp) + rho_val * dt * qc_tend(i,j,k), 0.0_rt);
                }
                if (has_qi) {
                    cc(i,j,k,qi_comp) = amrex::max(
                        cc(i,j,k,qi_comp) + rho_val * dt * qi_tend(i,j,k), 0.0_rt);
                }
                cc(i,j,k,RhoKE_comp) = amrex::max(
                    cc(i,j,k,RhoKE_comp) + rho_val * dt * tke_tend(i,j,k), 0.0_rt);
            });
        }
    }

    void apply_state_update_face_velocity_impl (MultiFab& vel,
                                                const MultiFab& vel_tend,
                                                Real dt)
    {
        for (MFIter mfi(vel_tend, false); mfi.isValid(); ++mfi) {
            auto v = vel.array(mfi);
            const auto tend = vel_tend.const_array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                v(i,j,k) += dt * tend(i,j,k);
            });
        }
    }
}

ShocDriver::ShocDriver (int lev, const SolverChoice& solver_choice)
    : m_lev(lev),
      m_moisture_type(solver_choice.moisture_type),
      m_moisture_indices(solver_choice.moisture_indices)
{
    read_shoc_runtime_options(m_opts);
    validate_shoc_runtime_options(m_opts);

    if (uses_host_diffusion() && m_moisture_type != MoistureType::None) {
        amrex::Abort(
            "Native SHOC host_diffusion with moisture is not yet supported because SHOC does not own cloud macrophysics in this mode while SHOC-family microphysics condensation is suppressed. Use state_update for moist SHOC runs, or run host_diffusion only for dry cases until a transport-aware microphysics ownership predicate is implemented.");
    }
}

void
ShocDriver::ensure_storage (const MultiFab& cons,
                            const MultiFab& xvel,
                            const MultiFab& yvel,
                            const MultiFab& eddy_diffs)
{
    if (!m_theta_tend_cc.isDefined()) {
        m_theta_tend_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_qv_tend_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_qc_tend_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_qi_tend_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_tke_tend_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_u_tend_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 1);
        m_v_tend_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 1);
        m_u_tend_fc.define(xvel.boxArray(), xvel.DistributionMap(), 1, 0);
        m_v_tend_fc.define(yvel.boxArray(), yvel.DistributionMap(), 1, 0);
        m_eddy_coeffs_cc.define(eddy_diffs.boxArray(), eddy_diffs.DistributionMap(),
                                EddyDiff::NumDiffs, 0);
        m_prev_turb_cc.define(cons.boxArray(), cons.DistributionMap(), 2, 0);
        m_prev_wthv_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_pblh_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_shoc_cldfrac_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_shoc_ql_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_shoc_ql2_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_shoc_cond_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_w_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_wqls_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_wthv_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_thl_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_qw_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_qwthl_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_wthl_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_wqw_sec_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_w3_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_brunt_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_isotropy_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_shear_prod_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_buoy_prod_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_diss_tke_cc.define(cons.boxArray(), cons.DistributionMap(), 1, 0);
        m_prev_turb_cc.setVal(0.0);
        m_prev_wthv_sec_cc.setVal(0.0);
    }
    m_theta_tend_cc.setVal(0.0);
    m_qv_tend_cc.setVal(0.0);
    m_qc_tend_cc.setVal(0.0);
    m_qi_tend_cc.setVal(0.0);
    m_tke_tend_cc.setVal(0.0);
    m_u_tend_cc.setVal(0.0);
    m_v_tend_cc.setVal(0.0);
    m_u_tend_fc.setVal(0.0);
    m_v_tend_fc.setVal(0.0);
    m_eddy_coeffs_cc.setVal(0.0);
    m_pblh_cc.setVal(0.0);
    m_shoc_cldfrac_cc.setVal(0.0);
    m_shoc_ql_cc.setVal(0.0);
    m_shoc_ql2_cc.setVal(0.0);
    m_shoc_cond_cc.setVal(0.0);
    m_w_sec_cc.setVal(0.0);
    m_wqls_sec_cc.setVal(0.0);
    m_wthv_sec_cc.setVal(0.0);
    m_thl_sec_cc.setVal(0.0);
    m_qw_sec_cc.setVal(0.0);
    m_qwthl_sec_cc.setVal(0.0);
    m_wthl_sec_cc.setVal(0.0);
    m_wqw_sec_cc.setVal(0.0);
    m_w3_cc.setVal(0.0);
    m_brunt_cc.setVal(0.0);
    m_isotropy_cc.setVal(0.0);
    m_shear_prod_cc.setVal(0.0);
    m_buoy_prod_cc.setVal(0.0);
    m_diss_tke_cc.setVal(0.0);
}

void
ShocDriver::seed_carried_turbulence (ShocColumnData& col,
                                     const MFIter& mfi,
                                     const MultiFab& cons,
                                     const MultiFab& eddy_diffs) const
{
    const auto rho_host = cons.const_array(mfi);
    const auto carried = m_prev_turb_cc.const_array(mfi);
    const auto host_diff = eddy_diffs.const_array(mfi);
    auto tk = col.tk.array();
    auto tkh = col.tkh.array();
    const bool prev_turb_valid = m_prev_turb_valid;
    const auto layout = col.layout;
    const Box xy_box = amrex::makeSlab(mfi.validbox(), 2, layout.kmin);

    seed_carried_turbulence_impl(xy_box, layout, rho_host, carried, host_diff,
                                 tk, tkh, prev_turb_valid);
}

void
ShocDriver::store_carried_turbulence (const ShocColumnData& col,
                                      const MFIter& mfi)
{
    auto carried = m_prev_turb_cc.array(mfi);
    const auto tk = col.tk.const_array();
    const auto tkh = col.tkh.const_array();
    const auto layout = col.layout;
    const Box xy_box = amrex::makeSlab(mfi.validbox(), 2, layout.kmin);

    store_carried_turbulence_impl(xy_box, layout, carried, tk, tkh);
}

void
ShocDriver::seed_carried_buoyancy_flux (ShocColumnData& col,
                                        const MFIter& mfi) const
{
    if (!m_prev_turb_valid) {
        return;
    }

    const auto carried = m_prev_wthv_sec_cc.const_array(mfi);
    auto wthv_sec = col.wthv_sec.array();
    const auto layout = col.layout;
    const Box xy_box = amrex::makeSlab(mfi.validbox(), 2, layout.kmin);

    seed_carried_buoyancy_flux_impl(xy_box, layout, carried, wthv_sec);
}

void
ShocDriver::store_carried_buoyancy_flux (const ShocColumnData& col,
                                         const MFIter& mfi)
{
    auto carried = m_prev_wthv_sec_cc.array(mfi);
    const auto wthv_sec = col.wthv_sec.const_array();
    const auto layout = col.layout;
    const Box xy_box = amrex::makeSlab(mfi.validbox(), 2, layout.kmin);

    store_carried_buoyancy_flux_impl(xy_box, layout, carried, wthv_sec);
}

void
ShocDriver::advance (MultiFab& cons,
                     MultiFab& xvel,
                     MultiFab& yvel,
                     MultiFab& zvel,
                     MultiFab* tau13,
                     MultiFab* tau23,
                     MultiFab* hfx3,
                     MultiFab* qfx3,
                     MultiFab* eddy_diffs,
                     MultiFab& z_phys_nd,
                     const Geometry& geom,
                     Real dt)
{
    BL_PROFILE("SHOC::advance");

    require_full_height_shoc_boxes(cons.boxArray(), geom.Domain());

    m_cons_ptr = &cons;
    m_hfx3_ptr = hfx3;
    m_qfx3_ptr = qfx3;
    m_tau13_ptr = tau13;
    m_tau23_ptr = tau23;
    m_eddy_diffs_ptr = eddy_diffs;

    sync_face_multifab_impl(xvel, geom);
    sync_face_multifab_impl(yvel, geom);
    if (m_tau13_ptr) {
        sync_face_multifab_impl(*m_tau13_ptr, geom);
    }
    if (m_tau23_ptr) {
        sync_face_multifab_impl(*m_tau23_ptr, geom);
    }

    {
        BL_PROFILE("SHOC::advance::ensure_storage");
        ensure_storage(cons, xvel, yvel, *eddy_diffs);
    }

    if (uses_state_update() &&
        shoc_layout_requires_number_closure(m_moisture_indices, cons.nComp())) {
        amrex::Abort(
            "Native SHOC state_update does not yet support number-aware microphysics layouts with cloud/ice number concentrations. A number closure is required before SHOC can update cloud mass in these layouts.");
    }

    // Reused across the current host-serial MFIter loop to avoid repeated
    // SHOC scratch allocation. If this loop is parallelized on the host,
    // make the workspace thread-private.
    ShocColumnWorkspace workspace;

    for (MFIter mfi(cons, false); mfi.isValid(); ++mfi) {
        const Box& vbx = mfi.validbox();
        const ShocColumnLayout active_layout = make_shoc_layout(vbx, geom);
        {
            BL_PROFILE("SHOC::advance::define_column_data");
            workspace.ensure_capacity(active_layout, amrex::The_Async_Arena(),
                                      shoc::default_init_run_on());
        }
        ShocColumnData& col = workspace.col;
        {
            BL_PROFILE("SHOC::advance::preprocess");
            ShocPreprocess::fill_columns(col, mfi, cons, xvel, yvel, zvel,
                                         hfx3, qfx3, tau13, tau23, z_phys_nd, geom,
                                         m_moisture_indices);
        }
        {
            BL_PROFILE("SHOC::advance::seed_carried_buoyancy_flux");
            seed_carried_buoyancy_flux(col, mfi);
        }
        {
            BL_PROFILE("SHOC::advance::seed_carried_turbulence");
            seed_carried_turbulence(col, mfi, cons, *eddy_diffs);
        }
        const auto dx = geom.CellSizeArray();
        if (uses_state_update()) {
            BL_PROFILE("SHOC::advance::cache_baseline_state");
            ShocImplicit::cache_baseline_state(col);
        }
        ShocDiagnostics::diagnose_pre_implicit(col, m_opts, dx[0], dx[1], dt);
        if (uses_state_update()) {
            BL_PROFILE("SHOC::advance::implicit");
            ShocImplicit::advance_implicit_state(col, m_opts, dt);
        }
        ShocDiagnostics::diagnose_post_implicit(col, m_opts, dt);
        if (uses_state_update()) {
            BL_PROFILE("SHOC::advance::finalize");
            ShocImplicit::finalize_from_pdf(col, m_opts, dt);
        }
        {
            BL_PROFILE("SHOC::advance::store_carried_buoyancy_flux");
            store_carried_buoyancy_flux(col, mfi);
        }
        {
            BL_PROFILE("SHOC::advance::store_carried_turbulence");
            store_carried_turbulence(col, mfi);
        }

        auto tk_arr = m_eddy_coeffs_cc.array(mfi);
        auto th_tend = m_theta_tend_cc.array(mfi);
        auto qv_tend = m_qv_tend_cc.array(mfi);
        auto qc_tend = m_qc_tend_cc.array(mfi);
        auto qi_tend = m_qi_tend_cc.array(mfi);
        auto tke_tend = m_tke_tend_cc.array(mfi);
        auto u_tend_cc = m_u_tend_cc.array(mfi);
        auto v_tend_cc = m_v_tend_cc.array(mfi);
        auto pblh_arr = m_pblh_cc.array(mfi);
        auto shoc_cldfrac_arr = m_shoc_cldfrac_cc.array(mfi);
        auto shoc_ql_arr = m_shoc_ql_cc.array(mfi);
        auto shoc_ql2_arr = m_shoc_ql2_cc.array(mfi);
        auto shoc_cond_arr = m_shoc_cond_cc.array(mfi);
        auto w_sec_arr = m_w_sec_cc.array(mfi);
        auto wqls_sec_arr = m_wqls_sec_cc.array(mfi);
        auto wthv_sec_arr = m_wthv_sec_cc.array(mfi);
        auto thl_sec_arr = m_thl_sec_cc.array(mfi);
        auto qw_sec_arr = m_qw_sec_cc.array(mfi);
        auto qwthl_sec_arr = m_qwthl_sec_cc.array(mfi);
        auto wthl_sec_arr = m_wthl_sec_cc.array(mfi);
        auto wqw_sec_arr = m_wqw_sec_cc.array(mfi);
        auto w3_arr = m_w3_cc.array(mfi);
        auto brunt_arr = m_brunt_cc.array(mfi);
        auto isotropy_arr = m_isotropy_cc.array(mfi);
        auto shear_prod_arr = m_shear_prod_cc.array(mfi);
        auto buoy_prod_arr = m_buoy_prod_cc.array(mfi);
        auto diss_tke_arr = m_diss_tke_cc.array(mfi);

        const auto shoc_mix = col.shoc_mix.const_array();
        const auto pblh = col.pblh.const_array();
        const auto shoc_cldfrac = col.shoc_cldfrac.const_array();
        const auto shoc_ql = col.shoc_ql.const_array();
        const auto shoc_ql2 = col.shoc_ql2.const_array();
        const auto shoc_cond = col.shoc_cond.const_array();
        const auto w_sec = col.w_sec.const_array();
        const auto wqls_sec = col.wqls_sec.const_array();
        const auto wthv_sec = col.wthv_sec.const_array();
        const auto thl_sec = col.thl_sec.const_array();
        const auto qw_sec = col.qw_sec.const_array();
        const auto qwthl_sec = col.qwthl_sec.const_array();
        const auto wthl_sec = col.wthl_sec.const_array();
        const auto wqw_sec = col.wqw_sec.const_array();
        const auto w3 = col.w3.const_array();
        const auto brunt = col.brunt.const_array();
        const auto isotropy = col.isotropy.const_array();
        const auto shear_prod = col.shear_prod.const_array();
        const auto buoy_prod = col.buoy_prod.const_array();
        const auto diss_tke = col.diss_tke.const_array();
        const auto tk = col.tk.const_array();
        const auto tkh = col.tkh.const_array();
        const auto rho = col.rho.const_array();
        const auto zt = col.zt.const_array();
        const auto zi = col.zi.const_array();
        const auto col_theta_tend = col.theta_tend.const_array();
        const auto col_qv_tend = col.qv_tend.const_array();
        const auto col_qc_tend = col.qc_tend.const_array();
        const auto col_qi_tend = col.qi_tend.const_array();
        const auto col_tke_tend = col.tke_tend.const_array();
        const auto col_u_tend = col.u_tend.const_array();
        const auto col_v_tend = col.v_tend.const_array();
        const auto layout = col.layout;
        const Box xy_box = amrex::makeSlab(vbx, 2, layout.kmin);
        {
            BL_PROFILE("SHOC::advance::writeback");
            ParallelFor(xy_box, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                const int ic = shoc_column_index(layout, i, j);
                for (int kk = 0; kk < layout.nlev; ++kk) {
                    const int k = layout.kmin + kk;
                    const Real l = shoc_mix(ic,kk,0);
                    const Real km = tk(ic,kk,0);
                    const Real kh = tkh(ic,kk,0);

                    tk_arr(i,j,k,EddyDiff::Mom_v) = rho(ic,kk,0) * km;
                    tk_arr(i,j,k,EddyDiff::Theta_v) = rho(ic,kk,0) * kh;
                    tk_arr(i,j,k,EddyDiff::KE_v) = rho(ic,kk,0) * kh;
                    tk_arr(i,j,k,EddyDiff::Scalar_v) = rho(ic,kk,0) * kh;
                    tk_arr(i,j,k,EddyDiff::Q_v) = rho(ic,kk,0) * kh;
                    tk_arr(i,j,k,EddyDiff::Turb_lengthscale) = l;

                    th_tend(i,j,k) = col_theta_tend(ic,kk,0);
                    qv_tend(i,j,k) = col_qv_tend(ic,kk,0);
                    qc_tend(i,j,k) = col_qc_tend(ic,kk,0);
                    qi_tend(i,j,k) = col_qi_tend(ic,kk,0);
                    tke_tend(i,j,k) = col_tke_tend(ic,kk,0);
                    u_tend_cc(i,j,k) = col_u_tend(ic,kk,0);
                    v_tend_cc(i,j,k) = col_v_tend(ic,kk,0);
                    pblh_arr(i,j,k) = pblh(ic,0,0);
                    shoc_cldfrac_arr(i,j,k) = shoc_cldfrac(ic,kk,0);
                    shoc_ql_arr(i,j,k) = shoc_ql(ic,kk,0);
                    shoc_ql2_arr(i,j,k) = shoc_ql2(ic,kk,0);
                    shoc_cond_arr(i,j,k) = shoc_cond(ic,kk,0);
                    w_sec_arr(i,j,k) = w_sec(ic,kk,0);
                    wqls_sec_arr(i,j,k) = wqls_sec(ic,kk,0);
                    wthv_sec_arr(i,j,k) = wthv_sec(ic,kk,0);
                    brunt_arr(i,j,k) = brunt(ic,kk,0);
                    isotropy_arr(i,j,k) = isotropy(ic,kk,0);
                    thl_sec_arr(i,j,k) = weighted_linear_interp(zi(ic,kk,0), zi(ic,kk+1,0),
                                                                thl_sec(ic,kk,0), thl_sec(ic,kk+1,0),
                                                                zt(ic,kk,0));
                    qw_sec_arr(i,j,k) = weighted_linear_interp(zi(ic,kk,0), zi(ic,kk+1,0),
                                                               qw_sec(ic,kk,0), qw_sec(ic,kk+1,0),
                                                               zt(ic,kk,0));
                    qwthl_sec_arr(i,j,k) = weighted_linear_interp(zi(ic,kk,0), zi(ic,kk+1,0),
                                                                  qwthl_sec(ic,kk,0), qwthl_sec(ic,kk+1,0),
                                                                  zt(ic,kk,0));
                    wthl_sec_arr(i,j,k) = weighted_linear_interp(zi(ic,kk,0), zi(ic,kk+1,0),
                                                                 wthl_sec(ic,kk,0), wthl_sec(ic,kk+1,0),
                                                                 zt(ic,kk,0));
                    wqw_sec_arr(i,j,k) = weighted_linear_interp(zi(ic,kk,0), zi(ic,kk+1,0),
                                                                wqw_sec(ic,kk,0), wqw_sec(ic,kk+1,0),
                                                                zt(ic,kk,0));
                    w3_arr(i,j,k) = weighted_linear_interp(zi(ic,kk,0), zi(ic,kk+1,0),
                                                           w3(ic,kk,0), w3(ic,kk+1,0),
                                                           zt(ic,kk,0));
                    shear_prod_arr(i,j,k) = shear_prod(ic,kk,0);
                    buoy_prod_arr(i,j,k) = buoy_prod(ic,kk,0);
                    diss_tke_arr(i,j,k) = diss_tke(ic,kk,0);
                }
                });
        }

    }

    {
        BL_PROFILE("SHOC::advance::tendency_interpolation");
        m_u_tend_cc.FillBoundary(geom.periodicity());
        m_v_tend_cc.FillBoundary(geom.periodicity());

        const auto dom = geom.Domain();
        const int ilo = dom.smallEnd(0);
        const int ihi = dom.bigEnd(0);
        const int jlo = dom.smallEnd(1);
        const int jhi = dom.bigEnd(1);
        const bool xper = geom.isPeriodic(0);
        const bool yper = geom.isPeriodic(1);

        for (MFIter mfi(m_u_tend_cc, false); mfi.isValid(); ++mfi) {
            const Box& cc_bx = mfi.validbox();
            const Box xface_bx = amrex::surroundingNodes(cc_bx, 0);
            const Box yface_bx = amrex::surroundingNodes(cc_bx, 1);
            const auto u_cc = m_u_tend_cc.const_array(mfi);
            const auto v_cc = m_v_tend_cc.const_array(mfi);
            auto u_fc = m_u_tend_fc.array(mfi);
            auto v_fc = m_v_tend_fc.array(mfi);

            ParallelFor(xface_bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const int il = face_neighbor_cell(i - 1, ilo, ihi, xper);
                const int ir = face_neighbor_cell(i,     ilo, ihi, xper);
                u_fc(i,j,k) = 0.5_rt * (u_cc(il,j,k) + u_cc(ir,j,k));
            });

            ParallelFor(yface_bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const int jb = face_neighbor_cell(j - 1, jlo, jhi, yper);
                const int jt = face_neighbor_cell(j,     jlo, jhi, yper);
                v_fc(i,j,k) = 0.5_rt * (v_cc(i,jb,k) + v_cc(i,jt,k));
            });
        }
    }

    if (uses_state_update()) {
        sync_face_multifab_impl(m_u_tend_fc, geom);
        sync_face_multifab_impl(m_v_tend_fc, geom);
    }

    if (uses_state_update()) {
        BL_PROFILE("SHOC::advance::state_update");
        apply_state_update(cons, xvel, yvel, dt);
    }

    if (uses_momentum_state_update()) {
        sync_face_multifab_impl(xvel, geom);
        sync_face_multifab_impl(yvel, geom);
    }

    m_prev_turb_valid = true;
    ++m_advance_calls;
    if (m_opts.debug_summary) {
        BL_PROFILE("SHOC::advance::debug_summary");
        print_debug_summary(dt);
    }
}

void
ShocDriver::set_eddy_diffs () const
{
    BL_PROFILE("SHOC::set_eddy_diffs");

    AMREX_ALWAYS_ASSERT(m_eddy_diffs_ptr != nullptr);
    m_eddy_diffs_ptr->setVal(0.0, k_shoc_vertical_diff_comp, k_shoc_vertical_diff_count,
                             m_eddy_diffs_ptr->nGrow());
    MultiFab::Copy(*m_eddy_diffs_ptr, m_eddy_coeffs_cc,
                   EddyDiff::Turb_lengthscale, EddyDiff::Turb_lengthscale, 1, 0);

    if (uses_host_diffusion()) {
        MultiFab::Copy(*m_eddy_diffs_ptr, m_eddy_coeffs_cc,
                       k_shoc_vertical_diff_comp, k_shoc_vertical_diff_comp,
                       k_shoc_vertical_diff_count, 0);
    } else if (uses_momentum_host_diffusion()) {
        MultiFab::Copy(*m_eddy_diffs_ptr, m_eddy_coeffs_cc,
                       EddyDiff::Mom_v, EddyDiff::Mom_v, 1, 0);
    }
}

void
ShocDriver::set_diff_stresses () const
{
    BL_PROFILE("SHOC::set_diff_stresses");

    if (uses_host_diffusion()) {
        // Full host-diffusion mode preserves the existing host-owned lower
        // boundary fluxes and stresses.
        return;
    }

    if (!m_hfx3_ptr || !m_qfx3_ptr) {
        return;
    }

    for (MFIter mfi(*m_hfx3_ptr, false); mfi.isValid(); ++mfi) {
        const Box& vbx_cc = mfi.validbox();
        const Box& vbx_xz = convert(vbx_cc, IntVect(1,0,1));
        const Box& vbx_yz = convert(vbx_cc, IntVect(0,1,1));
        auto hfx = m_hfx3_ptr->array(mfi);
        auto qfx = m_qfx3_ptr->array(mfi);

        ParallelFor(vbx_cc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            hfx(i,j,k) = 0.0;
            qfx(i,j,k) = 0.0;
        });

        if ((owns_momentum_surface_stresses() || disables_momentum_transport()) &&
            m_tau13_ptr && m_tau23_ptr) {
            auto tau13 = m_tau13_ptr->array(mfi);
            auto tau23 = m_tau23_ptr->array(mfi);
            ParallelFor(vbx_xz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                tau13(i,j,k) = 0.0;
            });
            ParallelFor(vbx_yz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                tau23(i,j,k) = 0.0;
            });
        }
    }
}

void
ShocDriver::add_fast_tend (Vector<MultiFab>& S_rhs) const
{
    AMREX_ALWAYS_ASSERT(m_cons_ptr != nullptr);

    amrex::ignore_unused(S_rhs);
}

void
ShocDriver::add_slow_tend (const MFIter& mfi,
                           const Box& tbx,
                           const Array4<Real>& cell_rhs) const
{
    AMREX_ALWAYS_ASSERT(m_cons_ptr != nullptr);
    amrex::ignore_unused(mfi, tbx, cell_rhs);
}

bool
ShocDriver::uses_host_diffusion () const
{
    return shoc_uses_host_diffusion(m_opts.transport_mode);
}

bool
ShocDriver::uses_state_update () const
{
    return shoc_uses_state_update(m_opts.transport_mode);
}

bool
ShocDriver::uses_momentum_state_update () const
{
    return shoc_uses_momentum_state_update(m_opts.momentum_transport);
}

bool
ShocDriver::uses_momentum_host_diffusion () const
{
    return shoc_uses_momentum_host_diffusion(m_opts.momentum_transport);
}

bool
ShocDriver::disables_momentum_transport () const
{
    return shoc_disables_momentum_transport(m_opts.momentum_transport);
}

bool
ShocDriver::owns_scalar_surface_fluxes () const
{
    return uses_state_update();
}

bool
ShocDriver::owns_momentum_surface_stresses () const
{
    return uses_momentum_state_update();
}

bool
ShocDriver::needs_host_surface_momentum_stresses () const
{
    return uses_momentum_host_diffusion();
}

bool
ShocDriver::owns_surface_fluxes () const
{
    return owns_scalar_surface_fluxes();
}

void
ShocDriver::apply_state_update (MultiFab& cons,
                                MultiFab& xvel,
                                MultiFab& yvel,
                                Real dt) const
{
    AMREX_ALWAYS_ASSERT(dt > 0.0);

    apply_state_update_cons_impl(cons,
                                 m_theta_tend_cc,
                                 m_qv_tend_cc,
                                 m_qc_tend_cc,
                                 m_qi_tend_cc,
                                 m_tke_tend_cc,
                                 m_moisture_indices.qv,
                                 m_moisture_indices.qc,
                                 m_moisture_indices.qi,
                                 dt);

    if (uses_momentum_state_update()) {
        apply_state_update_face_velocity_impl(xvel, m_u_tend_fc, dt);
        apply_state_update_face_velocity_impl(yvel, m_v_tend_fc, dt);
    }
}

void
ShocDriver::print_debug_summary (Real dt) const
{
    BL_PROFILE("SHOC::print_debug_summary");

    auto print_minmax = [] (const char* name, const MultiFab& mf, int comp) {
        amrex::Print() << "    " << std::setw(12) << std::left << name
                       << " min=" << mf.min(comp)
                       << " max=" << mf.max(comp) << "\n";
    };

    amrex::Print() << "SHOC debug summary:"
                   << " level=" << m_lev
                   << " call=" << m_advance_calls
                   << " dt=" << dt
                   << " transport_mode=" << shoc_transport_mode_name(m_opts.transport_mode)
                   << " momentum_transport=" << shoc_momentum_transport_name(m_opts.momentum_transport)
                   << " state_update=" << (uses_state_update() ? "on" : "off")
                   << "\n";

    print_minmax("Kmv", m_eddy_coeffs_cc, EddyDiff::Mom_v);
    print_minmax("Khv", m_eddy_coeffs_cc, EddyDiff::Theta_v);
    print_minmax("KE_v", m_eddy_coeffs_cc, EddyDiff::KE_v);
    print_minmax("Q_v", m_eddy_coeffs_cc, EddyDiff::Q_v);
    print_minmax("l_turb", m_eddy_coeffs_cc, EddyDiff::Turb_lengthscale);
    print_minmax("pblh", m_pblh_cc, 0);

    print_minmax("th_tend", m_theta_tend_cc, 0);
    print_minmax("qv_tend", m_qv_tend_cc, 0);
    print_minmax("qc_tend", m_qc_tend_cc, 0);
    print_minmax("qi_tend", m_qi_tend_cc, 0);
    print_minmax("tke_tend", m_tke_tend_cc, 0);
    print_minmax("u_tend", m_u_tend_fc, 0);
    print_minmax("v_tend", m_v_tend_fc, 0);
    print_minmax("w_sec", m_w_sec_cc, 0);
    print_minmax("thl_sec", m_thl_sec_cc, 0);
    print_minmax("qw_sec", m_qw_sec_cc, 0);
    print_minmax("qwthl_sec", m_qwthl_sec_cc, 0);
    print_minmax("wthl_sec", m_wthl_sec_cc, 0);
    print_minmax("wqw_sec", m_wqw_sec_cc, 0);
    print_minmax("w3", m_w3_cc, 0);
    print_minmax("brunt", m_brunt_cc, 0);
    print_minmax("isotropy", m_isotropy_cc, 0);
    print_minmax("shear_prod", m_shear_prod_cc, 0);
    print_minmax("buoy_prod", m_buoy_prod_cc, 0);
    print_minmax("diss_tke", m_diss_tke_cc, 0);
    print_minmax("cldfrac", m_shoc_cldfrac_cc, 0);
    print_minmax("shoc_ql", m_shoc_ql_cc, 0);
    print_minmax("shoc_ql2", m_shoc_ql2_cc, 0);
    print_minmax("shoc_cond", m_shoc_cond_cc, 0);
    print_minmax("wqls_sec", m_wqls_sec_cc, 0);
    print_minmax("wthv_sec", m_wthv_sec_cc, 0);

    if (m_hfx3_ptr)  print_minmax("hfx3_in", *m_hfx3_ptr, 0);
    if (m_qfx3_ptr)  print_minmax("qfx3_in", *m_qfx3_ptr, 0);
    if (m_tau13_ptr) print_minmax("tau13_in", *m_tau13_ptr, 0);
    if (m_tau23_ptr) print_minmax("tau23_in", *m_tau23_ptr, 0);
}
