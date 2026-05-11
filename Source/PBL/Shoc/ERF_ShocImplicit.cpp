#include "ERF_ShocImplicit.H"

#include "ERF_Constants.H"
#include "ERF_ShocEnergyFixer.H"

#include <algorithm>
#include <cmath>

using namespace amrex;

namespace
{
    constexpr Real k_shoc_zvir = 0.61;
    constexpr Real k_shoc_u_ws_min = 1.0;
    constexpr Real k_shoc_ksrf_min = 1.0e-4;
    constexpr Real k_shoc_ustar_min = 0.01;
    constexpr Real k_shoc_min_tke = 4.0e-4;
    constexpr Real k_shoc_max_tke = 50.0;
    constexpr Real k_shoc_min_qw = 0.0;
    constexpr Real k_shoc_min_temp = 180.0;
    constexpr Real k_shoc_lat_ice = 3.34e5;
    constexpr Real k_shoc_freezing_temp = 273.15;

    AMREX_FORCE_INLINE
    Real weighted_linear_interp (Real x0, Real x1, Real y0, Real y1, Real x)
    {
        const Real denom = x1 - x0;
        if (std::abs(denom) <= 1.0e-12) {
            return 0.5 * (y0 + y1);
        }
        return y0 + (y1 - y0) * (x - x0) / denom;
    }

    void reconstruct_moisture_from_pdf_and_mean_state (Real thetal,
                                                       Real qw,
                                                       Real exner,
                                                       Real qi_seed,
                                                       Real pdf_ql,
                                                       Real& tabs,
                                                       Real& qv,
                                                       Real& qc,
                                                       Real& qi)
    {
        const Real ql_total = std::clamp(pdf_ql, 0.0_rt, std::max(0.0_rt, qw));
        tabs = ShocImplicit::compute_temperature(thetal, ql_total, exner);

        const bool use_ice = (tabs < k_shoc_freezing_temp && qi_seed > 0.0);
        qc = use_ice ? 0.0_rt : ql_total;
        qi = use_ice ? ql_total : 0.0_rt;
        qv = ShocImplicit::compute_vapor(qw, ql_total);
    }

    void interpolate_cc_to_iface (const ShocColumnData& col,
                                  const FArrayBox& cc_in,
                                  FArrayBox& iface_out)
    {
        const auto zt = col.zt.const_array();
        const auto zi = col.zi.const_array();
        const auto in = cc_in.const_array();
        auto out = iface_out.array();

        for (int ic = 0; ic < col.layout.ncell; ++ic) {
            if (col.layout.nlev == 1) {
                out(ic,0,0) = in(ic,0,0);
                out(ic,1,0) = in(ic,0,0);
                continue;
            }

            out(ic,0,0) = weighted_linear_interp(zt(ic,0,0), zt(ic,1,0),
                                                 in(ic,0,0), in(ic,1,0),
                                                 zi(ic,0,0));
            for (int k = 1; k < col.layout.nlev; ++k) {
                out(ic,k,0) = weighted_linear_interp(zt(ic,k-1,0), zt(ic,k,0),
                                                     in(ic,k-1,0), in(ic,k,0),
                                                     zi(ic,k,0));
            }
            out(ic,col.layout.nlev,0) =
                weighted_linear_interp(zt(ic,col.layout.nlev-2,0), zt(ic,col.layout.nlev-1,0),
                                       in(ic,col.layout.nlev-2,0), in(ic,col.layout.nlev-1,0),
                                       zi(ic,col.layout.nlev,0));
        }
    }

    AMREX_FORCE_INLINE
    Real interface_spacing (const ShocColumnData& col, int ic, int k)
    {
        const auto zt = col.zt.const_array();
        const auto zi = col.zi.const_array();
        if (k <= 0) {
            return std::max(zt(ic,0,0) - zi(ic,0,0), 1.0e-12);
        }
        if (k >= col.layout.nlev) {
            return std::max(zi(ic,col.layout.nlev,0) - zt(ic,col.layout.nlev-1,0), 1.0e-12);
        }
        return std::max(zt(ic,k,0) - zt(ic,k-1,0), 1.0e-12);
    }

    void setup_tridiagonal (int nlev,
                            const Vector<Real>& kv_zi,
                            const Vector<Real>& tmpi,
                            const Vector<Real>& rdp_zt,
                            Real dt,
                            Real flux_diag,
                            Vector<Real>& dl,
                            Vector<Real>& d,
                            Vector<Real>& du)
    {
        dl.assign(nlev, 0.0);
        d.assign(nlev, 1.0);
        du.assign(nlev, 0.0);

        for (int k = 0; k < nlev; ++k) {
            if (k > 0) {
                dl[k] = -kv_zi[k] * tmpi[k] * rdp_zt[k];
            }
            if (k < nlev - 1) {
                du[k] = -kv_zi[k+1] * tmpi[k+1] * rdp_zt[k];
            }
            d[k] = 1.0 - dl[k] - du[k];
        }

        if (nlev > 0) {
            d[0] += flux_diag * dt * CONST_GRAV * rdp_zt[0];
        }
    }

    void solve_tridiagonal (const Vector<Real>& dl_in,
                            const Vector<Real>& d_in,
                            const Vector<Real>& du_in,
                            Vector<Real>& rhs)
    {
        const int n = static_cast<int>(rhs.size());
        if (n == 0) {
            return;
        }

        Vector<Real> cprime(n, 0.0);
        Vector<Real> dprime(n, 0.0);

        Real denom = d_in[0];
        AMREX_ALWAYS_ASSERT(std::abs(denom) > 1.0e-14);
        cprime[0] = (n > 1) ? du_in[0] / denom : 0.0;
        dprime[0] = rhs[0] / denom;

        for (int k = 1; k < n; ++k) {
            denom = d_in[k] - dl_in[k] * cprime[k-1];
            AMREX_ALWAYS_ASSERT(std::abs(denom) > 1.0e-14);
            cprime[k] = (k < n - 1) ? du_in[k] / denom : 0.0;
            dprime[k] = (rhs[k] - dl_in[k] * dprime[k-1]) / denom;
        }

        rhs[n-1] = dprime[n-1];
        for (int k = n - 2; k >= 0; --k) {
            rhs[k] = dprime[k] - cprime[k] * rhs[k+1];
        }
    }
}

void
ShocImplicit::cache_baseline_state (ShocColumnData& col)
{
    auto thetal = col.thetal.const_array();
    auto theta = col.theta.const_array();
    auto qv = col.qv.const_array();
    auto qc = col.qc.const_array();
    auto qi = col.qi.const_array();
    auto u = col.u.const_array();
    auto v = col.v.const_array();
    auto tke = col.tke.const_array();

    auto thetal_base = col.thetal_base.array();
    auto theta_base = col.theta_base.array();
    auto qv_base = col.qv_base.array();
    auto qc_base = col.qc_base.array();
    auto qi_base = col.qi_base.array();
    auto u_base = col.u_base.array();
    auto v_base = col.v_base.array();
    auto tke_base = col.tke_base_state.array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 0; k < col.layout.nlev; ++k) {
            thetal_base(ic,k,0) = thetal(ic,k,0);
            theta_base(ic,k,0) = theta(ic,k,0);
            qv_base(ic,k,0) = qv(ic,k,0);
            qc_base(ic,k,0) = qc(ic,k,0);
            qi_base(ic,k,0) = qi(ic,k,0);
            u_base(ic,k,0) = u(ic,k,0);
            v_base(ic,k,0) = v(ic,k,0);
            tke_base(ic,k,0) = tke(ic,k,0);
        }
    }
}

void
ShocImplicit::compute_tmpi (const ShocColumnData& col,
                            int ic,
                            Real dt,
                            const Vector<Real>& rho_zi,
    Vector<Real>& tmpi)
{
    tmpi.assign(col.layout.nlev + 1, 0.0);
    for (int k = 0; k <= col.layout.nlev; ++k) {
        tmpi[k] = dt * CONST_GRAV * std::max(rho_zi[k], 1.0e-12) /
                  interface_spacing(col, ic, k);
    }
}

void
ShocImplicit::compute_dp_inverse (const ShocColumnData& col,
                                  int ic,
                                  Vector<Real>& rdp_zt)
{
    rdp_zt.assign(col.layout.nlev, 0.0);
    const auto rho = col.rho.const_array();
    const auto dz = col.dz.const_array();
    for (int k = 0; k < col.layout.nlev; ++k) {
        rdp_zt[k] = 1.0 / std::max(CONST_GRAV * rho(ic,k,0) * dz(ic,k,0), 1.0e-12);
    }
}

Real
ShocImplicit::compute_temperature (Real thetal,
                                   Real ql,
                                   Real exner)
{
    return std::max(k_shoc_min_temp,
                    thetal * std::max(exner, 1.0e-12) + (L_v / Cp_d) * ql);
}

Real
ShocImplicit::compute_vapor (Real qw,
                             Real ql)
{
    return std::max(0.0_rt, qw - std::max(0.0_rt, ql));
}

void
ShocImplicit::advance_implicit_state (ShocColumnData& col,
                                      const ShocRuntimeOptions& opts,
                                      Real dt)
{
    AMREX_ALWAYS_ASSERT(dt > 0.0);
    static_cast<void>(opts);

    const int nlev = col.layout.nlev;
    if (nlev == 0) {
        return;
    }

    const Box iface_box(IntVect(0,0,0), IntVect(col.layout.ncell - 1, nlev, 0));
    FArrayBox rho_zi_fab(iface_box, 1);
    FArrayBox tk_zi_fab(iface_box, 1);
    FArrayBox tkh_zi_fab(iface_box, 1);

    interpolate_cc_to_iface(col, col.rho, rho_zi_fab);
    interpolate_cc_to_iface(col, col.tk, tk_zi_fab);
    interpolate_cc_to_iface(col, col.tkh, tkh_zi_fab);

    auto thetal = col.thetal.array();
    auto qw = col.qw.array();
    auto tke = col.tke.array();
    auto u = col.u.array();
    auto v = col.v.array();

    const auto surf_tau_u = col.surf_tau_u.const_array();
    const auto surf_tau_v = col.surf_tau_v.const_array();
    const auto surf_sens_flux = col.surf_sens_flux.const_array();
    const auto surf_lat_flux = col.surf_lat_flux.const_array();
    const auto rho_zi = rho_zi_fab.const_array();
    const auto tk_zi = tk_zi_fab.const_array();
    const auto tkh_zi = tkh_zi_fab.const_array();
    const auto explicit_tke_delta = col.tke_tend.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        Vector<Real> rho_zi_col(nlev + 1, 0.0);
        for (int k = 0; k <= nlev; ++k) {
            rho_zi_col[k] = std::max(rho_zi(ic,k,0), 1.0e-12);
        }
        Vector<Real> rdp_zt;
        Vector<Real> tmpi;
        compute_dp_inverse(col, ic, rdp_zt);
        compute_tmpi(col, ic, dt, rho_zi_col, tmpi);

        Vector<Real> thl_rhs(nlev, 0.0);
        Vector<Real> qw_rhs(nlev, 0.0);
        Vector<Real> tke_rhs(nlev, 0.0);
        Vector<Real> u_rhs(nlev, 0.0);
        Vector<Real> v_rhs(nlev, 0.0);
        Vector<Real> tke_new(nlev, 0.0);
        Vector<Real> tk_zi_col(nlev + 1, 0.0);
        Vector<Real> tkh_zi_col(nlev + 1, 0.0);

        for (int k = 0; k <= nlev; ++k) {
            tk_zi_col[k] = std::max(0.0, tk_zi(ic,k,0));
            tkh_zi_col[k] = std::max(0.0, tkh_zi(ic,k,0));
        }

        for (int k = 0; k < nlev; ++k) {
            thl_rhs[k] = thetal(ic,k,0);
            qw_rhs[k] = qw(ic,k,0);
            tke_rhs[k] = std::clamp(tke(ic,k,0), k_shoc_min_tke, k_shoc_max_tke);
            u_rhs[k] = u(ic,k,0);
            v_rhs[k] = v(ic,k,0);
        }

        // E3SM: cmnfac = dtime * g * rho_zi(nlevi-1) * rdp_zt(nlev-1)
        // The tmpi array includes a 1/dz_zi interface-spacing factor that
        // belongs only in the tridiagonal diffusion matrix, NOT in the
        // explicit surface-flux injection.  Using tmpi[0]*rdp_zt[0] here
        // previously introduced an extra 1/dz_half that weakened the
        // surface fluxes by ~dz/2.
        const Real cmnfac = dt * CONST_GRAV * rho_zi_col[0] * rdp_zt[0];
        thl_rhs[0] += cmnfac * surf_sens_flux(ic,0,0);
        qw_rhs[0] += cmnfac * surf_lat_flux(ic,0,0);

        const Real uw_sfc = surf_tau_u(ic,0,0);
        const Real vw_sfc = surf_tau_v(ic,0,0);
        const Real stress_mag = std::sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc);
        Real ksrf = 0.0;
        Real wtke_sfc = 0.0;
        if (stress_mag > 1.0e-12) {
            const Real ws = std::max(std::sqrt(u_rhs[0] * u_rhs[0] + v_rhs[0] * v_rhs[0]), k_shoc_u_ws_min);
            const Real tau = std::sqrt(std::pow(rho_zi_col[0] * uw_sfc, 2) +
                                       std::pow(rho_zi_col[0] * vw_sfc, 2));
            ksrf = std::max(tau / ws, k_shoc_ksrf_min);
            const Real ustar = std::max(std::sqrt(stress_mag), k_shoc_ustar_min);
            wtke_sfc = ustar * ustar * ustar;
        }
        tke_rhs[0] += cmnfac * wtke_sfc;

        Vector<Real> dl, d, du;
        setup_tridiagonal(nlev, tk_zi_col, tmpi, rdp_zt, dt, ksrf, dl, d, du);
        solve_tridiagonal(dl, d, du, u_rhs);
        solve_tridiagonal(dl, d, du, v_rhs);

        setup_tridiagonal(nlev, tkh_zi_col, tmpi, rdp_zt, dt, 0.0, dl, d, du);
        solve_tridiagonal(dl, d, du, thl_rhs);
        solve_tridiagonal(dl, d, du, qw_rhs);
        solve_tridiagonal(dl, d, du, tke_rhs);

        for (int k = 0; k < nlev; ++k) {
            const Real tke_new_loc = std::clamp(tke_rhs[k], k_shoc_min_tke, k_shoc_max_tke);
            tke_new[k] = tke_new_loc;
        }

        for (int k = 0; k < nlev; ++k) {
            thetal(ic,k,0) = thl_rhs[k];
            qw(ic,k,0) = std::max(qw_rhs[k], k_shoc_min_qw);
            tke(ic,k,0) = tke_new[k];
            u(ic,k,0) = u_rhs[k];
            v(ic,k,0) = v_rhs[k];
        }
    }
}

void
ShocImplicit::finalize_from_pdf (ShocColumnData& col,
                                 const ShocRuntimeOptions& opts,
                                 Real dt)
{
    AMREX_ALWAYS_ASSERT(dt > 0.0);
    static_cast<void>(opts);

    const int nlev = col.layout.nlev;
    if (nlev == 0) {
        return;
    }

    auto thetal = col.thetal.array();
    auto theta = col.theta.array();
    auto theta_v = col.theta_v.array();
    auto qv = col.qv.array();
    auto qc = col.qc.array();
    auto qi = col.qi.array();
    auto qw = col.qw.array();
    auto tabs = col.tabs.array();
    auto host_dse = col.host_dse.array();
    auto tke = col.tke.array();
    auto u = col.u.array();
    auto v = col.v.array();
    auto shoc_ql = col.shoc_ql.array();
    auto theta_tend = col.theta_tend.array();
    auto qv_tend = col.qv_tend.array();
    auto qc_tend = col.qc_tend.array();
    auto qi_tend = col.qi_tend.array();
    auto u_tend = col.u_tend.array();
    auto v_tend = col.v_tend.array();
    auto tke_tend = col.tke_tend.array();

    const auto zt = col.zt.const_array();
    const auto exner = col.exner.const_array();
    const auto shoc_ql_pdf = col.shoc_ql.const_array();
    const auto thetal_base = col.thetal_base.const_array();
    const auto theta_base = col.theta_base.const_array();
    const auto qv_base = col.qv_base.const_array();
    const auto qc_base = col.qc_base.const_array();
    const auto qi_base = col.qi_base.const_array();
    const auto u_base = col.u_base.const_array();
    const auto v_base = col.v_base.const_array();
    const auto tke_base = col.tke_base_state.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        Vector<Real> thl_old(nlev, 0.0);
        Vector<Real> qv_old(nlev, 0.0);
        Vector<Real> qc_old(nlev, 0.0);
        Vector<Real> qi_old(nlev, 0.0);
        Vector<Real> u_old(nlev, 0.0);
        Vector<Real> v_old(nlev, 0.0);
        Vector<Real> tke_old(nlev, 0.0);
        Vector<Real> thl_new(nlev, 0.0);
        Vector<Real> qv_new(nlev, 0.0);
        Vector<Real> qc_new(nlev, 0.0);
        Vector<Real> qi_new(nlev, 0.0);
        Vector<Real> u_new(nlev, 0.0);
        Vector<Real> v_new(nlev, 0.0);
        Vector<Real> tke_new(nlev, 0.0);

        for (int k = 0; k < nlev; ++k) {
            thl_old[k] = thetal_base(ic,k,0);
            qv_old[k] = qv_base(ic,k,0);
            qc_old[k] = qc_base(ic,k,0);
            qi_old[k] = qi_base(ic,k,0);
            u_old[k] = u_base(ic,k,0);
            v_old[k] = v_base(ic,k,0);
            tke_old[k] = tke_base(ic,k,0);

            thl_new[k] = thetal(ic,k,0);
            u_new[k] = u(ic,k,0);
            v_new[k] = v(ic,k,0);
            tke_new[k] = tke(ic,k,0);

            const Real qw_new_loc = std::max(qw(ic,k,0), k_shoc_min_qw);
            const Real pdf_ql = std::clamp(shoc_ql_pdf(ic,k,0), 0.0_rt, qw_new_loc);
            Real tabs_new = 0.0;
            reconstruct_moisture_from_pdf_and_mean_state(thl_new[k], qw_new_loc,
                                                         exner(ic,k,0), qi_old[k], pdf_ql,
                                                         tabs_new, qv_new[k], qc_new[k], qi_new[k]);
        }

        ShocEnergyFixer::apply_column(col, ic, dt,
                                      thl_old, qv_old, qc_old, qi_old,
                                      u_old, v_old, tke_old,
                                      thl_new, qv_new, qc_new, qi_new,
                                      u_new, v_new, tke_new);

        for (int k = 0; k < nlev; ++k) {
            const Real qw_new_loc = std::max(qw(ic,k,0), k_shoc_min_qw);
            const Real pdf_ql = std::clamp(shoc_ql_pdf(ic,k,0), 0.0_rt, qw_new_loc);
            Real tabs_new = 0.0;
            Real qv_new_loc = 0.0;
            Real qc_new_loc = 0.0;
            Real qi_new_loc = 0.0;
            reconstruct_moisture_from_pdf_and_mean_state(thl_new[k], qw_new_loc,
                                                         exner(ic,k,0), qi_old[k], pdf_ql,
                                                         tabs_new, qv_new_loc, qc_new_loc, qi_new_loc);

            const Real ql_total = qc_new_loc + qi_new_loc;
            thetal(ic,k,0) = thl_new[k];
            tabs(ic,k,0) = tabs_new;
            theta(ic,k,0) = tabs_new / std::max(exner(ic,k,0), 1.0e-12);
            qw(ic,k,0) = qw_new_loc;
            qv(ic,k,0) = qv_new_loc;
            qc(ic,k,0) = qc_new_loc;
            qi(ic,k,0) = qi_new_loc;
            shoc_ql(ic,k,0) = ql_total;
            theta_v(ic,k,0) = theta(ic,k,0) * (1.0 + k_shoc_zvir * qv_new_loc - ql_total);
            host_dse(ic,k,0) = Cp_d * tabs_new + CONST_GRAV * zt(ic,k,0);
            tke(ic,k,0) = tke_new[k];
            u(ic,k,0) = u_new[k];
            v(ic,k,0) = v_new[k];

            theta_tend(ic,k,0) = (theta(ic,k,0) - theta_base(ic,k,0)) / dt;
            qv_tend(ic,k,0) = (qv_new_loc - qv_base(ic,k,0)) / dt;
            qc_tend(ic,k,0) = (qc_new_loc - qc_base(ic,k,0)) / dt;
            qi_tend(ic,k,0) = (qi_new_loc - qi_base(ic,k,0)) / dt;
            u_tend(ic,k,0) = (u_new[k] - u_base(ic,k,0)) / dt;
            v_tend(ic,k,0) = (v_new[k] - v_base(ic,k,0)) / dt;
            tke_tend(ic,k,0) = (tke_new[k] - tke_base(ic,k,0)) / dt;
        }
    }
}

void
ShocImplicit::update_prognostics (ShocColumnData& col,
                                  const ShocRuntimeOptions& opts,
                                  Real dt)
{
    cache_baseline_state(col);
    advance_implicit_state(col, opts, dt);
    finalize_from_pdf(col, opts, dt);
}
