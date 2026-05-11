#include "ERF_ShocMoments.H"

#include "ERF_Constants.H"

#include <algorithm>
#include <cmath>

using namespace amrex;

namespace
{
    constexpr Real k_shoc_min_tke = 4.0e-4;
    constexpr Real k_shoc_large_neg = -99999999.99;
    constexpr Real k_shoc_w3clip = 1.2;
    constexpr Real k_shoc_w3clipdef = 0.02;
    constexpr Real k_shoc_base_temp = 300.0;
    constexpr Real k_shoc_ufmin = 0.01;

    AMREX_FORCE_INLINE
    Real weighted_linear_interp (Real x0, Real x1, Real y0, Real y1, Real x)
    {
        const Real denom = x1 - x0;
        if (std::abs(denom) <= 1.0e-12) {
            return 0.5 * (y0 + y1);
        }
        return y0 + (y1 - y0) * (x - x0) / denom;
    }

    AMREX_FORCE_INLINE
    Real signed_denominator (Real value, Real eps = 1.0e-12)
    {
        if (std::abs(value) >= eps) {
            return value;
        }
        return std::copysign(eps, value == 0.0 ? 1.0 : value);
    }

    void interpolate_cc_to_iface (const ShocColumnData& col,
                                  const FArrayBox& cc_in,
                                  FArrayBox& iface_out,
                                  Real min_thresh)
    {
        const auto zt = col.zt.const_array();
        const auto zi = col.zi.const_array();
        const auto in = cc_in.const_array();
        auto out = iface_out.array();

        for (int ic = 0; ic < col.layout.ncell; ++ic) {
            if (col.layout.nlev == 1) {
                out(ic,0,0) = std::max(min_thresh, in(ic,0,0));
                out(ic,1,0) = std::max(min_thresh, in(ic,0,0));
                continue;
            }

            out(ic,0,0) = std::max(min_thresh,
                                   weighted_linear_interp(zt(ic,0,0), zt(ic,1,0),
                                                          in(ic,0,0), in(ic,1,0),
                                                          zi(ic,0,0)));
            for (int k = 1; k < col.layout.nlev; ++k) {
                const Real interp = weighted_linear_interp(zt(ic,k-1,0), zt(ic,k,0),
                                                           in(ic,k-1,0), in(ic,k,0),
                                                           zi(ic,k,0));
                out(ic,k,0) = std::max(min_thresh, interp);
            }
            out(ic,col.layout.nlev,0) = std::max(min_thresh,
                                                 weighted_linear_interp(zt(ic,col.layout.nlev-2,0),
                                                                        zt(ic,col.layout.nlev-1,0),
                                                                        in(ic,col.layout.nlev-2,0),
                                                                        in(ic,col.layout.nlev-1,0),
                                                                        zi(ic,col.layout.nlev,0)));
        }
    }

    AMREX_FORCE_INLINE
    Real cell_spacing_at_iface (const ShocColumnData& col, int ic, int k)
    {
        const auto zt = col.zt.const_array();
        if (k <= 0) {
            return std::max(zt(ic,0,0) - col.zi.const_array()(ic,0,0), 1.0e-12);
        }
        if (k >= col.layout.nlev) {
            return std::max(col.zi.const_array()(ic,col.layout.nlev,0) - zt(ic,col.layout.nlev-1,0), 1.0e-12);
        }
        return std::max(zt(ic,k,0) - zt(ic,k-1,0), 1.0e-12);
    }

    AMREX_FORCE_INLINE
    Real top_taper_from_distance (const ShocRuntimeOptions& opts,
                                  Real distance_from_top)
    {
        if (opts.top_taper_depth <= 0.0) {
            return 1.0;
        }

        const Real x = std::clamp(std::max(0.0, distance_from_top) / opts.top_taper_depth,
                                  0.0, 1.0);
        const Real smooth = x * x * (3.0 - 2.0 * x);
        return opts.top_taper_min_factor + (1.0 - opts.top_taper_min_factor) * smooth;
    }

    AMREX_FORCE_INLINE
    Real top_taper_cell (const ShocColumnData& col,
                         const ShocRuntimeOptions& opts,
                         int ic,
                         int k)
    {
        const auto zt = col.zt.const_array();
        const auto zi = col.zi.const_array();
        return top_taper_from_distance(opts, zi(ic,col.layout.nlev,0) - zt(ic,k,0));
    }

    AMREX_FORCE_INLINE
    Real top_taper_iface (const ShocColumnData& col,
                          const ShocRuntimeOptions& opts,
                          int ic,
                          int k)
    {
        const auto zi = col.zi.const_array();
        return top_taper_from_distance(opts, zi(ic,col.layout.nlev,0) - zi(ic,k,0));
    }

    void diagnose_surface_moment_scales (const ShocColumnData& col,
                                         Real& ustar2,
                                         Real& wstar)
    {
        const auto wthl_sfc = col.surf_sens_flux.const_array()(0,0,0);
        const auto uw_sfc = col.surf_tau_u.const_array()(0,0,0);
        const auto vw_sfc = col.surf_tau_v.const_array()(0,0,0);

        ustar2 = std::sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc);
        if (wthl_sfc >= 0.0) {
            wstar = std::cbrt((CONST_GRAV / k_shoc_base_temp) * wthl_sfc);
        } else {
            wstar = 0.0;
        }
    }

    void apply_second_moment_boundary_conditions (ShocColumnData& col)
    {
        auto thl_sec = col.thl_sec.array();
        auto qw_sec = col.qw_sec.array();
        auto qwthl_sec = col.qwthl_sec.array();
        auto wthl_sec = col.wthl_sec.array();
        auto wqw_sec = col.wqw_sec.array();
        auto uw_sec = col.uw_sec.array();
        auto vw_sec = col.vw_sec.array();
        auto wtke_sec = col.wtke_sec.array();

        const auto sflux = col.surf_sens_flux.const_array();
        const auto lflux = col.surf_lat_flux.const_array();
        const auto tauu = col.surf_tau_u.const_array();
        const auto tauv = col.surf_tau_v.const_array();

        for (int ic = 0; ic < col.layout.ncell; ++ic) {
            const Real wthl_sfc = sflux(ic,0,0);
            const Real wqw_sfc = lflux(ic,0,0);
            const Real uw_sfc = tauu(ic,0,0);
            const Real vw_sfc = tauv(ic,0,0);
            const Real ustar2 = std::sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc);
            const Real wstar = (wthl_sfc >= 0.0)
                ? std::cbrt((CONST_GRAV / k_shoc_base_temp) * wthl_sfc)
                : 0.0;
            const Real uf = std::max(k_shoc_ufmin,
                                     std::sqrt(ustar2 + 0.3 * wstar * wstar));

            thl_sec(ic,0,0) = 0.72 * std::pow(wthl_sfc / uf, 2);
            qw_sec(ic,0,0) = 0.72 * std::pow(wqw_sfc / uf, 2);
            qwthl_sec(ic,0,0) = 0.36 * (wthl_sfc / uf) * (wqw_sfc / uf);
            wthl_sec(ic,0,0) = wthl_sfc;
            wqw_sec(ic,0,0) = wqw_sfc;
            uw_sec(ic,0,0) = uw_sfc;
            vw_sec(ic,0,0) = vw_sfc;
            wtke_sec(ic,0,0) = std::pow(std::max(std::sqrt(ustar2), k_shoc_ufmin), 3);

            const int ktop = col.layout.nlev;
            thl_sec(ic,ktop,0) = 0.0;
            qw_sec(ic,ktop,0) = 0.0;
            qwthl_sec(ic,ktop,0) = 0.0;
            wthl_sec(ic,col.layout.nlev,0) = 0.0;
            wqw_sec(ic,col.layout.nlev,0) = 0.0;
            uw_sec(ic,col.layout.nlev,0) = 0.0;
            vw_sec(ic,col.layout.nlev,0) = 0.0;
            wtke_sec(ic,col.layout.nlev,0) = 0.0;
        }
    }

    void apply_top_taper_to_second_moments (ShocColumnData& col,
                                            const ShocRuntimeOptions& opts)
    {
        if (opts.top_taper_depth <= 0.0) {
            return;
        }

        auto thl_sec = col.thl_sec.array();
        auto qw_sec = col.qw_sec.array();
        auto qwthl_sec = col.qwthl_sec.array();
        auto wthl_sec = col.wthl_sec.array();
        auto wqw_sec = col.wqw_sec.array();
        auto uw_sec = col.uw_sec.array();
        auto vw_sec = col.vw_sec.array();
        auto wtke_sec = col.wtke_sec.array();
        auto w_sec = col.w_sec.array();

        for (int ic = 0; ic < col.layout.ncell; ++ic) {
            for (int k = 0; k < col.layout.nlev; ++k) {
                w_sec(ic,k,0) *= top_taper_cell(col, opts, ic, k);
            }
            for (int k = 0; k <= col.layout.nlev; ++k) {
                const Real factor = top_taper_iface(col, opts, ic, k);
                thl_sec(ic,k,0) *= factor;
                qw_sec(ic,k,0) *= factor;
                qwthl_sec(ic,k,0) *= factor;
                wthl_sec(ic,k,0) *= factor;
                wqw_sec(ic,k,0) *= factor;
                uw_sec(ic,k,0) *= factor;
                vw_sec(ic,k,0) *= factor;
                wtke_sec(ic,k,0) *= factor;
            }
        }
    }

    void apply_top_taper_to_third_moments (ShocColumnData& col,
                                           const ShocRuntimeOptions& opts)
    {
        if (opts.top_taper_depth <= 0.0) {
            return;
        }

        auto w3 = col.w3.array();
        for (int ic = 0; ic < col.layout.ncell; ++ic) {
            for (int k = 0; k <= col.layout.nlev; ++k) {
                w3(ic,k,0) *= top_taper_iface(col, opts, ic, k);
            }
        }
    }
}

void
ShocMoments::calc_var_or_covar (const ShocColumnData& col,
                                Real tunefac,
                                const FArrayBox& isotropy_zi,
                                const FArrayBox& tkh_zi,
                                const FArrayBox& invar1,
                                const FArrayBox& invar2,
                                FArrayBox& outvar)
{
    auto out = outvar.array();
    const auto iso = isotropy_zi.const_array();
    const auto tkh = tkh_zi.const_array();
    const auto v1 = invar1.const_array();
    const auto v2 = invar2.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 1; k < col.layout.nlev; ++k) {
            const Real dz_zi = cell_spacing_at_iface(col, ic, k);
            const Real grid_dz2 = 1.0 / (dz_zi * dz_zi);
            out(ic,k,0) = tunefac * (iso(ic,k,0) * tkh(ic,k,0)) * grid_dz2 *
                          (v1(ic,k-1,0) - v1(ic,k,0)) * (v2(ic,k-1,0) - v2(ic,k,0));
        }
    }
}

void
ShocMoments::calc_vertflux (const ShocColumnData& col,
                            const FArrayBox& tkh_zi,
                            const FArrayBox& invar,
                            FArrayBox& vertflux)
{
    auto out = vertflux.array();
    const auto tkh = tkh_zi.const_array();
    const auto v = invar.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 1; k < col.layout.nlev; ++k) {
            const Real dz_zi = cell_spacing_at_iface(col, ic, k);
            // E3SM uses top-down vertical indexing. ERF columns are bottom-up,
            // so the interface gradient must use upper-minus-lower to preserve
            // the same physical downgradient flux sign.
            out(ic,k,0) = -(tkh(ic,k,0) / dz_zi) * (v(ic,k,0) - v(ic,k-1,0));
        }
    }
}

void
ShocMoments::diagnose_second_moments (ShocColumnData& col,
                                      const ShocRuntimeOptions& opts)
{
    FArrayBox isotropy_zi, tkh_zi, tk_zi;
    const Box iface_box(IntVect(0,0,0), IntVect(col.layout.ncell - 1, col.layout.nlev, 0));
    isotropy_zi.resize(iface_box, 1);
    tkh_zi.resize(iface_box, 1);
    tk_zi.resize(iface_box, 1);

    interpolate_cc_to_iface(col, col.isotropy, isotropy_zi, 0.0);
    interpolate_cc_to_iface(col, col.tkh, tkh_zi, 0.0);
    interpolate_cc_to_iface(col, col.tk, tk_zi, 0.0);

    auto thl_sec = col.thl_sec.array();
    auto qw_sec = col.qw_sec.array();
    auto qwthl_sec = col.qwthl_sec.array();
    auto wthl_sec = col.wthl_sec.array();
    auto wqw_sec = col.wqw_sec.array();
    auto uw_sec = col.uw_sec.array();
    auto vw_sec = col.vw_sec.array();
    auto wtke_sec = col.wtke_sec.array();
    auto w_sec = col.w_sec.array();

    const auto thetal = col.thetal.const_array();
    const auto qw = col.qw.const_array();
    const auto tke = col.tke.const_array();
    const auto u = col.u.const_array();
    const auto v = col.v.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 0; k < col.layout.nlev; ++k) {
            w_sec(ic,k,0) = opts.shoc_1p5tke ? 0.0 : opts.w2tune * (2.0 / 3.0) * tke(ic,k,0);
        }
    }

    if (opts.shoc_1p5tke) {
        for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 0; k <= col.layout.nlev; ++k) {
            thl_sec(ic,k,0) = 0.0;
            qw_sec(ic,k,0) = 0.0;
            qwthl_sec(ic,k,0) = 0.0;
            }
        }
    } else {
        calc_var_or_covar(col, opts.thl2tune, isotropy_zi, tkh_zi, col.thetal, col.thetal, col.thl_sec);
        calc_var_or_covar(col, opts.qw2tune, isotropy_zi, tkh_zi, col.qw, col.qw, col.qw_sec);
        calc_var_or_covar(col, opts.qwthl2tune, isotropy_zi, tkh_zi, col.thetal, col.qw, col.qwthl_sec);
    }

    calc_vertflux(col, tkh_zi, col.thetal, col.wthl_sec);
    calc_vertflux(col, tkh_zi, col.qw, col.wqw_sec);
    calc_vertflux(col, tkh_zi, col.tke, col.wtke_sec);
    calc_vertflux(col, tk_zi, col.u, col.uw_sec);
    calc_vertflux(col, tk_zi, col.v, col.vw_sec);

    apply_second_moment_boundary_conditions(col);
    apply_top_taper_to_second_moments(col, opts);
}

void
ShocMoments::clip_third_moments (const ShocColumnData& col,
                                 const FArrayBox& w_sec_zi,
                                 FArrayBox& w3_fab)
{
    auto w3 = w3_fab.array();
    const auto wsec = w_sec_zi.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 0; k <= col.layout.nlev; ++k) {
            const Real clip_cond = k_shoc_w3clip * std::sqrt(std::max(0.0, 2.0 * std::pow(wsec(ic,k,0), 3)));
            if (std::abs(w3(ic,k,0)) > clip_cond) {
                w3(ic,k,0) = k_shoc_w3clipdef;
            }
        }
    }
}

void
ShocMoments::diagnose_third_moments (ShocColumnData& col,
                                     const ShocRuntimeOptions& opts)
{
    FArrayBox isotropy_zi, brunt_zi, w_sec_zi, thetal_zi;
    const Box iface_box(IntVect(0,0,0), IntVect(col.layout.ncell - 1, col.layout.nlev, 0));
    isotropy_zi.resize(iface_box, 1);
    brunt_zi.resize(iface_box, 1);
    w_sec_zi.resize(iface_box, 1);
    thetal_zi.resize(iface_box, 1);

    interpolate_cc_to_iface(col, col.isotropy, isotropy_zi, 0.0);
    interpolate_cc_to_iface(col, col.brunt, brunt_zi, k_shoc_large_neg);
    interpolate_cc_to_iface(col, col.w_sec, w_sec_zi, (2.0 / 3.0) * k_shoc_min_tke);
    interpolate_cc_to_iface(col, col.thetal, thetal_zi, 0.0);

    auto w3 = col.w3.array();
    const auto w_sec = col.w_sec.const_array();
    const auto thl_sec = col.thl_sec.const_array();
    const auto wthl_sec = col.wthl_sec.const_array();
    const auto tke = col.tke.const_array();
    const auto dz = col.dz.const_array();
    const auto isotropy_i = isotropy_zi.const_array();
    const auto brunt_i = brunt_zi.const_array();
    const auto w_sec_i = w_sec_zi.const_array();
    const auto thetal_i = thetal_zi.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        w3(ic,0,0) = 0.0;
        w3(ic,col.layout.nlev,0) = 0.0;

        for (int k = 1; k < col.layout.nlev; ++k) {
            if (opts.shoc_1p5tke) {
                w3(ic,k,0) = 0.0;
                continue;
            }

            const Real dz_zt_k = std::max(dz(ic,k,0), 1.0e-12);
            const Real dz_zt_km1 = std::max(dz(ic,k-1,0), 1.0e-12);
            const Real dz_zi = cell_spacing_at_iface(col, ic, k);
            const Real thedz = 1.0 / dz_zi;
            const Real thedz2 = 1.0 / (dz_zt_k + dz_zt_km1);
            const Real iso = isotropy_i(ic,k,0);
            const Real isosq = iso * iso;
            const Real buoy_sgs2 = isosq * brunt_i(ic,k,0);
            const Real bet2 = CONST_GRAV / std::max(thetal_i(ic,k,0), 1.0e-12);

            // E3SM's top-down indexing forms above-minus-below centered
            // differences here. In ERF's bottom-up ordering, the upper
            // neighbor is k+1 and the lower neighbor is k-1, while the cell
            // just above interface k is k and the one below is k-1.
            const Real thl_sec_diff = thl_sec(ic,std::min(k+1, col.layout.nlev),0) - thl_sec(ic,k-1,0);
            const Real wthl_sec_diff = wthl_sec(ic,std::min(k+1, col.layout.nlev),0) - wthl_sec(ic,k-1,0);
            const Real wsec_diff = w_sec(ic,k,0) - w_sec(ic,k-1,0);
            const Real tke_diff = tke(ic,k,0) - tke(ic,k-1,0);

            const Real c = opts.c_diag_3rd_mom;
            const Real a0 = (0.52 / (c * c)) / std::max(c - 2.0, 1.0e-12);
            const Real a1 = 0.87 / (c * c);
            const Real a2 = 0.5 / c;
            const Real a3 = 0.6 / (c * std::max(c - 2.0, 1.0e-12));
            const Real a4 = 2.4 / (3.0 * c + 5.0);
            const Real a5 = 0.6 / (c * (3.0 + 5.0 * c));

            const Real f0 = thedz2 * std::pow(bet2, 3) * std::pow(isosq, 2) *
                            wthl_sec(ic,k,0) * thl_sec_diff;
            const Real f1 = thedz2 * (bet2 * bet2) * std::pow(iso, 3) *
                            (wthl_sec(ic,k,0) * wthl_sec_diff +
                             0.5 * w_sec_i(ic,k,0) * thl_sec_diff);
            const Real f2 = thedz * bet2 * isosq * wthl_sec(ic,k,0) * wsec_diff +
                            2.0 * thedz2 * bet2 * isosq * w_sec_i(ic,k,0) * wthl_sec_diff;
            const Real f3 = thedz2 * bet2 * isosq * w_sec_i(ic,k,0) * wthl_sec_diff +
                            thedz * bet2 * isosq * (wthl_sec(ic,k,0) * tke_diff);
            const Real f4 = thedz * iso * w_sec_i(ic,k,0) * (wsec_diff + tke_diff);
            const Real f5 = thedz * iso * w_sec_i(ic,k,0) * wsec_diff;

            const Real denom0 = signed_denominator(1.0 - (a1 + a3) * buoy_sgs2);
            const Real omega0 = a4 / signed_denominator(1.0 - a5 * buoy_sgs2);
            const Real omega1 = omega0 / (2.0 * c);
            const Real omega2 = omega1 * f3 + 1.25 * omega0 * f4;
            const Real x0 = (a2 * buoy_sgs2 * (1.0 - a3 * buoy_sgs2)) / denom0;
            const Real y0 = (2.0 * a2 * buoy_sgs2 * x0) / signed_denominator(1.0 - a3 * buoy_sgs2);
            const Real x1 = (a0 * f0 + a1 * f1 + a2 * (1.0 - a3 * buoy_sgs2) * f2) / denom0;
            const Real y1 = (2.0 * a2 * (buoy_sgs2 * x1 + (a0 / std::max(a1, 1.0e-12)) * f0 + f1)) /
                            signed_denominator(1.0 - a3 * buoy_sgs2);
            const Real aa0 = omega0 * x0 + omega1 * y0;
            const Real aa1 = omega0 * x1 + omega1 * y1 + omega2;

            const Real denom = signed_denominator(c - 1.2 * x0 + aa0);
            Real w3_val = (aa1 - 1.2 * x1 - 1.5 * f5) / denom;
            w3(ic,k,0) = w3_val;
        }
    }

    clip_third_moments(col, w_sec_zi, col.w3);
    apply_top_taper_to_third_moments(col, opts);
}

void
ShocMoments::diagnose_moments (ShocColumnData& col,
                               const ShocRuntimeOptions& opts)
{
    diagnose_second_moments(col, opts);
    diagnose_third_moments(col, opts);
}
