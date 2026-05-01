#include "ERF_ShocTKE.H"

#include "ERF_Constants.H"

#include <algorithm>
#include <cmath>

using namespace amrex;

namespace
{
    constexpr Real k_shoc_basetemp = 300.0;
    constexpr Real k_shoc_min_tke = 4.0e-4;
    constexpr Real k_shoc_max_tke = 50.0;
    constexpr Real k_shoc_max_iso = 2.0e4;
    constexpr Real k_shoc_tabs_crit = 182.0;
    constexpr Real k_shoc_pbl_trans = 200.0;
    constexpr Real k_shoc_stable_ckh = 0.1;
    constexpr Real k_shoc_stable_ckm = 0.1;
    constexpr Real k_shoc_shear_ck = 0.1;
    constexpr Real k_shoc_tke_cs = 0.15;
    constexpr Real k_shoc_tke_ce =
        (k_shoc_shear_ck * k_shoc_shear_ck * k_shoc_shear_ck) /
        ((k_shoc_tke_cs * k_shoc_tke_cs) * (k_shoc_tke_cs * k_shoc_tke_cs));
    constexpr Real k_shoc_tke_cee =
        (k_shoc_tke_ce / 0.7) * 0.19 + (k_shoc_tke_ce / 0.7) * 0.51;
    constexpr Real k_shoc_trop_pres = 8.0e4;

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

    void interpolate_iface_to_cc (const ShocColumnData& col,
                                  const FArrayBox& iface_in,
                                  FArrayBox& cc_out,
                                  Real min_thresh)
    {
        const auto zt = col.zt.const_array();
        const auto zi = col.zi.const_array();
        const auto in = iface_in.const_array();
        auto out = cc_out.array();

        for (int ic = 0; ic < col.layout.ncell; ++ic) {
            for (int k = 0; k < col.layout.nlev; ++k) {
                const Real interp = weighted_linear_interp(zi(ic,k,0), zi(ic,k+1,0),
                                                           in(ic,k,0), in(ic,k+1,0),
                                                           zt(ic,k,0));
                out(ic,k,0) = std::max(min_thresh, interp);
            }
        }
    }
}

void
ShocTKE::compute_shear_production (const ShocColumnData& col,
                                   FArrayBox& sterm_iface)
{
    const Box iface_box(IntVect(0,0,0), IntVect(col.layout.ncell - 1, col.layout.nlev, 0));
    sterm_iface.resize(iface_box, 1);
    sterm_iface.setVal<RunOn::Host>(0.0);

    auto sterm = sterm_iface.array();
    const auto u = col.u.const_array();
    const auto v = col.v.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        sterm(ic,0,0) = 0.0;
        sterm(ic,col.layout.nlev,0) = 0.0;
        for (int k = 1; k < col.layout.nlev; ++k) {
            const Real dz_zi = interface_spacing(col, ic, k);
            const Real du_dz = (u(ic,k-1,0) - u(ic,k,0)) / dz_zi;
            const Real dv_dz = (v(ic,k-1,0) - v(ic,k,0)) / dz_zi;
            sterm(ic,k,0) = k_shoc_shear_ck * (du_dz * du_dz + dv_dz * dv_dz);
        }
    }
}

void
ShocTKE::integrate_column_stability (const ShocColumnData& col,
                                     Vector<Real>& brunt_int)
{
    brunt_int.assign(col.layout.ncell, 0.0);
    const auto dz = col.dz.const_array();
    const auto p_mid = col.p_mid.const_array();
    const auto brunt = col.brunt.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 0; k < col.layout.nlev; ++k) {
            if (p_mid(ic,k,0) > k_shoc_trop_pres) {
                brunt_int[ic] += dz(ic,k,0) * brunt(ic,k,0);
            }
        }
    }
}

void
ShocTKE::diagnose_tke_and_diffusivities (ShocColumnData& col,
                                         const ShocRuntimeOptions& opts,
                                         Real dt)
{
    AMREX_ALWAYS_ASSERT(dt > 0.0);

    const Box cell_box(IntVect(0,0,0), IntVect(col.layout.ncell - 1, col.layout.nlev - 1, 0));
    FArrayBox sterm_iface;
    FArrayBox sterm_zt(cell_box, 1);
    FArrayBox a_diss(cell_box, 1);

    compute_shear_production(col, sterm_iface);
    interpolate_iface_to_cc(col, sterm_iface, sterm_zt, 0.0);

    Vector<Real> brunt_int;
    integrate_column_stability(col, brunt_int);

    auto tke = col.tke.array();
    auto isotropy = col.isotropy.array();
    auto tk = col.tk.array();
    auto tkh = col.tkh.array();
    auto tke_tend = col.tke_tend.array();
    auto shear_prod_arr = col.shear_prod.array();
    auto buoy_prod_arr = col.buoy_prod.array();
    auto diss_arr = col.diss_tke.array();

    const auto wthv_sec = col.wthv_sec.const_array();
    const auto shoc_mix = col.shoc_mix.const_array();
    const auto brunt = col.brunt.const_array();
    const auto zt = col.zt.const_array();
    const auto tabs = col.tabs.const_array();
    const auto pblh = col.pblh.const_array();
    const auto sterm = sterm_zt.const_array();
    auto a_diss_arr = a_diss.array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 0; k < col.layout.nlev; ++k) {
            const Real old_tke = std::max(0.0, tke(ic,k,0));
            const Real buoy_prod = opts.shoc_1p5tke
                ? -old_tke * brunt(ic,k,0)
                : (CONST_GRAV / k_shoc_basetemp) * wthv_sec(ic,k,0);
            const Real shear_prod = tk(ic,k,0) * sterm(ic,k,0);
            const Real mix = std::max(shoc_mix(ic,k,0), 1.0e-12);
            const Real diss = k_shoc_tke_cee / mix * std::pow(old_tke, 1.5);
            const Real net_prod = opts.signed_tke_production
                ? (shear_prod + buoy_prod)
                : std::max(0.0, shear_prod + buoy_prod);
            const Real new_tke = std::clamp(old_tke + dt * (net_prod - diss),
                                            k_shoc_min_tke, k_shoc_max_tke);

            a_diss_arr(ic,k,0) = diss;
            shear_prod_arr(ic,k,0) = shear_prod;
            buoy_prod_arr(ic,k,0) = buoy_prod;
            diss_arr(ic,k,0) = diss;
            tke_tend(ic,k,0) = new_tke - old_tke;
            tke(ic,k,0) = new_tke;
        }

        Real lambda = opts.lambda_low + ((brunt_int[ic] / CONST_GRAV) - opts.lambda_thresh) * opts.lambda_slope;
        lambda = std::clamp(lambda, opts.lambda_low, opts.lambda_high);

        for (int k = 0; k < col.layout.nlev; ++k) {
            const Real diss = std::max(a_diss_arr(ic,k,0), 1.0e-12);
            const Real tscale = 2.0 * tke(ic,k,0) / diss;
            const Real local_lambda = (brunt(ic,k,0) <= 0.0) ? 0.0 : lambda;
            isotropy(ic,k,0) = std::min(k_shoc_max_iso,
                                        tscale / (1.0 + local_lambda * brunt(ic,k,0) * tscale * tscale));

            const bool use_stable_mix = (zt(ic,k,0) < pblh(ic,0,0) + k_shoc_pbl_trans) &&
                                        (tabs(ic,0,0) < k_shoc_tabs_crit);
            if (use_stable_mix) {
                const Real sterm_sqrt = std::sqrt(std::max(sterm(ic,k,0), 0.0));
                tkh(ic,k,0) = k_shoc_stable_ckh * shoc_mix(ic,k,0) * shoc_mix(ic,k,0) * sterm_sqrt;
                tk(ic,k,0) = k_shoc_stable_ckm * shoc_mix(ic,k,0) * shoc_mix(ic,k,0) * sterm_sqrt;
            } else {
                tkh(ic,k,0) = opts.coeff_kh * isotropy(ic,k,0) * tke(ic,k,0);
                tk(ic,k,0) = opts.coeff_km * isotropy(ic,k,0) * tke(ic,k,0);
            }

            tkh(ic,k,0) = std::max(0.0, tkh(ic,k,0));
            tk(ic,k,0) = std::max(0.0, tk(ic,k,0));
        }
    }
}
