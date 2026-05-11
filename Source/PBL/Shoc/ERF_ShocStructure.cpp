#include "ERF_ShocStructure.H"

#include "ERF_Constants.H"

#include <algorithm>
#include <cmath>

using namespace amrex;

namespace
{
    constexpr Real k_shoc_zvir = 0.61;
    constexpr Real k_shoc_u_star_min = 0.01;
    constexpr Real k_shoc_min_tke = 4.0e-4;
    constexpr Real k_shoc_min_len = 20.0;
    constexpr Real k_shoc_max_len = 2.0e4;
    constexpr Real k_shoc_pblmaxp = 4.0e4;
    constexpr Real k_shoc_pbl_ricr = 0.3;
    constexpr Real k_shoc_pbl_fac = 100.0;
    constexpr Real k_shoc_pbl_fak = 8.5;
    constexpr Real k_shoc_pbl_betam = 15.0;
    constexpr Real k_shoc_pbl_sffrac = 0.1;
    constexpr Real k_shoc_kbfs_eps = 1.0e-10;

    AMREX_FORCE_INLINE
    Real weighted_linear_interp (Real x0, Real x1, Real y0, Real y1, Real x)
    {
        const Real denom = x1 - x0;
        if (std::abs(denom) <= 1.0e-12) {
            return 0.5 * (y0 + y1);
        }
        return y0 + (y1 - y0) * (x - x0) / denom;
    }

    int diagnose_npbl (const ShocColumnData& col, int ic)
    {
        const auto p_mid = col.p_mid.const_array();
        int npbl = 1;
        for (int k = 0; k < col.layout.nlev; ++k) {
            if (p_mid(ic,k,0) >= k_shoc_pblmaxp) {
                npbl = k + 1;
            }
        }
        return npbl;
    }

    AMREX_FORCE_INLINE
    Real theta_from_shoc_state (Real thetal, Real ql, Real exner)
    {
        return thetal + (L_v / Cp_d) * ql / std::max(exner, 1.0e-12);
    }

    AMREX_FORCE_INLINE
    Real virtual_theta_from_shoc_state (Real thetal, Real ql, Real qv, Real exner)
    {
        const Real theta = theta_from_shoc_state(thetal, ql, exner);
        return theta * (1.0 + k_shoc_zvir * qv - ql);
    }
}

void
ShocStructure::diagnose_surface_layer (ShocColumnData& col)
{
    auto ustar = col.ustar.array();
    auto obklen = col.obklen.array();
    auto wthv_sec = col.wthv_sec.array();
    auto pblh = col.pblh.array();
    const auto thetal = col.thetal.const_array();
    const auto qc = col.qc.const_array();
    const auto qi = col.qi.const_array();
    const auto qv = col.qv.const_array();
    const auto exner = col.exner.const_array();
    const auto sflux = col.surf_sens_flux.const_array();
    const auto lflux = col.surf_lat_flux.const_array();
    const auto tauu = col.surf_tau_u.const_array();
    const auto tauv = col.surf_tau_v.const_array();
    const auto zt = col.zt.const_array();
    const int nlev = col.layout.nlev;

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        const Real cldliq_sfc = qc(ic,0,0) + qi(ic,0,0);
        const Real th_sfc = theta_from_shoc_state(thetal(ic,0,0), cldliq_sfc, exner(ic,0,0));
        const Real thv_sfc = th_sfc * (1.0 + k_shoc_zvir * qv(ic,0,0) - cldliq_sfc);
        const Real stress_mag = std::sqrt(tauu(ic,0,0) * tauu(ic,0,0) +
                                          tauv(ic,0,0) * tauv(ic,0,0));
        const Real ustar_val = std::sqrt(stress_mag);
        const Real kbfs = sflux(ic,0,0) + k_shoc_zvir * th_sfc * lflux(ic,0,0);
        const Real sign_val = (kbfs >= 0.0) ? k_shoc_kbfs_eps : -k_shoc_kbfs_eps;

        ustar(ic,0,0) = std::max(k_shoc_u_star_min, ustar_val);
        obklen(ic,0,0) = -thv_sfc * std::pow(ustar(ic,0,0), 3) /
                         (CONST_GRAV * KAPPA * (kbfs + sign_val));
        pblh(ic,0,0) = zt(ic,0,0);

        // E3SM SHOC advances TKE using the carried buoyancy-flux profile from
        // the previous SHOC call. ERF overwrites only the surface entry here
        // with the current lower-boundary forcing and preserves the interior
        // profile that was diagnosed later in the previous SHOC call.
        wthv_sec(ic,0,0) = kbfs;
    }
}

void
ShocStructure::diagnose_pblh (ShocColumnData& col)
{
    auto pblh = col.pblh.array();
    const auto zt = col.zt.const_array();
    const auto zi = col.zi.const_array();
    const auto u = col.u.const_array();
    const auto v = col.v.const_array();
    const auto ustar = col.ustar.const_array();
    const auto obklen = col.obklen.const_array();
    const auto wthv_sec = col.wthv_sec.const_array();
    const auto thetal = col.thetal.const_array();
    const auto qc = col.qc.const_array();
    const auto qi = col.qi.const_array();
    const auto qv = col.qv.const_array();
    const auto exner = col.exner.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        const int npbl = diagnose_npbl(col, ic);
        const Real ustar_loc = ustar(ic,0,0);
        Real pblh_loc = zt(ic,npbl-1,0);
        Vector<Real> rino(col.layout.nlev, 0.0);
        Vector<Real> thv(col.layout.nlev, 0.0);

        for (int k = 0; k < col.layout.nlev; ++k) {
            thv[k] = virtual_theta_from_shoc_state(thetal(ic,k,0), qc(ic,k,0) + qi(ic,k,0),
                                                   qv(ic,k,0), exner(ic,k,0));
        }

        bool found_pblh = false;
        for (int k = 1; k < npbl; ++k) {
            const Real vvk = std::max(1.0e-36,
                                      std::pow(u(ic,k,0) - u(ic,0,0), 2) +
                                      std::pow(v(ic,k,0) - v(ic,0,0), 2) +
                                      k_shoc_pbl_fac * ustar_loc * ustar_loc);
            rino[k] = CONST_GRAV * (thv[k] - thv[0]) * (zt(ic,k,0) - zt(ic,0,0)) /
                      (std::max(thv[0], 1.0e-12) * vvk);
            if (rino[k] >= k_shoc_pbl_ricr) {
                const Real r0 = rino[k-1];
                const Real r1 = rino[k];
                if (k == 1 || std::abs(r1 - r0) <= 1.0e-12) {
                    pblh_loc = zt(ic,k,0);
                } else {
                    pblh_loc = zt(ic,k-1,0) + (k_shoc_pbl_ricr - r0) * (zt(ic,k,0) - zt(ic,k-1,0)) / (r1 - r0);
                }
                found_pblh = true;
                break;
            }
        }

        if (!found_pblh) {
            pblh_loc = zt(ic,npbl-1,0);
        }

        if (wthv_sec(ic,0,0) > 0.0) {
            const Real obk = std::copysign(std::max(std::abs(obklen(ic,0,0)), 1.0e-6), obklen(ic,0,0));
            const Real binm = k_shoc_pbl_betam * k_shoc_pbl_sffrac;
            const Real phiminv = std::cbrt(std::max(1.0e-12, 1.0 - binm * pblh_loc / obk));
            const Real tlv = thv[0] + wthv_sec(ic,0,0) * k_shoc_pbl_fak /
                                       (std::max(ustar_loc, k_shoc_u_star_min) * phiminv);
            for (int k = 1; k < npbl; ++k) {
                const Real vvk = std::max(1.0e-36,
                                          std::pow(u(ic,k,0) - u(ic,0,0), 2) +
                                          std::pow(v(ic,k,0) - v(ic,0,0), 2) +
                                          k_shoc_pbl_fac * ustar_loc * ustar_loc);
                rino[k] = CONST_GRAV * (thv[k] - tlv) * (zt(ic,k,0) - zt(ic,0,0)) /
                          (std::max(thv[0], 1.0e-12) * vvk);
                if (rino[k] >= k_shoc_pbl_ricr) {
                    const Real r0 = rino[k-1];
                    const Real r1 = rino[k];
                    if (k == 1 || std::abs(r1 - r0) <= 1.0e-12) {
                        pblh_loc = zt(ic,k,0);
                    } else {
                        pblh_loc = zt(ic,k-1,0) + (k_shoc_pbl_ricr - r0) * (zt(ic,k,0) - zt(ic,k-1,0)) / (r1 - r0);
                    }
                    break;
                }
            }
        }

        pblh_loc = std::max(pblh_loc, 700.0 * ustar_loc);
        if (qc(ic,0,0) + qi(ic,0,0) > 0.0) {
            pblh_loc = std::max(pblh_loc, zi(ic,1,0) + 50.0);
        }
        pblh(ic,0,0) = pblh_loc;
    }
}

void
ShocStructure::diagnose_length_and_brunt (ShocColumnData& col,
                                          const ShocRuntimeOptions& opts,
                                          Real dx,
                                          Real dy)
{
    auto brunt = col.brunt.array();
    auto shoc_mix = col.shoc_mix.array();
    const auto zt = col.zt.const_array();
    const auto zi = col.zi.const_array();
    const auto dz = col.dz.const_array();
    const auto thetal = col.thetal.const_array();
    const auto qc = col.qc.const_array();
    const auto qi = col.qi.const_array();
    const auto qv = col.qv.const_array();
    const auto exner = col.exner.const_array();
    const auto tke = col.tke.const_array();
    const Real max_horiz_len = std::sqrt(dx * dy);

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        Vector<Real> theta_v_cc(col.layout.nlev, 0.0);
        Vector<Real> theta_v_iface(col.layout.nlev + 1, 0.0);
        for (int k = 0; k < col.layout.nlev; ++k) {
            theta_v_cc[k] = virtual_theta_from_shoc_state(thetal(ic,k,0), qc(ic,k,0) + qi(ic,k,0),
                                                          qv(ic,k,0), exner(ic,k,0));
        }
        if (col.layout.nlev == 1) {
            theta_v_iface[0] = theta_v_cc[0];
            theta_v_iface[1] = theta_v_cc[0];
        } else {
            theta_v_iface[0] = weighted_linear_interp(zt(ic,0,0), zt(ic,1,0),
                                                      theta_v_cc[0], theta_v_cc[1],
                                                      zi(ic,0,0));
            for (int k = 1; k < col.layout.nlev; ++k) {
                theta_v_iface[k] = weighted_linear_interp(zt(ic,k-1,0), zt(ic,k,0),
                                                          theta_v_cc[k-1], theta_v_cc[k],
                                                          zi(ic,k,0));
            }
            theta_v_iface[col.layout.nlev] =
                weighted_linear_interp(zt(ic,col.layout.nlev-2,0), zt(ic,col.layout.nlev-1,0),
                                       theta_v_cc[col.layout.nlev-2], theta_v_cc[col.layout.nlev-1],
                                       zi(ic,col.layout.nlev,0));
        }

        Real numer = 0.0;
        Real denom = 0.0;
        for (int k = 0; k < col.layout.nlev; ++k) {
            const Real tke_sqrt = std::sqrt(std::max(tke(ic,k,0), k_shoc_min_tke));
            numer += tke_sqrt * zt(ic,k,0) * dz(ic,k,0);
            denom += tke_sqrt * dz(ic,k,0);
        }
        const Real l_inf = (denom > 0.0) ? 0.1 * numer / denom : k_shoc_min_len;

        for (int k = 0; k < col.layout.nlev; ++k) {
            // ERF columns use bottom-up indexing, so the upper interface is
            // k+1 and the lower interface is k. Stable stratification must
            // therefore yield positive Brunt-Vaisala frequency.
            brunt(ic,k,0) = (CONST_GRAV / std::max(theta_v_cc[k], 1.0e-12)) *
                            (theta_v_iface[k+1] - theta_v_iface[k]) / std::max(dz(ic,k,0), 1.0e-12);
            const Real tkes = std::sqrt(std::max(tke(ic,k,0), k_shoc_min_tke));
            const Real brunt_pos = std::max(brunt(ic,k,0), 0.0);
            const Real inv_term = (1.0 / std::max(400.0 * tkes * KAPPA * std::max(zt(ic,k,0), 1.0), 1.0e-12)) +
                                  (1.0 / std::max(400.0 * tkes * std::max(l_inf, k_shoc_min_len), 1.0e-12)) +
                                  0.01 * brunt_pos / std::max(tke(ic,k,0), k_shoc_min_tke);
            Real mix = std::min(k_shoc_max_len,
                                2.8284 * std::sqrt(1.0 / std::max(inv_term, 1.0e-12)) /
                                std::max(opts.length_fac, 1.0e-12));
            mix = std::min(max_horiz_len, std::max(k_shoc_min_len, mix));
            shoc_mix(ic,k,0) = mix;
        }
    }
}
