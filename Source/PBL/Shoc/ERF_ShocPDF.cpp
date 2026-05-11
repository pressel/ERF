#include "ERF_ShocPDF.H"

#include "ERF_Constants.H"
#include "ERF_MicrophysicsUtils.H"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace amrex;

namespace
{
    constexpr Real k_shoc_thl_tol = 1.0e-2;
    constexpr Real k_shoc_rt_tol = 1.0e-4;
    constexpr Real k_shoc_w_tol_sqd = 4.0e-4;
    constexpr Real k_shoc_w_thresh = 0.0;
    constexpr Real k_shoc_large_neg = -99999999.99;
    constexpr Real k_shoc_base_temp = 300.0;
    constexpr Real k_shoc_tl_min = 100.0;
    constexpr Real k_shoc_pdf_tmp = 0.4;
    constexpr Real k_shoc_sqrt_pdf_tmp = 0.77459666924148337704; // sqrt(1-0.4)
    constexpr bool k_shoc_do_thetal_skew = false;
    constexpr Real k_inv_sqrt_two = 0.70710678118654752440;
    constexpr Real k_inv_sqrt_two_pi = 0.39894228040143267794;

    AMREX_FORCE_INLINE
    Real clamp01 (Real x) { return std::clamp(x, 0.0, 1.0); }

    AMREX_FORCE_INLINE
    Real signed_denominator (Real value, Real eps = 1.0e-12)
    {
        if (std::abs(value) >= eps) {
            return value;
        }
        return std::copysign(eps, value == 0.0 ? 1.0 : value);
    }

    AMREX_FORCE_INLINE
    Real weighted_linear_interp (Real x0, Real x1, Real y0, Real y1, Real x)
    {
        const Real denom = x1 - x0;
        if (std::abs(denom) <= 1.0e-12) {
            return 0.5 * (y0 + y1);
        }
        return y0 + (y1 - y0) * (x - x0) / denom;
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

    AMREX_FORCE_INLINE
    void vv_parameters (Real w_first, Real w_sec, Real w3var,
                        Real& skew_w, Real& w1_1, Real& w1_2,
                        Real& w2_1, Real& w2_2, Real& a)
    {
        skew_w = 0.0;
        w1_1 = w_first;
        w1_2 = w_first;
        w2_1 = 0.0;
        w2_2 = 0.0;
        a = 0.5;

        if (w_sec > k_shoc_w_tol_sqd) {
            const Real skew_denom = std::sqrt(std::max(std::pow(w_sec, 3), 1.0e-18));
            skew_w = w3var / skew_denom;
            const Real skew_term = std::sqrt(1.0 /
                (4.0 * std::pow(1.0 - k_shoc_pdf_tmp, 3) + skew_w * skew_w));
            a = std::clamp(0.5 * (1.0 - skew_w * skew_term), 0.01, 0.99);
            w1_1 = std::sqrt((1.0 - a) / a) * k_shoc_sqrt_pdf_tmp;
            w1_2 = -std::sqrt(a / (1.0 - a)) * k_shoc_sqrt_pdf_tmp;
            w2_1 = k_shoc_pdf_tmp * w_sec;
            w2_2 = k_shoc_pdf_tmp * w_sec;
        }
    }

    AMREX_FORCE_INLINE
    void thl_parameters (Real wthlsec, Real sqrtw2, Real sqrtthl,
                         Real thlsec, Real thl_first, Real w1_1, Real w1_2,
                         Real skew_w, Real a,
                         Real& thl1_1, Real& thl1_2, Real& thl2_1,
                         Real& thl2_2, Real& sqrtthl2_1, Real& sqrtthl2_2)
    {
        thl1_1 = thl_first;
        thl1_2 = thl_first;
        thl2_1 = 0.0;
        thl2_2 = 0.0;
        sqrtthl2_1 = 0.0;
        sqrtthl2_2 = 0.0;

        if (thlsec > k_shoc_thl_tol * k_shoc_thl_tol &&
            std::abs(w1_2 - w1_1) > k_shoc_w_thresh &&
            sqrtw2 > 0.0 && sqrtthl > 0.0) {
            const Real corr = std::clamp(wthlsec / (sqrtw2 * sqrtthl), -1.0, 1.0);
            const Real tmp1 = -corr / w1_1;
            const Real tmp2 = -corr / w1_2;
            const Real tsign = std::abs(tmp1 - tmp2);
            Real skew_thl = 0.0;
            if (k_shoc_do_thetal_skew) {
                if (tsign > 0.4) {
                    skew_thl = 1.2 * skew_w;
                } else if (tsign > 0.2) {
                    skew_thl = ((1.2 * skew_w) / 0.2) * (tsign - 0.2);
                }
            }

            thl2_1 = std::min(100.0, std::max(0.0,
                (3.0 * tmp1 * (1.0 - a * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1)
                 - (skew_thl - a * tmp2 * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1 * tmp1)) /
                signed_denominator(3.0 * a * (tmp1 - tmp2)))) * thlsec;

            thl2_2 = std::min(100.0, std::max(0.0,
                (-3.0 * tmp2 * (1.0 - a * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1)
                 + (skew_thl - a * tmp2 * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1 * tmp1)) /
                signed_denominator(3.0 * (1.0 - a) * (tmp1 - tmp2)))) * thlsec;

            thl1_1 = tmp2 * sqrtthl + thl_first;
            thl1_2 = tmp1 * sqrtthl + thl_first;
            sqrtthl2_1 = std::sqrt(std::max(thl2_1, 0.0));
            sqrtthl2_2 = std::sqrt(std::max(thl2_2, 0.0));
        }
    }

    AMREX_FORCE_INLINE
    void qw_parameters (Real wqwsec, Real sqrtw2, Real skew_w, Real sqrtqt,
                        Real qwsec, Real w1_2, Real w1_1, Real qw_first, Real a,
                        Real& qw1_1, Real& qw1_2, Real& qw2_1, Real& qw2_2,
                        Real& sqrtqw2_1, Real& sqrtqw2_2)
    {
        qw1_1 = qw_first;
        qw1_2 = qw_first;
        qw2_1 = 0.0;
        qw2_2 = 0.0;
        sqrtqw2_1 = 0.0;
        sqrtqw2_2 = 0.0;

        if (qwsec > k_shoc_rt_tol * k_shoc_rt_tol &&
            std::abs(w1_2 - w1_1) > k_shoc_w_thresh &&
            sqrtw2 > 0.0 && sqrtqt > 0.0) {
            const Real corr = std::clamp(wqwsec / (sqrtw2 * sqrtqt), -1.0, 1.0);
            const Real tmp1 = -corr / w1_1;
            const Real tmp2 = -corr / w1_2;
            const Real tsign = std::abs(tmp1 - tmp2);
            Real skew_qw = 0.0;
            if (tsign > 0.4) {
                skew_qw = 1.2 * skew_w;
            } else if (tsign > 0.2) {
                skew_qw = ((1.2 * skew_w) / 0.2) * (tsign - 0.2);
            }

            qw2_1 = std::min(100.0, std::max(0.0,
                (3.0 * tmp1 * (1.0 - a * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1)
                 - (skew_qw - a * tmp2 * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1 * tmp1)) /
                signed_denominator(3.0 * a * (tmp1 - tmp2)))) * qwsec;

            qw2_2 = std::min(100.0, std::max(0.0,
                (-3.0 * tmp2 * (1.0 - a * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1)
                 + (skew_qw - a * tmp2 * tmp2 * tmp2 - (1.0 - a) * tmp1 * tmp1 * tmp1)) /
                signed_denominator(3.0 * (1.0 - a) * (tmp1 - tmp2)))) * qwsec;

            qw1_1 = tmp2 * sqrtqt + qw_first;
            qw1_2 = tmp1 * sqrtqt + qw_first;
            sqrtqw2_1 = std::sqrt(std::max(qw2_1, 0.0));
            sqrtqw2_2 = std::sqrt(std::max(qw2_2, 0.0));
        }
    }

    AMREX_FORCE_INLINE
    Real inplume_correlation (Real sqrtqw2_1, Real sqrtthl2_1, Real a,
                              Real sqrtqw2_2, Real sqrtthl2_2,
                              Real qwthlsec, Real qw1_1, Real qw_first,
                              Real thl1_1, Real thl_first,
                              Real qw1_2, Real thl1_2)
    {
        const Real denom = a * sqrtqw2_1 * sqrtthl2_1 + (1.0 - a) * sqrtqw2_2 * sqrtthl2_2;
        if (std::abs(denom) <= 1.0e-18) {
            return 0.0;
        }
        return std::clamp((qwthlsec
                          - a * (qw1_1 - qw_first) * (thl1_1 - thl_first)
                          - (1.0 - a) * (qw1_2 - qw_first) * (thl1_2 - thl_first)) / denom,
                          -1.0, 1.0);
    }

    AMREX_FORCE_INLINE
    Real compute_temperature (Real thl1, Real pval)
    {
        return thl1 / std::pow(p_0 / std::max(pval, 1.0e-12), R_d / Cp_d);
    }

    AMREX_FORCE_INLINE
    void compute_qs_beta (Real temp, Real pval, Real& qs, Real& beta)
    {
        qs = 0.0;
        erf_qsatw(temp, 0.01 * pval, qs);
        qs = std::max(0.0, qs);
        beta = (R_d / R_v) * (L_v / (R_d * temp)) * (L_v / (Cp_d * temp));
    }

    AMREX_FORCE_INLINE
    void compute_s_terms (Real qw1, Real qs, Real beta, Real pval,
                          Real thl2, Real qw2, Real sqrtthl2, Real sqrtqw2,
                          Real r_qwthl, Real& s, Real& std_s, Real& qn, Real& cfrac)
    {
        const Real cthl = ((1.0 + beta * qw1) / std::pow(1.0 + beta * qs, 2))
                        * (Cp_d / L_v) * beta * qs * std::pow(pval / p_0, R_d / Cp_d);
        const Real cqt = 1.0 / (1.0 + beta * qs);
        std_s = std::sqrt(std::max(0.0, cthl * cthl * thl2
                                        + cqt * cqt * qw2
                                        - 2.0 * cthl * sqrtthl2 * cqt * sqrtqw2 * r_qwthl));
        s = qw1 - qs * ((1.0 + beta * qw1) / (1.0 + beta * qs));

        cfrac = 0.0;
        qn = 0.0;
        if (std_s > std::sqrt(std::numeric_limits<Real>::min()) * 100.0) {
            cfrac = 0.5 * (1.0 + std::erf(s / (std::sqrt(2.0) * std_s)));
            if (cfrac != 0.0) {
                qn = s * cfrac + std_s * k_inv_sqrt_two_pi * std::exp(-0.5 * (s / std_s) * (s / std_s));
            }
        } else if (s > 0.0) {
            cfrac = 1.0;
            qn = s;
        }

        if (qn <= 0.0) {
            cfrac = 0.0;
            qn = 0.0;
        }
    }

    AMREX_FORCE_INLINE
    Real compute_buoyancy_flux (Real wthlsec, Real wqwsec, Real pval, Real wqls)
    {
        const Real epsterm = R_d / R_v;
        return wthlsec
             + ((1.0 - epsterm) / epsterm) * k_shoc_base_temp * wqwsec
             + ((L_v / Cp_d) * std::pow(p_0 / std::max(pval, 1.0e-12), R_d / Cp_d)
                - (1.0 / epsterm) * k_shoc_base_temp) * wqls;
    }
}

void
ShocPDF::diagnose_pdf (ShocColumnData& col,
                       const ShocRuntimeOptions& opts,
                       Real dt)
{
    FArrayBox w3_zt, thl_sec_zt, qw_sec_zt, wthl_zt, wqw_zt, qwthl_zt;
    const Box cell_box(IntVect(0,0,0), IntVect(col.layout.ncell - 1, col.layout.nlev - 1, 0));
    w3_zt.resize(cell_box, 1);
    thl_sec_zt.resize(cell_box, 1);
    qw_sec_zt.resize(cell_box, 1);
    wthl_zt.resize(cell_box, 1);
    wqw_zt.resize(cell_box, 1);
    qwthl_zt.resize(cell_box, 1);

    interpolate_iface_to_cc(col, col.w3, w3_zt, k_shoc_large_neg);
    interpolate_iface_to_cc(col, col.thl_sec, thl_sec_zt, 0.0);
    interpolate_iface_to_cc(col, col.qw_sec, qw_sec_zt, 0.0);
    interpolate_iface_to_cc(col, col.wthl_sec, wthl_zt, k_shoc_large_neg);
    interpolate_iface_to_cc(col, col.wqw_sec, wqw_zt, k_shoc_large_neg);
    interpolate_iface_to_cc(col, col.qwthl_sec, qwthl_zt, k_shoc_large_neg);

    auto shoc_cldfrac = col.shoc_cldfrac.array();
    auto shoc_ql = col.shoc_ql.array();
    auto shoc_ql2 = col.shoc_ql2.array();
    auto shoc_cond = col.shoc_cond.array();
    auto shoc_evap = col.shoc_evap.array();
    auto wqls_sec = col.wqls_sec.array();
    auto wthv_sec = col.wthv_sec.array();

    const auto thetal = col.thetal.const_array();
    const auto qw = col.qw.const_array();
    const auto w_field = col.w.const_array();
    const auto p_mid = col.p_mid.const_array();
    const auto thl_sec = thl_sec_zt.const_array();
    const auto qw_sec = qw_sec_zt.const_array();
    const auto qwthl_sec = qwthl_zt.const_array();
    const auto wthl_sec = wthl_zt.const_array();
    const auto wqw_sec = wqw_zt.const_array();
    const auto w_sec = col.w_sec.const_array();
    const auto w3 = w3_zt.const_array();

    for (int ic = 0; ic < col.layout.ncell; ++ic) {
        for (int k = 0; k < col.layout.nlev; ++k) {
            const Real thl_first = thetal(ic,k,0);
            const Real qw_first = qw(ic,k,0);
            const Real w_first = w_field(ic,k,0);
            const Real w2sec = std::max(w_sec(ic,k,0), 0.0);
            const Real sqrtw2 = std::sqrt(w2sec);
            const Real sqrtthl = std::max(k_shoc_thl_tol, std::sqrt(std::max(thl_sec(ic,k,0), 0.0)));
            const Real sqrtqt = std::max(k_shoc_rt_tol, std::sqrt(std::max(qw_sec(ic,k,0), 0.0)));

            Real skew_w = 0.0, w1_1 = w_first, w1_2 = w_first, w2_1 = 0.0, w2_2 = 0.0, a = 0.5;
            vv_parameters(w_first, w2sec, w3(ic,k,0), skew_w, w1_1, w1_2, w2_1, w2_2, a);

            Real thl1_1 = thl_first, thl1_2 = thl_first, thl2_1 = 0.0, thl2_2 = 0.0;
            Real sqrtthl2_1 = 0.0, sqrtthl2_2 = 0.0;
            thl_parameters(wthl_sec(ic,k,0), sqrtw2, sqrtthl, std::max(thl_sec(ic,k,0), 0.0),
                           thl_first, w1_1, w1_2, skew_w, a,
                           thl1_1, thl1_2, thl2_1, thl2_2, sqrtthl2_1, sqrtthl2_2);

            Real qw1_1 = qw_first, qw1_2 = qw_first, qw2_1 = 0.0, qw2_2 = 0.0;
            Real sqrtqw2_1 = 0.0, sqrtqw2_2 = 0.0;
            qw_parameters(wqw_sec(ic,k,0), sqrtw2, skew_w, sqrtqt, std::max(qw_sec(ic,k,0), 0.0),
                          w1_2, w1_1, qw_first, a,
                          qw1_1, qw1_2, qw2_1, qw2_2, sqrtqw2_1, sqrtqw2_2);

            w1_1 = w1_1 * sqrtw2 + w_first;
            w1_2 = w1_2 * sqrtw2 + w_first;

            const Real r_qwthl_1 = inplume_correlation(sqrtqw2_1, sqrtthl2_1, a,
                                                       sqrtqw2_2, sqrtthl2_2,
                                                       qwthl_sec(ic,k,0), qw1_1, qw_first,
                                                       thl1_1, thl_first, qw1_2, thl1_2);

            Real Tl1_1 = std::max(k_shoc_tl_min, compute_temperature(thl1_1, p_mid(ic,k,0)));
            Real Tl1_2 = std::max(k_shoc_tl_min, compute_temperature(thl1_2, p_mid(ic,k,0)));
            Real qs1 = 0.0, beta1 = 0.0, qs2 = 0.0, beta2 = 0.0;
            compute_qs_beta(Tl1_1, p_mid(ic,k,0), qs1, beta1);
            compute_qs_beta(Tl1_2, p_mid(ic,k,0), qs2, beta2);

            Real s1 = 0.0, std_s1 = 0.0, qn1 = 0.0, c1 = 0.0;
            Real s2 = 0.0, std_s2 = 0.0, qn2 = 0.0, c2 = 0.0;
            compute_s_terms(qw1_1, qs1, beta1, p_mid(ic,k,0), thl2_1, qw2_1,
                            sqrtthl2_1, sqrtqw2_1, r_qwthl_1, s1, std_s1, qn1, c1);

            if (qw1_1 == qw1_2 && thl2_1 == thl2_2 && qs1 == qs2) {
                s2 = s1;
                std_s2 = std_s1;
                qn2 = qn1;
                c2 = c1;
            } else {
                compute_s_terms(qw1_2, qs2, beta2, p_mid(ic,k,0), thl2_2, qw2_2,
                                sqrtthl2_2, sqrtqw2_2, r_qwthl_1, s2, std_s2, qn2, c2);
            }

            const Real ql1 = std::min(qn1, qw1_1);
            const Real ql2 = std::min(qn2, qw1_2);
            const Real old_ql = shoc_ql(ic,k,0);
            const Real cldfrac = std::min(1.0, a * c1 + (1.0 - a) * c2);
            const Real ql = std::max(0.0, a * ql1 + (1.0 - a) * ql2);
            const Real ql2_var = std::max(0.0, a * (s1 * ql1 + c1 * std_s1 * std_s1)
                                               + (1.0 - a) * (s2 * ql2 + c2 * std_s2 * std_s2)
                                               - ql * ql);
            const Real wqls = a * ((w1_1 - w_first) * ql1) + (1.0 - a) * ((w1_2 - w_first) * ql2);

            shoc_cldfrac(ic,k,0) = clamp01(cldfrac);
            shoc_ql(ic,k,0) = ql;
            shoc_ql2(ic,k,0) = ql2_var;
            wqls_sec(ic,k,0) = wqls;
            wthv_sec(ic,k,0) = compute_buoyancy_flux(wthl_sec(ic,k,0), wqw_sec(ic,k,0),
                                                     p_mid(ic,k,0), wqls);

            if (opts.extra_shoc_diags) {
                const Real ql_change = ql - old_ql;
                shoc_cond(ic,k,0) = std::max(0.0, ql_change / std::max(dt, 1.0e-12));
                shoc_evap(ic,k,0) = std::max(0.0, -ql_change / std::max(dt, 1.0e-12));
            } else {
                shoc_cond(ic,k,0) = 0.0;
                shoc_evap(ic,k,0) = 0.0;
            }
        }
    }
}
