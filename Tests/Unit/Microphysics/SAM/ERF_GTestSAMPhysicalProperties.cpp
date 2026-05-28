#include <array>

#include <gtest/gtest.h>

#include "ERF_GTestSAMCommon.H"

using namespace sam_test;

namespace {

void expect_nonnegative_species (const SAMCellState& state)
{
    EXPECT_GE(state.qv, -exact_zero_or_near_zero_tol());
    EXPECT_GE(state.qcl, -exact_zero_or_near_zero_tol());
    EXPECT_GE(state.qci, -exact_zero_or_near_zero_tol());
    EXPECT_GE(state.qpr, -exact_zero_or_near_zero_tol());
    EXPECT_GE(state.qps, -exact_zero_or_near_zero_tol());
    EXPECT_GE(state.qpg, -exact_zero_or_near_zero_tol());
}

amrex::Real mixed_phase_tabs ()
{
    const amrex::Real lower = std::max(tprmin, tgrmin) + amrex::Real(1.0e-3);
    const amrex::Real upper = std::min(tprmax, tgrmax) - amrex::Real(1.0e-3);
    EXPECT_LT(lower, upper);
    return amrex::Real(0.5) * (lower + upper);
}

amrex::Real mixed_qsat_for_state (const int sam_mode,
                                  const amrex::Real tabs,
                                  const amrex::Real pres_mbar)
{
    amrex::Real qsatw;
    amrex::Real qsati;
    erf_qsatw(tabs, pres_mbar, qsatw);
    erf_qsati(tabs, pres_mbar, qsati);
    const amrex::Real omn = sam_cloud_liquid_fraction(sam_mode, tabs, a_bg, tbgmin * a_bg);
    return sam_mixed_qsat(omn, qsatw, qsati);
}

SAMCellState make_cell_state (const amrex::Real tabs,
                              const amrex::Real pres_mbar,
                              const amrex::Real qv,
                              const amrex::Real qcl,
                              const amrex::Real qci,
                              const amrex::Real qpr,
                              const amrex::Real qps,
                              const amrex::Real qpg)
{
    SAMCellState state{};
    state.tabs = tabs;
    state.pres_mbar = pres_mbar;
    state.qv = qv;
    state.qcl = qcl;
    state.qci = qci;
    state.qpr = qpr;
    state.qps = qps;
    state.qpg = qpg;
    state.qn = qcl + qci;
    state.qt = qv + state.qn;
    state.qp = qpr + qps + qpg;
    state.theta = sam_theta_from_stored_mbar_converted_to_pa(tabs, pres_mbar, kRdOcp);
    state.rho = getRhogivenTandPress(tabs, sam_mbar_to_pa(pres_mbar), qv);
    return state;
}

SAMPrecipConfig make_precip_config (const int sam_mode,
                                    const bool enable_precip = true,
                                    const amrex::Real dtn = amrex::Real(1.0))
{
    return {
        sam_mode,
        enable_precip,
        dtn,
        kRdOcp,
        kFacCond,
        kFacFus,
        kFacSub,
        std::numeric_limits<amrex::Real>::epsilon(),
        (three + b_rain) / amrex::Real(4.0),
        (three + b_snow) / amrex::Real(4.0),
        (three + b_grau) / amrex::Real(4.0),
        (amrex::Real(5.0) + b_rain) / amrex::Real(8.0),
        (amrex::Real(5.0) + b_snow) / amrex::Real(8.0),
        (amrex::Real(5.0) + b_grau) / amrex::Real(8.0)};
}

SAMCoefficientRow make_coeffs (const amrex::Real scale = amrex::Real(1.0))
{
    return {
        amrex::Real(2.0) * scale,
        amrex::Real(2.5) * scale,
        amrex::Real(2.2) * scale,
        amrex::Real(1.8) * scale,
        amrex::Real(1.1) * scale,
        amrex::Real(1.4) * scale,
        amrex::Real(2.4) * scale,
        amrex::Real(2.1) * scale,
        amrex::Real(1.2) * scale,
        amrex::Real(1.6) * scale,
        amrex::Real(1.3) * scale,
        amrex::Real(1.7) * scale};
}

} // namespace

// Motivation:
// Local SAM precipitation sources should exercise all autoconversion,
// accretion, and evaporation pathways on a composed mixed-phase cell without
// creating negative species or violating total-water conservation.
TEST(SAMPhysicalProperties, PrecipSources_AllSpeciesNonzeroAllSourceTermsActive)
{
    const amrex::Real tabs = mixed_phase_tabs();
    const amrex::Real pres_mbar = amrex::Real(900.0);
    const amrex::Real qsat = mixed_qsat_for_state(kSAMWithIceMode, tabs, pres_mbar);
    const SAMCellState before = make_cell_state(
        tabs, pres_mbar,
        amrex::Real(0.5) * qsat,
        qcw0 + amrex::Real(8.0e-4),
        qci0 + amrex::Real(7.0e-4),
        amrex::Real(4.0e-4),
        amrex::Real(5.0e-4),
        amrex::Real(6.0e-4));

    SAMPrecipCellDiagnostics diagnostics;
    const SAMCellState after = sam_precip_cell_update(
        before, make_coeffs(), make_precip_config(kSAMWithIceMode), &diagnostics);

    EXPECT_GT(diagnostics.autoconversion.dqca, amrex::Real(0.0));
    EXPECT_GT(diagnostics.autoconversion.dqia, amrex::Real(0.0));
    EXPECT_GT(diagnostics.accretion.dprc, amrex::Real(0.0));
    EXPECT_GT(diagnostics.accretion.dpsc, amrex::Real(0.0));
    EXPECT_GT(diagnostics.accretion.dpgc, amrex::Real(0.0));
    EXPECT_GT(diagnostics.accretion.dpsi, amrex::Real(0.0));
    EXPECT_GT(diagnostics.accretion.dpgi, amrex::Real(0.0));
    EXPECT_GT(diagnostics.evaporation.dqpr, amrex::Real(0.0));
    EXPECT_GT(diagnostics.evaporation.dqps, amrex::Real(0.0));
    EXPECT_GT(diagnostics.evaporation.dqpg, amrex::Real(0.0));

    expect_nonnegative_species(after);
    EXPECT_NEAR(sam_total_water(after),
                sam_total_water(before),
                formula_abs_tol(sam_total_water(before)));
}

// Motivation:
// Local SAM precipitation sources exchange water among qv, qcl, qci, qpr,
// qps, and qpg only. They should conserve total water across representative
// warm, cold, mixed, limiter-active, no-precip, and no-ice cases.
TEST(SAMPhysicalProperties, PrecipSources_ConserveTotalWaterAcrossRepresentativeCases)
{
    const amrex::Real mixed_tabs = mixed_phase_tabs();
    const amrex::Real warm_tabs = tbgmax + amrex::Real(2.0);
    const amrex::Real cold_tabs = tgrmin + amrex::Real(0.25) * (tgrmax - tgrmin);
    const amrex::Real pres_mbar = amrex::Real(900.0);

    struct ConservationCase {
        const char* label;
        SAMCellState state;
        SAMCoefficientRow coeffs;
        SAMPrecipConfig config;
    };

    const std::array<ConservationCase, 7> cases = {{
        {"warm-rain-only",
         make_cell_state(warm_tabs, pres_mbar,
                         amrex::Real(0.8) * mixed_qsat_for_state(kSAMNoIceMode, warm_tabs, pres_mbar),
                         qcw0 + amrex::Real(7.0e-4),
                         amrex::Real(0.0),
                         amrex::Real(4.0e-4),
                         amrex::Real(0.0),
                         amrex::Real(0.0)),
         make_coeffs(),
         make_precip_config(kSAMNoIceMode)},
        {"cold-snow-graupel-only",
         make_cell_state(cold_tabs, pres_mbar,
                         amrex::Real(0.7) * mixed_qsat_for_state(kSAMWithIceMode, cold_tabs, pres_mbar),
                         amrex::Real(0.0),
                         qci0 + amrex::Real(6.0e-4),
                         amrex::Real(0.0),
                         amrex::Real(4.0e-4),
                         amrex::Real(5.0e-4)),
         make_coeffs(),
         make_precip_config(kSAMWithIceMode)},
        {"mixed-all-species",
         make_cell_state(mixed_tabs, pres_mbar,
                         amrex::Real(0.6) * mixed_qsat_for_state(kSAMWithIceMode, mixed_tabs, pres_mbar),
                         qcw0 + amrex::Real(8.0e-4),
                         qci0 + amrex::Real(7.0e-4),
                         amrex::Real(4.0e-4),
                         amrex::Real(5.0e-4),
                         amrex::Real(6.0e-4)),
         make_coeffs(),
         make_precip_config(kSAMWithIceMode)},
        {"cloud-sink-limiter",
         make_cell_state(mixed_tabs, pres_mbar,
                         amrex::Real(0.7) * mixed_qsat_for_state(kSAMWithIceMode, mixed_tabs, pres_mbar),
                         amrex::Real(3.0e-5),
                         amrex::Real(2.0e-5),
                         amrex::Real(4.0e-4),
                         amrex::Real(5.0e-4),
                         amrex::Real(6.0e-4)),
         make_coeffs(amrex::Real(1.0e3)),
         make_precip_config(kSAMWithIceMode, true, amrex::Real(10.0))},
        {"evaporation-limiter",
         make_cell_state(mixed_tabs, pres_mbar,
                         amrex::Real(1.0e-6),
                         qcw0 + amrex::Real(2.0e-4),
                         qci0 + amrex::Real(2.0e-4),
                         amrex::Real(2.0e-6),
                         amrex::Real(3.0e-6),
                         amrex::Real(4.0e-6)),
         make_coeffs(amrex::Real(5.0e3)),
         make_precip_config(kSAMWithIceMode, true, amrex::Real(5.0))},
        {"no-precip-mode",
         make_cell_state(mixed_tabs, pres_mbar,
                         amrex::Real(0.5) * mixed_qsat_for_state(kSAMWithIceMode, mixed_tabs, pres_mbar),
                         qcw0 + amrex::Real(4.0e-4),
                         qci0 + amrex::Real(3.0e-4),
                         amrex::Real(2.0e-4),
                         amrex::Real(2.0e-4),
                         amrex::Real(2.0e-4)),
         make_coeffs(),
         make_precip_config(kSAMWithIceMode, false)},
        {"no-ice-mode",
         make_cell_state(warm_tabs, pres_mbar,
                         amrex::Real(0.75) * mixed_qsat_for_state(kSAMNoIceMode, warm_tabs, pres_mbar),
                         qcw0 + amrex::Real(5.0e-4),
                         amrex::Real(0.0),
                         amrex::Real(3.0e-4),
                         amrex::Real(0.0),
                         amrex::Real(0.0)),
         make_coeffs(),
         make_precip_config(kSAMNoIceMode)}
    }};

    for (const ConservationCase& test_case : cases) {
        SCOPED_TRACE(test_case.label);
        const amrex::Real before_total = sam_total_water(test_case.state);
        const SAMCellState after = sam_precip_cell_update(test_case.state,
                                                          test_case.coeffs,
                                                          test_case.config);
        expect_nonnegative_species(after);
        EXPECT_NEAR(sam_total_water(after),
                    before_total,
                    formula_abs_tol(before_total));
    }
}

// Motivation:
// The composed SAM precipitation source update should conserve the latent proxy
// T + fac_cond*qv - fac_fus*(qci + qps + qpg). A mixed-phase case with cloud-
// liquid accretion onto snow and graupel directly exposes the candidate missing
// fusion-heating term if that contract is violated.
TEST(SAMPhysicalProperties, PrecipSources_ConserveLatentProxyOrExposeRimingHeatingBug)
{
    const amrex::Real tabs = mixed_phase_tabs();
    const amrex::Real pres_mbar = amrex::Real(900.0);
    const amrex::Real qsat = mixed_qsat_for_state(kSAMWithIceMode, tabs, pres_mbar);
    const SAMCellState before = make_cell_state(
        tabs, pres_mbar,
        qsat,
        qcw0 + amrex::Real(8.0e-4),
        qci0 + amrex::Real(2.0e-4),
        amrex::Real(2.0e-4),
        amrex::Real(6.0e-4),
        amrex::Real(7.0e-4));

    SAMPrecipCellDiagnostics diagnostics;
    const SAMCellState after = sam_precip_cell_update(
        before, make_coeffs(), make_precip_config(kSAMWithIceMode), &diagnostics);

    ASSERT_GT(diagnostics.limited_sources.dpsc, amrex::Real(0.0));
    ASSERT_GT(diagnostics.limited_sources.dpgc, amrex::Real(0.0));

    const amrex::Real before_proxy = sam_latent_proxy(before, kFacCond, kFacFus);
    const amrex::Real after_proxy = sam_latent_proxy(after, kFacCond, kFacFus);
    const amrex::Real expected_missing =
        -kFacFus * (diagnostics.limited_sources.dpsc + diagnostics.limited_sources.dpgc);

    EXPECT_NEAR(after_proxy,
                before_proxy,
                formula_abs_tol(before_proxy))
        << "expected_missing_if_unheated=" << expected_missing
        << " dpsc=" << diagnostics.limited_sources.dpsc
        << " dpgc=" << diagnostics.limited_sources.dpgc;
}
