#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "ERF_GTestSAMCommon.H"

using namespace sam_test;

namespace {

void expect_near_roundoff (const amrex::Real actual,
                           const amrex::Real expected)
{
    EXPECT_NEAR(actual, expected, roundoff_tol(expected));
}

void expect_near_pow_sqrt (const amrex::Real actual,
                           const amrex::Real expected)
{
    EXPECT_NEAR(actual, expected, pow_sqrt_tol(expected));
}

} // namespace

// Motivation:
// This test protects the SAM pressure-unit contract: SAM stores pressure in
// mbar, while EOS helpers expect Pa.
TEST(SAMScalar, PressureRoundtrip)
{
    const std::array<amrex::Real, 4> pressures_mbar = {
        amrex::Real(50.0), amrex::Real(500.0), amrex::Real(900.0), amrex::Real(1013.25)};

    for (const amrex::Real pres_mbar : pressures_mbar) {
        SCOPED_TRACE("pres_mbar=" + std::to_string(static_cast<double>(pres_mbar)));
        const amrex::Real pres_pa = sam_mbar_to_pa(pres_mbar);
        expect_near_roundoff(sam_pa_to_mbar(pres_pa), pres_mbar);
    }
}

// Motivation:
// The mbar-converting theta helper must match direct EOS usage with 100*p_mbar.
TEST(SAMScalar, ThetaPressureUnitHelpers)
{
    const amrex::Real tabs = amrex::Real(289.5);
    const amrex::Real pres_mbar = amrex::Real(875.0);
    const amrex::Real pres_pa = sam_mbar_to_pa(pres_mbar);
    const amrex::Real expected_theta = getThgivenTandP(tabs, pres_pa, kRdOcp);

    expect_near_roundoff(
        sam_theta_from_stored_mbar_converted_to_pa(tabs, pres_mbar, kRdOcp),
        expected_theta);
}

// Motivation:
// SAM::Cloud stores pressure in mbar on this path, so its transient theta
// update must use the mbar-converting helper rather than treating the stored
// value as Pa.
TEST(SAMScalar, CloudThetaUsesMbarConversion)
{
    const amrex::Real tabs = amrex::Real(268.0);
    const amrex::Real pres_mbar = amrex::Real(900.0);
    const amrex::Real expected_theta =
        getThgivenTandP(tabs, sam_mbar_to_pa(pres_mbar), kRdOcp);
    const amrex::Real wrong_theta =
        getThgivenTandP(tabs, pres_mbar, kRdOcp);
    const amrex::Real converted_theta =
        sam_theta_from_stored_mbar_converted_to_pa(tabs, pres_mbar, kRdOcp);

    expect_near_roundoff(converted_theta, expected_theta);
    EXPECT_GT(std::abs(wrong_theta - expected_theta), amrex::Real(1.0));
}

// Motivation:
// Conserved-to-primitive conversion must preserve valid moist species, clip
// negative moisture, and store pressure in mbar.
TEST(SAMScalar, PrimitiveClipsNegativeMoisture)
{
    const amrex::Real rho = amrex::Real(1.15);
    const amrex::Real theta = amrex::Real(300.0);
    const amrex::Real rho_theta = rho * theta;

    const SAMPrimitiveCell clipped = sam_cons_to_primitive(
        rho, rho_theta,
        -rho * amrex::Real(2.0e-3),
        -rho * amrex::Real(3.0e-4),
        rho * amrex::Real(5.0e-4),
        -rho * amrex::Real(1.0e-4),
        rho * amrex::Real(2.0e-4),
        -rho * amrex::Real(3.0e-4));

    EXPECT_EQ(clipped.qv, amrex::Real(0.0));
    EXPECT_EQ(clipped.qcl, amrex::Real(0.0));
    EXPECT_EQ(clipped.qpr, amrex::Real(0.0));
    EXPECT_EQ(clipped.qpg, amrex::Real(0.0));
    expect_near_roundoff(clipped.qn, clipped.qcl + clipped.qci);
    expect_near_roundoff(clipped.qt, clipped.qv + clipped.qn);
    expect_near_roundoff(clipped.qp, clipped.qpr + clipped.qps + clipped.qpg);

    const amrex::Real expected_pres_mbar = sam_pa_to_mbar(getPgivenRTh(rho_theta, clipped.qv));
    expect_near_roundoff(clipped.pres_mbar, expected_pres_mbar);

    SAMPrimitiveCell primitive_to_write{};
    primitive_to_write.rho = rho;
    primitive_to_write.theta = theta;
    primitive_to_write.qv = -amrex::Real(1.0e-3);
    primitive_to_write.qcl = amrex::Real(2.5e-4);
    primitive_to_write.qci = -amrex::Real(5.0e-5);
    primitive_to_write.qpr = -amrex::Real(4.0e-4);
    primitive_to_write.qps = amrex::Real(3.0e-4);
    primitive_to_write.qpg = -amrex::Real(2.0e-4);

    amrex::FArrayBox state_fab(single_cell_box(), RhoQ6_comp + 1);
    auto state = single_cell_state_array(state_fab);
    sam_primitive_to_cons(primitive_to_write, state, 0, 0, 0);

    expect_near_roundoff(state(0,0,0,RhoTheta_comp), rho_theta);
    EXPECT_EQ(state(0,0,0,RhoQ1_comp), amrex::Real(0.0));
    expect_near_roundoff(state(0,0,0,RhoQ2_comp), rho * primitive_to_write.qcl);
    EXPECT_EQ(state(0,0,0,RhoQ3_comp), amrex::Real(0.0));
    EXPECT_EQ(state(0,0,0,RhoQ4_comp), amrex::Real(0.0));
    expect_near_roundoff(state(0,0,0,RhoQ5_comp), rho * primitive_to_write.qps);
    EXPECT_EQ(state(0,0,0,RhoQ6_comp), amrex::Real(0.0));
}

// Motivation:
// Positive conserved moisture values should roundtrip through the primitive
// helper surface within roundoff.
TEST(SAMScalar, PrimitiveConservedMoistureRoundtrip)
{
    const amrex::Real rho = amrex::Real(0.95);
    const amrex::Real theta = amrex::Real(304.0);
    const amrex::Real rho_theta = rho * theta;
    const amrex::Real qv = amrex::Real(1.2e-2);
    const amrex::Real qcl = amrex::Real(6.0e-4);
    const amrex::Real qci = amrex::Real(2.5e-4);
    const amrex::Real qpr = amrex::Real(3.0e-4);
    const amrex::Real qps = amrex::Real(1.0e-4);
    const amrex::Real qpg = amrex::Real(2.0e-4);

    const SAMPrimitiveCell primitive = sam_cons_to_primitive(
        rho, rho_theta, rho * qv, rho * qcl, rho * qci, rho * qpr, rho * qps, rho * qpg);

    expect_near_roundoff(primitive.theta, theta);
    expect_near_roundoff(primitive.qv, qv);
    expect_near_roundoff(primitive.qcl, qcl);
    expect_near_roundoff(primitive.qci, qci);
    expect_near_roundoff(primitive.qn, qcl + qci);
    expect_near_roundoff(primitive.qt, qv + qcl + qci);
    expect_near_roundoff(primitive.qpr, qpr);
    expect_near_roundoff(primitive.qps, qps);
    expect_near_roundoff(primitive.qpg, qpg);
    expect_near_roundoff(primitive.qp, qpr + qps + qpg);

    amrex::FArrayBox state_fab(single_cell_box(), RhoQ6_comp + 1);
    auto state = single_cell_state_array(state_fab);
    sam_primitive_to_cons(primitive, state, 0, 0, 0);

    expect_near_roundoff(state(0,0,0,RhoTheta_comp), rho_theta);
    expect_near_roundoff(state(0,0,0,RhoQ1_comp), rho * qv);
    expect_near_roundoff(state(0,0,0,RhoQ2_comp), rho * qcl);
    expect_near_roundoff(state(0,0,0,RhoQ3_comp), rho * qci);
    expect_near_roundoff(state(0,0,0,RhoQ4_comp), rho * qpr);
    expect_near_roundoff(state(0,0,0,RhoQ5_comp), rho * qps);
    expect_near_roundoff(state(0,0,0,RhoQ6_comp), rho * qpg);
}

// Motivation:
// Cloud liquid fraction has exact warm/cold caps and a linear interior branch.
TEST(SAMScalar, CloudLiquidFractionBranches)
{
    const amrex::Real an = a_bg;
    const amrex::Real bn = tbgmin * a_bg;
    const auto samples = threshold_samples(tbgmin, tbgmax);

    for (const amrex::Real tabs : samples) {
        SCOPED_TRACE(branch_label("sam_cloud_liquid_fraction", tabs));
        const amrex::Real actual = sam_cloud_liquid_fraction(kSAMWithIceMode, tabs, an, bn);
        EXPECT_GE(actual, amrex::Real(0.0));
        EXPECT_LE(actual, amrex::Real(1.0));

        if (tabs <= tbgmin) {
            EXPECT_NEAR(actual, amrex::Real(0.0), exact_zero_tol());
        } else if (tabs >= tbgmax) {
            EXPECT_NEAR(actual, amrex::Real(1.0), exact_zero_tol());
        } else {
            expect_near_roundoff(actual, an * tabs - bn);
        }
    }

    EXPECT_NEAR(sam_cloud_liquid_fraction(kSAMNoIceMode, amrex::Real(240.0), an, bn),
                amrex::Real(1.0), exact_zero_tol());
}

// Motivation:
// Rain fraction must stay in [0,1], clip at the warm/cold caps, and become all
// rain in no-ice mode.
TEST(SAMScalar, PrecipRainFractionBranches)
{
    const auto samples = threshold_samples(tprmin, tprmax);

    for (const amrex::Real tabs : samples) {
        SCOPED_TRACE(branch_label("sam_precip_rain_fraction", tabs));
        const amrex::Real actual = sam_precip_rain_fraction(kSAMWithIceMode, tabs);
        EXPECT_GE(actual, amrex::Real(0.0));
        EXPECT_LE(actual, amrex::Real(1.0));

        const amrex::Real expected = std::max(amrex::Real(0.0),
                                              std::min(amrex::Real(1.0), (tabs - tprmin) * a_pr));
        expect_near_roundoff(actual, expected);
    }

    EXPECT_NEAR(sam_precip_rain_fraction(kSAMNoIceMode, amrex::Real(250.0)),
                amrex::Real(1.0), exact_zero_tol());
}

// Motivation:
// Graupel fraction must stay in [0,1], clip at the warm/cold caps, and vanish
// in no-ice mode.
TEST(SAMScalar, GraupelFractionBranches)
{
    const auto samples = threshold_samples(tgrmin, tgrmax);

    for (const amrex::Real tabs : samples) {
        SCOPED_TRACE(branch_label("sam_graupel_fraction", tabs));
        const amrex::Real actual = sam_graupel_fraction(kSAMWithIceMode, tabs);
        EXPECT_GE(actual, amrex::Real(0.0));
        EXPECT_LE(actual, amrex::Real(1.0));

        const amrex::Real expected = std::max(amrex::Real(0.0),
                                              std::min(amrex::Real(1.0), (tabs - tgrmin) * a_gr));
        expect_near_roundoff(actual, expected);
    }

    EXPECT_NEAR(sam_graupel_fraction(kSAMNoIceMode, amrex::Real(260.0)),
                amrex::Real(0.0), exact_zero_tol());
}

// Motivation:
// Cloud partition must conserve qcl + qci, stay nonnegative for valid inputs,
// and follow the documented warm, cold, mixed, and no-ice temperature-update
// contracts.
TEST(SAMScalar, CloudPartitionBranches)
{
    const amrex::Real an = a_bg;
    const amrex::Real bn = tbgmin * a_bg;
    const amrex::Real qcl = amrex::Real(8.0e-4);
    const amrex::Real qci = amrex::Real(2.0e-4);
    const amrex::Real qn = qcl + qci;

    const SAMCloudPhaseChange warm = sam_partition_cloud_phase(
        kSAMWithIceMode, tbgmax + amrex::Real(1.0), qn, qcl, qci, kFacCond, kFacFus, an, bn);
    EXPECT_NEAR(warm.qcl, qn, roundoff_tol(qn));
    EXPECT_NEAR(warm.qci, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(warm.tabs, tbgmax + amrex::Real(1.0) - kFacFus * qci,
                newton_temperature_tol(tbgmax + amrex::Real(1.0)));

    const SAMCloudPhaseChange cold = sam_partition_cloud_phase(
        kSAMWithIceMode, tbgmin - amrex::Real(1.0), qn, qcl, qci, kFacCond, kFacFus, an, bn);
    EXPECT_NEAR(cold.qcl, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(cold.qci, qn, roundoff_tol(qn));
    EXPECT_NEAR(cold.tabs, tbgmin - amrex::Real(1.0) + kFacFus * qcl,
                newton_temperature_tol(tbgmin - amrex::Real(1.0)));

    const amrex::Real tabs_mixed = amrex::Real(0.5) * (tbgmin + tbgmax);
    const amrex::Real omn = sam_cloud_liquid_fraction(kSAMWithIceMode, tabs_mixed, an, bn);
    const SAMCloudPhaseChange mixed = sam_partition_cloud_phase(
        kSAMWithIceMode, tabs_mixed, qn, qcl, qci, kFacCond, kFacFus, an, bn);
    EXPECT_NEAR(mixed.qcl, qn * omn, roundoff_tol(qn * omn));
    EXPECT_NEAR(mixed.qci, qn * (one - omn), roundoff_tol(qn * (one - omn)));
    EXPECT_NEAR(mixed.tabs, tabs_mixed + kFacFus * (qcl - qn * omn),
                newton_temperature_tol(tabs_mixed));

    const SAMCloudPhaseChange no_ice = sam_partition_cloud_phase(
        kSAMNoIceMode, amrex::Real(250.0), qn, qcl, qci, kFacCond, kFacFus, an, bn);
    EXPECT_NEAR(no_ice.qcl, qn, roundoff_tol(qn));
    EXPECT_NEAR(no_ice.qci, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(no_ice.tabs, amrex::Real(250.0) + kFacCond * (qcl - qn),
                newton_temperature_tol(amrex::Real(250.0)));

    for (const SAMCloudPhaseChange& branch : {warm, cold, mixed, no_ice}) {
        EXPECT_GE(branch.qcl, -exact_zero_tol());
        EXPECT_GE(branch.qci, -exact_zero_tol());
        EXPECT_NEAR(branch.qcl + branch.qci, qn, roundoff_tol(qn));
    }
}

// Motivation:
// The mixed saturation derivative should match the symbolic derivative of
// qsat = omn*qsatw + (1-omn)*qsati.
TEST(SAMScalar, MixedQsatDerivativeMatchesSymbolicDerivative)
{
    const amrex::Real omn = amrex::Real(0.35);
    const amrex::Real domn = amrex::Real(1.5e-2);
    const amrex::Real qsatw = amrex::Real(9.0e-3);
    const amrex::Real qsati = amrex::Real(6.5e-3);
    const amrex::Real dqsatw = amrex::Real(4.0e-4);
    const amrex::Real dqsati = amrex::Real(2.5e-4);

    const amrex::Real expected_qsat = omn * qsatw + (one - omn) * qsati;
    const amrex::Real expected_dqsat =
        omn * dqsatw + (one - omn) * dqsati + domn * (qsatw - qsati);

    expect_near_roundoff(sam_mixed_qsat(omn, qsatw, qsati), expected_qsat);
    expect_near_roundoff(
        sam_mixed_dqsat_dT(omn, domn, qsatw, qsati, dqsatw, dqsati),
        expected_dqsat);
}

// Motivation:
// The Newton residual derivative should match the symbolic derivative of
// F(T) = -T + Told + L*(qv - qsat).
TEST(SAMScalar, NewtonResidualDerivativeMatchesSymbolicDerivative)
{
    const amrex::Real tabs_new = amrex::Real(268.0);
    const amrex::Real tabs_old = amrex::Real(270.0);
    const amrex::Real lstar = amrex::Real(2600.0);
    const amrex::Real dlstar = amrex::Real(-2.0);
    const amrex::Real qv = amrex::Real(5.0e-3);
    const amrex::Real qsat = amrex::Real(4.6e-3);
    const amrex::Real dqsat = amrex::Real(1.2e-4);

    const amrex::Real expected_residual = -tabs_new + tabs_old + lstar * (qv - qsat);
    const amrex::Real expected_derivative = -one + dlstar * (qv - qsat) - lstar * dqsat;

    expect_near_roundoff(
        sam_newton_residual(tabs_new, tabs_old, lstar, qv, qsat),
        expected_residual);
    expect_near_roundoff(
        sam_newton_residual_derivative(lstar, dlstar, qv, qsat, dqsat),
        expected_derivative);
}

// Motivation:
// SAM autoconversion is zero at and below threshold, then linear above it.
TEST(SAMScalar, AutoconversionThresholds)
{
    const amrex::Real dtn = amrex::Real(2.0);
    const amrex::Real coefice = amrex::Real(1.7);

    const SAMPrecipSources below = sam_autoconversion_rates(dtn, qcw0, qci0, coefice);
    EXPECT_NEAR(below.dqca, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(below.dqia, amrex::Real(0.0), exact_zero_tol());

    const amrex::Real qcc = qcw0 + amrex::Real(2.0e-4);
    const amrex::Real qii = qci0 + amrex::Real(3.0e-5);
    const SAMPrecipSources above = sam_autoconversion_rates(dtn, qcc, qii, coefice);
    expect_near_roundoff(above.dqca, dtn * alphaelq * (qcc - qcw0));
    expect_near_roundoff(above.dqia, dtn * betaelq * coefice * (qii - qci0));
}

// Motivation:
// Accretion activation depends on omp/omg thresholds rather than sign-only
// behavior.
TEST(SAMScalar, AccretionActivationThresholds)
{
    const amrex::Real dtn = amrex::Real(1.5);
    const amrex::Real qcc = amrex::Real(8.0e-4);
    const amrex::Real qii = amrex::Real(3.0e-4);
    const amrex::Real qpr = amrex::Real(2.0e-3);
    const amrex::Real qps = amrex::Real(2.5e-3);
    const amrex::Real qpg = amrex::Real(3.0e-3);

    const SAMPrecipSources rain_off = sam_accretion_rates(
        dtn, qcc, qii, qpr, qps, qpg,
        one, one, one,
        kPhaseActivationLow, amrex::Real(0.5),
        amrex::Real(2.0), amrex::Real(3.0), amrex::Real(4.0), amrex::Real(5.0), amrex::Real(6.0));
    EXPECT_NEAR(rain_off.dprc, amrex::Real(0.0), exact_zero_tol());
    EXPECT_GT(rain_off.dpsc, amrex::Real(0.0));
    EXPECT_GT(rain_off.dpsi, amrex::Real(0.0));
    EXPECT_GT(rain_off.dpgc, amrex::Real(0.0));
    EXPECT_GT(rain_off.dpgi, amrex::Real(0.0));

    const SAMPrecipSources rain_on = sam_accretion_rates(
        dtn, qcc, qii, qpr, qps, qpg,
        one, one, one,
        kPhaseActivationLow + amrex::Real(1.0e-6), amrex::Real(0.5),
        amrex::Real(2.0), amrex::Real(3.0), amrex::Real(4.0), amrex::Real(5.0), amrex::Real(6.0));
    EXPECT_GT(rain_on.dprc, amrex::Real(0.0));

    const SAMPrecipSources snow_graupel_off = sam_accretion_rates(
        dtn, qcc, qii, qpr, qps, qpg,
        one, one, one,
        kPhaseActivationHigh, kPhaseActivationLow,
        amrex::Real(2.0), amrex::Real(3.0), amrex::Real(4.0), amrex::Real(5.0), amrex::Real(6.0));
    EXPECT_NEAR(snow_graupel_off.dpsc, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(snow_graupel_off.dpsi, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(snow_graupel_off.dpgc, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(snow_graupel_off.dpgi, amrex::Real(0.0), exact_zero_tol());
}

// Motivation:
// The cloud-sink limiter must not remove more cloud liquid or ice than is
// available in the donor cell.
TEST(SAMScalar, CloudSinkLimiterCapsSinks)
{
    SAMPrecipSources sources{};
    sources.dqca = amrex::Real(3.0e-4);
    sources.dprc = amrex::Real(4.0e-4);
    sources.dpsc = amrex::Real(5.0e-4);
    sources.dpgc = amrex::Real(6.0e-4);
    sources.dqia = amrex::Real(2.0e-4);
    sources.dpsi = amrex::Real(3.0e-4);
    sources.dpgi = amrex::Real(4.0e-4);

    const amrex::Real qcl = amrex::Real(8.0e-4);
    const amrex::Real qci = amrex::Real(5.0e-4);
    const SAMPrecipSources limited = sam_rescale_cloud_sinks(qcl, qci, exact_zero_tol(), sources);

    EXPECT_LE(limited.dqc, qcl + flux_conservation_tol(qcl));
    EXPECT_LE(limited.dqi, qci + flux_conservation_tol(qci));
    EXPECT_NEAR(limited.dqc, qcl, flux_conservation_tol(qcl));
    EXPECT_NEAR(limited.dqi, qci, flux_conservation_tol(qci));
}

// Motivation:
// Partitioning the autoconverted precip increments among rain, snow, and
// graupel must conserve the total precip source.
TEST(SAMScalar, PartitionedPrecipConservesSources)
{
    SAMPrecipSources sources{};
    sources.dqca = amrex::Real(2.0e-4);
    sources.dqia = amrex::Real(1.5e-4);
    sources.dprc = amrex::Real(3.0e-4);
    sources.dpsc = amrex::Real(4.0e-4);
    sources.dpgc = amrex::Real(5.0e-4);
    sources.dpsi = amrex::Real(6.0e-4);
    sources.dpgi = amrex::Real(7.0e-4);

    const SAMPrecipSources partitioned = sam_partition_autoconverted_precip(
        sources, amrex::Real(0.35), amrex::Real(0.4));

    const amrex::Real total_input =
        sources.dqca + sources.dqia + sources.dprc + sources.dpsc +
        sources.dpgc + sources.dpsi + sources.dpgi;
    const amrex::Real total_output = partitioned.dqpr + partitioned.dqps + partitioned.dqpg;

    EXPECT_NEAR(total_output, total_input, flux_conservation_tol(total_input));
}

// Motivation:
// Evaporation rates must be finite and nonnegative for positive precip, and the
// limiter must cap each precip species independently.
TEST(SAMScalar, PrecipEvaporationLimiterCapsSpecies)
{
    const amrex::Real powr2 = (amrex::Real(5.0) + b_rain) / amrex::Real(8.0);
    const amrex::Real pows2 = (amrex::Real(5.0) + b_snow) / amrex::Real(8.0);
    const amrex::Real powg2 = (amrex::Real(5.0) + b_grau) / amrex::Real(8.0);

    const SAMPrecipSources zero_rates = sam_precip_evaporation_rates(
        amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
        powr2, pows2, powg2, amrex::Real(1.0), amrex::Real(1.0),
        amrex::Real(1.0), amrex::Real(1.0), amrex::Real(1.0), amrex::Real(1.0));
    EXPECT_NEAR(zero_rates.dqpr, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(zero_rates.dqps, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(zero_rates.dqpg, amrex::Real(0.0), exact_zero_tol());

    SAMPrecipSources positive = sam_precip_evaporation_rates(
        amrex::Real(4.0e-4), amrex::Real(5.0e-4), amrex::Real(6.0e-4),
        powr2, pows2, powg2,
        amrex::Real(2.0), amrex::Real(3.0),
        amrex::Real(4.0), amrex::Real(5.0),
        amrex::Real(6.0), amrex::Real(7.0));
    EXPECT_TRUE(std::isfinite(positive.dqpr));
    EXPECT_TRUE(std::isfinite(positive.dqps));
    EXPECT_TRUE(std::isfinite(positive.dqpg));
    EXPECT_GE(positive.dqpr, amrex::Real(0.0));
    EXPECT_GE(positive.dqps, amrex::Real(0.0));
    EXPECT_GE(positive.dqpg, amrex::Real(0.0));

    positive.dqpr = amrex::Real(9.0e-4);
    positive.dqps = amrex::Real(1.0e-3);
    positive.dqpg = amrex::Real(1.1e-3);
    const SAMPrecipSources limited = sam_apply_precip_evaporation_limiter(
        amrex::Real(4.0e-4), amrex::Real(5.0e-4), amrex::Real(6.0e-4), positive);
    EXPECT_NEAR(limited.dqpr, amrex::Real(4.0e-4), roundoff_tol(amrex::Real(4.0e-4)));
    EXPECT_NEAR(limited.dqps, amrex::Real(5.0e-4), roundoff_tol(amrex::Real(5.0e-4)));
    EXPECT_NEAR(limited.dqpg, amrex::Real(6.0e-4), roundoff_tol(amrex::Real(6.0e-4)));
    EXPECT_NEAR(limited.dqp, limited.dqpr + limited.dqps + limited.dqpg,
                flux_conservation_tol(limited.dqp));
}

// Motivation:
// The coefficient-row helper should stay finite and nonnegative over a
// representative atmospheric range.
TEST(SAMScalar, CoefficientRowRepresentativeStatesFinite)
{
    const amrex::Real gamr1 = erf_gammafff(three + b_rain);
    const amrex::Real gamr2 = erf_gammafff((amrex::Real(5.0) + b_rain) / two);
    const amrex::Real gams1 = erf_gammafff(three + b_snow);
    const amrex::Real gams2 = erf_gammafff((amrex::Real(5.0) + b_snow) / two);
    const amrex::Real gamg1 = erf_gammafff(three + b_grau);
    const amrex::Real gamg2 = erf_gammafff((amrex::Real(5.0) + b_grau) / two);

    for (const amrex::Real rho : {amrex::Real(0.4), amrex::Real(0.8), amrex::Real(1.4)}) {
        for (const amrex::Real tabs : {amrex::Real(190.0), amrex::Real(230.0), tbgmax}) {
            SCOPED_TRACE("rho=" + std::to_string(static_cast<double>(rho)) +
                         " tabs=" + std::to_string(static_cast<double>(tabs)));
            const SAMCoefficientRow row = sam_compute_coefficient_row(
                rho, tabs, gamr1, gamr2, gams1, gams2, gamg1, gamg2);

            for (const amrex::Real coeff : {
                     row.accrrc, row.accrsi, row.accrsc, row.coefice,
                     row.evaps1, row.evaps2, row.accrgi, row.accrgc,
                     row.evapg1, row.evapg2, row.evapr1, row.evapr2}) {
                EXPECT_TRUE(std::isfinite(coeff));
                EXPECT_GE(coeff, amrex::Real(0.0));
            }
        }
    }
}

// Motivation:
// The sedimentation helpers have exact one-sided face rules, a documented ceil
// rule for substeps, a zero-flux threshold, and a closed-form flux-divergence
// tendency.
TEST(SAMScalar, SedimentationHelpersMatchDocumentedContracts)
{
    const amrex::Real clipped_velocity = sam_cloud_ice_terminal_velocity(-amrex::Real(1.0e-6));
    const amrex::Real expected_clipped =
        std::min(amrex::Real(0.4), amrex::Real(8.66) * std::pow(amrex::Real(1.0e-10), amrex::Real(0.24)));
    expect_near_pow_sqrt(clipped_velocity, expected_clipped);

    const amrex::Real capped_velocity = sam_cloud_ice_terminal_velocity(amrex::Real(1.0e3));
    EXPECT_NEAR(capped_velocity, amrex::Real(0.4), exact_zero_tol());

    const SAMFaceState bottom = sam_face_average_state(
        0, 0, 2,
        amrex::Real(0.8), amrex::Real(1.0),
        amrex::Real(250.0), amrex::Real(255.0),
        amrex::Real(1.0e-4), amrex::Real(2.0e-4),
        amrex::Real(3.0e-4), amrex::Real(4.0e-4));
    expect_near_roundoff(bottom.rho_avg, amrex::Real(1.0));
    expect_near_roundoff(bottom.tabs_avg, amrex::Real(255.0));

    const SAMFaceState interior = sam_face_average_state(
        1, 0, 2,
        amrex::Real(0.8), amrex::Real(1.0),
        amrex::Real(250.0), amrex::Real(255.0),
        amrex::Real(1.0e-4), amrex::Real(2.0e-4),
        amrex::Real(3.0e-4), amrex::Real(4.0e-4));
    expect_near_roundoff(interior.rho_avg, amrex::Real(0.9));
    expect_near_roundoff(interior.tabs_avg, amrex::Real(252.5));

    const SAMFaceState top = sam_face_average_state(
        3, 0, 2,
        amrex::Real(0.8), amrex::Real(1.0),
        amrex::Real(250.0), amrex::Real(255.0),
        amrex::Real(1.0e-4), amrex::Real(2.0e-4),
        amrex::Real(3.0e-4), amrex::Real(4.0e-4));
    expect_near_roundoff(top.rho_avg, amrex::Real(0.8));
    expect_near_roundoff(top.tabs_avg, amrex::Real(250.0));

    EXPECT_EQ(sam_substep_count_from_reduced_flux(amrex::Real(0.0), amrex::Real(5.0), amrex::Real(100.0)), 0);
    EXPECT_EQ(sam_substep_count_from_reduced_flux(amrex::Real(12.0), amrex::Real(5.0), amrex::Real(100.0)),
              static_cast<int>(std::ceil(amrex::Real(12.0) * (amrex::Real(5.0) / amrex::Real(100.0)) / myhalf)));

    amrex::Box col_box(amrex::IntVect(0, 0, 0), amrex::IntVect(0, 0, 2));
    amrex::FArrayBox rho_fab(col_box, 1);
    amrex::FArrayBox tabs_fab(col_box, 1);
    amrex::FArrayBox qp_fab(col_box, 1);
    auto rho = rho_fab.array();
    auto tabs = tabs_fab.array();
    auto qp = qp_fab.array();

    rho(0,0,0) = amrex::Real(1.0);
    rho(0,0,1) = amrex::Real(1.2);
    rho(0,0,2) = amrex::Real(1.4);
    tabs(0,0,0) = amrex::Real(265.0);
    tabs(0,0,1) = amrex::Real(270.0);
    tabs(0,0,2) = amrex::Real(275.0);
    qp(0,0,0) = amrex::Real(2.0e-7);
    qp(0,0,1) = amrex::Real(4.0e-7);
    qp(0,0,2) = amrex::Real(6.0e-7);

    const SAMPrecipFaceState precip_bottom = sam_precip_face_state(
        kSAMWithIceMode, rho_fab.const_array(), tabs_fab.const_array(), qp_fab.const_array(), 0, 0, 0, 0, 2);
    expect_near_roundoff(precip_bottom.rho_avg, rho(0,0,0));
    expect_near_roundoff(precip_bottom.qp_avg, qp(0,0,0));

    const SAMPrecipFaceState precip_interior = sam_precip_face_state(
        kSAMWithIceMode, rho_fab.const_array(), tabs_fab.const_array(), qp_fab.const_array(), 0, 0, 1, 0, 2);
    expect_near_roundoff(precip_interior.rho_avg, amrex::Real(1.1));
    expect_near_roundoff(precip_interior.qp_avg, amrex::Real(3.0e-7));
    expect_near_roundoff(precip_interior.qrr, precip_interior.omp * precip_interior.qp_avg);

    const SAMPrecipFaceState tiny_qp = precip_interior;
    const SAMPrecipFaceState threshold_state{
        precip_interior.rho_avg,
        precip_interior.tabs_avg,
        kQpThresholdForBranchSampling,
        precip_interior.omp,
        precip_interior.omg,
        tiny_qp.qrr,
        tiny_qp.qss,
        tiny_qp.qgg};
    EXPECT_NEAR(sam_precip_flux_from_face_state(threshold_state,
                                                amrex::Real(1.0), amrex::Real(2.0), amrex::Real(3.0)),
                amrex::Real(0.0), exact_zero_tol());

    SAMPrecipFaceState flux_state = precip_interior;
    flux_state.qp_avg = amrex::Real(5.0e-5);
    flux_state.qrr = flux_state.omp * flux_state.qp_avg;
    flux_state.qss = (one - flux_state.omp) * (one - flux_state.omg) * flux_state.qp_avg;
    flux_state.qgg = (one - flux_state.omp) * flux_state.omg * flux_state.qp_avg;
    const amrex::Real precip_flux = sam_precip_flux_from_face_state(
        flux_state, amrex::Real(2.0), amrex::Real(3.0), amrex::Real(4.0));
    const amrex::Real expected_flux =
        flux_state.omp * amrex::Real(2.0) * std::pow(flux_state.rho_avg * flux_state.qrr, one + crain) +
        (one - flux_state.omp) *
            ((one - flux_state.omg) * amrex::Real(3.0) * std::pow(flux_state.rho_avg * flux_state.qss, one + csnow) +
             flux_state.omg * amrex::Real(4.0) * std::pow(flux_state.rho_avg * flux_state.qgg, one + cgrau));
    expect_near_pow_sqrt(precip_flux, expected_flux);

    const amrex::Real density_corrected = sam_precip_flux_density_corrected(
        precip_flux, amrex::Real(1.2), flux_state.rho_avg);
    expect_near_pow_sqrt(density_corrected,
                         precip_flux * std::sqrt(amrex::Real(1.2) / flux_state.rho_avg));

    EXPECT_NEAR(sam_sedimentation_tendency(amrex::Real(2.0), amrex::Real(2.0), amrex::Real(1.1), amrex::Real(0.9), amrex::Real(1.4)),
                amrex::Real(0.0), exact_zero_tol());
    expect_near_roundoff(
        sam_sedimentation_tendency(amrex::Real(3.0), amrex::Real(1.0), amrex::Real(1.25), amrex::Real(0.8), amrex::Real(1.5)),
        amrex::Real(0.8) * (one / amrex::Real(1.25)) * amrex::Real(2.0) * amrex::Real(1.5));
}

// Motivation:
// Surface accumulation should follow the same precip-flux activation branch as
// the bottom sedimentation flux. Below the flux threshold, no surface depth
// should accumulate.
TEST(SAMScalar, SurfaceAccumulationZeroBelowFluxThreshold)
{
    const amrex::Real rho0 = amrex::Real(1.29);
    const amrex::Real dt = amrex::Real(2.0);
    const amrex::Real vrain = amrex::Real(2.0);
    const amrex::Real vsnow = amrex::Real(3.0);
    const amrex::Real vgrau = amrex::Real(4.0);

    const std::array<SAMPrecipFaceState, 3> pure_phase_states = {{
        {amrex::Real(1.0), amrex::Real(280.0), kQpThresholdForBranchSampling,
         amrex::Real(1.0), amrex::Real(0.0),
         kQpThresholdForBranchSampling, amrex::Real(0.0), amrex::Real(0.0)},
        {amrex::Real(0.8), amrex::Real(260.0), kQpThresholdForBranchSampling,
         amrex::Real(0.0), amrex::Real(0.0),
         amrex::Real(0.0), kQpThresholdForBranchSampling, amrex::Real(0.0)},
        {amrex::Real(0.9), amrex::Real(255.0), kQpThresholdForBranchSampling,
         amrex::Real(0.0), amrex::Real(1.0),
         amrex::Real(0.0), amrex::Real(0.0), kQpThresholdForBranchSampling}
    }};

    for (const SAMPrecipFaceState& face_state : pure_phase_states) {
        const amrex::Real bottom_flux = sam_precip_flux_density_corrected(
            sam_precip_flux_from_face_state(face_state, vrain, vsnow, vgrau),
            rho0, face_state.rho_avg);
        EXPECT_NEAR(bottom_flux, amrex::Real(0.0), exact_zero_tol());

        const SAMSurfaceAccumulation accum = sam_surface_accumulation(
            face_state, rho0,
            vrain, vsnow, vgrau, dt);

        EXPECT_NEAR(accum.rain, amrex::Real(0.0), exact_zero_tol());
        EXPECT_NEAR(accum.snow, amrex::Real(0.0), exact_zero_tol());
        EXPECT_NEAR(accum.graupel, amrex::Real(0.0), exact_zero_tol());
    }
}

// Motivation:
// For pure rain, snow, and graupel, surface accumulation should equal the same
// bottom outgoing precip flux used by sedimentation, integrated over dt and
// converted to depth.
TEST(SAMScalar, SurfaceAccumulationMatchesPurePhaseBottomFlux)
{
    const amrex::Real rho0 = amrex::Real(1.29);
    const amrex::Real dt = amrex::Real(3.0);
    const amrex::Real vrain = amrex::Real(2.0);
    const amrex::Real vsnow = amrex::Real(3.0);
    const amrex::Real vgrau = amrex::Real(4.0);

    struct PurePhaseCase {
        const char* label;
        SAMPrecipFaceState face_state;
        amrex::Real density;
    };

    const std::array<PurePhaseCase, 3> cases = {{
        {"rain", {amrex::Real(1.0), amrex::Real(285.0), amrex::Real(6.0e-5),
                   amrex::Real(1.0), amrex::Real(0.0),
                   amrex::Real(6.0e-5), amrex::Real(0.0), amrex::Real(0.0)}, rhor},
        {"snow", {amrex::Real(0.85), amrex::Real(260.0), amrex::Real(7.0e-5),
                   amrex::Real(0.0), amrex::Real(0.0),
                   amrex::Real(0.0), amrex::Real(7.0e-5), amrex::Real(0.0)}, rhos},
        {"graupel", {amrex::Real(0.95), amrex::Real(245.0), amrex::Real(8.0e-5),
                      amrex::Real(0.0), amrex::Real(1.0),
                      amrex::Real(0.0), amrex::Real(0.0), amrex::Real(8.0e-5)}, rhog}
    }};

    for (const PurePhaseCase& test_case : cases) {
        SCOPED_TRACE(test_case.label);
        const amrex::Real bottom_flux = sam_precip_flux_density_corrected(
            sam_precip_flux_from_face_state(test_case.face_state, vrain, vsnow, vgrau),
            rho0, test_case.face_state.rho_avg);
        const amrex::Real expected_depth = bottom_flux * dt / test_case.density * amrex::Real(1000.0);

        const SAMSurfaceAccumulation accum = sam_surface_accumulation(
            test_case.face_state, rho0,
            vrain, vsnow, vgrau, dt);

        if (std::string(test_case.label) == "rain") {
            EXPECT_NEAR(accum.rain, expected_depth, pow_sqrt_tol(expected_depth));
            EXPECT_NEAR(accum.snow, amrex::Real(0.0), exact_zero_tol());
            EXPECT_NEAR(accum.graupel, amrex::Real(0.0), exact_zero_tol());
        } else if (std::string(test_case.label) == "snow") {
            EXPECT_NEAR(accum.snow, expected_depth, pow_sqrt_tol(expected_depth));
            EXPECT_NEAR(accum.rain, amrex::Real(0.0), exact_zero_tol());
            EXPECT_NEAR(accum.graupel, amrex::Real(0.0), exact_zero_tol());
        } else {
            EXPECT_NEAR(accum.graupel, expected_depth, pow_sqrt_tol(expected_depth));
            EXPECT_NEAR(accum.rain, amrex::Real(0.0), exact_zero_tol());
            EXPECT_NEAR(accum.snow, amrex::Real(0.0), exact_zero_tol());
        }
    }
}

// Motivation:
// Conservative component sedimentation uses the same bottom/interior/top face
// selection rules as the total-qp face helper, but reads qpr, qps, and qpg
// directly from the neighboring cell states.
TEST(SAMScalar, PrecipComponentFaceState_BottomTopInteriorBranches)
{
    const amrex::Box column_box(amrex::IntVect(0, 0, 0), amrex::IntVect(0, 0, 1));
    amrex::FArrayBox rho_fab(column_box, 1);
    amrex::FArrayBox tabs_fab(column_box, 1);
    amrex::FArrayBox qpr_fab(column_box, 1);
    amrex::FArrayBox qps_fab(column_box, 1);
    amrex::FArrayBox qpg_fab(column_box, 1);

    auto rho = rho_fab.array();
    auto tabs = tabs_fab.array();
    auto qpr = qpr_fab.array();
    auto qps = qps_fab.array();
    auto qpg = qpg_fab.array();

    rho(0,0,0) = amrex::Real(1.0);
    tabs(0,0,0) = amrex::Real(280.0);
    qpr(0,0,0) = amrex::Real(1.0e-4);
    qps(0,0,0) = amrex::Real(2.0e-4);
    qpg(0,0,0) = amrex::Real(3.0e-4);

    rho(0,0,1) = amrex::Real(1.2);
    tabs(0,0,1) = amrex::Real(260.0);
    qpr(0,0,1) = amrex::Real(4.0e-4);
    qps(0,0,1) = amrex::Real(5.0e-4);
    qpg(0,0,1) = amrex::Real(6.0e-4);

    const auto bottom = sam_precip_component_face_state(rho, tabs, qpr, qps, qpg, 0, 0, 0, 0, 1);
    expect_near_roundoff(bottom.rho_avg, rho(0,0,0));
    expect_near_roundoff(bottom.qpr_avg, qpr(0,0,0));
    expect_near_roundoff(bottom.qps_avg, qps(0,0,0));
    expect_near_roundoff(bottom.qpg_avg, qpg(0,0,0));

    const auto interior = sam_precip_component_face_state(rho, tabs, qpr, qps, qpg, 0, 0, 1, 0, 1);
    expect_near_roundoff(interior.rho_avg, myhalf * (rho(0,0,0) + rho(0,0,1)));
    expect_near_roundoff(interior.qpr_avg, myhalf * (qpr(0,0,0) + qpr(0,0,1)));
    expect_near_roundoff(interior.qps_avg, myhalf * (qps(0,0,0) + qps(0,0,1)));
    expect_near_roundoff(interior.qpg_avg, myhalf * (qpg(0,0,0) + qpg(0,0,1)));

    const auto top = sam_precip_component_face_state(rho, tabs, qpr, qps, qpg, 0, 0, 2, 0, 1);
    expect_near_roundoff(top.rho_avg, rho(0,0,1));
    expect_near_roundoff(top.qpr_avg, qpr(0,0,1));
    expect_near_roundoff(top.qps_avg, qps(0,0,1));
    expect_near_roundoff(top.qpg_avg, qpg(0,0,1));

    EXPECT_EQ(sam_precip_face_donor_k(0, 0, 1), 0);
    EXPECT_EQ(sam_precip_face_donor_k(1, 0, 1), 1);
    EXPECT_EQ(sam_precip_face_donor_k(2, 0, 1), 1);
}

// Motivation:
// Conservative component fluxes must vanish exactly when the corresponding
// component mass is zero.
TEST(SAMScalar, PrecipComponentFlux_ZeroWhenComponentIsZero)
{
    const SAMPrecipComponentFaceState face_state{
        amrex::Real(1.0), amrex::Real(270.0),
        amrex::Real(0.0), amrex::Real(4.0e-4), amrex::Real(5.0e-4)};
    const SAMPrecipFluxComponents fluxes = sam_precip_component_fluxes_from_face_state(
        face_state, amrex::Real(2.0), amrex::Real(3.0), amrex::Real(4.0));

    EXPECT_NEAR(fluxes.rain, amrex::Real(0.0), exact_zero_tol());
    EXPECT_GT(fluxes.snow, amrex::Real(0.0));
    EXPECT_GT(fluxes.graupel, amrex::Real(0.0));
}

// Motivation:
// For fixed rho, each conservative component flux must increase with its own
// component mass.
TEST(SAMScalar, PrecipComponentFlux_MonotoneInComponentMass)
{
    const amrex::Real vrain = amrex::Real(2.0);
    const amrex::Real vsnow = amrex::Real(3.0);
    const amrex::Real vgrau = amrex::Real(4.0);

    const SAMPrecipComponentFaceState base_state{
        amrex::Real(1.0), amrex::Real(270.0),
        amrex::Real(3.0e-4), amrex::Real(4.0e-4), amrex::Real(5.0e-4)};
    SAMPrecipComponentFaceState larger_rain = base_state;
    SAMPrecipComponentFaceState larger_snow = base_state;
    SAMPrecipComponentFaceState larger_graupel = base_state;
    larger_rain.qpr_avg *= amrex::Real(1.5);
    larger_snow.qps_avg *= amrex::Real(1.5);
    larger_graupel.qpg_avg *= amrex::Real(1.5);

    const SAMPrecipFluxComponents base_fluxes =
        sam_precip_component_fluxes_from_face_state(base_state, vrain, vsnow, vgrau);
    const SAMPrecipFluxComponents rain_fluxes =
        sam_precip_component_fluxes_from_face_state(larger_rain, vrain, vsnow, vgrau);
    const SAMPrecipFluxComponents snow_fluxes =
        sam_precip_component_fluxes_from_face_state(larger_snow, vrain, vsnow, vgrau);
    const SAMPrecipFluxComponents graupel_fluxes =
        sam_precip_component_fluxes_from_face_state(larger_graupel, vrain, vsnow, vgrau);

    EXPECT_GT(rain_fluxes.rain, base_fluxes.rain);
    EXPECT_GT(snow_fluxes.snow, base_fluxes.snow);
    EXPECT_GT(graupel_fluxes.graupel, base_fluxes.graupel);
}

// Motivation:
// The component donor limiter must never export more component mass through a
// face than the donor cell contains over one substep.
TEST(SAMScalar, PrecipComponentFluxLimiter_CapsAtAvailableDonorMass)
{
    const amrex::Real rho_donor = amrex::Real(1.0);
    const amrex::Real q_donor = amrex::Real(3.0e-4);
    const amrex::Real detJ_donor = amrex::Real(0.8);
    const amrex::Real coef = amrex::Real(0.25);
    const amrex::Real available_flux = rho_donor * q_donor * detJ_donor / coef;

    const amrex::Real inactive = sam_limit_precip_component_flux(
        amrex::Real(0.5) * available_flux, rho_donor, q_donor, detJ_donor, coef);
    const amrex::Real active = sam_limit_precip_component_flux(
        amrex::Real(2.0) * available_flux, rho_donor, q_donor, detJ_donor, coef);

    expect_near_roundoff(inactive, amrex::Real(0.5) * available_flux);
    expect_near_roundoff(active, available_flux);
    EXPECT_LE(active, available_flux + exact_zero_or_near_zero_tol());
}

// Motivation:
// Conservative surface accumulation must be computed from the same limited
// density-corrected component fluxes used by the state update.
TEST(SAMScalar, SurfaceAccumulation_UsesLimitedComponentFluxes)
{
    const SAMPrecipFluxComponents limited_fluxes{
        amrex::Real(0.12), amrex::Real(0.09), amrex::Real(0.06)};
    const amrex::Real dt = amrex::Real(2.5);

    const SAMSurfaceAccumulation accum =
        sam_surface_accumulation_from_component_fluxes(limited_fluxes, dt);

    expect_near_pow_sqrt(accum.rain, limited_fluxes.rain * dt / rhor * amrex::Real(1000.0));
    expect_near_pow_sqrt(accum.snow, limited_fluxes.snow * dt / rhos * amrex::Real(1000.0));
    expect_near_pow_sqrt(accum.graupel, limited_fluxes.graupel * dt / rhog * amrex::Real(1000.0));
}

// Motivation:
// Away from the qp activation threshold, the density-corrected pure-phase SAM
// precip flux has a closed-form derivative in both q and rho. This verifies
// the documented rain, snow, and graupel derivative contracts directly.
TEST(SAMScalar, PrecipFlux_PurePhaseDerivativeInPositiveDomain)
{
    const amrex::Real rho0 = amrex::Real(1.29);
    const amrex::Real vrain = amrex::Real(2.0);
    const amrex::Real vsnow = amrex::Real(3.0);
    const amrex::Real vgrau = amrex::Real(4.0);

    struct PurePhaseCase {
        const char* label;
        amrex::Real tabs;
        amrex::Real qp;
        amrex::Real rho;
        amrex::Real omp;
        amrex::Real omg;
        amrex::Real exponent;
    };

    const std::array<PurePhaseCase, 3> cases = {{
        {"rain", amrex::Real(285.0), amrex::Real(1.0e-3), amrex::Real(1.0), amrex::Real(1.0), amrex::Real(0.0), crain},
        {"snow", amrex::Real(260.0), amrex::Real(1.2e-3), amrex::Real(0.85), amrex::Real(0.0), amrex::Real(0.0), csnow},
        {"graupel", amrex::Real(245.0), amrex::Real(1.4e-3), amrex::Real(0.95), amrex::Real(0.0), amrex::Real(1.0), cgrau}
    }};

    for (const PurePhaseCase& test_case : cases) {
        SCOPED_TRACE(test_case.label);
        auto flux = [&](const amrex::Real qp, const amrex::Real rho) {
            SAMPrecipFaceState face_state{};
            face_state.rho_avg = rho;
            face_state.tabs_avg = test_case.tabs;
            face_state.qp_avg = qp;
            face_state.omp = test_case.omp;
            face_state.omg = test_case.omg;
            face_state.qrr = face_state.omp * qp;
            face_state.qss = (one - face_state.omp) * (one - face_state.omg) * qp;
            face_state.qgg = (one - face_state.omp) * face_state.omg * qp;
            const amrex::Real precip_flux =
                sam_precip_flux_from_face_state(face_state, vrain, vsnow, vgrau);
            return sam_precip_flux_density_corrected(precip_flux, rho0, rho);
        };

        const amrex::Real base_flux = flux(test_case.qp, test_case.rho);
        const amrex::Real h_q = std::min(derivative_step(test_case.qp), amrex::Real(0.25) * test_case.qp);
        const amrex::Real h_rho = std::min(derivative_step(test_case.rho), amrex::Real(0.25) * test_case.rho);
        const amrex::Real dF_dq = central_difference([&](const amrex::Real qp) {
            return flux(qp, test_case.rho);
        }, test_case.qp, h_q);
        const amrex::Real dF_drho = central_difference([&](const amrex::Real rho) {
            return flux(test_case.qp, rho);
        }, test_case.rho, h_rho);

        const amrex::Real expected_dF_dq = (one + test_case.exponent) * base_flux / test_case.qp;
        const amrex::Real expected_dF_drho = (test_case.exponent + myhalf) * base_flux / test_case.rho;

        EXPECT_NEAR(dF_dq, expected_dF_dq, amrex::Real(2.0) * finite_difference_tol(expected_dF_dq, h_q));
        EXPECT_NEAR(dF_drho, expected_dF_drho, finite_difference_tol(expected_dF_drho, h_rho));
    }
}

// Motivation:
// In the positive-domain pure-phase regime, the density-corrected SAM precip
// flux should increase monotonically with both q and rho.
TEST(SAMScalar, PrecipFlux_MonotoneInQpAndDensity)
{
    const amrex::Real rho0 = amrex::Real(1.29);
    const amrex::Real vrain = amrex::Real(2.0);
    const amrex::Real vsnow = amrex::Real(3.0);
    const amrex::Real vgrau = amrex::Real(4.0);

    struct PurePhaseCase {
        const char* label;
        amrex::Real tabs;
        amrex::Real qp;
        amrex::Real rho;
        amrex::Real omp;
        amrex::Real omg;
    };

    const std::array<PurePhaseCase, 3> cases = {{
        {"rain", amrex::Real(285.0), amrex::Real(1.0e-3), amrex::Real(1.0), amrex::Real(1.0), amrex::Real(0.0)},
        {"snow", amrex::Real(260.0), amrex::Real(1.2e-3), amrex::Real(0.85), amrex::Real(0.0), amrex::Real(0.0)},
        {"graupel", amrex::Real(245.0), amrex::Real(1.4e-3), amrex::Real(0.95), amrex::Real(0.0), amrex::Real(1.0)}
    }};

    for (const PurePhaseCase& test_case : cases) {
        SCOPED_TRACE(test_case.label);
        auto flux = [&](const amrex::Real qp, const amrex::Real rho) {
            SAMPrecipFaceState face_state{};
            face_state.rho_avg = rho;
            face_state.tabs_avg = test_case.tabs;
            face_state.qp_avg = qp;
            face_state.omp = test_case.omp;
            face_state.omg = test_case.omg;
            face_state.qrr = face_state.omp * qp;
            face_state.qss = (one - face_state.omp) * (one - face_state.omg) * qp;
            face_state.qgg = (one - face_state.omp) * face_state.omg * qp;
            const amrex::Real precip_flux =
                sam_precip_flux_from_face_state(face_state, vrain, vsnow, vgrau);
            return sam_precip_flux_density_corrected(precip_flux, rho0, rho);
        };

        const amrex::Real base_flux = flux(test_case.qp, test_case.rho);
        const amrex::Real larger_q_flux = flux(amrex::Real(1.5) * test_case.qp, test_case.rho);
        const amrex::Real larger_rho_flux = flux(test_case.qp, amrex::Real(1.2) * test_case.rho);

        EXPECT_GT(larger_q_flux, base_flux - backend_math_abs_tol(base_flux));
        EXPECT_GT(larger_rho_flux, base_flux - backend_math_abs_tol(base_flux));
    }
}

// Motivation:
// The current sedimentation substep helper is documented as a reduced-flux
// ceil rule rather than a fall-speed CFL. This test characterizes that legacy
// reduced-flux behavior explicitly.
TEST(SAMScalar, SubstepCountCharacterizesReducedFluxRule)
{
    const amrex::Real dt = amrex::Real(5.0);
    const amrex::Real dz = amrex::Real(100.0);

    EXPECT_EQ(sam_substep_count_from_reduced_flux(amrex::Real(0.0), dt, dz), 0);
    EXPECT_EQ(sam_substep_count_from_reduced_flux(amrex::Real(1.0), dt, dz), 1);
    EXPECT_EQ(sam_substep_count_from_reduced_flux(amrex::Real(10.0), dt, dz), 1);
    EXPECT_EQ(sam_substep_count_from_reduced_flux(amrex::Real(10.0001), dt, dz), 2);
    EXPECT_EQ(sam_substep_count_from_reduced_flux(amrex::Real(25.0), dt, dz), 3);
}