#include <array>
#include <memory>
#include <string>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>

#include <gtest/gtest.h>

#include "ERF_GTestKesslerCommon.H"

using namespace kessler_test;

namespace {

void expect_near_formula (const amrex::Real actual,
                          const amrex::Real expected)
{
    EXPECT_NEAR(actual, expected, std::max(formula_abs_tol(expected), formula_rel_tol(expected)));
}

void expect_near_backend_math (const amrex::Real actual,
                               const amrex::Real expected)
{
    EXPECT_NEAR(actual, expected, backend_math_abs_tol(expected));
}

void expect_source_terms_match (const KesslerSourceTerms& actual,
                                const KesslerSourceTerms& expected)
{
    expect_near_formula(actual.dq_vapor_to_cloud, expected.dq_vapor_to_cloud);
    expect_near_formula(actual.dq_cloud_to_vapor, expected.dq_cloud_to_vapor);
    expect_near_backend_math(actual.dq_cloud_to_rain, expected.dq_cloud_to_rain);
    expect_near_backend_math(actual.dq_rain_to_vapor, expected.dq_rain_to_vapor);
}

void expect_sat_adjustment_match (const KesslerSaturationAdjustment& actual,
                                  const KesslerSaturationAdjustment& expected)
{
    expect_near_formula(actual.dq_vapor_to_cloud, expected.dq_vapor_to_cloud);
    expect_near_formula(actual.dq_cloud_to_vapor, expected.dq_cloud_to_vapor);
}

} // namespace

// Motivation: Saturation adjustment should be an exact identity at qv == qsat.
// This protects the no-op branch and the physical contract that no phase
// change occurs when the parcel is already saturated.
TEST(KesslerScalar, SaturationAdjustment_NoOpAtSaturation)
{
    const amrex::Real qv = amrex::Real(1.0e-2);
    const amrex::Real qc = amrex::Real(2.0e-3);
    const amrex::Real qsat = qv;
    const amrex::Real dtqsat = amrex::Real(0.02);

    const KesslerSaturationAdjustment actual =
        kessler_saturation_adjustment(qv, qc, qsat, dtqsat, true);

    EXPECT_NEAR(actual.dq_vapor_to_cloud, amrex::Real(0.0), exact_zero_or_near_zero_tol());
    EXPECT_NEAR(actual.dq_cloud_to_vapor, amrex::Real(0.0), exact_zero_or_near_zero_tol());
}

// Motivation: The smooth condensation branch has a closed-form update.
// This checks the helper against an independent derivation away from caps and
// threshold boundaries.
TEST(KesslerScalar, SaturationAdjustment_CondensationMatchesIndependentReference)
{
    const amrex::Real qv = amrex::Real(1.2e-2);
    const amrex::Real qc = amrex::Real(7.0e-4);
    const amrex::Real qsat = amrex::Real(1.0e-2);
    const amrex::Real dtqsat = amrex::Real(0.03);

    const KesslerSaturationAdjustment actual =
        kessler_saturation_adjustment(qv, qc, qsat, dtqsat, true);
    const KesslerSaturationAdjustment expected =
        reference_saturation_adjustment(qv, qc, qsat, dtqsat, true);

    expect_sat_adjustment_match(actual, expected);
}

// Motivation: Subsaturated cloudy air should evaporate cloud water toward the
// linearized saturation target. This protects the cloud-evaporation branch.
TEST(KesslerScalar, SaturationAdjustment_CloudEvaporationMatchesIndependentReference)
{
    const amrex::Real qv = amrex::Real(9.0e-3);
    const amrex::Real qc = amrex::Real(2.0e-3);
    const amrex::Real qsat = amrex::Real(1.0e-2);
    const amrex::Real dtqsat = amrex::Real(0.02);

    const KesslerSaturationAdjustment actual =
        kessler_saturation_adjustment(qv, qc, qsat, dtqsat, true);
    const KesslerSaturationAdjustment expected =
        reference_saturation_adjustment(qv, qc, qsat, dtqsat, true);

    expect_sat_adjustment_match(actual, expected);
}

// Motivation: Cloud evaporation is limited by the available cloud water. This
// protects the cap branch instead of relying on sign-only behavior.
TEST(KesslerScalar, SaturationAdjustment_CloudEvaporationCappedByCloudWater)
{
    const amrex::Real qv = amrex::Real(7.0e-3);
    const amrex::Real qc = amrex::Real(1.0e-4);
    const amrex::Real qsat = amrex::Real(1.0e-2);
    const amrex::Real dtqsat = amrex::Real(0.01);

    const KesslerSaturationAdjustment actual =
        kessler_saturation_adjustment(qv, qc, qsat, dtqsat, true);

    EXPECT_NEAR(actual.dq_cloud_to_vapor, qc, formula_abs_tol(qc));
    EXPECT_NEAR(actual.dq_vapor_to_cloud, amrex::Real(0.0), exact_zero_or_near_zero_tol());
}

// Motivation: When SHOC owns condensation, both saturation-adjustment source
// terms must be suppressed even for supersaturated or subsaturated inputs.
TEST(KesslerScalar, SaturationAdjustment_DoCondFalseSuppressesCondensationAndCloudEvaporation)
{
    const std::array<std::tuple<amrex::Real, amrex::Real, amrex::Real>, 2> cases = {{
        {amrex::Real(1.2e-2), amrex::Real(6.0e-4), amrex::Real(1.0e-2)},
        {amrex::Real(9.0e-3), amrex::Real(2.0e-3), amrex::Real(1.0e-2)}
    }};

    for (const auto& [qv, qc, qsat] : cases) {
        SCOPED_TRACE("qv=" + std::to_string(static_cast<double>(qv)));
        const KesslerSaturationAdjustment actual =
            kessler_saturation_adjustment(qv, qc, qsat, amrex::Real(0.03), false);
        EXPECT_NEAR(actual.dq_vapor_to_cloud, amrex::Real(0.0), exact_zero_or_near_zero_tol());
        EXPECT_NEAR(actual.dq_cloud_to_vapor, amrex::Real(0.0), exact_zero_or_near_zero_tol());
    }
}

// Motivation: The smooth condensation branch has analytic derivatives with
// respect to qv, qsat, and dqsdT. This protects the algebraic form away from
// caps and branch switches; the finite difference stays strictly inside the
// smooth branch.
TEST(KesslerScalar, SaturationAdjustment_AnalyticDerivativeInSmoothCondensationBranch)
{
    const amrex::Real qv = amrex::Real(1.12e-2);
    const amrex::Real qc = amrex::Real(4.0e-4);
    const amrex::Real qsat = amrex::Real(1.00e-2);
    const amrex::Real dtqsat = amrex::Real(0.02);
    const amrex::Real denom = amrex::Real(1.0) + kSatAdjLatentOverCp * dtqsat;

    const amrex::Real h_qv = derivative_step(qv);
    const amrex::Real h_qsat = derivative_step(qsat);
    const amrex::Real h_dtqsat = derivative_step(dtqsat);

    const auto branch = [=] (const amrex::Real qv_value,
                             const amrex::Real qsat_value,
                             const amrex::Real dtqsat_value) {
        return kessler_saturation_adjustment(qv_value, qc, qsat_value, dtqsat_value, true).dq_vapor_to_cloud;
    };

    const amrex::Real d_dq_dqv = central_difference(
        [=] (const amrex::Real value) { return branch(value, qsat, dtqsat); }, qv, h_qv);
    const amrex::Real d_dq_dqsat = central_difference(
        [=] (const amrex::Real value) { return branch(qv, value, dtqsat); }, qsat, h_qsat);
    const amrex::Real d_dq_ddtqsat = central_difference(
        [=] (const amrex::Real value) { return branch(qv, qsat, value); }, dtqsat, h_dtqsat);

    const amrex::Real expected_dqv = amrex::Real(1.0) / denom;
    const amrex::Real expected_dqsat = -amrex::Real(1.0) / denom;
    const amrex::Real expected_ddtqsat =
        -(qv - qsat) * kSatAdjLatentOverCp / (denom * denom);

    EXPECT_NEAR(d_dq_dqv, expected_dqv, finite_difference_tol(expected_dqv, h_qv));
    EXPECT_NEAR(d_dq_dqsat, expected_dqsat, finite_difference_tol(expected_dqsat, h_qsat));
    EXPECT_NEAR(d_dq_ddtqsat, expected_ddtqsat, finite_difference_tol(expected_ddtqsat, h_dtqsat));
}

// Motivation: For valid qsat >= 0 and dqsdT >= 0, the condensation update
// cannot be limited by the qv cap. This documents the contract instead of
// forcing an invalid-state branch hit.
TEST(KesslerScalar, SaturationAdjustment_CondensationQvCapUnreachableForValidPositiveQsat)
{
    const std::array<std::tuple<amrex::Real, amrex::Real, amrex::Real>, 4> cases = {{
        {amrex::Real(1.2e-2), amrex::Real(1.0e-2), amrex::Real(0.0)},
        {amrex::Real(1.2e-2), amrex::Real(1.0e-2), amrex::Real(0.01)},
        {amrex::Real(8.0e-3), amrex::Real(0.0), amrex::Real(0.02)},
        {amrex::Real(5.0e-3), amrex::Real(2.0e-3), amrex::Real(0.05)}
    }};

    for (const auto& [qv, qsat, dtqsat] : cases) {
        SCOPED_TRACE("qv=" + std::to_string(static_cast<double>(qv)) +
                     " qsat=" + std::to_string(static_cast<double>(qsat)));
        const KesslerSaturationAdjustment actual =
            kessler_saturation_adjustment(qv, amrex::Real(1.0e-3), qsat, dtqsat, true);
        EXPECT_LE(actual.dq_vapor_to_cloud, qv + formula_abs_tol(qv));
        EXPECT_NEAR(actual.dq_vapor_to_cloud,
                    (qv > qsat) ? (qv - qsat) / (amrex::Real(1.0) + kSatAdjLatentOverCp * dtqsat)
                                : amrex::Real(0.0),
                    formula_abs_tol(qv));
    }
}

// Motivation: The local Kessler source update exchanges water among qv, qc,
// and qp only. This protects the algebraic total-water conservation contract
// before sedimentation.
TEST(KesslerScalar, WarmRainSources_TotalWaterConservedByLocalSources)
{
    const KesslerSourceTerms source_terms = kessler_warm_rain_sources(
        amrex::Real(9.0e-3), amrex::Real(2.0e-3), amrex::Real(1.5e-3),
        amrex::Real(1.1), amrex::Real(900.0), amrex::Real(1.0e-2),
        amrex::Real(0.02), amrex::Real(2.0), true);

    const amrex::Real source_sum =
        (-source_terms.dq_vapor_to_cloud + source_terms.dq_cloud_to_vapor + source_terms.dq_rain_to_vapor)
      + ( source_terms.dq_vapor_to_cloud - source_terms.dq_cloud_to_vapor - source_terms.dq_cloud_to_rain)
      + ( source_terms.dq_cloud_to_rain - source_terms.dq_rain_to_vapor);

    EXPECT_NEAR(source_sum, amrex::Real(0.0), exact_zero_or_near_zero_tol());
}

// Motivation: The autoconversion trigger uses qc > qcw0, not >=. The below,
// equal, and above cases protect this threshold branch with independent values
// instead of sign-only assertions.
TEST(KesslerScalar, WarmRainSources_AutoconversionThresholdBelowEqualAbove)
{
    const std::array<amrex::Real, 3> qc_cases = {
        qcw0 - amrex::Real(1.0e-6), qcw0, qcw0 + amrex::Real(1.0e-6)};

    for (const amrex::Real qc : qc_cases) {
        SCOPED_TRACE("qc=" + std::to_string(static_cast<double>(qc)));
        const KesslerSourceTerms actual = kessler_warm_rain_sources(
            amrex::Real(1.0e-2), qc, amrex::Real(0.0), amrex::Real(1.0),
            amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(1.0), true);
        const KesslerSourceTerms expected = reference_warm_rain_sources(
            amrex::Real(1.0e-2), qc, amrex::Real(0.0), amrex::Real(1.0),
            amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(1.0), true);
        expect_source_terms_match(actual, expected);
    }
}

// Motivation: Cloud-to-rain conversion cannot remove more cloud water than is
// available. This protects the cap branch for the combined accretion and
// autoconversion source.
TEST(KesslerScalar, WarmRainSources_CloudToRainCappedByCloudWater)
{
    const amrex::Real qc = amrex::Real(2.0e-4);
    const KesslerSourceTerms actual = kessler_warm_rain_sources(
        amrex::Real(1.0e-2), qc, amrex::Real(1.0), amrex::Real(1.0),
        amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.01), amrex::Real(10.0), true);

    EXPECT_NEAR(actual.dq_cloud_to_rain, qc, formula_abs_tol(qc));
}

// Motivation: For fixed cloud water with autoconversion disabled, accretion is
// monotone in rain water. This protects a physical monotonicity property of
// the warm-rain closure.
TEST(KesslerScalar, WarmRainSources_AccretionMonotoneInRainWater)
{
    const amrex::Real qv = amrex::Real(1.0e-2);
    const amrex::Real qc = amrex::Real(8.0e-4);
    const std::array<amrex::Real, 4> qp_values = {
        amrex::Real(0.0), amrex::Real(1.0e-4), amrex::Real(5.0e-4), amrex::Real(1.5e-3)};

    amrex::Real previous = -amrex::Real(1.0);
    for (const amrex::Real qp : qp_values) {
        SCOPED_TRACE("qp=" + std::to_string(static_cast<double>(qp)));
        const KesslerSourceTerms actual = kessler_warm_rain_sources(
            qv, qc, qp, amrex::Real(1.1), amrex::Real(900.0), qv, amrex::Real(0.02), amrex::Real(1.0), true);
        EXPECT_GE(actual.dq_cloud_to_rain, amrex::Real(0.0));
        if (previous >= amrex::Real(0.0)) {
            EXPECT_GE(actual.dq_cloud_to_rain + formula_abs_tol(actual.dq_cloud_to_rain), previous);
        }
        previous = actual.dq_cloud_to_rain;
    }
}

// Motivation: Rain evaporation requires both rain water and subsaturation.
// This protects three distinct branch guards in the composed warm-rain helper.
TEST(KesslerScalar, WarmRainSources_RainEvaporationRequiresRainAndSubsaturation)
{
    struct Case {
        amrex::Real qp;
        amrex::Real qv;
        amrex::Real qsat;
        bool expect_positive;
    };

    const std::array<Case, 3> cases = {{
        {amrex::Real(0.0), amrex::Real(9.0e-3), amrex::Real(1.0e-2), false},
        {amrex::Real(2.0e-3), amrex::Real(1.0e-2), amrex::Real(1.0e-2), false},
        {amrex::Real(2.0e-3), amrex::Real(9.0e-3), amrex::Real(1.0e-2), true}
    }};

    for (const auto& test_case : cases) {
        SCOPED_TRACE("qp=" + std::to_string(static_cast<double>(test_case.qp)) +
                     " qv=" + std::to_string(static_cast<double>(test_case.qv)));
        const KesslerSourceTerms actual = kessler_warm_rain_sources(
            test_case.qv, amrex::Real(5.0e-4), test_case.qp, amrex::Real(1.1),
            amrex::Real(900.0), test_case.qsat, amrex::Real(0.02), amrex::Real(2.0), true);
        if (test_case.expect_positive) {
            EXPECT_GT(actual.dq_rain_to_vapor, amrex::Real(0.0));
        } else {
            EXPECT_NEAR(actual.dq_rain_to_vapor, amrex::Real(0.0), exact_zero_or_near_zero_tol());
        }
    }
}

// Motivation: Rain evaporation cannot exceed the available rain water. This
// protects the availability cap in the rain-evaporation branch.
TEST(KesslerScalar, WarmRainSources_RainEvaporationCappedByRainWater)
{
    const amrex::Real qp = amrex::Real(1.0e-4);
    const KesslerSourceTerms actual = kessler_warm_rain_sources(
        amrex::Real(1.0e-3), amrex::Real(5.0e-4), qp, amrex::Real(1.2),
        amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.01), amrex::Real(200.0), true);

    EXPECT_NEAR(actual.dq_rain_to_vapor, qp, formula_abs_tol(qp));
}

// Motivation: For valid nonnegative inputs, all Kessler source magnitudes are
// nonnegative. This is secondary physical-sanity coverage for the composed
// helper, not the primary source-term verification.
TEST(KesslerScalar, WarmRainSources_NoNegativeTendenciesForValidInputs)
{
    const std::array<KernelCase, 4> cases = {make_kernel_cases()[0], make_kernel_cases()[1],
                                             make_kernel_cases()[2], make_kernel_cases()[3]};

    for (const KernelCase& test_case : cases) {
        SCOPED_TRACE(case_label(test_case));
        const KesslerSourceTerms actual = kessler_warm_rain_sources(
            test_case.qv, test_case.qc, test_case.qp, test_case.rho, test_case.pressure_mbar,
            test_case.qsat, test_case.dtqsat, test_case.dt, test_case.do_cond);

        EXPECT_GE(actual.dq_vapor_to_cloud, -exact_zero_or_near_zero_tol());
        EXPECT_GE(actual.dq_cloud_to_vapor, -exact_zero_or_near_zero_tol());
        EXPECT_GE(actual.dq_cloud_to_rain, -exact_zero_or_near_zero_tol());
        EXPECT_GE(actual.dq_rain_to_vapor, -exact_zero_or_near_zero_tol());
    }
}

// Motivation: The precipitation flux must be zero when qp == 0 even if one
// avoids making a stronger claim about the terminal-velocity value itself.
TEST(KesslerScalar, TerminalVelocity_ZeroRainGivesZeroFlux)
{
    const amrex::Real rho = amrex::Real(1.16);
    const amrex::Real velocity = kessler_terminal_velocity(rho, amrex::Real(0.0));
    const amrex::Real flux = kessler_precip_flux(rho, velocity, amrex::Real(0.0));

    EXPECT_NEAR(flux, amrex::Real(0.0), exact_zero_or_near_zero_tol());
}

// Motivation: Terminal velocity should be nondecreasing in rain water for
// fixed positive density. This protects a basic physical monotonicity contract.
TEST(KesslerScalar, TerminalVelocity_MonotoneInQp)
{
    const amrex::Real rho = amrex::Real(1.16);
    const std::array<amrex::Real, 4> qp_values = {
        amrex::Real(1.0e-6), amrex::Real(1.0e-4), amrex::Real(1.0e-3), amrex::Real(1.0e-2)};

    amrex::Real previous = -amrex::Real(1.0);
    for (const amrex::Real qp : qp_values) {
        const amrex::Real velocity = kessler_terminal_velocity(rho, qp);
        EXPECT_GE(velocity, amrex::Real(0.0));
        if (previous >= amrex::Real(0.0)) {
            EXPECT_GE(velocity + backend_math_abs_tol(velocity), previous);
        }
        previous = velocity;
    }
}

// Motivation: The positive-domain terminal-velocity formula has analytic
// derivatives in qp and rho. This protects the functional form away from qp ==
// 0, where the derivative is singular.
TEST(KesslerScalar, TerminalVelocity_AnalyticDerivativeInPositiveDomain)
{
    const amrex::Real rho = amrex::Real(1.16);
    const amrex::Real qp = amrex::Real(1.0e-2);
    const amrex::Real h_rho = derivative_step(rho);
    const amrex::Real h_qp = derivative_step(qp);

    const amrex::Real d_dqp = central_difference(
        [=] (const amrex::Real value) { return kessler_terminal_velocity(rho, value); }, qp, h_qp);
    const amrex::Real d_drho = central_difference(
        [=] (const amrex::Real value) { return kessler_terminal_velocity(value, qp); }, rho, h_rho);

    const amrex::Real expected_dqp = reference_terminal_velocity_dqp(rho, qp);
    const amrex::Real expected_drho = reference_terminal_velocity_drho(rho, qp);

    EXPECT_NEAR(d_dqp, expected_dqp, finite_difference_tol(expected_dqp, h_qp));
    EXPECT_NEAR(d_drho, expected_drho, finite_difference_tol(expected_drho, h_rho));
}

// Motivation: The terminal-velocity formula can be inverted analytically for
// qp at fixed density. This roundtrip protects the power-law exponents.
TEST(KesslerScalar, TerminalVelocity_InverseRoundtripQp)
{
    const amrex::Real rho = amrex::Real(1.25);
    const std::array<amrex::Real, 3> qp_values = {
        amrex::Real(2.0e-5), amrex::Real(5.0e-4), amrex::Real(2.0e-3)};

    for (const amrex::Real qp : qp_values) {
        SCOPED_TRACE("qp=" + std::to_string(static_cast<double>(qp)));
        const amrex::Real velocity = kessler_terminal_velocity(rho, qp);
        const amrex::Real recovered_qp = inverse_qp_from_terminal_velocity(velocity, rho);
        EXPECT_NEAR(recovered_qp, qp, backend_math_abs_tol(qp));
    }
}

// Motivation: Precipitation flux is defined as rho * V * qp. This protects the
// functional identity independently of the production helper composition.
TEST(KesslerScalar, PrecipFlux_EqualsRhoTimesTerminalVelocityTimesQp)
{
    const amrex::Real rho = amrex::Real(1.2);
    const amrex::Real qp = amrex::Real(1.0e-3);
    const amrex::Real velocity = reference_terminal_velocity(rho, qp);
    const amrex::Real expected = reference_precip_flux(rho, velocity, qp);
    const amrex::Real actual = kessler_precip_flux(rho, kessler_terminal_velocity(rho, qp), qp);

    expect_near_backend_math(actual, expected);
}

// Motivation: The positive-domain precipitation flux formula has analytic
// derivatives in qp and rho. This protects the power-law composition of rho,
// V, and qp away from qp == 0.
TEST(KesslerScalar, PrecipFlux_AnalyticDerivativeInPositiveDomain)
{
    const amrex::Real rho = amrex::Real(1.16);
    const amrex::Real qp = amrex::Real(1.0e-2);
    const amrex::Real h_rho = derivative_step(rho);
    const amrex::Real h_qp = derivative_step(qp);

    const amrex::Real d_dqp = central_difference(
        [=] (const amrex::Real value) {
            return kessler_precip_flux(rho, kessler_terminal_velocity(rho, value), value);
        },
        qp, h_qp);
    const amrex::Real d_drho = central_difference(
        [=] (const amrex::Real value) {
            return kessler_precip_flux(value, kessler_terminal_velocity(value, qp), qp);
        },
        rho, h_rho);

    const amrex::Real expected_dqp = reference_precip_flux_dqp(rho, qp);
    const amrex::Real expected_drho = reference_precip_flux_drho(rho, qp);

    EXPECT_NEAR(d_dqp, expected_dqp, finite_difference_tol(expected_dqp, h_qp));
    EXPECT_NEAR(d_drho, expected_drho, finite_difference_tol(expected_drho, h_rho));
}

// Motivation: Face-state selection has three branches: bottom, top, and
// interior. This protects the public sedimentation path's boundary logic.
TEST(KesslerScalar, FaceState_BottomTopInteriorBranches)
{
    const KesslerFaceState bottom =
        kessler_face_state(0, 0, 2, amrex::Real(1.0), amrex::Real(1.1), amrex::Real(0.2), amrex::Real(0.3));
    const KesslerFaceState top =
        kessler_face_state(3, 0, 2, amrex::Real(1.0), amrex::Real(1.1), amrex::Real(0.2), amrex::Real(0.3));
    const KesslerFaceState interior =
        kessler_face_state(1, 0, 2, amrex::Real(1.0), amrex::Real(1.1), amrex::Real(0.2), amrex::Real(0.3));

    EXPECT_NEAR(bottom.rho, amrex::Real(1.1), formula_abs_tol(amrex::Real(1.1)));
    EXPECT_NEAR(bottom.qp, amrex::Real(0.3), formula_abs_tol(amrex::Real(0.3)));
    EXPECT_NEAR(top.rho, amrex::Real(1.0), formula_abs_tol(amrex::Real(1.0)));
    EXPECT_NEAR(top.qp, amrex::Real(0.2), formula_abs_tol(amrex::Real(0.2)));
    EXPECT_NEAR(interior.rho, amrex::Real(1.05), formula_abs_tol(amrex::Real(1.05)));
    EXPECT_NEAR(interior.qp, amrex::Real(0.25), formula_abs_tol(amrex::Real(0.25)));
}

// Motivation: Face rain water is clipped nonnegative after branch selection.
// This protects the physical sanity of the precipitating face state.
TEST(KesslerScalar, FaceState_ClipsNegativeFaceRainWater)
{
    const KesslerFaceState actual =
        kessler_face_state(1, 0, 2, amrex::Real(1.0), amrex::Real(1.1), -amrex::Real(0.2), -amrex::Real(0.1));

    EXPECT_NEAR(actual.qp, amrex::Real(0.0), exact_zero_or_near_zero_tol());
}

// Motivation: The current face-state contract averages neighboring qp first,
// then clips the averaged face value. This protects the branch order, not a
// physically stronger invariant.
TEST(KesslerScalar, FaceState_AveragesThenClips)
{
    // MUST MATCH: production face-state averaging and clipping order.
    const KesslerFaceState actual =
        kessler_face_state(1, 0, 2, amrex::Real(1.0), amrex::Real(1.0), -amrex::Real(0.2), amrex::Real(0.1));

    EXPECT_NEAR(actual.qp, amrex::Real(0.0), exact_zero_or_near_zero_tol());
}

// Motivation: Equal face fluxes imply zero sedimentation tendency. This is the
// identity branch for the flux-divergence update.
TEST(KesslerScalar, SedimentationTendency_ZeroForEqualFaceFluxes)
{
    const amrex::Real actual = kessler_sedimentation_tendency(
        amrex::Real(3.0e-3), amrex::Real(3.0e-3), amrex::Real(1.1), amrex::Real(1.0), amrex::Real(0.5));

    EXPECT_NEAR(actual, amrex::Real(0.0), exact_zero_or_near_zero_tol());
}

// Motivation: The sedimentation tendency should scale linearly with Jinv. This
// protects the detJ weighting in the local divergence formula.
TEST(KesslerScalar, SedimentationTendency_DetJScalesFluxDivergence)
{
    const amrex::Real base = kessler_sedimentation_tendency(
        amrex::Real(4.0e-3), amrex::Real(1.0e-3), amrex::Real(1.0), amrex::Real(1.0), amrex::Real(0.5));
    const amrex::Real doubled = kessler_sedimentation_tendency(
        amrex::Real(4.0e-3), amrex::Real(1.0e-3), amrex::Real(1.0), amrex::Real(2.0), amrex::Real(0.5));

    EXPECT_NEAR(doubled, amrex::Real(2.0) * base, formula_abs_tol(doubled));
}

// Motivation: The sedimentation zero threshold uses a strict < comparison.
// This protects the documented cutoff branch on both sides of the boundary.
TEST(KesslerScalar, SedimentationThreshold_StrictLessThanAbsoluteCutoff)
{
    // MUST MATCH: production sedimentation zero threshold.
    const amrex::Real cutoff = amrex::Real(1.0e-14);

    EXPECT_TRUE(kessler_is_small_sedimentation_value(cutoff - amrex::Real(1.0e-16)));
    EXPECT_FALSE(kessler_is_small_sedimentation_value(cutoff));
    EXPECT_FALSE(kessler_is_small_sedimentation_value(cutoff + amrex::Real(1.0e-16)));
    EXPECT_TRUE(kessler_is_small_sedimentation_value(-cutoff + amrex::Real(1.0e-16)));
}

// Motivation: This tests the integer helper exactly as written for a supplied
// reduced value. It does not claim the current call path uses the physically
// correct reduced quantity.
TEST(KesslerScalar, SubstepCount_IntegerFormulaForProvidedReducedValue)
{
    const amrex::Real reduced_value = amrex::Real(0.375);
    const int actual = kessler_num_sedimentation_substeps(reduced_value, amrex::Real(2.0), amrex::Real(0.5));
    const int expected = reference_substeps_from_reduced_value(reduced_value, amrex::Real(2.0), amrex::Real(0.5));

    EXPECT_EQ(actual, expected);
}

// Motivation: Copy_State_to_Micro must store pressure in mbar / hPa because
// qsat pressure inputs use mbar while EOS pressure utilities use Pa.
TEST(KesslerScalar, CopyStateToMicro_PressureStoredInMbar)
{
    const amrex::Geometry geom = make_geometry(1, 1, 1);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));

    const amrex::Real tabs = amrex::Real(290.0);
    const amrex::Real pres_mbar = amrex::Real(900.0);
    amrex::Real qsat;
    amrex::Real dtqsat;
    erf_qsatw(tabs, pres_mbar, qsat);
    erf_dtqsatw(tabs, pres_mbar, dtqsat);
    const PrimitiveState state = make_primitive_state(
        tabs, pres_mbar, qsat + amrex::Real(2.0e-4), amrex::Real(5.0e-5), amrex::Real(0.0));

    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);

    kessler.Define(sc);
    kessler.Set_dzmin(geom.CellSize(2));
    kessler.Init(cons, ba, geom, kDefaultDt, z_phys_nd, detJ_cc);
    run_and_sync([&] {
        kessler.Copy_State_to_Micro(cons);
        kessler.Advance(kDefaultDt, sc);
        kessler.Copy_Micro_to_State(cons);
    });

    const KesslerSourceTerms expected_sources = reference_warm_rain_sources(
        state.qv, state.qc, state.qp, state.rho, state.pres_mbar, qsat, dtqsat, kDefaultDt, true);
    amrex::Real qv_expected = state.qv;
    amrex::Real qc_expected = state.qc;
    amrex::Real qp_expected = state.qp;
    apply_local_sources(qv_expected, qc_expected, qp_expected, expected_sources);

    amrex::Real theta_expected = state.theta;
    theta_expected += (state.theta / state.tabs) * (lcond / sc.c_p)
        * (expected_sources.dq_vapor_to_cloud
           - expected_sources.dq_cloud_to_vapor
           - expected_sources.dq_rain_to_vapor);

    EXPECT_NEAR(cons.max(RhoQ1_comp), state.rho * qv_expected, formula_abs_tol(state.rho * qv_expected));
    EXPECT_NEAR(cons.max(RhoQ2_comp), state.rho * qc_expected, formula_abs_tol(state.rho * qc_expected));
    EXPECT_NEAR(cons.max(RhoQ3_comp), state.rho * qp_expected, exact_zero_or_near_zero_tol());
    EXPECT_NEAR(cons.max(RhoTheta_comp), state.rho * theta_expected,
                formula_abs_tol(state.rho * theta_expected));
}

// Motivation: Copy_Micro_to_State must write qv, qc, and qp back as rho-
// weighted conserved scalars. This protects the public copy-back contract.
TEST(KesslerScalar, CopyMicroToState_WritesQvQcQpAsRhoWeightedConservedScalars)
{
    const amrex::Geometry geom = make_geometry(1, 1, 1);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));

    const PrimitiveState state = make_primitive_state(
        amrex::Real(289.0), amrex::Real(910.0), amrex::Real(8.0e-3), amrex::Real(4.0e-4), amrex::Real(7.0e-4));
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);

    kessler.Define(sc);
    kessler.Set_dzmin(geom.CellSize(2));
    kessler.Init(cons, ba, geom, kDefaultDt, z_phys_nd, detJ_cc);

    run_and_sync([&] {
        kessler.Copy_State_to_Micro(cons);
        kessler.Copy_Micro_to_State(cons);
    });

    EXPECT_NEAR(cons.max(RhoQ1_comp), state.rho * state.qv, formula_abs_tol(state.rho * state.qv));
    EXPECT_NEAR(cons.max(RhoQ2_comp), state.rho * state.qc, formula_abs_tol(state.rho * state.qc));
    EXPECT_NEAR(cons.max(RhoQ3_comp), state.rho * state.qp, formula_abs_tol(state.rho * state.qp));
}

// Motivation: Copy_Micro_to_State writes the current microphysics theta back to
// rho*theta. This is a documentation contract, not a claim that theta is a
// roundtrip invariant through phase change.
TEST(KesslerScalar, CopyMicroToState_WritesCurrentMicroThetaToRhoTheta)
{
    const amrex::Geometry geom = make_geometry(1, 1, 1);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));

    const PrimitiveState state = make_primitive_state(
        amrex::Real(292.0), amrex::Real(940.0), amrex::Real(6.0e-3), amrex::Real(1.0e-4), amrex::Real(0.0));
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                set_conserved_cell(arr, i, j, k, state);
            });
        });
    }

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);

    kessler.Define(sc);
    kessler.Set_dzmin(geom.CellSize(2));
    kessler.Init(cons, ba, geom, kDefaultDt, z_phys_nd, detJ_cc);

    run_and_sync([&] {
        kessler.Copy_State_to_Micro(cons);
        kessler.Copy_Micro_to_State(cons);
    });

    EXPECT_NEAR(cons.max(RhoTheta_comp), state.rho * state.theta,
                formula_abs_tol(state.rho * state.theta));
}

// Motivation: This is a characterization test, not a correctness test. It
// pins current behavior so that any future change must also address
// `DISABLED_SubstepCountUsesTerminalVelocityCFL`.
TEST(KesslerScalar, Characterization_SubstepCountCurrentlyUsesFluxLikeReducedValue)
{
    const amrex::Real rho = amrex::Real(1.16);
    const amrex::Real qp = amrex::Real(1.0e-3);
    const amrex::Real dt = amrex::Real(1.0);
    const amrex::Real dz = amrex::Real(1.0);
    const amrex::Real velocity = kessler_terminal_velocity(rho, qp);
    const amrex::Real reduced_flux_like_value = kessler_precip_flux(rho, velocity, qp);

    EXPECT_EQ(kessler_num_sedimentation_substeps(reduced_flux_like_value, dt, dz),
              reference_substeps_from_reduced_value(reduced_flux_like_value, dt, dz));
    EXPECT_EQ(kessler_num_sedimentation_substeps(reduced_flux_like_value, dt, dz), 1);
}

// Motivation: Full cloud and rain evaporation currently both use the original
// subsaturation deficit. This is a characterization test, not a correctness
// test. It pins current behavior so that any future change must also address
// `DISABLED_CloudAndRainEvaporationDoNotBothConsumeSameSubsaturation`.
TEST(KesslerScalar, Characterization_CloudAndRainEvaporationCurrentlyShareOriginalDeficit)
{
    const amrex::Real qv = amrex::Real(8.0e-3);
    const amrex::Real qc = amrex::Real(4.0e-3);
    const amrex::Real qp = amrex::Real(2.0e-3);
    const amrex::Real qsat = amrex::Real(1.0e-2);
    const amrex::Real dtqsat = amrex::Real(0.02);
    const amrex::Real rho = amrex::Real(1.0);
    const amrex::Real dt = amrex::Real(80.0);
    const KesslerSourceTerms actual = kessler_warm_rain_sources(
        qv, qc, qp, rho, amrex::Real(900.0), qsat, dtqsat, dt, true);
    const amrex::Real linearized_capacity = (qsat - qv) / (amrex::Real(1.0) + kSatAdjLatentOverCp * dtqsat);

    EXPECT_GT(actual.dq_cloud_to_vapor + actual.dq_rain_to_vapor, linearized_capacity);
}

// Motivation: This is a characterization test, not a correctness test. It
// pins current behavior so that any future change must also address
// `DISABLED_SaturationThetaUsesConsistentLatentHeatFactor`.
TEST(KesslerScalar, Characterization_SaturationThetaCurrentlyUseDifferentLatentFactors)
{
    EXPECT_NE(kSatAdjLatentOverCp, kThetaLatentOverCp);
}

// Motivation: This is a characterization test, not a correctness test. It
// documents the current absolute sedimentation threshold so any future change
// must also address a matching contract test.
TEST(KesslerScalar, Characterization_SedimentationThresholdUsesAbsoluteOneEMinusFourteen)
{
    // MUST MATCH: production sedimentation zero threshold.
    EXPECT_TRUE(kessler_is_small_sedimentation_value(amrex::Real(9.9e-15)));
    EXPECT_FALSE(kessler_is_small_sedimentation_value(amrex::Real(1.0e-14)));
}

// Motivation: Rain accumulation currently preserves the exact
// `/ Real(1000.0) * Real(1000.0)` cancellation expression. This is a
// characterization test, not a correctness test.
TEST(KesslerScalar, Characterization_RainAccumulationPreservesCurrentUnitCancellationExpression)
{
    const amrex::Real rho = amrex::Real(1.2);
    const amrex::Real qp = amrex::Real(8.0e-4);
    const amrex::Real velocity = kessler_terminal_velocity(rho, qp);
    const amrex::Real dt = amrex::Real(2.0);
    const amrex::Real current_expression = rho * qp * velocity * dt / amrex::Real(1000.0) * amrex::Real(1000.0);

    EXPECT_NEAR(current_expression, rho * qp * velocity * dt, backend_math_abs_tol(current_expression));
}

// Motivation: Expected behavior:
//   The sedimentation substep count should be based on a terminal-velocity CFL,
//   n = ceil((V_terminal + eps) * dt / dz / CFL_MAX).
// Observed current behavior:
//   The current call path reduces a flux-like value rho * V_terminal * qp.
// Minimal reproducer:
//   rho = 1.16, qp = 1.0e-3, dt = 1.0, dz = 1.0, CFL_MAX = 0.5.
// Why disabled:
//   Current development uses the reduced flux-like value, so the physical CFL
//   contract does not hold yet.
// Enable trigger:
//   Production should reduce a terminal velocity rather than rho*V*qp.
TEST(KesslerScalar, DISABLED_SubstepCountUsesTerminalVelocityCFL)
{
    const amrex::Real rho = amrex::Real(1.16);
    const amrex::Real qp = amrex::Real(1.0e-3);
    const amrex::Real dt = amrex::Real(1.0);
    const amrex::Real dz = amrex::Real(1.0);
    const amrex::Real velocity = reference_terminal_velocity(rho, qp);
    const amrex::Real reduced_flux_like_value = reference_precip_flux(rho, velocity, qp);

    EXPECT_EQ(kessler_num_sedimentation_substeps(reduced_flux_like_value, dt, dz),
              reference_velocity_cfl_substeps(velocity, dt, dz));
}

// Motivation: Expected behavior:
//   Total vapor supplied by cloud evaporation plus rain evaporation should not
//   exceed the linearized subsaturation capacity.
// Observed current behavior:
//   Cloud evaporation and rain evaporation both consume the original deficit.
// Minimal reproducer:
//   qv = 8.0e-3, qc = 4.0e-3, qp = 2.0e-3, qsat = 1.0e-2, dtqsat = 0.02,
//   rho = 1.0, dt = 80.0.
// Why disabled:
//   Current development allows both evaporation paths to contribute from the
//   same original deficit.
// Enable trigger:
//   Production should re-evaluate subsaturation capacity after each phase
//   change or otherwise enforce a shared-deficit contract.
TEST(KesslerScalar, DISABLED_CloudAndRainEvaporationDoNotBothConsumeSameSubsaturation)
{
    const amrex::Real qv = amrex::Real(8.0e-3);
    const amrex::Real qc = amrex::Real(4.0e-3);
    const amrex::Real qp = amrex::Real(2.0e-3);
    const amrex::Real qsat = amrex::Real(1.0e-2);
    const amrex::Real dtqsat = amrex::Real(0.02);
    const amrex::Real rho = amrex::Real(1.0);
    const amrex::Real dt = amrex::Real(80.0);
    const KesslerSourceTerms actual = kessler_warm_rain_sources(
        qv, qc, qp, rho, amrex::Real(900.0), qsat, dtqsat, dt, true);
    const amrex::Real linearized_capacity = (qsat - qv) / (amrex::Real(1.0) + kSatAdjLatentOverCp * dtqsat);

    EXPECT_LE(actual.dq_cloud_to_vapor + actual.dq_rain_to_vapor,
              linearized_capacity + formula_abs_tol(linearized_capacity));
}

// Motivation: Expected behavior:
//   Saturation adjustment and theta update should use thermodynamically
//   consistent latent-heat-over-cp factors.
// Observed current behavior:
//   The saturation helper uses L_v / Cp_d while the theta update uses
//   lcond / sc.c_p.
// Minimal reproducer:
//   Compare the two factors with the default dry-air cp.
// Why disabled:
//   Current development intentionally preserves the existing factors.
// Enable trigger:
//   Production should adopt one thermodynamically consistent latent factor for
//   both the saturation and theta updates.
TEST(KesslerScalar, DISABLED_SaturationThetaUsesConsistentLatentHeatFactor)
{
    EXPECT_EQ(kSatAdjLatentOverCp, kThetaLatentOverCp);
}