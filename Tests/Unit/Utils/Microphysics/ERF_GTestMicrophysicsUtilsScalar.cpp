#include <array>

#include <gtest/gtest.h>

#include "ERF_GTestMicrophysicsCommon.H"

using namespace microphysics_test;

namespace {

void expect_near_relative (const amrex::Real actual,
                           const amrex::Real expected,
                           const amrex::Real factor = kValueRelTol)
{
    EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, factor));
}

} // namespace

// Motivation: Morrison and SAM only call this gamma wrapper with positive
// arguments. The test locks in that positive-only contract and key identities.
TEST(MicrophysicsGamma, PositiveIdentities)
{
    expect_near_relative(erf_gammafff(amrex::Real(1.0)), amrex::Real(1.0));
    expect_near_relative(erf_gammafff(amrex::Real(0.5)), kSqrtPi);
    expect_near_relative(erf_gammafff(amrex::Real(2.5)), amrex::Real(1.5) * erf_gammafff(amrex::Real(1.5)));
}

#if GTEST_HAS_DEATH_TEST
// Motivation: std::lgamma returns log(abs(Gamma(x))) for negative non-integers,
// so the wrapper must reject that unsupported branch explicitly.
TEST(MicrophysicsGamma, RejectsNegativeArgument)
{
    EXPECT_DEATH(
        {
            volatile amrex::Real value = erf_gammafff(-amrex::Real(0.5));
            (void)value;
        },
        "");
}
#endif

// Motivation: The Clausius-Clapeyron fallback is the cold and hot water-vapor
// pressure closure. Its derivative should match both the analytic formula and
// a finite-difference check.
TEST(MicrophysicsWaterCC, DerivativeMatchesAnalyticAndFiniteDifference)
{
    const std::array<amrex::Real, 4> temperatures = {
        amrex::Real(190.0), amrex::Real(240.0), amrex::Real(300.0), amrex::Real(350.0)};

    for (const amrex::Real temperature : temperatures) {
        const amrex::Real esat = erf_esatw_cc(temperature);
        const amrex::Real derivative = erf_dtesatw_cc(temperature);
        const amrex::Real expected = cc_derivative_formula(temperature);
        const amrex::Real finite_difference =
            central_difference([] (const amrex::Real value) { return erf_esatw_cc(value); }, temperature);

        EXPECT_GT(esat, amrex::Real(0.0));
        EXPECT_GT(derivative, amrex::Real(0.0));
        expect_near_relative(derivative, expected, kDerivativeRelTol);
        expect_near_relative(derivative, finite_difference, kDerivativeRelTol);
    }
}

// Motivation: The empirical branch is selectable at runtime. It must report the
// same mbar units as the default branch or the HSE initialization uses the
// wrong vapor pressure by a factor of 100.
TEST(MicrophysicsWaterSaturation, EmpiricalBranchUsesMbarUnits)
{
    expect_near_relative(erf_esatw(amrex::Real(273.15), true), kEmpiricalTriplePointMbar);

    const std::array<amrex::Real, 3> temperatures = {
        amrex::Real(273.15), amrex::Real(300.0), amrex::Real(343.15)};

    for (const amrex::Real temperature : temperatures) {
        const amrex::Real empirical = erf_esatw(temperature, true);
        const amrex::Real default_fit = erf_esatw(temperature);

        EXPECT_GT(empirical, amrex::Real(0.0));
        expect_near_relative(empirical, default_fit, kEmpiricalAgreementRelTol);
    }
}

// Motivation: The old low-temperature polynomial produced negative vapor
// pressures near 188 K. These regression points must stay positive.
TEST(MicrophysicsWaterSaturation, RegressionCasesStayPositive)
{
    const std::array<amrex::Real, 4> regression_temperatures = {
        amrex::Real(188.16), amrex::Real(188.17), amrex::Real(189.0), amrex::Real(189.17)};

    for (const amrex::Real temperature : regression_temperatures) {
        SCOPED_TRACE(std::to_string(temperature));
        EXPECT_GT(erf_esatw(temperature), amrex::Real(0.0));
    }
}

// Motivation: The water saturation helper should never go negative anywhere in
// the supported cold-to-warm range used by the current branch logic.
TEST(MicrophysicsWaterSaturation, NoNegativeValuesOnRepresentativeGrid)
{
    for (int index = 0; index <= 64; ++index) {
        const amrex::Real temperature = amrex::Real(188.16) + amrex::Real(2.5) * index;
        EXPECT_GT(erf_esatw(temperature), amrex::Real(0.0));
    }
}

// Motivation: The low and high water-branch switches do not need to be exactly
// continuous, but they must avoid the large jump created by the old cold fit.
TEST(MicrophysicsWaterSaturation, SwitchJumpsStayBounded)
{
    const amrex::Real low_below = erf_esatw(kWaterLowerSwitchK - amrex::Real(1.0e-3));
    const amrex::Real low_above = erf_esatw(kWaterLowerSwitchK + amrex::Real(1.0e-3));
    const amrex::Real high_below = erf_esatw(kWaterUpperSwitchK - amrex::Real(1.0e-3));
    const amrex::Real high_above = erf_esatw(kWaterUpperSwitchK + amrex::Real(1.0e-3));

    EXPECT_NEAR(low_above, low_below, scaled_tol(low_above, low_below, kSwitchRelTol));
    EXPECT_NEAR(high_above, high_below, scaled_tol(high_above, high_below, kSwitchRelTol));
}

// Motivation: The water saturation helper and its derivative must represent the
// same piecewise function away from the explicit branch transitions.
TEST(MicrophysicsWaterSaturation, DerivativeMatchesFiniteDifferenceAwayFromSwitches)
{
    const std::array<amrex::Real, 4> temperatures = {
        amrex::Real(190.0), amrex::Real(210.0), amrex::Real(300.0), amrex::Real(345.0)};
    amrex::Real previous = -amrex::Real(1.0);

    for (const amrex::Real temperature : temperatures) {
        const amrex::Real esat = erf_esatw(temperature);
        const amrex::Real derivative = erf_dtesatw(temperature);
        const amrex::Real finite_difference =
            central_difference([] (const amrex::Real value) { return erf_esatw(value); }, temperature);

        EXPECT_GT(esat, amrex::Real(0.0));
        EXPECT_GT(derivative, amrex::Real(0.0));
        expect_near_relative(derivative, finite_difference, kDerivativeRelTol);

        if (previous > amrex::Real(0.0)) {
            EXPECT_GT(esat, previous);
        }
        previous = esat;
    }
}

// Motivation: Ice saturation intentionally clamps to zero above freezing.
// These checks document that preserved discontinuity explicitly.
TEST(MicrophysicsIceSaturation, PreservesAboveFreezingClamp)
{
    expect_near_relative(erf_esati(amrex::Real(273.16)), amrex::Real(6.11147274));
    EXPECT_EQ(erf_esati(amrex::Real(273.160001)), amrex::Real(0.0));
    expect_near_relative(erf_dtesati(amrex::Real(273.16)), amrex::Real(0.503223089));
    EXPECT_EQ(erf_dtesati(amrex::Real(273.160001)), amrex::Real(0.0));
}

// Motivation: The ice value and derivative polynomials should stay positive and
// agree with finite differences over the documented derivative domain.
TEST(MicrophysicsIceSaturation, DerivativeMatchesFiniteDifferenceWithinContract)
{
    const std::array<amrex::Real, 4> temperatures = {
        amrex::Real(188.1601), amrex::Real(210.0), amrex::Real(240.0), amrex::Real(260.0)};
    amrex::Real previous = -amrex::Real(1.0);

    for (const amrex::Real temperature : temperatures) {
        const amrex::Real esat = erf_esati(temperature);
        const amrex::Real derivative = erf_dtesati(temperature);

        EXPECT_GT(esat, amrex::Real(0.0));
        EXPECT_GT(derivative, amrex::Real(0.0));

        if (temperature > amrex::Real(188.16)) {
            const amrex::Real finite_difference =
                central_difference([] (const amrex::Real value) { return erf_esati(value); }, temperature);
            expect_near_relative(derivative, finite_difference, kDerivativeRelTol);
        }

        if (previous > amrex::Real(0.0)) {
            EXPECT_GT(esat, previous);
        }
        previous = esat;
    }
}

#if GTEST_HAS_DEATH_TEST
// Motivation: The derivative polynomial has a narrower cold contract than the
// value polynomial. The assert documents that contract instead of leaving it
// implicit in a comment/code mismatch.
TEST(MicrophysicsIceSaturation, RejectsTemperaturesBelowDerivativeContract)
{
    EXPECT_DEATH(
        {
            volatile amrex::Real value = erf_dtesati(amrex::Real(188.15));
            (void)value;
        },
        "");
}
#endif

// Motivation: The qsat helpers should differentiate the actual capped
// implementation, not the uncapped algebraic form.
TEST(MicrophysicsQSat, WaterNormalBranchMatchesFiniteDifference)
{
    const std::array<std::pair<amrex::Real, amrex::Real>, 2> states = {{
        {amrex::Real(260.0), amrex::Real(800.0)},
        {amrex::Real(300.0), amrex::Real(800.0)}
    }};

    for (const auto& state : states) {
        const amrex::Real temperature = state.first;
        const amrex::Real pressure = state.second;
        const amrex::Real esat = erf_esatw(temperature);
        SCOPED_TRACE(tp_label("water-normal", temperature, pressure));
        ASSERT_TRUE(uses_normal_qsat_branch(esat, pressure));

        amrex::Real qsat;
        amrex::Real dqsat;
        erf_qsatw(temperature, pressure, qsat);
        erf_dtqsatw(temperature, pressure, dqsat);

        const amrex::Real expected = Rd_on_Rv * erf_dtesatw(temperature) * pressure /
                                     ((pressure - esat) * (pressure - esat));
        const amrex::Real finite_difference = central_difference(
            [pressure] (const amrex::Real value) {
                amrex::Real qsat_local;
                erf_qsatw(value, pressure, qsat_local);
                return qsat_local;
            },
            temperature);

        EXPECT_GE(qsat, amrex::Real(0.0));
        EXPECT_LE(qsat, Rd_on_Rv);
        EXPECT_GT(dqsat, amrex::Real(0.0));
        expect_near_relative(dqsat, expected, kDerivativeRelTol);
        expect_near_relative(dqsat, finite_difference, kDerivativeRelTol);
    }
}

// Motivation: Once the qsat cap is active, qsat should be fixed at Rd_on_Rv and
// its temperature derivative should collapse to zero.
TEST(MicrophysicsQSat, WaterCappedBranchIsConstant)
{
    const amrex::Real temperature = amrex::Real(300.0);
    const amrex::Real pressure = capped_branch_pressure(erf_esatw(temperature));

    amrex::Real qsat;
    amrex::Real dqsat;
    erf_qsatw(temperature, pressure, qsat);
    erf_dtqsatw(temperature, pressure, dqsat);

    expect_near_relative(qsat, Rd_on_Rv);
    expect_near_relative(dqsat, amrex::Real(0.0));
}

// Motivation: Ice qsat uses the same capped formula and must preserve the same
// normal-branch and capped-branch contracts.
TEST(MicrophysicsQSat, IceNormalAndCappedBranchesAreConsistent)
{
    const amrex::Real temperature = amrex::Real(260.0);
    const amrex::Real normal_pressure = amrex::Real(800.0);
    const amrex::Real capped_pressure = microphysics_test::capped_branch_pressure(erf_esati(temperature));

    amrex::Real qsat_normal;
    amrex::Real dqsat_normal;
    amrex::Real qsat_capped;
    amrex::Real dqsat_capped;
    erf_qsati(temperature, normal_pressure, qsat_normal);
    erf_dtqsati(temperature, normal_pressure, dqsat_normal);
    erf_qsati(temperature, capped_pressure, qsat_capped);
    erf_dtqsati(temperature, capped_pressure, dqsat_capped);

    const amrex::Real esat = erf_esati(temperature);
    ASSERT_TRUE(uses_normal_qsat_branch(esat, normal_pressure));
    EXPECT_GT(dqsat_normal, amrex::Real(0.0));
    EXPECT_GE(qsat_normal, amrex::Real(0.0));
    EXPECT_LE(qsat_normal, Rd_on_Rv);
    expect_near_relative(qsat_capped, Rd_on_Rv);
    expect_near_relative(dqsat_capped, amrex::Real(0.0));
}

// Motivation: These are physical property checks rather than regression values:
// qsat should stay bounded, increase with temperature, and decrease with
// pressure while it remains on the uncapped branch.
TEST(MicrophysicsQSat, BoundsAndMonotonicityHoldOnNormalBranch)
{
    const std::array<amrex::Real, 4> temperatures = {
        amrex::Real(240.0), amrex::Real(260.0), amrex::Real(280.0), amrex::Real(300.0)};
    const std::array<amrex::Real, 3> pressures = {
        amrex::Real(500.0), amrex::Real(800.0), amrex::Real(1000.0)};

    amrex::Real previous_water = -amrex::Real(1.0);
    amrex::Real previous_ice = -amrex::Real(1.0);

    for (const amrex::Real temperature : temperatures) {
        amrex::Real qsat_water;
        amrex::Real qsat_ice;
        erf_qsatw(temperature, amrex::Real(800.0), qsat_water);
        erf_qsati(std::min(temperature, amrex::Real(260.0)), amrex::Real(800.0), qsat_ice);

        EXPECT_GE(qsat_water, amrex::Real(0.0));
        EXPECT_LE(qsat_water, Rd_on_Rv);

        if (temperature <= amrex::Real(260.0)) {
            EXPECT_GE(qsat_ice, amrex::Real(0.0));
            EXPECT_LE(qsat_ice, Rd_on_Rv);
            if (previous_ice > amrex::Real(0.0)) {
                EXPECT_GT(qsat_ice, previous_ice);
            }
            previous_ice = qsat_ice;
        }

        if (previous_water > amrex::Real(0.0)) {
            EXPECT_GT(qsat_water, previous_water);
        }
        previous_water = qsat_water;
    }

    for (const amrex::Real temperature : {amrex::Real(260.0), amrex::Real(300.0)}) {
        amrex::Real previous = -amrex::Real(1.0);

        for (const amrex::Real pressure : pressures) {
            amrex::Real qsat;
            erf_qsatw(temperature, pressure, qsat);

            if (previous > amrex::Real(0.0)) {
                EXPECT_LT(qsat, previous);
            }
            previous = qsat;
        }
    }
}

// Motivation: The empirical branch is only useful if the downstream HSE helper
// converts both water-pressure fits from mbar to Pa the same way.
TEST(MicrophysicsHSE, SaturationPressureUsesMbarToPaForBothBranches)
{
    const std::array<amrex::Real, 2> temperatures = {
        amrex::Real(273.15), amrex::Real(300.0)};

    for (const amrex::Real temperature : temperatures) {
        expect_near_relative(HSEutils::compute_saturation_pressure(temperature, false),
                             amrex::Real(100.0) * erf_esatw(temperature, false));
        expect_near_relative(HSEutils::compute_saturation_pressure(temperature, true),
                             amrex::Real(100.0) * erf_esatw(temperature, true));
    }

    expect_near_relative(HSEutils::compute_saturation_pressure(amrex::Real(273.15), true),
                         kEmpiricalTriplePointPa);
}