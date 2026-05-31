#include <array>
#include <algorithm>

#include <gtest/gtest.h>

#include "ERF_GTestMorrisonCommon.H"

using namespace morrison_test;

namespace {

void expect_near_formula (const amrex::Real actual,
                          const amrex::Real expected)
{
    EXPECT_NEAR(actual, expected, formula_abs_tol(expected));
}

void expect_distribution_matches (const MorrisonDistributionParameters& actual,
                                  const amrex::Real expected_lambda,
                                  const amrex::Real expected_number,
                                  const amrex::Real expected_intercept)
{
    EXPECT_NEAR(actual.lambda, expected_lambda, backend_math_abs_tol(expected_lambda));
    EXPECT_NEAR(actual.number, expected_number, backend_math_abs_tol(expected_number));
    EXPECT_NEAR(actual.intercept, expected_intercept, backend_math_abs_tol(expected_intercept));
}

} // namespace

// Motivation: Morrison uses strict less-than qsmall cleanup. The below, equal,
// and above cases protect the branch threshold and the mass-number cleanup
// policy for all five hydrometeor moments.
TEST(MorrisonScalar, QSmallCleanup_BelowEqualAboveThreshold)
{
    const amrex::Real below = kQSmall * amrex::Real(0.5);
    const amrex::Real equal = kQSmall;
    const amrex::Real above = kQSmall * amrex::Real(2.0);

    MorrisonCellState below_state = make_state(amrex::Real(1.0e-2), below, below, below, below, below,
                                               amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0),
                                               amrex::Real(40.0), amrex::Real(50.0));
    MorrisonEffectiveRadii below_eff = make_effective_radii();
    const MorrisonQSmallCleanupDiagnostics below_diag =
        morrison_apply_qsmall_mass_number_cleanup(below_state, below_eff, kQSmall);

    EXPECT_EQ(below_diag.qc_zeroed, 1);
    EXPECT_EQ(below_diag.qi_zeroed, 1);
    EXPECT_EQ(below_diag.qr_zeroed, 1);
    EXPECT_EQ(below_diag.qs_zeroed, 1);
    EXPECT_EQ(below_diag.qg_zeroed, 1);
    EXPECT_NEAR(below_state.qc, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(below_state.qi, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(below_state.qr, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(below_state.qs, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(below_state.qg, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(below_state.nc + below_state.ni + below_state.nr + below_state.ns + below_state.ng,
                amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(below_eff.effc + below_eff.effi + below_eff.effr + below_eff.effs + below_eff.effg,
                amrex::Real(0.0), exact_zero_tol());

    for (const amrex::Real mass : {equal, above}) {
        SCOPED_TRACE("mass=" + std::to_string(static_cast<double>(mass)));
        MorrisonCellState state = make_state(amrex::Real(1.0e-2), mass, mass, mass, mass, mass,
                                             amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0),
                                             amrex::Real(40.0), amrex::Real(50.0));
        MorrisonEffectiveRadii eff = make_effective_radii();
        const MorrisonQSmallCleanupDiagnostics diag =
            morrison_apply_qsmall_mass_number_cleanup(state, eff, kQSmall);

        EXPECT_EQ(diag.qc_zeroed + diag.qi_zeroed + diag.qr_zeroed + diag.qs_zeroed + diag.qg_zeroed, 0);
        expect_near_formula(state.qc, mass);
        expect_near_formula(state.qi, mass);
        expect_near_formula(state.qr, mass);
        expect_near_formula(state.qs, mass);
        expect_near_formula(state.qg, mass);
        expect_near_formula(state.nc, amrex::Real(10.0));
        expect_near_formula(state.ni, amrex::Real(20.0));
        expect_near_formula(state.nr, amrex::Real(30.0));
        expect_near_formula(state.ns, amrex::Real(40.0));
        expect_near_formula(state.ng, amrex::Real(50.0));
    }
}

// Motivation: The subsaturated tiny-hydrometeor cleanup is a phase-change
// branch. It must conserve total water while applying the intended latent
// cooling sign for liquid evaporation and ice sublimation.
TEST(MorrisonScalar, SubsaturationCleanup_ConservesWaterAndCoolsByLatentCoefficient)
{
    MorrisonCellState state = make_state(amrex::Real(1.0e-2), amrex::Real(2.0e-9),
                                         amrex::Real(3.0e-9), amrex::Real(4.0e-9),
                                         amrex::Real(5.0e-9), amrex::Real(6.0e-9),
                                         amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0),
                                         amrex::Real(40.0), amrex::Real(50.0));
    const amrex::Real total_before = morrison_total_water_full(state);
    const amrex::Real temperature_before = state.temperature;
    const amrex::Real liquid_removed = state.qc + state.qr;
    const amrex::Real ice_removed = state.qi + state.qs + state.qg;

    morrison_apply_subsaturation_small_hydrometeor_cleanup(
        state, morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
        morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
        kLatentVaporization, kLatentSublimation, kCpm);

    EXPECT_NEAR(morrison_total_water_full(state), total_before, property_tol(6, total_before));
    EXPECT_LT(state.temperature, temperature_before);
    const amrex::Real latent_residual =
        kCpm * (state.temperature - temperature_before)
      + liquid_removed * kLatentVaporization
      + ice_removed * kLatentSublimation;
        EXPECT_NEAR(latent_residual, amrex::Real(0.0), latent_proxy_tol(6, temperature_before, kCpm));
}

// Motivation: Both saturation-ratio tests use strict less-than. Exactly at the
// production threshold must be a no-op, while just below must clean up eligible
// tiny hydrometeors.
TEST(MorrisonScalar, SubsaturationCleanup_ThresholdEqualityNoOp)
{
    MorrisonCellState at_threshold = make_state(amrex::Real(1.0e-2), amrex::Real(2.0e-9),
                                                amrex::Real(3.0e-9), amrex::Real(4.0e-9),
                                                amrex::Real(5.0e-9), amrex::Real(6.0e-9),
                                                amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0),
                                                amrex::Real(40.0), amrex::Real(50.0));
    const MorrisonCellState before = at_threshold;

    morrison_apply_subsaturation_small_hydrometeor_cleanup(
        at_threshold, morrison_subsaturation_ratio_threshold,
        morrison_subsaturation_ratio_threshold, kLatentVaporization, kLatentSublimation, kCpm);

    expect_near_formula(at_threshold.qv, before.qv);
    expect_near_formula(at_threshold.qc, before.qc);
    expect_near_formula(at_threshold.qi, before.qi);
    expect_near_formula(at_threshold.qr, before.qr);
    expect_near_formula(at_threshold.qs, before.qs);
    expect_near_formula(at_threshold.qg, before.qg);
    expect_near_formula(at_threshold.temperature, before.temperature);
}

// Motivation: Warm-branch tiny snow/graupel melt transfers mass and number to
// rain and applies fusion cooling. Equality at the threshold is intentionally
// inactive because production uses <, not <=.
TEST(MorrisonScalar, WarmSmallIceMelt_TransfersMassNumberAndAppliesFusionCooling)
{
    MorrisonCellState state = make_state(amrex::Real(1.0e-2), amrex::Real(0.0), amrex::Real(0.0),
                                         amrex::Real(2.0e-4), amrex::Real(2.0e-7),
                                         amrex::Real(3.0e-7), amrex::Real(0.0), amrex::Real(0.0),
                                         amrex::Real(100.0), amrex::Real(20.0), amrex::Real(30.0));
    const amrex::Real total_before = morrison_total_water_full(state);
    const amrex::Real temperature_before = state.temperature;
    const amrex::Real melted_mass = state.qs + state.qg;
    const amrex::Real expected_nr = state.nr + state.ns + state.ng;

    morrison_apply_warm_small_ice_melt_to_rain(state, kLatentFusion, kCpm);

    EXPECT_NEAR(morrison_total_water_full(state), total_before, property_tol(6, total_before));
    expect_near_formula(state.qs, amrex::Real(0.0));
    expect_near_formula(state.qg, amrex::Real(0.0));
    expect_near_formula(state.nr, expected_nr);
    expect_near_formula(state.ns, amrex::Real(0.0));
    expect_near_formula(state.ng, amrex::Real(0.0));
    const amrex::Real latent_residual = kCpm * (state.temperature - temperature_before)
                                      + melted_mass * kLatentFusion;
    EXPECT_NEAR(latent_residual, amrex::Real(0.0), latent_proxy_tol(3, temperature_before, kCpm));

    MorrisonCellState equality = make_state(amrex::Real(1.0e-2), amrex::Real(0.0), amrex::Real(0.0),
                                            amrex::Real(2.0e-4), morrison_warm_small_ice_melt_threshold,
                                            morrison_warm_small_ice_melt_threshold,
                                            amrex::Real(0.0), amrex::Real(0.0), amrex::Real(100.0),
                                            amrex::Real(20.0), amrex::Real(30.0));
    morrison_apply_warm_small_ice_melt_to_rain(equality, kLatentFusion, kCpm);
    expect_near_formula(equality.qs, morrison_warm_small_ice_melt_threshold);
    expect_near_formula(equality.qg, morrison_warm_small_ice_melt_threshold);
    expect_near_formula(equality.nr, amrex::Real(100.0));
}

// Motivation: Rain, snow, and graupel exponential PSD helpers enforce lambda
// bounds by changing number, not mass. These tests protect lower clamp, upper
// clamp, and smooth in-range formulas with independent mass-number values.
TEST(MorrisonScalar, ExponentialDistribution_EnforcesSlopeBoundsAndMassNumberConsistency)
{
    struct SpeciesCase {
        amrex::Real coefficient;
        amrex::Real lambda_min;
        amrex::Real lambda_max;
    };

    const std::array<SpeciesCase, 3> species_cases = {{{kRainCoefficient, kLamMinRain, kLamMaxRain},
                                                       {kSnowCoefficient, kLamMinSnow, kLamMaxSnow},
                                                       {kGraupelCoefficient, kLamMinGraupel, kLamMaxGraupel}}};
    const amrex::Real mass = amrex::Real(2.0e-4);

    for (const SpeciesCase& species : species_cases) {
        SCOPED_TRACE("coefficient=" + std::to_string(static_cast<double>(species.coefficient)));

        const amrex::Real in_range_lambda = amrex::Real(0.5) * (species.lambda_min + species.lambda_max);
        const amrex::Real in_range_number = number_for_lambda(mass, species.coefficient, in_range_lambda);
        const MorrisonDistributionParameters in_range = morrison_exponential_distribution_parameters(
            mass, in_range_number, species.coefficient, species.lambda_min, species.lambda_max, amrex::Real(3.0));
        expect_distribution_matches(in_range, in_range_lambda, in_range_number, in_range_number * in_range_lambda);
        EXPECT_EQ(in_range.limited_to_min, 0);
        EXPECT_EQ(in_range.limited_to_max, 0);

        const MorrisonDistributionParameters lower = morrison_exponential_distribution_parameters(
            mass, number_for_lambda(mass, species.coefficient, species.lambda_min * amrex::Real(0.25)),
            species.coefficient, species.lambda_min, species.lambda_max, amrex::Real(3.0));
        expect_distribution_matches(lower, species.lambda_min,
                                    expected_adjusted_number(mass, species.coefficient, species.lambda_min),
                                    expected_intercept(mass, species.coefficient, species.lambda_min));
        EXPECT_EQ(lower.limited_to_min, 1);
        EXPECT_EQ(lower.limited_to_max, 0);

        const MorrisonDistributionParameters upper = morrison_exponential_distribution_parameters(
            mass, number_for_lambda(mass, species.coefficient, species.lambda_max * amrex::Real(4.0)),
            species.coefficient, species.lambda_min, species.lambda_max, amrex::Real(3.0));
        expect_distribution_matches(upper, species.lambda_max,
                                    expected_adjusted_number(mass, species.coefficient, species.lambda_max),
                                    expected_intercept(mass, species.coefficient, species.lambda_max));
        EXPECT_EQ(upper.limited_to_min, 0);
        EXPECT_EQ(upper.limited_to_max, 1);
    }
}