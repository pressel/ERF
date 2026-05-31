#include <array>

#include <gtest/gtest.h>

#include "ERF_GTestMorrisonCommon.H"

using namespace morrison_test;

// Motivation: Morrison full-ice and no-ice modes use different total-water
// definitions. This test records the verified species mapping from the current
// ERF Morrison state: qv, qc, qi, qr, qs, qg for full mode; qv, qc, qr for
// Morrison_NoIce.
TEST(MorrisonPhysicalProperties, TotalWaterDefinitionsSeparateFullAndNoIceModes)
{
    const MorrisonCellState state = make_state(amrex::Real(1.0e-2), amrex::Real(1.0e-4),
                                               amrex::Real(2.0e-4), amrex::Real(3.0e-4),
                                               amrex::Real(4.0e-4), amrex::Real(5.0e-4),
                                               amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0),
                                               amrex::Real(40.0), amrex::Real(50.0));
    EXPECT_NEAR(morrison_total_water_full(state), amrex::Real(1.15e-2), formula_abs_tol(amrex::Real(1.15e-2)));
    EXPECT_NEAR(morrison_total_water_no_ice(state), amrex::Real(1.04e-2), formula_abs_tol(amrex::Real(1.04e-2)));
}

// Motivation: Autoconversion and accretion are local cloud-to-rain transfers.
// The extracted helper contract is that the mass tendencies are exactly paired:
// cloud water lost by these terms appears as rain before other processes act.
TEST(MorrisonPhysicalProperties, WarmRainSourcesConserveLocalLiquidMass)
{
    const MorrisonAutoconversionRates autoconversion = morrison_compute_warm_rain_autoconversion(
        amrex::Real(1.0e-4), amrex::Real(8.0e7), amrex::Real(1.1), amrex::Real(1.0), kCons29);
    const MorrisonAccretionRates accretion = morrison_compute_cloud_rain_accretion(
        amrex::Real(1.0e-4), amrex::Real(2.0e-4), amrex::Real(8.0e7));

    const amrex::Real qc_tendency = -autoconversion.prc - accretion.pra;
    const amrex::Real qr_tendency = autoconversion.prc + accretion.pra;
    EXPECT_NEAR(qc_tendency + qr_tendency, amrex::Real(0.0), exact_zero_tol());
    EXPECT_LE(qc_tendency, amrex::Real(0.0));
    EXPECT_GE(qr_tendency, amrex::Real(0.0));
}

// Motivation: qsmall cleanup is a mass-number consistency limiter. Whenever a
// mass species is removed, the corresponding number concentration must be
// removed too; species at or above threshold must remain untouched.
TEST(MorrisonPhysicalProperties, QSmallCleanupPreservesMassNumberConsistencyPolicy)
{
    MorrisonCellState state = make_state(amrex::Real(1.0e-2), kQSmall * amrex::Real(0.25),
                                         kQSmall, kQSmall * amrex::Real(2.0),
                                         kQSmall * amrex::Real(0.25), kQSmall,
                                         amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0),
                                         amrex::Real(40.0), amrex::Real(50.0));
    MorrisonEffectiveRadii effective_radii = make_effective_radii();
    morrison_apply_qsmall_mass_number_cleanup(state, effective_radii, kQSmall);

    EXPECT_NEAR(state.qc, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(state.nc, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(state.qs, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(state.ns, amrex::Real(0.0), exact_zero_tol());
    EXPECT_NEAR(state.qi, kQSmall, formula_abs_tol(kQSmall));
    EXPECT_NEAR(state.ni, amrex::Real(20.0), formula_abs_tol(amrex::Real(20.0)));
    EXPECT_NEAR(state.qr, kQSmall * amrex::Real(2.0), formula_abs_tol(kQSmall));
    EXPECT_NEAR(state.nr, amrex::Real(30.0), formula_abs_tol(amrex::Real(30.0)));
    EXPECT_NEAR(state.qg, kQSmall, formula_abs_tol(kQSmall));
    EXPECT_NEAR(state.ng, amrex::Real(50.0), formula_abs_tol(amrex::Real(50.0)));
}

// Motivation: The extracted local cleanup and warm tiny-ice melt helpers must
// not create negative valid species. This protects the physical-property
// contract separately from exact formula checks.
TEST(MorrisonPhysicalProperties, ExtractedHelpersDoNotCreateNegativeMassOrNumber)
{
    std::array<MorrisonCellState, 4> states = {{
        make_state(amrex::Real(1.0e-2), kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5),
                   kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5),
                   amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
        make_state(amrex::Real(1.0e-2), amrex::Real(2.0e-9), amrex::Real(3.0e-9),
                   amrex::Real(4.0e-9), amrex::Real(5.0e-9), amrex::Real(6.0e-9),
                   amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
        make_state(amrex::Real(1.0e-2), amrex::Real(1.0e-4), amrex::Real(0.0),
                   amrex::Real(2.0e-4), amrex::Real(2.0e-7), amrex::Real(3.0e-7),
                   amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
        make_state(amrex::Real(1.0e-2), kQSmall, kQSmall, kQSmall,
                   morrison_warm_small_ice_melt_threshold, morrison_warm_small_ice_melt_threshold,
                   amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0))
    }};

    for (MorrisonCellState& state : states) {
        SCOPED_TRACE(state_label(state));
        MorrisonEffectiveRadii effective_radii = make_effective_radii();
        morrison_apply_subsaturation_small_hydrometeor_cleanup(
            state, morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
            morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
            kLatentVaporization, kLatentSublimation, kCpm);
        morrison_apply_qsmall_mass_number_cleanup(state, effective_radii, kQSmall);
        morrison_apply_warm_small_ice_melt_to_rain(state, kLatentFusion, kCpm);

        EXPECT_GE(state.qv, -exact_zero_tol());
        EXPECT_GE(state.qc, -exact_zero_tol());
        EXPECT_GE(state.qi, -exact_zero_tol());
        EXPECT_GE(state.qr, -exact_zero_tol());
        EXPECT_GE(state.qs, -exact_zero_tol());
        EXPECT_GE(state.qg, -exact_zero_tol());
        EXPECT_GE(state.nc, -exact_zero_tol());
        EXPECT_GE(state.ni, -exact_zero_tol());
        EXPECT_GE(state.nr, -exact_zero_tol());
        EXPECT_GE(state.ns, -exact_zero_tol());
        EXPECT_GE(state.ng, -exact_zero_tol());
    }
}

// Motivation: Slope limiting changes number concentration to preserve the mass
// with a bounded exponential distribution. This is a double-moment contract:
// the helper must not assert number conservation, but it must maintain the
// mass-number relationship implied by the bounded lambda.
TEST(MorrisonPhysicalProperties, ExponentialDistributionMaintainsMassNumberRelationshipUnderLimits)
{
    const amrex::Real mass = amrex::Real(2.0e-4);
    const MorrisonDistributionParameters rain_min = morrison_exponential_distribution_parameters(
        mass, number_for_lambda(mass, kRainCoefficient, kLamMinRain * amrex::Real(0.5)),
        kRainCoefficient, kLamMinRain, kLamMaxRain);
    const MorrisonDistributionParameters rain_max = morrison_exponential_distribution_parameters(
        mass, number_for_lambda(mass, kRainCoefficient, kLamMaxRain * amrex::Real(2.0)),
        kRainCoefficient, kLamMinRain, kLamMaxRain);

    for (const MorrisonDistributionParameters& params : {rain_min, rain_max}) {
        SCOPED_TRACE("lambda=" + std::to_string(static_cast<double>(params.lambda)));
        const amrex::Real reconstructed_mass = params.number * kRainCoefficient /
            (params.lambda * params.lambda * params.lambda);
        EXPECT_NEAR(reconstructed_mass, mass, backend_math_abs_tol(mass));
        EXPECT_GE(params.lambda, kLamMinRain - backend_math_abs_tol(kLamMinRain));
        EXPECT_LE(params.lambda, kLamMaxRain + backend_math_abs_tol(kLamMaxRain));
        EXPECT_GE(params.number, -exact_zero_tol());
        EXPECT_GE(params.intercept, -exact_zero_tol());
    }
}

// Motivation: Sedimentation is conservative inside a column: summing density
// content changes over all levels leaves only the boundary fluxes. Morrison's
// top boundary has zero incoming flux, so the column loss equals bottom fallout.
TEST(MorrisonPhysicalProperties, SedimentationBudgetTelescopesToBottomFallout)
{
    const std::array<amrex::Real, 3> fallout_to_below = {{amrex::Real(2.0e-5),
                                                          amrex::Real(4.0e-5),
                                                          amrex::Real(7.0e-5)}};
    const amrex::Real dz = amrex::Real(80.0);
    const amrex::Real rho = amrex::Real(1.0);
    const amrex::Real dt = amrex::Real(2.5);
    const int nstep = 2;

    amrex::Real column_integrated_density_content_delta = amrex::Real(0);
    for (int k = static_cast<int>(fallout_to_below.size()) - 1; k >= 0; --k) {
        const amrex::Real fallout_from_above = (k + 1 < static_cast<int>(fallout_to_below.size()))
            ? fallout_to_below[static_cast<std::size_t>(k + 1)] : amrex::Real(0);
        const MorrisonSedimentationBudget budget = morrison_sedimentation_budget(
            fallout_from_above, fallout_to_below[static_cast<std::size_t>(k)], dz, rho, dt, nstep);
        column_integrated_density_content_delta += budget.density_content_delta * dz;
        EXPECT_NEAR(budget.mixing_ratio_tendency * rho * dt,
                    budget.density_content_delta,
                    backend_math_abs_tol(budget.density_content_delta));
    }

    const amrex::Real expected_column_loss = -fallout_to_below[0] * dt / static_cast<amrex::Real>(nstep);
    EXPECT_NEAR(column_integrated_density_content_delta, expected_column_loss,
                property_tol(static_cast<int>(fallout_to_below.size()), expected_column_loss));
}

// Motivation: Surface accumulators must use the same bottom fallout fluxes as
// sedimentation. This protects the rain/snow/graupel partitioning formulas.
TEST(MorrisonPhysicalProperties, SurfacePrecipitationIncrementUsesBottomFluxPartition)
{
    const amrex::Real rain = amrex::Real(3.0e-5);
    const amrex::Real cloud_water = amrex::Real(2.0e-6);
    const amrex::Real snow = amrex::Real(4.0e-6);
    const amrex::Real cloud_ice = amrex::Real(5.0e-6);
    const amrex::Real graupel = amrex::Real(6.0e-6);
    const amrex::Real dt = amrex::Real(5.0);
    const int nstep = 4;
    const amrex::Real step_dt = dt / static_cast<amrex::Real>(nstep);

    const MorrisonSurfacePrecipitationIncrement increment = morrison_surface_precipitation_increment(
        rain, cloud_water, snow, cloud_ice, graupel, dt, nstep);

    EXPECT_NEAR(increment.precipitation, (rain + cloud_water + snow + cloud_ice + graupel) * step_dt,
                formula_abs_tol(increment.precipitation));
    EXPECT_NEAR(increment.snow, (snow + cloud_ice + graupel) * step_dt,
                formula_abs_tol(increment.snow));
    EXPECT_NEAR(increment.snow_plus_ice, (cloud_ice + snow) * step_dt,
                formula_abs_tol(increment.snow_plus_ice));
    EXPECT_NEAR(increment.graupel, graupel * step_dt, formula_abs_tol(increment.graupel));
    EXPECT_LE(increment.snow_plus_ice, increment.snow + property_tol(2, increment.snow));
    EXPECT_LE(increment.graupel, increment.snow + property_tol(2, increment.snow));
}