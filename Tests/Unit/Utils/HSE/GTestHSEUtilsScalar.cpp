#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "GTestHSEUtilsCommon.H"

using namespace hse_test;

namespace {

void expect_near_scale (const Real actual, const Real expected, const Real tol)
{
    EXPECT_NEAR(actual, expected, tol);
}

} // namespace

// Motivation: HSE initialization relies on saturation pressure in Pa on both
// warm-water branches. This locks in that unit contract against independent
// reference values rather than reusing production formulas.
TEST(HSEUtils, SaturationPressureUnits)
{
    for (const SaturationReference& sample : saturation_reference_cases()) {
        SCOPED_TRACE(label("psat", sample.temperature));
        const Real default_pressure = HSEutils::compute_saturation_pressure(sample.temperature, false);
        const Real empirical_pressure = HSEutils::compute_saturation_pressure(sample.temperature, true);
        const Real tol = Real(0.08) * sample.expected_pa + host_device_tol(sample.expected_pa);

        EXPECT_GT(default_pressure, Real(0.0));
        EXPECT_GT(empirical_pressure, Real(0.0));
        expect_near_scale(default_pressure, sample.expected_pa, tol);
        expect_near_scale(empirical_pressure, sample.expected_pa, tol);
    }
}

// Motivation: The broad meteorological references above catch unit mistakes,
// while this tighter pointwise check anchors the water-saturation contract at
// well-known temperatures using an independent Magnus-style fit.
TEST(HSEUtils, SaturationPressureReferencePoints)
{
    for (const SaturationReference& sample : saturation_formula_reference_cases()) {
        SCOPED_TRACE(label("psat-tight", sample.temperature));
        const Real default_pressure = HSEutils::compute_saturation_pressure(sample.temperature, false);
        const Real empirical_pressure = HSEutils::compute_saturation_pressure(sample.temperature, true);
        const Real tol = saturation_formula_tol(sample.expected_pa);

        expect_near_scale(default_pressure, sample.expected_pa, tol);
        expect_near_scale(empirical_pressure, sample.expected_pa, tol);
    }
}

// Motivation: compute_theta is piecewise, so continuity and branch-local
// inverse/derivative contracts are more informative than a few golden values.
TEST(HSEUtils, ComputeThetaBranchContracts)
{
    const Real theta_0 = Real(300.0);
    const Real theta_tr = Real(320.0);
    const Real z_tr = Real(12000.0);
    const Real t_tr = Real(295.0);
    const Real left = HSEutils::compute_theta(Real(1.0), theta_0, theta_tr, z_tr, t_tr);
    const Real right = theta_profile(Real(1.0), theta_0, theta_tr, z_tr, t_tr);

    expect_near_scale(left, right, machine_precision_tol(scale_from_values({left, right})));

    for (const Real scaled_height : {Real(0.25), Real(0.75)}) {
        const Real theta = HSEutils::compute_theta(scaled_height, theta_0, theta_tr, z_tr, t_tr);
        const Real recovered = theta_profile_lower_inverse(theta, theta_0, theta_tr);
        const Real step = Real(32.0) * std::sqrt(std::numeric_limits<Real>::epsilon());
        const Real derivative = central_difference(
            [=] (const Real s) { return HSEutils::compute_theta(s, theta_0, theta_tr, z_tr, t_tr); },
            scaled_height, step);

        expect_near_scale(recovered, scaled_height, machine_precision_tol(scaled_height));
        expect_near_scale(derivative, theta_profile_lower_derivative(scaled_height, theta_0, theta_tr),
                          fd_truncation_tol(step, theta));
    }

    for (const Real scaled_height : {Real(1.25), Real(1.75)}) {
        const Real theta = HSEutils::compute_theta(scaled_height, theta_0, theta_tr, z_tr, t_tr);
        const Real recovered = theta_profile_upper_inverse(theta, theta_tr, z_tr, t_tr);
        const Real step = Real(32.0) * std::sqrt(std::numeric_limits<Real>::epsilon());
        const Real derivative = central_difference(
            [=] (const Real s) { return HSEutils::compute_theta(s, theta_0, theta_tr, z_tr, t_tr); },
            scaled_height, step);

        expect_near_scale(recovered, scaled_height, machine_precision_tol(scaled_height));
        expect_near_scale(derivative, theta_profile_upper_derivative(scaled_height, theta_0, theta_tr, z_tr, t_tr),
                          fd_truncation_tol(step, theta));
    }
}

// Motivation: compute_F switches between a first-cell half-step residual and
// an interior trapezoidal residual. This test verifies both branches directly.
TEST(HSEUtils, ComputeF_FirstAndInteriorResiduals)
{
    Real theta = Real(0.0);
    Real rho = Real(0.0);
    Real qv = Real(0.0);
    Real t_dp = Real(0.0);
    Real temperature = Real(0.0);
    const Real pressure = Real(8.0e4);
    const Real pressure_minus_1 = pressure + Real(600.0);
    const Real dz = Real(400.0);
    const Real q_t = kZone1QV;
    const Real eq_pot_temp = equivalent_potential_temperature_from_state(Real(288.0), pressure, kZone1QV, q_t);
    const bool use_empirical = false;
    const int which_zone = 1;
    const Real scaled_height = Real(0.5);
    const bool t_from_theta = true;
    const Real theta_0 = Real(300.0);
    const Real theta_tr = Real(312.0);
    const Real z_tr = Real(12000.0);
    const Real t_tr = Real(295.0);

    const Real first = HSEutils::compute_F(pressure, pressure_minus_1, theta, rho, qv, t_dp, temperature, dz,
                                           Real(0.0), q_t, eq_pot_temp, use_empirical, which_zone, scaled_height,
                                           t_from_theta, theta_0, theta_tr, z_tr, t_tr);
    const RhoState ref_state = compute_rho_reference(pressure, q_t, eq_pot_temp, use_empirical, which_zone,
                                                     scaled_height, t_from_theta, theta_0, theta_tr, z_tr, t_tr);
    const Real first_expected = pressure - pressure_minus_1 + Real(0.5) * ref_state.rho * kConstGrav * dz;
    const Real interior = HSEutils::compute_F(pressure, pressure_minus_1, theta, rho, qv, t_dp, temperature, dz,
                                              Real(1.1), q_t, eq_pot_temp, use_empirical, which_zone, scaled_height,
                                              t_from_theta, theta_0, theta_tr, z_tr, t_tr);
    const Real interior_expected = pressure - pressure_minus_1 +
                                   Real(0.5) * (ref_state.rho + Real(1.1)) * kConstGrav * dz;

    expect_near_scale(first, first_expected, residual_tol(scale_from_values({first, first_expected})));
    expect_near_scale(interior, interior_expected,
                      residual_tol(scale_from_values({interior, interior_expected})));
}

// Motivation: The Newton finite-difference perturbations must survive machine
// rounding. This is the minimal regression guard for the float-step bug.
TEST(HSEUtils, FDPerturbationChangesInput)
{
    const Real temperature = Real(200.0);
    const Real pressure = kP0;

    EXPECT_NE(temperature + fd_step_temperature(temperature), temperature);
    EXPECT_NE(pressure + fd_step_pressure(pressure), pressure);
}

// Motivation: This constructed-root test checks that compute_temperature solves
// the intended moist equivalent-potential-temperature equation, not just that
// its iterations terminate.
TEST(HSEUtils, ComputeTemperatureConstructedRoot)
{
    const Real pressure = Real(9.2e4);
    const Real q_t = kZone1QV;
    const Real temperature_star = Real(287.5);
    const bool use_empirical = true;
    const int which_zone = 1;
    const Real scaled_height = Real(0.4);
    const Real eq_pot_temp = equivalent_potential_temperature_from_profile(temperature_star, pressure, q_t,
                                                                           use_empirical, which_zone,
                                                                           scaled_height);
    const Real temperature = HSEutils::compute_temperature(pressure, q_t, eq_pot_temp, use_empirical,
                                                           which_zone, scaled_height);
    const Real scale = scale_from_values({temperature_star, temperature, Real(300.0)});
    const Real tol = Real(2.0) * fd_step_temperature(temperature_star) +
                     Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;

    expect_near_scale(temperature, temperature_star, tol);
}

// Motivation: Newton_Raphson_hse should recover a known moist pressure root
// when the residual is constructed independently from the EOS contract.
TEST(HSEUtils, NewtonRaphsonHSEKnownRootWithMoisture)
{
    const Real tolerance = machine_precision_tol(kP0);
    const Real dz = Real(5.0e4);
    const Real pressure_star = Real(8.0e4);
    const Real theta = Real(300.0);
    const Real qv = Real(0.01);
    const Real qt = Real(0.5);
    const Real rho_star = dry_density_from_theta_pressure_qv(theta, pressure_star, qv);
    const Real c = -pressure_star - Real(0.5) * rho_star * (Real(1.0) + qt) * kConstGrav * dz;
    Real pressure = Real(5.0e4);
    Real rho = dry_density_from_theta_pressure_qv(theta, pressure, qv);
    Real residual = pressure + Real(0.5) * rho * (Real(1.0) + qt) * kConstGrav * dz + c;

    HSEutils::Newton_Raphson_hse(tolerance, kRdOcp, dz, kConstGrav, c, theta, qt, qv, pressure, rho, residual);

    const Real scale = scale_from_values({pressure_star, rho_star * kConstGrav * dz});
    expect_near_scale(pressure, pressure_star, independent_root_tol(scale));
    EXPECT_NEAR(residual, Real(0.0), residual_tol(scale));
}

// Motivation: The moist Newton derivative should use density implied by the
// current pressure iterate, even when the caller passes a stale starting rho.
TEST(HSEUtils, NewtonRaphsonHSEIgnoresStaleInputDensity)
{
    const Real tolerance = machine_precision_tol(kP0);
    const Real dz = Real(5.0e4);
    const Real pressure_star = Real(8.0e4);
    const Real theta = Real(300.0);
    const Real qv = Real(0.01);
    const Real qt = Real(0.5);
    const Real rho_star = dry_density_from_theta_pressure_qv(theta, pressure_star, qv);
    const Real c = -pressure_star - Real(0.5) * rho_star * (Real(1.0) + qt) * kConstGrav * dz;
    Real pressure = Real(5.0e4);
    const Real rho_consistent = dry_density_from_theta_pressure_qv(theta, pressure, qv);
    Real rho = Real(0.35) * rho_consistent;
    Real residual = pressure + Real(0.5) * rho_consistent * (Real(1.0) + qt) * kConstGrav * dz + c;

    HSEutils::Newton_Raphson_hse(tolerance, kRdOcp, dz, kConstGrav, c, theta, qt, qv, pressure, rho, residual);

    const Real scale = scale_from_values({pressure_star, rho_star * kConstGrav * dz});
    expect_near_scale(pressure, pressure_star, independent_root_tol(scale));
    EXPECT_NEAR(residual, Real(0.0), residual_tol(scale));
}

// Motivation: The flat-column interior HSE solve should satisfy the discrete
// trapezoidal balance after initialization for a moderately stretched layer.
TEST(HSEUtils, IsentropicInteriorResidualModerateDz)
{
    constexpr int klo = 0;
    constexpr int khi = 1;
    constexpr Real dz = Real(1.2e4);
    std::array<Real, khi + 2> rho{};
    std::array<Real, khi + 2> pressure{};

    HSEutils::init_isentropic_hse(Real(0.8), Real(300.0), rho.data(), pressure.data(), dz, klo, khi);

    for (int k = 1; k <= khi; ++k) {
        const Real residual = pressure[k] - pressure[k - 1] +
                              Real(0.5) * dz * kConstGrav * (rho[k] + rho[k - 1]);
        const Real scale = scale_from_values({pressure[k], pressure[k - 1], rho[k] * kConstGrav * dz});
        EXPECT_NEAR(residual, Real(0.0), residual_tol(scale)) << "k=" << k;
    }
}

// Motivation: This is the stiffer companion to the moderate-dz case above and
// is the more sensitive regression check for the interior Jacobian factor.
TEST(HSEUtils, IsentropicInteriorResidualStiffDz)
{
    constexpr int klo = 0;
    constexpr int khi = 1;
    constexpr Real dz = Real(1.4e4);
    std::array<Real, khi + 2> rho{};
    std::array<Real, khi + 2> pressure{};

    HSEutils::init_isentropic_hse(Real(0.65), Real(300.0), rho.data(), pressure.data(), dz, klo, khi);

    for (int k = 1; k <= khi; ++k) {
        const Real residual = pressure[k] - pressure[k - 1] +
                              Real(0.5) * dz * kConstGrav * (rho[k] + rho[k - 1]);
        const Real scale = scale_from_values({pressure[k], pressure[k - 1], rho[k] * kConstGrav * dz});
        EXPECT_NEAR(residual, Real(0.0), residual_tol(scale)) << "k=" << k;
    }
}

// Motivation: Uniform terrain heights should reduce the terrain HSE path to
// the same equations as the flat-column path, including the top ghost state.
TEST(HSEUtils, TerrainUniformMatchesFlat)
{
    constexpr int klo = 0;
    constexpr int khi = 4;
    constexpr Real dz = Real(1500.0);
    std::array<Real, khi + 2> flat_rho{};
    std::array<Real, khi + 2> flat_pressure{};
    std::array<Real, khi + 2> terrain_rho{};
    std::array<Real, khi + 2> terrain_pressure{};

    amrex::Box box(amrex::IntVect(0, 0, 0), amrex::IntVect(0, 0, khi));
    amrex::BaseFab<Real> zfab(box, 1);
    auto zcc = zfab.array();
    for (int k = 0; k <= khi; ++k) {
        zcc(0, 0, k) = (Real(k) + Real(0.5)) * dz;
    }

    HSEutils::init_isentropic_hse(Real(1.18), Real(300.0), flat_rho.data(), flat_pressure.data(), dz, klo, khi);
    HSEutils::init_isentropic_hse_terrain(0, 0, Real(1.18), Real(300.0), terrain_rho.data(), terrain_pressure.data(),
                                          zfab.const_array(), klo, khi);

    for (int k = klo; k <= khi + 1; ++k) {
        const Real rho_scale = scale_from_values({flat_rho[k], terrain_rho[k], Real(1.0)});
        const Real pressure_scale = scale_from_values({flat_pressure[std::min(k, khi)],
                                                       terrain_pressure[std::min(k, khi)], Real(1.0)});
        EXPECT_NEAR(terrain_rho[k], flat_rho[k], machine_precision_tol(rho_scale)) << "k=" << k;
        if (k <= khi) {
            EXPECT_NEAR(terrain_pressure[k], flat_pressure[k], machine_precision_tol(pressure_scale)) << "k=" << k;
        }
    }
}

// Motivation: The terrain HSE path should preserve the same top ghost-fill
// contract as the flat-column path unless explicitly documented otherwise.
TEST(HSEUtils, TerrainGhostFill)
{
    constexpr int klo = 0;
    constexpr int khi = 3;
    constexpr Real dz = Real(1500.0);
    std::array<Real, khi + 2> rho{};
    std::array<Real, khi + 2> pressure{};
    rho[khi + 1] = Real(-999.0);

    amrex::Box box(amrex::IntVect(0, 0, 0), amrex::IntVect(0, 0, khi));
    amrex::BaseFab<Real> zfab(box, 1);
    auto zcc = zfab.array();
    for (int k = 0; k <= khi; ++k) {
        zcc(0, 0, k) = (Real(k) + Real(0.5)) * dz;
    }

    HSEutils::init_isentropic_hse_terrain(0, 0, Real(1.18), Real(300.0), rho.data(), pressure.data(),
                                          zfab.const_array(), klo, khi);

    EXPECT_EQ(rho[khi + 1], rho[khi]);
}

// Motivation: compute_p_k uses Newton iteration internally, so this test uses
// an independent bracketed solve to check the governing residual instead of
// duplicating the production iteration.
TEST(HSEUtils, ComputePKIndependentRoot)
{
    Real theta = Real(0.0);
    Real rho = Real(0.0);
    Real qv = Real(0.0);
    Real t_dp = Real(0.0);
    Real temperature = Real(0.0);
    const Real pressure_minus_1 = Real(8.1e4);
    Real pressure = pressure_minus_1 - Real(750.0);
    const Real dz = Real(500.0);
    const Real rho_minus_1 = Real(1.08);
    const Real q_t = kZone1QV;
    const Real scaled_height = Real(0.5);
    const Real theta_0 = Real(302.0);
    const Real theta_tr = Real(314.0);
    const Real z_tr = Real(12000.0);
    const Real t_tr = Real(295.0);
    const int which_zone = 1;
    const bool t_from_theta = true;
    const bool use_empirical = true;
    const Real theta_target = theta_profile(scaled_height, theta_0, theta_tr, z_tr, t_tr);
    const Real temperature_target = temperature_from_theta_pressure(theta_target, pressure);
    const Real eq_pot_temp = equivalent_potential_temperature_from_state(temperature_target, pressure, kZone1QV, q_t);
    const Real expected = compute_pk_reference(pressure, pressure_minus_1, dz, rho_minus_1, q_t, eq_pot_temp,
                                               use_empirical, which_zone, scaled_height, t_from_theta,
                                               theta_0, theta_tr, z_tr, t_tr);
    const Real actual = HSEutils::compute_p_k(pressure, pressure_minus_1, theta, rho, qv, t_dp, temperature,
                                              dz, rho_minus_1, q_t, eq_pot_temp, use_empirical, which_zone,
                                              scaled_height, t_from_theta, theta_0, theta_tr, z_tr, t_tr);
    const Real scale = scale_from_values({actual, expected, pressure_minus_1});

    expect_near_scale(actual, expected, independent_root_tol(scale));
}

// Motivation: compute_rho has two top-level branches. This verifies that each
// one is internally consistent with the corresponding independent EOS/profile
// closure for theta, temperature, moisture, and density.
TEST(HSEUtils, ComputeRhoBranchContracts)
{
    {
        Real theta = Real(0.0);
        Real rho = Real(0.0);
        Real qv = Real(0.0);
        Real t_dp = Real(0.0);
        Real temperature = Real(0.0);
        const Real pressure = Real(8.5e4);
        const Real q_t = kZone1QV;
        const Real eq_pot_temp = equivalent_potential_temperature_from_state(Real(286.0), pressure, kZone1QV, q_t);

        HSEutils::compute_rho(pressure, theta, rho, qv, t_dp, temperature, q_t, eq_pot_temp, true, 1, Real(0.5), true,
                              Real(301.0), Real(314.0), Real(12000.0), Real(295.0));
        const RhoState ref = compute_rho_reference(pressure, q_t, eq_pot_temp, true, 1, Real(0.5), true,
                                                   Real(301.0), Real(314.0), Real(12000.0), Real(295.0));

        EXPECT_NEAR(theta, ref.theta, machine_precision_tol(ref.theta));
        EXPECT_NEAR(temperature, ref.temperature, machine_precision_tol(ref.temperature));
        EXPECT_NEAR(qv, ref.qv, machine_precision_tol(ref.qv));
        EXPECT_NEAR(rho, ref.rho, machine_precision_tol(ref.rho));
    }

    {
        Real theta = Real(0.0);
        Real rho = Real(0.0);
        Real qv = Real(0.0);
        Real t_dp = Real(0.0);
        Real temperature = Real(0.0);
        const Real pressure = Real(9.1e4);
        const Real q_t = kZone1QV;
        const Real temperature_star = Real(287.0);
        const Real eq_pot_temp = equivalent_potential_temperature_from_profile(temperature_star, pressure, q_t,
                                               false, 1, Real(0.4));

        HSEutils::compute_rho(pressure, theta, rho, qv, t_dp, temperature, q_t, eq_pot_temp, false, 1, Real(0.4), false,
                              Real(301.0), Real(314.0), Real(12000.0), Real(295.0));
        const RhoState ref = compute_rho_reference(pressure, q_t, eq_pot_temp, false, 1, Real(0.4), false,
                                                   Real(301.0), Real(314.0), Real(12000.0), Real(295.0));

        EXPECT_NEAR(theta, ref.theta, machine_precision_tol(ref.theta));
        EXPECT_NEAR(temperature, ref.temperature, fd_truncation_tol(fd_step_temperature(ref.temperature), ref.temperature));
        EXPECT_NEAR(qv, ref.qv, machine_precision_tol(ref.qv));
        EXPECT_NEAR(rho, ref.rho, machine_precision_tol(ref.rho));
    }
}