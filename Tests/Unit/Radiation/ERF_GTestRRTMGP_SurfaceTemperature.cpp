#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <ERF_Constants.H>
#include <ERF_MicrophysicsUtils.H>
#include <ERF_RRTMGP_SurfaceTemperature.H>
#include <ERF_SurfaceTemperature.H>

namespace {

constexpr amrex::Real kSurfacePressure = amrex::Real(0.9) * p_0;
constexpr amrex::Real kSurfaceTheta = amrex::Real(300.0);
constexpr amrex::Real kDefaultTemperature = amrex::Real(280.0);

} // namespace

// Motivation: a surface value supplied in Kelvin must be normalized to the
// same theta representation used by MOST and reconstruct to the same physical
// temperature at a non-reference pressure. This catches both skipped and
// repeated Exner conversions without using the implementation as the oracle.
TEST(SurfaceTemperatureContract, AbsoluteTemperatureRoundTripsAtReducedPressure)
{
    const amrex::Real pressure = amrex::Real(0.9) * p_0;
    const amrex::Real absolute_temperature = amrex::Real(290.0);
    const amrex::Real expected_theta =
        absolute_temperature * std::pow(p_0 / pressure, RdoCp);
    const amrex::Real tolerance =
        amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon() *
        absolute_temperature;

    amrex::Real theta = -1.0;
    ASSERT_TRUE(erf_surface_temperature::absolute_to_theta(
        absolute_temperature, pressure, theta));
    EXPECT_NEAR(theta, expected_theta,
                amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon() *
                expected_theta);

    amrex::Real reconstructed_temperature = -1.0;
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        theta, pressure, reconstructed_temperature));
    EXPECT_NEAR(reconstructed_temperature, absolute_temperature, tolerance);
}

// Motivation: WRFInput and wrflowinp normalize SST/TSK to theta before the
// SurfaceLayer sees them. The radiation consumer must convert that theta once,
// rather than treating it as Kelvin and applying another normalization.
TEST(SurfaceTemperatureContract, WrfInputThetaIsConvertedExactlyOnce)
{
    const amrex::Real pressure = amrex::Real(0.9) * p_0;
    const amrex::Real expected_temperature = amrex::Real(290.0);
    const amrex::Real theta_from_wrfinput =
        expected_temperature * std::pow(p_0 / pressure, RdoCp);
    const amrex::Real tolerance =
        amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon() *
        expected_temperature;

    amrex::Real reconstructed_temperature = -1.0;
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        theta_from_wrfinput, pressure, reconstructed_temperature));
    EXPECT_NEAR(reconstructed_temperature, expected_temperature, tolerance);
}

// Motivation: WRFInput-style values are already theta, whereas Metgrid,
// text-file SST, and coupled SST arrive in Kelvin. Each absolute producer
// must be normalized once so all three source families retain the same
// physical temperature at the radiation and saturation consumers.
TEST(SurfaceTemperatureContract, AbsoluteProducersAreNormalizedOnce)
{
    const amrex::Real pressure = amrex::Real(0.9) * p_0;
    const amrex::Real expected_temperature = amrex::Real(290.0);
    const amrex::Real expected_theta =
        expected_temperature * std::pow(p_0 / pressure, RdoCp);
    const amrex::Real tolerance =
        amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon();

    amrex::Real metgrid_theta = -1.0;
    amrex::Real text_sst_theta = -1.0;
    amrex::Real coupled_sst_theta = -1.0;
    ASSERT_TRUE(erf_surface_temperature::absolute_to_theta(
        expected_temperature, pressure, metgrid_theta));
    ASSERT_TRUE(erf_surface_temperature::absolute_to_theta(
        expected_temperature, pressure, text_sst_theta));
    ASSERT_TRUE(erf_surface_temperature::absolute_to_theta(
        expected_temperature, pressure, coupled_sst_theta));

    EXPECT_NEAR(metgrid_theta, expected_theta, tolerance * expected_theta);
    EXPECT_NEAR(text_sst_theta, expected_theta, tolerance * expected_theta);
    EXPECT_NEAR(coupled_sst_theta, expected_theta, tolerance * expected_theta);

    amrex::Real metgrid_temperature = -1.0;
    amrex::Real text_temperature = -1.0;
    amrex::Real coupled_temperature = -1.0;
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        metgrid_theta, pressure, metgrid_temperature));
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        text_sst_theta, pressure, text_temperature));
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        coupled_sst_theta, pressure, coupled_temperature));
    EXPECT_NEAR(metgrid_temperature, expected_temperature, tolerance * expected_temperature);
    EXPECT_NEAR(text_temperature, expected_temperature, tolerance * expected_temperature);
    EXPECT_NEAR(coupled_temperature, expected_temperature, tolerance * expected_temperature);
}

// Motivation: SLM exposes a field named theta, while Noah-MP exposes t_sfc in
// absolute Kelvin. The shared surface contract must preserve that distinction:
// SLM theta is reconstructed for absolute-temperature consumers, but a valid
// Noah-MP value passes through without an Exner conversion.
TEST(SurfaceTemperatureContract, SlmThetaAndNoahMpTemperatureUseDeclaredContracts)
{
    const amrex::Real pressure = amrex::Real(0.9) * p_0;
    const amrex::Real slm_absolute_temperature = amrex::Real(290.0);
    const amrex::Real slm_theta =
        slm_absolute_temperature * std::pow(p_0 / pressure, RdoCp);
    const amrex::Real noah_temperature = amrex::Real(282.0);
    const amrex::Real tolerance =
        amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon();

    amrex::Real reconstructed_slm_temperature = -1.0;
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        slm_theta, pressure, reconstructed_slm_temperature));
    EXPECT_NEAR(reconstructed_slm_temperature, slm_absolute_temperature,
                tolerance * slm_absolute_temperature);

    amrex::Real unchanged_noah_temperature = -1.0;
    amrex::Real lsm_t_sfc = noah_temperature;
    rrtmgp::resolve_surface_temperature(
        true, true, true, noah_temperature,
        true, slm_theta, pressure, kDefaultTemperature,
        unchanged_noah_temperature, &lsm_t_sfc);
    EXPECT_EQ(unchanged_noah_temperature, noah_temperature);
    EXPECT_EQ(lsm_t_sfc, noah_temperature);
}

// Motivation: coupled SST coverage is per cell. The uncovered cell must keep
// its existing theta source while the covered cell is normalized from Kelvin;
// a single provenance flag for the whole surface field cannot express this.
TEST(SurfaceTemperatureContract, CoupledSstPartialCoverageUsesPerCellConversion)
{
    const amrex::Real pressure = amrex::Real(0.9) * p_0;
    const amrex::Real covered_temperature = amrex::Real(290.0);
    const amrex::Real uncovered_temperature = amrex::Real(286.0);
    const amrex::Real covered_theta_expected =
        covered_temperature * std::pow(p_0 / pressure, RdoCp);
    const amrex::Real uncovered_theta =
        uncovered_temperature * std::pow(p_0 / pressure, RdoCp);
    const amrex::Real tolerance =
        amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon();

    amrex::Real covered_theta = -1.0;
    ASSERT_TRUE(erf_surface_temperature::absolute_to_theta(
        covered_temperature, pressure, covered_theta));
    EXPECT_NEAR(covered_theta, covered_theta_expected,
                tolerance * covered_theta_expected);
    EXPECT_NEAR(uncovered_theta,
                uncovered_temperature * std::pow(p_0 / pressure, RdoCp),
                tolerance * uncovered_theta);

    amrex::Real covered_reconstructed = -1.0;
    amrex::Real uncovered_reconstructed = -1.0;
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        covered_theta, pressure, covered_reconstructed));
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        uncovered_theta, pressure, uncovered_reconstructed));
    EXPECT_NEAR(covered_reconstructed, covered_temperature,
                tolerance * covered_temperature);
    EXPECT_NEAR(uncovered_reconstructed, uncovered_temperature,
                tolerance * uncovered_temperature);
}

// Motivation: saturation vapor pressure is a temperature-sensitive consumer
// of SurfaceLayer data. Passing theta directly would produce a different
// humidity boundary at reduced pressure than passing the reconstructed Kelvin
// temperature.
TEST(SurfaceTemperatureContract, SaturationHumidityReceivesAbsoluteTemperature)
{
    const amrex::Real pressure = amrex::Real(0.9) * p_0;
    const amrex::Real temperature = amrex::Real(290.0);
    const amrex::Real theta = temperature * std::pow(p_0 / pressure, RdoCp);
    amrex::Real reconstructed_temperature = -1.0;
    ASSERT_TRUE(erf_surface_temperature::theta_to_absolute(
        theta, pressure, reconstructed_temperature));

    amrex::Real expected_qsat = -1.0;
    amrex::Real actual_qsat = -1.0;
    erf_qsatw(temperature, pressure * amrex::Real(0.01), expected_qsat);
    erf_qsatw(reconstructed_temperature, pressure * amrex::Real(0.01), actual_qsat);
    EXPECT_NEAR(actual_qsat, expected_qsat,
                amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon() * expected_qsat);

    amrex::Real theta_qsat = -1.0;
    erf_qsatw(theta, pressure * amrex::Real(0.01), theta_qsat);
    EXPECT_GT(std::abs(theta_qsat - expected_qsat), amrex::Real(1.0e-4));
}

// Motivation: a failed Exner conversion must not mutate a caller's value or
// persist NaN/Inf to an LSM fallback. The atmospheric column remains invalid,
// so the radiation resolver uses its already-validated absolute default.
TEST(SurfaceTemperatureContract, InvalidPressureDoesNotPersistConvertedTemperature)
{
    amrex::Real theta = amrex::Real(301.0);
    EXPECT_FALSE(erf_surface_temperature::absolute_to_theta(
        amrex::Real(290.0), amrex::Real(0.0), theta));
    EXPECT_EQ(theta, amrex::Real(301.0));
    EXPECT_FALSE(erf_surface_temperature::absolute_to_theta(
        amrex::Real(290.0), -amrex::Real(1.0), theta));
    EXPECT_EQ(theta, amrex::Real(301.0));

    amrex::Real t_sfc = -1.0;
    amrex::Real lsm_t_sfc = lsm_undefined;
    rrtmgp::resolve_surface_temperature(
        true, true, false, amrex::Real(0.0), true, amrex::Real(300.0),
        std::numeric_limits<amrex::Real>::quiet_NaN(), kDefaultTemperature,
        t_sfc, &lsm_t_sfc);
    EXPECT_EQ(t_sfc, kDefaultTemperature);
    EXPECT_EQ(lsm_t_sfc, lsm_undefined);
}

// Motivation: SurfaceLayer stores potential temperature, but RRTMGP's
// surface boundary inputs are absolute temperature. At a pressure below p0,
// using theta directly changes both the surface emission and the bottom
// layer boundary. The fallback and its LSM writeback must use the same
// Exner-converted temperature.
TEST(RRTMGP_SurfaceTemperature, SurfaceLayerFallbackConvertsThetaAndWritesLsmAbsoluteTemperature)
{
    const amrex::Real expected_temperature =
        kSurfaceTheta * std::pow(kSurfacePressure / p_0, RdoCp);
    const amrex::Real expected_tolerance =
        amrex::Real(64.0) * std::numeric_limits<amrex::Real>::epsilon() *
        expected_temperature;
    ASSERT_NE(expected_temperature, kSurfaceTheta);

    amrex::Real t_sfc = -1.0;
    amrex::Real lsm_t_sfc = lsm_undefined;
    rrtmgp::resolve_surface_temperature(
        true, true, false, lsm_t_sfc,
        true, kSurfaceTheta, kSurfacePressure, kDefaultTemperature,
        t_sfc, &lsm_t_sfc);

    EXPECT_NEAR(t_sfc, expected_temperature, expected_tolerance);
    EXPECT_NEAR(lsm_t_sfc, expected_temperature, expected_tolerance);
}

// Motivation: water columns do not use an available LSM surface temperature
// in this fallback chain. SurfaceLayer theta must take precedence, be
// converted at the physical-surface pressure, and write back absolute temperature
// to the available LSM field.
TEST(RRTMGP_SurfaceTemperature, SurfaceLayerFallbackTakesPrecedenceOverWaterLsmTemperature)
{
    const amrex::Real expected_temperature =
        kSurfaceTheta * std::pow(kSurfacePressure / p_0, RdoCp);
    const amrex::Real valid_lsm_temperature = amrex::Real(282.0);
    const amrex::Real expected_tolerance =
        amrex::Real(64.0) * std::numeric_limits<amrex::Real>::epsilon() *
        expected_temperature;

    amrex::Real t_sfc = -1.0;
    amrex::Real lsm_t_sfc = valid_lsm_temperature;
    rrtmgp::resolve_surface_temperature(
        false, true, true, valid_lsm_temperature,
        true, kSurfaceTheta, kSurfacePressure, kDefaultTemperature,
        t_sfc, &lsm_t_sfc);

    EXPECT_NEAR(t_sfc, expected_temperature, expected_tolerance);
    EXPECT_NE(t_sfc, valid_lsm_temperature);
    EXPECT_NEAR(lsm_t_sfc, expected_temperature, expected_tolerance);
}

// Motivation: LSM t_sfc is already an absolute-temperature contract. The
// SurfaceLayer conversion must be limited to the fallback source, so a valid
// LSM value must pass through unchanged even when SurfaceLayer is available.
TEST(RRTMGP_SurfaceTemperature, ValidLsmTemperatureIsNotExnerConverted)
{
    const amrex::Real valid_lsm_temperature = amrex::Real(282.0);
    amrex::Real t_sfc = -1.0;
    amrex::Real lsm_t_sfc = valid_lsm_temperature;
    rrtmgp::resolve_surface_temperature(
        true, true, true, valid_lsm_temperature,
        true, kSurfaceTheta, kSurfacePressure, kDefaultTemperature,
        t_sfc, &lsm_t_sfc);

    EXPECT_EQ(t_sfc, valid_lsm_temperature);
    EXPECT_EQ(lsm_t_sfc, valid_lsm_temperature);
}

// Motivation: the default RRTMGP surface temperature is already absolute
// temperature. It must remain unchanged when no SurfaceLayer or valid LSM
// value supplies the boundary condition.
TEST(RRTMGP_SurfaceTemperature, DefaultTemperatureIsAlreadyAbsolute)
{
    amrex::Real t_sfc = -1.0;
    rrtmgp::resolve_surface_temperature(
        true, false, false, 0.0,
        false, 0.0, kSurfacePressure, kDefaultTemperature,
        t_sfc, nullptr);

    EXPECT_EQ(t_sfc, kDefaultTemperature);
}
