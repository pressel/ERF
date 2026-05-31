#include <vector>

#include <gtest/gtest.h>

#include "ERF_GTestMorrisonCommon.H"

using namespace morrison_test;

namespace {

struct KernelCase {
    MorrisonCellState state;
    amrex::Real qvqvs;
    amrex::Real qvqvsi;
    amrex::Real autoconversion_qc;
    amrex::Real autoconversion_nc;
    amrex::Real autoconversion_rho;
    amrex::Real autoconversion_dt;
    amrex::Real accretion_qc;
    amrex::Real accretion_qr;
    amrex::Real accretion_nc;
    amrex::Real limiter_qc;
    amrex::Real limiter_dt;
    amrex::Real limiter_prc;
    amrex::Real limiter_pra;
    amrex::Real sedimentation_fallout_from_above;
    amrex::Real sedimentation_fallout_to_below;
    amrex::Real sedimentation_dz;
    amrex::Real sedimentation_rho;
    amrex::Real sedimentation_dt;
    int sedimentation_nstep;
    amrex::Real surface_fallout_rain;
    amrex::Real surface_fallout_cloud_water;
    amrex::Real surface_fallout_snow;
    amrex::Real surface_fallout_cloud_ice;
    amrex::Real surface_fallout_graupel;
    amrex::Real surface_dt;
    int surface_nstep;
    amrex::Real distribution_mass;
    amrex::Real distribution_number;
    amrex::Real coefficient;
    amrex::Real lambda_min;
    amrex::Real lambda_max;
};

struct KernelOutputs {
    MorrisonCellState state;
    MorrisonEffectiveRadii effective_radii;
    MorrisonQSmallCleanupDiagnostics cleanup;
    amrex::Real total_water_full;
    amrex::Real total_water_no_ice;
    MorrisonAutoconversionRates autoconversion;
    MorrisonAccretionRates accretion;
    MorrisonCloudWaterLimiterDiagnostics limiter;
    amrex::Real limited_prc;
    amrex::Real limited_pra;
    MorrisonSedimentationBudget sedimentation;
    MorrisonSurfacePrecipitationIncrement surface;
    MorrisonDistributionParameters distribution;
};

void launch_morrison_helper_kernel (const int ncases,
                                    const KernelCase* cases_ptr,
                                    KernelOutputs* outputs_ptr)
{
    amrex::ParallelFor(ncases, [=] AMREX_GPU_DEVICE (int idx) noexcept {
        MorrisonCellState state = cases_ptr[idx].state;
        MorrisonEffectiveRadii effective_radii = make_effective_radii();

        morrison_apply_subsaturation_small_hydrometeor_cleanup(
            state, cases_ptr[idx].qvqvs, cases_ptr[idx].qvqvsi,
            kLatentVaporization, kLatentSublimation, kCpm);
        const MorrisonQSmallCleanupDiagnostics cleanup =
            morrison_apply_qsmall_mass_number_cleanup(state, effective_radii, kQSmall);
        morrison_apply_warm_small_ice_melt_to_rain(state, kLatentFusion, kCpm);
        const amrex::Real total_water_full = morrison_total_water_full(state);
        const amrex::Real total_water_no_ice = morrison_total_water_no_ice(state);
        const MorrisonAutoconversionRates autoconversion = morrison_compute_warm_rain_autoconversion(
            cases_ptr[idx].autoconversion_qc, cases_ptr[idx].autoconversion_nc,
            cases_ptr[idx].autoconversion_rho, cases_ptr[idx].autoconversion_dt, kCons29);
        const MorrisonAccretionRates accretion = morrison_compute_cloud_rain_accretion(
            cases_ptr[idx].accretion_qc, cases_ptr[idx].accretion_qr, cases_ptr[idx].accretion_nc);
        amrex::Real limited_prc = cases_ptr[idx].limiter_prc;
        amrex::Real limited_pra = cases_ptr[idx].limiter_pra;
        const MorrisonCloudWaterLimiterDiagnostics limiter = morrison_apply_cloud_water_sink_limiter(
            cases_ptr[idx].limiter_qc, cases_ptr[idx].limiter_dt, kQSmall, limited_prc, limited_pra);
        const MorrisonSedimentationBudget sedimentation = morrison_sedimentation_budget(
            cases_ptr[idx].sedimentation_fallout_from_above,
            cases_ptr[idx].sedimentation_fallout_to_below,
            cases_ptr[idx].sedimentation_dz, cases_ptr[idx].sedimentation_rho,
            cases_ptr[idx].sedimentation_dt, cases_ptr[idx].sedimentation_nstep);
        const MorrisonSurfacePrecipitationIncrement surface = morrison_surface_precipitation_increment(
            cases_ptr[idx].surface_fallout_rain, cases_ptr[idx].surface_fallout_cloud_water,
            cases_ptr[idx].surface_fallout_snow, cases_ptr[idx].surface_fallout_cloud_ice,
            cases_ptr[idx].surface_fallout_graupel, cases_ptr[idx].surface_dt, cases_ptr[idx].surface_nstep);
        const MorrisonDistributionParameters distribution = morrison_exponential_distribution_parameters(
            cases_ptr[idx].distribution_mass, cases_ptr[idx].distribution_number,
            cases_ptr[idx].coefficient, cases_ptr[idx].lambda_min, cases_ptr[idx].lambda_max);

        outputs_ptr[idx] = KernelOutputs{state, effective_radii, cleanup, total_water_full, total_water_no_ice,
                                         autoconversion, accretion, limiter, limited_prc, limited_pra,
                                         sedimentation, surface, distribution};
    });
    morrison_test::sync();
}

KernelCase make_kernel_case (const MorrisonCellState& state,
                             const amrex::Real qvqvs,
                             const amrex::Real qvqvsi,
                             const amrex::Real autoconversion_qc,
                             const amrex::Real autoconversion_nc,
                             const amrex::Real autoconversion_rho,
                             const amrex::Real autoconversion_dt,
                             const amrex::Real accretion_qc,
                             const amrex::Real accretion_qr,
                             const amrex::Real accretion_nc,
                             const amrex::Real limiter_qc,
                             const amrex::Real limiter_dt,
                             const amrex::Real limiter_prc,
                             const amrex::Real limiter_pra,
                             const amrex::Real sedimentation_fallout_from_above,
                             const amrex::Real sedimentation_fallout_to_below,
                             const amrex::Real sedimentation_dz,
                             const amrex::Real sedimentation_rho,
                             const amrex::Real sedimentation_dt,
                             const int sedimentation_nstep,
                             const amrex::Real surface_fallout_rain,
                             const amrex::Real surface_fallout_cloud_water,
                             const amrex::Real surface_fallout_snow,
                             const amrex::Real surface_fallout_cloud_ice,
                             const amrex::Real surface_fallout_graupel,
                             const amrex::Real surface_dt,
                             const int surface_nstep,
                             const amrex::Real distribution_mass,
                             const amrex::Real distribution_number,
                             const amrex::Real coefficient,
                             const amrex::Real lambda_min,
                             const amrex::Real lambda_max)
{
    return KernelCase{state, qvqvs, qvqvsi,
                      autoconversion_qc, autoconversion_nc, autoconversion_rho, autoconversion_dt,
                      accretion_qc, accretion_qr, accretion_nc,
                      limiter_qc, limiter_dt, limiter_prc, limiter_pra,
                      sedimentation_fallout_from_above, sedimentation_fallout_to_below,
                      sedimentation_dz, sedimentation_rho, sedimentation_dt, sedimentation_nstep,
                      surface_fallout_rain, surface_fallout_cloud_water, surface_fallout_snow,
                      surface_fallout_cloud_ice, surface_fallout_graupel, surface_dt, surface_nstep,
                      distribution_mass, distribution_number, coefficient, lambda_min, lambda_max};
}

std::vector<KernelCase> make_kernel_cases ()
{
    const amrex::Real mass = amrex::Real(2.0e-4);
    return {
        make_kernel_case(
            make_state(amrex::Real(1.0e-2), kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5),
                       kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5),
                       amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
            morrison_subsaturation_ratio_threshold, morrison_subsaturation_ratio_threshold,
            morrison_autoconversion_cloud_water_threshold * amrex::Real(0.5), amrex::Real(1.0e8), amrex::Real(1.1), amrex::Real(2.0),
            morrison_cloud_rain_accretion_threshold * amrex::Real(0.5), amrex::Real(2.0e-4), amrex::Real(8.0e7),
            amrex::Real(3.0e-4), amrex::Real(10.0), amrex::Real(1.0e-7), amrex::Real(1.0e-7),
            amrex::Real(0.0), amrex::Real(7.0e-5), amrex::Real(80.0), amrex::Real(1.0), amrex::Real(2.5), 2,
            amrex::Real(3.0e-5), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(5.0), 4,
            mass, number_for_lambda(mass, kRainCoefficient, amrex::Real(0.5) * (kLamMinRain + kLamMaxRain)),
            kRainCoefficient, kLamMinRain, kLamMaxRain),
        make_kernel_case(
            make_state(amrex::Real(1.0e-2), amrex::Real(2.0e-9), amrex::Real(3.0e-9),
                       amrex::Real(4.0e-9), amrex::Real(5.0e-9), amrex::Real(6.0e-9),
                       amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
            morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
            morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
            amrex::Real(1.2e-6), amrex::Real(1.0e8), amrex::Real(1.1), amrex::Real(2.0),
            amrex::Real(1.0e-4), amrex::Real(2.0e-4), amrex::Real(8.0e7),
            amrex::Real(3.0e-4), amrex::Real(10.0), amrex::Real(7.0e-5), amrex::Real(5.0e-5),
            amrex::Real(9.0e-5), amrex::Real(4.0e-5), amrex::Real(80.0), amrex::Real(1.0), amrex::Real(2.5), 2,
            amrex::Real(3.0e-5), amrex::Real(2.0e-6), amrex::Real(4.0e-6), amrex::Real(5.0e-6), amrex::Real(6.0e-6), amrex::Real(5.0), 4,
            mass, number_for_lambda(mass, kSnowCoefficient, kLamMinSnow * amrex::Real(0.25)),
            kSnowCoefficient, kLamMinSnow, kLamMaxSnow),
        make_kernel_case(
            make_state(amrex::Real(1.0e-2), amrex::Real(1.0e-4), amrex::Real(0.0),
                       amrex::Real(2.0e-4), amrex::Real(2.0e-7), amrex::Real(3.0e-7),
                       amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
            morrison_subsaturation_ratio_threshold, morrison_subsaturation_ratio_threshold,
            amrex::Real(2.0e-3), amrex::Real(1.0e6), amrex::Real(1.1), amrex::Real(1000.0),
            amrex::Real(1.0e-4), morrison_cloud_rain_accretion_threshold * amrex::Real(0.5), amrex::Real(8.0e7),
            amrex::Real(3.0e-4), amrex::Real(10.0), amrex::Real(7.0e-5), amrex::Real(5.0e-5),
            amrex::Real(9.0e-5), amrex::Real(4.0e-5), amrex::Real(120.0), amrex::Real(0.8), amrex::Real(3.0), 3,
            amrex::Real(0.0), amrex::Real(0.0), amrex::Real(4.0e-6), amrex::Real(5.0e-6), amrex::Real(6.0e-6), amrex::Real(6.0), 3,
            mass, number_for_lambda(mass, kGraupelCoefficient, kLamMaxGraupel * amrex::Real(4.0)),
            kGraupelCoefficient, kLamMinGraupel, kLamMaxGraupel),
        make_kernel_case(
            make_state(amrex::Real(1.0e-2), kQSmall, kQSmall, kQSmall, morrison_warm_small_ice_melt_threshold,
                       morrison_warm_small_ice_melt_threshold, amrex::Real(10.0), amrex::Real(20.0),
                       amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
            morrison_subsaturation_ratio_threshold, morrison_subsaturation_ratio_threshold,
            amrex::Real(1.2e-6), amrex::Real(1.0e8), amrex::Real(1.1), amrex::Real(2.0),
            amrex::Real(1.0e-4), amrex::Real(2.0e-4), amrex::Real(8.0e7),
            amrex::Real(3.0e-4), amrex::Real(10.0), amrex::Real(1.0e-7), amrex::Real(1.0e-7),
            amrex::Real(0.0), amrex::Real(7.0e-5), amrex::Real(80.0), amrex::Real(1.0), amrex::Real(2.5), 2,
            amrex::Real(0.0), amrex::Real(2.0e-6), amrex::Real(0.0), amrex::Real(5.0e-6), amrex::Real(0.0), amrex::Real(6.0), 3,
            mass, number_for_lambda(mass, kRainCoefficient, kLamMinRain),
            kRainCoefficient, kLamMinRain, kLamMaxRain)
    };
}

KernelOutputs host_reference (const KernelCase& test_case)
{
    MorrisonCellState state = test_case.state;
    MorrisonEffectiveRadii effective_radii = make_effective_radii();
    morrison_apply_subsaturation_small_hydrometeor_cleanup(
        state, test_case.qvqvs, test_case.qvqvsi, kLatentVaporization, kLatentSublimation, kCpm);
    const MorrisonQSmallCleanupDiagnostics cleanup =
        morrison_apply_qsmall_mass_number_cleanup(state, effective_radii, kQSmall);
    morrison_apply_warm_small_ice_melt_to_rain(state, kLatentFusion, kCpm);
    const amrex::Real total_water_full = morrison_total_water_full(state);
    const amrex::Real total_water_no_ice = morrison_total_water_no_ice(state);
    const MorrisonAutoconversionRates autoconversion = morrison_compute_warm_rain_autoconversion(
        test_case.autoconversion_qc, test_case.autoconversion_nc, test_case.autoconversion_rho,
        test_case.autoconversion_dt, kCons29);
    const MorrisonAccretionRates accretion = morrison_compute_cloud_rain_accretion(
        test_case.accretion_qc, test_case.accretion_qr, test_case.accretion_nc);
    amrex::Real limited_prc = test_case.limiter_prc;
    amrex::Real limited_pra = test_case.limiter_pra;
    const MorrisonCloudWaterLimiterDiagnostics limiter = morrison_apply_cloud_water_sink_limiter(
        test_case.limiter_qc, test_case.limiter_dt, kQSmall, limited_prc, limited_pra);
    const MorrisonSedimentationBudget sedimentation = morrison_sedimentation_budget(
        test_case.sedimentation_fallout_from_above, test_case.sedimentation_fallout_to_below,
        test_case.sedimentation_dz, test_case.sedimentation_rho, test_case.sedimentation_dt,
        test_case.sedimentation_nstep);
    const MorrisonSurfacePrecipitationIncrement surface = morrison_surface_precipitation_increment(
        test_case.surface_fallout_rain, test_case.surface_fallout_cloud_water,
        test_case.surface_fallout_snow, test_case.surface_fallout_cloud_ice,
        test_case.surface_fallout_graupel, test_case.surface_dt, test_case.surface_nstep);
    const MorrisonDistributionParameters distribution = morrison_exponential_distribution_parameters(
        test_case.distribution_mass, test_case.distribution_number, test_case.coefficient,
        test_case.lambda_min, test_case.lambda_max);
    return KernelOutputs{state, effective_radii, cleanup, total_water_full, total_water_no_ice,
                         autoconversion, accretion, limiter, limited_prc, limited_pra,
                         sedimentation, surface, distribution};
}

void expect_state_near (const MorrisonCellState& actual,
                        const MorrisonCellState& expected)
{
    EXPECT_NEAR(actual.temperature, expected.temperature, backend_math_abs_tol(expected.temperature));
    EXPECT_NEAR(actual.qv, expected.qv, formula_abs_tol(expected.qv));
    EXPECT_NEAR(actual.qc, expected.qc, formula_abs_tol(expected.qc));
    EXPECT_NEAR(actual.qi, expected.qi, formula_abs_tol(expected.qi));
    EXPECT_NEAR(actual.qr, expected.qr, formula_abs_tol(expected.qr));
    EXPECT_NEAR(actual.qs, expected.qs, formula_abs_tol(expected.qs));
    EXPECT_NEAR(actual.qg, expected.qg, formula_abs_tol(expected.qg));
    EXPECT_NEAR(actual.nc, expected.nc, formula_abs_tol(expected.nc));
    EXPECT_NEAR(actual.ni, expected.ni, formula_abs_tol(expected.ni));
    EXPECT_NEAR(actual.nr, expected.nr, formula_abs_tol(expected.nr));
    EXPECT_NEAR(actual.ns, expected.ns, formula_abs_tol(expected.ns));
    EXPECT_NEAR(actual.ng, expected.ng, formula_abs_tol(expected.ng));
}

void expect_autoconversion_near (const MorrisonAutoconversionRates& actual,
                                 const MorrisonAutoconversionRates& expected)
{
    EXPECT_NEAR(actual.prc, expected.prc, backend_math_abs_tol(expected.prc));
    EXPECT_NEAR(actual.nprc, expected.nprc, backend_math_abs_tol(expected.nprc));
    EXPECT_NEAR(actual.nprc1, expected.nprc1, backend_math_abs_tol(expected.nprc1));
    EXPECT_EQ(actual.active, expected.active);
    EXPECT_EQ(actual.nprc_limited, expected.nprc_limited);
    EXPECT_EQ(actual.nprc1_limited, expected.nprc1_limited);
}

void expect_accretion_near (const MorrisonAccretionRates& actual,
                            const MorrisonAccretionRates& expected)
{
    EXPECT_NEAR(actual.pra, expected.pra, backend_math_abs_tol(expected.pra));
    EXPECT_NEAR(actual.npra, expected.npra, backend_math_abs_tol(expected.npra));
    EXPECT_EQ(actual.active, expected.active);
}

void expect_limiter_near (const MorrisonCloudWaterLimiterDiagnostics& actual,
                          const MorrisonCloudWaterLimiterDiagnostics& expected,
                          const amrex::Real actual_prc,
                          const amrex::Real expected_prc,
                          const amrex::Real actual_pra,
                          const amrex::Real expected_pra)
{
    EXPECT_NEAR(actual.ratio, expected.ratio, formula_abs_tol(expected.ratio));
    EXPECT_EQ(actual.limited, expected.limited);
    EXPECT_NEAR(actual_prc, expected_prc, formula_abs_tol(expected_prc));
    EXPECT_NEAR(actual_pra, expected_pra, formula_abs_tol(expected_pra));
}

void expect_sedimentation_budget_near (const MorrisonSedimentationBudget& actual,
                                       const MorrisonSedimentationBudget& expected)
{
    EXPECT_NEAR(actual.flux_divergence, expected.flux_divergence,
                formula_abs_tol(expected.flux_divergence));
    EXPECT_NEAR(actual.mixing_ratio_tendency, expected.mixing_ratio_tendency,
                formula_abs_tol(expected.mixing_ratio_tendency));
    EXPECT_NEAR(actual.density_content_delta, expected.density_content_delta,
                formula_abs_tol(expected.density_content_delta));
}

void expect_surface_increment_near (const MorrisonSurfacePrecipitationIncrement& actual,
                                    const MorrisonSurfacePrecipitationIncrement& expected)
{
    EXPECT_NEAR(actual.precipitation, expected.precipitation, formula_abs_tol(expected.precipitation));
    EXPECT_NEAR(actual.snow, expected.snow, formula_abs_tol(expected.snow));
    EXPECT_NEAR(actual.snow_plus_ice, expected.snow_plus_ice, formula_abs_tol(expected.snow_plus_ice));
    EXPECT_NEAR(actual.graupel, expected.graupel, formula_abs_tol(expected.graupel));
}

} // namespace

// Motivation: The extracted Morrison helpers are called from AMReX kernels in
// production. This test compares host/device parity for the total-water,
// cleanup/melt, warm-rain source, cloud-water limiter, sedimentation-budget,
// surface-increment, and PSD helpers. Formula correctness is covered by the
// scalar and physical-property tests.
TEST(MorrisonKernel, HostDeviceHelperEquivalenceCoversExtractedBranches)
{
    const std::vector<KernelCase> cases = make_kernel_cases();
    std::vector<KernelOutputs> host_outputs(cases.size());
    for (int idx = 0; idx < static_cast<int>(cases.size()); ++idx) {
        host_outputs[idx] = host_reference(cases[idx]);
    }

    amrex::Gpu::DeviceVector<KernelCase> d_cases(cases.size());
    amrex::Gpu::DeviceVector<KernelOutputs> d_outputs(cases.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, cases.begin(), cases.end(), d_cases.begin());
    launch_morrison_helper_kernel(static_cast<int>(cases.size()), d_cases.data(), d_outputs.data());

    std::vector<KernelOutputs> device_outputs(cases.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_outputs.begin(), d_outputs.end(), device_outputs.begin());

    for (int idx = 0; idx < static_cast<int>(cases.size()); ++idx) {
        SCOPED_TRACE(state_label(cases[idx].state));
        expect_state_near(device_outputs[idx].state, host_outputs[idx].state);
        EXPECT_NEAR(device_outputs[idx].total_water_full, host_outputs[idx].total_water_full,
                    formula_abs_tol(host_outputs[idx].total_water_full));
        EXPECT_NEAR(device_outputs[idx].total_water_no_ice, host_outputs[idx].total_water_no_ice,
                    formula_abs_tol(host_outputs[idx].total_water_no_ice));
        EXPECT_EQ(device_outputs[idx].cleanup.qc_zeroed, host_outputs[idx].cleanup.qc_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qi_zeroed, host_outputs[idx].cleanup.qi_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qr_zeroed, host_outputs[idx].cleanup.qr_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qs_zeroed, host_outputs[idx].cleanup.qs_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qg_zeroed, host_outputs[idx].cleanup.qg_zeroed);
        expect_autoconversion_near(device_outputs[idx].autoconversion, host_outputs[idx].autoconversion);
        expect_accretion_near(device_outputs[idx].accretion, host_outputs[idx].accretion);
        expect_limiter_near(device_outputs[idx].limiter, host_outputs[idx].limiter,
                            device_outputs[idx].limited_prc, host_outputs[idx].limited_prc,
                            device_outputs[idx].limited_pra, host_outputs[idx].limited_pra);
        expect_sedimentation_budget_near(device_outputs[idx].sedimentation, host_outputs[idx].sedimentation);
        expect_surface_increment_near(device_outputs[idx].surface, host_outputs[idx].surface);
        EXPECT_NEAR(device_outputs[idx].distribution.lambda, host_outputs[idx].distribution.lambda,
                    backend_math_abs_tol(host_outputs[idx].distribution.lambda));
        EXPECT_NEAR(device_outputs[idx].distribution.number, host_outputs[idx].distribution.number,
                    backend_math_abs_tol(host_outputs[idx].distribution.number));
        EXPECT_NEAR(device_outputs[idx].distribution.intercept, host_outputs[idx].distribution.intercept,
                    backend_math_abs_tol(host_outputs[idx].distribution.intercept));
        EXPECT_EQ(device_outputs[idx].distribution.limited_to_min, host_outputs[idx].distribution.limited_to_min);
        EXPECT_EQ(device_outputs[idx].distribution.limited_to_max, host_outputs[idx].distribution.limited_to_max);
    }
}