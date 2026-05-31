#include <vector>

#include <gtest/gtest.h>

#include "ERF_GTestMorrisonCommon.H"

using namespace morrison_test;

namespace {

struct KernelCase {
    MorrisonCellState state;
    amrex::Real qvqvs;
    amrex::Real qvqvsi;
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
        const MorrisonDistributionParameters distribution = morrison_exponential_distribution_parameters(
            cases_ptr[idx].distribution_mass, cases_ptr[idx].distribution_number,
            cases_ptr[idx].coefficient, cases_ptr[idx].lambda_min, cases_ptr[idx].lambda_max,
            amrex::Real(3.0));

        outputs_ptr[idx] = KernelOutputs{state, effective_radii, cleanup, distribution};
    });
    morrison_test::sync();
}

std::vector<KernelCase> make_kernel_cases ()
{
    const amrex::Real mass = amrex::Real(2.0e-4);
    return {
        {make_state(amrex::Real(1.0e-2), kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5),
                    kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5), kQSmall * amrex::Real(0.5),
                    amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
         morrison_subsaturation_ratio_threshold, morrison_subsaturation_ratio_threshold,
         mass, number_for_lambda(mass, kRainCoefficient, amrex::Real(0.5) * (kLamMinRain + kLamMaxRain)),
         kRainCoefficient, kLamMinRain, kLamMaxRain},
        {make_state(amrex::Real(1.0e-2), amrex::Real(2.0e-9), amrex::Real(3.0e-9),
                    amrex::Real(4.0e-9), amrex::Real(5.0e-9), amrex::Real(6.0e-9),
                    amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
         morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
         morrison_subsaturation_ratio_threshold - amrex::Real(1.0e-3),
         mass, number_for_lambda(mass, kSnowCoefficient, kLamMinSnow * amrex::Real(0.25)),
         kSnowCoefficient, kLamMinSnow, kLamMaxSnow},
        {make_state(amrex::Real(1.0e-2), amrex::Real(1.0e-4), amrex::Real(0.0),
                    amrex::Real(2.0e-4), amrex::Real(2.0e-7), amrex::Real(3.0e-7),
                    amrex::Real(10.0), amrex::Real(20.0), amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
         morrison_subsaturation_ratio_threshold, morrison_subsaturation_ratio_threshold,
         mass, number_for_lambda(mass, kGraupelCoefficient, kLamMaxGraupel * amrex::Real(4.0)),
         kGraupelCoefficient, kLamMinGraupel, kLamMaxGraupel},
        {make_state(amrex::Real(1.0e-2), kQSmall, kQSmall, kQSmall, morrison_warm_small_ice_melt_threshold,
                    morrison_warm_small_ice_melt_threshold, amrex::Real(10.0), amrex::Real(20.0),
                    amrex::Real(30.0), amrex::Real(40.0), amrex::Real(50.0)),
         morrison_subsaturation_ratio_threshold, morrison_subsaturation_ratio_threshold,
         mass, number_for_lambda(mass, kRainCoefficient, kLamMinRain),
         kRainCoefficient, kLamMinRain, kLamMaxRain}
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
    const MorrisonDistributionParameters distribution = morrison_exponential_distribution_parameters(
        test_case.distribution_mass, test_case.distribution_number, test_case.coefficient,
        test_case.lambda_min, test_case.lambda_max, amrex::Real(3.0));
    return KernelOutputs{state, effective_radii, cleanup, distribution};
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

} // namespace

// Motivation: The extracted Morrison helpers are called from AMReX kernels in
// production. This sweep covers qsmall cleanup, threshold equality, warm
// tiny-ice melt, subsaturated cleanup, and PSD lower/upper/in-range branches on
// the active backend, then compares against the same helper surface on host.
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
        EXPECT_EQ(device_outputs[idx].cleanup.qc_zeroed, host_outputs[idx].cleanup.qc_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qi_zeroed, host_outputs[idx].cleanup.qi_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qr_zeroed, host_outputs[idx].cleanup.qr_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qs_zeroed, host_outputs[idx].cleanup.qs_zeroed);
        EXPECT_EQ(device_outputs[idx].cleanup.qg_zeroed, host_outputs[idx].cleanup.qg_zeroed);
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