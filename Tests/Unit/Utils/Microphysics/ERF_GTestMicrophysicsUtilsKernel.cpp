#include <algorithm>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>

#include <gtest/gtest.h>

#include "ERF_GTestMicrophysicsCommon.H"

using namespace microphysics_test;

namespace {

constexpr int kNumKernelMetrics = 19;

void launch_microphysics_device_smoke (amrex::Gpu::DeviceVector<amrex::Real>& metrics)
{
    // Keep this device-path smoke test explicit. The scalar tests exercise the
    // physical properties in detail on the host, while this kernel test proves
    // the current microphysics utility surface is callable inside AMReX
    // ParallelFor and returns values that host-side GTest can inspect.
    auto* metric_ptr = metrics.data();

    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept {
        amrex::Real qsatw_normal;
        amrex::Real dtqsatw_normal;
        amrex::Real qsatw_capped;
        amrex::Real dtqsatw_capped;
        amrex::Real qsati_normal;
        amrex::Real dtqsati_normal;
        amrex::Real qsati_capped;
        amrex::Real dtqsati_capped;

        const amrex::Real water_normal_temperature = amrex::Real(300.0);
        const amrex::Real water_normal_pressure = amrex::Real(800.0);
        const amrex::Real water_capped_pressure = capped_branch_pressure(erf_esatw(water_normal_temperature));
        const amrex::Real ice_temperature = amrex::Real(260.0);
        const amrex::Real ice_normal_pressure = amrex::Real(800.0);
        const amrex::Real ice_capped_pressure = capped_branch_pressure(erf_esati(ice_temperature));

        erf_qsatw(water_normal_temperature, water_normal_pressure, qsatw_normal);
        erf_dtqsatw(water_normal_temperature, water_normal_pressure, dtqsatw_normal);
        erf_qsatw(water_normal_temperature, water_capped_pressure, qsatw_capped);
        erf_dtqsatw(water_normal_temperature, water_capped_pressure, dtqsatw_capped);
        erf_qsati(ice_temperature, ice_normal_pressure, qsati_normal);
        erf_dtqsati(ice_temperature, ice_normal_pressure, dtqsati_normal);
        erf_qsati(ice_temperature, ice_capped_pressure, qsati_capped);
        erf_dtqsati(ice_temperature, ice_capped_pressure, dtqsati_capped);

        metric_ptr[0] = erf_gammafff(amrex::Real(1.0));
        metric_ptr[1] = erf_gammafff(amrex::Real(0.5));
        metric_ptr[2] = erf_esati(ice_temperature);
        metric_ptr[3] = erf_esatw_cc(ice_temperature);
        metric_ptr[4] = erf_esatw(amrex::Real(190.0));
        metric_ptr[5] = erf_esatw(amrex::Real(273.15), true);
        metric_ptr[6] = erf_dtesati(ice_temperature);
        metric_ptr[7] = erf_dtesatw_cc(ice_temperature);
        metric_ptr[8] = erf_dtesatw(water_normal_temperature);
        metric_ptr[9] = qsatw_normal;
        metric_ptr[10] = dtqsatw_normal;
        metric_ptr[11] = qsatw_capped;
        metric_ptr[12] = dtqsatw_capped;
        metric_ptr[13] = qsati_normal;
        metric_ptr[14] = dtqsati_normal;
        metric_ptr[15] = qsati_capped;
        metric_ptr[16] = dtqsati_capped;
        metric_ptr[17] = HSEutils::compute_saturation_pressure(amrex::Real(300.0), false);
        metric_ptr[18] = HSEutils::compute_saturation_pressure(amrex::Real(300.0), true);
    });
}

} // namespace

// Motivation: This kernel smoke test is not a replacement for the scalar
// property tests. It exists to keep the microphysics utility surface callable
// from AMReX device code while all GTest assertions stay on the host.
TEST(MicrophysicsKernel, DevicePathSmoke)
{
    amrex::Gpu::DeviceVector<amrex::Real> device_metrics(kNumKernelMetrics, amrex::Real(0.0));
    amrex::Gpu::HostVector<amrex::Real> host_metrics(device_metrics.size(), amrex::Real(0.0));

    launch_microphysics_device_smoke(device_metrics);
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     device_metrics.begin(), device_metrics.end(), host_metrics.begin());

    EXPECT_NEAR(host_metrics[0], amrex::Real(1.0), scaled_tol(host_metrics[0], amrex::Real(1.0), kValueRelTol));
    EXPECT_NEAR(host_metrics[1], kSqrtPi, scaled_tol(host_metrics[1], kSqrtPi, kValueRelTol));
    EXPECT_GT(host_metrics[2], amrex::Real(0.0));
    EXPECT_GT(host_metrics[3], amrex::Real(0.0));
    EXPECT_GT(host_metrics[4], amrex::Real(0.0));
    EXPECT_NEAR(host_metrics[5], kEmpiricalTriplePointMbar,
                scaled_tol(host_metrics[5], kEmpiricalTriplePointMbar, kEmpiricalAgreementRelTol));
    EXPECT_GT(host_metrics[6], amrex::Real(0.0));
    EXPECT_GT(host_metrics[7], amrex::Real(0.0));
    EXPECT_GT(host_metrics[8], amrex::Real(0.0));
    EXPECT_GE(host_metrics[9], amrex::Real(0.0));
    EXPECT_LE(host_metrics[9], Rd_on_Rv);
    EXPECT_GT(host_metrics[10], amrex::Real(0.0));
    EXPECT_NEAR(host_metrics[11], Rd_on_Rv, scaled_tol(host_metrics[11], Rd_on_Rv, kValueRelTol));
    EXPECT_NEAR(host_metrics[12], amrex::Real(0.0), scaled_tol(host_metrics[12], amrex::Real(0.0), kValueRelTol));
    EXPECT_GE(host_metrics[13], amrex::Real(0.0));
    EXPECT_LE(host_metrics[13], Rd_on_Rv);
    EXPECT_GT(host_metrics[14], amrex::Real(0.0));
    EXPECT_NEAR(host_metrics[15], Rd_on_Rv, scaled_tol(host_metrics[15], Rd_on_Rv, kValueRelTol));
    EXPECT_NEAR(host_metrics[16], amrex::Real(0.0), scaled_tol(host_metrics[16], amrex::Real(0.0), kValueRelTol));
    EXPECT_NEAR(host_metrics[17], amrex::Real(100.0) * erf_esatw(amrex::Real(300.0), false),
                scaled_tol(host_metrics[17], amrex::Real(100.0) * erf_esatw(amrex::Real(300.0), false), kValueRelTol));
    EXPECT_NEAR(host_metrics[18], amrex::Real(100.0) * erf_esatw(amrex::Real(300.0), true),
                scaled_tol(host_metrics[18], amrex::Real(100.0) * erf_esatw(amrex::Real(300.0), true), kValueRelTol));
}