#include <algorithm>
#include <vector>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>

#include <gtest/gtest.h>

#include "GTestHSEUtilsCommon.H"

using namespace hse_test;

namespace {

void launch_case_sweep (amrex::Gpu::DeviceVector<KernelCase>& cases,
                        amrex::Gpu::DeviceVector<Real>& outputs)
{
    // Keep the device-path metric sweep explicit. The scalar tests check the
    // same contracts on the host, while this helper proves the HSE utilities
    // stay callable inside AMReX ParallelFor across multiple branch regimes.
    auto* case_ptr = cases.data();
    auto* output_ptr = outputs.data();

    amrex::ParallelFor(cases.size(), [=] AMREX_GPU_DEVICE (int index) noexcept {
        const KernelCase current = case_ptr[index];
        const bool use_empirical = current.use_empirical != 0;
        const bool t_from_theta = current.t_from_theta != 0;
        Real theta = Real(0.0);
        Real rho = Real(0.0);
        Real qv = Real(0.0);
        Real t_dp = Real(0.0);
        Real temperature = Real(0.0);

        HSEutils::compute_rho(current.pressure, theta, rho, qv, t_dp, temperature,
                              current.q_t, current.eq_pot_temp, use_empirical, current.which_zone,
                              current.scaled_height, t_from_theta, current.theta_0, current.theta_tr,
                              current.z_tr, current.t_tr);

        const Real saturation = HSEutils::compute_saturation_pressure(temperature, use_empirical);
        const Real humidity = HSEutils::compute_relative_humidity(current.pressure, temperature, use_empirical,
                                                                  current.which_zone, current.scaled_height);
        const Real residual = HSEutils::compute_F(current.pressure, current.pressure_minus_1, theta, rho, qv, t_dp,
                                                  temperature, current.dz, current.rho_minus_1, current.q_t,
                                                  current.eq_pot_temp, use_empirical, current.which_zone,
                                                  current.scaled_height, t_from_theta, current.theta_0,
                                                  current.theta_tr, current.z_tr, current.t_tr);

        const int offset = index * kKernelMetricsPerCase;
        output_ptr[offset + 0] = saturation;
        output_ptr[offset + 1] = humidity;
        output_ptr[offset + 2] = theta;
        output_ptr[offset + 3] = rho;
        output_ptr[offset + 4] = qv;
        output_ptr[offset + 5] = residual;
    });
}

std::string metric_label (const int metric)
{
    switch (metric) {
    case 0:
        return "saturation pressure";
    case 1:
        return "relative humidity";
    case 2:
        return "theta";
    case 3:
        return "rho";
    case 4:
        return "qv";
    default:
        return "F residual";
    }
}

} // namespace

// Motivation: HSE helpers are AMReX-portable utilities that run on both host
// and device paths. This multi-branch sweep checks that the device-compiled
// path matches the host reference over the intended branch coverage set.
TEST(HSEUtilsKernel, KernelHostDeviceBranchSweep)
{
    const std::vector<KernelCase> sample_cases = make_kernel_cases();
    amrex::Gpu::HostVector<KernelCase> host_cases(sample_cases.size());
    std::copy(sample_cases.begin(), sample_cases.end(), host_cases.begin());

    amrex::Gpu::DeviceVector<KernelCase> device_cases(host_cases.size());
    amrex::Gpu::DeviceVector<Real> device_outputs(host_cases.size() * kKernelMetricsPerCase, Real(0.0));
    amrex::Gpu::HostVector<Real> host_outputs(device_outputs.size(), Real(0.0));

    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_cases.begin(), host_cases.end(), device_cases.begin());
    launch_case_sweep(device_cases, device_outputs);
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_outputs.begin(), device_outputs.end(), host_outputs.begin());

    for (int index = 0; index < static_cast<int>(sample_cases.size()); ++index) {
        const KernelCase& current = sample_cases[index];
        Real theta = Real(0.0);
        Real rho = Real(0.0);
        Real qv = Real(0.0);
        Real t_dp = Real(0.0);
        Real temperature = Real(0.0);
        const bool use_empirical = current.use_empirical != 0;
        const bool t_from_theta = current.t_from_theta != 0;

        HSEutils::compute_rho(current.pressure, theta, rho, qv, t_dp, temperature,
                              current.q_t, current.eq_pot_temp, use_empirical, current.which_zone,
                              current.scaled_height, t_from_theta, current.theta_0, current.theta_tr,
                              current.z_tr, current.t_tr);
        const Real expected[kKernelMetricsPerCase] = {
            HSEutils::compute_saturation_pressure(temperature, use_empirical),
            HSEutils::compute_relative_humidity(current.pressure, temperature, use_empirical,
                                                current.which_zone, current.scaled_height),
            theta,
            rho,
            qv,
            HSEutils::compute_F(current.pressure, current.pressure_minus_1, theta, rho, qv, t_dp, temperature,
                                current.dz, current.rho_minus_1, current.q_t, current.eq_pot_temp,
                                use_empirical, current.which_zone, current.scaled_height, t_from_theta,
                                current.theta_0, current.theta_tr, current.z_tr, current.t_tr)
        };

        for (int metric = 0; metric < kKernelMetricsPerCase; ++metric) {
            const Real actual = host_outputs[index * kKernelMetricsPerCase + metric];
            const Real scale = scale_from_values({actual, expected[metric], Real(1.0)});
            EXPECT_NEAR(actual, expected[metric], host_device_tol(scale))
                << "case=" << index << " metric=" << metric_label(metric);
        }
    }
}