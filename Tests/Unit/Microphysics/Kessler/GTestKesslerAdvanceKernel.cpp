#include <vector>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>

#include <gtest/gtest.h>

#include <ERF_KesslerUtils.H>

#include "GTestKesslerCommon.H"

using namespace kessler_bfb_test;

namespace {

template <typename HostVectorT>
void copy_host_to_device (const HostVectorT& host,
                          amrex::Gpu::DeviceVector<amrex::Real>& device)
{
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), device.begin());
}

void launch_kessler_helper_kernel (const int ncases,
                                   const amrex::Real* tabs_ptr,
                                   const amrex::Real* pressure_ptr,
                                   const amrex::Real* qv_ptr,
                                   const amrex::Real* qc_ptr,
                                   const amrex::Real* qp_ptr,
                                   const amrex::Real* rho_ptr,
                                   const amrex::Real* dt_ptr,
                                   const amrex::Real* dzmin_ptr,
                                   const amrex::Real* reduced_ptr,
                                   const int* do_cond_ptr,
                                   amrex::Real* terminal_velocity_ptr,
                                   amrex::Real* flux_ptr,
                                   int* nsub_ptr,
                                   amrex::Real* dq_vapor_to_cloud_ptr,
                                   amrex::Real* dq_cloud_to_vapor_ptr,
                                   amrex::Real* dq_cloud_to_rain_ptr,
                                   amrex::Real* dq_rain_to_vapor_ptr)
{
    amrex::ParallelFor(ncases, [=] AMREX_GPU_DEVICE (int idx) noexcept {
        amrex::Real qsat_local;
        amrex::Real dtqsat_local;
        erf_qsatw(tabs_ptr[idx], pressure_ptr[idx], qsat_local);
        erf_dtqsatw(tabs_ptr[idx], pressure_ptr[idx], dtqsat_local);

        const amrex::Real terminal_velocity = kessler_terminal_velocity(rho_ptr[idx], qp_ptr[idx]);
        const KesslerSourceTerms source_terms = kessler_warm_rain_sources(
            qv_ptr[idx], qc_ptr[idx], qp_ptr[idx], rho_ptr[idx], pressure_ptr[idx],
            qsat_local, dtqsat_local, dt_ptr[idx], do_cond_ptr[idx] != 0);

        terminal_velocity_ptr[idx] = terminal_velocity;
        flux_ptr[idx] = kessler_precip_flux(rho_ptr[idx], terminal_velocity, qp_ptr[idx]);
        nsub_ptr[idx] = kessler_num_sedimentation_substeps(reduced_ptr[idx], dt_ptr[idx], dzmin_ptr[idx]);
        dq_vapor_to_cloud_ptr[idx] = source_terms.dq_vapor_to_cloud;
        dq_cloud_to_vapor_ptr[idx] = source_terms.dq_cloud_to_vapor;
        dq_cloud_to_rain_ptr[idx] = source_terms.dq_cloud_to_rain;
        dq_rain_to_vapor_ptr[idx] = source_terms.dq_rain_to_vapor;
    });

    kessler_bfb_test::sync();
}

} // namespace

TEST(KesslerKernel, HelperHostDeviceEquivalence)
{
    const std::vector<amrex::Real> tabs = {
        amrex::Real(290.0), amrex::Real(289.0), amrex::Real(287.0),
        amrex::Real(292.0), amrex::Real(285.0), amrex::Real(294.0)};
    const std::vector<amrex::Real> pressure = {
        amrex::Real(900.0), amrex::Real(890.0), amrex::Real(880.0),
        amrex::Real(910.0), amrex::Real(870.0), amrex::Real(920.0)};
    const std::vector<amrex::Real> qv = {
        qsat(tabs[0], pressure[0]) + amrex::Real(8.0e-4),
        amrex::Real(0.6) * qsat(tabs[1], pressure[1]),
        amrex::Real(0.1) * qsat(tabs[2], pressure[2]),
        qsat(tabs[3], pressure[3]),
        qsat(tabs[4], pressure[4]),
        amrex::Real(0.7) * qsat(tabs[5], pressure[5])};
    const std::vector<amrex::Real> qc = {
        amrex::Real(4.0e-4), amrex::Real(1.8e-3), amrex::Real(0.0),
        amrex::Real(0.5) * qcw0, amrex::Real(3.0) * qcw0, amrex::Real(1.2e-3)};
    const std::vector<amrex::Real> qp = {
        amrex::Real(1.0e-3), amrex::Real(0.0), amrex::Real(5.0e-2),
        amrex::Real(1.0e-3), amrex::Real(4.0e-3), amrex::Real(7.0e-2)};
    const std::vector<amrex::Real> rho = {
        amrex::Real(1.0), amrex::Real(0.95), amrex::Real(1.05),
        amrex::Real(1.1), amrex::Real(0.9), amrex::Real(1.15)};
    const std::vector<amrex::Real> dt = {
        amrex::Real(3.0), amrex::Real(3.0), amrex::Real(3.0),
        amrex::Real(3.0), amrex::Real(3.0), amrex::Real(3.0)};
    const std::vector<amrex::Real> dzmin = {
        amrex::Real(0.5), amrex::Real(0.5), amrex::Real(0.5),
        amrex::Real(0.5), amrex::Real(0.5), amrex::Real(0.5)};
    const std::vector<amrex::Real> reduced = {
        amrex::Real(0.01), amrex::Real(0.25), amrex::Real(0.5),
        amrex::Real(0.75), amrex::Real(1.0), amrex::Real(1.25)};
    const std::vector<int> do_cond = {1, 1, 1, 1, 0, 0};

    amrex::Gpu::DeviceVector<amrex::Real> d_tabs(tabs.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_pressure(pressure.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_qv(qv.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_qc(qc.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_qp(qp.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_rho(rho.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dt(dt.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dzmin(dzmin.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_reduced(reduced.size());
    amrex::Gpu::DeviceVector<int> d_do_cond(do_cond.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_terminal_velocity(tabs.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_flux(tabs.size());
    amrex::Gpu::DeviceVector<int> d_nsub(tabs.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dq_vapor_to_cloud(tabs.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dq_cloud_to_vapor(tabs.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dq_cloud_to_rain(tabs.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dq_rain_to_vapor(tabs.size());

    copy_host_to_device(tabs, d_tabs);
    copy_host_to_device(pressure, d_pressure);
    copy_host_to_device(qv, d_qv);
    copy_host_to_device(qc, d_qc);
    copy_host_to_device(qp, d_qp);
    copy_host_to_device(rho, d_rho);
    copy_host_to_device(dt, d_dt);
    copy_host_to_device(dzmin, d_dzmin);
    copy_host_to_device(reduced, d_reduced);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, do_cond.begin(), do_cond.end(), d_do_cond.begin());

    launch_kessler_helper_kernel(static_cast<int>(tabs.size()),
                                 d_tabs.data(), d_pressure.data(), d_qv.data(), d_qc.data(), d_qp.data(),
                                 d_rho.data(), d_dt.data(), d_dzmin.data(), d_reduced.data(), d_do_cond.data(),
                                 d_terminal_velocity.data(), d_flux.data(), d_nsub.data(),
                                 d_dq_vapor_to_cloud.data(), d_dq_cloud_to_vapor.data(),
                                 d_dq_cloud_to_rain.data(), d_dq_rain_to_vapor.data());

    std::vector<amrex::Real> terminal_velocity(tabs.size(), amrex::Real(0.0));
    std::vector<amrex::Real> flux(tabs.size(), amrex::Real(0.0));
    std::vector<int> nsub(tabs.size(), 0);
    std::vector<amrex::Real> dq_vapor_to_cloud(tabs.size(), amrex::Real(0.0));
    std::vector<amrex::Real> dq_cloud_to_vapor(tabs.size(), amrex::Real(0.0));
    std::vector<amrex::Real> dq_cloud_to_rain(tabs.size(), amrex::Real(0.0));
    std::vector<amrex::Real> dq_rain_to_vapor(tabs.size(), amrex::Real(0.0));

    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_terminal_velocity.begin(), d_terminal_velocity.end(), terminal_velocity.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_flux.begin(), d_flux.end(), flux.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_nsub.begin(), d_nsub.end(), nsub.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_dq_vapor_to_cloud.begin(), d_dq_vapor_to_cloud.end(), dq_vapor_to_cloud.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_dq_cloud_to_vapor.begin(), d_dq_cloud_to_vapor.end(), dq_cloud_to_vapor.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_dq_cloud_to_rain.begin(), d_dq_cloud_to_rain.end(), dq_cloud_to_rain.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_dq_rain_to_vapor.begin(), d_dq_rain_to_vapor.end(), dq_rain_to_vapor.begin());

    for (int idx = 0; idx < static_cast<int>(tabs.size()); ++idx) {
        amrex::Real qsat_local;
        amrex::Real dtqsat_local;
        erf_qsatw(tabs[idx], pressure[idx], qsat_local);
        erf_dtqsatw(tabs[idx], pressure[idx], dtqsat_local);
        const KesslerSourceTerms host_terms = kessler_warm_rain_sources(
            qv[idx], qc[idx], qp[idx], rho[idx], pressure[idx], qsat_local, dtqsat_local, dt[idx], do_cond[idx] != 0);
        const amrex::Real host_terminal_velocity = kessler_terminal_velocity(rho[idx], qp[idx]);
        const amrex::Real host_flux = kessler_precip_flux(rho[idx], host_terminal_velocity, qp[idx]);
        const int host_nsub = kessler_num_sedimentation_substeps(reduced[idx], dt[idx], dzmin[idx]);

        EXPECT_EQ(real_bits(terminal_velocity[idx]), real_bits(host_terminal_velocity)) << idx;
        EXPECT_EQ(real_bits(flux[idx]), real_bits(host_flux)) << idx;
        EXPECT_EQ(nsub[idx], host_nsub) << idx;
        EXPECT_EQ(real_bits(dq_vapor_to_cloud[idx]), real_bits(host_terms.dq_vapor_to_cloud)) << idx;
        EXPECT_EQ(real_bits(dq_cloud_to_vapor[idx]), real_bits(host_terms.dq_cloud_to_vapor)) << idx;
        EXPECT_EQ(real_bits(dq_cloud_to_rain[idx]), real_bits(host_terms.dq_cloud_to_rain)) << idx;
        EXPECT_EQ(real_bits(dq_rain_to_vapor[idx]), real_bits(host_terms.dq_rain_to_vapor)) << idx;
    }
}
