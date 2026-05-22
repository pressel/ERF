#include <cmath>
#include <vector>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>

#include <gtest/gtest.h>

#include <ERF_KesslerUtils.H>

#include "GTestKesslerCommon.H"

using namespace kessler_bfb_test;

namespace {

void expect_helper_real_match (const amrex::Real device_value,
                               const amrex::Real host_value,
                               const char* label,
                               const int idx)
{
#ifdef AMREX_USE_GPU
    const amrex::Real abs_tol = backend_math_abs_tol(host_value);
    EXPECT_LE(std::abs(device_value - host_value), abs_tol)
        << label << " idx=" << idx
        << " expected=" << host_value
        << " actual=" << device_value
        << " abs_tol=" << abs_tol;
#else
    EXPECT_EQ(real_bits(device_value), real_bits(host_value))
        << label << " idx=" << idx;
#endif
}

void expect_exact_helper_real_match (const amrex::Real device_value,
                                     const amrex::Real host_value,
                                     const char* label,
                                     const int idx)
{
    EXPECT_EQ(real_bits(device_value), real_bits(host_value))
        << label << " idx=" << idx;
}

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

void launch_kessler_scalar_helper_kernel (const int ncases,
                                          const amrex::Real* sat_qv_ptr,
                                          const amrex::Real* sat_qc_ptr,
                                          const amrex::Real* sat_qsat_ptr,
                                          const amrex::Real* sat_dtqsat_ptr,
                                          const int* sat_do_cond_ptr,
                                          const int* face_k_ptr,
                                          const int* face_k_lo_ptr,
                                          const int* face_k_hi_ptr,
                                          const amrex::Real* face_rho_km1_ptr,
                                          const amrex::Real* face_rho_k_ptr,
                                          const amrex::Real* face_qp_km1_ptr,
                                          const amrex::Real* face_qp_k_ptr,
                                          const amrex::Real* fz_hi_ptr,
                                          const amrex::Real* fz_lo_ptr,
                                          const amrex::Real* sed_rho_ptr,
                                          const amrex::Real* dJinv_ptr,
                                          const amrex::Real* coef_ptr,
                                          amrex::Real* sat_dq_vapor_to_cloud_ptr,
                                          amrex::Real* sat_dq_cloud_to_vapor_ptr,
                                          amrex::Real* face_rho_ptr,
                                          amrex::Real* face_qp_ptr,
                                          amrex::Real* dq_sed_ptr)
{
    amrex::ParallelFor(ncases, [=] AMREX_GPU_DEVICE (int idx) noexcept {
        const KesslerSaturationAdjustment saturation_adjustment =
            kessler_saturation_adjustment(sat_qv_ptr[idx], sat_qc_ptr[idx], sat_qsat_ptr[idx],
                                          sat_dtqsat_ptr[idx], sat_do_cond_ptr[idx] != 0);
        const KesslerFaceState face_state =
            kessler_face_state(face_k_ptr[idx], face_k_lo_ptr[idx], face_k_hi_ptr[idx],
                               face_rho_km1_ptr[idx], face_rho_k_ptr[idx],
                               face_qp_km1_ptr[idx], face_qp_k_ptr[idx]);

        sat_dq_vapor_to_cloud_ptr[idx] = saturation_adjustment.dq_vapor_to_cloud;
        sat_dq_cloud_to_vapor_ptr[idx] = saturation_adjustment.dq_cloud_to_vapor;
        face_rho_ptr[idx] = face_state.rho;
        face_qp_ptr[idx] = face_state.qp;
        dq_sed_ptr[idx] = kessler_sedimentation_tendency(
            fz_hi_ptr[idx], fz_lo_ptr[idx], sed_rho_ptr[idx], dJinv_ptr[idx], coef_ptr[idx]);
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

        expect_helper_real_match(terminal_velocity[idx], host_terminal_velocity, "terminal_velocity", idx);
        expect_helper_real_match(flux[idx], host_flux, "precip_flux", idx);
        EXPECT_EQ(nsub[idx], host_nsub) << idx;
        expect_helper_real_match(dq_vapor_to_cloud[idx], host_terms.dq_vapor_to_cloud, "dq_vapor_to_cloud", idx);
        expect_helper_real_match(dq_cloud_to_vapor[idx], host_terms.dq_cloud_to_vapor, "dq_cloud_to_vapor", idx);
        expect_helper_real_match(dq_cloud_to_rain[idx], host_terms.dq_cloud_to_rain, "dq_cloud_to_rain", idx);
        expect_helper_real_match(dq_rain_to_vapor[idx], host_terms.dq_rain_to_vapor, "dq_rain_to_vapor", idx);
    }
}

TEST(KesslerKernel, ScalarHelperHostDeviceEquivalence)
{
    const std::vector<amrex::Real> sat_qsat = {
        amrex::Real(0.010), amrex::Real(0.011), amrex::Real(0.009), amrex::Real(0.008), amrex::Real(0.012)};
    const std::vector<amrex::Real> sat_qv = {
        sat_qsat[0] + amrex::Real(8.0e-4),
        amrex::Real(0.6) * sat_qsat[1],
        amrex::Real(0.7) * sat_qsat[2],
        sat_qsat[3],
        sat_qsat[4] + amrex::Real(2.0e-4)};
    const std::vector<amrex::Real> sat_qc = {
        amrex::Real(4.0e-4), amrex::Real(1.2e-3), amrex::Real(0.0),
        amrex::Real(5.0e-4), amrex::Real(3.0e-4)};
    const std::vector<amrex::Real> sat_dtqsat = {
        amrex::Real(1.0e-4), amrex::Real(2.0e-4), amrex::Real(1.5e-4), amrex::Real(1.0e-4), amrex::Real(2.5e-4)};
    const std::vector<int> sat_do_cond = {1, 1, 0, 1, 1};

    const std::vector<int> face_k = {2, 5, 3, 1, 4};
    const std::vector<int> face_k_lo = {2, 2, 2, 1, 1};
    const std::vector<int> face_k_hi = {4, 4, 4, 3, 3};
    const std::vector<amrex::Real> face_rho_km1 = {
        amrex::Real(1.00), amrex::Real(1.10), amrex::Real(0.95), amrex::Real(1.20), amrex::Real(0.90)};
    const std::vector<amrex::Real> face_rho_k = {
        amrex::Real(1.05), amrex::Real(1.30), amrex::Real(1.15), amrex::Real(1.20), amrex::Real(0.92)};
    const std::vector<amrex::Real> face_qp_km1 = {
        amrex::Real(2.0e-3), amrex::Real(4.0e-3), amrex::Real(-5.0e-4), amrex::Real(3.0e-3), amrex::Real(2.0e-4)};
    const std::vector<amrex::Real> face_qp_k = {
        amrex::Real(1.0e-3), amrex::Real(8.0e-3), amrex::Real(-7.0e-4), amrex::Real(3.0e-3), amrex::Real(-1.0e-4)};

    const std::vector<amrex::Real> fz_hi = {
        amrex::Real(0.10), amrex::Real(0.02), amrex::Real(0.00), amrex::Real(-1.0e-4), amrex::Real(0.25)};
    const std::vector<amrex::Real> fz_lo = {
        amrex::Real(0.05), amrex::Real(0.01), amrex::Real(0.00), amrex::Real(-2.0e-4), amrex::Real(0.20)};
    const std::vector<amrex::Real> sed_rho = {
        amrex::Real(1.00), amrex::Real(0.95), amrex::Real(1.10), amrex::Real(1.20), amrex::Real(0.90)};
    const std::vector<amrex::Real> dJinv = {
        amrex::Real(1.0), amrex::Real(0.8), amrex::Real(1.2), amrex::Real(1.1), amrex::Real(0.9)};
    const std::vector<amrex::Real> coef = {
        amrex::Real(0.5), amrex::Real(0.25), amrex::Real(1.0), amrex::Real(0.75), amrex::Real(0.125)};

    amrex::Gpu::DeviceVector<amrex::Real> d_sat_qv(sat_qv.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_sat_qc(sat_qc.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_sat_qsat(sat_qsat.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_sat_dtqsat(sat_dtqsat.size());
    amrex::Gpu::DeviceVector<int> d_sat_do_cond(sat_do_cond.size());
    amrex::Gpu::DeviceVector<int> d_face_k(face_k.size());
    amrex::Gpu::DeviceVector<int> d_face_k_lo(face_k_lo.size());
    amrex::Gpu::DeviceVector<int> d_face_k_hi(face_k_hi.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_face_rho_km1(face_rho_km1.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_face_rho_k(face_rho_k.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_face_qp_km1(face_qp_km1.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_face_qp_k(face_qp_k.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_fz_hi(fz_hi.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_fz_lo(fz_lo.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_sed_rho(sed_rho.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dJinv(dJinv.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_coef(coef.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_sat_dq_vapor_to_cloud(sat_qv.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_sat_dq_cloud_to_vapor(sat_qv.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_face_rho(sat_qv.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_face_qp(sat_qv.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_dq_sed(sat_qv.size());

    copy_host_to_device(sat_qv, d_sat_qv);
    copy_host_to_device(sat_qc, d_sat_qc);
    copy_host_to_device(sat_qsat, d_sat_qsat);
    copy_host_to_device(sat_dtqsat, d_sat_dtqsat);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, sat_do_cond.begin(), sat_do_cond.end(), d_sat_do_cond.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, face_k.begin(), face_k.end(), d_face_k.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, face_k_lo.begin(), face_k_lo.end(), d_face_k_lo.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, face_k_hi.begin(), face_k_hi.end(), d_face_k_hi.begin());
    copy_host_to_device(face_rho_km1, d_face_rho_km1);
    copy_host_to_device(face_rho_k, d_face_rho_k);
    copy_host_to_device(face_qp_km1, d_face_qp_km1);
    copy_host_to_device(face_qp_k, d_face_qp_k);
    copy_host_to_device(fz_hi, d_fz_hi);
    copy_host_to_device(fz_lo, d_fz_lo);
    copy_host_to_device(sed_rho, d_sed_rho);
    copy_host_to_device(dJinv, d_dJinv);
    copy_host_to_device(coef, d_coef);

    launch_kessler_scalar_helper_kernel(
        static_cast<int>(sat_qv.size()),
        d_sat_qv.data(), d_sat_qc.data(), d_sat_qsat.data(), d_sat_dtqsat.data(), d_sat_do_cond.data(),
        d_face_k.data(), d_face_k_lo.data(), d_face_k_hi.data(),
        d_face_rho_km1.data(), d_face_rho_k.data(), d_face_qp_km1.data(), d_face_qp_k.data(),
        d_fz_hi.data(), d_fz_lo.data(), d_sed_rho.data(), d_dJinv.data(), d_coef.data(),
        d_sat_dq_vapor_to_cloud.data(), d_sat_dq_cloud_to_vapor.data(),
        d_face_rho.data(), d_face_qp.data(), d_dq_sed.data());

    std::vector<amrex::Real> sat_dq_vapor_to_cloud(sat_qv.size(), amrex::Real(0.0));
    std::vector<amrex::Real> sat_dq_cloud_to_vapor(sat_qv.size(), amrex::Real(0.0));
    std::vector<amrex::Real> face_rho(sat_qv.size(), amrex::Real(0.0));
    std::vector<amrex::Real> face_qp(sat_qv.size(), amrex::Real(0.0));
    std::vector<amrex::Real> dq_sed(sat_qv.size(), amrex::Real(0.0));

    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_sat_dq_vapor_to_cloud.begin(), d_sat_dq_vapor_to_cloud.end(), sat_dq_vapor_to_cloud.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_sat_dq_cloud_to_vapor.begin(), d_sat_dq_cloud_to_vapor.end(), sat_dq_cloud_to_vapor.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_face_rho.begin(), d_face_rho.end(), face_rho.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_face_qp.begin(), d_face_qp.end(), face_qp.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_dq_sed.begin(), d_dq_sed.end(), dq_sed.begin());

    for (int idx = 0; idx < static_cast<int>(sat_qv.size()); ++idx) {
        const KesslerSaturationAdjustment host_saturation_adjustment =
            kessler_saturation_adjustment(sat_qv[idx], sat_qc[idx], sat_qsat[idx], sat_dtqsat[idx], sat_do_cond[idx] != 0);
        const KesslerFaceState host_face_state =
            kessler_face_state(face_k[idx], face_k_lo[idx], face_k_hi[idx],
                               face_rho_km1[idx], face_rho_k[idx], face_qp_km1[idx], face_qp_k[idx]);
        const amrex::Real host_dq_sed =
            kessler_sedimentation_tendency(fz_hi[idx], fz_lo[idx], sed_rho[idx], dJinv[idx], coef[idx]);

        expect_exact_helper_real_match(
            sat_dq_vapor_to_cloud[idx], host_saturation_adjustment.dq_vapor_to_cloud, "sat_dq_vapor_to_cloud", idx);
        expect_exact_helper_real_match(
            sat_dq_cloud_to_vapor[idx], host_saturation_adjustment.dq_cloud_to_vapor, "sat_dq_cloud_to_vapor", idx);
        expect_exact_helper_real_match(face_rho[idx], host_face_state.rho, "face_rho", idx);
        expect_exact_helper_real_match(face_qp[idx], host_face_state.qp, "face_qp", idx);
        expect_exact_helper_real_match(dq_sed[idx], host_dq_sed, "dq_sed", idx);
    }
}
