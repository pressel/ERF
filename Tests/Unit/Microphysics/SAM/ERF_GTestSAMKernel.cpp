#include <string>
#include <vector>

#include <AMReX_GpuContainers.H>

#include <gtest/gtest.h>

#include "ERF_GTestSAMCommon.H"

using namespace sam_test;

namespace {

struct SAMKernelCase {
    int mode;
    amrex::Real tabs;
    amrex::Real rho;
    amrex::Real omn;
    amrex::Real domn;
    amrex::Real qsatw;
    amrex::Real qsati;
    amrex::Real dqsatw;
    amrex::Real dqsati;
    amrex::Real lstar;
    amrex::Real dlstar;
    amrex::Real qv;
    amrex::Real qsat;
    amrex::Real dtn;
    amrex::Real qcc;
    amrex::Real qii;
    amrex::Real coefice;
    amrex::Real qpr;
    amrex::Real qps;
    amrex::Real qpg;
    amrex::Real qci_avg;
    amrex::Real reduced_flux;
    amrex::Real dt;
    amrex::Real dz;
    SAMPrecipFaceState face_state;
    SAMFaceState face_average_input;
    int face_k;
    int face_k_lo;
    int face_k_hi;
    amrex::Real rho_km1;
    amrex::Real rho_k;
    amrex::Real tabs_km1;
    amrex::Real tabs_k;
    amrex::Real qci_km1;
    amrex::Real qci_k;
    amrex::Real qp_km1;
    amrex::Real qp_k;
    amrex::Real gamr1;
    amrex::Real gamr2;
    amrex::Real gams1;
    amrex::Real gams2;
    amrex::Real gamg1;
    amrex::Real gamg2;
};

struct SAMKernelOutputs {
    amrex::Real cloud_liquid_fraction;
    amrex::Real rain_fraction;
    amrex::Real graupel_fraction;
    amrex::Real mixed_qsat;
    amrex::Real mixed_dqsat;
    amrex::Real residual_derivative;
    amrex::Real autoconv_cloud;
    amrex::Real autoconv_ice;
    amrex::Real evap_rain;
    amrex::Real evap_snow;
    amrex::Real evap_graupel;
    amrex::Real coeff_evapr1;
    amrex::Real coeff_evaps1;
    amrex::Real coeff_evapg1;
    amrex::Real ice_velocity;
    amrex::Real precip_flux;
    amrex::Real density_corrected_flux;
    amrex::Real face_rho;
    amrex::Real face_tabs;
    amrex::Real face_qci;
    amrex::Real face_qp;
    int substeps;
};

SAMKernelOutputs host_reference (const SAMKernelCase& test_case)
{
    const SAMCoefficientRow coeffs = sam_compute_coefficient_row(
        test_case.rho, test_case.tabs,
        test_case.gamr1, test_case.gamr2,
        test_case.gams1, test_case.gams2,
        test_case.gamg1, test_case.gamg2);
    const SAMFaceState face = sam_face_average_state(
        test_case.face_k, test_case.face_k_lo, test_case.face_k_hi,
        test_case.rho_km1, test_case.rho_k,
        test_case.tabs_km1, test_case.tabs_k,
        test_case.qci_km1, test_case.qci_k,
        test_case.qp_km1, test_case.qp_k);
    const amrex::Real precip_flux = sam_precip_flux_from_face_state(
        test_case.face_state, amrex::Real(2.0), amrex::Real(3.0), amrex::Real(4.0));

    return SAMKernelOutputs{
        sam_cloud_liquid_fraction(test_case.mode, test_case.tabs, a_bg, tbgmin * a_bg),
        sam_precip_rain_fraction(test_case.mode, test_case.tabs),
        sam_graupel_fraction(test_case.mode, test_case.tabs),
        sam_mixed_qsat(test_case.omn, test_case.qsatw, test_case.qsati),
        sam_mixed_dqsat_dT(test_case.omn, test_case.domn,
                           test_case.qsatw, test_case.qsati,
                           test_case.dqsatw, test_case.dqsati),
        sam_newton_residual_derivative(test_case.lstar, test_case.dlstar,
                                       test_case.qv, test_case.qsat,
                                       test_case.dqsatw),
        sam_autoconversion_rates(test_case.dtn, test_case.qcc, test_case.qii,
                                 test_case.coefice).dqca,
        sam_autoconversion_rates(test_case.dtn, test_case.qcc, test_case.qii,
                                 test_case.coefice).dqia,
        sam_precip_evaporation_rates(
            test_case.qpr, test_case.qps, test_case.qpg,
            (amrex::Real(5.0) + b_rain) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_snow) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_grau) / amrex::Real(8.0),
            amrex::Real(2.0), amrex::Real(3.0),
            amrex::Real(4.0), amrex::Real(5.0),
            amrex::Real(6.0), amrex::Real(7.0)).dqpr,
        sam_precip_evaporation_rates(
            test_case.qpr, test_case.qps, test_case.qpg,
            (amrex::Real(5.0) + b_rain) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_snow) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_grau) / amrex::Real(8.0),
            amrex::Real(2.0), amrex::Real(3.0),
            amrex::Real(4.0), amrex::Real(5.0),
            amrex::Real(6.0), amrex::Real(7.0)).dqps,
        sam_precip_evaporation_rates(
            test_case.qpr, test_case.qps, test_case.qpg,
            (amrex::Real(5.0) + b_rain) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_snow) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_grau) / amrex::Real(8.0),
            amrex::Real(2.0), amrex::Real(3.0),
            amrex::Real(4.0), amrex::Real(5.0),
            amrex::Real(6.0), amrex::Real(7.0)).dqpg,
        coeffs.evapr1,
        coeffs.evaps1,
        coeffs.evapg1,
        sam_cloud_ice_terminal_velocity(test_case.qci_avg),
        precip_flux,
        sam_precip_flux_density_corrected(precip_flux, amrex::Real(1.2), test_case.face_state.rho_avg),
        face.rho_avg,
        face.tabs_avg,
        face.qci_avg,
        face.qp_avg,
        sam_substep_count_from_reduced_flux(test_case.reduced_flux, test_case.dt, test_case.dz)};
}

void launch_sam_helper_kernel (const int ncases,
                               const SAMKernelCase* cases_ptr,
                               SAMKernelOutputs* outputs_ptr)
{
    amrex::ParallelFor(ncases, [=] AMREX_GPU_DEVICE (int idx) noexcept {
        const SAMKernelCase test_case = cases_ptr[idx];
        const SAMCoefficientRow coeffs = sam_compute_coefficient_row(
            test_case.rho, test_case.tabs,
            test_case.gamr1, test_case.gamr2,
            test_case.gams1, test_case.gams2,
            test_case.gamg1, test_case.gamg2);
        const SAMFaceState face = sam_face_average_state(
            test_case.face_k, test_case.face_k_lo, test_case.face_k_hi,
            test_case.rho_km1, test_case.rho_k,
            test_case.tabs_km1, test_case.tabs_k,
            test_case.qci_km1, test_case.qci_k,
            test_case.qp_km1, test_case.qp_k);
        const amrex::Real precip_flux = sam_precip_flux_from_face_state(
            test_case.face_state, amrex::Real(2.0), amrex::Real(3.0), amrex::Real(4.0));
        const SAMPrecipSources auto_sources = sam_autoconversion_rates(
            test_case.dtn, test_case.qcc, test_case.qii, test_case.coefice);
        const SAMPrecipSources evap_sources = sam_precip_evaporation_rates(
            test_case.qpr, test_case.qps, test_case.qpg,
            (amrex::Real(5.0) + b_rain) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_snow) / amrex::Real(8.0),
            (amrex::Real(5.0) + b_grau) / amrex::Real(8.0),
            amrex::Real(2.0), amrex::Real(3.0),
            amrex::Real(4.0), amrex::Real(5.0),
            amrex::Real(6.0), amrex::Real(7.0));

        outputs_ptr[idx] = SAMKernelOutputs{
            sam_cloud_liquid_fraction(test_case.mode, test_case.tabs, a_bg, tbgmin * a_bg),
            sam_precip_rain_fraction(test_case.mode, test_case.tabs),
            sam_graupel_fraction(test_case.mode, test_case.tabs),
            sam_mixed_qsat(test_case.omn, test_case.qsatw, test_case.qsati),
            sam_mixed_dqsat_dT(test_case.omn, test_case.domn,
                               test_case.qsatw, test_case.qsati,
                               test_case.dqsatw, test_case.dqsati),
            sam_newton_residual_derivative(test_case.lstar, test_case.dlstar,
                                           test_case.qv, test_case.qsat,
                                           test_case.dqsatw),
            auto_sources.dqca,
            auto_sources.dqia,
            evap_sources.dqpr,
            evap_sources.dqps,
            evap_sources.dqpg,
            coeffs.evapr1,
            coeffs.evaps1,
            coeffs.evapg1,
            sam_cloud_ice_terminal_velocity(test_case.qci_avg),
            precip_flux,
            sam_precip_flux_density_corrected(precip_flux, amrex::Real(1.2), test_case.face_state.rho_avg),
            face.rho_avg,
            face.tabs_avg,
            face.qci_avg,
            face.qp_avg,
            sam_substep_count_from_reduced_flux(test_case.reduced_flux, test_case.dt, test_case.dz)};
    });

    amrex::Gpu::synchronize();
}

std::vector<SAMKernelCase> make_kernel_cases ()
{
    const amrex::Real gamr1 = erf_gammafff(three + b_rain);
    const amrex::Real gamr2 = erf_gammafff((amrex::Real(5.0) + b_rain) / two);
    const amrex::Real gams1 = erf_gammafff(three + b_snow);
    const amrex::Real gams2 = erf_gammafff((amrex::Real(5.0) + b_snow) / two);
    const amrex::Real gamg1 = erf_gammafff(three + b_grau);
    const amrex::Real gamg2 = erf_gammafff((amrex::Real(5.0) + b_grau) / two);

    return {
        {kSAMWithIceMode, tbgmin - amrex::Real(1.0), amrex::Real(0.8), amrex::Real(0.0), amrex::Real(1.0e-2),
         amrex::Real(6.0e-3), amrex::Real(4.0e-3), amrex::Real(3.0e-4), amrex::Real(2.0e-4),
         amrex::Real(2500.0), amrex::Real(-1.0), amrex::Real(5.0e-3), amrex::Real(4.5e-3),
         amrex::Real(2.0), qcw0, qci0, amrex::Real(1.0),
         amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
         -amrex::Real(1.0e-6), amrex::Real(0.0), amrex::Real(4.0), amrex::Real(100.0),
         {amrex::Real(0.9), tprmin - amrex::Real(1.0), kQpThresholdForBranchSampling,
          amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)},
         {}, 0, 0, 2,
         amrex::Real(0.7), amrex::Real(0.8),
         amrex::Real(250.0), amrex::Real(252.0),
         amrex::Real(1.0e-4), amrex::Real(2.0e-4),
         amrex::Real(1.0e-7), amrex::Real(2.0e-7),
         gamr1, gamr2, gams1, gams2, gamg1, gamg2},
        {kSAMWithIceMode, amrex::Real(0.5) * (tbgmin + tbgmax), amrex::Real(1.0), amrex::Real(0.4), amrex::Real(1.5e-2),
         amrex::Real(8.0e-3), amrex::Real(6.5e-3), amrex::Real(4.0e-4), amrex::Real(2.5e-4),
         amrex::Real(2700.0), amrex::Real(-2.0), amrex::Real(7.0e-3), amrex::Real(6.1e-3),
         amrex::Real(1.5), qcw0 + amrex::Real(3.0e-4), qci0 + amrex::Real(2.0e-4), amrex::Real(1.7),
         amrex::Real(4.0e-4), amrex::Real(5.0e-4), amrex::Real(6.0e-4),
         amrex::Real(3.0e-4), amrex::Real(12.0), amrex::Real(5.0), amrex::Real(100.0),
         {amrex::Real(1.0), amrex::Real(270.0), amrex::Real(5.0e-5),
          amrex::Real(0.45), amrex::Real(0.3),
          amrex::Real(2.25e-5), amrex::Real(1.925e-5), amrex::Real(8.25e-6)},
         {}, 1, 0, 2,
         amrex::Real(0.9), amrex::Real(1.1),
         amrex::Real(268.0), amrex::Real(270.0),
         amrex::Real(2.0e-4), amrex::Real(4.0e-4),
         amrex::Real(2.0e-5), amrex::Real(5.0e-5),
         gamr1, gamr2, gams1, gams2, gamg1, gamg2},
        {kSAMWithIceMode, tbgmax + amrex::Real(1.0), amrex::Real(1.2), amrex::Real(1.0), amrex::Real(0.0),
         amrex::Real(1.0e-2), amrex::Real(7.0e-3), amrex::Real(5.0e-4), amrex::Real(3.0e-4),
         amrex::Real(2400.0), amrex::Real(-1.5), amrex::Real(8.5e-3), amrex::Real(7.5e-3),
         amrex::Real(1.0), qcw0 + amrex::Real(1.0e-4), qci0, amrex::Real(0.9),
         amrex::Real(8.0e-4), amrex::Real(2.0e-4), amrex::Real(1.0e-4),
         amrex::Real(1.0e3), amrex::Real(20.0), amrex::Real(3.0), amrex::Real(50.0),
         {amrex::Real(1.1), tprmax + amrex::Real(1.0), amrex::Real(7.0e-5),
          amrex::Real(1.0), amrex::Real(0.0),
          amrex::Real(7.0e-5), amrex::Real(0.0), amrex::Real(0.0)},
         {}, 2, 0, 2,
         amrex::Real(1.1), amrex::Real(1.3),
         amrex::Real(274.0), amrex::Real(276.0),
         amrex::Real(3.0e-4), amrex::Real(5.0e-4),
         amrex::Real(4.0e-5), amrex::Real(7.0e-5),
         gamr1, gamr2, gams1, gams2, gamg1, gamg2},
        {kSAMNoIceMode, amrex::Real(245.0), amrex::Real(0.6), amrex::Real(1.0), amrex::Real(0.0),
         amrex::Real(5.0e-3), amrex::Real(3.0e-3), amrex::Real(2.0e-4), amrex::Real(1.0e-4),
         amrex::Real(2550.0), amrex::Real(-0.5), amrex::Real(4.0e-3), amrex::Real(3.5e-3),
         amrex::Real(0.75), qcw0 + amrex::Real(2.0e-4), qci0 + amrex::Real(1.0e-4), amrex::Real(1.3),
         amrex::Real(3.0e-4), amrex::Real(4.0e-4), amrex::Real(5.0e-4),
         amrex::Real(5.0e-5), amrex::Real(2.0), amrex::Real(2.0), amrex::Real(100.0),
         {amrex::Real(0.7), amrex::Real(245.0), amrex::Real(3.0e-5),
          amrex::Real(1.0), amrex::Real(0.0),
          amrex::Real(3.0e-5), amrex::Real(0.0), amrex::Real(0.0)},
         {}, 3, 0, 2,
         amrex::Real(0.5), amrex::Real(0.7),
         amrex::Real(244.0), amrex::Real(245.0),
         amrex::Real(5.0e-5), amrex::Real(5.0e-5),
         amrex::Real(2.0e-5), amrex::Real(3.0e-5),
         gamr1, gamr2, gams1, gams2, gamg1, gamg2}};
}

void expect_close (const char* helper_name,
                   const int idx,
                   const SAMKernelCase& test_case,
                   const amrex::Real host_value,
                   const amrex::Real device_value,
                   const amrex::Real tol)
{
    EXPECT_LE(std::abs(device_value - host_value), tol)
        << "helper=" << helper_name
        << " sample_index=" << idx
        << " mode=" << test_case.mode
        << " tabs=" << test_case.tabs
        << " rho=" << test_case.rho
        << " qpr=" << test_case.qpr
        << " qps=" << test_case.qps
        << " qpg=" << test_case.qpg
        << " qci_avg=" << test_case.qci_avg
        << " reduced_flux=" << test_case.reduced_flux
        << " host=" << host_value
        << " device=" << device_value
        << " normalized_error=" << normalized_error(device_value, host_value)
        << " tolerance=" << tol;
}

} // namespace

// Motivation:
// Production SAM calls these helpers from AMReX kernels. This host/device
// sweep covers threshold branches, no-ice mode, zero and positive precip,
// capped and clipped fall speeds, and representative coefficient states.
TEST(SAMKernel, SAMUtilsSweptHostDeviceEquivalence)
{
    const std::vector<SAMKernelCase> cases = make_kernel_cases();
    std::vector<SAMKernelOutputs> host_outputs(cases.size());

    for (int idx = 0; idx < static_cast<int>(cases.size()); ++idx) {
        host_outputs[idx] = host_reference(cases[idx]);
    }

    amrex::Gpu::DeviceVector<SAMKernelCase> d_cases(cases.size());
    amrex::Gpu::DeviceVector<SAMKernelOutputs> d_outputs(cases.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, cases.begin(), cases.end(), d_cases.begin());

    launch_sam_helper_kernel(static_cast<int>(cases.size()), d_cases.data(), d_outputs.data());

    std::vector<SAMKernelOutputs> device_outputs(cases.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_outputs.begin(), d_outputs.end(), device_outputs.begin());

    for (int idx = 0; idx < static_cast<int>(cases.size()); ++idx) {
        SCOPED_TRACE("sample_index=" + std::to_string(idx));

        expect_close("sam_cloud_liquid_fraction", idx, cases[idx],
                     host_outputs[idx].cloud_liquid_fraction, device_outputs[idx].cloud_liquid_fraction,
                     roundoff_tol(host_outputs[idx].cloud_liquid_fraction));
        expect_close("sam_precip_rain_fraction", idx, cases[idx],
                     host_outputs[idx].rain_fraction, device_outputs[idx].rain_fraction,
                     roundoff_tol(host_outputs[idx].rain_fraction));
        expect_close("sam_graupel_fraction", idx, cases[idx],
                     host_outputs[idx].graupel_fraction, device_outputs[idx].graupel_fraction,
                     roundoff_tol(host_outputs[idx].graupel_fraction));
        expect_close("sam_mixed_qsat", idx, cases[idx],
                     host_outputs[idx].mixed_qsat, device_outputs[idx].mixed_qsat,
                     roundoff_tol(host_outputs[idx].mixed_qsat));
        expect_close("sam_mixed_dqsat_dT", idx, cases[idx],
                     host_outputs[idx].mixed_dqsat, device_outputs[idx].mixed_dqsat,
                     roundoff_tol(host_outputs[idx].mixed_dqsat));
        expect_close("sam_newton_residual_derivative", idx, cases[idx],
                     host_outputs[idx].residual_derivative, device_outputs[idx].residual_derivative,
                     roundoff_tol(host_outputs[idx].residual_derivative));
        expect_close("sam_autoconversion_rates.dqca", idx, cases[idx],
                     host_outputs[idx].autoconv_cloud, device_outputs[idx].autoconv_cloud,
                     roundoff_tol(host_outputs[idx].autoconv_cloud));
        expect_close("sam_autoconversion_rates.dqia", idx, cases[idx],
                     host_outputs[idx].autoconv_ice, device_outputs[idx].autoconv_ice,
                     roundoff_tol(host_outputs[idx].autoconv_ice));
        expect_close("sam_precip_evaporation_rates.dqpr", idx, cases[idx],
                     host_outputs[idx].evap_rain, device_outputs[idx].evap_rain,
                     pow_sqrt_tol(host_outputs[idx].evap_rain));
        expect_close("sam_precip_evaporation_rates.dqps", idx, cases[idx],
                     host_outputs[idx].evap_snow, device_outputs[idx].evap_snow,
                     pow_sqrt_tol(host_outputs[idx].evap_snow));
        expect_close("sam_precip_evaporation_rates.dqpg", idx, cases[idx],
                     host_outputs[idx].evap_graupel, device_outputs[idx].evap_graupel,
                     pow_sqrt_tol(host_outputs[idx].evap_graupel));
        expect_close("sam_compute_coefficient_row.evapr1", idx, cases[idx],
                     host_outputs[idx].coeff_evapr1, device_outputs[idx].coeff_evapr1,
                     pow_sqrt_tol(host_outputs[idx].coeff_evapr1));
        expect_close("sam_compute_coefficient_row.evaps1", idx, cases[idx],
                     host_outputs[idx].coeff_evaps1, device_outputs[idx].coeff_evaps1,
                     pow_sqrt_tol(host_outputs[idx].coeff_evaps1));
        expect_close("sam_compute_coefficient_row.evapg1", idx, cases[idx],
                     host_outputs[idx].coeff_evapg1, device_outputs[idx].coeff_evapg1,
                     pow_sqrt_tol(host_outputs[idx].coeff_evapg1));
        expect_close("sam_cloud_ice_terminal_velocity", idx, cases[idx],
                     host_outputs[idx].ice_velocity, device_outputs[idx].ice_velocity,
                     pow_sqrt_tol(host_outputs[idx].ice_velocity));
        expect_close("sam_precip_flux_from_face_state", idx, cases[idx],
                     host_outputs[idx].precip_flux, device_outputs[idx].precip_flux,
                     pow_sqrt_tol(host_outputs[idx].precip_flux));
        expect_close("sam_precip_flux_density_corrected", idx, cases[idx],
                     host_outputs[idx].density_corrected_flux, device_outputs[idx].density_corrected_flux,
                     pow_sqrt_tol(host_outputs[idx].density_corrected_flux));
        expect_close("sam_face_average_state.rho_avg", idx, cases[idx],
                     host_outputs[idx].face_rho, device_outputs[idx].face_rho,
                     roundoff_tol(host_outputs[idx].face_rho));
        expect_close("sam_face_average_state.tabs_avg", idx, cases[idx],
                     host_outputs[idx].face_tabs, device_outputs[idx].face_tabs,
                     roundoff_tol(host_outputs[idx].face_tabs));
        expect_close("sam_face_average_state.qci_avg", idx, cases[idx],
                     host_outputs[idx].face_qci, device_outputs[idx].face_qci,
                     roundoff_tol(host_outputs[idx].face_qci));
        expect_close("sam_face_average_state.qp_avg", idx, cases[idx],
                     host_outputs[idx].face_qp, device_outputs[idx].face_qp,
                     roundoff_tol(host_outputs[idx].face_qp));
        EXPECT_EQ(device_outputs[idx].substeps, host_outputs[idx].substeps)
            << "helper=sam_substep_count_from_reduced_flux sample_index=" << idx;
    }
}