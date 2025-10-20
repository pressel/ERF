#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <AMReX_Array.H>

#include <ERF_ProbCommon.H>
#include <ERF_Constants.H>

struct ProbParm : ProbParmDefaults {
    amrex::Real p_surf     = 101780.0;   // Surface pressure (Pa) - RF02 spec: 1017.8 hPa
    amrex::Real sst        = 292.0;      // Sea surface temperature (K)
    amrex::Real u_star     = 0.25;       // Surface friction velocity (m/s) - RF02 spec
    amrex::Real divergence = 3.75e-6;    // Large-scale divergence (s^-1) - RF02 spec
    amrex::Real coriolis   = 1.03e-4;    // Coriolis parameter (s^-1)
    amrex::Real nc_cm3     = 55.0;       // Cloud droplet number concentration (cm^-3)
    amrex::Real z_damp     = 1250.0;     // Start height of Rayleigh damping (m)
    amrex::Real tau_damp   = 100.0;      // Rayleigh damping time scale (s)

    // Analytic thermodynamic and wind profile parameters
    amrex::Real zi                 = 795.0;   // Inversion height (m) - RF02 spec
    amrex::Real theta_l_bl         = 288.3;   // Liquid potential temperature below zi (K)
    amrex::Real theta_l_free_base  = 295.0;   // Offset above zi (K)
    amrex::Real qt_bl              = 9.45e-3; // Total water below zi (kg/kg)
    amrex::Real qt_free_asymptote  = 2.0e-3;  // Asymptotic qt aloft (kg/kg) - RF02 spec: 2 g/kg
    amrex::Real qt_free_excess     = 3.0e-3;  // Excess amount at the inversion (kg/kg)
    amrex::Real qt_free_scale      = 500.0;   // Scale height in exponential (m)

    amrex::Real geos_u    = 5.0;  // RF02 spec: 5.0 m/s (geostrophic U component)
    amrex::Real geos_v    = -5.5; // RF02 spec: -5.5 m/s (geostrophic V component)

    amrex::Real U_0 = 5.0;  // Initial U velocity (m/s)
    amrex::Real V_0 = -5.5; // Initial V velocity (m/s)
    amrex::Real W_0 = 0.0;

    // Radiation parameters (defaults; may be overridden via inputs)
    amrex::Real rad_F0        = 70.0;
    amrex::Real rad_F1        = 22.0;
    amrex::Real rad_kappa     = 85.0;
    amrex::Real rad_a         = 1.0;
    amrex::Real rho_i         = 1.12;  // RF02 spec: 1.12 kg/m³ (inversion density)
    amrex::Real rad_qc_thresh = 1.0e-7;

    // Perturbation parameters
    amrex::Real theta_pert_amp    = 0.1;   // K
    amrex::Real theta_pert_height = 200.0; // m
};

#endif
