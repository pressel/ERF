/** \file ERF_EddyViscosity.H */

#ifndef ERF_EDDY_VISCOSITY_H_
#define ERF_EDDY_VISCOSITY_H_

#include "AMReX_BCRec.H"
#include "ERF_SurfaceLayer.H"
#include "ERF_DataStruct.H"
#include "ERF_IndexDefines.H"
#include "ERF_Constants.H"
#include "ERF_EOS.H"
#include "ERF_MicrophysicsUtils.H"

// =============================================================================
// MOISTURE COMPONENT INDICES STRUCT
// =============================================================================
struct MoistureComponentIndices {
    int qv  = -1;  // Water vapor
    int qc  = -1;  // Cloud liquid water
    int qi  = -1;  // Cloud ice
    int qr  = -1;  // Rain
    int qs  = -1;  // Snow
    int qg  = -1;  // Graupel

    // Constructor for easy initialization
    MoistureComponentIndices(int qv_comp, int qc_comp, int qi_comp,
                           int qr_comp, int qs_comp, int qg_comp)
        : qv(qv_comp), qc(qc_comp), qi(qi_comp), qr(qr_comp), qs(qs_comp), qg(qg_comp) {}

    // Default constructor
    MoistureComponentIndices() = default;
};


void
ComputeTurbulentViscosity (const amrex::MultiFab& xvel ,
                           const amrex::MultiFab& yvel ,
                           amrex::Vector<std::unique_ptr<amrex::MultiFab>>& Tau_lev,
                           const amrex::MultiFab& cons_in,
                           const amrex::MultiFab& wdist,
                           amrex::MultiFab& eddyViscosity,
                           amrex::MultiFab& Hfx1,
                           amrex::MultiFab& Hfx2,
                           amrex::MultiFab& Hfx3,
                           amrex::MultiFab& Diss,
                           const amrex::Geometry& geom,
                           amrex::Vector<std::unique_ptr<amrex::MultiFab>>& mapfac,
                           const std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                           const SolverChoice& solverChoice,
                           std::unique_ptr<SurfaceLayer>& SurfLayer,
                           const amrex::FArrayBox* z_0,
                           const bool& use_terrain_fitted_coords,
                           const bool& use_moisture,
                           int level,
                           const amrex::BCRec* bc_ptr,
                           bool vert_only = false);

// =============================================================================
// MOIST TURBULENCE FUNCTIONS
// =============================================================================

/**
 * Extract moisture variables and partition into liquid/ice phases
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void GetMoistureAtLevel(int i, int j, int k,
                       const amrex::Array4<amrex::Real const>& cell_data,
                       amrex::Real& qv, amrex::Real& qc_liquid, amrex::Real& qc_ice,
                       const MoistureComponentIndices& moisture_indices)
{
    qv = 0.0; qc_liquid = 0.0; qc_ice = 0.0;

    // Water vapor
    if (moisture_indices.qv >= 0) {
        qv = cell_data(i,j,k,moisture_indices.qv)/cell_data(i,j,k,Rho_comp);
    }

    // Cloud liquid water
    if (moisture_indices.qc >= 0) {
        qc_liquid = cell_data(i,j,k,moisture_indices.qc)/cell_data(i,j,k,Rho_comp);
    }

    // Cloud ice (only if separate ice species exists)
    if (moisture_indices.qi >= 0) {
        qc_ice = cell_data(i,j,k,moisture_indices.qi)/cell_data(i,j,k,Rho_comp);
    }
    // No temperature-based partitioning - respect the microphysics scheme's decisions

    // Add precipitating species
    if (moisture_indices.qr >= 0) {  // Rain (always liquid)
        qc_liquid += cell_data(i,j,k,moisture_indices.qr)/cell_data(i,j,k,Rho_comp);
    }
    if (moisture_indices.qs >= 0) {  // Snow (always ice)
        qc_ice += cell_data(i,j,k,moisture_indices.qs)/cell_data(i,j,k,Rho_comp);
    }
    if (moisture_indices.qg >= 0) {  // Graupel (always ice)
        qc_ice += cell_data(i,j,k,moisture_indices.qg)/cell_data(i,j,k,Rho_comp);
    }
}

/**
 * Compute virtual potential temperature with moisture loading effects
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real ComputeVirtualPotentialTemperature(amrex::Real theta, amrex::Real qv,
                                              amrex::Real qc_liquid, amrex::Real qc_ice)
{
    amrex::Real qc_total = qc_liquid + qc_ice;
    return theta * (1.0 + 0.61*qv - qc_total);
}

/**
 * Compute moist stratification accounting for conditional instability
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real ComputeMoistStratification(int i, int j, int k,
                                      const amrex::Array4<amrex::Real const>& cell_data,
                                      amrex::Real dzInv, amrex::Real abs_g, amrex::Real inv_theta0,
                                      const MoistureComponentIndices& moisture_indices)
{
    // Get moisture at three levels
    amrex::Real qv, qc_liquid, qc_ice;
    GetMoistureAtLevel(i, j, k, cell_data, qv, qc_liquid, qc_ice, moisture_indices);

    amrex::Real qv_upper, qcl_upper, qci_upper;
    GetMoistureAtLevel(i, j, k+1, cell_data, qv_upper, qcl_upper, qci_upper, moisture_indices);

    amrex::Real qv_lower, qcl_lower, qci_lower;
    GetMoistureAtLevel(i, j, k-1, cell_data, qv_lower, qcl_lower, qci_lower, moisture_indices);

    // Compute virtual potential temperature gradient
    amrex::Real theta_upper = cell_data(i,j,k+1,RhoTheta_comp)/cell_data(i,j,k+1,Rho_comp);
    amrex::Real theta_lower = cell_data(i,j,k-1,RhoTheta_comp)/cell_data(i,j,k-1,Rho_comp);

    amrex::Real theta_v_upper = ComputeVirtualPotentialTemperature(theta_upper, qv_upper, qcl_upper, qci_upper);
    amrex::Real theta_v_lower = ComputeVirtualPotentialTemperature(theta_lower, qv_lower, qcl_lower, qci_lower);

    amrex::Real dthetav_dz = 0.5 * (theta_v_upper - theta_v_lower) * dzInv;
    amrex::Real stratification = abs_g * dthetav_dz * inv_theta0;

    // Apply conditional instability correction
    amrex::Real total_condensate = qc_liquid + qc_ice;
    if (total_condensate > 1e-8) {
        amrex::Real T_current = getTgivenRandRTh(cell_data(i,j,k,Rho_comp),
                                                 cell_data(i,j,k,RhoTheta_comp), qv);

        // Phase-weighted effective latent heat
        amrex::Real liquid_fraction = qc_liquid / (total_condensate + 1e-12);
        amrex::Real ice_fraction = 1.0 - liquid_fraction;
        amrex::Real L_eff = liquid_fraction * L_v + ice_fraction * (L_v + lat_ice);

        // Phase-weighted saturation mixing ratio
        amrex::Real qsat_liquid = 0.0, qsat_ice = 0.0;
        amrex::Real pres_current = getPgivenRTh(cell_data(i,j,k,RhoTheta_comp), qv) * 0.01;

        erf_qsatw(T_current, pres_current, qsat_liquid);
        erf_qsati(T_current, pres_current, qsat_ice);

        amrex::Real qsat_eff = liquid_fraction * qsat_liquid + ice_fraction * qsat_ice;

        // Moist adiabatic lapse rate correction
        amrex::Real gamma_moist_factor = (1.0 + L_eff*qsat_eff/(R_d*T_current)) /
                                         (1.0 + L_eff*L_eff*qsat_eff/(Cp_d*R_v*T_current*T_current));

        stratification *= gamma_moist_factor;
    }

    return stratification;
}

/**
 * Compute stratification for Smagorinsky scheme (moist or dry)
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real ComputeStratificationForSmagorinsky(int i, int j, int k,
                                               const amrex::Array4<amrex::Real const>& cell_data,
                                               amrex::Real dzInv, amrex::Real abs_g, amrex::Real inv_theta0,
                                               bool use_moisture, int rho_qv_comp,
                                               const MoistureComponentIndices& moisture_indices)
{
    if (use_moisture && (rho_qv_comp >= 0)) {
        // Moist stratification with virtual temperature and conditional instability
        return ComputeMoistStratification(i, j, k, cell_data, dzInv, abs_g, inv_theta0, moisture_indices);
    } else {
        // Dry stratification (original approach)
        amrex::Real theta_hi = cell_data(i,j,k+1,RhoTheta_comp)/cell_data(i,j,k+1,Rho_comp);
        amrex::Real theta_lo = cell_data(i,j,k-1,RhoTheta_comp)/cell_data(i,j,k-1,Rho_comp);
        amrex::Real dtheta_dz = 0.5 * (theta_hi - theta_lo) * dzInv;
        return abs_g * dtheta_dz * inv_theta0;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ComputeSmnSmn (int& i, int& j, int& k,
               const amrex::Array4<amrex::Real const>& tau11,
               const amrex::Array4<amrex::Real const>& tau22,
               const amrex::Array4<amrex::Real const>& tau33,
               const amrex::Array4<amrex::Real const>& tau12,
               const amrex::Array4<amrex::Real const>& tau13,
               const amrex::Array4<amrex::Real const>& tau23)
{
    amrex::Real s11bar = tau11(i,j,k);
    amrex::Real s22bar = tau22(i,j,k);
    amrex::Real s33bar = tau33(i,j,k);
    amrex::Real s12bar = 0.25 * ( tau12(i  , j  , k  ) + tau12(i  , j+1, k  )
                                + tau12(i+1, j  , k  ) + tau12(i+1, j+1, k  ) );
    amrex::Real s13bar = 0.25 * ( tau13(i  , j  , k  ) + tau13(i  , j  , k+1)
                                + tau13(i+1, j  , k  ) + tau13(i+1, j  , k+1) );
    amrex::Real s23bar = 0.25 * ( tau23(i  , j  , k  ) + tau23(i  , j  , k+1)
                                + tau23(i  , j+1, k  ) + tau23(i  , j+1, k+1) );

    amrex::Real SmnSmn = s11bar*s11bar + s22bar*s22bar + s33bar*s33bar
                       + 2.0*s12bar*s12bar + 2.0*s13bar*s13bar + 2.0*s23bar*s23bar;

    return SmnSmn;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ComputeSmnSmn2D (int& i, int& j, int& k,
                 const amrex::Array4<amrex::Real const>& tau11,
                 const amrex::Array4<amrex::Real const>& tau22,
                 const amrex::Array4<amrex::Real const>& tau12)
{
    amrex::Real sdiff = tau11(i,j,k) - tau22(i,j,k);
    amrex::Real s12bar = 0.25 * ( tau12(i  , j  , k  ) + tau12(i  , j+1, k  )
                                + tau12(i+1, j  , k  ) + tau12(i+1, j+1, k  ) );
    return 0.5 * sdiff*sdiff + 2.0*s12bar*s12bar;
}
#endif