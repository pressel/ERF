/**
 * @file ERF_Prob.cpp
 * 
 * Implementation file for the RICO (Rain In Cumulus over Ocean) case.
 * This case represents shallow cumulus clouds in the trade-wind region
 * of the Caribbean. Based on van Zanten et al. (2011) RICO intercomparison study.
 */
#include "ERF_Prob.H"
#include "AMReX_Random.H"
#include <Utils/ERF_ParFunctions.H>
using namespace amrex;

std::unique_ptr<ProblemBase>
amrex_probinit (const amrex_real* problo, const amrex_real* probhi)
{
    return std::make_unique<Problem>(problo, probhi);
}

/**
 * Problem constructor - initializes parameters for the RICO case
 */
Problem::Problem (const Real* problo, const Real* probhi)
{
    // Parse params from input file
    ParmParse pp("prob");
    pp.query("rho_0", parms.rho_0);        // Reference density
    pp.query("T_0", parms.T_0);            // Reference temperature
    pp.query("KE_0", parms.KE_0);          // Initial turbulent kinetic energy
    
    // Random perturbation magnitudes
    pp.query("T_0_Pert_Mag", parms.T_0_Pert_Mag);      // Temperature perturbation magnitude
    pp.query("pert_ref_height", parms.pert_ref_height);  // Reference height for perturbations
    
    // Options specific to the TKE initialization
    pp.query("custom_TKE", parms.custom_TKE);
    
    //===========================================================================
    // Initialize base thermodynamic parameters
    init_base_parms(parms.rho_0, parms.T_0);
}

/**
 * Utility function for linear interpolation between profile points
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real interpolate_linear(Real z, const Real* z_levels, const Real* values, int n_levels)
{
    // Handle boundary cases
    if (z <= z_levels[0]) return values[0];
    if (z >= z_levels[n_levels-1]) return values[n_levels-1];
    
    // Find the interval for interpolation
    for (int i = 0; i < n_levels-1; i++) {
        if (z >= z_levels[i] && z <= z_levels[i+1]) {
            Real frac = (z - z_levels[i]) / (z_levels[i+1] - z_levels[i]);
            return values[i] + frac * (values[i+1] - values[i]);
        }
    }
    return values[n_levels-1]; // fallback
}

/**
 * Initialize custom perturbations for the RICO case
 * 
 * This function applies random perturbations to temperature only
 * to help trigger convection in the model.
 */
void
Problem::init_custom_pert (
    const Box&  bx,
    const Box& xbx,
    const Box& ybx,
    const Box& zbx,
    Array4<Real const> const& state,
    Array4<Real      > const& state_pert,
    Array4<Real      > const& x_vel_pert,
    Array4<Real      > const& y_vel_pert,
    Array4<Real      > const& z_vel_pert,
    Array4<Real      > const& r_hse,
    Array4<Real      > const& /*p_hse*/,
    Array4<Real const> const& /*z_nd*/,
    Array4<Real const> const& /*z_cc*/,
    GeometryData const& geomdata,
    Array4<Real const> const& /*mf_m*/,
    Array4<Real const> const& /*mf_u*/,
    Array4<Real const> const& /*mf_v*/,
    const SolverChoice& sc)
{
    const bool use_moisture = (sc.moisture_type != MoistureType::None);
    
    // Apply perturbations to cell-centered variables
    ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        // Geometry
        const Real* prob_lo = geomdata.ProbLo();
        const Real* prob_hi = geomdata.ProbHi();
        const Real* dx = geomdata.CellSize();
        const Real z = prob_lo[2] + (k + 0.5) * dx[2];
        
        // Add potential temperature perturbations in the lower atmosphere 
        // Simple approach: perturb theta, hold density fixed
        if ((z <= parms_d.pert_ref_height) && (parms_d.T_0_Pert_Mag != 0.0)) {
            Real rho = state(i,j,k,Rho_comp);
            Real rand_double = amrex::Random(engine); 
            Real theta_pert = (rand_double*2.0 - 1.0)*parms_d.T_0_Pert_Mag;
            
            // Perturb potential temperature, hold density fixed
            state_pert(i, j, k, Rho_comp) = 0.0;                    // No density perturbation
            state_pert(i, j, k, RhoTheta_comp) = rho * theta_pert;  // Direct theta perturbation
        } else {
            state_pert(i, j, k, Rho_comp) = 0.0;
            state_pert(i, j, k, RhoTheta_comp) = 0.0;
        }
        
        // Set initial turbulent kinetic energy (TKE)
        if (parms_d.custom_TKE) {
            state_pert(i, j, k, RhoKE_comp) = (1.0 - z/prob_hi[2]) * r_hse(i,j,k);
        } else {
            state_pert(i, j, k, RhoKE_comp) = parms_d.KE_0;
        }
        
        // Initialize moisture perturbations to zero if moisture is enabled
        if (use_moisture) {
            state_pert(i, j, k, RhoQ1_comp) = 0.0;
            state_pert(i, j, k, RhoQ2_comp) = 0.0;
        }
    });
    
    // Set velocities to zero (no perturbations)
    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        x_vel_pert(i, j, k) = 0.0;
    });
    
    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        y_vel_pert(i, j, k) = 0.0;
    });
    
    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        z_vel_pert(i, j, k) = 0.0;
    });
}


//=============================================================================
// Temperature forcing - Large-scale advective cooling + radiative cooling
//=============================================================================
/**
 * Updates temperature tendency sources for RICO: Constant cooling of -2.5 K/day
 * Applied at surface (0m) and top (4000m) with linear interpolation
 */
void
Problem::update_rhotheta_sources (const Real& /*time*/,
                                  Vector<Real>& src,
                                  Gpu::DeviceVector<Real>& d_src,
                                  const Geometry& geom,
                                  std::unique_ptr<MultiFab>& z_phys_cc)
{
    if (src.empty()) return;
    const int khi = geom.Domain().bigEnd()[2];
    const Real* prob_lo = geom.ProbLo();
    const auto dx = geom.CellSize();
    
    // If using terrain, get the physical heights at cell centers
    if (z_phys_cc) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_cc);
    }
    
    // RICO temperature forcing profile points (from van Zanten et al. 2011 Table 2)
    const Real z_points[] = {0.0, 4000.0};
    const Real temp_forcing[] = {-2.5/86400.0,    // -2.5 K/day converted to K/s
                                 -2.5/86400.0};   // Constant throughout
    const int n_points = 2;
    
    for (int k = 0; k <= khi; k++) {
        const Real z_cc = (z_phys_cc) ? zlevels[k] : prob_lo[2] + (k+0.5)*dx[2];
        
        // Linear interpolation (constant in this case)
        src[k] = interpolate_linear(z_cc, z_points, temp_forcing, n_points);
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, src.begin(), src.end(), d_src.begin());
}

//=============================================================================
// Moisture forcing - Large-scale advection effects
//=============================================================================
/**
 * Updates moisture tendency sources for RICO:
 * -1.0 g/kg/day at surface (0m)
 * +0.3456 g/kg/day at 2980m and 4000m
 * Linear interpolation between points
 */
void
Problem::update_rhoqt_sources (const Real& /*time*/,
                               Vector<Real>& qsrc,
                               Gpu::DeviceVector<Real>& d_qsrc,
                               const Geometry& geom,
                               std::unique_ptr<MultiFab>& z_phys_cc)
{
    if (qsrc.empty()) return;
    const int khi = geom.Domain().bigEnd()[2];
    const Real* prob_lo = geom.ProbLo();
    const auto dx = geom.CellSize();
    
    // If using terrain, get the physical heights at cell centers
    if (z_phys_cc) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_cc);
    }
    
    // RICO moisture forcing profile points (from van Zanten et al. 2011 Table 2)
    const Real z_points[] = {0.0, 2980.0, 4000.0};
    const Real qv_forcing[] = {-1.0/86400.0/1000.0,     // -1.0 g/kg/day converted to kg/kg/s
                               0.3456/86400.0/1000.0,   // +0.3456 g/kg/day at 2980m  
                               0.3456/86400.0/1000.0};  // +0.3456 g/kg/day at 4000m
    const int n_points = 3;
    
    for (int k = 0; k <= khi; k++) {
        const Real z_cc = (z_phys_cc) ? zlevels[k] : prob_lo[2] + (k+0.5)*dx[2];
        
        // Linear interpolation between specified points
        qsrc[k] = interpolate_linear(z_cc, z_points, qv_forcing, n_points);
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, qsrc.begin(), qsrc.end(), d_qsrc.begin());
}

//=============================================================================
// Subsidence velocity profile
//=============================================================================
/**
 * Updates the large-scale subsidence velocity profile for RICO:
 * 0.0 cm/s at surface (0m)
 * -0.5 cm/s from 2260m to 4000m
 * Linear interpolation between points
 */
void
Problem::update_w_subsidence (const Real& /*time*/,
                              Vector<Real>& wbar,
                              Gpu::DeviceVector<Real>& d_wbar,
                              const Geometry& geom,
                              std::unique_ptr<MultiFab>& z_phys_nd)
{
    if (wbar.empty()) return;
    const int khi = geom.Domain().bigEnd()[2] + 1; // lives on z-faces
    const Real* prob_lo = geom.ProbLo();
    const auto dx = geom.CellSize();
    
    // If using terrain, get the physical heights at nodes
    if (z_phys_nd) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_nd);
    }
    
    // RICO subsidence profile points (from van Zanten et al. 2011 Table 2)
    const Real z_points[] = {0.0, 2260.0, 4000.0};
    const Real w_subsidence[] = {0.0,     // 0 cm/s at surface
                                 -0.005,  // -0.5 cm/s converted to m/s at 2260m
                                 -0.005}; // -0.5 cm/s converted to m/s at 4000m
    const int n_points = 3;
    
    for (int k = 0; k <= khi; k++) {
        const Real z_nd = (z_phys_nd) ? zlevels[k] : prob_lo[2] + k*dx[2];
        
        // Linear interpolation between specified points
        wbar[k] = interpolate_linear(z_nd, z_points, w_subsidence, n_points);
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, wbar.begin(), wbar.end(), d_wbar.begin());
}

//=============================================================================
// Geostrophic wind profile
//=============================================================================
/**
 * Updates the geostrophic wind profile for RICO:
 * u_g: -9.9 m/s at surface (0m), -1.9 m/s at top (4000m)
 * v_g: -3.8 m/s constant throughout domain
 * Linear interpolation between points
 */
void
Problem::update_geostrophic_profile (const Real& /*time*/,
                                     Vector<Real>& u_geos,
                                     Gpu::DeviceVector<Real>& d_u_geos,
                                     Vector<Real>& v_geos,
                                     Gpu::DeviceVector<Real>& d_v_geos,
                                     const Geometry& geom,
                                     std::unique_ptr<MultiFab>& z_phys_cc)
{
    if (u_geos.empty()) return;
    const int khi = geom.Domain().bigEnd()[2];
    const Real* prob_lo = geom.ProbLo();
    const auto dx = geom.CellSize();
    
    // If using terrain, get the physical heights at cell centers
    if (z_phys_cc) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_cc);
    }
    
    // RICO geostrophic wind profile points (from van Zanten et al. 2011 Table 2)
    const Real z_points[] = {0.0, 4000.0};
    const Real u_geo_values[] = {-9.9,  // -9.9 m/s at surface
                                 -1.9}; // -1.9 m/s at top
    const Real v_geo_values[] = {-3.8,  // -3.8 m/s constant
                                 -3.8}; // -3.8 m/s constant
    const int n_points = 2;
    
    for (int k = 0; k <= khi; k++) {
        const Real z_cc = (z_phys_cc) ? zlevels[k] : prob_lo[2] + (k+0.5)*dx[2];
        
        // Linear interpolation for both components
        u_geos[k] = interpolate_linear(z_cc, z_points, u_geo_values, n_points);
        v_geos[k] = interpolate_linear(z_cc, z_points, v_geo_values, n_points);
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, u_geos.begin(), u_geos.end(), d_u_geos.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, v_geos.begin(), v_geos.end(), d_v_geos.begin());
}