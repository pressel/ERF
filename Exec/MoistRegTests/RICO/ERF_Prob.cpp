/**
 * @file ERF_Prob.cpp
 * 
 * Implementation file for the RICO (Rain In Cumulus over Ocean) case.
 * This case represents shallow cumulus clouds in the trade-wind region
 * of the Caribbean. It is based on the specifications from 
 * van Zanten et al. (2011) in the RICO intercomparison study.
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
 * 
 * @param problo Lower bounds of the domain
 * @param probhi Upper bounds of the domain
 */
Problem::Problem (const Real* problo, const Real* probhi)
{
    // Parse params from input file
    ParmParse pp("prob");
    pp.query("rho_0", parms.rho_0);        // Reference density
    pp.query("T_0", parms.T_0);            // Reference temperature
    pp.query("A_0", parms.A_0);            // Scalar parameter
    pp.query("KE_0", parms.KE_0);          // Initial turbulent kinetic energy
    
    // Mean horizontal velocity components
    pp.query("U_0", parms.U_0);            // Initial mean U velocity
    pp.query("V_0", parms.V_0);            // Initial mean V velocity
    pp.query("W_0", parms.W_0);            // Initial mean W velocity
    
    // Random perturbation magnitudes
    pp.query("U_0_Pert_Mag", parms.U_0_Pert_Mag);      // U velocity perturbation magnitude
    pp.query("V_0_Pert_Mag", parms.V_0_Pert_Mag);      // V velocity perturbation magnitude
    pp.query("W_0_Pert_Mag", parms.W_0_Pert_Mag);      // W velocity perturbation magnitude
    pp.query("T_0_Pert_Mag", parms.T_0_Pert_Mag);      // Temperature perturbation magnitude
    pp.query("qv_0_Pert_Mag", parms.qv_0_Pert_Mag);    // Water vapor perturbation magnitude
    
    // Structured perturbation parameters
    pp.query("pert_deltaU", parms.pert_deltaU);          // Amplitude of U structured perturbation
    pp.query("pert_deltaV", parms.pert_deltaV);          // Amplitude of V structured perturbation
    pp.query("pert_periods_U", parms.pert_periods_U);    // Number of periods in Y direction for U
    pp.query("pert_periods_V", parms.pert_periods_V);    // Number of periods in X direction for V
    pp.query("pert_ref_height", parms.pert_ref_height);  // Reference height for perturbations
    
    // Calculate derived perturbation parameters
    parms.aval = parms.pert_periods_U * 2.0 * PI / (probhi[1] - problo[1]);
    parms.bval = parms.pert_periods_V * 2.0 * PI / (probhi[0] - problo[0]);
    parms.ufac = parms.pert_deltaU * std::exp(0.5) / parms.pert_ref_height;
    parms.vfac = parms.pert_deltaV * std::exp(0.5) / parms.pert_ref_height;
    
    //===========================================================================
    // READ USER-DEFINED INPUTS FOR RICO CASE
    //===========================================================================
    
    // Temperature and moisture forcing parameters
    pp.query("advection_heating_rate", parms.advection_heating_rate);  // Large-scale cooling rate (K/s)
    pp.query("restart_time", parms.restart_time);                     // Restart time if applicable
    pp.query("source_cutoff", parms.cutoff);                          // Height cutoff for thermal forcing
    pp.query("source_cutoff_transition", parms.cutoff_transition);    // Transition layer thickness
    pp.query("advection_moisture_rate", parms.advection_moisture_rate); // Large-scale drying rate (g/kg/s)
    pp.query("moisture_source_cutoff", parms.moisture_cutoff);        // Height cutoff for moisture forcing
    pp.query("moisture_source_cutoff_transition", parms.moisture_cutoff_transition); // Moisture transition layer
    
    // Subsidence parameters
    pp.query("wbar_sub_max", parms.wbar_sub_max);        // Maximum subsidence velocity
    pp.query("wbar_cutoff_max", parms.wbar_cutoff_max);  // Height of maximum subsidence
    pp.query("wbar_cutoff_min", parms.wbar_cutoff_min);  // Height where subsidence reduces to zero
    
    // Safety check for subsidence parameters
    AMREX_ASSERT_WITH_MESSAGE(parms.wbar_cutoff_min > parms.wbar_cutoff_max, 
                             "ERROR: wbar_cutoff_min < wbar_cutoff_max");
    
    // TKE initialization option
    pp.query("custom_TKE", parms.custom_TKE);
    
    //===========================================================================
    // Initialize base thermodynamic parameters
    init_base_parms(parms.rho_0, parms.T_0);
}

/**
 * Initialize custom perturbations for the RICO case
 * 
 * This function applies both random and structured perturbations to the initial state
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
    const Real rdOcp = sc.rdOcp;  // R_d/c_p ratio for thermodynamic calculations
    
    // Apply perturbations to cell-centered variables
    ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        // Geometry
        const Real* prob_lo = geomdata.ProbLo();
        const Real* prob_hi = geomdata.ProbHi();
        const Real* dx = geomdata.CellSize();
        const Real x = prob_lo[0] + (i + 0.5) * dx[0];
        const Real y = prob_lo[1] + (j + 0.5) * dx[1];
        const Real z = prob_lo[2] + (k + 0.5) * dx[2];
        
        // Define a point (xc,yc,zc) at the center of the domain
        const Real xc = 0.5 * (prob_lo[0] + prob_hi[0]);
        const Real yc = 0.5 * (prob_lo[1] + prob_hi[1]);
        const Real zc = 0.5 * (prob_lo[2] + prob_hi[2]);
        const Real r  = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));
        
        //-----------------------------------------------------------------------
        // Add temperature perturbations 
        // (Constant pressure perturbations that result in density changes)
        //-----------------------------------------------------------------------
        if ((z <= parms_d.pert_ref_height) && (parms_d.T_0_Pert_Mag != 0.0)) {
            Real rhotheta  = state(i,j,k,RhoTheta_comp);
            Real rho       = state(i,j,k,Rho_comp);
            Real qv        = state(i,j,k,RhoQ1_comp) / rho;
            Real Told      = getTgivenRandRTh(rho,rhotheta,qv);
            Real P         = getPgivenRTh(rhotheta,qv);
            Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
            Real Tpert    = (rand_double*2.0 - 1.0)*parms_d.T_0_Pert_Mag;
            Real Tnew     = Told + Tpert;
            Real theta_new = getThgivenPandT(Tnew,P,rdOcp);
            Real rhonew    = getRhogivenThetaPress(theta_new,P,rdOcp,qv);
            state_pert(i, j, k, Rho_comp) = rhonew - rho;
            
            // Note: we do not perturb rho*theta directly
            state_pert(i, j, k, RhoTheta_comp) = 0.0;
            
            // Instead of perturbing (rho*theta) we perturb T and hold (rho*theta) fixed,
            // which ends up being stored as a perturbation in rho
        }
        
        //-----------------------------------------------------------------------
        // Set scalar field = A_0*exp(-10r^2) - a Gaussian blob centered in domain
        //-----------------------------------------------------------------------
        state_pert(i, j, k, RhoScalar_comp) = parms_d.A_0 * exp(-10.*r*r);
        
        //-----------------------------------------------------------------------
        // Set an initial value for turbulent kinetic energy (TKE)
        //-----------------------------------------------------------------------
        if (parms_d.custom_TKE) {
            // TKE proportional to height from surface (higher near surface)
            state_pert(i, j, k, RhoKE_comp) = (1.0 - z/prob_hi[2]) * r_hse(i,j,k);
        } else {
            // Constant TKE throughout domain
            state_pert(i, j, k, RhoKE_comp) = parms_d.KE_0;
        }
        
        //-----------------------------------------------------------------------
        // Apply moisture perturbations if moisture is enabled
        //-----------------------------------------------------------------------
        if (use_moisture) {
            state_pert(i, j, k, RhoQ1_comp) = 0.0;  // Default: no perturbation to water vapor
            state_pert(i, j, k, RhoQ2_comp) = 0.0;  // Default: no perturbation to cloud water
            
            // Add random perturbations to moisture in lower levels to help trigger convection
            if ((z <= parms_d.pert_ref_height) && (parms_d.qv_0_Pert_Mag != 0.0))
            {
                Real rhoold = state(i,j,k,Rho_comp);
                Real rhonew = rhoold + state_pert(i,j,k,Rho_comp);
                Real qvold = state(i,j,k,RhoQ1_comp) / rhoold;
                Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
                Real qvnew = qvold + (rand_double*2.0 - 1.0)*parms_d.qv_0_Pert_Mag;
                state_pert(i, j, k, RhoQ1_comp) = rhonew * qvnew - rhoold * qvold;
            }
        }
    });
    
    //-----------------------------------------------------------------------
    // Set the x-velocity perturbations
    //-----------------------------------------------------------------------
    ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx = geomdata.CellSize();
        const Real y = prob_lo[1] + (j + 0.5) * dx[1];
        const Real z = prob_lo[2] + (k + 0.5) * dx[2];
        
        // Start with mean horizontal velocity (typically -9.9 m/s for RICO)
        x_vel_pert(i, j, k) = parms_d.U_0;
        
        // Add random perturbations near surface to help trigger turbulence
        if ((z <= parms_d.pert_ref_height) && (parms_d.U_0_Pert_Mag != 0.0))
        {
            Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
            Real x_vel_prime = (rand_double*2.0 - 1.0)*parms_d.U_0_Pert_Mag;
            x_vel_pert(i, j, k) += x_vel_prime;
        }
        
        // Add structured wave-like perturbations in y-direction if specified
        if (parms_d.pert_deltaU != 0.0)
        {
            const amrex::Real yl = y - prob_lo[1];
            const amrex::Real zl = z / parms_d.pert_ref_height;
            const amrex::Real damp = std::exp(-0.5 * zl * zl);  // Height damping factor
            x_vel_pert(i, j, k) += parms_d.ufac * damp * z * std::cos(parms_d.aval * yl);
        }
    });
    
    //-----------------------------------------------------------------------
    // Set the y-velocity perturbations
    //-----------------------------------------------------------------------
    ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx = geomdata.CellSize();
        const Real x = prob_lo[0] + (i + 0.5) * dx[0];
        const Real z = prob_lo[2] + (k + 0.5) * dx[2];
        
        // Start with mean horizontal velocity (typically -3.8 m/s for RICO)
        y_vel_pert(i, j, k) = parms_d.V_0;
        
        // Add random perturbations near surface to help trigger turbulence
        if ((z <= parms_d.pert_ref_height) && (parms_d.V_0_Pert_Mag != 0.0))
        {
            Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
            Real y_vel_prime = (rand_double*2.0 - 1.0)*parms_d.V_0_Pert_Mag;
            y_vel_pert(i, j, k) += y_vel_prime;
        }
        
        // Add structured wave-like perturbations in x-direction if specified
        if (parms_d.pert_deltaV != 0.0)
        {
            const amrex::Real xl = x - prob_lo[0];
            const amrex::Real zl = z / parms_d.pert_ref_height;
            const amrex::Real damp = std::exp(-0.5 * zl * zl);  // Height damping factor
            y_vel_pert(i, j, k) += parms_d.vfac * damp * z * std::cos(parms_d.bval * xl);
        }
    });
    
    //-----------------------------------------------------------------------
    // Set the z-velocity perturbations
    //-----------------------------------------------------------------------
    ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        const int dom_lo_z = geomdata.Domain().smallEnd()[2];
        const int dom_hi_z = geomdata.Domain().bigEnd()[2];
        
        // Set the z-velocity
        if (k == dom_lo_z || k == dom_hi_z+1)
        {
            // Enforce w=0 at domain boundaries
            z_vel_pert(i, j, k) = 0.0;
        }
        else if (parms_d.W_0_Pert_Mag != 0.0)
        {
            // Add random perturbations to vertical velocity throughout domain
            Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
            Real z_vel_prime = (rand_double*2.0 - 1.0)*parms_d.W_0_Pert_Mag;
            z_vel_pert(i, j, k) = parms_d.W_0 + z_vel_prime;
        }
    });
}

//=============================================================================
// USER-DEFINED FUNCTION: Temperature forcing
//=============================================================================
/**
 * Updates temperature tendency sources to represent large-scale advective cooling
 * For RICO: Constant cooling of -2.5 K/day throughout the domain
 */
void
Problem::update_rhotheta_sources (const Real& /*time*/,
                                  Vector<Real>& src,
                                  Gpu::DeviceVector<Real>& d_src,
                                  const Geometry& geom,
                                  std::unique_ptr<MultiFab>& z_phys_cc)
{
    if (src.empty()) return;
    const int khi       = geom.Domain().bigEnd()[2];
    const Real* prob_lo = geom.ProbLo();
    const auto dx       = geom.CellSize();
    
    // If using terrain, get the physical heights at cell centers
    if (z_phys_cc) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_cc);
    }
    
    // For RICO: Apply constant radiative cooling of -2.5 K/day throughout the domain
    // Convert from K/day to K/s by dividing by seconds per day
    const Real cooling_rate = -2.5/86400.0; // -2.5 K/day = -2.89e-5 K/s
    
    for (int k = 0; k <= khi; k++) {
        src[k] = cooling_rate;
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, src.begin(), src.end(), d_src.begin());
}

//=============================================================================
// USER-DEFINED FUNCTION: Moisture forcing
//=============================================================================
/**
 * Updates moisture tendency sources to represent large-scale advection effects
 * For RICO: -1.0 g/kg/day near surface and a source of +0.3456 g/kg/day around 2980m and 4000m
 */
void
Problem::update_rhoqt_sources (const Real& /*time*/,
                               Vector<Real>& qsrc,
                               Gpu::DeviceVector<Real>& d_qsrc,
                               const Geometry& geom,
                               std::unique_ptr<MultiFab>& z_phys_cc)
{
    if (qsrc.empty()) return;
    const int khi       = geom.Domain().bigEnd()[2];
    const Real* prob_lo = geom.ProbLo();
    const auto dx       = geom.CellSize();
    
    // If using terrain, get the physical heights at cell centers
    if (z_phys_cc) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_cc);
    }
    
    // For RICO: Apply surface drying and upper level moistening
    // Constants from van Zanten et al. (2011) RICO LES intercomparison
    const Real drying_rate = -1.0/86400.0;     // -1.0 g/kg/day converted to g/kg/s
    const Real moistening_rate = 0.3456/86400.0; // +0.3456 g/kg/day converted to g/kg/s
    
    // Heights where upper-level moistening is applied
    const Real moisture_source_lower = 2800.0; // m, around 2980m in table
    const Real moisture_source_upper = 4200.0; // m, around 4000m in table
    
    for (int k = 0; k <= khi; k++) {
        const Real z_cc = (z_phys_cc) ? zlevels[k] : prob_lo[2] + (k+0.5)* dx[2];
        
        // Default: apply drying everywhere (representing large-scale subsidence)
        qsrc[k] = drying_rate;
        
        // Apply moisture source in upper levels to represent horizontal advection of moist air
        if (z_cc > moisture_source_lower && z_cc < moisture_source_upper) {
            qsrc[k] = moistening_rate;
        }
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, qsrc.begin(), qsrc.end(), d_qsrc.begin());
}

//=============================================================================
// USER-DEFINED FUNCTION: Subsidence velocity
//=============================================================================
/**
 * Updates the large-scale subsidence velocity profile
 * For RICO: -0.5 cm/s subsidence between 2260m and 4000m
 */
void
Problem::update_w_subsidence (const Real& /*time*/,
                              Vector<Real>& wbar,
                              Gpu::DeviceVector<Real>& d_wbar,
                              const Geometry& geom,
                              std::unique_ptr<MultiFab>& z_phys_nd)
{
    if (wbar.empty()) return;
    const int khi       = geom.Domain().bigEnd()[2] + 1; // lives on z-faces
    const Real* prob_lo = geom.ProbLo();
    const auto dx       = geom.CellSize();
    
    // If using terrain, get the physical heights at cell centers
    if (z_phys_nd) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_nd);
    }
    
    // For RICO: Apply subsidence of -0.5 cm/s at heights between 2260m and 4000m
    // Constants from van Zanten et al. (2011) RICO LES intercomparison
    const Real subsidence_velocity = -0.005; // -0.5 cm/s = -0.005 m/s
    const Real subsidence_start = 2260.0;    // m, height where subsidence begins
    const Real subsidence_end = 4000.0;      // m, height where subsidence ends
    
    // At surface, subsidence is always zero
    wbar[0] = 0.0;
    
    for (int k = 1; k <= khi; k++) {
        const Real z_cc = (z_phys_nd) ? zlevels[k] : prob_lo[2] + k*dx[2];
        
        // Default: no subsidence
        wbar[k] = 0.0;
        
        // Apply subsidence in middle levels as specified in RICO case
        if (z_cc >= subsidence_start && z_cc <= subsidence_end) {
            wbar[k] = subsidence_velocity;
        }
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, wbar.begin(), wbar.end(), d_wbar.begin());
}

//=============================================================================
// USER-DEFINED FUNCTION: Geostrophic wind profile
//=============================================================================
/**
 * Updates the geostrophic wind profile for Coriolis force calculation
 * For RICO: Constant u_g = -9.9 m/s and v_g = -3.8 m/s throughout domain
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
    const int khi       = geom.Domain().bigEnd()[2];
    const Real* prob_lo = geom.ProbLo();
    const auto dx       = geom.CellSize();
    
    // If using terrain, get the physical heights at cell centers
    if (z_phys_cc) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_cc);
    }

    // For RICO: Use a constant geostrophic wind profile matching the mean flow
    // Values from van Zanten et al. (2011) RICO LES intercomparison
    const Real u_geostrophic = -9.9; // m/s
    const Real v_geostrophic = -3.8; // m/s
    
    // Apply constant geostrophic wind throughout domain
    for (int k = 0; k <= khi; k++) {
        u_geos[k] = u_geostrophic;
        v_geos[k] = v_geostrophic;
    }
    
    // Copy from host version to device version
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, u_geos.begin(), u_geos.end(), d_u_geos.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, v_geos.begin(), v_geos.end(), d_v_geos.begin());
}