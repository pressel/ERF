#include "ERF_Prob.H"

#include <ERF_EOS.H>
#include <ERF_Constants.H>
#include <ERF_HSEUtils.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Gpu.H>
#include <AMReX_Random.H>
#include <Utils/ERF_ParFunctions.H>
#include <algorithm>
#include <cctype>

using namespace amrex;

std::unique_ptr<ProblemBase>
amrex_probinit (const amrex_real* /*problo*/, const amrex_real* /*probhi*/)
{
    return std::make_unique<Problem>();
}

Problem::Problem ()
{
    ParmParse pp("prob");
    pp.query("rho_0", parms.rho_0);
    pp.query("T_0", parms.T_0);
    pp.query("U_0", parms.U_0);
    pp.query("V_0", parms.V_0);
    pp.query("W_0", parms.W_0);
    init_base_parms(parms.rho_0, parms.T_0);

    ParmParse pdy("prob.dycoms");
    pdy.query("p_surf",     parms.p_surf);
    pdy.query("sst",        parms.sst);
    pdy.query("u_star",     parms.u_star);
    pdy.query("divergence", parms.divergence);
    pdy.query("f_coriolis", parms.coriolis);
    pdy.query("nc",         parms.nc_cm3);
    pdy.query("zi",         parms.zi);
    pdy.query("theta_l_bl", parms.theta_l_bl);
    pdy.query("theta_l_free_base", parms.theta_l_free_base);
    pdy.query("qt_bl",              parms.qt_bl);
    pdy.query("qt_free_asymptote",  parms.qt_free_asymptote);
    pdy.query("qt_free_excess",     parms.qt_free_excess);
    pdy.query("qt_free_scale",      parms.qt_free_scale);
    pdy.query("geos_u",     parms.geos_u);
    pdy.query("geos_v",     parms.geos_v);

    pdy.query("rad_F0",          parms.rad_F0);
    pdy.query("rad_F1",          parms.rad_F1);
    pdy.query("rad_kappa",       parms.rad_kappa);
    pdy.query("rad_a",           parms.rad_a);
    pdy.query("rho_i",           parms.rho_i);
    pdy.query("rad_qc_threshold",parms.rad_qc_thresh);
    pdy.query("theta_pert_amp",    parms.theta_pert_amp);
    pdy.query("theta_pert_height", parms.theta_pert_height);

    parms.U_0 = parms.geos_u;
    parms.V_0 = parms.geos_v;
}

void
Problem::erf_init_dens_hse (MultiFab& rho_hse,
                            std::unique_ptr<MultiFab>& z_phys_nd,
                            std::unique_ptr<MultiFab>& /*z_phys_cc*/,
                            Geometry const& geom)
{
    /**
     * DyCOMS RF02 Dry Adiabatic Reference State (for use_moist_background=false)
     * ===========================================================================
     * 
     * **Purpose:** Compute smooth ρ₀(z) for anelastic simulations while preserving
     * RF02 moist discontinuities as perturbations in init_custom_pert().
     * 
     * **Strategy:** Use dry isentropic atmosphere with BL-averaged thermodynamics:
     * - θ₀ = constant (BL mean: ~288.5 K)
     * - Hydrostatic: dp/dz = -ρ₀·g
     * - Ideal gas: ρ₀ = p/(R_d·T)
     * - T = θ·(p/p₀)^(R/c_p)
     * 
     * **Result:** C^∞ smooth ρ₀(z) suitable for anelastic base state.
     * RF02 moist structure (sharp θ_ℓ, qt jumps) applied as ρ', θ' perturbations.
     */
    
    const int khi = geom.Domain().bigEnd()[2];
    
    // Extract z-levels
    Vector<Real> h_z_faces(khi+2);
    if (z_phys_nd) {
        bool filled = false;
        for (MFIter mfi(*z_phys_nd); mfi.isValid(); ++mfi) {
            const auto& z_arr = z_phys_nd->const_array(mfi);
            const Box& vbx = mfi.validbox();
            const int i0 = vbx.smallEnd(0);
            const int j0 = vbx.smallEnd(1);
            for (int k = 0; k <= khi+1; ++k) {
                h_z_faces[k] = z_arr(i0, j0, k);
            }
            filled = true;
            break;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(filled, "Failed to populate z-levels from z_phys_nd.");
    } else {
        const Real dz = geom.CellSize(2);
        const Real z0 = geom.ProbLo(2);
        for (int k = 0; k <= khi+1; ++k) {
            h_z_faces[k] = z0 + k * dz;
        }
    }
    
    // Use BL-representative dry adiabatic profile
    // θ₀ = constant ≈ 288.5 K (midpoint between surface 287K and pre-inversion 290K)
    const Real theta_ref = 288.5;  // Dry potential temperature reference [K]
    const Real rdOcp = R_d / Cp_d;
    
    Vector<Real> h_rho(khi+1);
    Real pressure = parms.p_surf;  // Start from RF02 surface pressure
    
    for (int k = 0; k <= khi; ++k) {
        const Real z_cc = 0.5 * (h_z_faces[k] + h_z_faces[k+1]);
        
        // Dry adiabatic: T = θ₀ · (p/p₀)^(R/c_p)
        const Real exner = getExnergivenP(pressure, rdOcp);
        const Real temp = theta_ref * exner;
        
        // Dry ideal gas law: ρ = p/(R_d·T)
        const Real rho_level = pressure / (R_d * temp);
        h_rho[k] = rho_level;
        
        // Hydrostatic integration: dp/dz = -ρ·g
        if (k < khi) {
            const Real dz = h_z_faces[k+1] - h_z_faces[k];
            pressure = std::max(pressure - rho_level * CONST_GRAV * dz, 1.0e3);
        }
    }
    
    // Copy to device and fill MultiFab
    Gpu::DeviceVector<Real> d_rho(khi+1);
    Gpu::copy(Gpu::hostToDevice, h_rho.begin(), h_rho.end(), d_rho.begin());
    Real* rho_ptr = d_rho.data();
    
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rho_hse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.growntilebox(1);
        auto rho_arr = rho_hse.array(mfi);
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const int kk = amrex::max(0, amrex::min(k, khi));
            rho_arr(i,j,k) = rho_ptr[kk];
        });
    }
}

void
Problem::erf_init_dens_hse_moist (MultiFab& rho_hse,
                                  std::unique_ptr<MultiFab>& z_phys_nd,
                                  Geometry const& geom)
{
    /**
     * DyCOMS RF02 Moist HSE with Smoothed Density Integration
     * ========================================================
     * 
     * **Problem:** RF02 sharp discontinuities (θ_ℓ: 288→295K, qt: 9.45→5 g/kg at zi=795m)
     * create unphysical density jumps when integrated hydrostatically.
     * 
     * **Solution:** Apply vertical smoothing (running mean) to thermodynamic profiles
     * during hydrostatic integration to compute smooth ρ₀(z), while preserving
     * sharp RF02 profiles in the actual state via init_custom_pert().
     * 
     * **Method:**
     * 1. Compute RF02 (θ_ℓ, qt) at each level
     * 2. Apply vertical smoothing filter (Δz ~ 50-100m) for HSE integration only
     * 3. Integrate smooth profiles hydrostatically → smooth ρ₀(z)
     * 4. init_custom_pert() applies sharp RF02 profiles → ρ' contains discontinuities
     * 
     * **Result:** Smooth base state ρ₀(z) for anelastic, sharp inversion preserved.
     */
    
    const int khi = geom.Domain().bigEnd()[2];

    Vector<Real> h_z_faces(khi+2);

    if (z_phys_nd) {
        bool filled = false;
        for (MFIter mfi(*z_phys_nd); mfi.isValid(); ++mfi) {
            const auto& z_arr = z_phys_nd->const_array(mfi);
            const Box& vbx = mfi.validbox();
            const int i0 = vbx.smallEnd(0);
            const int j0 = vbx.smallEnd(1);
            for (int k = 0; k <= khi+1; ++k) {
                h_z_faces[k] = z_arr(i0, j0, k);
            }
            filled = true;
            break;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(filled,
            "Failed to populate DyCOMS RF02 z-levels from z_phys_nd.");
    } else {
        const Real dz = geom.CellSize(2);
        const Real z0 = geom.ProbLo(2);
        for (int k = 0; k <= khi+1; ++k) {
            h_z_faces[k] = z0 + k * dz;
        }
    }

    // Step 1: Compute raw RF02 profiles at each level
    Vector<Real> theta_l_raw(khi+1);
    Vector<Real> qt_raw(khi+1);
    Vector<Real> z_cc(khi+1);
    
    for (int k = 0; k <= khi; ++k) {
        z_cc[k] = 0.5 * (h_z_faces[k] + h_z_faces[k+1]);
        theta_l_raw[k] = dycoms_theta_l(parms, z_cc[k]);
        qt_raw[k] = dycoms_qt(parms, z_cc[k]);
    }
    
    // Step 2: Apply vertical smoothing for HSE integration
    // Smoothing scale: ~5 grid points (typically 50-70m for DyCOMS grids)
    const int smooth_radius = 5;
    Vector<Real> theta_l_smooth(khi+1);
    Vector<Real> qt_smooth(khi+1);
    
    for (int k = 0; k <= khi; ++k) {
        const int k_lo = std::max(0, k - smooth_radius);
        const int k_hi = std::min(khi, k + smooth_radius);
        
        Real sum_theta = 0.0;
        Real sum_qt = 0.0;
        int count = 0;
        
        for (int kk = k_lo; kk <= k_hi; ++kk) {
            sum_theta += theta_l_raw[kk];
            sum_qt += qt_raw[kk];
            count++;
        }
        
        theta_l_smooth[k] = sum_theta / count;
        qt_smooth[k] = sum_qt / count;
    }

    // Step 3: Hydrostatic integration with SMOOTHED profiles
    Vector<Real> h_rho(khi+1);
    Real pressure = parms.p_surf;
    
    for (int k = 0; k <= khi; ++k) {
        Real rho_level, theta_level, temp_level, qv_level, qc_level;
        dycoms_compute_moist_state(pressure, theta_l_smooth[k], qt_smooth[k], R_d/Cp_d,
                                   rho_level, theta_level, temp_level,
                                   qv_level, qc_level);
        h_rho[k] = rho_level;

        if (k < khi) {
            const Real dz = h_z_faces[k+1] - h_z_faces[k];
            pressure = std::max(pressure - rho_level * CONST_GRAV * dz, 1.0e3);
        }
    }

    Gpu::DeviceVector<Real> d_rho(khi+1);
    Gpu::copy(Gpu::hostToDevice, h_rho.begin(), h_rho.end(), d_rho.begin());
    Real* rho_ptr = d_rho.data();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rho_hse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.growntilebox(1);
        auto rho_arr = rho_hse.array(mfi);
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const int kk = amrex::max(0, amrex::min(k, khi));
            rho_arr(i,j,k) = rho_ptr[kk];
        });
    }
}

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
    Array4<Real      > const& /*r_hse*/,
    Array4<Real      > const& p_hse,
    Array4<Real const> const& z_nd,
    Array4<Real const> const& /*z_cc*/,
    GeometryData const& geomdata,
    Array4<Real const> const& /*mf_m*/,
    Array4<Real const> const& /*mf_u*/,
    Array4<Real const> const& /*mf_v*/,
    const SolverChoice& sc)
{
    const bool has_terrain = z_nd.dataPtr() != nullptr;
    const Real rdOcp = sc.rdOcp;
    const bool use_moisture = (sc.moisture_type != MoistureType::None);
    const MoistureComponentIndices midx = sc.moisture_indices;
    const int dom_hi_z = geomdata.Domain().bigEnd()[2];

    const ProbParm parms_d = parms;

    ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, const RandomEngine& engine) noexcept
    {
        const Real z = dycoms_cell_height(i, j, k, z_nd, geomdata, has_terrain);
        Real theta_l = dycoms_theta_l(parms_d, z);
        const Real qt = amrex::max(dycoms_qt(parms_d, z), 0.0);

        if (z <= parms_d.theta_pert_height) {
            const Real perturb = parms_d.theta_pert_amp * (2.0 * Random(engine) - 1.0);
            theta_l = theta_l + perturb;
        }

        const Real pressure = p_hse(i,j,k);
        Real rho, theta, temp, qv, qc;
        dycoms_compute_moist_state(pressure, theta_l, qt, rdOcp,
                                   rho, theta, temp, qv, qc);

        static_cast<void>(temp);

        const Real rhotheta = rho * theta;
        Real rhoqv = rho * qv;
        Real rhoqc = rho * qc;

        const Real rho_bg      = state(i,j,k,Rho_comp);
        const Real rhotheta_bg = state(i,j,k,RhoTheta_comp);
        const Real rhoqv_bg    = (use_moisture && midx.qv >= 0) ? state(i,j,k,midx.qv) : 0.0;
        const Real rhoqc_bg    = (use_moisture && midx.qc >= 0) ? state(i,j,k,midx.qc) : 0.0;

        state_pert(i,j,k,Rho_comp)      = rho - rho_bg;
        state_pert(i,j,k,RhoTheta_comp) = rhotheta - rhotheta_bg;
        if (use_moisture && midx.qv >= 0) {
            state_pert(i,j,k,midx.qv) = rhoqv - rhoqv_bg;
            if (midx.qc >= 0) {
                state_pert(i,j,k,midx.qc) = rhoqc - rhoqc_bg;
            }
            if (midx.qr >= 0) {
                state_pert(i,j,k,midx.qr) = -state(i,j,k,midx.qr);
            }
        }

        state_pert(i,j,k,RhoScalar_comp) = -state(i,j,k,RhoScalar_comp);
        state_pert(i,j,k,RhoKE_comp)     = -state(i,j,k,RhoKE_comp);
    });

    ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        const int kk = amrex::max(0, amrex::min(k, dom_hi_z));
        const Real z = dycoms_cell_height(i, j, kk, z_nd, geomdata, has_terrain);
        x_vel_pert(i,j,k) = dycoms_u(parms_d, z);
    });

    ParallelFor(ybx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        const int kk = amrex::max(0, amrex::min(k, dom_hi_z));
        const Real z = dycoms_cell_height(i, j, kk, z_nd, geomdata, has_terrain);
        y_vel_pert(i,j,k) = dycoms_v(parms_d, z);
    });

    ParallelFor(zbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        z_vel_pert(i,j,k) = 0.0;
    });
}

void
Problem::update_w_subsidence (const Real& /*time*/,
                              Vector<Real>& wbar,
                              Gpu::DeviceVector<Real>& d_wbar,
                              const Geometry& geom,
                              std::unique_ptr<MultiFab>& z_phys_nd)
{
    if (wbar.empty()) return;

    const int khi = geom.Domain().bigEnd()[2] + 1;
    const Real* prob_lo = geom.ProbLo();
    const Real dz = geom.CellSize(2);

    if (z_phys_nd) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_nd);
    }

    wbar[0] = 0.0;
    for (int k = 1; k <= khi; ++k) {
        const Real z = (z_phys_nd) ? zlevels[k] : prob_lo[2] + k * dz;
        wbar[k] = -parms.divergence * z;
    }

    Gpu::copy(Gpu::hostToDevice, wbar.begin(), wbar.end(), d_wbar.begin());
}

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
    const Real dz = geom.CellSize(2);
    
    if (z_phys_cc) {
        zlevels.resize(khi+1);
        reduce_to_max_per_height(zlevels, z_phys_cc);
    }

    // RF02 spec: geostrophic wind follows initial wind profiles u(z), v(z)
    // u(z) = 3 + 4.3*z/1000  [m/s]
    // v(z) = -9 + 5.6*z/1000  [m/s]
    for (int k = 0; k <= khi; ++k) {
        const Real z = (z_phys_cc) ? zlevels[k] : prob_lo[2] + (k + 0.5) * dz;
        u_geos[k] = dycoms_u(parms, z);
        v_geos[k] = dycoms_v(parms, z);
    }

    Gpu::copy(Gpu::hostToDevice, u_geos.begin(), u_geos.end(), d_u_geos.begin());
    Gpu::copy(Gpu::hostToDevice, v_geos.begin(), v_geos.end(), d_v_geos.begin());
}
