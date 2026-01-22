#include "ERF_InitLESForcing.H"
#include "ERFVerticalInterpolation.H"
#include "ERF_IndexDefines.H"
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Gpu.H>
#include <cmath>

ERFInitLESForcing::ERFInitLESForcing()
{
}

void ERFInitLESForcing::define(const std::string& filename)
{
    amrex::Print() << "Initializing LES forcing from file: " << filename << std::endl;
    m_reader = std::make_shared<ERForcingReader>(filename);
}

void ERFInitLESForcing::initialize_domain(const amrex::Geometry& geom,
                                          const amrex::MultiFab& z_phys_cc,
                                          amrex::MultiFab& u_cc,
                                          amrex::MultiFab& v_cc,
                                          amrex::MultiFab& w_cc,
                                          amrex::MultiFab& cons_cc,
                                          amrex::Real start_time)
{
    if (!m_reader) {
        amrex::Abort("ERFInitLESForcing::initialize_domain: reader not initialized. Call define() first.");
    }

    // Initialization must use the forcing-relative clock (time=0 at the first slice).
    // Do NOT offset by start_time because when start_datetime is set, start_time holds
    // an absolute epoch value. Runtime calls use cur_time (seconds since model start),
    // so we mirror that here.
    (void) start_time;
    const auto& data = m_reader->get_data();
    amrex::Real init_time = data.time.empty() ? 0.0 : data.time.front();

    // Extract profiles at init_time
    // We need to interpolate profiles to ERF grid z_phys_cc
    // Since z_phys_cc can vary in x,y, we should do this on GPU per cell.
    
    // Get forcing height grid and profiles at init_time
    const auto& z_forcing = data.height;
    int n_forcing = z_forcing.size();
    
    amrex::Vector<amrex::Real> u_prof(n_forcing);
    amrex::Vector<amrex::Real> v_prof(n_forcing);
    amrex::Vector<amrex::Real> T_prof(n_forcing);
    amrex::Vector<amrex::Real> p_prof(n_forcing);
    amrex::Vector<amrex::Real> qv_prof(n_forcing);
    
    // Interpolate in time first to get the profile at init_time on forcing grid
    int idx;
    amrex::Real weight;
    m_reader->get_time_interp(init_time, idx, weight);
    
    for (int k = 0; k < n_forcing; ++k) {
        u_prof[k] = data.u[idx][k] * (1.0 - weight) + data.u[idx+1][k] * weight;
        v_prof[k] = data.v[idx][k] * (1.0 - weight) + data.v[idx+1][k] * weight;
        T_prof[k] = data.T[idx][k] * (1.0 - weight) + data.T[idx+1][k] * weight;
        p_prof[k] = data.p[idx][k] * (1.0 - weight) + data.p[idx+1][k] * weight;
        qv_prof[k] = data.qv[idx][k] * (1.0 - weight) + data.qv[idx+1][k] * weight;
    }
    
    // Copy profiles to GPU
    amrex::Gpu::DeviceVector<amrex::Real> d_z_forcing(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_u_prof(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_v_prof(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_T_prof(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_p_prof(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_qv_prof(n_forcing);
    
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, z_forcing.begin(), z_forcing.end(), d_z_forcing.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, u_prof.begin(), u_prof.end(), d_u_prof.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, v_prof.begin(), v_prof.end(), d_v_prof.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, T_prof.begin(), T_prof.end(), d_T_prof.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, p_prof.begin(), p_prof.end(), d_p_prof.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, qv_prof.begin(), qv_prof.end(), d_qv_prof.begin());
    
    amrex::Real* p_z = d_z_forcing.data();
    amrex::Real* p_u = d_u_prof.data();
    amrex::Real* p_v = d_v_prof.data();
    amrex::Real* p_T = d_T_prof.data();
    amrex::Real* p_p = d_p_prof.data();
    amrex::Real* p_qv = d_qv_prof.data();
    
    amrex::Real p0 = 100000.0;
    amrex::Real Rd = 287.0;
    amrex::Real Cp = 1004.0;

    // Initialize MultiFabs
    for (amrex::MFIter mfi(u_cc); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& z_arr = z_phys_cc.array(mfi);
        const auto& u_arr = u_cc.array(mfi);
        const auto& v_arr = v_cc.array(mfi);
        const auto& w_arr = w_cc.array(mfi);
        const auto& cons_arr = cons_cc.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::Real z = z_arr(i,j,k);
            
            // Interpolate to z
            amrex::Real u_val = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_u, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            amrex::Real v_val = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_v, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            amrex::Real T_val = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_T, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            amrex::Real p_val = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_p, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            amrex::Real qv_val = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_qv, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            
            // Calculate potential temperature
            amrex::Real theta_val = T_val * std::pow(p0 / p_val, Rd / Cp);
            
            // Calculate density
            amrex::Real rho_val = p_val / (Rd * T_val * (1.0 + 0.61 * qv_val));

            u_arr(i,j,k) = u_val;
            v_arr(i,j,k) = v_val;
            w_arr(i,j,k) = 0.0;
            
            cons_arr(i,j,k,Rho_comp) = rho_val;
            cons_arr(i,j,k,RhoTheta_comp) = rho_val * theta_val;
            cons_arr(i,j,k,RhoQ1_comp) = rho_val * qv_val;
            // Initialize other scalars to 0
            for (int n = RhoQ2_comp; n < cons_arr.nComp(); ++n) {
                cons_arr(i,j,k,n) = 0.0;
            }
        });
    }
}
