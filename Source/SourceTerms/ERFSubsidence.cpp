#include "ERFSubsidence.H"
#include "ERFVerticalInterpolation.H"
#include "ERF_Constants.H"

ERFSubsidence::ERFSubsidence(std::shared_ptr<ERForcingReader> reader)
    : m_reader(reader)
{
}

void ERFSubsidence::compute_subsidence_tendency(const amrex::Real& time,
                                                const amrex::Geometry& geom,
                                                const amrex::MultiFab& z_phys_cc,
                                                amrex::MultiFab& theta_rhs)
{
    if (!m_reader) return;

    // Get forcing data
    const auto& data = m_reader->get_data();
    const auto& z_forcing = data.height;
    int n_forcing = z_forcing.size();

    // Extract profile at current time
    amrex::Vector<amrex::Real> w_sub_prof(n_forcing);
    int idx;
    amrex::Real weight;
    m_reader->get_time_interp(time, idx, weight);
    for (int k = 0; k < n_forcing; ++k) {
        w_sub_prof[k] = data.w_sub[idx][k] * (1.0 - weight) + data.w_sub[idx+1][k] * weight;
    }
    
    // Copy to GPU
    amrex::Gpu::DeviceVector<amrex::Real> d_z_forcing(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_w_sub(n_forcing);
    
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, z_forcing.begin(), z_forcing.end(), d_z_forcing.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, w_sub_prof.begin(), w_sub_prof.end(), d_w_sub.begin());
    
    amrex::Real* p_z = d_z_forcing.data();
    amrex::Real* p_w_sub = d_w_sub.data();
    
    amrex::Real g = CONST_GRAV;
    amrex::Real Cp = Cp_d;
    
    for (amrex::MFIter mfi(theta_rhs); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& z_arr = z_phys_cc.array(mfi);
        const auto& th_rhs_arr = theta_rhs.array(mfi);
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::Real z = z_arr(i,j,k);
            
            // Interpolate w_sub
            amrex::Real w_sub = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_w_sub, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            
            // Adiabatic heating: dtheta/dt = (g/Cp) * w_sub
            // w_sub is positive downward. Descent causes warming.
            amrex::Real dtheta_dt = (g / Cp) * w_sub;
            
            th_rhs_arr(i,j,k) += dtheta_dt;
        });
    }
}
