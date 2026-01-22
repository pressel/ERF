#include "ERFAdvectionTendencies.H"
#include "ERFVerticalInterpolation.H"
#include "ERF_Constants.H"

ERFAdvectionTendencies::ERFAdvectionTendencies(std::shared_ptr<ERForcingReader> reader)
    : m_reader(reader)
{
}

void ERFAdvectionTendencies::compute_tendencies(const amrex::Real& time,
                                                const amrex::Geometry& geom,
                                                const amrex::MultiFab& z_phys_cc,
                                                amrex::MultiFab& theta_rhs,
                                                amrex::MultiFab& qv_rhs)
{
    if (!m_reader) return;

    // Get forcing data
    const auto& data = m_reader->get_data();
    const auto& z_forcing = data.height;
    int n_forcing = z_forcing.size();

    // Interpolate profiles at current time
    // We need to interpolate to each cell center height
    // Since z_phys_cc varies in x,y (terrain), we do this point-wise or column-wise
    
    // Optimization: If z_phys is flat, we can compute profile once.
    // But for generality, let's do it per column or per point.
    // Since we have interpolate_1d_linear helper, we can use it in GPU kernel.
    
    // However, interpolate_1d_linear takes pointers to arrays.
    // We need to pass the forcing profile arrays to the GPU.
    // The ForcingData struct has Vector<Real> which are on CPU.
    // We should probably copy the current time profile to a Gpu::DeviceVector or similar.
    // Or, since n_forcing is small (~50), we can maybe put it in constant memory or just pass as argument?
    // AMReX Gpu::DeviceVector is good.
    
    // Let's extract the profiles at current time on CPU first
    amrex::Vector<amrex::Real> theta_adv_h_prof(n_forcing);
    amrex::Vector<amrex::Real> theta_adv_v_prof(n_forcing);
    amrex::Vector<amrex::Real> qv_adv_h_prof(n_forcing);
    amrex::Vector<amrex::Real> qv_adv_v_prof(n_forcing);
    
    int idx;
    amrex::Real weight;
    m_reader->get_time_interp(time, idx, weight);

    for (int k = 0; k < n_forcing; ++k) {
        theta_adv_h_prof[k] = data.theta_adv_h[idx][k] * (1.0 - weight) + data.theta_adv_h[idx+1][k] * weight;
        theta_adv_v_prof[k] = data.theta_adv_v[idx][k] * (1.0 - weight) + data.theta_adv_v[idx+1][k] * weight;
        qv_adv_h_prof[k] = data.qv_adv_h[idx][k] * (1.0 - weight) + data.qv_adv_h[idx+1][k] * weight;
        qv_adv_v_prof[k] = data.qv_adv_v[idx][k] * (1.0 - weight) + data.qv_adv_v[idx+1][k] * weight;
    }
    
    // Copy to GPU
    amrex::Gpu::DeviceVector<amrex::Real> d_z_forcing(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_theta_adv_h(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_theta_adv_v(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_qv_adv_h(n_forcing);
    amrex::Gpu::DeviceVector<amrex::Real> d_qv_adv_v(n_forcing);
    
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, z_forcing.begin(), z_forcing.end(), d_z_forcing.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, theta_adv_h_prof.begin(), theta_adv_h_prof.end(), d_theta_adv_h.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, theta_adv_v_prof.begin(), theta_adv_v_prof.end(), d_theta_adv_v.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, qv_adv_h_prof.begin(), qv_adv_h_prof.end(), d_qv_adv_h.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, qv_adv_v_prof.begin(), qv_adv_v_prof.end(), d_qv_adv_v.begin());
    
    amrex::Real* p_z = d_z_forcing.data();
    amrex::Real* p_th_h = d_theta_adv_h.data();
    amrex::Real* p_th_v = d_theta_adv_v.data();
    amrex::Real* p_qv_h = d_qv_adv_h.data();
    amrex::Real* p_qv_v = d_qv_adv_v.data();
    
    for (amrex::MFIter mfi(theta_rhs); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& z_arr = z_phys_cc.array(mfi);
        const auto& th_rhs_arr = theta_rhs.array(mfi);
        const auto& qv_rhs_arr = qv_rhs.array(mfi);
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::Real z = z_arr(i,j,k);
            
            // Interpolate tendencies
            amrex::Real th_h = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_th_h, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            amrex::Real th_v = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_th_v, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            amrex::Real qv_h = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_qv_h, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            amrex::Real qv_v = ERFVerticalInterpolation::interpolate_1d_linear(z, p_z, p_qv_v, n_forcing, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Constant);
            
            th_rhs_arr(i,j,k) += th_h + th_v;
            qv_rhs_arr(i,j,k) += qv_h + qv_v;
        });
    }
}
