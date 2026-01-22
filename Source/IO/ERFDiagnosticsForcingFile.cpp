#include "ERFDiagnosticsForcingFile.H"
#include "ERF_Constants.H"
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>

ERFDiagnosticsForcingFile::ERFDiagnosticsForcingFile(std::shared_ptr<ERForcingReader> reader)
    : m_reader(reader)
{
}

void ERFDiagnosticsForcingFile::write_diagnostics(const amrex::Real& time,
                                                    const amrex::Geometry& geom,
                                                    const amrex::MultiFab& z_phys_cc,
                                                    const amrex::MultiFab& theta,
                                                    const amrex::MultiFab& qv,
                                                    const amrex::MultiFab& u,
                                                    const amrex::MultiFab& v)
{
    if (!m_reader) return;

    // Only compute every so often (e.g. at plot_int or fixed interval)
    // For now, compute if called (caller controls frequency)

    int idx;
    amrex::Real weight;
    m_reader->get_time_interp(time, idx, weight);
    const auto& data = m_reader->get_data();

    // 1. Comparison to Input Data (RMSE of profiles)
    // We want to compare horizontal average of ERF variables vs Forcing file variables (which are 1D profiles)
    // Forcing file variables: T(t, z), qv(t, z), u(t, z), v(t, z)
    // We need to interpolate Forcing vars to ERF grid height, OR compare at nearest levels.
    // Simpler: Interpolate Forcing vars to z_phys_cc at domain center or average z_phys.
    
    // Compute horizontal averages of ERF state
    // Note: This is expensive if done every step. We should assume caller controls frequency.
    
    // We already have code in ERF::MakeHorizontalAverages or similar. 
    // Here we will do a simple comparison at a few levels or full profile RMSE if possible.

    // Let's compute global min/max/mean of theta
    amrex::Real vol_inv = 1.0 / geom.Domain().numPts();
    amrex::Real theta_mean = theta.sum(0) * vol_inv;
    amrex::Real qv_mean = qv.sum(0) * vol_inv;
    amrex::Real u_mean = u.sum(0) * vol_inv;
    amrex::Real v_mean = v.sum(0) * vol_inv;

    // Get interpolated forcing values at a reference height (e.g. middle of domain) to check consistency
    // Ideally we would loop over vertical levels, compute horizontal avg, and compare with forcing profile.

    amrex::Print() << "\n[Diagnostics] Time: " << time << " s\n";
    amrex::Print() << "  ERF Mean State: Theta=" << theta_mean << " K, Qv=" << qv_mean << " kg/kg, U=" << u_mean << ", V=" << v_mean << "\n";
    
    // TODO: Full profile comparison could be implemented here using parallel reduce over levels.
}

void ERFDiagnosticsForcingFile::initial_diagnostics(const amrex::Real& time,
                                                    const amrex::Geometry& geom,
                                                    const amrex::MultiFab& z_phys_cc,
                                                    const amrex::MultiFab& u,
                                                    const amrex::MultiFab& v,
                                                    const amrex::MultiFab& theta,
                                                    const amrex::MultiFab& qv)
{
    if (!m_reader) return;

    amrex::Print() << "\n[Initial Diagnostics] vs Forcing File at t=" << time << "\n";
    
    amrex::Real vol_inv = 1.0 / geom.Domain().numPts();
    amrex::Real theta_mean = theta.sum(0) * vol_inv;
    amrex::Real qv_mean = qv.sum(0) * vol_inv;
    
    amrex::Print() << "  ERF Initial Mean: Theta=" << theta_mean << ", Qv=" << qv_mean << "\n";
}
