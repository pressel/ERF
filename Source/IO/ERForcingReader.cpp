#include "ERForcingReader.H"
#include "ERFVerticalInterpolation.H"
#include <AMReX_Print.H>
#include <AMReX.H>
#include <iostream>

ERForcingReader::ERForcingReader(const std::string& filename)
    : m_filename(filename), m_ntime(0), m_nheight(0)
{
    read_file();
}

void ERForcingReader::read_file()
{
    amrex::Print() << "Reading forcing file: " << m_filename << std::endl;

#ifdef ERF_USE_NETCDF
    // Diagnostic test: call nc_open directly
    int ncid;
    amrex::Print() << "Calling nc_open..." << std::endl;
    int status = nc_open(m_filename.c_str(), NC_NOWRITE, &ncid);
    if (status != NC_NOERR) {
        amrex::Print() << "nc_open failed: " << nc_strerror(status) << std::endl;
        amrex::Abort("ERForcingReader: nc_open failed");
    }
    amrex::Print() << "nc_open success, ncid=" << ncid << std::endl;

    // Now use ncutils wrapper (which wraps the ncid we just opened? No, we need to close it first or use it)
    // Since ncutils::NCFile owns the ncid, we can't easily transfer ownership unless we use the constructor taking ncid.
    // But NCFile constructor is protected.
    // So we close it and let NCFile open it again (hopefully it works this time if it was a fluke, but likely it will fail again if we don't fix the root cause).
    // Actually, let's just use the raw ncid for now to verify we can read.
    
    // But to keep code clean, let's close and try NCFile::open (serial) again, but this time we know if nc_open works.
    nc_close(ncid);

    auto nc_file = ncutils::NCFile::open(m_filename, NC_NOWRITE);

    // Read dimensions
    const auto& dims = nc_file.all_dims();
    for (const auto& dim : dims) {
        if (dim.name() == "time") m_ntime = dim.len();
        if (dim.name() == "height") m_nheight = dim.len();
    }

    if (m_ntime == 0 || m_nheight == 0) {
        amrex::Abort("ERForcingReader: Invalid dimensions (time or height is 0)");
    }

    // Helper macro to read 1D variable
    #define READ_1D(NAME, VEC) \
    if (nc_file.has_var(NAME)) { \
        VEC.resize(m_ntime); \
        std::vector<double> tmp(m_ntime); \
        nc_file.var(NAME).get(tmp.data()); \
        for (int i = 0; i < m_ntime; ++i) VEC[i] = static_cast<amrex::Real>(tmp[i]); \
    } else { \
        amrex::Print() << "Warning: Variable " << NAME << " not found in forcing file." << std::endl; \
    }

    READ_1D("time", m_data.time);
    READ_1D("sfc_sens_flux", m_data.sfc_sens_flux);
    READ_1D("sfc_lat_flux", m_data.sfc_lat_flux);
    READ_1D("sfc_pres", m_data.sfc_pres);
    READ_1D("sfc_temp", m_data.sfc_temp);

    #undef READ_1D

    // Read height (1D but size nheight)
    if (nc_file.has_var("height")) {
        m_data.height.resize(m_nheight);
        std::vector<double> tmp(m_nheight);
        nc_file.var("height").get(tmp.data());
        for (int k = 0; k < m_nheight; ++k) m_data.height[k] = static_cast<amrex::Real>(tmp[k]);
    } else {
        amrex::Abort("ERForcingReader: Variable 'height' not found");
    }

    // Helper macro to read 2D variable
    #define READ_2D(NAME, VEC) \
    if (nc_file.has_var(NAME)) { \
        VEC.resize(m_ntime); \
        std::vector<double> tmp(m_ntime * m_nheight); \
        nc_file.var(NAME).get(tmp.data()); \
        for (int i = 0; i < m_ntime; ++i) { \
            VEC[i].resize(m_nheight); \
            for (int k = 0; k < m_nheight; ++k) { \
                VEC[i][k] = static_cast<amrex::Real>(tmp[i * m_nheight + k]); \
            } \
        } \
    } else { \
        amrex::Print() << "Warning: Variable " << NAME << " not found in forcing file." << std::endl; \
    }

    READ_2D("T", m_data.T);
    READ_2D("p", m_data.p);
    READ_2D("qv", m_data.qv);
    READ_2D("u", m_data.u);
    READ_2D("v", m_data.v);
    READ_2D("u_g", m_data.u_g);
    READ_2D("v_g", m_data.v_g);
    READ_2D("w_sub", m_data.w_sub);
    READ_2D("theta_adv_h", m_data.theta_adv_h);
    READ_2D("theta_adv_v", m_data.theta_adv_v);
    READ_2D("qv_adv_h", m_data.qv_adv_h);
    READ_2D("qv_adv_v", m_data.qv_adv_v);

    #undef READ_2D

    nc_file.close();
#else
    amrex::Abort("ERForcingReader: ERF_USE_NETCDF not defined");
#endif
}

void ERForcingReader::get_time_interp(amrex::Real time, int& idx, amrex::Real& weight) const
{
    // Find index such that m_data.time[idx] <= time < m_data.time[idx+1]
    // Assumes time is monotonic
    int n = m_data.time.size();
    if (n < 2) {
        idx = 0;
        weight = 0.0;
        return;
    }

    if (time <= m_data.time[0]) {
        idx = 0;
        weight = 0.0;
        return;
    }

    if (time >= m_data.time[n-1]) {
        idx = n - 2;
        weight = 1.0;
        return;
    }

    // Linear search (could be binary search)
    idx = 0;
    while (idx < n - 2 && m_data.time[idx+1] < time) {
        idx++;
    }

    weight = (time - m_data.time[idx]) / (m_data.time[idx+1] - m_data.time[idx]);
}

amrex::Real ERForcingReader::interpolate_scalar(const amrex::Vector<amrex::Real>& var, amrex::Real time) const
{
    if (var.empty()) return 0.0;
    int idx;
    amrex::Real weight;
    get_time_interp(time, idx, weight);
    return var[idx] * (1.0 - weight) + var[idx+1] * weight;
}

amrex::Real ERForcingReader::interpolate_profile(const amrex::Vector<amrex::Vector<amrex::Real> >& var,
                                                 amrex::Real time, amrex::Real height) const
{
    if (var.empty()) return 0.0;
    int idx;
    amrex::Real weight;
    get_time_interp(time, idx, weight);

    // Interpolate in height for both time slices
    // Use ERFVerticalInterpolation helper
    amrex::Real val0 = ERFVerticalInterpolation::interpolate_1d_linear(height, m_data.height.data(), var[idx].data(), m_nheight, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Linear);
    amrex::Real val1 = ERFVerticalInterpolation::interpolate_1d_linear(height, m_data.height.data(), var[idx+1].data(), m_nheight, ERFVerticalInterpolation::ExtrapType::Constant, ERFVerticalInterpolation::ExtrapType::Linear);

    // Interpolate in time
    return val0 * (1.0 - weight) + val1 * weight;
}
