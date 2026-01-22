#ifndef ERF_FORCING_READER_H
#define ERF_FORCING_READER_H

#include <string>
#include <vector>
#include <AMReX_Vector.H>
#include <AMReX_RealVect.H>
#include "ERF_NCInterface.H"

struct ForcingData {
    // 1D arrays (time)
    amrex::Vector<amrex::Real> time; // seconds since reference
    amrex::Vector<amrex::Real> sfc_sens_flux;
    amrex::Vector<amrex::Real> sfc_lat_flux;
    amrex::Vector<amrex::Real> sfc_pres;
    amrex::Vector<amrex::Real> sfc_temp;

    // 1D arrays (height)
    amrex::Vector<amrex::Real> height;

    // 2D arrays (time, height) - flattened or vector of vectors?
    // Using flat vector with access helper is usually better for GPU,
    // but here we read on CPU and interpolate.
    // Let's use Vector<Vector<Real>> for simplicity in reading/storage on CPU,
    // as this data is not heavy (O(MB)).
    // Outer vector is time, inner is height.
    amrex::Vector<amrex::Vector<amrex::Real>> T;
    amrex::Vector<amrex::Vector<amrex::Real>> p;
    amrex::Vector<amrex::Vector<amrex::Real>> qv;
    amrex::Vector<amrex::Vector<amrex::Real>> u;
    amrex::Vector<amrex::Vector<amrex::Real>> v;
    amrex::Vector<amrex::Vector<amrex::Real>> u_g;
    amrex::Vector<amrex::Vector<amrex::Real>> v_g;
    amrex::Vector<amrex::Vector<amrex::Real>> w_sub;
    amrex::Vector<amrex::Vector<amrex::Real>> theta_adv_h;
    amrex::Vector<amrex::Vector<amrex::Real>> theta_adv_v;
    amrex::Vector<amrex::Vector<amrex::Real>> qv_adv_h;
    amrex::Vector<amrex::Vector<amrex::Real>> qv_adv_v;
};

class ERForcingReader {
public:
    ERForcingReader(const std::string& filename);

    // Read the file and populate data
    void read_file();

    // Interpolate scalar at a given time
    amrex::Real interpolate_scalar(const amrex::Vector<amrex::Real>& var, amrex::Real time) const;

    // Interpolate profile at a given time and height
    amrex::Real interpolate_profile(const amrex::Vector<amrex::Vector<amrex::Real>>& var,
                                    amrex::Real time, amrex::Real height) const;

    // Get raw data
    const ForcingData& get_data() const { return m_data; }

    // Get time index and weight for interpolation
    void get_time_interp(amrex::Real time, int& idx, amrex::Real& weight) const;

private:
    std::string m_filename;
    ForcingData m_data;
    int m_ntime;
    int m_nheight;
};

#endif // ERF_FORCING_READER_H
