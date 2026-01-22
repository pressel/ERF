#ifndef ERF_VERTICAL_INTERPOLATION_H
#define ERF_VERTICAL_INTERPOLATION_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Gpu.H>

struct ERFVerticalInterpolation {

    enum class InterpType {
        Linear,
        CubicSpline
    };

    enum class ExtrapType {
        Constant,
        Linear,
        LogLinear // For wind profiles
    };

    /**
     * Interpolate a 1D profile from source height grid to target height grid.
     *
     * @param[in] z_src Source height grid (must be monotonic increasing)
     * @param[in] f_src Source variable values
     * @param[in] z_tgt Target height grid
     * @param[out] f_tgt Target variable values (resized if needed)
     * @param[in] interp_type Interpolation method (Linear or CubicSpline)
     * @param[in] extrap_lo Extrapolation method below z_src[0]
     * @param[in] extrap_hi Extrapolation method above z_src[end]
     */
    static void interpolate_vertical(const amrex::Vector<amrex::Real>& z_src,
                              const amrex::Vector<amrex::Real>& f_src,
                              const amrex::Vector<amrex::Real>& z_tgt,
                              amrex::Vector<amrex::Real>& f_tgt,
                              InterpType interp_type = InterpType::Linear,
                              ExtrapType extrap_lo = ExtrapType::Constant,
                              ExtrapType extrap_hi = ExtrapType::Linear);

    /**
     * Single point interpolation helper
     */
    AMREX_GPU_HOST_DEVICE
    static amrex::Real interpolate_1d_linear(amrex::Real z_target,
                                      const amrex::Real* z_src,
                                      const amrex::Real* f_src,
                                      int n_src,
                                      ExtrapType extrap_lo,
                                      ExtrapType extrap_hi);

};

#endif // ERF_VERTICAL_INTERPOLATION_H
