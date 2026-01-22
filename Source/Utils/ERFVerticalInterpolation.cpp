#include "ERFVerticalInterpolation.H"
#include <cmath>
#include <AMReX_Print.H>

    amrex::Real ERFVerticalInterpolation::interpolate_1d_linear(amrex::Real z_target,
                                      const amrex::Real* z_src,
                                      const amrex::Real* f_src,
                                      int n_src,
                                      ExtrapType extrap_lo,
                                      ExtrapType extrap_hi)
    {
        // Handle below surface
        if (z_target < z_src[0]) {
            if (extrap_lo == ExtrapType::Constant) {
                return f_src[0];
            } else if (extrap_lo == ExtrapType::Linear) {
                amrex::Real slope = (f_src[1] - f_src[0]) / (z_src[1] - z_src[0]);
                return f_src[0] + slope * (z_target - z_src[0]);
            } else if (extrap_lo == ExtrapType::LogLinear) {
                // Simple log profile assumption: u = u_ref * ln(z/z0) / ln(z_ref/z0)
                // Here we just extrapolate linearly for now as log requires z0 info
                // which we don't have here. Fallback to linear.
                amrex::Real slope = (f_src[1] - f_src[0]) / (z_src[1] - z_src[0]);
                return f_src[0] + slope * (z_target - z_src[0]);
            }
        }

        // Handle above top
        if (z_target > z_src[n_src-1]) {
            if (extrap_hi == ExtrapType::Constant) {
                return f_src[n_src-1];
            } else if (extrap_hi == ExtrapType::Linear) {
                amrex::Real slope = (f_src[n_src-1] - f_src[n_src-2]) / (z_src[n_src-1] - z_src[n_src-2]);
                return f_src[n_src-1] + slope * (z_target - z_src[n_src-1]);
            } else if (extrap_hi == ExtrapType::LogLinear) {
                // Power law extrapolation: u = u_ref * (z/z_ref)^alpha
                // Estimate alpha from last two points
                amrex::Real z2 = z_src[n_src-1];
                amrex::Real z1 = z_src[n_src-2];
                amrex::Real u2 = f_src[n_src-1];
                amrex::Real u1 = f_src[n_src-2];
                if (u1 > 1e-6 && u2 > 1e-6 && z1 > 1e-6 && z2 > 1e-6) {
                    amrex::Real alpha = std::log(u2/u1) / std::log(z2/z1);
                    return u2 * std::pow(z_target/z2, alpha);
                } else {
                    // Fallback to linear
                    amrex::Real slope = (u2 - u1) / (z2 - z1);
                    return u2 + slope * (z_target - z2);
                }
            }
        }

        // Find bracketing indices
        int j = 0;
        // Simple linear search since n_src is small (~50)
        while (j < n_src - 1 && z_src[j+1] < z_target) {
            j++;
        }

        // Linear interpolation
        amrex::Real w = (z_target - z_src[j]) / (z_src[j+1] - z_src[j]);
        return f_src[j] * (1.0 - w) + f_src[j+1] * w;
    }

    void ERFVerticalInterpolation::interpolate_vertical(const amrex::Vector<amrex::Real>& z_src,
                              const amrex::Vector<amrex::Real>& f_src,
                              const amrex::Vector<amrex::Real>& z_tgt,
                              amrex::Vector<amrex::Real>& f_tgt,
                              InterpType interp_type,
                              ExtrapType extrap_lo,
                              ExtrapType extrap_hi)
    {
        int n_src = z_src.size();
        int n_tgt = z_tgt.size();
        f_tgt.resize(n_tgt);

        if (n_src < 2) {
            amrex::Abort("interpolate_vertical: source grid must have at least 2 points");
        }

        if (interp_type == InterpType::Linear) {
            for (int i = 0; i < n_tgt; ++i) {
                f_tgt[i] = interpolate_1d_linear(z_tgt[i], z_src.data(), f_src.data(), n_src, extrap_lo, extrap_hi);
            }
        } else {
            amrex::Abort("interpolate_vertical: CubicSpline not implemented yet");
        }
    }
