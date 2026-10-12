#include <cmath>
#include <memory>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include "ERF_ThinBody.H"

using namespace amrex;

/*
 * Correct an existing ground-wall distance with the exact distance to finite
 * Cartesian face rectangles. The caller restricts this setup-only correction
 * to terrain-free ConstantDz geometry.
 */
void
thinbody_wall_dist (std::unique_ptr<MultiFab>& wdist,
                    Vector<IntVect>& xfaces,
                    Vector<IntVect>& yfaces,
                    Vector<IntVect>& zfaces,
                    const Geometry& geomdata,
                    bool cartesian_constant_dz)
{
    BL_PROFILE("thinbody_wall_dist()");

    if (!cartesian_constant_dz) {
        Abort("Thin-body RANS wall-distance correction requires Cartesian ConstantDz geometry");
    }
    if (!wdist) {
        Abort("Thin-body wall-distance correction received an unallocated distance field");
    }
    if (!wdist->is_finite(0, 1, 0)) {
        Abort("Thin-body wall-distance correction requires finite ground-wall distances");
    }
    if (wdist->min(0) < Real(0.0)) {
        Abort("Thin-body wall-distance correction requires nonnegative ground-wall distances");
    }

    const auto prob_lo = geomdata.ProbLoArray();
    const auto dx = geomdata.CellSizeArray();
    const auto periodic = geomdata.isPeriodicArray();
    const GpuArray<Real, AMREX_SPACEDIM> period_length{
        AMREX_D_DECL(geomdata.ProbLength(0), geomdata.ProbLength(1), geomdata.ProbLength(2))};
    const IntVect domain_lo = geomdata.Domain().smallEnd();

    Gpu::DeviceVector<IntVect> xfaces_d(xfaces.size());
    Gpu::DeviceVector<IntVect> yfaces_d(yfaces.size());
    Gpu::DeviceVector<IntVect> zfaces_d(zfaces.size());
    Gpu::copyAsync(Gpu::hostToDevice, xfaces.begin(), xfaces.end(), xfaces_d.begin());
    Gpu::copyAsync(Gpu::hostToDevice, yfaces.begin(), yfaces.end(), yfaces_d.begin());
    Gpu::copyAsync(Gpu::hostToDevice, zfaces.begin(), zfaces.end(), zfaces_d.begin());
    const auto* xfaces_d_ptr = xfaces_d.data();
    const auto* yfaces_d_ptr = yfaces_d.data();
    const auto* zfaces_d_ptr = zfaces_d.data();
    const auto n_xfaces = xfaces_d.size();
    const auto n_yfaces = yfaces_d.size();
    const auto n_zfaces = zfaces_d.size();

    for (MFIter mfi(*wdist); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto wd_arr = wdist->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const Real x = prob_lo[0] + (i - domain_lo[0] + Real(0.5)) * dx[0];
            const Real y = prob_lo[1] + (j - domain_lo[1] + Real(0.5)) * dx[1];
            const Real z = prob_lo[2] + (k - domain_lo[2] + Real(0.5)) * dx[2];
            Real distance_squared = wd_arr(i,j,k) * wd_arr(i,j,k);

            for (std::size_t iface = 0; iface < n_xfaces; ++iface) {
                const int fi = xfaces_d_ptr[iface][0];
                const int fj = xfaces_d_ptr[iface][1];
                const int fk = xfaces_d_ptr[iface][2];
                const Real xf = prob_lo[0] + (fi - domain_lo[0]) * dx[0];
                const Real y0 = prob_lo[1] + (fj - domain_lo[1]) * dx[1];
                const Real y1 = y0 + dx[1];
                const Real z0 = prob_lo[2] + (fk - domain_lo[2]) * dx[2];
                const Real z1 = z0 + dx[2];
                const Real gap_x = ERFThinBody::periodic_gap(
                    x, xf, xf, periodic[0] != 0, period_length[0]);
                const Real gap_y = ERFThinBody::periodic_gap(
                    y, y0, y1, periodic[1] != 0, period_length[1]);
                const Real gap_z = ERFThinBody::periodic_gap(
                    z, z0, z1, periodic[2] != 0, period_length[2]);
                distance_squared = min(distance_squared,
                    gap_x*gap_x + gap_y*gap_y + gap_z*gap_z);
            }

            for (std::size_t iface = 0; iface < n_yfaces; ++iface) {
                const int fi = yfaces_d_ptr[iface][0];
                const int fj = yfaces_d_ptr[iface][1];
                const int fk = yfaces_d_ptr[iface][2];
                const Real yf = prob_lo[1] + (fj - domain_lo[1]) * dx[1];
                const Real x0 = prob_lo[0] + (fi - domain_lo[0]) * dx[0];
                const Real x1 = x0 + dx[0];
                const Real z0 = prob_lo[2] + (fk - domain_lo[2]) * dx[2];
                const Real z1 = z0 + dx[2];
                const Real gap_x = ERFThinBody::periodic_gap(
                    x, x0, x1, periodic[0] != 0, period_length[0]);
                const Real gap_y = ERFThinBody::periodic_gap(
                    y, yf, yf, periodic[1] != 0, period_length[1]);
                const Real gap_z = ERFThinBody::periodic_gap(
                    z, z0, z1, periodic[2] != 0, period_length[2]);
                distance_squared = min(distance_squared,
                    gap_x*gap_x + gap_y*gap_y + gap_z*gap_z);
            }

            for (std::size_t iface = 0; iface < n_zfaces; ++iface) {
                const int fi = zfaces_d_ptr[iface][0];
                const int fj = zfaces_d_ptr[iface][1];
                const int fk = zfaces_d_ptr[iface][2];
                const Real zf = prob_lo[2] + (fk - domain_lo[2]) * dx[2];
                const Real x0 = prob_lo[0] + (fi - domain_lo[0]) * dx[0];
                const Real x1 = x0 + dx[0];
                const Real y0 = prob_lo[1] + (fj - domain_lo[1]) * dx[1];
                const Real y1 = y0 + dx[1];
                const Real gap_x = ERFThinBody::periodic_gap(
                    x, x0, x1, periodic[0] != 0, period_length[0]);
                const Real gap_y = ERFThinBody::periodic_gap(
                    y, y0, y1, periodic[1] != 0, period_length[1]);
                const Real gap_z = ERFThinBody::periodic_gap(
                    z, zf, zf, periodic[2] != 0, period_length[2]);
                distance_squared = min(distance_squared,
                    gap_x*gap_x + gap_y*gap_y + gap_z*gap_z);
            }

            wd_arr(i,j,k) = std::sqrt(distance_squared);
        });
    }

    // The face lists back asynchronous kernels and must remain alive until all
    // setup launches have consumed them.
    Gpu::streamSynchronize();
}
