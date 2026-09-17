#ifndef ERF_SBM_ERF_BOUNDARY_H_
#define ERF_SBM_ERF_BOUNDARY_H_

#include "ERF_IndexDefines.H"
#include "ERF_SBMBoundary.H"

#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Orientation.H"

namespace erf_sbm {

/** Map ERF's physical boundary vocabulary to the provider-neutral SBM policy. */
inline BoundaryKind classify_erf_boundary(const ERF_BC boundary) noexcept
{
    if (boundary == ERF_BC::symmetry ||
        boundary == ERF_BC::no_slip_wall ||
        boundary == ERF_BC::slip_wall) {
        return BoundaryKind::ImpermeableWall;
    }
    if (boundary == ERF_BC::outflow ||
        boundary == ERF_BC::ho_outflow ||
        boundary == ERF_BC::open) {
        return BoundaryKind::AdvectiveOutflow;
    }
    if (boundary == ERF_BC::periodic) return BoundaryKind::Periodic;
    return BoundaryKind::PrescribedSpectralInflow;
}

/**
 * Build the one ERF-to-SBM boundary policy used by transport and host CFL.
 * Periodicity is authoritative when a direction is periodic; otherwise the
 * corresponding ERF physical face is classified explicitly.
 */
inline TransportBoundaryPolicy make_erf_transport_boundary_policy(
    const amrex::Geometry& geometry,
    const amrex::GpuArray<ERF_BC, AMREX_SPACEDIM*2>& phys_bc_type) noexcept
{
    TransportBoundaryPolicy policy;
    policy.configured = true;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (geometry.isPeriodic(dir)) {
            policy.face_kind[2*dir] = BoundaryKind::Periodic;
            policy.face_kind[2*dir+1] = BoundaryKind::Periodic;
        } else {
            policy.face_kind[2*dir] = classify_erf_boundary(
                phys_bc_type[amrex::Orientation(dir, amrex::Orientation::low)]);
            policy.face_kind[2*dir+1] = classify_erf_boundary(
                phys_bc_type[amrex::Orientation(dir, amrex::Orientation::high)]);
        }
    }
    return policy;
}

} // namespace erf_sbm

#endif
