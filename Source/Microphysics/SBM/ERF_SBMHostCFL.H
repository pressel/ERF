#ifndef ERF_SBM_HOST_CFL_H_
#define ERF_SBM_HOST_CFL_H_

#include "ERF_IndexDefines.H"
#include "ERF_SBMBoundary.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace erf_sbm {

/** Fixed-size result for one active host-CFL face. */
struct HostFaceDemand {
    amrex::Real advective{0.0};
    amrex::Real diffusive{0.0};
    int valid{1};
};

/**
 * Evaluate one cell-face contribution using ERF's mass-carrier convention.
 * The neighbor is required only for an active face; callers must skip a
 * suppressed physical wall face before invoking this function.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HostFaceDemand host_face_demand(const amrex::Real rho_cell,
                                const amrex::Real rho_neighbor,
                                const amrex::Real velocity,
                                const int boundary_kind,
                                const bool low_face,
                                const amrex::Real inverse_spacing,
                                const amrex::Real diffusion_coefficient) noexcept
{
    HostFaceDemand result;
    const amrex::Real rho_face = amrex::Real(0.5) * (rho_cell + rho_neighbor);
    if (rho_cell <= amrex::Real(0.0) || rho_neighbor <= amrex::Real(0.0) ||
        rho_face <= amrex::Real(0.0) ||
        amrex::isnan(rho_cell) || amrex::isinf(rho_cell) ||
        amrex::isnan(rho_neighbor) || amrex::isinf(rho_neighbor) ||
        amrex::isnan(rho_face) || amrex::isinf(rho_face) ||
        amrex::isnan(velocity) || amrex::isinf(velocity)) {
        result.valid = 0;
        return result;
    }

    const int outflow = static_cast<int>(BoundaryKind::AdvectiveOutflow);
    // An outflow face is qualified only for its outward sign.  An inward
    // carrier would require a resolved inflow state, so fail closed instead
    // of silently dropping that part of the demand.
    if (boundary_kind == outflow &&
        ((low_face && velocity > amrex::Real(0.0)) ||
         (!low_face && velocity < amrex::Real(0.0)))) {
        result.valid = 0;
        return result;
    }

    const amrex::Real face_mass_flux = velocity * rho_face;
    result.advective = (low_face ? amrex::max(-face_mass_flux, amrex::Real(0.0)) :
                                   amrex::max(face_mass_flux, amrex::Real(0.0))) *
                       inverse_spacing / rho_cell;
    if (!suppress_normal_sbm_diffusion(boundary_kind)) {
        result.diffusive = diffusion_coefficient * rho_face *
                           inverse_spacing * inverse_spacing / rho_cell;
    }
    if (amrex::isnan(result.advective) || amrex::isinf(result.advective) ||
        amrex::isnan(result.diffusive) || amrex::isinf(result.diffusive)) {
        result.valid = 0;
    }
    return result;
}

/** Fixed-size result for all active faces incident on one cell. */
struct HostCellDemand {
    amrex::Real advective{0.0};
    amrex::Real diffusive{0.0};
    amrex::Real total{0.0};
    int valid{1};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void accumulate_host_face(HostCellDemand& result,
                          const HostFaceDemand& face) noexcept
{
    result.advective += face.advective;
    result.diffusive += face.diffusive;
    result.valid = result.valid && face.valid;
}

/**
 * Evaluate the variable-density low-order host demand for one cell.  Physical
 * wall faces are deliberately skipped before any physical-side ghost is read.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HostCellDemand host_cell_demand(
    const amrex::Array4<const amrex::Real>& state,
    const amrex::Array4<const amrex::Real>& xvel,
    const amrex::Array4<const amrex::Real>& yvel,
    const amrex::Array4<const amrex::Real>& zvel,
    const int i, const int j, const int k,
    const int xlo, const int xhi, const int ylo, const int yhi,
    const int zlo, const int zhi,
    const int periodic_x, const int periodic_y, const int periodic_z,
    const int xlow_kind, const int xhigh_kind,
    const int ylow_kind, const int yhigh_kind,
    const int zlow_kind, const int zhigh_kind,
    const amrex::Real diffusion_coefficient,
    const amrex::Real dxi, const amrex::Real dyi, const amrex::Real dzi) noexcept
{
    HostCellDemand result;
    const amrex::Real rho_i = state(i,j,k,Rho_comp);
    if (rho_i <= amrex::Real(0.0) || amrex::isnan(rho_i) || amrex::isinf(rho_i)) {
        result.valid = 0;
        return result;
    }

    const int wall = static_cast<int>(BoundaryKind::ImpermeableWall);
    const bool xlow_physical = !periodic_x && i == xlo;
    const bool xhigh_physical = !periodic_x && i == xhi;
    if (!(xlow_physical && xlow_kind == wall)) {
        accumulate_host_face(result, host_face_demand(
            rho_i, state(i-1,j,k,Rho_comp), xvel(i,j,k),
            xlow_physical ? xlow_kind : static_cast<int>(BoundaryKind::Periodic),
            true, dxi, diffusion_coefficient));
    }
    if (!(xhigh_physical && xhigh_kind == wall)) {
        accumulate_host_face(result, host_face_demand(
            rho_i, state(i+1,j,k,Rho_comp), xvel(i+1,j,k),
            xhigh_physical ? xhigh_kind : static_cast<int>(BoundaryKind::Periodic),
            false, dxi, diffusion_coefficient));
    }

    const bool ylow_physical = !periodic_y && j == ylo;
    const bool yhigh_physical = !periodic_y && j == yhi;
    if (!(ylow_physical && ylow_kind == wall)) {
        accumulate_host_face(result, host_face_demand(
            rho_i, state(i,j-1,k,Rho_comp), yvel(i,j,k),
            ylow_physical ? ylow_kind : static_cast<int>(BoundaryKind::Periodic),
            true, dyi, diffusion_coefficient));
    }
    if (!(yhigh_physical && yhigh_kind == wall)) {
        accumulate_host_face(result, host_face_demand(
            rho_i, state(i,j+1,k,Rho_comp), yvel(i,j+1,k),
            yhigh_physical ? yhigh_kind : static_cast<int>(BoundaryKind::Periodic),
            false, dyi, diffusion_coefficient));
    }

    const bool zlow_physical = !periodic_z && k == zlo;
    const bool zhigh_physical = !periodic_z && k == zhi;
    if (!(zlow_physical && zlow_kind == wall)) {
        accumulate_host_face(result, host_face_demand(
            rho_i, state(i,j,k-1,Rho_comp), zvel(i,j,k),
            zlow_physical ? zlow_kind : static_cast<int>(BoundaryKind::Periodic),
            true, dzi, diffusion_coefficient));
    }
    if (!(zhigh_physical && zhigh_kind == wall)) {
        accumulate_host_face(result, host_face_demand(
            rho_i, state(i,j,k+1,Rho_comp), zvel(i,j,k+1),
            zhigh_physical ? zhigh_kind : static_cast<int>(BoundaryKind::Periodic),
            false, dzi, diffusion_coefficient));
    }

    result.total = result.advective + result.diffusive;
    if (amrex::isnan(result.total) || amrex::isinf(result.total)) result.valid = 0;
    return result;
}

} // namespace erf_sbm

#endif
