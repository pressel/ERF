#ifndef ERF_SBM_BOUNDARY_H_
#define ERF_SBM_BOUNDARY_H_

#include "ERF_SBMConstraintGroups.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>
#include <vector>

namespace erf_sbm {

enum class BoundaryKind { Periodic, PrescribedSpectralInflow, AdvectiveOutflow, ImpermeableWall };

/** Shared production/oracle rule for an impermeable normal face. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool suppress_normal_sbm_transfer(const int boundary_kind) noexcept
{
    return boundary_kind == static_cast<int>(BoundaryKind::ImpermeableWall);
}

/** Production omits normal diffusion at walls and advective outflows. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool suppress_normal_sbm_diffusion(const int boundary_kind) noexcept
{
    return suppress_normal_sbm_transfer(boundary_kind) ||
           boundary_kind == static_cast<int>(BoundaryKind::AdvectiveOutflow);
}

/** Physical face policy used by the production auxiliary transport hook. */
struct TransportBoundaryPolicy {
    // Face numbering is 2*dir for low and 2*dir+1 for high.  The zero value
    // intentionally makes the default policy periodic in every direction.
    std::array<BoundaryKind, AMREX_SPACEDIM*2> face_kind{};
    bool configured{false};
};

struct BoundaryDescriptor {
    BoundaryKind kind{BoundaryKind::Periodic};
    int face_id{0};
    std::string semantic_id;
    std::vector<amrex::Real> prescribed_state;
};

struct BoundaryTransferBudget {
    std::vector<amrex::Real> auxiliary;
    amrex::Real qc{0.0};
    amrex::Real qr{0.0};
};

[[nodiscard]] bool validate_prescribed_inflow(const BoundaryDescriptor& boundary,
                                              const std::vector<ConstraintGroup>& groups,
                                              std::string* failure = nullptr);

/** Convert a boundary face flux into the same signed integrated-transfer ledger. */
[[nodiscard]] BoundaryTransferBudget
make_boundary_budget(const BoundaryDescriptor& boundary,
                     const std::vector<amrex::Real>& face_flux,
                     const std::vector<ConstraintGroup>& groups,
                     const std::vector<amrex::Real>& qc_projection,
                     const std::vector<amrex::Real>& qr_projection,
                     amrex::Real area, amrex::Real dt);

} // namespace erf_sbm

#endif
