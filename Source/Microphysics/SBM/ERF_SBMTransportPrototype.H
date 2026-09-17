#ifndef ERF_SBM_TRANSPORT_PROTOTYPE_H_
#define ERF_SBM_TRANSPORT_PROTOTYPE_H_

#include "ERF_AuxiliaryStageContext.H"
#include "ERF_AuxiliaryFaceTransfer.H"
#include "ERF_AuxiliaryStateManager.H"
#include "ERF_SBMLayout.H"
#include "ERF_SBMFCT.H"
#include "ERF_SBMBoundary.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <cstddef>
#include <vector>

namespace erf_sbm {

using HostState = std::vector<amrex::Real>;

/**
 * Translation-covariant finite-volume WENO-Z3 face reconstruction.  The
 * production face kernel and host qualification tests use this same helper;
 * its smoothness indicators and regularizer depend only on differences and
 * machine precision, never on the absolute tracer amplitude.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno_z3_face_from_stencil(const amrex::Real qm2,
                                      const amrex::Real qm1,
                                      const amrex::Real q,
                                      const amrex::Real qp1,
                                      const amrex::Real carrier) noexcept
{
    const bool positive = carrier >= amrex::Real(0.0);
    const amrex::Real q0 = positive ?
        amrex::Real(0.5) * (-qm2 + amrex::Real(3.0) * qm1) :
        amrex::Real(0.5) * (amrex::Real(3.0) * q - qp1);
    const amrex::Real q1 = amrex::Real(0.5) * (qm1 + q);
    const amrex::Real beta0 = positive ? (qm1 - qm2) * (qm1 - qm2) :
        (qp1 - q) * (qp1 - q);
    const amrex::Real beta1 = (q - qm1) * (q - qm1);
    // Match ERF's canonical WENO_Z3 regularizer exactly: fixed precision
    // epsilon, with no beta-dependent floor or tracer-amplitude dependence.
    // Keeping this algebra identical makes the SBM adapter interchangeable
    // with ERF's native scalar reconstruction.
#ifdef AMREX_USE_FLOAT
    constexpr amrex::Real epsilon = amrex::Real(1.0e-12);
#else
    constexpr amrex::Real epsilon = amrex::Real(1.0e-40);
#endif
    const amrex::Real tau = amrex::Math::abs(beta1 - beta0);
    const amrex::Real w0 = (amrex::Real(1.0) / amrex::Real(3.0)) *
        (amrex::Real(1.0) + (tau * tau) /
         ((epsilon + beta0) * (epsilon + beta0)));
    const amrex::Real w1 = (amrex::Real(2.0) / amrex::Real(3.0)) *
        (amrex::Real(1.0) + (tau * tau) /
         ((epsilon + beta1) * (epsilon + beta1)));
    return (w0 * q0 + w1 * q1) / (w0 + w1);
}

/** Exact reduced-state recurrence used by the production auxiliary hook. */
void update_stage(const ::erf_auxiliary::StageContext& context,
                  const HostState& old_state,
                  const HostState& evaluation_state,
                  const HostState& rhs,
                  HostState& output_state);

/** Return the accepted full-step ledger for a sequence of stage face fluxes. */
HostState accepted_ledger(const ::erf_auxiliary::StageContext& context,
                          const std::vector<HostState>& stage_fluxes,
                          double full_step);

/**
 * First-order donor transport using ERF's already-constructed dry-air carrier
 * mass fluxes.  This is intentionally restricted to the qualified static
 * Cartesian P1/P2 auxiliary contract.
 */
void advance_stage(::erf_auxiliary::AuxiliaryStateManager& manager,
                   const SBMLayout& layout,
                   const ::erf_auxiliary::StageContext& context,
                   const amrex::MultiFab& rho_anchor,
                   const amrex::MultiFab& rho_input,
                   const amrex::MultiFab& rho_target,
                   amrex::MultiFab& core_state,
                   const amrex::MultiFab& carrier_x,
                   const amrex::MultiFab& carrier_y,
                   const amrex::MultiFab& carrier_z,
                   const amrex::Geometry& geometry,
                   ::erf_auxiliary::AuxiliaryFaceTransfer& stage_flux,
                   TransportMethod method = TransportMethod::DonorCell,
                   int level = 0,
                   amrex::Real diffusion_coefficient = amrex::Real(0.0),
                   int chunk_size = 0,
                   amrex::Real* minimum_accepted_limiter = nullptr,
                   const TransportBoundaryPolicy& boundary_policy = {});

/** Compatibility entry point for the pre-P2 two-density call contract. */
void advance_stage(::erf_auxiliary::AuxiliaryStateManager& manager,
                   const SBMLayout& layout,
                   const ::erf_auxiliary::StageContext& context,
                   const amrex::MultiFab& rho_anchor,
                   const amrex::MultiFab& rho_input,
                   amrex::MultiFab& core_state,
                   const amrex::MultiFab& carrier_x,
                   const amrex::MultiFab& carrier_y,
                   const amrex::MultiFab& carrier_z,
                   const amrex::Geometry& geometry,
                   ::erf_auxiliary::AuxiliaryFaceTransfer& stage_flux,
                   TransportMethod method = TransportMethod::DonorCell,
                   int level = 0,
                   amrex::Real diffusion_coefficient = amrex::Real(0.0),
                   int chunk_size = 0,
                   amrex::Real* minimum_accepted_limiter = nullptr,
                   const TransportBoundaryPolicy& boundary_policy = {});

/**
 * Compatibility entry point for callers that have one density view.  New
 * production callers must use the explicit anchor/input overload above.
 */
void advance_stage(::erf_auxiliary::AuxiliaryStateManager& manager,
                   const SBMLayout& layout,
                   const ::erf_auxiliary::StageContext& context,
                   const amrex::MultiFab& rho_evaluation,
                   amrex::MultiFab& core_state,
                   const amrex::MultiFab& carrier_x,
                   const amrex::MultiFab& carrier_y,
                   const amrex::MultiFab& carrier_z,
                   const amrex::Geometry& geometry,
                   ::erf_auxiliary::AuxiliaryFaceTransfer& stage_flux,
                   TransportMethod method = TransportMethod::DonorCell,
                   int level = 0,
                   amrex::Real diffusion_coefficient = amrex::Real(0.0),
                   int chunk_size = 0,
                   amrex::Real* minimum_accepted_limiter = nullptr,
                   const TransportBoundaryPolicy& boundary_policy = {});

/** Fail-closed finite/nonnegative check for authoritative spectral state. */
void validate_nonnegative_state(const amrex::MultiFab& state, int ncomp,
                                const char* context = "SBM auxiliary state");

/** Validate physical storage and two-moment endpoint realizability. */
void validate_admissible_state(::erf_auxiliary::AuxiliaryStateManager& manager,
                               const SBMLayout& layout, int level);

/**
 * Return the allocation-bound temporary payload for one production grouped
 * FCT chunk.  The estimate includes grown ratio/budget FABs and the three
 * chunk-local face candidates; authoritative state and accepted ledgers are
 * intentionally excluded because they are persistent state, not temporary
 * chunk work.
 */
[[nodiscard]] std::size_t grouped_fct_peak_working_bytes(
    const SBMLayout& layout, const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm, int chunk_size);

} // namespace erf_sbm

#endif
