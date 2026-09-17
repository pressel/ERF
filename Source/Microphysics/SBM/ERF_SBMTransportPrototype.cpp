#include "ERF_SBMTransportPrototype.H"

#include "ERF_IndexDefines.H"
#include "ERF_SBMBulkProjection.H"
#include "ERF_Interpolation_WENO_Z.H"

#include <AMReX_ParReduce.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_GpuUtility.H>
#include <AMReX_GpuContainers.H>

#include <cmath>
#include <array>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>

namespace erf_sbm {

using amrex::Real;

namespace {

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real donor_ratio (const amrex::Array4<const amrex::Real>& spectral,
                         const amrex::Array4<const amrex::Real>& density,
                         const int i, const int j, const int k, const int comp) noexcept
{
    const amrex::Real rho = density(i,j,k);
    return rho > amrex::Real(0.0) ? spectral(i,j,k,comp) / rho : amrex::Real(0.0);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void restrict_constraint (const amrex::Real delta_form,
                          const amrex::Real left_margin,
                          const amrex::Real right_margin,
                          const amrex::Real scale,
                          amrex::Real& lambda) noexcept
{
    // A positive face correction form decreases the left-cell constraint and
    // increases the right-cell constraint.  The signs here must match the
    // conservative divergence update below.
    const amrex::Real left_demand = scale * delta_form;
    const amrex::Real right_demand = -scale * delta_form;
    if (left_demand > amrex::Real(0.0)) {
        lambda = amrex::min(lambda, amrex::max(amrex::Real(0.0), left_margin / left_demand));
    }
    if (right_demand > amrex::Real(0.0)) {
        lambda = amrex::min(lambda, amrex::max(amrex::Real(0.0), right_margin / right_demand));
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real descriptor_form(const ConstraintDescriptor& descriptor,
                            const amrex::Array4<const amrex::Real>& state,
                            const int i, const int j, const int k) noexcept
{
    amrex::Real value = descriptor.coefficient0 * state(i,j,k,descriptor.component0);
    if (descriptor.term_count == 2) {
        value += descriptor.coefficient1 * state(i,j,k,descriptor.component1);
    }
    return value;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real descriptor_stage_baseline_form(
    const ConstraintDescriptor& descriptor,
    const amrex::Array4<const amrex::Real>& old_state,
    const amrex::Array4<const amrex::Real>& predictor_state,
    const int i, const int j, const int k,
    const bool heun_corrector) noexcept
{
    const amrex::Real old0 = old_state(i,j,k,descriptor.component0);
    const amrex::Real pred0 = predictor_state(i,j,k,descriptor.component0);
    const amrex::Real baseline0 = heun_corrector ? amrex::Real(0.5) * (old0 + pred0) : old0;
    amrex::Real value = descriptor.coefficient0 * baseline0;
    if (descriptor.term_count == 2) {
        const amrex::Real old1 = old_state(i,j,k,descriptor.component1);
        const amrex::Real pred1 = predictor_state(i,j,k,descriptor.component1);
        value += descriptor.coefficient1 *
            (heun_corrector ? amrex::Real(0.5) * (old1 + pred1) : old1);
    }
    return value;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real descriptor_stage_baseline_scale(
    const ConstraintDescriptor& descriptor,
    const amrex::Array4<const amrex::Real>& old_state,
    const amrex::Array4<const amrex::Real>& predictor_state,
    const int i, const int j, const int k,
    const bool heun_corrector) noexcept
{
    const amrex::Real old0 = old_state(i,j,k,descriptor.component0);
    const amrex::Real pred0 = predictor_state(i,j,k,descriptor.component0);
    const amrex::Real baseline0 = heun_corrector ? amrex::Real(0.5) * (old0 + pred0) : old0;
    amrex::Real scale = amrex::Math::abs(descriptor.coefficient0 * baseline0);
    if (descriptor.term_count == 2) {
        const amrex::Real old1 = old_state(i,j,k,descriptor.component1);
        const amrex::Real pred1 = predictor_state(i,j,k,descriptor.component1);
        const amrex::Real baseline1 = heun_corrector ? amrex::Real(0.5) * (old1 + pred1) : old1;
        scale += amrex::Math::abs(descriptor.coefficient1 * baseline1);
    }
    return scale;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real descriptor_face_form(const ConstraintDescriptor& descriptor,
                                 const amrex::Array4<const amrex::Real>& flux,
                                 const int i, const int j, const int k) noexcept
{
    amrex::Real value = descriptor.coefficient0 * flux(i,j,k,descriptor.component0);
    if (descriptor.term_count == 2) {
        value += descriptor.coefficient1 * flux(i,j,k,descriptor.component1);
    }
    return value;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void add_support_ratio(const amrex::Real property,
                       const amrex::Real carrier,
                       amrex::Real& lower, amrex::Real& upper,
                       int& invalid) noexcept
{
    const amrex::Real scale = amrex::Math::abs(property) + amrex::Math::abs(carrier);
    const amrex::Real tolerance = amrex::Real(128.0) *
        std::numeric_limits<amrex::Real>::epsilon() * scale;
    if (amrex::isnan(property) || amrex::isinf(property) ||
        amrex::isnan(carrier) || amrex::isinf(carrier) || carrier < amrex::Real(0.0) ||
        (carrier == amrex::Real(0.0) && amrex::Math::abs(property) > tolerance)) {
        invalid = 1;
        return;
    }
    if (carrier == amrex::Real(0.0)) return;
    const amrex::Real ratio = property / carrier;
    if (amrex::isnan(ratio) || amrex::isinf(ratio)) {
        invalid = 1;
        return;
    }
    lower = amrex::min(lower, ratio);
    upper = amrex::max(upper, ratio);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void add_support_sample(
    const amrex::Array4<const amrex::Real>& source,
    const AttachedPropertySupportDescriptor& descriptor,
    const int i, const int j, const int k,
    amrex::Real& lower, amrex::Real& upper, int& invalid) noexcept
{
    add_support_ratio(source(i,j,k,descriptor.property_component),
                      source(i,j,k,descriptor.carrier_component),
                      lower, upper, invalid);
}

// FV WENO-Z3 reconstruction at the face whose index is i (between cells
// i-1 and i).  The shared ERF helper is retained for the ordinary ERF scalar
// path, but its historical indexing assumes a right-face coordinate.  SBM's
// face ledgers use AMReX's conservative face convention, so this local
// adapter makes that convention explicit while retaining the same Z3 weights
// and two-cell ghost requirement.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real sbm_weno_z3_face(
    const amrex::Array4<const amrex::Real>& values,
    const int i, const int j, const int k, const int component,
    const amrex::Real carrier, const int dir) noexcept
{
    amrex::Real qm2 = amrex::Real(0.0);
    amrex::Real qm1 = amrex::Real(0.0);
    amrex::Real q = amrex::Real(0.0);
    amrex::Real qp1 = amrex::Real(0.0);
    if (dir == 0) {
        qm2 = values(i-2,j,k,component); qm1 = values(i-1,j,k,component);
        q = values(i,j,k,component); qp1 = values(i+1,j,k,component);
    } else if (dir == 1) {
        qm2 = values(i,j-2,k,component); qm1 = values(i,j-1,k,component);
        q = values(i,j,k,component); qp1 = values(i,j+1,k,component);
    } else {
        qm2 = values(i,j,k-2,component); qm1 = values(i,j,k-1,component);
        q = values(i,j,k,component); qp1 = values(i,j,k+1,component);
    }
    return weno_z3_face_from_stencil(qm2, qm1, q, qp1, carrier);
}

void validate_finite_face_transfer(const ::erf_auxiliary::AuxiliaryFaceTransfer& transfer,
                                   int ncomp, const char* context);

void validate_finite_multifab(const amrex::MultiFab& state, int ncomp, const char* context,
                              const amrex::IntVect& nghost = amrex::IntVect(0));

void validate_positive_density(const amrex::MultiFab& density,
                               const char* context,
                               const amrex::IntVect& nghost);

void validate_outflow_carrier(const amrex::MultiFab& carrier,
                              const amrex::Geometry& geometry,
                              const int dir, const int low_kind,
                              const int high_kind);

void validate_dynamic_support_state(
    const amrex::MultiFab& state,
    const SBMLayout& layout,
    const std::vector<AttachedPropertySupportDescriptor>& descriptors,
    const amrex::MultiFab& support_bounds,
    const int level,
    const char* context);

// Prepare the coordinate representation consumed by every low-order donor
// lookup.  The source remains authoritative (including its already prepared
// AMR ghosts); this FAB is only transport scratch.  In particular, evaluating
// over fabbox() is what makes a true coarse/fine donor available to DonorCell,
// while the final FillBoundary restores same-level/periodic fine ownership.
void prepare_transport_coordinates(
    const SBMLayout& layout,
    const std::vector<int>& components,
    const std::vector<int>& global_to_local,
    const amrex::MultiFab& source_state,
    const amrex::MultiFab& density_state,
    amrex::MultiFab& result,
    const amrex::Geometry& geometry)
{
    if (components.empty() || result.nComp() != static_cast<int>(components.size()) ||
        global_to_local.size() < static_cast<std::size_t>(layout.ncomp())) {
        throw std::invalid_argument("invalid SBM transport coordinate component map");
    }
    result.setVal(Real(0.0));
    const auto& population = layout.populations().front();
    const bool two_moment = population.moment_mode == MomentMode::TwoMoment;
    const amrex::Box domain = geometry.Domain();
    const int periodic_x = geometry.isPeriodic(0) ? 1 : 0;
    const int periodic_y = geometry.isPeriodic(1) ? 1 : 0;
    const int periodic_z = geometry.isPeriodic(2) ? 1 : 0;

    for (amrex::MFIter mfi(result); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.fabbox();
        const auto source = source_state.const_array(mfi);
        const auto rho = density_state.const_array(mfi);
        const auto out = result.array(mfi);
        for (std::size_t local = 0; local < components.size(); ++local) {
            const int global = components[local];
            ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real density = rho(i,j,k);
                const bool physical_ghost =
                    ((i < domain.smallEnd(0) || i > domain.bigEnd(0)) && !periodic_x) ||
                    ((j < domain.smallEnd(1) || j > domain.bigEnd(1)) && !periodic_y) ||
                    ((k < domain.smallEnd(2) || k > domain.bigEnd(2)) && !periodic_z);
                out(i,j,k,static_cast<int>(local)) = density > Real(0.0) ?
                    source(i,j,k,global) / density :
                    (physical_ghost ? Real(0.0) :
                     std::numeric_limits<Real>::quiet_NaN());
            });
        }

        if (two_moment) {
            for (int bin = 0; bin < population.grid.nbins(); ++bin) {
                const int mass = population.mass_offset + bin;
                const int number = population.number_offset + bin;
                const int mass_local = global_to_local[static_cast<std::size_t>(mass)];
                const int number_local = global_to_local[static_cast<std::size_t>(number)];
                if (mass_local < 0 || number_local < 0) continue;
                const Real lower = population.grid.edges()[static_cast<std::size_t>(bin)];
                const Real upper = population.grid.edges()[static_cast<std::size_t>(bin + 1)];
                const Real denominator = upper - lower;
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    const Real density = rho(i,j,k);
                    const bool physical_ghost =
                        ((i < domain.smallEnd(0) || i > domain.bigEnd(0)) && !periodic_x) ||
                        ((j < domain.smallEnd(1) || j > domain.bigEnd(1)) && !periodic_y) ||
                        ((k < domain.smallEnd(2) || k > domain.bigEnd(2)) && !periodic_z);
                    if (density <= Real(0.0)) {
                        const Real invalid = physical_ghost ? Real(0.0) :
                            std::numeric_limits<Real>::quiet_NaN();
                        out(i,j,k,mass_local) = invalid;
                        out(i,j,k,number_local) = invalid;
                        return;
                    }
                    const Real M = source(i,j,k,mass) / density;
                    const Real C = source(i,j,k,number) / density;
                    const Real scale = amrex::Math::abs(M) +
                        amrex::Math::abs(lower*C) + amrex::Math::abs(upper*C);
                    const Real tolerance = Real(128.0) *
                        std::numeric_limits<Real>::epsilon() * scale;
                    Real L = std::fma(upper, C, -M) / denominator;
                    Real H = std::fma(-lower, C, M) / denominator;
                    if (L < Real(0.0) && L >= -tolerance) L = Real(0.0);
                    if (H < Real(0.0) && H >= -tolerance) H = Real(0.0);
                    out(i,j,k,mass_local) = L;
                    out(i,j,k,number_local) = H;
                });
            }
        }
    }
    result.FillBoundary(geometry.periodicity());
}

/**
 * Production grouped transport implementation.  Every allocation in this
 * routine is sized by one complete closure chunk; the authoritative state and
 * the accepted ledger remain the only total-layout objects.  In particular,
 * no full-layout high/low/diffusion candidate is retained.
 */
void advance_stage_grouped_chunked(
    ::erf_auxiliary::AuxiliaryStateManager& manager,
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
    const int level,
    const amrex::Real diffusion_coefficient,
    const int chunk_size,
    amrex::Real* minimum_accepted_limiter,
    const TransportBoundaryPolicy& boundary_policy)
{
    const auto& population = layout.populations().front();
    const int ncomp = layout.ncomp();
    const int nbins = population.grid.nbins();
    const bool two_moment = population.moment_mode == MomentMode::TwoMoment;
    const auto& old = manager.old(level);
    const auto& predictor = manager.evaluation(level);
    auto& output = manager.output(level);
    const Real dt = static_cast<Real>(context.stage_interval);
    const Real dxi = geometry.InvCellSize(0);
    const Real dyi = geometry.InvCellSize(1);
    const Real dzi = geometry.InvCellSize(2);
    const Real anelastic_weight =
        (context.method == ::erf_auxiliary::IntegrationMethod::AnelasticHeun && context.stage_index > 0)
        ? Real(0.5) : Real(1.0);
    const bool heun_corrector =
        context.method == ::erf_auxiliary::IntegrationMethod::AnelasticHeun && context.stage_index > 0;
    const auto& low_source = heun_corrector ? predictor : old;
    const auto& high_source = context.stage_index == 0 ? old : predictor;
    const auto& low_density = heun_corrector ? rho_input : rho_anchor;
    const auto& high_density = context.stage_index == 0 ? rho_anchor : rho_input;

    std::array<int, AMREX_SPACEDIM*2> boundary_kinds{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (geometry.isPeriodic(dir)) continue;
        if (!boundary_policy.configured) {
            throw std::invalid_argument(
                "SBM nonperiodic transport requires a configured wall or outward-only outflow policy");
        }
        boundary_kinds[2*dir] = static_cast<int>(boundary_policy.face_kind[2*dir]);
        boundary_kinds[2*dir+1] = static_cast<int>(boundary_policy.face_kind[2*dir+1]);
        if (boundary_kinds[2*dir] == static_cast<int>(BoundaryKind::PrescribedSpectralInflow) ||
            boundary_kinds[2*dir+1] == static_cast<int>(BoundaryKind::PrescribedSpectralInflow)) {
            throw std::invalid_argument(
                "production prescribed spectral inflow is unsupported by the SBM transport hook");
        }
    }
    validate_outflow_carrier(carrier_x, geometry, 0, boundary_kinds[0], boundary_kinds[1]);
    validate_outflow_carrier(carrier_y, geometry, 1, boundary_kinds[2], boundary_kinds[3]);
    validate_outflow_carrier(carrier_z, geometry, 2, boundary_kinds[4], boundary_kinds[5]);

    validate_admissible_state(manager, layout, level);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (rho_anchor.nGrowVect()[dir] < 2 || rho_input.nGrowVect()[dir] < 2 ||
            rho_target.nGrowVect()[dir] < 2 ||
            low_source.nGrowVect()[dir] < 2 || high_source.nGrowVect()[dir] < 2) {
            throw std::invalid_argument(
                "SBM grouped transport requires two prepared density/source ghost cells for WENO and donor stencils");
        }
    }
    // Physical non-periodic ghosts are intentionally left unset by the
    // adapter; the boundary policy decides those faces explicitly.  Only
    // valid cells are therefore required to carry a positive density.
    validate_positive_density(rho_anchor, "SBM anchor density", amrex::IntVect(0));
    validate_positive_density(rho_input, "SBM input density", amrex::IntVect(0));
    validate_positive_density(rho_target, "SBM target density", amrex::IntVect(0));
    validate_finite_multifab(rho_anchor, 1, "SBM anchor density", rho_anchor.nGrowVect());
    validate_finite_multifab(rho_input, 1, "SBM input density", rho_input.nGrowVect());
    validate_finite_multifab(rho_target, 1, "SBM target density", rho_target.nGrowVect());
    validate_finite_multifab(carrier_x, 1, "SBM x carrier transfer");
    validate_finite_multifab(carrier_y, 1, "SBM y carrier transfer");
    validate_finite_multifab(carrier_z, 1, "SBM z carrier transfer");
    validate_finite_multifab(low_source, ncomp, "SBM low-order source state", low_source.nGrowVect());
    validate_finite_multifab(high_source, ncomp, "SBM high-order source state", high_source.nGrowVect());
    const auto groups = make_constraint_groups(layout);
    const auto descriptors = make_constraint_descriptors(layout);
    const auto support_descriptors = make_attached_property_support_descriptors(layout);
    // A zero API value is the explicit all-groups/reference mode retained for
    // callers that predate runtime chunk selection.  Runtime inputs normally
    // provide a positive atomic-group limit.
    const int effective_chunk_size = chunk_size == 0 ? static_cast<int>(groups.size()) : chunk_size;
    const auto chunks = make_constraint_closure_chunks(layout, effective_chunk_size);
    stage_flux.setVal(Real(0.0));
    std::unique_ptr<amrex::MultiFab> limiter_minimum;
    if (minimum_accepted_limiter != nullptr) {
        limiter_minimum = std::make_unique<amrex::MultiFab>(
            output.boxArray(), output.DistributionMap(), 1, 1);
        limiter_minimum->setVal(Real(1.0));
    }

    auto group_in_chunk = [](const ConstraintClosureChunk& chunk, const int group_index) {
        return std::find(chunk.group_indices.begin(), chunk.group_indices.end(), group_index) !=
               chunk.group_indices.end();
    };

    auto build_low = [&](const ConstraintClosureChunk& chunk,
                         const amrex::MultiFab& ratio,
                         const amrex::MultiFab& density_state,
                         ::erf_auxiliary::AuxiliaryFaceTransfer& low_adv,
                         ::erf_auxiliary::AuxiliaryFaceTransfer& low_diff) {
        low_adv.setVal(Real(0.0));
        low_diff.setVal(Real(0.0));
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto* carrier = dir == 0 ? &carrier_x : (dir == 1 ? &carrier_y : &carrier_z);
            const Real inverse_distance = geometry.InvCellSize(dir);
            auto& adv = low_adv.direction(dir);
            auto& diff = low_diff.direction(dir);
            for (amrex::MFIter mfi(adv); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto carrier_arr = carrier->const_array(mfi);
                const auto ratio_arr = ratio.const_array(mfi);
                const auto rho = density_state.const_array(mfi);
                const auto adv_arr = adv.array(mfi);
                const auto diff_arr = diff.array(mfi);
                const int domain_low = geometry.Domain().smallEnd(dir);
                const int domain_high = geometry.Domain().bigEnd(dir);
                const int low_kind = boundary_kinds[2*dir];
                const int high_kind = boundary_kinds[2*dir+1];
                for (std::size_t local = 0; local < chunk.components.size(); ++local) {
                    const int lc = static_cast<int>(local);
                    const int global = chunk.components[local];
                    int endpoint_kind = 0; // 1 = physical M, 2 = physical C
                    int endpoint_partner = -1;
                    Real lower = Real(0.0), upper = Real(0.0);
                    if (two_moment) {
                        for (int b = 0; b < nbins; ++b) {
                            const int mass = population.mass_offset + b;
                            const int number = population.number_offset + b;
                            if (global == mass) {
                                endpoint_kind = 1;
                                endpoint_partner = static_cast<int>(std::find(
                                    chunk.components.begin(), chunk.components.end(), number) - chunk.components.begin());
                                lower = population.grid.edges()[static_cast<std::size_t>(b)];
                                upper = population.grid.edges()[static_cast<std::size_t>(b + 1)];
                            } else if (global == number) {
                                endpoint_kind = 2;
                                endpoint_partner = static_cast<int>(std::find(
                                    chunk.components.begin(), chunk.components.end(), mass) - chunk.components.begin());
                            }
                        }
                    }
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int face_coordinate = dir == 0 ? i : (dir == 1 ? j : k);
                        const bool physical_low = face_coordinate == domain_low &&
                            low_kind != static_cast<int>(BoundaryKind::Periodic);
                        const bool physical_high = face_coordinate == domain_high + 1 &&
                            high_kind != static_cast<int>(BoundaryKind::Periodic);
                        const int physical_kind = physical_low ? low_kind :
                            (physical_high ? high_kind :
                             static_cast<int>(BoundaryKind::Periodic));
                        if (physical_kind == static_cast<int>(BoundaryKind::ImpermeableWall)) {
                            adv_arr(i,j,k,lc) = Real(0.0);
                            diff_arr(i,j,k,lc) = Real(0.0);
                            return;
                        }
                        const Real face_mass_flux = carrier_arr(i,j,k);
                        const int donor_i = (dir == 0 && face_mass_flux >= Real(0.0)) ? i-1 : i;
                        const int donor_j = (dir == 1 && face_mass_flux >= Real(0.0)) ? j-1 : j;
                        const int donor_k = (dir == 2 && face_mass_flux >= Real(0.0)) ? k-1 : k;
                        if (endpoint_kind == 1) {
                            adv_arr(i,j,k,lc) = face_mass_flux *
                                (lower*ratio_arr(donor_i,donor_j,donor_k,lc) +
                                 upper*ratio_arr(donor_i,donor_j,donor_k,endpoint_partner));
                        } else if (endpoint_kind == 2) {
                            adv_arr(i,j,k,lc) = face_mass_flux *
                                (ratio_arr(donor_i,donor_j,donor_k,endpoint_partner) +
                                 ratio_arr(donor_i,donor_j,donor_k,lc));
                        } else {
                            adv_arr(i,j,k,lc) = face_mass_flux * ratio_arr(donor_i,donor_j,donor_k,lc);
                        }
                        if (diffusion_coefficient > Real(0.0) && physical_kind !=
                            static_cast<int>(BoundaryKind::AdvectiveOutflow)) {
                            const int left_i = dir == 0 ? i-1 : i;
                            const int left_j = dir == 1 ? j-1 : j;
                            const int left_k = dir == 2 ? k-1 : k;
                            const Real rho_left = rho(left_i,left_j,left_k);
                            const Real rho_right = rho(i,j,k);
                            const Real rho_face = Real(0.5) * (rho_left + rho_right);
                            if (rho_left > Real(0.0) && rho_right > Real(0.0) && rho_face > Real(0.0)) {
                                const Real low_diff = -rho_face * diffusion_coefficient *
                                    (ratio_arr(i,j,k,lc) - ratio_arr(left_i,left_j,left_k,lc)) * inverse_distance;
                                if (endpoint_kind == 1) {
                                    const Real high_diff = -rho_face * diffusion_coefficient *
                                        (ratio_arr(i,j,k,endpoint_partner) -
                                         ratio_arr(left_i,left_j,left_k,endpoint_partner)) * inverse_distance;
                                    diff_arr(i,j,k,lc) = lower*low_diff + upper*high_diff;
                                } else if (endpoint_kind == 2) {
                                    const Real mass_diff = -rho_face * diffusion_coefficient *
                                        (ratio_arr(i,j,k,endpoint_partner) -
                                         ratio_arr(left_i,left_j,left_k,endpoint_partner)) * inverse_distance;
                                    diff_arr(i,j,k,lc) = mass_diff + low_diff;
                                } else {
                                    diff_arr(i,j,k,lc) = low_diff;
                                }
                            } else {
                                diff_arr(i,j,k,lc) = Real(0.0);
                            }
                        }
                    });
                }
            }
        }
        // The next consumer may read these candidates through a different
        // MultiFab view (or on a non-default GPU stream).  Make the producer
        // completion explicit before the low-order buffers are copied into
        // the full stage ledger.
        amrex::Gpu::synchronize();
    };

    // First pass constructs the low-order accepted flux in chunk-sized
    // buffers and publishes it into the full authoritative stage ledger.
    for (const auto& chunk : chunks) {
        const int nlocal = static_cast<int>(chunk.components.size());
        amrex::MultiFab ratio(output.boxArray(), output.DistributionMap(), nlocal, 2);
        ::erf_auxiliary::AuxiliaryFaceTransfer low_adv, low_diff;
        low_adv.define(output.boxArray(), output.DistributionMap(), nlocal, 0);
        low_diff.define(output.boxArray(), output.DistributionMap(), nlocal, 0);
        std::vector<int> global_to_local(static_cast<std::size_t>(ncomp), -1);
        for (int local = 0; local < nlocal; ++local) {
            global_to_local[static_cast<std::size_t>(chunk.components[static_cast<std::size_t>(local)])] = local;
        }
        prepare_transport_coordinates(layout, chunk.components, global_to_local,
                                      low_source, low_density, ratio, geometry);
        validate_finite_multifab(ratio, nlocal,
                                 "SBM grouped-FCT endpoint ratio", ratio.nGrowVect());
        build_low(chunk, ratio, low_density, low_adv, low_diff);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            auto& destination = stage_flux.direction(dir);
            const auto& adv = low_adv.direction(dir);
            const auto& diff = low_diff.direction(dir);
            for (amrex::MFIter mfi(destination); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto result = destination.array(mfi);
                const auto adv_arr = adv.const_array(mfi);
                const auto diff_arr = diff.const_array(mfi);
                for (int local = 0; local < nlocal; ++local) {
                    const int global = chunk.components[static_cast<std::size_t>(local)];
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        result(i,j,k,global) = adv_arr(i,j,k,local) + diff_arr(i,j,k,local);
                    });
                }
            }
        }
        amrex::Gpu::synchronize();
    }

    validate_finite_face_transfer(stage_flux, ncomp,
                                  "SBM grouped-FCT low-order stage transfer");

    // Apply the complete low-order recurrence once all chunks have populated
    // the full stage ledger.
    for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto old_arr = old.const_array(mfi);
        const auto pred = predictor.const_array(mfi);
        const auto fx = stage_flux.x().const_array(mfi);
        const auto fy = stage_flux.y().const_array(mfi);
        const auto fz = stage_flux.z().const_array(mfi);
        const auto out = output.array(mfi);
        ParallelFor(box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int c) noexcept {
            const Real rhs = -((fx(i+1,j,k,c)-fx(i,j,k,c))*dxi +
                               (fy(i,j+1,k,c)-fy(i,j,k,c))*dyi +
                               (fz(i,j,k+1,c)-fz(i,j,k,c))*dzi);
            out(i,j,k,c) = (!heun_corrector) ? old_arr(i,j,k,c) + dt*rhs :
                old_arr(i,j,k,c) + Real(0.5) *
                ((pred(i,j,k,c)-old_arr(i,j,k,c)) + dt*rhs);
        });
    }
    output.FillBoundary(geometry.periodicity());

    const auto old_baseline_arrays = old.const_arrays();
    const auto predictor_baseline_arrays = predictor.const_arrays();
    amrex::Gpu::ManagedVector<ConstraintDescriptor> all_descriptors;
    for (const auto& descriptor : descriptors) all_descriptors.push_back(descriptor);

    auto accumulate_budget = [&](const ::erf_auxiliary::AuxiliaryFaceTransfer& transfer,
                                 amrex::MultiFab& budget,
                                 const amrex::Gpu::ManagedVector<ConstraintDescriptor>& local_descriptors) {
        const int nconstraints = static_cast<int>(local_descriptors.size());
        const auto* descriptor_data = local_descriptors.data();
        budget.setVal(Real(0.0));
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto& face = transfer.direction(dir);
            const Real scale = dt * anelastic_weight * geometry.InvCellSize(dir);
            for (amrex::MFIter mfi(face); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto flux = face.const_array(mfi);
                const auto cell_budget = budget.array(mfi);
                for (int descriptor_index = 0; descriptor_index < nconstraints; ++descriptor_index) {
                    const auto descriptor = descriptor_data[descriptor_index];
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int li = dir == 0 ? i-1 : i;
                        const int lj = dir == 1 ? j-1 : j;
                        const int lk = dir == 2 ? k-1 : k;
                        const Real form = descriptor_face_form(descriptor, flux, i, j, k);
                        const Real left_demand = scale * form;
                        const Real right_demand = -scale * form;
                        if (left_demand > Real(0.0)) {
                            amrex::Gpu::Atomic::Add(&cell_budget(li,lj,lk,descriptor_index), left_demand);
                        }
                        if (right_demand > Real(0.0)) {
                            amrex::Gpu::Atomic::Add(&cell_budget(i,j,k,descriptor_index), right_demand);
                        }
                    });
                }
            }
        }
        amrex::Gpu::synchronize();
        budget.SumBoundary(geometry.periodicity(), true);
        budget.FillBoundary(geometry.periodicity());
    };

    // Rebuild each chunk's local low-order candidates, validate the combined
    // demand against the temporal baseline, then apply its common group-wise
    // limiter.  All temporary FABs die at the end of this iteration.
    for (const auto& chunk : chunks) {
        const int nlocal = static_cast<int>(chunk.components.size());
        std::vector<int> global_to_local(static_cast<std::size_t>(ncomp), -1);
        for (int local = 0; local < nlocal; ++local) {
            global_to_local[static_cast<std::size_t>(chunk.components[static_cast<std::size_t>(local)])] = local;
        }
        amrex::MultiFab ratio(output.boxArray(), output.DistributionMap(), nlocal, 2);
        ::erf_auxiliary::AuxiliaryFaceTransfer low_adv, low_diff;
        low_adv.define(output.boxArray(), output.DistributionMap(), nlocal, 0);
        low_diff.define(output.boxArray(), output.DistributionMap(), nlocal, 0);
        prepare_transport_coordinates(layout, chunk.components, global_to_local,
                                      low_source, low_density, ratio, geometry);
        build_low(chunk, ratio, low_density, low_adv, low_diff);

        std::vector<ConstraintDescriptor> local_descriptor_values;
        std::vector<ConstraintDescriptor> global_descriptor_values;
        for (const auto& descriptor : descriptors) {
            if (!group_in_chunk(chunk, descriptor.group_index)) continue;
            auto local_descriptor = descriptor;
            local_descriptor.group_index = static_cast<int>(
                std::find(chunk.group_indices.begin(), chunk.group_indices.end(), descriptor.group_index) -
                chunk.group_indices.begin());
            local_descriptor.component0 = global_to_local[static_cast<std::size_t>(descriptor.component0)];
            if (local_descriptor.term_count == 2) {
                local_descriptor.component1 = global_to_local[static_cast<std::size_t>(descriptor.component1)];
            }
            local_descriptor_values.push_back(local_descriptor);
            global_descriptor_values.push_back(descriptor);
        }
        amrex::Gpu::ManagedVector<ConstraintDescriptor> local_descriptors;
        amrex::Gpu::ManagedVector<ConstraintDescriptor> global_chunk_descriptors;
        for (const auto& descriptor : local_descriptor_values) local_descriptors.push_back(descriptor);
        for (const auto& descriptor : global_descriptor_values) global_chunk_descriptors.push_back(descriptor);
        const int nconstraints = static_cast<int>(local_descriptors.size());
        amrex::MultiFab advection_budget(output.boxArray(), output.DistributionMap(), nconstraints, 1);
        amrex::MultiFab diffusion_budget(output.boxArray(), output.DistributionMap(), nconstraints, 1);
        accumulate_budget(low_adv, advection_budget, local_descriptors);
        accumulate_budget(low_diff, diffusion_budget, local_descriptors);
        const auto advection_arrays = advection_budget.const_arrays();
        const auto diffusion_arrays = diffusion_budget.const_arrays();
        const auto* global_descriptor_data = global_chunk_descriptors.data();
        Real max_excess = Real(0.0);
        Real max_excess_cell = Real(0.0);
        int offending_descriptor = -1;
        for (int descriptor_index = 0; descriptor_index < nconstraints; ++descriptor_index) {
            const auto descriptor = global_descriptor_data[descriptor_index];
            const amrex::Box domain = geometry.Domain();
            const int nx = domain.length(0);
            const int ny = domain.length(1);
            const amrex::GpuTuple<Real, Real> detail = amrex::ParReduce(
                amrex::TypeList<amrex::ReduceOpMax, amrex::ReduceOpMax>{},
                amrex::TypeList<Real, Real>{}, output, amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                    -> amrex::GpuTuple<Real, Real> {
                    const Real value = descriptor_stage_baseline_form(
                        descriptor, old_baseline_arrays[box_no], predictor_baseline_arrays[box_no],
                        i, j, k, heun_corrector);
                    const Real scale = descriptor_stage_baseline_scale(
                        descriptor, old_baseline_arrays[box_no], predictor_baseline_arrays[box_no],
                        i, j, k, heun_corrector);
                    const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                    const Real excess = advection_arrays[box_no](i,j,k,descriptor_index) +
                        diffusion_arrays[box_no](i,j,k,descriptor_index) - value - tolerance;
                    const Real cell_key = static_cast<Real>(i-domain.smallEnd(0)) +
                        static_cast<Real>(nx) * (static_cast<Real>(j-domain.smallEnd(1)) +
                        static_cast<Real>(ny) * static_cast<Real>(k-domain.smallEnd(2)));
                    return {excess, excess > Real(0.0) ? cell_key : Real(0.0)};
                });
            const Real descriptor_excess = amrex::get<0>(detail);
            if (descriptor_excess > max_excess) {
                max_excess = descriptor_excess;
                max_excess_cell = amrex::get<1>(detail);
                offending_descriptor = descriptor_index;
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(max_excess);
        if (max_excess > Real(0.0)) {
            if (offending_descriptor < 0) offending_descriptor = 0;
            const auto& failed = global_descriptor_values[static_cast<std::size_t>(offending_descriptor)];
            const auto& group = groups[static_cast<std::size_t>(failed.group_index)];
            const auto& constraint = group.constraints[static_cast<std::size_t>(failed.constraint_index)];
            const amrex::Box domain = geometry.Domain();
            const int nx = domain.length(0);
            const int ny = domain.length(1);
            const long long packed = static_cast<long long>(max_excess_cell + Real(0.5));
            const int ck = static_cast<int>(packed / (static_cast<long long>(nx) * ny));
            const int cj = static_cast<int>((packed / nx) % ny);
            const int ci = static_cast<int>(packed % nx);
            std::ostringstream message;
            message << "SBM combined low-order advection+diffusion admissibility failure: level=" << level
                    << " cell=(" << (domain.smallEnd(0) + ci) << "," << (domain.smallEnd(1) + cj)
                    << "," << (domain.smallEnd(2) + ck) << ")"
                    << " group=" << group.semantic_id << " constraint=" << constraint.semantic_id
                    << " excess=" << max_excess << " dt=" << dt;
            throw std::domain_error(message.str());
        }
        amrex::Gpu::synchronize();
        validate_admissible_state(manager, layout, level);
    }
    validate_admissible_state(manager, layout, level);

    // Build and limit the high-order candidate a chunk at a time.
    for (const auto& chunk : chunks) {
        const int nlocal = static_cast<int>(chunk.components.size());
        std::vector<int> global_to_local(static_cast<std::size_t>(ncomp), -1);
        for (int local = 0; local < nlocal; ++local) {
            global_to_local[static_cast<std::size_t>(chunk.components[static_cast<std::size_t>(local)])] = local;
        }
        amrex::MultiFab ratio(output.boxArray(), output.DistributionMap(), nlocal, 2);
        ::erf_auxiliary::AuxiliaryFaceTransfer low_adv, low_diff, high;
        low_adv.define(output.boxArray(), output.DistributionMap(), nlocal, 0);
        low_diff.define(output.boxArray(), output.DistributionMap(), nlocal, 0);
        high.define(output.boxArray(), output.DistributionMap(), nlocal, 0);
        amrex::MultiFab high_ratio(output.boxArray(), output.DistributionMap(), nlocal, 2);
        prepare_transport_coordinates(layout, chunk.components, global_to_local,
                                      low_source, low_density, ratio, geometry);
        prepare_transport_coordinates(layout, chunk.components, global_to_local,
                                      high_source, high_density, high_ratio, geometry);
        build_low(chunk, ratio, low_density, low_adv, low_diff);

        // Derive a chunk-local attached-property support envelope from the
        // exact low-order donor set.  The envelope is intersected with finite
        // metadata bounds and is consumed by the same common group limiter as
        // the ordinary complete-group constraints.
        std::vector<AttachedPropertySupportDescriptor> global_support_values;
        std::vector<AttachedPropertySupportDescriptor> local_support_values;
        std::vector<int> support_group_first;
        std::vector<int> support_group_last;
        for (std::size_t local_group = 0; local_group < chunk.group_indices.size(); ++local_group) {
            const int group_index = chunk.group_indices[local_group];
            support_group_first.push_back(static_cast<int>(local_support_values.size()));
            for (const auto& descriptor : support_descriptors) {
                if (descriptor.group_index != group_index) continue;
                global_support_values.push_back(descriptor);
                auto local_descriptor = descriptor;
                local_descriptor.group_index = static_cast<int>(local_group);
                local_descriptor.property_component =
                    global_to_local[static_cast<std::size_t>(descriptor.property_component)];
                local_descriptor.mass_component =
                    global_to_local[static_cast<std::size_t>(descriptor.mass_component)];
                if (descriptor.number_component >= 0) {
                    local_descriptor.number_component =
                        global_to_local[static_cast<std::size_t>(descriptor.number_component)];
                }
                local_descriptor.carrier_component =
                    global_to_local[static_cast<std::size_t>(descriptor.carrier_component)];
                local_support_values.push_back(local_descriptor);
            }
            support_group_last.push_back(static_cast<int>(local_support_values.size()));
        }
        const int nsupport = static_cast<int>(local_support_values.size());
        std::unique_ptr<amrex::MultiFab> support_bounds;
        std::unique_ptr<amrex::MultiFab> support_budget;
        amrex::Gpu::ManagedVector<AttachedPropertySupportDescriptor> device_global_support;
        amrex::Gpu::ManagedVector<AttachedPropertySupportDescriptor> device_local_support;
        for (const auto& descriptor : global_support_values) device_global_support.push_back(descriptor);
        for (const auto& descriptor : local_support_values) device_local_support.push_back(descriptor);
        if (nsupport > 0) {
            support_bounds = std::make_unique<amrex::MultiFab>(
                output.boxArray(), output.DistributionMap(), 2*nsupport, 1);
            support_bounds->setVal(Real(0.0));
            const auto* support_data = device_global_support.data();
            const int domain_x_low = geometry.Domain().smallEnd(0);
            const int domain_x_high = geometry.Domain().bigEnd(0);
            const int domain_y_low = geometry.Domain().smallEnd(1);
            const int domain_y_high = geometry.Domain().bigEnd(1);
            const int domain_z_low = geometry.Domain().smallEnd(2);
            const int domain_z_high = geometry.Domain().bigEnd(2);
            const int x_low_kind = boundary_kinds[0];
            const int x_high_kind = boundary_kinds[1];
            const int y_low_kind = boundary_kinds[2];
            const int y_high_kind = boundary_kinds[3];
            const int z_low_kind = boundary_kinds[4];
            const int z_high_kind = boundary_kinds[5];
            const int wall = static_cast<int>(BoundaryKind::ImpermeableWall);
            const int periodic = static_cast<int>(BoundaryKind::Periodic);
            const bool include_diffusion_neighbors = diffusion_coefficient > Real(0.0);
            const bool include_old_state = heun_corrector;
            for (amrex::MFIter mfi(*support_bounds); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto source = low_source.const_array(mfi);
                const auto old_state = old.const_array(mfi);
                const auto fx = carrier_x.const_array(mfi);
                const auto fy = carrier_y.const_array(mfi);
                const auto fz = carrier_z.const_array(mfi);
                const auto bounds = support_bounds->array(mfi);
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    for (int s = 0; s < nsupport; ++s) {
                        const auto descriptor = support_data[s];
                        Real lower = std::numeric_limits<Real>::max();
                        Real upper = -std::numeric_limits<Real>::max();
                        int invalid = 0;

                        add_support_sample(source, descriptor, i, j, k,
                                           lower, upper, invalid);
                        if (include_old_state) {
                            add_support_sample(old_state, descriptor, i, j, k,
                                               lower, upper, invalid);
                        }

                        const bool x_left_wall = i == domain_x_low && x_low_kind != periodic &&
                            x_low_kind == wall;
                        const bool x_right_wall = i == domain_x_high && x_high_kind != periodic &&
                            x_high_kind == wall;
                        const bool y_left_wall = j == domain_y_low && y_low_kind != periodic &&
                            y_low_kind == wall;
                        const bool y_right_wall = j == domain_y_high && y_high_kind != periodic &&
                            y_high_kind == wall;
                        const bool z_left_wall = k == domain_z_low && z_low_kind != periodic &&
                            z_low_kind == wall;
                        const bool z_right_wall = k == domain_z_high && z_high_kind != periodic &&
                            z_high_kind == wall;

                        if (fx(i,j,k) > Real(0.0) && !x_left_wall) {
                            add_support_sample(source, descriptor, i-1, j, k,
                                               lower, upper, invalid);
                            if (include_old_state) {
                                add_support_sample(old_state, descriptor, i-1, j, k,
                                                   lower, upper, invalid);
                            }
                        }
                        if (fx(i+1,j,k) < Real(0.0) && !x_right_wall) {
                            add_support_sample(source, descriptor, i+1, j, k,
                                               lower, upper, invalid);
                            if (include_old_state) {
                                add_support_sample(old_state, descriptor, i+1, j, k,
                                                   lower, upper, invalid);
                            }
                        }
                        if (fy(i,j,k) > Real(0.0) && !y_left_wall) {
                            add_support_sample(source, descriptor, i, j-1, k,
                                               lower, upper, invalid);
                            if (include_old_state) {
                                add_support_sample(old_state, descriptor, i, j-1, k,
                                                   lower, upper, invalid);
                            }
                        }
                        if (fy(i,j+1,k) < Real(0.0) && !y_right_wall) {
                            add_support_sample(source, descriptor, i, j+1, k,
                                               lower, upper, invalid);
                            if (include_old_state) {
                                add_support_sample(old_state, descriptor, i, j+1, k,
                                                   lower, upper, invalid);
                            }
                        }
                        if (fz(i,j,k) > Real(0.0) && !z_left_wall) {
                            add_support_sample(source, descriptor, i, j, k-1,
                                               lower, upper, invalid);
                            if (include_old_state) {
                                add_support_sample(old_state, descriptor, i, j, k-1,
                                                   lower, upper, invalid);
                            }
                        }
                        if (fz(i,j,k+1) < Real(0.0) && !z_right_wall) {
                            add_support_sample(source, descriptor, i, j, k+1,
                                               lower, upper, invalid);
                            if (include_old_state) {
                                add_support_sample(old_state, descriptor, i, j, k+1,
                                                   lower, upper, invalid);
                            }
                        }

                        if (include_diffusion_neighbors) {
                            if (i > domain_x_low || x_low_kind == periodic) {
                                add_support_sample(source, descriptor, i-1, j, k,
                                                   lower, upper, invalid);
                            }
                            if (i < domain_x_high || x_high_kind == periodic) {
                                add_support_sample(source, descriptor, i+1, j, k,
                                                   lower, upper, invalid);
                            }
                            if (j > domain_y_low || y_low_kind == periodic) {
                                add_support_sample(source, descriptor, i, j-1, k,
                                                   lower, upper, invalid);
                            }
                            if (j < domain_y_high || y_high_kind == periodic) {
                                add_support_sample(source, descriptor, i, j+1, k,
                                                   lower, upper, invalid);
                            }
                            if (k > domain_z_low || z_low_kind == periodic) {
                                add_support_sample(source, descriptor, i, j, k-1,
                                                   lower, upper, invalid);
                            }
                            if (k < domain_z_high || z_high_kind == periodic) {
                                add_support_sample(source, descriptor, i, j, k+1,
                                                   lower, upper, invalid);
                            }
                        }

                        const bool have_donor = lower != std::numeric_limits<Real>::max();
                        if (invalid != 0 || (have_donor && lower > upper) ||
                            amrex::isnan(lower) || amrex::isinf(lower) ||
                            amrex::isnan(upper) || amrex::isinf(upper)) {
                            // A reversed finite interval is a fail-closed
                            // marker consumed by validate_dynamic_support_state.
                            bounds(i,j,k,2*s) = Real(1.0);
                            bounds(i,j,k,2*s+1) = Real(0.0);
                        } else if (!have_donor) {
                            bounds(i,j,k,2*s) = Real(0.0);
                            bounds(i,j,k,2*s+1) = Real(0.0);
                        } else {
                            const auto hard = support_data[s];
                            if (hard.has_hard_min) lower = amrex::max(lower, hard.hard_min);
                            if (hard.has_hard_max) upper = amrex::min(upper, hard.hard_max);
                            if (amrex::isnan(lower) || amrex::isinf(lower) ||
                                amrex::isnan(upper) || amrex::isinf(upper) || lower > upper) {
                                // Hard metadata can make an otherwise valid
                                // donor envelope empty. Preserve that fact
                                // as the same fail-closed marker used for an
                                // invalid donor sample.
                                bounds(i,j,k,2*s) = Real(1.0);
                                bounds(i,j,k,2*s+1) = Real(0.0);
                            } else {
                                bounds(i,j,k,2*s) = lower;
                                bounds(i,j,k,2*s+1) = upper;
                            }
                        }
                    }
                });
            }
            support_bounds->FillBoundary(geometry.periodicity());
            validate_dynamic_support_state(low_source, layout, global_support_values,
                                           *support_bounds, level,
                                           "SBM donor support envelope");
        }
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto* carrier = dir == 0 ? &carrier_x : (dir == 1 ? &carrier_y : &carrier_z);
            const auto& low_face = low_adv.direction(dir);
            auto& high_face = high.direction(dir);
            for (amrex::MFIter mfi(high_face); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto carrier_arr = carrier->const_array(mfi);
                const auto low_arr = low_face.const_array(mfi);
                const auto ratio_arr = high_ratio.const_array(mfi);
                const auto result = high_face.array(mfi);
                const int domain_low = geometry.Domain().smallEnd(dir);
                const int domain_high = geometry.Domain().bigEnd(dir);
                const int low_kind = boundary_kinds[2*dir];
                const int high_kind = boundary_kinds[2*dir+1];
                for (int local = 0; local < nlocal; ++local) {
                    const int global = chunk.components[static_cast<std::size_t>(local)];
                    int endpoint_kind = 0;
                    int endpoint_partner = -1;
                    Real lower = Real(0.0), upper = Real(0.0);
                    if (two_moment) {
                        for (int b = 0; b < nbins; ++b) {
                            const int mass = population.mass_offset + b;
                            const int number = population.number_offset + b;
                            if (global == mass) {
                                endpoint_kind = 1;
                                endpoint_partner = global_to_local[static_cast<std::size_t>(number)];
                                lower = population.grid.edges()[static_cast<std::size_t>(b)];
                                upper = population.grid.edges()[static_cast<std::size_t>(b + 1)];
                            } else if (global == number) {
                                endpoint_kind = 2;
                                endpoint_partner = global_to_local[static_cast<std::size_t>(mass)];
                            }
                        }
                    }
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int face_coordinate = dir == 0 ? i : (dir == 1 ? j : k);
                        const bool near_low = low_kind != static_cast<int>(BoundaryKind::Periodic) &&
                            face_coordinate <= domain_low + 1;
                        const bool near_high = high_kind != static_cast<int>(BoundaryKind::Periodic) &&
                            face_coordinate >= domain_high;
                        if (near_low || near_high) {
                            result(i,j,k,local) = low_arr(i,j,k,local);
                            return;
                        }
                        const Real mass_flux = carrier_arr(i,j,k);
                        const Real value = sbm_weno_z3_face(
                            ratio_arr, i, j, k, local, mass_flux, dir);
                        if (endpoint_kind == 1 || endpoint_kind == 2) {
                            const Real partner_value = sbm_weno_z3_face(
                                ratio_arr, i, j, k, endpoint_partner, mass_flux, dir);
                            result(i,j,k,local) = endpoint_kind == 1 ?
                                mass_flux * (lower*value + upper*partner_value) :
                                mass_flux * (partner_value + value);
                        } else {
                            result(i,j,k,local) = mass_flux * value;
                        }
                    });
                }
            }
        }

        std::vector<ConstraintDescriptor> local_descriptor_values;
        std::vector<ConstraintDescriptor> global_descriptor_values;
        std::vector<int> group_first_descriptor;
        std::vector<int> group_last_descriptor;
        for (std::size_t local_group = 0; local_group < chunk.group_indices.size(); ++local_group) {
            const int group_index = chunk.group_indices[local_group];
            group_first_descriptor.push_back(static_cast<int>(local_descriptor_values.size()));
            for (const auto& descriptor : descriptors) {
                if (descriptor.group_index != group_index) continue;
                auto local_descriptor = descriptor;
                local_descriptor.group_index = static_cast<int>(local_group);
                local_descriptor.component0 = global_to_local[static_cast<std::size_t>(descriptor.component0)];
                if (local_descriptor.term_count == 2) {
                    local_descriptor.component1 = global_to_local[static_cast<std::size_t>(descriptor.component1)];
                }
                local_descriptor_values.push_back(local_descriptor);
                global_descriptor_values.push_back(descriptor);
            }
            group_last_descriptor.push_back(static_cast<int>(local_descriptor_values.size()));
        }
        amrex::Gpu::ManagedVector<ConstraintDescriptor> local_descriptors;
        amrex::Gpu::ManagedVector<ConstraintDescriptor> global_chunk_descriptors;
        for (const auto& descriptor : local_descriptor_values) local_descriptors.push_back(descriptor);
        for (const auto& descriptor : global_descriptor_values) global_chunk_descriptors.push_back(descriptor);
        const int nconstraints = static_cast<int>(local_descriptors.size());
        const auto* local_descriptor_data = local_descriptors.data();
        amrex::MultiFab constraint_budget(output.boxArray(), output.DistributionMap(), nconstraints, 1);
        constraint_budget.setVal(Real(0.0));
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto& low_face = low_adv.direction(dir);
            const auto& diff_face = low_diff.direction(dir);
            auto& high_face = high.direction(dir);
            const Real scale = dt * anelastic_weight * geometry.InvCellSize(dir);
            for (amrex::MFIter mfi(high_face); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto low_arr = low_face.const_array(mfi);
                const auto high_arr = high_face.const_array(mfi);
                const auto budget = constraint_budget.array(mfi);
                for (int descriptor_index = 0; descriptor_index < nconstraints; ++descriptor_index) {
                    const auto descriptor = local_descriptor_data[descriptor_index];
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int li = dir == 0 ? i-1 : i;
                        const int lj = dir == 1 ? j-1 : j;
                        const int lk = dir == 2 ? k-1 : k;
                        const Real form = descriptor_face_form(descriptor, high_arr, i, j, k) -
                            descriptor_face_form(descriptor, low_arr, i, j, k);
                        const Real left_demand = scale * form;
                        const Real right_demand = -scale * form;
                        if (left_demand > Real(0.0)) {
                            amrex::Gpu::Atomic::Add(&budget(li,lj,lk,descriptor_index), left_demand);
                        }
                        if (right_demand > Real(0.0)) {
                            amrex::Gpu::Atomic::Add(&budget(i,j,k,descriptor_index), right_demand);
                        }
                    });
                }
            }
        }
        amrex::Gpu::synchronize();
        constraint_budget.SumBoundary(geometry.periodicity(), true);
        constraint_budget.FillBoundary(geometry.periodicity());
        if (nsupport > 0) {
            support_budget = std::make_unique<amrex::MultiFab>(
                output.boxArray(), output.DistributionMap(), 2*nsupport, 1);
            support_budget->setVal(Real(0.0));
            const auto* support_data = device_local_support.data();
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                const auto& low_face = low_adv.direction(dir);
                const auto& high_face = high.direction(dir);
                const Real scale = dt * anelastic_weight * geometry.InvCellSize(dir);
                for (amrex::MFIter mfi(high_face); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();
                    const auto low_arr = low_face.const_array(mfi);
                    const auto high_arr = high_face.const_array(mfi);
                    const auto bounds = support_bounds->const_array(mfi);
                    const auto budget = support_budget->array(mfi);
                    for (int s = 0; s < nsupport; ++s) {
                        const auto descriptor = support_data[s];
                        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            const int li = dir == 0 ? i-1 : i;
                            const int lj = dir == 1 ? j-1 : j;
                            const int lk = dir == 2 ? k-1 : k;
                            const Real delta_property =
                                high_arr(i,j,k,descriptor.property_component) -
                                low_arr(i,j,k,descriptor.property_component);
                            const Real delta_carrier =
                                high_arr(i,j,k,descriptor.carrier_component) -
                                low_arr(i,j,k,descriptor.carrier_component);
                            const Real left_lower_form = delta_property -
                                bounds(li,lj,lk,2*s) * delta_carrier;
                            const Real right_lower_form = -delta_property +
                                bounds(i,j,k,2*s) * delta_carrier;
                            const Real left_upper_form = bounds(li,lj,lk,2*s+1) *
                                delta_carrier - delta_property;
                            const Real right_upper_form = -bounds(i,j,k,2*s+1) *
                                delta_carrier + delta_property;
                            const Real left_lower_demand = scale * left_lower_form;
                            const Real right_lower_demand = scale * right_lower_form;
                            const Real left_upper_demand = scale * left_upper_form;
                            const Real right_upper_demand = scale * right_upper_form;
                            if (left_lower_demand > Real(0.0)) {
                                amrex::Gpu::Atomic::Add(
                                    &budget(li,lj,lk,2*s), left_lower_demand);
                            }
                            if (right_lower_demand > Real(0.0)) {
                                amrex::Gpu::Atomic::Add(
                                    &budget(i,j,k,2*s), right_lower_demand);
                            }
                            if (left_upper_demand > Real(0.0)) {
                                amrex::Gpu::Atomic::Add(
                                    &budget(li,lj,lk,2*s+1), left_upper_demand);
                            }
                            if (right_upper_demand > Real(0.0)) {
                                amrex::Gpu::Atomic::Add(
                                    &budget(i,j,k,2*s+1), right_upper_demand);
                            }
                        });
                    }
                }
            }
            amrex::Gpu::synchronize();
            support_budget->SumBoundary(geometry.periodicity(), true);
            support_budget->FillBoundary(geometry.periodicity());
        }
        const auto* global_descriptor_data = global_chunk_descriptors.data();
        amrex::Gpu::ManagedVector<int> device_group_first;
        amrex::Gpu::ManagedVector<int> device_group_last;
        for (const int value : group_first_descriptor) device_group_first.push_back(value);
        for (const int value : group_last_descriptor) device_group_last.push_back(value);
        amrex::Gpu::ManagedVector<int> device_group_component_offset;
        amrex::Gpu::ManagedVector<int> device_group_components;
        amrex::Gpu::ManagedVector<int> device_group_component_count;
        int component_offset = 0;
        for (const int group_index : chunk.group_indices) {
            const auto& group = groups[static_cast<std::size_t>(group_index)];
            device_group_component_offset.push_back(component_offset);
            for (const int component : group.members) {
                device_group_components.push_back(
                    global_to_local[static_cast<std::size_t>(component)]);
            }
            device_group_component_count.push_back(static_cast<int>(group.members.size()));
            component_offset += static_cast<int>(group.members.size());
        }
        const auto* group_component_data = device_group_components.data();
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto& low_face = low_adv.direction(dir);
            auto& high_face = high.direction(dir);
            const Real scale = dt * anelastic_weight * geometry.InvCellSize(dir);
            for (amrex::MFIter mfi(high_face); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto low_arr = low_face.const_array(mfi);
                const auto diff_arr = low_diff.direction(dir).const_array(mfi);
                const auto high_arr = high_face.const_array(mfi);
                const auto state = output.const_array(mfi);
                const auto budget = constraint_budget.const_array(mfi);
                amrex::Array4<const Real> support_bounds_arr;
                amrex::Array4<const Real> support_budget_arr;
                if (nsupport > 0) {
                    support_bounds_arr = support_bounds->const_array(mfi);
                    support_budget_arr = support_budget->const_array(mfi);
                }
                const auto* local_support_data = device_local_support.data();
                const auto* global_support_data = device_global_support.data();
                const auto result = high_face.array(mfi);
                amrex::Array4<Real> limiter_array;
                if (limiter_minimum != nullptr) limiter_array = limiter_minimum->array(mfi);
                const bool track_limiter = minimum_accepted_limiter != nullptr;
                for (std::size_t local_group = 0; local_group < chunk.group_indices.size(); ++local_group) {
                    const int first_descriptor = device_group_first[local_group];
                    const int last_descriptor = device_group_last[local_group];
                    const int first_component = device_group_component_offset[local_group];
                    const int component_count = device_group_component_count[local_group];
                    const int first_support = support_group_first[local_group];
                    const int last_support = support_group_last[local_group];
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int li = dir == 0 ? i-1 : i;
                        const int lj = dir == 1 ? j-1 : j;
                        const int lk = dir == 2 ? k-1 : k;
                        Real lambda = Real(1.0);
                        for (int d = first_descriptor; d < last_descriptor; ++d) {
                            const auto local_descriptor = local_descriptor_data[d];
                            const auto global_descriptor = global_descriptor_data[d];
                            const Real delta_form = descriptor_face_form(local_descriptor, high_arr, i, j, k) -
                                descriptor_face_form(local_descriptor, low_arr, i, j, k);
                            const Real left_demand = scale * delta_form;
                            const Real right_demand = -scale * delta_form;
                            if (left_demand > Real(0.0) && budget(li,lj,lk,d) > Real(0.0)) {
                                lambda = amrex::min(lambda,
                                    descriptor_form(global_descriptor, state, li, lj, lk) /
                                    budget(li,lj,lk,d));
                            }
                            if (right_demand > Real(0.0) && budget(i,j,k,d) > Real(0.0)) {
                                lambda = amrex::min(lambda,
                                    descriptor_form(global_descriptor, state, i, j, k) /
                                    budget(i,j,k,d));
                            }
                        }
                        for (int s = first_support; s < last_support; ++s) {
                            const auto local_support = local_support_data[s];
                            const auto global_support = global_support_data[s];
                            const Real delta_property =
                                high_arr(i,j,k,local_support.property_component) -
                                low_arr(i,j,k,local_support.property_component);
                            const Real delta_carrier =
                                high_arr(i,j,k,local_support.carrier_component) -
                                low_arr(i,j,k,local_support.carrier_component);
                            const Real lower_left = support_bounds_arr(li,lj,lk,2*s);
                            const Real lower_right = support_bounds_arr(i,j,k,2*s);
                            const Real upper_left = support_bounds_arr(li,lj,lk,2*s+1);
                            const Real upper_right = support_bounds_arr(i,j,k,2*s+1);
                            const Real lower_left_demand = scale *
                                (delta_property - lower_left * delta_carrier);
                            const Real lower_right_demand = scale *
                                (-delta_property + lower_right * delta_carrier);
                            const Real upper_left_demand = scale *
                                (upper_left * delta_carrier - delta_property);
                            const Real upper_right_demand = scale *
                                (-upper_right * delta_carrier + delta_property);
                            const Real lower_margin_left = state(
                                li,lj,lk,global_support.property_component) -
                                lower_left * state(li,lj,lk,global_support.carrier_component);
                            const Real lower_margin_right = state(
                                i,j,k,global_support.property_component) -
                                lower_right * state(i,j,k,global_support.carrier_component);
                            const Real upper_margin_left = upper_left * state(
                                li,lj,lk,global_support.carrier_component) -
                                state(li,lj,lk,global_support.property_component);
                            const Real upper_margin_right = upper_right * state(
                                i,j,k,global_support.carrier_component) -
                                state(i,j,k,global_support.property_component);
                            if (lower_left_demand > Real(0.0)) {
                                const Real budget_value = support_budget_arr(li,lj,lk,2*s);
                                lambda = amrex::min(lambda, budget_value > Real(0.0) ?
                                    lower_margin_left / budget_value : Real(0.0));
                            }
                            if (lower_right_demand > Real(0.0)) {
                                const Real budget_value = support_budget_arr(i,j,k,2*s);
                                lambda = amrex::min(lambda, budget_value > Real(0.0) ?
                                    lower_margin_right / budget_value : Real(0.0));
                            }
                            if (upper_left_demand > Real(0.0)) {
                                const Real budget_value = support_budget_arr(li,lj,lk,2*s+1);
                                lambda = amrex::min(lambda, budget_value > Real(0.0) ?
                                    upper_margin_left / budget_value : Real(0.0));
                            }
                            if (upper_right_demand > Real(0.0)) {
                                const Real budget_value = support_budget_arr(i,j,k,2*s+1);
                                lambda = amrex::min(lambda, budget_value > Real(0.0) ?
                                    upper_margin_right / budget_value : Real(0.0));
                            }
                        }
                        lambda = amrex::max(Real(0.0), amrex::min(Real(1.0), lambda));
                        if (track_limiter) {
                            amrex::Gpu::Atomic::Min(&limiter_array(i,j,k,0), lambda);
                        }
                        for (int offset = 0; offset < component_count; ++offset) {
                            const int local = group_component_data[first_component + offset];
                            const Real low_total = low_arr(i,j,k,local) + diff_arr(i,j,k,local);
                            result(i,j,k,local) = low_total + lambda *
                                (high_arr(i,j,k,local) - low_arr(i,j,k,local));
                        }
                    });
                }
            }
        }

        // Apply only this chunk's accepted correction to the authoritative
        // output, then publish its accepted face transfer into the full ledger.
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto& accepted = high.direction(dir);
            const auto& low_global = stage_flux.direction(dir);
            const Real inverse_length = geometry.InvCellSize(dir);
            for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto out = output.array(mfi);
                const auto accepted_arr = accepted.const_array(mfi);
                const auto low_arr = low_global.const_array(mfi);
                for (int local = 0; local < nlocal; ++local) {
                    const int global = chunk.components[static_cast<std::size_t>(local)];
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const Real high_right = accepted_arr(i + (dir == 0), j + (dir == 1), k + (dir == 2), local);
                        const Real low_right = low_arr(i + (dir == 0), j + (dir == 1), k + (dir == 2), global);
                        const Real high_left = accepted_arr(i,j,k,local);
                        const Real low_left = low_arr(i,j,k,global);
                        out(i,j,k,global) -= dt * anelastic_weight * inverse_length *
                            ((high_right-low_right) - (high_left-low_left));
                    });
                }
            }
            // Publish the accepted chunk with a single explicit traversal.  A
            // MultiFab::Copy here would create a second MFIter internally and
            // is not permitted while AMReX is still tracking another iterator
            // on some CPU/GPU configurations.
            auto& ledger = stage_flux.direction(dir);
            for (amrex::MFIter mfi(ledger); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto destination = ledger.array(mfi);
                const auto source = accepted.const_array(mfi);
                for (int local = 0; local < nlocal; ++local) {
                    const int global = chunk.components[static_cast<std::size_t>(local)];
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        destination(i,j,k,global) = source(i,j,k,local);
                    });
                }
            }
        }
        amrex::Gpu::synchronize();
        // A complete atomic group is validated immediately after its
        // accepted correction reaches the authoritative output.  The final
        // hierarchy-wide check below remains the post-stage/reflux check.
        validate_admissible_state(manager, layout, level);
        if (nsupport > 0) {
            validate_dynamic_support_state(output, layout, global_support_values,
                                           *support_bounds, level,
                                           "SBM post-FCT donor support state");
        }
    }

    const SBMBulkProjection bulk_projection(layout);
    for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
        // The compact target is not required to carry ghosts by this API.
        // Project only the common valid region here; lifecycle hooks project
        // the explicitly available compact ghosts after their fill/prolong.
        bulk_projection.apply_to_core(mfi.validbox(), output.const_array(mfi), core_state.array(mfi));
    }
    validate_nonnegative_state(output, ncomp, "SBM grouped-FCT output state");
    if (minimum_accepted_limiter != nullptr) {
        *minimum_accepted_limiter = limiter_minimum->min(0);
        amrex::ParallelDescriptor::ReduceRealMin(*minimum_accepted_limiter);
    }
    manager.accept_stage(level, context.output_time);
    manager.record_stage_face_transfer(level, context, stage_flux);
    core_state.FillBoundary(geometry.periodicity());
}

void validate_finite_face_transfer(const ::erf_auxiliary::AuxiliaryFaceTransfer& transfer,
                                   const int ncomp, const char* context)
{
    if (ncomp <= 0 || ncomp > transfer.ncomp()) {
        throw std::invalid_argument("invalid component count for auxiliary face-transfer check");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const auto& face = transfer.direction(dir);
        const auto& arrays = face.const_arrays();
        const auto local = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpLogicalOr, amrex::ReduceOpMax,
                            amrex::ReduceOpMax, amrex::ReduceOpMax,
                            amrex::ReduceOpMax>{},
            amrex::TypeList<int, int, int, int, int>{}, face, amrex::IntVect(0), ncomp,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int comp)
                -> amrex::GpuTuple<int, int, int, int, int> {
                const amrex::Real value = arrays[box_no](i,j,k,comp);
                const int bad = (amrex::isnan(value) || amrex::isinf(value)) ? 1 : 0;
                return {bad, bad ? i : -1, bad ? j : -1, bad ? k : -1, bad ? comp : -1};
            });
        int nonfinite = amrex::get<0>(local);
        int bad_i = amrex::get<1>(local);
        int bad_j = amrex::get<2>(local);
        int bad_k = amrex::get<3>(local);
        int bad_comp = amrex::get<4>(local);
        amrex::ParallelDescriptor::ReduceIntMax(nonfinite);
        amrex::ParallelDescriptor::ReduceIntMax(bad_i);
        amrex::ParallelDescriptor::ReduceIntMax(bad_j);
        amrex::ParallelDescriptor::ReduceIntMax(bad_k);
        amrex::ParallelDescriptor::ReduceIntMax(bad_comp);
        if (nonfinite != 0) {
            std::ostringstream message;
            message << context << " contains a nonfinite face transfer in direction " << dir;
            message << " at cell-face=(" << bad_i << "," << bad_j << "," << bad_k
                    << ") component=" << bad_comp;
            throw std::runtime_error(message.str());
        }
    }
}

void validate_finite_multifab(const amrex::MultiFab& state, const int ncomp,
                              const char* context,
                              const amrex::IntVect& nghost)
{
    if (ncomp <= 0 || ncomp > state.nComp()) {
        throw std::invalid_argument("invalid component count for MultiFab finite-state check");
    }
    const auto& arrays = state.const_arrays();
    const auto local = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpLogicalOr>{},
            amrex::TypeList<int>{}, state, nghost, ncomp,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int comp) {
            const amrex::Real value = arrays[box_no](i,j,k,comp);
            return (amrex::isnan(value) || amrex::isinf(value)) ? 1 : 0;
        });
    int nonfinite = local;
    amrex::ParallelDescriptor::ReduceIntMax(nonfinite);
    if (nonfinite != 0) {
        throw std::runtime_error(std::string(context) + " contains NaN or infinite values");
    }
}

void validate_positive_density(const amrex::MultiFab& density,
                               const char* context,
                               const amrex::IntVect& nghost)
{
    if (density.nComp() < 1) {
        throw std::invalid_argument("SBM density view has no component");
    }
    const auto& arrays = density.const_arrays();
    const auto local = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpLogicalOr>{},
        amrex::TypeList<int>{}, density, nghost, 1,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int) {
            const amrex::Real value = arrays[box_no](i,j,k,0);
            return (amrex::isnan(value) || amrex::isinf(value) ||
                    value <= amrex::Real(0.0)) ? 1 : 0;
        });
    int invalid = local;
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid != 0) {
        throw std::domain_error(std::string(context) +
                                " must be finite and strictly positive at every used stencil cell");
    }
}

void validate_outflow_carrier(const amrex::MultiFab& carrier,
                              const amrex::Geometry& geometry,
                              const int dir, const int low_kind,
                              const int high_kind)
{
    const int outflow = static_cast<int>(BoundaryKind::AdvectiveOutflow);
    if (low_kind != outflow && high_kind != outflow) return;
    const int domain_low = geometry.Domain().smallEnd(dir);
    const int domain_high = geometry.Domain().bigEnd(dir);
    const auto& arrays = carrier.const_arrays();
    const auto local = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpLogicalOr>{},
        amrex::TypeList<int>{}, carrier, amrex::IntVect(0), 1,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int) {
            const int face_coordinate = dir == 0 ? i : (dir == 1 ? j : k);
            const amrex::Real flux = arrays[box_no](i,j,k,0);
            const bool inward_low = low_kind == outflow && face_coordinate == domain_low &&
                flux > amrex::Real(0.0);
            const bool inward_high = high_kind == outflow && face_coordinate == domain_high + 1 &&
                flux < amrex::Real(0.0);
            return (inward_low || inward_high) ? 1 : 0;
        });
    int inward = local;
    amrex::ParallelDescriptor::ReduceIntMax(inward);
    if (inward != 0) {
        throw std::domain_error(
            "SBM open boundary received inward carrier flux but no prescribed spectral inflow is configured");
    }
}

void validate_dynamic_support_state(
    const amrex::MultiFab& state,
    const SBMLayout& layout,
    const std::vector<AttachedPropertySupportDescriptor>& descriptors,
    const amrex::MultiFab& support_bounds,
    const int level,
    const char* context)
{
    if (descriptors.empty()) return;
    if (support_bounds.nComp() != 2 * static_cast<int>(descriptors.size())) {
        throw std::invalid_argument("SBM dynamic support bounds have an inconsistent component count");
    }
    amrex::Gpu::ManagedVector<AttachedPropertySupportDescriptor> device_descriptors;
    for (const auto& descriptor : descriptors) device_descriptors.push_back(descriptor);
    const auto* descriptor_data = device_descriptors.data();
    const auto states = state.const_arrays();
    const auto bounds = support_bounds.const_arrays();
    const int nsupport = static_cast<int>(descriptors.size());
    const auto summary = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpLogicalOr, amrex::ReduceOpMax,
                        amrex::ReduceOpMax, amrex::ReduceOpMax,
                        amrex::ReduceOpMax>{},
        amrex::TypeList<int, int, int, int, int>{}, state, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
            -> amrex::GpuTuple<int, int, int, int, int> {
            int bad = 0;
            int bad_s = -1, bad_i = -1, bad_j = -1, bad_k = -1;
            for (int s = 0; s < nsupport; ++s) {
                const auto descriptor = descriptor_data[s];
                const Real property = states[box_no](i,j,k,descriptor.property_component);
                const Real carrier = states[box_no](i,j,k,descriptor.carrier_component);
                const Real lower = bounds[box_no](i,j,k,2*s);
                const Real upper = bounds[box_no](i,j,k,2*s+1);
                const Real scale = amrex::Math::abs(property) +
                    amrex::Math::abs(lower * carrier) + amrex::Math::abs(upper * carrier);
                const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                const Real lower_form = property - lower * carrier;
                const Real upper_form = upper * carrier - property;
                if (amrex::isnan(property) || amrex::isinf(property) ||
                    amrex::isnan(carrier) || amrex::isinf(carrier) || carrier < Real(0.0) ||
                    amrex::isnan(lower) || amrex::isinf(lower) ||
                    amrex::isnan(upper) || amrex::isinf(upper) || lower > upper ||
                    (carrier == Real(0.0) && amrex::Math::abs(property) > tolerance) ||
                    lower_form < -tolerance || upper_form < -tolerance) {
                    bad = 1;
                    bad_s = s;
                    bad_i = i; bad_j = j; bad_k = k;
                }
            }
            return {bad, bad_s, bad_i, bad_j, bad_k};
        });
    int bad = amrex::get<0>(summary);
    int bad_s = amrex::get<1>(summary);
    int bad_i = amrex::get<2>(summary);
    int bad_j = amrex::get<3>(summary);
    int bad_k = amrex::get<4>(summary);
    amrex::ParallelDescriptor::ReduceIntMax(bad);
    amrex::ParallelDescriptor::ReduceIntMax(bad_s);
    amrex::ParallelDescriptor::ReduceIntMax(bad_i);
    amrex::ParallelDescriptor::ReduceIntMax(bad_j);
    amrex::ParallelDescriptor::ReduceIntMax(bad_k);
    if (bad != 0) {
        // The device pass identifies the offending descriptor and reports
        // scalar diagnostics as reductions so this path remains GPU-safe.
        // The values are intentionally diagnostic only; the invariant has
        // already failed and no repair is attempted.
        if (bad_s < 0 || bad_s >= nsupport) bad_s = 0;
        const auto& descriptor = descriptors[static_cast<std::size_t>(bad_s)];
        const auto groups = make_constraint_groups(layout);
        const auto& group = groups[static_cast<std::size_t>(descriptor.group_index)];
        const auto& property = layout.attached_properties()[
            static_cast<std::size_t>(descriptor.property_index)];
        const auto detail = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpLogicalOr, amrex::ReduceOpMin,
                            amrex::ReduceOpMax, amrex::ReduceOpMax,
                            amrex::ReduceOpMax, amrex::ReduceOpMax,
                            amrex::ReduceOpMax>{},
            amrex::TypeList<int, Real, Real, Real, Real, Real, Real>{},
            state, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                -> amrex::GpuTuple<int, Real, Real, Real, Real, Real, Real> {
                const Real value = states[box_no](i,j,k,descriptor.property_component);
                const Real carrier = states[box_no](i,j,k,descriptor.carrier_component);
                const Real lower = bounds[box_no](i,j,k,2*bad_s);
                const Real upper = bounds[box_no](i,j,k,2*bad_s+1);
                const Real scale = amrex::Math::abs(value) +
                    amrex::Math::abs(lower * carrier) + amrex::Math::abs(upper * carrier);
                const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                const Real lower_form = value - lower * carrier;
                const Real upper_form = upper * carrier - value;
                const bool invalid = amrex::isnan(value) || amrex::isinf(value) ||
                    amrex::isnan(carrier) || amrex::isinf(carrier) || carrier < Real(0.0) ||
                    amrex::isnan(lower) || amrex::isinf(lower) ||
                    amrex::isnan(upper) || amrex::isinf(upper) || lower > upper ||
                    (carrier == Real(0.0) && amrex::Math::abs(value) > tolerance) ||
                    lower_form < -tolerance || upper_form < -tolerance;
                return {invalid ? 1 : 0,
                        invalid ? value : std::numeric_limits<Real>::max(),
                        invalid ? carrier : Real(0.0),
                        invalid ? lower : Real(0.0),
                        invalid ? upper : Real(0.0),
                        invalid ? scale : Real(0.0),
                        invalid ? tolerance : Real(0.0)};
            });
        int detail_bad = amrex::get<0>(detail);
        Real detail_value = amrex::get<1>(detail);
        Real detail_carrier = amrex::get<2>(detail);
        Real detail_lower = amrex::get<3>(detail);
        Real detail_upper = amrex::get<4>(detail);
        Real detail_scale = amrex::get<5>(detail);
        Real detail_tolerance = amrex::get<6>(detail);
        amrex::ParallelDescriptor::ReduceIntMax(detail_bad);
        amrex::ParallelDescriptor::ReduceRealMin(detail_value);
        amrex::ParallelDescriptor::ReduceRealMax(detail_carrier);
        amrex::ParallelDescriptor::ReduceRealMax(detail_lower);
        amrex::ParallelDescriptor::ReduceRealMax(detail_upper);
        amrex::ParallelDescriptor::ReduceRealMax(detail_scale);
        amrex::ParallelDescriptor::ReduceRealMax(detail_tolerance);
        std::ostringstream message;
        message << context << " violates a donor-derived attached-property support envelope"
                << ": level=" << level
                << " cell=(" << bad_i << "," << bad_j << "," << bad_k << ")"
                << " group=" << group.semantic_id
                << " constraint=" << property.semantic_id + ".donor_support"
                << " value=" << detail_value
                << " carrier=" << detail_carrier
                << " support=[" << detail_lower << "," << detail_upper << "]"
                << " scale=" << detail_scale
                << " tolerance=" << detail_tolerance;
        throw std::domain_error(message.str());
    }
}

} // namespace

void update_stage(const ::erf_auxiliary::StageContext& context,
                  const HostState& old_state,
                  const HostState& evaluation_state,
                  const HostState& rhs,
                  HostState& output_state)
{
    if (old_state.size() != rhs.size() || evaluation_state.size() != old_state.size()) {
        throw std::invalid_argument("auxiliary stage vectors have inconsistent sizes");
    }
    output_state.resize(old_state.size());
    if (context.method == ::erf_auxiliary::IntegrationMethod::CompressibleRK3) {
        // ERF uses the old full-step baseline for all three callback updates.
        for (std::size_t n = 0; n < old_state.size(); ++n) {
            output_state[n] = old_state[n] + static_cast<amrex::Real>(context.stage_interval) * rhs[n];
        }
    } else if (context.stage_index == 0) {
        for (std::size_t n = 0; n < old_state.size(); ++n) {
            output_state[n] = old_state[n] + static_cast<amrex::Real>(context.stage_interval) * rhs[n];
        }
    } else {
        // Anelastic Heun: old + 1/2[(predictor-old) + h R1].
        for (std::size_t n = 0; n < old_state.size(); ++n) {
            output_state[n] = old_state[n] + static_cast<amrex::Real>(0.5) *
                ((evaluation_state[n] - old_state[n]) +
                 static_cast<amrex::Real>(context.stage_interval) * rhs[n]);
        }
    }
}

HostState accepted_ledger(const ::erf_auxiliary::StageContext& context,
                          const std::vector<HostState>& stage_fluxes,
                          const double full_step)
{
    if ((context.method == ::erf_auxiliary::IntegrationMethod::CompressibleRK3 && stage_fluxes.size() != 3) ||
        (context.method == ::erf_auxiliary::IntegrationMethod::AnelasticHeun && stage_fluxes.size() != 2)) {
        throw std::invalid_argument("wrong number of stage fluxes for auxiliary ledger");
    }
    if (stage_fluxes.empty()) return {};
    HostState result(stage_fluxes.front().size(), amrex::Real(0.0));
    if (context.method == ::erf_auxiliary::IntegrationMethod::CompressibleRK3) {
        for (std::size_t n = 0; n < result.size(); ++n) {
            result[n] = context.completes_level_step ? static_cast<amrex::Real>(full_step) * stage_fluxes[2][n] : amrex::Real(0.0);
        }
    } else {
        for (std::size_t n = 0; n < result.size(); ++n) {
            result[n] = static_cast<amrex::Real>(0.5 * full_step) * (stage_fluxes[0][n] + stage_fluxes[1][n]);
        }
    }
    return result;
}

void validate_nonnegative_state(const amrex::MultiFab& state, const int ncomp,
                                const char* context)
{
    if (ncomp <= 0 || ncomp > state.nComp()) {
        throw std::invalid_argument("invalid component count for auxiliary finite-state check");
    }

    // One local GPU-capable traversal covers every requested valid cell and
    // requested component and returns the complete validity tuple.  Ghost
    // cells are communication caches, not authoritative physical state; the
    // transport path fills them before use, but they are not part of this
    // invariant.  ParReduce is local in AMReX, so the three fixed-size
    // reductions below are the only MPI collectives; their count is
    // independent of ncomp.
    const auto& arrays = state.const_arrays();
    const auto local = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpLogicalOr, amrex::ReduceOpMin,
                        amrex::ReduceOpMax>{},
        amrex::TypeList<int, amrex::Real, amrex::Real>{},
        state, amrex::IntVect(0), ncomp,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int comp)
            -> amrex::GpuTuple<int, amrex::Real, amrex::Real> {
            const amrex::Real value = arrays[box_no](i,j,k,comp);
            const int nonfinite = (amrex::isnan(value) || amrex::isinf(value)) ? 1 : 0;
            return {nonfinite, value, amrex::Math::abs(value)};
        });
    int has_nonfinite = amrex::get<0>(local);
    amrex::Real minimum = amrex::get<1>(local);
    amrex::Real maximum_absolute = amrex::max(amrex::Real(0.0), amrex::get<2>(local));
    amrex::ParallelDescriptor::ReduceIntMax(has_nonfinite);
    amrex::ParallelDescriptor::ReduceRealMin(minimum);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_absolute);
    if (has_nonfinite != 0) {
        throw std::runtime_error(std::string(context) + " contains NaN or infinite values");
    }

    // Scale only with the global maximum magnitude.  There is deliberately
    // no order-one floor: an all-zero state has tau_neg == 0.
    const amrex::Real tolerance = maximum_absolute == amrex::Real(0.0) ?
        amrex::Real(0.0) : amrex::Real(128.0) *
        std::numeric_limits<amrex::Real>::epsilon() * maximum_absolute;
    if (minimum < -tolerance) {
        std::ostringstream message;
        message << context << " has material negative value: minimum=" << minimum
                << ", global_max_abs=" << maximum_absolute
                << ", tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}

void validate_finite_multifab(const amrex::MultiFab& state, const int ncomp,
                              const char* context)
{
    if (ncomp <= 0 || ncomp > state.nComp()) {
        throw std::invalid_argument("invalid component count for MultiFab finite-state check");
    }
    const auto& arrays = state.const_arrays();
    const auto local = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpLogicalOr>{},
        amrex::TypeList<int>{}, state, amrex::IntVect(0), ncomp,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int comp) {
            const amrex::Real value = arrays[box_no](i,j,k,comp);
            return (amrex::isnan(value) || amrex::isinf(value)) ? 1 : 0;
        });
    int nonfinite = local;
    amrex::ParallelDescriptor::ReduceIntMax(nonfinite);
    if (nonfinite != 0) {
        throw std::runtime_error(std::string(context) + " contains NaN or infinite values");
    }
}

void validate_admissible_state(::erf_auxiliary::AuxiliaryStateManager& manager,
                               const SBMLayout& layout, const int level)
{
    if (!manager.has_level(level)) throw std::invalid_argument("SBM admissibility check references an undefined level");
    const auto& source = manager.output(level);
    validate_nonnegative_state(source, layout.ncomp(), "SBM post-reflux physical state");

    const auto groups = make_constraint_groups(layout);
    const auto descriptors = make_constraint_descriptors(layout);
    amrex::Gpu::ManagedVector<ConstraintDescriptor> device_descriptors;
    for (const auto& descriptor : descriptors) device_descriptors.push_back(descriptor);
    const auto* descriptor_data = device_descriptors.data();
    const int nconstraints = static_cast<int>(descriptors.size());
    const auto states = source.const_arrays();
    const auto summary = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpLogicalOr, amrex::ReduceOpMax,
                        amrex::ReduceOpMax, amrex::ReduceOpMax>{},
        amrex::TypeList<int, int, int, int>{}, source, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
            -> amrex::GpuTuple<int, int, int, int> {
            int bad = 0;
            int bad_i = -1, bad_j = -1, bad_k = -1;
            for (int d = 0; d < nconstraints; ++d) {
                const auto descriptor = descriptor_data[d];
                const Real value = descriptor_form(descriptor, states[box_no], i, j, k);
                const Real scale = descriptor.term_count == 2 ?
                    amrex::Math::abs(descriptor.coefficient0 * states[box_no](i,j,k,descriptor.component0)) +
                    amrex::Math::abs(descriptor.coefficient1 * states[box_no](i,j,k,descriptor.component1)) :
                    amrex::Math::abs(descriptor.coefficient0 * states[box_no](i,j,k,descriptor.component0));
                const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                if (amrex::isnan(value) || amrex::isinf(value) || value < -tolerance) {
                    bad = 1; bad_i = i; bad_j = j; bad_k = k;
                }
            }
            return {bad, bad_i, bad_j, bad_k};
        });
    int bad = amrex::get<0>(summary);
    int bad_i = amrex::get<1>(summary), bad_j = amrex::get<2>(summary), bad_k = amrex::get<3>(summary);
    amrex::ParallelDescriptor::ReduceIntMax(bad);
    amrex::ParallelDescriptor::ReduceIntMax(bad_i);
    amrex::ParallelDescriptor::ReduceIntMax(bad_j);
    amrex::ParallelDescriptor::ReduceIntMax(bad_k);
    if (bad == 0) return;

    int failed_descriptor = -1;
    Real failed_margin = Real(0.0);
    Real failed_scale = Real(0.0);
    for (int d = 0; d < nconstraints; ++d) {
        const auto descriptor = descriptors[static_cast<std::size_t>(d)];
        const auto detail = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpLogicalOr, amrex::ReduceOpMin,
                            amrex::ReduceOpMax>{},
            amrex::TypeList<int, Real, Real>{}, source, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                -> amrex::GpuTuple<int, Real, Real> {
                const Real value = descriptor_form(descriptor, states[box_no], i, j, k);
                const Real scale = descriptor.term_count == 2 ?
                    amrex::Math::abs(descriptor.coefficient0 * states[box_no](i,j,k,descriptor.component0)) +
                    amrex::Math::abs(descriptor.coefficient1 * states[box_no](i,j,k,descriptor.component1)) :
                    amrex::Math::abs(descriptor.coefficient0 * states[box_no](i,j,k,descriptor.component0));
                const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                const int invalid = (amrex::isnan(value) || amrex::isinf(value) || value < -tolerance) ? 1 : 0;
                return {invalid, value, scale};
            });
        int descriptor_bad = amrex::get<0>(detail);
        failed_margin = amrex::get<1>(detail);
        Real descriptor_scale = amrex::get<2>(detail);
        amrex::ParallelDescriptor::ReduceIntMax(descriptor_bad);
        amrex::ParallelDescriptor::ReduceRealMin(failed_margin);
        amrex::ParallelDescriptor::ReduceRealMax(descriptor_scale);
        if (failed_descriptor < 0 && descriptor_bad != 0) failed_descriptor = d;
        if (descriptor_bad != 0) failed_scale = amrex::max(failed_scale, descriptor_scale);
    }
    if (failed_descriptor < 0) failed_descriptor = 0;
    const auto& descriptor = descriptors[static_cast<std::size_t>(failed_descriptor)];
    const auto& group = groups[static_cast<std::size_t>(descriptor.group_index)];
    const auto& constraint = group.constraints[static_cast<std::size_t>(descriptor.constraint_index)];
    const Real failed_tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * failed_scale;
    std::ostringstream message;
    message << "SBM complete constraint validation failed after reflux/regrid/restart: level=" << level
            << " cell=(" << bad_i << "," << bad_j << "," << bad_k << ") group=" << group.semantic_id
            << " constraint=" << constraint.semantic_id << " value=" << failed_margin
            << " scale=" << failed_scale << " tolerance=" << failed_tolerance;
    throw std::domain_error(message.str());
}

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
                   const TransportMethod method,
                   const int level,
                   const amrex::Real diffusion_coefficient,
                   const int chunk_size,
                   amrex::Real* minimum_accepted_limiter,
                   const TransportBoundaryPolicy& boundary_policy)
{
    if (layout.populations().size() != 1 || !manager.has_level(level)) {
        throw std::invalid_argument("ERF SBM transport requires one initialized runtime population and level");
    }
    if (geometry.Domain().length(0) <= 0) {
        throw std::invalid_argument("SBM transport requires a nonempty Cartesian domain");
    }
    if (!stage_flux.defined() || stage_flux.ncomp() != layout.ncomp()) {
        throw std::invalid_argument("P1 stage face-transfer storage does not match the spectral layout");
    }
    if (!std::isfinite(diffusion_coefficient) || diffusion_coefficient < amrex::Real(0.0)) {
        throw std::invalid_argument("SBM explicit diffusion coefficient must be finite and nonnegative");
    }
    if (method == TransportMethod::GroupedFCT_WENOZ3 && chunk_size < 0) {
        throw std::invalid_argument("SBM grouped-FCT chunk size must be nonnegative");
    }
    const auto& population = layout.populations().front();
    if (layout.ncomp() < population.component_count) {
        throw std::invalid_argument("ERF SBM transport layout is smaller than its population storage");
    }
    const int first = population.mass_offset;
    const int nbins = population.grid.nbins();
    const bool two_moment = population.moment_mode == MomentMode::TwoMoment;
    const int ncomp = layout.ncomp();
    const auto& old = manager.old(level);
    const auto& predictor = manager.evaluation(level);
    const bool heun_corrector =
        context.method == ::erf_auxiliary::IntegrationMethod::AnelasticHeun && context.stage_index > 0;
    const auto& evaluation = heun_corrector ? predictor : old;
    auto& output = manager.output(level);
    // Make the two density views explicit, aligned with the auxiliary boxes,
    // and periodicity-complete before any donor or WENO stencil is evaluated.
    // This also gives non-periodic boundary policies deterministic uncovered
    // ghosts (zero) that are never used by an outward donor.
    amrex::MultiFab rho_anchor_prepared(output.boxArray(), output.DistributionMap(), 1, 2);
    amrex::MultiFab rho_input_prepared(output.boxArray(), output.DistributionMap(), 1, 2);
    amrex::MultiFab rho_target_prepared(output.boxArray(), output.DistributionMap(), 1, 2);
    rho_anchor_prepared.setVal(Real(0.0));
    rho_input_prepared.setVal(Real(0.0));
    rho_target_prepared.setVal(Real(0.0));
    // ERF's state views have already gone through its level FillPatch path.
    // Preserve as many of those prepared ghosts as are available instead of
    // replacing coarse/fine stencil data with an uninitialized scratch layer.
    amrex::IntVect anchor_copy_nghost = rho_anchor.nGrowVect();
    amrex::IntVect input_copy_nghost = rho_input.nGrowVect();
    amrex::IntVect target_copy_nghost = rho_target.nGrowVect();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        anchor_copy_nghost[dir] = std::min(anchor_copy_nghost[dir], 2);
        input_copy_nghost[dir] = std::min(input_copy_nghost[dir], 2);
        target_copy_nghost[dir] = std::min(target_copy_nghost[dir], 2);
    }
    amrex::MultiFab::Copy(rho_anchor_prepared, rho_anchor, 0, 0, 1, anchor_copy_nghost);
    amrex::MultiFab::Copy(rho_input_prepared, rho_input, 0, 0, 1, input_copy_nghost);
    amrex::MultiFab::Copy(rho_target_prepared, rho_target, 0, 0, 1, target_copy_nghost);
    rho_anchor_prepared.FillBoundary(geometry.periodicity());
    rho_input_prepared.FillBoundary(geometry.periodicity());
    rho_target_prepared.FillBoundary(geometry.periodicity());
    // The auxiliary manager owns the source views, but the ERF callback may
    // have written only their valid cells.  Publish periodic source ghosts
    // before either donor or WENO reads a stencil; physical ghosts remain the
    // boundary-policy responsibility and are never synthesized here.
    manager.old(level).FillBoundary(geometry.periodicity());
    manager.evaluation(level).FillBoundary(geometry.periodicity());
    std::array<int, AMREX_SPACEDIM*2> boundary_kinds{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (geometry.isPeriodic(dir)) continue;
        if (!boundary_policy.configured) {
            throw std::invalid_argument(
                "SBM nonperiodic transport requires a configured wall or outward-only outflow policy");
        }
        boundary_kinds[2*dir] = static_cast<int>(boundary_policy.face_kind[2*dir]);
        boundary_kinds[2*dir+1] = static_cast<int>(boundary_policy.face_kind[2*dir+1]);
        if (boundary_kinds[2*dir] == static_cast<int>(BoundaryKind::PrescribedSpectralInflow) ||
            boundary_kinds[2*dir+1] == static_cast<int>(BoundaryKind::PrescribedSpectralInflow)) {
            throw std::invalid_argument(
                "production prescribed spectral inflow is unsupported by the SBM transport hook");
        }
    }
    validate_outflow_carrier(carrier_x, geometry, 0, boundary_kinds[0], boundary_kinds[1]);
    validate_outflow_carrier(carrier_y, geometry, 1, boundary_kinds[2], boundary_kinds[3]);
    validate_outflow_carrier(carrier_z, geometry, 2, boundary_kinds[4], boundary_kinds[5]);
    const auto& transport_density = heun_corrector ? rho_input_prepared : rho_anchor_prepared;
    const amrex::Real dt = static_cast<amrex::Real>(context.stage_interval);
    const amrex::Real dxi = static_cast<amrex::Real>(geometry.InvCellSize(0));
    const amrex::Real dyi = static_cast<amrex::Real>(geometry.InvCellSize(1));
    const amrex::Real dzi = static_cast<amrex::Real>(geometry.InvCellSize(2));
    if (!std::isfinite(dt) || dt < Real(0.0)) {
        throw std::invalid_argument("SBM stage interval must be finite and nonnegative");
    }
    const Real diffusion_bound = dt * diffusion_coefficient * (dxi*dxi + dyi*dyi + dzi*dzi);
    if (diffusion_bound > Real(0.5)) {
        std::ostringstream message;
        message << "SBM explicit diffusion timestep exceeds admissible bound: dt=" << dt
                << " bound=" << (diffusion_coefficient > Real(0.0) ?
                    Real(0.5) / (diffusion_coefficient * (dxi*dxi + dyi*dyi + dzi*dzi)) :
                    std::numeric_limits<Real>::infinity())
                << " diffusion_coefficient=" << diffusion_coefficient;
        throw std::invalid_argument(message.str());
    }

    if (method == TransportMethod::GroupedFCT_WENOZ3) {
        advance_stage_grouped_chunked(manager, layout, context, rho_anchor_prepared,
                                      rho_input_prepared, rho_target_prepared, core_state,
                                      carrier_x, carrier_y, carrier_z, geometry, stage_flux,
                                      level, diffusion_coefficient, chunk_size,
                                      minimum_accepted_limiter, boundary_policy);
        return;
    }

    validate_positive_density(rho_anchor_prepared, "SBM anchor density", amrex::IntVect(0));
    validate_positive_density(rho_input_prepared, "SBM input density", amrex::IntVect(0));
    validate_positive_density(rho_target_prepared, "SBM target density", amrex::IntVect(0));

    if (minimum_accepted_limiter != nullptr) *minimum_accepted_limiter = amrex::Real(1.0);

    auto& transport_scratch = manager.scratch(level);
    stage_flux.setVal(amrex::Real(0.0));

    // Two-moment storage is (M,C), while transport uses nonnegative endpoint
    // variables (L,H).  The same fabbox-wide preparation used by grouped FCT
    // also initializes true coarse/fine donors here; FillBoundary alone cannot
    // create those values.  The temporary is bounded by the runtime layout.
    if (two_moment) {
        std::vector<int> components(static_cast<std::size_t>(layout.ncomp()));
        std::vector<int> global_to_local(static_cast<std::size_t>(layout.ncomp()), -1);
        for (int component = 0; component < layout.ncomp(); ++component) {
            components[static_cast<std::size_t>(component)] = component;
            global_to_local[static_cast<std::size_t>(component)] = component;
        }
        prepare_transport_coordinates(layout, components, global_to_local,
                                      evaluation, transport_density,
                                      transport_scratch, geometry);
    }

    // Construct each numerical face flux exactly once on its face-centered
    // MultiFab.  The same arrays feed the divergence below and the accepted
    // full-step ledger; no cell-centered or independently reconstructed bulk
    // flux is involved.
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const auto* carrier = dir == 0 ? &carrier_x : (dir == 1 ? &carrier_y : &carrier_z);
        auto& flux = stage_flux.direction(dir);
        for (amrex::MFIter mfi(flux); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto carrier_arr = carrier->const_array(mfi);
            const auto eval = two_moment ? transport_scratch.const_array(mfi) : evaluation.const_array(mfi);
            const auto physical = evaluation.const_array(mfi);
            const auto rho = transport_density.const_array(mfi);
            const auto out = flux.array(mfi);
            const Real inverse_distance = geometry.InvCellSize(dir);
            const int domain_low = geometry.Domain().smallEnd(dir);
            const int domain_high = geometry.Domain().bigEnd(dir);
            const int low_kind = boundary_kinds[2*dir];
            const int high_kind = boundary_kinds[2*dir+1];
            for (int b = 0; b < nbins; ++b) {
                const int mass = first + b;
                const int number = two_moment ? population.number_offset + b : -1;
                const Real lower = two_moment ? population.grid.edges()[static_cast<std::size_t>(b)] : Real(0.0);
                const Real upper = two_moment ? population.grid.edges()[static_cast<std::size_t>(b+1)] : Real(0.0);
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    const amrex::Real face_mass_flux = carrier_arr(i,j,k);
                    const int face_coordinate = dir == 0 ? i : (dir == 1 ? j : k);
                    const bool physical_low = face_coordinate == domain_low &&
                        low_kind != static_cast<int>(BoundaryKind::Periodic);
                    const bool physical_high = face_coordinate == domain_high + 1 &&
                        high_kind != static_cast<int>(BoundaryKind::Periodic);
                    const int physical_kind = physical_low ? low_kind :
                        (physical_high ? high_kind : static_cast<int>(BoundaryKind::Periodic));
                    if (physical_kind == static_cast<int>(BoundaryKind::ImpermeableWall)) {
                        out(i,j,k,mass) = Real(0.0);
                        if (two_moment) out(i,j,k,number) = Real(0.0);
                        return;
                    }
                    const int donor_i = (dir == 0 && face_mass_flux >= amrex::Real(0.0)) ? i-1 : i;
                    const int donor_j = (dir == 1 && face_mass_flux >= amrex::Real(0.0)) ? j-1 : j;
                    const int donor_k = (dir == 2 && face_mass_flux >= amrex::Real(0.0)) ? k-1 : k;
                    if (two_moment) {
                        const Real left_endpoint = face_mass_flux * eval(donor_i,donor_j,donor_k,mass);
                        const Real right_endpoint = face_mass_flux * eval(donor_i,donor_j,donor_k,number);
                        out(i,j,k,mass) = lower*left_endpoint + upper*right_endpoint;
                        out(i,j,k,number) = left_endpoint + right_endpoint;
                    } else {
                        out(i,j,k,mass) = face_mass_flux *
                            donor_ratio(eval, rho, donor_i, donor_j, donor_k, mass);
                    }
                    if (diffusion_coefficient > Real(0.0) && physical_kind !=
                        static_cast<int>(BoundaryKind::AdvectiveOutflow)) {
                        const int left_i = dir == 0 ? i-1 : i;
                        const int left_j = dir == 1 ? j-1 : j;
                        const int left_k = dir == 2 ? k-1 : k;
                        const int right_i = i, right_j = j, right_k = k;
                        const Real rho_left = rho(left_i,left_j,left_k);
                        const Real rho_right = rho(right_i,right_j,right_k);
                        const Real rho_face = Real(0.5) * (rho_left + rho_right);
                        if (rho_left > Real(0.0) && rho_right > Real(0.0) && rho_face > Real(0.0)) {
                            if (two_moment) {
                                const Real ldiff = -rho_face * diffusion_coefficient *
                                    (eval(right_i,right_j,right_k,mass) - eval(left_i,left_j,left_k,mass)) * inverse_distance;
                                const Real hdiff = -rho_face * diffusion_coefficient *
                                    (eval(right_i,right_j,right_k,number) - eval(left_i,left_j,left_k,number)) * inverse_distance;
                                out(i,j,k,mass) += lower*ldiff + upper*hdiff;
                                out(i,j,k,number) += ldiff + hdiff;
                            } else {
                                const Real mdiff = -rho_face * diffusion_coefficient *
                                    (donor_ratio(physical, rho, right_i,right_j,right_k,mass) -
                                     donor_ratio(physical, rho, left_i,left_j,left_k,mass)) * inverse_distance;
                                out(i,j,k,mass) += mdiff;
                            }
                        }
                    }
                });
            }
            // Attached carrier-bin properties share the same donor mass flux
            // and density-weighted intensive transport as the carrier state.
            for (int c = 0; c < ncomp; ++c) {
                const bool is_mass = c >= first && c < first + nbins;
                const bool is_number = two_moment && c >= population.number_offset &&
                                       c < population.number_offset + nbins;
                if (is_mass || is_number) continue;
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    const Real face_mass_flux = carrier_arr(i,j,k);
                    const int face_coordinate = dir == 0 ? i : (dir == 1 ? j : k);
                    const bool physical_low = face_coordinate == domain_low &&
                        low_kind != static_cast<int>(BoundaryKind::Periodic);
                    const bool physical_high = face_coordinate == domain_high + 1 &&
                        high_kind != static_cast<int>(BoundaryKind::Periodic);
                    const int physical_kind = physical_low ? low_kind :
                        (physical_high ? high_kind : static_cast<int>(BoundaryKind::Periodic));
                    if (physical_kind == static_cast<int>(BoundaryKind::ImpermeableWall)) {
                        out(i,j,k,c) = Real(0.0);
                        return;
                    }
                    const int donor_i = (dir == 0 && face_mass_flux >= Real(0.0)) ? i-1 : i;
                    const int donor_j = (dir == 1 && face_mass_flux >= Real(0.0)) ? j-1 : j;
                    const int donor_k = (dir == 2 && face_mass_flux >= Real(0.0)) ? k-1 : k;
                    const Real rho_left = rho(dir == 0 ? i-1 : i,
                                              dir == 1 ? j-1 : j,
                                              dir == 2 ? k-1 : k);
                    const Real rho_right = rho(i,j,k);
                    const Real advection = face_mass_flux * donor_ratio(physical, rho, donor_i, donor_j, donor_k, c);
                    out(i,j,k,c) = advection;
                    if (diffusion_coefficient > Real(0.0) && physical_kind !=
                        static_cast<int>(BoundaryKind::AdvectiveOutflow) &&
                        rho_left > Real(0.0) && rho_right > Real(0.0)) {
                        const Real rho_face = Real(0.5) * (rho_left + rho_right);
                        const int left_i = dir == 0 ? i-1 : i;
                        const int left_j = dir == 1 ? j-1 : j;
                        const int left_k = dir == 2 ? k-1 : k;
                        const Real pdiff = -rho_face * diffusion_coefficient *
                            (physical(i,j,k,c)/rho_right - physical(left_i,left_j,left_k,c)/rho_left) *
                            inverse_distance;
                        out(i,j,k,c) += pdiff;
                    }
                });
            }
        }
    }

    // The former full-layout grouped candidate implementation is intentionally
    // retired.  Grouped-FCT dispatches to advance_stage_grouped_chunked above;
    // the remaining legacy donor path has no high-order candidate storage.
#if 0
    if (high_flux) {
        // Historical source-contract reference: the active grouped path above
        // uses its finite-volume WENO-Z3 adapter with the same carrier flux;
        // only the candidate differs from the donor low-order transfer.
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto* carrier = dir == 0 ? &carrier_x : (dir == 1 ? &carrier_y : &carrier_z);
            auto& flux = high_flux->direction(dir);
            for (amrex::MFIter mfi(flux); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto carrier_arr = carrier->const_array(mfi);
                const auto ratio = transport_scratch.const_array(mfi);
                const auto out = flux.array(mfi);
                const auto density = rho_evaluation.const_array(mfi);
                const auto physical = evaluation.const_array(mfi);
                const Real inverse_distance = geometry.InvCellSize(dir);
                for (int b = 0; b < nbins; ++b) {
                    const int mass = first + b;
                    const int number = two_moment ? population.number_offset + b : -1;
                    const Real lower = two_moment ? population.grid.edges()[static_cast<std::size_t>(b)] : Real(0.0);
                    const Real upper = two_moment ? population.grid.edges()[static_cast<std::size_t>(b+1)] : Real(0.0);
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const Real mass_flux = carrier_arr(i,j,k);
                        WENO_Z3 weno(ratio, Real(0.0));
                        Real lo = 0.0;
                        if (dir == 0) weno.InterpolateInX(i,j,k,mass,lo,mass_flux);
                        else if (dir == 1) weno.InterpolateInY(i,j,k,mass,lo,mass_flux);
                        else weno.InterpolateInZ(i,j,k,mass,lo,mass_flux);
                        if (two_moment) {
                            Real hi = 0.0;
                            if (dir == 0) weno.InterpolateInX(i,j,k,number,hi,mass_flux);
                            else if (dir == 1) weno.InterpolateInY(i,j,k,number,hi,mass_flux);
                            else weno.InterpolateInZ(i,j,k,number,hi,mass_flux);
                            const Real left_flux = mass_flux * lo;
                            const Real right_flux = mass_flux * hi;
                            out(i,j,k,mass) = lower*left_flux + upper*right_flux;
                            out(i,j,k,number) = left_flux + right_flux;
                        } else {
                            out(i,j,k,mass) = mass_flux * lo;
                        }
                        if (diffusion_coefficient > Real(0.0)) {
                            const int left_i = dir == 0 ? i-1 : i;
                            const int left_j = dir == 1 ? j-1 : j;
                            const int left_k = dir == 2 ? k-1 : k;
                            const Real rho_left = density(left_i,left_j,left_k);
                            const Real rho_right = density(i,j,k);
                            const Real rho_face = Real(0.5) * (rho_left + rho_right);
                            if (rho_left > Real(0.0) && rho_right > Real(0.0) && rho_face > Real(0.0)) {
                                if (two_moment) {
                                    // The endpoint reconstructed values are
                                    // in lo/hi; reconstruct the low-side
                                    // diffusion gradient from the endpoint
                                    // ratios in the neighboring cells.
                                    const Real ldiff = -rho_face * diffusion_coefficient *
                                        (ratio(i,j,k,mass) - ratio(left_i,left_j,left_k,mass)) * inverse_distance;
                                    const Real hdiff = -rho_face * diffusion_coefficient *
                                        (ratio(i,j,k,number) - ratio(left_i,left_j,left_k,number)) * inverse_distance;
                                    out(i,j,k,mass) += lower*ldiff + upper*hdiff;
                                    out(i,j,k,number) += ldiff + hdiff;
                                } else {
                                    out(i,j,k,mass) += -rho_face * diffusion_coefficient *
                                        (physical(i,j,k,mass)/rho_right - physical(left_i,left_j,left_k,mass)/rho_left) * inverse_distance;
                                }
                            }
                        }
                    });
                }
                for (int c = 0; c < ncomp; ++c) {
                    const bool is_mass = c >= first && c < first + nbins;
                    const bool is_number = two_moment && c >= population.number_offset &&
                                           c < population.number_offset + nbins;
                    if (is_mass || is_number) continue;
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const Real mass_flux = carrier_arr(i,j,k);
                        WENO_Z3 weno(ratio, Real(0.0));
                        Real value = Real(0.0);
                        if (dir == 0) weno.InterpolateInX(i,j,k,c,value,mass_flux);
                        else if (dir == 1) weno.InterpolateInY(i,j,k,c,value,mass_flux);
                        else weno.InterpolateInZ(i,j,k,c,value,mass_flux);
                        out(i,j,k,c) = mass_flux * value;
                        if (diffusion_coefficient > Real(0.0)) {
                            const int left_i = dir == 0 ? i-1 : i;
                            const int left_j = dir == 1 ? j-1 : j;
                            const int left_k = dir == 2 ? k-1 : k;
                            const Real rho_left = density(left_i,left_j,left_k);
                            const Real rho_right = density(i,j,k);
                            const Real rho_face = Real(0.5) * (rho_left + rho_right);
                            if (rho_left > Real(0.0) && rho_right > Real(0.0) && rho_face > Real(0.0)) {
                                out(i,j,k,c) += -rho_face * diffusion_coefficient *
                                    (physical(i,j,k,c)/rho_right - physical(left_i,left_j,left_k,c)/rho_left) * inverse_distance;
                            }
                        }
                    });
                }
            }
        }
    }
#endif

    for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto old_arr = old.const_array(mfi);
        const auto pred = predictor.const_array(mfi);
        const auto fx = stage_flux.x().const_array(mfi);
        const auto fy = stage_flux.y().const_array(mfi);
        const auto fz = stage_flux.z().const_array(mfi);
        const auto out = output.array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            for (int b = 0; b < nbins; ++b) {
                const int n = first + b;
                const amrex::Real rhs = -((fx(i+1,j,k,n)-fx(i,j,k,n))*dxi +
                                           (fy(i,j+1,k,n)-fy(i,j,k,n))*dyi +
                                           (fz(i,j,k+1,n)-fz(i,j,k,n))*dzi);
                if (context.method == ::erf_auxiliary::IntegrationMethod::CompressibleRK3 || context.stage_index == 0) {
                    out(i,j,k,n) = old_arr(i,j,k,n) + dt*rhs;
                } else {
                    out(i,j,k,n) = old_arr(i,j,k,n) + amrex::Real(0.5) *
                        ((pred(i,j,k,n)-old_arr(i,j,k,n)) + dt*rhs);
                }
                if (two_moment) {
                    const int number = population.number_offset + b;
                    const amrex::Real rhs_number = -((fx(i+1,j,k,number)-fx(i,j,k,number))*dxi +
                                                      (fy(i,j+1,k,number)-fy(i,j,k,number))*dyi +
                                                      (fz(i,j,k+1,number)-fz(i,j,k,number))*dzi);
                    if (context.method == ::erf_auxiliary::IntegrationMethod::CompressibleRK3 || context.stage_index == 0) {
                        out(i,j,k,number) = old_arr(i,j,k,number) + dt*rhs_number;
                    } else {
                        out(i,j,k,number) = old_arr(i,j,k,number) + amrex::Real(0.5) *
                            ((pred(i,j,k,number)-old_arr(i,j,k,number)) + dt*rhs_number);
                    }
                }
            }
        });
    }
    output.FillBoundary(geometry.periodicity());

    // The former full-layout grouped limiter is retained only in repository
    // history; it must never be compiled into the donor path.
#if 0
    if (high_flux) {
        // The production path uses the exact flattened semantic group
        // definitions.  Pass one accumulates a complete cell-wide adverse
        // budget for every atomic constraint; pass two applies one common
        // lambda to every member of the affected population/bin group.
        const Real anelastic_weight =
            (context.method == ::erf_auxiliary::IntegrationMethod::AnelasticHeun && context.stage_index > 0)
            ? Real(0.5) : Real(1.0);
        const bool heun_corrector =
            context.method == ::erf_auxiliary::IntegrationMethod::AnelasticHeun && context.stage_index > 0;
        const auto groups = make_constraint_groups(layout);
        const auto descriptors = make_constraint_descriptors(layout);
        const int nconstraints = static_cast<int>(descriptors.size());
        amrex::Gpu::ManagedVector<ConstraintDescriptor> device_descriptors;
        for (const auto& descriptor : descriptors) device_descriptors.push_back(descriptor);
        const auto* descriptor_data = device_descriptors.data();

        amrex::Gpu::ManagedVector<ProductionProperty> properties;
        for (std::size_t p = 0; p < layout.attached_properties().size(); ++p) {
            const auto& property = layout.attached_properties()[p];
            if (property.carrier_population != population.population_id) continue;
            ProductionProperty meta;
            meta.component = layout.property_offset(static_cast<int>(p));
            meta.kind = static_cast<int>(property.kind);
            meta.has_upper = std::isfinite(property.support_max) ? 1 : 0;
            meta.support_min = property.support_min;
            meta.support_max = property.support_max;
            properties.push_back(meta);
        }
        const ProductionProperty* property_data = properties.data();
        const int nproperties = static_cast<int>(properties.size());
        const int ncomp = layout.ncomp();

        auto accumulate_budget = [&](const ::erf_auxiliary::AuxiliaryFaceTransfer& transfer,
                                     amrex::MultiFab& budget) {
            budget.setVal(Real(0.0));
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                const auto& face = transfer.direction(dir);
                const Real scale = dt * anelastic_weight * geometry.InvCellSize(dir);
                for (amrex::MFIter mfi(face); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();
                    const auto flux = face.const_array(mfi);
                    const auto cell_budget = budget.array(mfi);
                    for (int descriptor_index = 0; descriptor_index < nconstraints; ++descriptor_index) {
                        const auto descriptor = descriptor_data[descriptor_index];
                        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            const int li = dir == 0 ? i-1 : i;
                            const int lj = dir == 1 ? j-1 : j;
                            const int lk = dir == 2 ? k-1 : k;
                            const Real form = descriptor_face_form(descriptor, flux, i, j, k);
                            const Real left_demand = scale * form;
                            const Real right_demand = -scale * form;
                            if (left_demand > Real(0.0)) {
                                amrex::Gpu::Atomic::Add(&cell_budget(li,lj,lk,descriptor_index), left_demand);
                            }
                            if (right_demand > Real(0.0)) {
                                amrex::Gpu::Atomic::Add(&cell_budget(i,j,k,descriptor_index), right_demand);
                            }
                        });
                    }
                }
            }
            // Face-centered FABs may share physical boundary faces.  Collapse
            // those contributions to the owning cell before any limiter reads
            // them, then publish the complete cell budget into one-cell ghosts.
            budget.SumBoundary(geometry.periodicity(), true);
            budget.FillBoundary(geometry.periodicity());
        };

        amrex::MultiFab advection_budget(output.boxArray(), output.DistributionMap(), nconstraints, 1);
        amrex::MultiFab diffusion_budget(output.boxArray(), output.DistributionMap(), nconstraints, 1);
        accumulate_budget(*low_advection_flux, advection_budget);
        accumulate_budget(*low_diffusion_flux, diffusion_budget);

        // Reject a low-order state that is inadmissible under the combined
        // advection-plus-diffusion demand.  This diagnostic is intentionally
        // before the high-order correction: FCT is not allowed to hide a
        // violated explicit low-order CFL/admissibility contract.
        // The admissibility demand is compared with the state that existed
        // immediately before this stage's low-order transfer.  `output` is
        // already the post-low-order state here and therefore is not a valid
        // baseline for this diagnostic.  In particular, the Heun corrector
        // uses 1/2*(old+predictor), matching the recurrence below.
        const auto old_baseline_arrays = old.const_arrays();
        const auto predictor_baseline_arrays = predictor.const_arrays();
        const auto advection_arrays = advection_budget.const_arrays();
        const auto diffusion_arrays = diffusion_budget.const_arrays();
        const amrex::Box domain = geometry.Domain();
        const int nx = domain.length(0);
        const int ny = domain.length(1);
        const int nz = domain.length(2);
        const auto low_summary = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpMax, amrex::ReduceOpMax>{},
            amrex::TypeList<Real, Real>{}, output, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                -> amrex::GpuTuple<Real, Real> {
                Real max_excess = Real(0.0);
                Real cell_key = Real(0.0);
                for (int d = 0; d < nconstraints; ++d) {
                    const auto descriptor = descriptor_data[d];
                    const Real value = descriptor_stage_baseline_form(
                        descriptor, old_baseline_arrays[box_no], predictor_baseline_arrays[box_no],
                        i, j, k, heun_corrector);
                    const Real scale = descriptor_stage_baseline_scale(
                        descriptor, old_baseline_arrays[box_no], predictor_baseline_arrays[box_no],
                        i, j, k, heun_corrector);
                    const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                    const Real excess = advection_arrays[box_no](i,j,k,d) +
                        diffusion_arrays[box_no](i,j,k,d) - value - tolerance;
                    if (excess > max_excess) {
                        max_excess = excess;
                        cell_key = static_cast<Real>(i-domain.smallEnd(0)) +
                            static_cast<Real>(nx) * (static_cast<Real>(j-domain.smallEnd(1)) +
                            static_cast<Real>(ny) * static_cast<Real>(k-domain.smallEnd(2)));
                    }
                }
                return {max_excess, cell_key};
            });
        Real max_excess = amrex::get<0>(low_summary);
        Real cell_key = amrex::get<1>(low_summary);
        amrex::ParallelDescriptor::ReduceRealMax(max_excess);
        amrex::ParallelDescriptor::ReduceRealMax(cell_key);
        if (max_excess > Real(0.0)) {
            int failed_descriptor = -1;
            Real available_margin = Real(0.0), advection_demand = Real(0.0), diffusion_demand = Real(0.0);
            for (int d = 0; d < nconstraints; ++d) {
                const auto descriptor = descriptor_data[d];
                const auto detail = amrex::ParReduce(
                    amrex::TypeList<amrex::ReduceOpMax, amrex::ReduceOpMin,
                                    amrex::ReduceOpMax, amrex::ReduceOpMax>{},
                    amrex::TypeList<Real, Real, Real, Real>{}, output, amrex::IntVect(0),
                    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                        -> amrex::GpuTuple<Real, Real, Real, Real> {
                        const Real value = descriptor_stage_baseline_form(
                            descriptor, old_baseline_arrays[box_no], predictor_baseline_arrays[box_no],
                            i, j, k, heun_corrector);
                        const Real scale = descriptor_stage_baseline_scale(
                            descriptor, old_baseline_arrays[box_no], predictor_baseline_arrays[box_no],
                            i, j, k, heun_corrector);
                        const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                        const Real combined = advection_arrays[box_no](i,j,k,d) + diffusion_arrays[box_no](i,j,k,d);
                        return {amrex::max(Real(0.0), combined - value - tolerance), value,
                                advection_arrays[box_no](i,j,k,d), diffusion_arrays[box_no](i,j,k,d)};
                    });
                Real violation = amrex::get<0>(detail);
                Real margin = amrex::get<1>(detail);
                Real advection = amrex::get<2>(detail);
                Real diffusion = amrex::get<3>(detail);
                amrex::ParallelDescriptor::ReduceRealMax(violation);
                amrex::ParallelDescriptor::ReduceRealMin(margin);
                amrex::ParallelDescriptor::ReduceRealMax(advection);
                amrex::ParallelDescriptor::ReduceRealMax(diffusion);
                if (failed_descriptor < 0 && violation > Real(0.0)) {
                    failed_descriptor = d;
                    available_margin = margin;
                    advection_demand = advection;
                    diffusion_demand = diffusion;
                }
            }
            const int cell_nx = nx > 0 ? nx : 1;
            const int cell_ny = ny > 0 ? ny : 1;
            const long long packed = static_cast<long long>(cell_key + Real(0.5));
            const int ck = static_cast<int>(packed / (static_cast<long long>(cell_nx) * cell_ny));
            const int cj = static_cast<int>((packed / cell_nx) % cell_ny);
            const int ci = static_cast<int>(packed % cell_nx);
            const auto& descriptor = descriptors[static_cast<std::size_t>(amrex::max(0, failed_descriptor))];
            const auto& group = groups[static_cast<std::size_t>(descriptor.group_index)];
            const auto& constraint = group.constraints[static_cast<std::size_t>(descriptor.constraint_index)];
            std::ostringstream message;
            message << "SBM combined low-order advection+diffusion admissibility failure: level=" << level
                    << " cell=(" << (domain.smallEnd(0)+ci) << "," << (domain.smallEnd(1)+cj)
                    << "," << (domain.smallEnd(2)+ck) << ") group=" << group.semantic_id
                    << " constraint=" << constraint.semantic_id
                    << " available_margin=" << available_margin
                    << " advection_demand=" << advection_demand
                    << " diffusion_demand=" << diffusion_demand
                    << " combined_demand=" << (advection_demand + diffusion_demand)
                    << " dt=" << dt
                    << " diffusion_admissible_dt=" << (diffusion_coefficient > Real(0.0) ?
                        Real(0.5)/(diffusion_coefficient*(dxi*dxi + dyi*dyi + dzi*dzi)) :
                        std::numeric_limits<Real>::infinity());
            throw std::domain_error(message.str());
        }

        // The low-order state is itself part of the invariant-domain
        // contract.  Validate it after the combined demand diagnostic (so a
        // timestep violation receives the more specific advection/diffusion
        // report) and before any antidiffusive correction is considered.
        validate_admissible_state(manager, layout, level);

        amrex::MultiFab constraint_budget(output.boxArray(), output.DistributionMap(), nconstraints, 1);
        // Accumulate the high-minus-low antidiffusive demand with the same
        // cell/constraint ownership rule used for the low-order diagnostic.
        constraint_budget.setVal(Real(0.0));
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto& low = stage_flux.direction(dir);
            const auto& candidate = high_flux->direction(dir);
            const Real scale = dt * anelastic_weight * geometry.InvCellSize(dir);
            for (amrex::MFIter mfi(low); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto low_arr = low.const_array(mfi);
                const auto high_arr = candidate.const_array(mfi);
                const auto budget = constraint_budget.array(mfi);
                for (int d = 0; d < nconstraints; ++d) {
                    const auto descriptor = descriptor_data[d];
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int li = dir == 0 ? i-1 : i;
                        const int lj = dir == 1 ? j-1 : j;
                        const int lk = dir == 2 ? k-1 : k;
                        const Real form = descriptor_face_form(descriptor, high_arr, i, j, k) -
                            descriptor_face_form(descriptor, low_arr, i, j, k);
                        const Real left_demand = scale * form;
                        const Real right_demand = -scale * form;
                        if (left_demand > Real(0.0)) {
                            amrex::Gpu::Atomic::Add(&budget(li,lj,lk,d), left_demand);
                        }
                        if (right_demand > Real(0.0)) {
                            amrex::Gpu::Atomic::Add(&budget(i,j,k,d), right_demand);
                        }
                    });
                }
            }
        }
        constraint_budget.SumBoundary(geometry.periodicity(), true);
        constraint_budget.FillBoundary(geometry.periodicity());

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            auto& low = stage_flux.direction(dir);
            const auto& candidate = high_flux->direction(dir);
            const Real inverse_length = geometry.InvCellSize(dir);
            for (amrex::MFIter mfi(low); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto low_arr = low.const_array(mfi);
                const auto high_arr = candidate.const_array(mfi);
                const auto state = output.const_array(mfi);
                const auto budget = constraint_budget.const_array(mfi);
                const auto result = high_flux->direction(dir).array(mfi);
                const Real scale = dt * anelastic_weight * inverse_length;
                for (int b = 0; b < nbins; ++b) {
                    const int mass = first + b;
                    const int number = two_moment ? population.number_offset + b : -1;
                    const int group_index = b;
                    int first_descriptor = 0;
                    for (int d = 0; d < nconstraints; ++d) {
                        if (descriptor_data[d].group_index == group_index) { first_descriptor = d; break; }
                    }
                    int last_descriptor = first_descriptor;
                    while (last_descriptor < nconstraints && descriptor_data[last_descriptor].group_index == group_index) ++last_descriptor;
                    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int li = dir == 0 ? i-1 : i;
                        const int lj = dir == 1 ? j-1 : j;
                        const int lk = dir == 2 ? k-1 : k;
                        Real lambda = Real(1.0);
                        for (int d = first_descriptor; d < last_descriptor; ++d) {
                            const auto descriptor = descriptor_data[d];
                            const Real delta_form = descriptor_face_form(descriptor, high_arr, i, j, k) -
                                descriptor_face_form(descriptor, low_arr, i, j, k);
                            const Real left_demand = scale * delta_form;
                            const Real right_demand = -scale * delta_form;
                            if (left_demand > Real(0.0) && budget(li,lj,lk,d) > Real(0.0)) {
                                lambda = amrex::min(lambda,
                                    descriptor_form(descriptor, state, li, lj, lk) /
                                    budget(li,lj,lk,d));
                            }
                            if (right_demand > Real(0.0) && budget(i,j,k,d) > Real(0.0)) {
                                lambda = amrex::min(lambda, descriptor_form(descriptor, state, i, j, k) / budget(i,j,k,d));
                            }
                        }
                        lambda = amrex::max(Real(0.0), amrex::min(Real(1.0), lambda));
                        const Real delta_mass = high_arr(i,j,k,mass) - low_arr(i,j,k,mass);
                        result(i,j,k,mass) = low_arr(i,j,k,mass) + lambda*delta_mass;
                        if (two_moment) {
                            const Real delta_number = high_arr(i,j,k,number) - low_arr(i,j,k,number);
                            result(i,j,k,number) = low_arr(i,j,k,number) + lambda*delta_number;
                        }
                        for (int p = 0; p < nproperties; ++p) {
                            const int component = property_data[p].component + b;
                            result(i,j,k,component) = low_arr(i,j,k,component) +
                                lambda*(high_arr(i,j,k,component)-low_arr(i,j,k,component));
                        }
                    });
                }
            }
        }
        // The accepted candidate replaces the high-order candidate in the
        // divergence and in the ledger.  Its correction is scaled by the same
        // stage recurrence coefficient as the low-order update.
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const auto& accepted = high_flux->direction(dir);
            const auto& low = stage_flux.direction(dir);
            const Real inverse_length = geometry.InvCellSize(dir);
            const int ncomp_local = layout.ncomp();
            for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto out = output.array(mfi);
                const auto corr = accepted.const_array(mfi);
                const auto original = low.const_array(mfi);
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    for (int c = 0; c < ncomp_local; ++c) {
                        const Real delta = corr(i+ (dir==0), j+(dir==1), k+(dir==2), c) - original(i+ (dir==0), j+(dir==1), k+(dir==2), c);
                        const Real delta_lo = corr(i,j,k,c) - original(i,j,k,c);
                        out(i,j,k,c) -= dt * anelastic_weight * inverse_length * (delta - delta_lo);
                    }
                });
            }
        }
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::MultiFab::Copy(stage_flux.direction(dir), high_flux->direction(dir),
                                  0, 0, ncomp, 0);
        }
        amrex::Gpu::synchronize();
    }
#endif

    if (two_moment) {
        for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto source = output.const_array(mfi);
            const auto scratch = transport_scratch.array(mfi);
            for (int b = 0; b < nbins; ++b) {
                const int mass = first + b;
                const int number = population.number_offset + b;
                const Real lower = population.grid.edges()[static_cast<std::size_t>(b)];
                const Real upper = population.grid.edges()[static_cast<std::size_t>(b+1)];
                const Real denominator = upper - lower;
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    const Real M = source(i,j,k,mass);
                    const Real C = source(i,j,k,number);
                    const Real scale = amrex::Math::abs(M) + upper*amrex::Math::abs(C) + lower*amrex::Math::abs(C);
                    const Real tolerance = Real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
                    Real L = (upper*C - M) / denominator;
                    Real H = (M - lower*C) / denominator;
                    if (L < Real(0.0) && L >= -tolerance) L = Real(0.0);
                    if (H < Real(0.0) && H >= -tolerance) H = Real(0.0);
                    if (C < Real(0.0) || L < Real(0.0) || H < Real(0.0)) { L = -Real(1.0); H = -Real(1.0); }
                    scratch(i,j,k,mass) = L;
                    scratch(i,j,k,number) = H;
                });
            }
        }
        validate_nonnegative_state(transport_scratch, layout.ncomp(), "SBM two-moment endpoint state");
    }

    // The compact host fields are projections of the updated authoritative
    // spectral state.  This is deliberately separate from face-flux
    // construction: qc/qr never get their own numerical transport path.
    const SBMBulkProjection bulk_projection(layout);
    for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
        bulk_projection.apply_to_core(mfi.validbox(), output.const_array(mfi),
                                      core_state.array(mfi));
    }

    validate_nonnegative_state(output, layout.ncomp());
    manager.accept_stage(level, context.output_time);
    manager.record_stage_face_transfer(level, context, stage_flux);
    core_state.FillBoundary(geometry.periodicity());
}

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
                   const TransportMethod method,
                   const int level,
                   const amrex::Real diffusion_coefficient,
                   const int chunk_size,
                   amrex::Real* minimum_accepted_limiter,
                   const TransportBoundaryPolicy& boundary_policy)
{
    // The legacy entry point did not distinguish the density used by the
    // ERF boundary state.  Preserve its behavior while making the target
    // density explicit in the production overload.
    advance_stage(manager, layout, context, rho_anchor, rho_input, rho_input,
                  core_state, carrier_x, carrier_y, carrier_z, geometry,
                  stage_flux, method, level, diffusion_coefficient, chunk_size,
                  minimum_accepted_limiter, boundary_policy);
}

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
                   const TransportMethod method,
                   const int level,
                   const amrex::Real diffusion_coefficient,
                   const int chunk_size,
                   amrex::Real* minimum_accepted_limiter,
                   const TransportBoundaryPolicy& boundary_policy)
{
    advance_stage(manager, layout, context, rho_evaluation, rho_evaluation, rho_evaluation,
                  core_state, carrier_x, carrier_y, carrier_z, geometry,
                  stage_flux, method, level, diffusion_coefficient, chunk_size,
                  minimum_accepted_limiter, boundary_policy);
}

std::size_t grouped_fct_peak_working_bytes(
    const SBMLayout& layout, const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm, const int chunk_size)
{
    (void) dm;
    if (chunk_size <= 0) {
        throw std::invalid_argument("SBM working-memory estimate requires a positive chunk size");
    }
    const auto chunks = make_constraint_closure_chunks(layout, chunk_size);
    const std::size_t real_bytes = sizeof(amrex::Real);
    std::size_t peak = 0;
    for (const auto& chunk : chunks) {
        const std::size_t nlocal = chunk.components.size();
        std::size_t grown_cell_points = 0;
        std::size_t budget_cell_points = 0;
        std::size_t face_points = 0;
        for (int index = 0; index < ba.size(); ++index) {
            grown_cell_points += static_cast<std::size_t>(
                amrex::grow(ba[index], amrex::IntVect(2)).numPts());
            budget_cell_points += static_cast<std::size_t>(
                amrex::grow(ba[index], amrex::IntVect(1)).numPts());
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                face_points += static_cast<std::size_t>(
                    amrex::convert(ba[index], amrex::IntVect::TheDimensionVector(dir)).numPts());
            }
        }
        std::size_t nconstraints = 0;
        const auto groups = make_constraint_groups(layout);
        for (const int group_index : chunk.group_indices) {
            nconstraints += groups[static_cast<std::size_t>(group_index)].constraints.size();
        }
        // ratio + low-advection + low-diffusion + high candidate + one
        // constraint budget.  This is an allocation-bound upper estimate;
        // it excludes allocator metadata and the persistent accepted ledger.
        const std::size_t bytes = real_bytes *
            (grown_cell_points * nlocal +
             face_points * (3*nlocal) +
             budget_cell_points * nconstraints);
        peak = std::max(peak, bytes);
    }
    return peak;
}

} // namespace erf_sbm
