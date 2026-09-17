#include "ERF_AuxiliaryStateManager.H"

#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace erf_auxiliary {

namespace {

void validate_carrier_density(const amrex::MultiFab& density,
                              const amrex::IntVect& nghost,
                              const char* context)
{
    if (density.nComp() < 1) {
        throw std::invalid_argument(std::string(context) + " has no carrier component");
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
                                " must be finite and strictly positive wherever transferred");
    }
}

std::unique_ptr<amrex::MultiFab> temporal_view(
    const amrex::MultiFab& old_state, const amrex::MultiFab& output_state,
    const double old_time, const double output_time, const double time,
    const amrex::Geometry& geometry, const int ncomp)
{
    auto result = std::make_unique<amrex::MultiFab>(
        old_state.boxArray(), old_state.DistributionMap(), ncomp,
        old_state.nGrowVect());
    const double scale = 1.0 + std::max(std::abs(old_time), std::abs(output_time));
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * scale;
    const double denominator = output_time - old_time;
    if (std::abs(denominator) <= tolerance) {
        amrex::MultiFab::Copy(*result, output_state, 0, 0, ncomp, output_state.nGrowVect());
    } else {
        const amrex::Real theta = static_cast<amrex::Real>((time - old_time) / denominator);
        amrex::MultiFab::LinComb(*result, amrex::Real(1.0) - theta, old_state, 0,
                                 theta, output_state, 0, 0, ncomp, old_state.nGrowVect());
    }
    result->FillBoundary(geometry.periodicity());
    return result;
}

void carrier_weighted_fill(
    amrex::MultiFab& target, const amrex::MultiFab& coarse_ratio,
    const amrex::Geometry& coarse_geometry, const amrex::Geometry& fine_geometry,
    const amrex::IntVect& ref_ratio, const amrex::MultiFab& fine_density,
    const amrex::Periodicity& fine_periodicity, const int ncomp,
    const bool fill_valid_cells)
{
    bool insufficient_ghosts = false;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        insufficient_ghosts = insufficient_ghosts ||
            fine_density.nGrowVect()[dir] < target.nGrowVect()[dir];
    }
    if (fine_density.boxArray() != target.boxArray() ||
        fine_density.DistributionMap() != target.DistributionMap() || insufficient_ghosts) {
        throw std::invalid_argument(
            "carrier-weighted auxiliary transfer requires a target-aligned density with sufficient ghosts");
    }
    validate_carrier_density(fine_density, target.nGrowVect(), "fine carrier density");
    amrex::MultiFab fine_ratio(target.boxArray(), target.DistributionMap(), ncomp,
                               target.nGrowVect());
    fine_ratio.setVal(amrex::Real(0.0));
    amrex::Vector<amrex::BCRec> bcs(static_cast<std::size_t>(ncomp));
    amrex::InterpFromCoarseLevel(fine_ratio, fine_ratio.nGrowVect(), amrex::IntVect(0),
                                 coarse_ratio, 0, 0, ncomp, coarse_geometry,
                                 fine_geometry, ref_ratio, &amrex::pc_interp, bcs, 0);
    fine_ratio.FillBoundary(fine_periodicity);
    for (amrex::MFIter mfi(target); mfi.isValid(); ++mfi) {
        const amrex::Box valid_box = mfi.validbox();
        const amrex::Box box = fill_valid_cells
            ? mfi.fabbox()
            : amrex::grow(valid_box, target.nGrowVect());
        const auto ratio = fine_ratio.const_array(mfi);
        const auto rho = fine_density.const_array(mfi);
        const auto out = target.array(mfi);
        amrex::ParallelFor(box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                if (!fill_valid_cells && valid_box.contains(i, j, k)) return;
                out(i,j,k,comp) = ratio(i,j,k,comp) * rho(i,j,k);
            });
    }
    // A stage FillPatch updates only the target ghost region.  A subsequent
    // same-level FillBoundary would copy the still-authoritative fine valid
    // cells back over those newly prepared coarse/fine interface values.
    if (fill_valid_cells) target.FillBoundary(fine_periodicity);
}

std::unique_ptr<amrex::MultiFab> make_carrier_ratio(
    const amrex::MultiFab& extensive, const amrex::MultiFab& density,
    const amrex::Periodicity& periodicity, const int ncomp)
{
    bool insufficient_ghosts = false;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        insufficient_ghosts = insufficient_ghosts ||
            density.nGrowVect()[dir] < extensive.nGrowVect()[dir];
    }
    if (density.boxArray() != extensive.boxArray() ||
        density.DistributionMap() != extensive.DistributionMap() || insufficient_ghosts) {
        throw std::invalid_argument(
            "carrier-weighted auxiliary transfer requires an extensive-state-aligned density");
    }
    validate_carrier_density(density, extensive.nGrowVect(), "coarse carrier density");
    auto ratio = std::make_unique<amrex::MultiFab>(
        extensive.boxArray(), extensive.DistributionMap(), ncomp,
        extensive.nGrowVect());
    ratio->setVal(amrex::Real(0.0));
    for (amrex::MFIter mfi(*ratio); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.fabbox();
        const auto source = extensive.const_array(mfi);
        const auto rho = density.const_array(mfi);
        const auto out = ratio->array(mfi);
        amrex::ParallelFor(box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const amrex::Real carrier = rho(i,j,k);
                out(i,j,k,comp) = carrier > amrex::Real(0.0) ?
                    source(i,j,k,comp) / carrier :
                    std::numeric_limits<amrex::Real>::quiet_NaN();
            });
    }
    ratio->FillBoundary(periodicity);
    return ratio;
}

} // namespace

void AuxiliaryStateManager::define_level(const int level,
                                         const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm,
                                         const int ngrow,
                                         const int scratch_ncomp)
{
    if (level < 0 || m_layout.ncomp() <= 0 || ngrow < 0) {
        throw std::invalid_argument("invalid auxiliary state definition");
    }
    const auto n = static_cast<std::size_t>(level + 1);
    m_old.resize(n);
    m_evaluation.resize(n);
    m_output.resize(n);
    m_scratch.resize(n);
    m_face_ledgers.resize(n);
    m_old_time.resize(n, 0.0);
    m_evaluation_time.resize(n, 0.0);
    m_output_time.resize(n, 0.0);

    m_old[static_cast<std::size_t>(level)] = std::make_unique<amrex::MultiFab>(ba, dm, m_layout.ncomp(), ngrow);
    m_evaluation[static_cast<std::size_t>(level)] = std::make_unique<amrex::MultiFab>(ba, dm, m_layout.ncomp(), ngrow);
    m_output[static_cast<std::size_t>(level)] = std::make_unique<amrex::MultiFab>(ba, dm, m_layout.ncomp(), ngrow);
    const int scratch_components = scratch_ncomp < 0 ? m_layout.ncomp() : scratch_ncomp;
    if (scratch_components > 0) {
        m_scratch[static_cast<std::size_t>(level)] =
            std::make_unique<amrex::MultiFab>(ba, dm, scratch_components, ngrow);
    } else {
        m_scratch[static_cast<std::size_t>(level)].reset();
    }
    m_old[static_cast<std::size_t>(level)]->setVal(0.0);
    m_evaluation[static_cast<std::size_t>(level)]->setVal(0.0);
    m_output[static_cast<std::size_t>(level)]->setVal(0.0);
    if (m_scratch[static_cast<std::size_t>(level)]) {
        m_scratch[static_cast<std::size_t>(level)]->setVal(0.0);
    }
    m_face_ledgers[static_cast<std::size_t>(level)] = std::make_unique<AuxiliaryFaceTransferLedger>();
    m_face_ledgers[static_cast<std::size_t>(level)]->define(ba, dm, m_layout.ncomp(), 3, 0);

    recompute_resident_bytes();
}

void AuxiliaryStateManager::remake_level(const int level, const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm,
                                         const int ngrow, const amrex::Periodicity& periodicity,
                                         const int scratch_ncomp)
{
    if (level < 0 || m_layout.ncomp() <= 0 || ngrow < 0) {
        throw std::invalid_argument("invalid auxiliary state remake");
    }
    const bool had_level = has_level(level);
    const auto n = static_cast<std::size_t>(level + 1);
    m_old.resize(n); m_evaluation.resize(n); m_output.resize(n);
    m_scratch.resize(n); m_face_ledgers.resize(n);
    m_old_time.resize(n, 0.0); m_evaluation_time.resize(n, 0.0); m_output_time.resize(n, 0.0);
    const double saved_old_time = had_level ? m_old_time[static_cast<std::size_t>(level)] : 0.0;
    const double saved_evaluation_time = had_level ? m_evaluation_time[static_cast<std::size_t>(level)] : 0.0;
    const double saved_output_time = had_level ? m_output_time[static_cast<std::size_t>(level)] : 0.0;
    auto new_old = std::make_unique<amrex::MultiFab>(ba, dm, m_layout.ncomp(), ngrow);
    auto new_evaluation = std::make_unique<amrex::MultiFab>(ba, dm, m_layout.ncomp(), ngrow);
    auto new_output = std::make_unique<amrex::MultiFab>(ba, dm, m_layout.ncomp(), ngrow);
    const int scratch_components = scratch_ncomp < 0 ? m_layout.ncomp() : scratch_ncomp;
    std::unique_ptr<amrex::MultiFab> new_scratch;
    if (scratch_components > 0) {
        new_scratch = std::make_unique<amrex::MultiFab>(ba, dm, scratch_components, ngrow);
    }
    new_old->setVal(0.0); new_evaluation->setVal(0.0); new_output->setVal(0.0);
    if (new_scratch) new_scratch->setVal(0.0);
    if (had_level) {
        new_old->ParallelCopy(*m_old[static_cast<std::size_t>(level)], 0, 0, m_layout.ncomp(),
                              amrex::IntVect(0), amrex::IntVect(0),
                              amrex::Periodicity::NonPeriodic());
        new_evaluation->ParallelCopy(*m_evaluation[static_cast<std::size_t>(level)], 0, 0, m_layout.ncomp(),
                                     amrex::IntVect(0), amrex::IntVect(0),
                                     amrex::Periodicity::NonPeriodic());
        new_output->ParallelCopy(*m_output[static_cast<std::size_t>(level)], 0, 0, m_layout.ncomp(),
                                 amrex::IntVect(0), amrex::IntVect(0),
                                 amrex::Periodicity::NonPeriodic());
        new_old->FillBoundary(periodicity); new_evaluation->FillBoundary(periodicity); new_output->FillBoundary(periodicity);
    }
    // ERF invokes regrid/remake at a coarse-step boundary, before the next
    // level advance begins.  The pre-remake old/evaluation views therefore
    // belong to the completed step and must not be carried as temporal
    // baselines into the next step.  Use the accepted remade output for all
    // three views; this preserves accepted overlap while giving the next
    // begin_step a coherent authoritative state.
    amrex::MultiFab::Copy(*new_old, *new_output, 0, 0, m_layout.ncomp(), new_output->nGrowVect());
    amrex::MultiFab::Copy(*new_evaluation, *new_output, 0, 0, m_layout.ncomp(), new_output->nGrowVect());
    new_old->FillBoundary(periodicity);
    new_evaluation->FillBoundary(periodicity);
    m_old[static_cast<std::size_t>(level)] = std::move(new_old);
    m_evaluation[static_cast<std::size_t>(level)] = std::move(new_evaluation);
    m_output[static_cast<std::size_t>(level)] = std::move(new_output);
    m_scratch[static_cast<std::size_t>(level)] = std::move(new_scratch);
    m_face_ledgers[static_cast<std::size_t>(level)] = std::make_unique<AuxiliaryFaceTransferLedger>();
    m_face_ledgers[static_cast<std::size_t>(level)]->define(ba, dm, m_layout.ncomp(), 3, 0);
    m_old_time[static_cast<std::size_t>(level)] = saved_old_time;
    m_evaluation_time[static_cast<std::size_t>(level)] = saved_evaluation_time;
    m_output_time[static_cast<std::size_t>(level)] = saved_output_time;
    recompute_resident_bytes();
}

void AuxiliaryStateManager::remake_level_from_coarse(
    const int level, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
    const int ngrow, const amrex::Periodicity& periodicity, const int coarse_level,
    const amrex::Geometry& coarse_geometry, const amrex::Geometry& fine_geometry,
    const amrex::IntVect& ref_ratio, const double time, const int scratch_ncomp,
    const amrex::MultiFab* coarse_rho_old,
    const amrex::MultiFab* coarse_rho_output,
    const amrex::MultiFab* fine_rho_target)
{
    if (!has_level(coarse_level) || level <= coarse_level || ref_ratio.min() <= 0 ||
        !coarse_geometry.isAllPeriodic() || !fine_geometry.isAllPeriodic()) {
        throw std::invalid_argument("invalid auxiliary coarse-filled remake contract");
    }
    const double coarse_old_time = old_time(coarse_level);
    const double coarse_new_time = output_time(coarse_level);
    const double time_scale = 1.0 + std::max(std::abs(coarse_old_time), std::abs(coarse_new_time));
    if (time < coarse_old_time - 128.0 * std::numeric_limits<double>::epsilon() * time_scale ||
        time > coarse_new_time + 128.0 * std::numeric_limits<double>::epsilon() * time_scale) {
        throw std::invalid_argument("auxiliary remake time is outside the authoritative coarse time bracket");
    }

    const bool had_level = has_level(level);
    const auto n = static_cast<std::size_t>(level + 1);
    m_old.resize(n); m_evaluation.resize(n); m_output.resize(n); m_scratch.resize(n);
    m_face_ledgers.resize(n); m_old_time.resize(n, 0.0); m_evaluation_time.resize(n, 0.0); m_output_time.resize(n, 0.0);
    const double saved_old_time = had_level ? m_old_time[static_cast<std::size_t>(level)] : time;
    const double saved_evaluation_time = had_level ? m_evaluation_time[static_cast<std::size_t>(level)] : time;
    const double saved_output_time = had_level ? m_output_time[static_cast<std::size_t>(level)] : time;

    auto make_state = [&]() {
        auto state = std::make_unique<amrex::MultiFab>(ba, dm, m_layout.ncomp(), ngrow);
        state->setVal(0.0);
        return state;
    };
    auto new_old = make_state();
    auto new_evaluation = make_state();
    auto new_output = make_state();
    const int scratch_components = scratch_ncomp < 0 ? m_layout.ncomp() : scratch_ncomp;
    std::unique_ptr<amrex::MultiFab> new_scratch;
    if (scratch_components > 0) {
        new_scratch = std::make_unique<amrex::MultiFab>(ba, dm, scratch_components, ngrow);
        new_scratch->setVal(0.0);
    }
    amrex::Vector<amrex::BCRec> bcs(static_cast<std::size_t>(m_layout.ncomp()));

    auto coarse_at_time = [&](const amrex::MultiFab& coarse_old,
                              const amrex::MultiFab& coarse_output) {
        auto interpolated = std::make_unique<amrex::MultiFab>(
            coarse_old.boxArray(), coarse_old.DistributionMap(), m_layout.ncomp(), coarse_old.nGrowVect());
        const double denominator = coarse_new_time - coarse_old_time;
        if (std::abs(denominator) <= 128.0 * std::numeric_limits<double>::epsilon() * time_scale) {
            amrex::MultiFab::Copy(*interpolated, coarse_output, 0, 0, m_layout.ncomp(), coarse_output.nGrowVect());
        } else {
            const amrex::Real theta = static_cast<amrex::Real>((time - coarse_old_time) / denominator);
            amrex::MultiFab::LinComb(*interpolated, amrex::Real(1.0) - theta, coarse_old,
                                     0, theta, coarse_output, 0, 0, m_layout.ncomp(), coarse_old.nGrowVect());
        }
        interpolated->FillBoundary(coarse_geometry.periodicity());
        return interpolated;
    };
    auto fill_from_coarse = [&](amrex::MultiFab& fine, const amrex::MultiFab& coarse) {
        if (coarse_rho_old == nullptr || coarse_rho_output == nullptr || fine_rho_target == nullptr) {
            amrex::InterpFromCoarseLevel(fine, fine.nGrowVect(), amrex::IntVect(0), coarse,
                                         0, 0, m_layout.ncomp(), coarse_geometry, fine_geometry,
                                         ref_ratio, &amrex::pc_interp, bcs, 0);
            fine.FillBoundary(periodicity);
            return;
        }
        const auto coarse_rho = temporal_view(*coarse_rho_old, *coarse_rho_output,
                                              coarse_old_time, coarse_new_time, time,
                                              coarse_geometry, 1);
        const auto coarse_ratio = make_carrier_ratio(coarse, *coarse_rho,
                                                     coarse_geometry.periodicity(),
                                                     m_layout.ncomp());
        carrier_weighted_fill(fine, *coarse_ratio, coarse_geometry, fine_geometry,
                              ref_ratio, *fine_rho_target, periodicity, m_layout.ncomp(), true);
    };

    // New coverage is created at one regrid time, so every newly introduced
    // semantic state is initialized from the same temporally interpolated
    // authoritative coarse spectrum.  Existing fine valid cells are overlaid
    // below, preserving their old/evaluation/output meanings.
    auto coarse_state = coarse_at_time(old(coarse_level), output(coarse_level));
    fill_from_coarse(*new_old, *coarse_state);
    fill_from_coarse(*new_evaluation, *coarse_state);
    fill_from_coarse(*new_output, *coarse_state);
    if (had_level) {
        new_old->ParallelCopy(*m_old[static_cast<std::size_t>(level)], 0, 0, m_layout.ncomp(),
                              amrex::IntVect(0), amrex::IntVect(0),
                              amrex::Periodicity::NonPeriodic());
        new_evaluation->ParallelCopy(*m_evaluation[static_cast<std::size_t>(level)], 0, 0, m_layout.ncomp(),
                                     amrex::IntVect(0), amrex::IntVect(0),
                                     amrex::Periodicity::NonPeriodic());
        new_output->ParallelCopy(*m_output[static_cast<std::size_t>(level)], 0, 0, m_layout.ncomp(),
                                 amrex::IntVect(0), amrex::IntVect(0),
                                 amrex::Periodicity::NonPeriodic());
        new_old->FillBoundary(periodicity); new_evaluation->FillBoundary(periodicity); new_output->FillBoundary(periodicity);
    }
    // A regrid is performed at a level-step boundary.  The old and
    // evaluation views from the completed step are not valid temporal views
    // for the next advance; the accepted remade output is the sole state to
    // carry into all three slots.
    amrex::MultiFab::Copy(*new_old, *new_output, 0, 0, m_layout.ncomp(), new_output->nGrowVect());
    amrex::MultiFab::Copy(*new_evaluation, *new_output, 0, 0, m_layout.ncomp(), new_output->nGrowVect());
    new_old->FillBoundary(periodicity);
    new_evaluation->FillBoundary(periodicity);
    m_old[static_cast<std::size_t>(level)] = std::move(new_old);
    m_evaluation[static_cast<std::size_t>(level)] = std::move(new_evaluation);
    m_output[static_cast<std::size_t>(level)] = std::move(new_output);
    m_scratch[static_cast<std::size_t>(level)] = std::move(new_scratch);
    m_face_ledgers[static_cast<std::size_t>(level)] = std::make_unique<AuxiliaryFaceTransferLedger>();
    m_face_ledgers[static_cast<std::size_t>(level)]->define(ba, dm, m_layout.ncomp(), 3, 0);
    m_old_time[static_cast<std::size_t>(level)] = had_level ? saved_old_time : time;
    m_evaluation_time[static_cast<std::size_t>(level)] = had_level ? saved_evaluation_time : time;
    m_output_time[static_cast<std::size_t>(level)] = had_level ? saved_output_time : time;
    recompute_resident_bytes();
}

void AuxiliaryStateManager::destroy_level(const int level)
{
    if (level < 0 || !has_level(level)) return;
    m_old[static_cast<std::size_t>(level)].reset();
    m_evaluation[static_cast<std::size_t>(level)].reset();
    m_output[static_cast<std::size_t>(level)].reset();
    m_scratch[static_cast<std::size_t>(level)].reset();
    m_face_ledgers[static_cast<std::size_t>(level)].reset();
    recompute_resident_bytes();
}

void AuxiliaryStateManager::average_down_to(const int coarse_level, const int fine_level,
                                            const amrex::IntVect& ref_ratio)
{
    if (!has_level(coarse_level) || !has_level(fine_level) || coarse_level >= fine_level) {
        throw std::invalid_argument("invalid auxiliary average-down levels");
    }
    for (auto* coarse : {m_old[static_cast<std::size_t>(coarse_level)].get(),
                         m_evaluation[static_cast<std::size_t>(coarse_level)].get(),
                         m_output[static_cast<std::size_t>(coarse_level)].get()}) {
        const amrex::MultiFab* fine = coarse == m_old[static_cast<std::size_t>(coarse_level)].get()
            ? m_old[static_cast<std::size_t>(fine_level)].get()
            : (coarse == m_evaluation[static_cast<std::size_t>(coarse_level)].get()
                ? m_evaluation[static_cast<std::size_t>(fine_level)].get()
                : m_output[static_cast<std::size_t>(fine_level)].get());
        amrex::average_down(*fine, *coarse, 0, m_layout.ncomp(), ref_ratio);
        coarse->FillBoundary(amrex::Periodicity::NonPeriodic());
    }
}

void AuxiliaryStateManager::prolong_from_coarse(const int coarse_level, const int fine_level,
                                                const amrex::Geometry& coarse_geometry,
                                                const amrex::Geometry& fine_geometry,
                                                const amrex::IntVect& ref_ratio,
                                                const double requested_time,
                                                const amrex::MultiFab* coarse_rho_old,
                                                const amrex::MultiFab* coarse_rho_output,
                                                const amrex::MultiFab* fine_rho_target)
{
    if (!has_level(coarse_level) || !has_level(fine_level) || coarse_level >= fine_level ||
        ref_ratio.min() <= 0) {
        throw std::invalid_argument("invalid auxiliary prolongation levels or refinement ratio");
    }
    // P2 supports only fully periodic static Cartesian geometry.  The
    // no-physical-BC overload is consequently the precise piecewise-constant
    // injection required here and does not invent a provider boundary rule.
    if (!coarse_geometry.isAllPeriodic() || !fine_geometry.isAllPeriodic()) {
        throw std::invalid_argument("SBM auxiliary prolongation requires periodic geometry");
    }
    const double coarse_old_time = old_time(coarse_level);
    const double coarse_new_time = output_time(coarse_level);
    const double time = requested_time >= 0.0 ? requested_time : coarse_new_time;
    const double time_scale = 1.0 + std::max(std::abs(coarse_old_time), std::abs(coarse_new_time));
    if (time < coarse_old_time - 128.0 * std::numeric_limits<double>::epsilon() * time_scale ||
        time > coarse_new_time + 128.0 * std::numeric_limits<double>::epsilon() * time_scale) {
        throw std::invalid_argument("auxiliary prolongation time is outside the authoritative coarse time bracket");
    }
    auto coarse_state = std::make_unique<amrex::MultiFab>(
        old(coarse_level).boxArray(), old(coarse_level).DistributionMap(), m_layout.ncomp(), old(coarse_level).nGrowVect());
    const double denominator = coarse_new_time - coarse_old_time;
    if (std::abs(denominator) <= 128.0 * std::numeric_limits<double>::epsilon() * time_scale) {
        amrex::MultiFab::Copy(*coarse_state, output(coarse_level), 0, 0, m_layout.ncomp(), output(coarse_level).nGrowVect());
    } else {
        const amrex::Real theta = static_cast<amrex::Real>((time - coarse_old_time) / denominator);
        amrex::MultiFab::LinComb(*coarse_state, amrex::Real(1.0) - theta, old(coarse_level),
                                 0, theta, output(coarse_level), 0, 0, m_layout.ncomp(), old(coarse_level).nGrowVect());
    }
    coarse_state->FillBoundary(coarse_geometry.periodicity());
    amrex::Vector<amrex::BCRec> bcs(static_cast<std::size_t>(m_layout.ncomp()));
    std::unique_ptr<amrex::MultiFab> coarse_ratio;
    if (coarse_rho_old != nullptr || coarse_rho_output != nullptr || fine_rho_target != nullptr) {
        if (coarse_rho_old == nullptr || coarse_rho_output == nullptr || fine_rho_target == nullptr) {
            throw std::invalid_argument("carrier-weighted auxiliary prolongation requires all density views");
        }
        const auto coarse_rho = temporal_view(*coarse_rho_old, *coarse_rho_output,
                                              coarse_old_time, coarse_new_time, time,
                                              coarse_geometry, 1);
        coarse_ratio = make_carrier_ratio(*coarse_state, *coarse_rho,
                                          coarse_geometry.periodicity(), m_layout.ncomp());
    }
    auto prolong = [this, &coarse_geometry, &fine_geometry, &ref_ratio, &bcs,
                    &coarse_state, &coarse_ratio, fine_rho_target](amrex::MultiFab& fine) {
        if (coarse_ratio) {
            carrier_weighted_fill(fine, *coarse_ratio, coarse_geometry, fine_geometry,
                                  ref_ratio, *fine_rho_target,
                                  fine_geometry.periodicity(), m_layout.ncomp(), true);
        } else {
            amrex::InterpFromCoarseLevel(fine, fine.nGrowVect(), amrex::IntVect(0), *coarse_state,
                                         0, 0, m_layout.ncomp(), coarse_geometry, fine_geometry,
                                         ref_ratio, &amrex::pc_interp, bcs, 0);
            fine.FillBoundary(fine_geometry.periodicity());
        }
    };
    prolong(output(fine_level));
    prolong(old(fine_level));
    prolong(evaluation(fine_level));
    if (m_scratch[static_cast<std::size_t>(fine_level)]) {
        scratch(fine_level).setVal(0.0);
        scratch(fine_level).FillBoundary(fine_geometry.periodicity());
    }
}

void AuxiliaryStateManager::fill_stage_from_coarse(
    const int coarse_level, const int fine_level, const double time,
    const amrex::Geometry& coarse_geometry, const amrex::Geometry& fine_geometry,
    const amrex::IntVect& ref_ratio)
{
    fill_stage_from_coarse(coarse_level, fine_level, time, coarse_geometry,
                           fine_geometry, ref_ratio, AuxiliaryTimeView::Evaluation);
}

void AuxiliaryStateManager::fill_stage_from_coarse(
    const int coarse_level, const int fine_level, const double time,
    const amrex::Geometry& coarse_geometry, const amrex::Geometry& fine_geometry,
    const amrex::IntVect& ref_ratio, const AuxiliaryTimeView target_view,
    const amrex::MultiFab* coarse_rho_old,
    const amrex::MultiFab* coarse_rho_output,
    const amrex::MultiFab* fine_rho_target)
{
    if (!has_level(coarse_level) || !has_level(fine_level) || coarse_level >= fine_level ||
        ref_ratio.min() <= 0 || !coarse_geometry.isAllPeriodic() || !fine_geometry.isAllPeriodic()) {
        throw std::invalid_argument("invalid auxiliary stage FillPatch levels or geometry");
    }
    const double coarse_old_time = old_time(coarse_level);
    const double coarse_new_time = output_time(coarse_level);
    const double time_scale = 1.0 + std::max(std::abs(coarse_old_time), std::abs(coarse_new_time));
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * time_scale;
    if (time < coarse_old_time - tolerance || time > coarse_new_time + tolerance) {
        throw std::invalid_argument("auxiliary stage time is outside the authoritative coarse time bracket");
    }
    old(coarse_level).FillBoundary(coarse_geometry.periodicity());
    output(coarse_level).FillBoundary(coarse_geometry.periodicity());
    auto& target = target_view == AuxiliaryTimeView::Old ? old(fine_level) : evaluation(fine_level);
    target.FillBoundary(fine_geometry.periodicity());
    const bool carrier_weighted = coarse_rho_old != nullptr || coarse_rho_output != nullptr ||
        fine_rho_target != nullptr;
    if (carrier_weighted) {
        if (coarse_rho_old == nullptr || coarse_rho_output == nullptr || fine_rho_target == nullptr) {
            throw std::invalid_argument("carrier-weighted auxiliary stage fill requires all density views");
        }
        const auto coarse_state = temporal_view(old(coarse_level), output(coarse_level),
                                                coarse_old_time, coarse_new_time, time,
                                                coarse_geometry, m_layout.ncomp());
        const auto coarse_rho = temporal_view(*coarse_rho_old, *coarse_rho_output,
                                              coarse_old_time, coarse_new_time, time,
                                              coarse_geometry, 1);
        const auto coarse_ratio = make_carrier_ratio(*coarse_state, *coarse_rho,
                                                     coarse_geometry.periodicity(), m_layout.ncomp());
        carrier_weighted_fill(target, *coarse_ratio, coarse_geometry, fine_geometry,
                              ref_ratio, *fine_rho_target,
                              fine_geometry.periodicity(), m_layout.ncomp(), false);
        // The coarse write intentionally prepares every uncovered ghost in the
        // grown target box.  Restore same-level and periodic fine authority
        // after that write; a true coarse/fine ghost has no same-level valid
        // owner and therefore remains the carrier-weighted interpolation.
        target.FillBoundary(fine_geometry.periodicity());
        return;
    }
    amrex::Vector<amrex::MultiFab*> coarse_states{&old(coarse_level), &output(coarse_level)};
    amrex::Vector<amrex::Real> coarse_times{static_cast<amrex::Real>(coarse_old_time),
                                            static_cast<amrex::Real>(coarse_new_time)};
    amrex::Vector<amrex::MultiFab*> fine_states{&target, &target};
    amrex::Vector<amrex::Real> fine_times{static_cast<amrex::Real>(time), static_cast<amrex::Real>(time)};
    amrex::Vector<amrex::BCRec> bcs(static_cast<std::size_t>(m_layout.ncomp()));
    amrex::FillPatchTwoLevels(target, target.nGrowVect(),
                              amrex::IntVect(0), static_cast<amrex::Real>(time),
                              coarse_states, coarse_times, fine_states, fine_times,
                              0, 0, m_layout.ncomp(), coarse_geometry, fine_geometry,
                              ref_ratio, &amrex::pc_interp, bcs, 0);
    target.FillBoundary(fine_geometry.periodicity());
}

void AuxiliaryStateManager::recompute_resident_bytes() noexcept
{
    m_state_resident_bytes = 0;
    m_face_transfer_resident_bytes = 0;
    for (std::size_t level = 0; level < m_output.size(); ++level) {
        if (m_old[level]) {
            m_state_resident_bytes += allocated_payload_bytes(*m_old[level]) +
                allocated_payload_bytes(*m_evaluation[level]) + allocated_payload_bytes(*m_output[level]) +
                (m_scratch[level] ? allocated_payload_bytes(*m_scratch[level]) : 0U);
        }
        if (m_face_ledgers[level]) m_face_transfer_resident_bytes += m_face_ledgers[level]->resident_bytes();
    }
    m_resident_bytes = m_state_resident_bytes + m_face_transfer_resident_bytes;
}

void AuxiliaryStateManager::begin_step(const int level, const double old_time_value,
                                       const bool reset_face_transfer)
{
    amrex::MultiFab::Copy(old(level), output(level), 0, 0, m_layout.ncomp(), output(level).nGrowVect());
    amrex::MultiFab::Copy(evaluation(level), output(level), 0, 0, m_layout.ncomp(), output(level).nGrowVect());
    face_transfer_ledger(level).begin_step(reset_face_transfer);
    m_old_time[static_cast<std::size_t>(level)] = old_time_value;
    m_evaluation_time[static_cast<std::size_t>(level)] = old_time_value;
    m_output_time[static_cast<std::size_t>(level)] = old_time_value;
}

void AuxiliaryStateManager::set_time_views(const int level, const double old_time_value,
                                           const double evaluation_time_value,
                                           const double output_time_value)
{
    if (level < 0 || !has_level(level)) {
        throw std::invalid_argument("cannot restore time views for an undefined auxiliary level");
    }
    m_old_time[static_cast<std::size_t>(level)] = old_time_value;
    m_evaluation_time[static_cast<std::size_t>(level)] = evaluation_time_value;
    m_output_time[static_cast<std::size_t>(level)] = output_time_value;
}

void AuxiliaryStateManager::accept_stage(const int level, const double stage_time)
{
    // Keep output as the accepted state and publish it as the evaluation
    // state for the next callback.  The old full-step baseline remains
    // untouched throughout all explicit stages.
    amrex::MultiFab::Copy(evaluation(level), output(level), 0, 0, m_layout.ncomp(), output(level).nGrowVect());
    m_evaluation_time[static_cast<std::size_t>(level)] = stage_time;
    m_output_time[static_cast<std::size_t>(level)] = stage_time;
}

void AuxiliaryStateManager::record_stage_face_transfer(const int level,
                                                       const StageContext& context,
                                                       const AuxiliaryFaceTransfer& stage_flux)
{
    face_transfer_ledger(level).record_stage(context, stage_flux);
}

} // namespace erf_auxiliary
