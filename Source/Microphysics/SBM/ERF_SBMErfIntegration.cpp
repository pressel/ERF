#include "ERF.H"

#include "ERF_SBMBulkProjection.H"
#include "ERF_SBMContracts.H"
#include "ERF_SBMTransferClosure.H"
#include "ERF_SBMTransportPrototype.H"
#include "ERF_SBMErfBoundary.H"

#include <AMReX_MFParallelFor.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

using namespace amrex;

namespace {

amrex::Real sbm_total_mass(const amrex::MultiFab& state, const int ncomp,
                           const amrex::Real cell_volume)
{
    amrex::Real total = amrex::Real(0.0);
    for (int comp = 0; comp < ncomp; ++comp) total += state.sum(comp);
    return cell_volume * total;
}

amrex::Real sbm_max_projection_error(const erf_sbm::SBMLayout& layout,
                                     const amrex::MultiFab& spectral,
                                     const amrex::MultiFab& core)
{
    const auto& projection = layout.liquid_projection();
    const auto liquid = std::find_if(layout.populations().begin(), layout.populations().end(),
        [&](const erf_sbm::PopulationLayout& population) {
            return population.population_id == projection.population_id;
        });
    const int first = liquid->mass_offset;
    const int split = projection.cloud_rain_split;
    const int nbins = liquid->grid.nbins();
    amrex::MultiFab error(core.boxArray(), core.DistributionMap(), 1, 0);
    error.setVal(amrex::Real(0.0));
    for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto spectrum = spectral.const_array(mfi);
        const auto compact = core.const_array(mfi);
        const auto result = error.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real qc = amrex::Real(0.0);
            amrex::Real qr = amrex::Real(0.0);
            for (int b = 0; b < split; ++b) qc += spectrum(i,j,k,first+b);
            for (int b = split; b < nbins; ++b) qr += spectrum(i,j,k,first+b);
            result(i,j,k) = amrex::max(
                amrex::Math::abs(compact(i,j,k,RhoQ2_comp) - qc),
                amrex::Math::abs(compact(i,j,k,RhoQ3_comp) - qr));
        });
    }
    return error.max(0);
}

amrex::Real sbm_max_change(const amrex::MultiFab& current,
                           const amrex::MultiFab& baseline,
                           const int ncomp)
{
    amrex::MultiFab difference(current.boxArray(), current.DistributionMap(), 1, 0);
    amrex::Real maximum = amrex::Real(0.0);
    for (int comp = 0; comp < ncomp; ++comp) {
        amrex::MultiFab::Copy(difference, current, comp, 0, 1, 0);
        amrex::MultiFab::Subtract(difference, baseline, comp, 0, 1, 0);
        maximum = amrex::max(maximum, difference.norm0(0));
    }
    return maximum;
}

amrex::Real sbm_max_manufactured_ratio_error(const erf_sbm::SBMLayout& layout,
                                              const amrex::MultiFab& spectral,
                                              const amrex::MultiFab& density)
{
    const auto& population = layout.populations().front();
    const int nbins = population.grid.nbins();
    amrex::Real maximum = amrex::Real(0.0);
    for (int comp = 0; comp < layout.ncomp(); ++comp) {
        amrex::Real expected_ratio = amrex::Real(0.0);
        if (comp >= population.mass_offset && comp < population.mass_offset + nbins) {
            expected_ratio = amrex::Real(1.0e-6) *
                amrex::Real(comp - population.mass_offset + 1);
        } else if (population.number_offset >= 0 &&
                   comp >= population.number_offset && comp < population.number_offset + nbins) {
            const int bin = comp - population.number_offset;
            expected_ratio = amrex::Real(1.0e-6) * amrex::Real(bin + 1) /
                population.grid.pivot(bin);
        } else {
            continue;
        }
        amrex::MultiFab error(spectral.boxArray(), spectral.DistributionMap(), 1, 0);
        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            const auto result = error.array(mfi);
            const auto state = spectral.const_array(mfi);
            const auto rho = density.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real cell_density = rho(i,j,k,0);
                result(i,j,k) = cell_density > amrex::Real(0.0) ?
                    amrex::Math::abs(state(i,j,k,comp) / cell_density - expected_ratio) :
                    amrex::Real(0.0);
            });
        }
        maximum = amrex::max(maximum, error.max(0));
    }
    return maximum;
}

std::size_t sbm_cell_bytes(const amrex::MultiFab& state)
{
    return ::erf_auxiliary::allocated_payload_bytes(state);
}

struct UniqueFaceTransferDiagnostics {
    std::vector<amrex::Real> sum;
    amrex::Real l1{0.0};
    amrex::Real maximum{0.0};
};

// A face-centered MultiFab made with convert(cell_ba, direction) contains
// both copies of an interface face when the cell BoxArray is split.  That is
// the correct storage for conservative cell updates, but a raw MultiFab
// norm/sum counts the shared face once per adjacent FAB and therefore changes
// with max_grid_size.  Diagnostics must describe unique physical faces, so
// assign each face to the FAB containing its canonical adjacent cell.  For a
// periodic direction the high endpoint is the same physical face as the low
// endpoint and is intentionally omitted.
UniqueFaceTransferDiagnostics unique_face_transfer_diagnostics(
    const amrex::MultiFab& face,
    const amrex::Geometry& geometry,
    const int dir,
    const int ncomp)
{
    UniqueFaceTransferDiagnostics result;
    result.sum.assign(static_cast<std::size_t>(ncomp), amrex::Real(0.0));
    amrex::MultiFab unique(face.boxArray(), face.DistributionMap(), ncomp, 0);
    unique.setVal(amrex::Real(0.0));
    const amrex::Box domain = geometry.Domain();
    const bool periodic = geometry.isPeriodic(dir);
    for (amrex::MFIter mfi(face); mfi.isValid(); ++mfi) {
        const amrex::Box face_box = mfi.validbox();
        const amrex::Box owner_box = amrex::enclosedCells(face_box);
        const auto source = face.const_array(mfi);
        const auto destination = unique.array(mfi);
        amrex::ParallelFor(face_box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const int coordinate = dir == 0 ? i : (dir == 1 ? j : k);
                if (periodic && coordinate == domain.bigEnd(dir) + 1) return;
                const int owner_coordinate = coordinate == domain.bigEnd(dir) + 1 ?
                    domain.bigEnd(dir) : coordinate;
                int owner_i = i;
                int owner_j = j;
                int owner_k = k;
                if (dir == 0) owner_i = owner_coordinate;
                if (dir == 1) owner_j = owner_coordinate;
                if (dir == 2) owner_k = owner_coordinate;
                if (owner_box.contains(owner_i, owner_j, owner_k)) {
                    destination(i,j,k,comp) = source(i,j,k,comp);
                }
            });
    }
    amrex::Gpu::synchronize();
    for (int comp = 0; comp < ncomp; ++comp) {
        result.l1 += unique.norm1(comp);
        result.maximum = amrex::max(result.maximum, unique.norm0(comp));
        result.sum[static_cast<std::size_t>(comp)] = unique.sum(comp);
    }
    return result;
}

void write_sbm_diagnostic(const std::string& path,
                          const bool anelastic,
                          const int nbins,
                          const int step_count,
                          const amrex::Real initial_mass,
                          const amrex::Real final_mass,
                          const amrex::Real initial_variation,
                          const amrex::Real transport_change,
                          const amrex::Real projection_error,
                          const amrex::Real face_projection_error,
                          const erf_sbm::AcceptedTransferClosure& closure,
                          const amrex::Real compact_mass,
                          const amrex::Real cell_volume,
                          const std::size_t cell_state_bytes,
                          const std::size_t face_transfer_bytes,
                          const std::size_t total_auxiliary_bytes)
{
    const amrex::Real mass_error = amrex::Math::abs(final_mass - initial_mass);
    const amrex::Real compact_mass_error = amrex::Math::abs(compact_mass - final_mass);
    const amrex::Real mass_tolerance = amrex::Real(1.0e-10) *
        amrex::max(amrex::Real(1.0), amrex::Math::abs(initial_mass));
    const amrex::Real projection_tolerance = amrex::Real(1.0e-12) *
        amrex::max(amrex::Real(1.0), amrex::Math::abs(compact_mass));
    const amrex::Real face_tolerance = amrex::Real(1.0e-12) *
        amrex::max(amrex::Real(1.0), amrex::Math::abs(final_mass));
    const bool passed = initial_variation > amrex::Real(0.0) &&
                        transport_change > amrex::Real(0.0) &&
                        mass_error <= mass_tolerance &&
                        compact_mass_error <= mass_tolerance &&
                        projection_error <= projection_tolerance &&
                        face_projection_error <= face_tolerance &&
                        closure.passes();

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream output(path);
        if (!output) amrex::Error("unable to write SBM P1 diagnostic: " + path);
        output << std::setprecision(17)
               << "format=erf-sbm-p1-diagnostic-v2\n"
               << "method=" << (anelastic ? "anelastic" : "compressible") << '\n'
               << "nbins=" << nbins << '\n'
               << "step_count=" << step_count << '\n'
               << "cell_volume=" << cell_volume << '\n'
               << "initial_mass=" << initial_mass << '\n'
               << "final_mass=" << final_mass << '\n'
               << "mass_error=" << mass_error << '\n'
               << "mass_tolerance=" << mass_tolerance << '\n'
               << "compact_mass=" << compact_mass << '\n'
               << "compact_mass_error=" << compact_mass_error << '\n'
               << "initial_variation=" << initial_variation << '\n'
               << "transport_change=" << transport_change << '\n'
               << "projection_error=" << projection_error << '\n'
               << "projection_tolerance=" << projection_tolerance << '\n'
               << "face_projection_error=" << face_projection_error << '\n'
               << "face_tolerance=" << face_tolerance << '\n'
               << "spectral_transfer_closure_error=" << closure.spectral_max << '\n'
               << "qc_transfer_closure_error=" << closure.qc_max << '\n'
               << "qr_transfer_closure_error=" << closure.qr_max << '\n'
               << "spectral_transfer_closure_tolerance=" << closure.spectral_tolerance << '\n'
               << "qc_transfer_closure_tolerance=" << closure.qc_tolerance << '\n'
               << "qr_transfer_closure_tolerance=" << closure.qr_tolerance << '\n'
               << "cell_state_bytes=" << cell_state_bytes << '\n'
               << "face_transfer_bytes=" << face_transfer_bytes << '\n'
               << "total_auxiliary_bytes=" << total_auxiliary_bytes << '\n'
               << "passed=" << (passed ? 1 : 0) << '\n';
    }
    amrex::ParallelDescriptor::Barrier("SBM P1 diagnostic");
    if (!passed) {
        std::ostringstream message;
        message << "SBM P1 numerical qualification failed: step_count=" << step_count
                << ", mass_error=" << mass_error
                << ", compact_mass_error=" << compact_mass_error
                << ", initial_variation=" << initial_variation
                << ", transport_change=" << transport_change
                << ", projection_error=" << projection_error
                << ", face_projection_error=" << face_projection_error
                << ", spectral_transfer_closure_error=" << closure.spectral_max
                << ", qc_transfer_closure_error=" << closure.qc_max
                << ", qr_transfer_closure_error=" << closure.qr_max;
        amrex::Error(message.str());
    }
}

} // namespace

void ERF::write_sbm_composite_diagnostic(const int nstep, const double time,
                                         const double dt_lev0)
{
    // The composite/reflux oracle is a completed-coarse-step qualification
    // diagnostic.  Do not evaluate it at an intermediate post_timestep call,
    // before the run has executed its final requested step and before all
    // expected interface transfers have been accumulated.
    if (max_step > 0 && nstep + 1 < max_step) return;
    if (sbm_layout == nullptr || sbm_auxiliary == nullptr || finest_level < 0) {
        amrex::Error("SBM composite diagnostic requested before auxiliary hierarchy exists");
    }
    const int ncomp = sbm_layout->ncomp();
    if (!sbm_composite_snapshot_valid ||
        sbm_composite_initial_totals.size() != static_cast<Long>(ncomp)) {
        amrex::Error("SBM composite diagnostic has no coarse-step initial snapshot");
    }
    std::vector<Real> initial(sbm_composite_initial_totals.begin(),
                              sbm_composite_initial_totals.end());
    std::vector<Real> final(static_cast<std::size_t>(ncomp), Real(0.0));
    Real face_projection_error = Real(0.0);
    Real accepted_face_transfer_l1 = Real(0.0);
    Real accepted_face_transfer_max = Real(0.0);
    Real accepted_bulk_transfer_l1 = Real(0.0);
    Real accepted_bulk_transfer_max = Real(0.0);
    std::vector<Real> accepted_face_transfer_sum(static_cast<std::size_t>(ncomp), Real(0.0));
    std::vector<Real> accepted_bulk_transfer_sum(2, Real(0.0));
    std::vector<Real> boundary_outward(static_cast<std::size_t>(ncomp), Real(0.0));
    const auto boundary_policy =
        ::erf_sbm::make_erf_transport_boundary_policy(geom[0], phys_bc_type);
    const auto boundary_kind_name = [](const ::erf_sbm::BoundaryKind kind) {
        switch (kind) {
        case ::erf_sbm::BoundaryKind::Periodic: return "Periodic";
        case ::erf_sbm::BoundaryKind::ImpermeableWall: return "ImpermeableWall";
        case ::erf_sbm::BoundaryKind::AdvectiveOutflow: return "AdvectiveOutflow";
        case ::erf_sbm::BoundaryKind::PrescribedSpectralInflow: return "PrescribedSpectralInflow";
        }
        return "Unknown";
    };
    std::array<const char*, AMREX_SPACEDIM*2> boundary_policy_names{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (geom[0].isPeriodic(dir)) {
            boundary_policy_names[2*dir] = "Periodic";
            boundary_policy_names[2*dir+1] = "Periodic";
        } else {
            boundary_policy_names[2*dir] = boundary_kind_name(
                boundary_policy.face_kind[2*dir]);
            boundary_policy_names[2*dir+1] = boundary_kind_name(
                boundary_policy.face_kind[2*dir+1]);
        }
    }
    for (int lev = 0; lev <= finest_level; ++lev) {
        const bool mask_coarse = lev < finest_level && fine_mask[lev+1] != nullptr;
        for (int comp = 0; comp < ncomp; ++comp) {
            final[static_cast<std::size_t>(comp)] += volWgtSumMF(
                lev, sbm_auxiliary->output(lev), comp, *detJ_cc[lev],
                *mapfac[lev][MapFacType::m_x], *mapfac[lev][MapFacType::m_y],
                mask_coarse, false);
        }
        if (static_cast<Long>(lev) < sbm_accepted_bulk_face_transfer.size() &&
            sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)] != nullptr) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                const auto& accepted_face = sbm_auxiliary->face_transfer_ledger(lev).accepted().direction(dir);
                const auto accepted_face_diagnostics = unique_face_transfer_diagnostics(
                    accepted_face, geom[lev], dir, ncomp);
                accepted_face_transfer_l1 += accepted_face_diagnostics.l1;
                accepted_face_transfer_max = amrex::max(accepted_face_transfer_max,
                                                        accepted_face_diagnostics.maximum);
                for (int comp = 0; comp < ncomp; ++comp) {
                    accepted_face_transfer_sum[static_cast<std::size_t>(comp)] +=
                        accepted_face_diagnostics.sum[static_cast<std::size_t>(comp)];
                }
                const auto& bulk_face = sbm_accepted_bulk_face_transfer[
                    static_cast<std::size_t>(lev)]->direction(dir);
                const auto bulk_face_diagnostics = unique_face_transfer_diagnostics(
                    bulk_face, geom[lev], dir, 2);
                accepted_bulk_transfer_l1 += bulk_face_diagnostics.l1;
                accepted_bulk_transfer_max = amrex::max(accepted_bulk_transfer_max,
                                                        bulk_face_diagnostics.maximum);
                for (int comp = 0; comp < 2; ++comp) {
                    accepted_bulk_transfer_sum[static_cast<std::size_t>(comp)] +=
                        bulk_face_diagnostics.sum[static_cast<std::size_t>(comp)];
                }
                // The accepted ledger stores flux integrated in time but not
                // multiplied by face area.  For the qualified static
                // Cartesian single-level boundary policy, this inventory is
                // the exact outward amount used by the conservation oracle.
                if (lev == 0 && !geom[lev].isPeriodic(dir)) {
                    const auto low_kind = boundary_policy.face_kind[2*dir];
                    const auto high_kind = boundary_policy.face_kind[2*dir+1];
                    const Real face_area = [&]() {
                        Real area = Real(1.0);
                        for (int transverse = 0; transverse < AMREX_SPACEDIM; ++transverse) {
                            if (transverse != dir) area *= geom[lev].CellSize(transverse);
                        }
                        return area;
                    }();
                    const auto boundary_sum = [&](const int component, const int coordinate) {
                        MultiFab selected(bulk_face.boxArray(), bulk_face.DistributionMap(), 1, 0);
                        selected.setVal(Real(0.0));
                        for (MFIter mfi(bulk_face); mfi.isValid(); ++mfi) {
                            const auto source = accepted_face.const_array(mfi);
                            const auto destination = selected.array(mfi);
                            const Box box = mfi.validbox();
                            ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                                const int face_coordinate = dir == 0 ? i : (dir == 1 ? j : k);
                                if (face_coordinate == coordinate) {
                                    destination(i,j,k,0) = source(i,j,k,component);
                                }
                            });
                        }
                        return selected.sum(0) * face_area;
                    };
                    for (int comp = 0; comp < ncomp; ++comp) {
                        if (low_kind == ::erf_sbm::BoundaryKind::AdvectiveOutflow) {
                            boundary_outward[static_cast<std::size_t>(comp)] -=
                                boundary_sum(comp, geom[lev].Domain().smallEnd(dir));
                        }
                        if (high_kind == ::erf_sbm::BoundaryKind::AdvectiveOutflow) {
                            boundary_outward[static_cast<std::size_t>(comp)] +=
                                boundary_sum(comp, geom[lev].Domain().bigEnd(dir) + 1);
                        }
                    }
                }
            }
            face_projection_error = amrex::max(face_projection_error,
                sbm_layout->liquid_projection().cloud_rain_split >= 0 ?
                ::erf_sbm::SBMBulkProjection(*sbm_layout).max_face_projection_error(
                    sbm_auxiliary->face_transfer_ledger(lev).accepted(),
                    *sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)]) : Real(0.0));
        }
    }

    const auto& population = sbm_layout->populations().front();
    const auto& projection = sbm_layout->liquid_projection();
    Real compact_projection_error = Real(0.0);
    Real compact_initial = Real(0.0);
    Real compact_final = Real(0.0);
    Real compact_final_qc = Real(0.0);
    Real compact_final_qr = Real(0.0);
    Real spectral_cloud_initial = Real(0.0);
    Real spectral_cloud_final = Real(0.0);
    Real spectral_rain_initial = Real(0.0);
    Real spectral_rain_final = Real(0.0);
    for (int bin = 0; bin < projection.cloud_rain_split; ++bin) {
        spectral_cloud_initial += initial[static_cast<std::size_t>(population.mass_offset + bin)];
    }
    for (int bin = projection.cloud_rain_split; bin < population.grid.nbins(); ++bin) {
        spectral_rain_initial += initial[static_cast<std::size_t>(population.mass_offset + bin)];
    }
    compact_initial = spectral_cloud_initial + spectral_rain_initial;
    for (int lev = 0; lev <= finest_level; ++lev) {
        const bool mask_coarse = lev < finest_level && fine_mask[lev+1] != nullptr;
        compact_final_qc += volWgtSumMF(lev, vars_new[lev][Vars::cons], RhoQ2_comp,
                                     *detJ_cc[lev], *mapfac[lev][MapFacType::m_x],
                                     *mapfac[lev][MapFacType::m_y], mask_coarse, false);
        compact_final_qr += volWgtSumMF(lev, vars_new[lev][Vars::cons], RhoQ3_comp,
                                     *detJ_cc[lev], *mapfac[lev][MapFacType::m_x],
                                     *mapfac[lev][MapFacType::m_y], mask_coarse, false);
        compact_final = compact_final_qc + compact_final_qr;
        spectral_cloud_final += volWgtSumMF(lev, sbm_auxiliary->output(lev),
                                            population.mass_offset,
                                            *detJ_cc[lev], *mapfac[lev][MapFacType::m_x],
                                            *mapfac[lev][MapFacType::m_y], mask_coarse, false);
        for (int bin = 1; bin < projection.cloud_rain_split; ++bin) {
            spectral_cloud_final += volWgtSumMF(lev, sbm_auxiliary->output(lev),
                                                population.mass_offset + bin,
                                                *detJ_cc[lev], *mapfac[lev][MapFacType::m_x],
                                                *mapfac[lev][MapFacType::m_y], mask_coarse, false);
        }
        for (int bin = projection.cloud_rain_split; bin < population.grid.nbins(); ++bin) {
            spectral_rain_final += volWgtSumMF(lev, sbm_auxiliary->output(lev),
                                               population.mass_offset + bin,
                                               *detJ_cc[lev], *mapfac[lev][MapFacType::m_x],
                                               *mapfac[lev][MapFacType::m_y], mask_coarse, false);
        }
    }
    compact_projection_error = amrex::max(
        amrex::Math::abs(compact_final - spectral_cloud_final - spectral_rain_final),
        amrex::Math::abs(compact_initial - spectral_cloud_initial - spectral_rain_initial));
    Real variable_density_ratio_error = Real(0.0);
    if (solverChoice.sbm_manufactured_variable_density) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            variable_density_ratio_error = amrex::max(
                variable_density_ratio_error,
                sbm_max_manufactured_ratio_error(
                    *sbm_layout, sbm_auxiliary->output(lev),
                    vars_new[lev][Vars::cons]));
        }
    }
    const Real variable_density_ratio_tolerance =
        solverChoice.sbm_manufactured_variable_density ?
        Real(4096.0) * std::numeric_limits<Real>::epsilon() * Real(1.0e-6) : Real(0.0);

    std::vector<Real> component_scale(static_cast<std::size_t>(ncomp), Real(0.0));
    std::vector<Real> component_tolerance(static_cast<std::size_t>(ncomp), Real(0.0));
    const Real tolerance_factor = Real(1.0e-12);
    const Real compact_scale = amrex::max(amrex::Math::abs(compact_initial),
                                           amrex::Math::abs(compact_final));
    const Real compact_tolerance = compact_scale == Real(0.0) ? Real(0.0) :
        tolerance_factor * compact_scale;
    const Real face_scale = amrex::max(accepted_face_transfer_max,
                                       accepted_face_transfer_l1);
    const Real face_tolerance = face_scale == Real(0.0) ? Real(0.0) :
        tolerance_factor * face_scale;
    const Real composite_error = [&]() {
        Real error = Real(0.0);
        for (int comp = 0; comp < ncomp; ++comp) {
            const auto index = static_cast<std::size_t>(comp);
            const Real conservation_residual = final[index] - initial[index] +
                boundary_outward[index];
            const Real component_error = amrex::Math::abs(conservation_residual);
            component_scale[index] = amrex::max(amrex::Math::abs(initial[index]),
                                                amrex::Math::abs(final[index]));
            component_tolerance[index] = component_scale[index] == Real(0.0) ?
                Real(0.0) : tolerance_factor * component_scale[index];
            error = amrex::max(error, component_error);
        }
        return error;
    }();
    // The independent oracle compares two separately reduced paths (fine
    // averaging and authoritative reflux).  Use a bounded machine-scale
    // allowance for their different summation order, while retaining a
    // relative tolerance with no order-one floor.
    const Real interface_tolerance = Real(4096.0) * std::numeric_limits<Real>::epsilon();
    bool interface_passed = finest_level == 0;
    if (finest_level > 0) {
        interface_passed = sbm_interface_oracle_available &&
                           sbm_interface_oracle_error.size() == static_cast<Long>(ncomp);
        if (interface_passed) {
            for (int comp = 0; comp < ncomp; ++comp) {
                const auto index = static_cast<std::size_t>(comp);
                const Real interface_scale = amrex::max(
                    accepted_face_transfer_max,
                    amrex::max(sbm_interface_coarse_scale[index],
                               amrex::max(sbm_interface_fine_scale[index],
                                          sbm_interface_reflux_scale[index])));
                interface_passed = interface_passed &&
                    sbm_interface_oracle_error[index] <= interface_tolerance * interface_scale;
            }
        }
    }
    bool component_residuals_passed = true;
    for (int comp = 0; comp < ncomp; ++comp) {
        const auto index = static_cast<std::size_t>(comp);
        component_residuals_passed = component_residuals_passed &&
            amrex::Math::abs(final[index] - initial[index] + boundary_outward[index]) <=
                component_tolerance[index];
    }
    const bool passed = component_residuals_passed &&
                        compact_projection_error <= compact_tolerance &&
                        face_projection_error <= face_tolerance &&
                        interface_passed &&
                        variable_density_ratio_error <= variable_density_ratio_tolerance;
    const std::size_t temporary_bytes = ::erf_sbm::grouped_fct_peak_working_bytes(
        *sbm_layout, grids[0], dmap[0], solverChoice.sbm_chunk_size);
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream output(solverChoice.sbm_composite_diagnostic_file);
        if (!output) amrex::Error("unable to write SBM composite diagnostic: " +
                                  solverChoice.sbm_composite_diagnostic_file);
        output << std::setprecision(17)
               << "format=erf-sbm-p2-composite-v1\n"
               << "coarse_step=" << nstep + 1 << '\n'
               << "coarse_steps=" << nstep + 1 << '\n'
               << "fine_steps=" << (finest_level > 0 ? istep[1] : 0) << '\n'
               << "fine_substeps_per_coarse=" << (finest_level > 0 ? nsubsteps[1] : 1) << '\n'
               << "time=" << time << '\n'
               << "dt_lev0=" << dt_lev0 << '\n'
               << "boundary_policy_xlo=" << boundary_policy_names[0] << '\n'
               << "boundary_policy_xhi=" << boundary_policy_names[1] << '\n'
               << "boundary_policy_ylo=" << boundary_policy_names[2] << '\n'
               << "boundary_policy_yhi=" << boundary_policy_names[3] << '\n'
               << "boundary_policy_zlo=" << boundary_policy_names[4] << '\n'
               << "boundary_policy_zhi=" << boundary_policy_names[5] << '\n'
               << "finest_level=" << finest_level << '\n'
               << "level_count=" << finest_level + 1 << '\n'
               << "nbins=" << population.grid.nbins() << '\n'
               << "moment_mode=" << (population.number_offset >= 0 ? 2 : 1) << '\n'
               << "composite_error=" << composite_error << '\n'
               << "composite_tolerance_factor=" << tolerance_factor << '\n'
               << "composite_residuals_passed=" << (component_residuals_passed ? 1 : 0) << '\n'
               << "compact_tolerance=" << compact_tolerance << '\n'
               << "accepted_face_projection_tolerance=" << face_tolerance << '\n'
               << "compact_projection_error=" << compact_projection_error << '\n'
               << "accepted_face_projection_error=" << face_projection_error << '\n'
               << "variable_density_ratio_error=" << variable_density_ratio_error << '\n'
               << "variable_density_ratio_tolerance=" << variable_density_ratio_tolerance << '\n'
               << "accepted_face_transfer_l1=" << accepted_face_transfer_l1 << '\n'
               << "accepted_face_transfer_max=" << accepted_face_transfer_max << '\n'
               << "accepted_bulk_transfer_l1=" << accepted_bulk_transfer_l1 << '\n'
               << "accepted_bulk_transfer_max=" << accepted_bulk_transfer_max << '\n'
               << "interface_oracle_available=" << (sbm_interface_oracle_available ? 1 : 0) << '\n'
               << "interface_oracle_passed=" << (interface_passed ? 1 : 0) << '\n'
               << "interface_oracle_tolerance=" << interface_tolerance << '\n'
               << "minimum_accepted_limiter=" << sbm_minimum_accepted_limiter << '\n'
               << "post_reflux_validation_count=" << sbm_post_reflux_validation_count << '\n'
               << "post_reflux_material_rejection_count=" << sbm_post_reflux_material_rejection_count << '\n'
               << "active_limiter=" << ((sbm_minimum_accepted_limiter > Real(0.0) &&
                                          sbm_minimum_accepted_limiter < Real(1.0)) ? 1 : 0) << '\n'
               << "persistent_cell_state_bytes=" << sbm_auxiliary->state_resident_bytes() << '\n'
               << "persistent_accepted_ledger_bytes=" << sbm_auxiliary->face_transfer_resident_bytes() << '\n'
               << "temporary_p2_working_bytes=" << temporary_bytes << '\n'
               << "composite_initial_total=" << std::accumulate(initial.begin(), initial.end(), Real(0.0)) << '\n'
               << "composite_final_total=" << std::accumulate(final.begin(), final.end(), Real(0.0)) << '\n'
               << "compact_initial_total=" << compact_initial << '\n'
               << "compact_final_total=" << compact_final << '\n'
               << "compact_final_qc=" << compact_final_qc << '\n'
               << "compact_final_qr=" << compact_final_qr << '\n'
               << "spectral_cloud_initial=" << spectral_cloud_initial << '\n'
               << "spectral_cloud_final=" << spectral_cloud_final << '\n'
               << "spectral_rain_initial=" << spectral_rain_initial << '\n'
               << "spectral_rain_final=" << spectral_rain_final << '\n';
        for (int comp = 0; comp < ncomp; ++comp) {
            output << "composite_initial_comp_" << comp << '=' << initial[static_cast<std::size_t>(comp)] << '\n'
                   << "composite_final_comp_" << comp << '=' << final[static_cast<std::size_t>(comp)] << '\n'
                   << "composite_error_comp_" << comp << '=' << amrex::Math::abs(
                       final[static_cast<std::size_t>(comp)] - initial[static_cast<std::size_t>(comp)] +
                       boundary_outward[static_cast<std::size_t>(comp)]) << '\n'
                   << "boundary_inventory_outward_comp_" << comp << '=' <<
                       boundary_outward[static_cast<std::size_t>(comp)] << '\n'
                   << "boundary_inventory_closure_error_comp_" << comp << '=' <<
                       amrex::Math::abs(final[static_cast<std::size_t>(comp)] -
                                        initial[static_cast<std::size_t>(comp)] +
                                        boundary_outward[static_cast<std::size_t>(comp)]) << '\n'
                   << "composite_scale_comp_" << comp << '=' << component_scale[static_cast<std::size_t>(comp)] << '\n'
                   << "composite_tolerance_comp_" << comp << '=' << component_tolerance[static_cast<std::size_t>(comp)] << '\n';
            output << "accepted_face_transfer_sum_comp_" << comp << '=' <<
                accepted_face_transfer_sum[static_cast<std::size_t>(comp)] << '\n';
            if (sbm_interface_oracle_available) {
                const auto index = static_cast<std::size_t>(comp);
                output << "interface_coarse_transfer_comp_" << comp << '=' <<
                           sbm_interface_coarse_transfer[index] << '\n'
                       << "interface_fine_transfer_comp_" << comp << '=' <<
                           sbm_interface_fine_transfer[index] << '\n'
                       << "interface_mismatch_comp_" << comp << '=' <<
                           sbm_interface_fine_transfer[index] - sbm_interface_coarse_transfer[index] << '\n'
                       << "interface_reflux_correction_comp_" << comp << '=' <<
                           sbm_interface_reflux_correction[index] << '\n'
                       << "interface_oracle_error_comp_" << comp << '=' <<
                           sbm_interface_oracle_error[index] << '\n'
                       << "interface_scale_comp_" << comp << '=' <<
                           amrex::max(accepted_face_transfer_max,
                                      amrex::max(sbm_interface_coarse_scale[index],
                                                 amrex::max(sbm_interface_fine_scale[index],
                                                            sbm_interface_reflux_scale[index]))) << '\n';
            }
        }
        output << "accepted_bulk_transfer_sum_qc=" << accepted_bulk_transfer_sum[0] << '\n'
               << "accepted_bulk_transfer_sum_qr=" << accepted_bulk_transfer_sum[1] << '\n';
        output << "interface_oracle=accepted-transfer-mismatch-vs-authoritative-reflux-correction\n"
               << "passed=" << (passed ? 1 : 0) << '\n';
    }
    ParallelDescriptor::Barrier("SBM P2 composite diagnostic");
    if (!passed) {
        std::ostringstream message;
        message << "SBM P2 composite qualification failed: composite_error=" << composite_error
                << ", component_residuals_passed=" << (component_residuals_passed ? 1 : 0)
                << ", compact_tolerance=" << compact_tolerance
                << ", compact_projection_error=" << compact_projection_error
                << ", accepted_face_projection_error=" << face_projection_error
                << ", variable_density_ratio_error=" << variable_density_ratio_error
                << ", interface_oracle_passed=" << (interface_passed ? 1 : 0);
        amrex::Error(message.str());
    }
}

void ERF::begin_sbm_reflux_oracle(const int lev)
{
    if (lev < 0 || lev >= finest_level || sbm_auxiliary == nullptr ||
        !sbm_auxiliary->has_level(lev) || !sbm_auxiliary->has_level(lev + 1) ||
        fine_mask[lev + 1] == nullptr) {
        amrex::Error("SBM reflux oracle requires a complete coarse/fine hierarchy");
    }
    const auto& coarse = sbm_auxiliary->output(lev);
    sbm_reflux_pre_state = std::make_unique<MultiFab>(
        coarse.boxArray(), coarse.DistributionMap(), coarse.nComp(), 0);
    MultiFab::Copy(*sbm_reflux_pre_state, coarse, 0, 0, coarse.nComp(), 0);
    sbm_reflux_level = lev;
}

void ERF::finish_sbm_reflux_oracle(const int lev)
{
    if (sbm_reflux_pre_state == nullptr || sbm_reflux_level != lev ||
        lev < 0 || lev >= finest_level || fine_mask[lev + 1] == nullptr) {
        amrex::Error("SBM reflux oracle has no matching pre-reflux state");
    }

    const int ncomp = sbm_layout->ncomp();
    if (sbm_interface_coarse_transfer.size() != static_cast<Long>(ncomp)) {
        sbm_interface_coarse_transfer.assign(static_cast<std::size_t>(ncomp), Real(0.0));
        sbm_interface_fine_transfer.assign(static_cast<std::size_t>(ncomp), Real(0.0));
        sbm_interface_reflux_correction.assign(static_cast<std::size_t>(ncomp), Real(0.0));
        sbm_interface_coarse_scale.assign(static_cast<std::size_t>(ncomp), Real(0.0));
        sbm_interface_fine_scale.assign(static_cast<std::size_t>(ncomp), Real(0.0));
        sbm_interface_reflux_scale.assign(static_cast<std::size_t>(ncomp), Real(0.0));
        sbm_interface_oracle_error.assign(static_cast<std::size_t>(ncomp), Real(0.0));
    }

    const auto& coarse_ledger = sbm_auxiliary->face_transfer_ledger(lev).accepted();
    const auto& fine_ledger = sbm_auxiliary->face_transfer_ledger(lev + 1).accepted();
    ::erf_auxiliary::AuxiliaryFaceTransfer fine_average;
    fine_average.define(grids[lev], dmap[lev], ncomp, 0);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::average_down_faces(fine_ledger.direction(dir), fine_average.direction(dir),
                                  refRatio(lev), geom[lev]);
    }

    // The fine mask has no ghosts because it is also used as a volume mask.
    // A periodic one-ghost view makes the interface test well-defined at a
    // periodic image while keeping all updates restricted to valid coarse
    // cells.  The qualification fixture keeps the refined footprint away
    // from the physical domain boundary, so every covered interface cell is
    // represented by a valid coarse index.
    MultiFab interface_mask(grids[lev], dmap[lev], 1, 1);
    MultiFab::Copy(interface_mask, *fine_mask[lev + 1], 0, 0, 1, 0);
    interface_mask.FillBoundary(geom[lev].periodicity());

    MultiFab coarse_transfer(grids[lev], dmap[lev], ncomp, 0);
    MultiFab fine_transfer(grids[lev], dmap[lev], ncomp, 0);
    coarse_transfer.setVal(Real(0.0));
    fine_transfer.setVal(Real(0.0));
    const auto dx = geom[lev].CellSizeArray();
    const Real face_area_x = dx[1] * dx[2];
    const Real face_area_y = dx[0] * dx[2];
    const Real face_area_z = dx[0] * dx[1];
    for (MFIter mfi(coarse_transfer); mfi.isValid(); ++mfi) {
        const Box box = mfi.validbox();
        const auto mask = interface_mask.const_array(mfi);
        const auto coarse_x = coarse_ledger.direction(0).const_array(mfi);
        const auto coarse_y = coarse_ledger.direction(1).const_array(mfi);
        const auto coarse_z = coarse_ledger.direction(2).const_array(mfi);
        const auto fine_x = fine_average.direction(0).const_array(mfi);
        const auto fine_y = fine_average.direction(1).const_array(mfi);
        const auto fine_z = fine_average.direction(2).const_array(mfi);
        const auto coarse_sum = coarse_transfer.array(mfi);
        const auto fine_sum = fine_transfer.array(mfi);
        ParallelFor(box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
            if (mask(i,j,k,0) == Real(0.0)) return;
            if (mask(i-1,j,k,0) == Real(0.0)) {
                coarse_sum(i,j,k,comp) += coarse_x(i,j,k,comp) * face_area_x;
                fine_sum(i,j,k,comp) += fine_x(i,j,k,comp) * face_area_x;
            }
            if (mask(i+1,j,k,0) == Real(0.0)) {
                coarse_sum(i,j,k,comp) -= coarse_x(i+1,j,k,comp) * face_area_x;
                fine_sum(i,j,k,comp) -= fine_x(i+1,j,k,comp) * face_area_x;
            }
            if (mask(i,j-1,k,0) == Real(0.0)) {
                coarse_sum(i,j,k,comp) += coarse_y(i,j,k,comp) * face_area_y;
                fine_sum(i,j,k,comp) += fine_y(i,j,k,comp) * face_area_y;
            }
            if (mask(i,j+1,k,0) == Real(0.0)) {
                coarse_sum(i,j,k,comp) -= coarse_y(i,j+1,k,comp) * face_area_y;
                fine_sum(i,j,k,comp) -= fine_y(i,j+1,k,comp) * face_area_y;
            }
            if (mask(i,j,k-1,0) == Real(0.0)) {
                coarse_sum(i,j,k,comp) += coarse_z(i,j,k,comp) * face_area_z;
                fine_sum(i,j,k,comp) += fine_z(i,j,k,comp) * face_area_z;
            }
            if (mask(i,j,k+1,0) == Real(0.0)) {
                coarse_sum(i,j,k,comp) -= coarse_z(i,j,k+1,comp) * face_area_z;
                fine_sum(i,j,k,comp) -= fine_z(i,j,k+1,comp) * face_area_z;
            }
        });
    }
    amrex::Gpu::synchronize();

    const Real cell_volume = dx[0] * dx[1] * dx[2];
    MultiFab actual_correction(grids[lev], dmap[lev], ncomp, 0);
    actual_correction.setVal(Real(0.0));
    for (MFIter mfi(actual_correction); mfi.isValid(); ++mfi) {
        const Box box = mfi.validbox();
        const auto actual = actual_correction.array(mfi);
        const auto post = sbm_auxiliary->output(lev).const_array(mfi);
        const auto pre = sbm_reflux_pre_state->const_array(mfi);
        const auto mask = fine_mask[lev + 1]->const_array(mfi);
        ParallelFor(box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
            if (mask(i, j, k, 0) != Real(0.0)) {
                actual(i, j, k, comp) = (post(i, j, k, comp) - pre(i, j, k, comp)) * cell_volume;
            }
        });
    }
    amrex::Gpu::synchronize();

    MultiFab expected_correction(grids[lev], dmap[lev], ncomp, 0);
    MultiFab::Copy(expected_correction, fine_transfer, 0, 0, ncomp, 0);
    MultiFab::Subtract(expected_correction, coarse_transfer, 0, 0, ncomp, 0);
    MultiFab correction_error(expected_correction.boxArray(), expected_correction.DistributionMap(),
                              ncomp, 0);
    MultiFab::Copy(correction_error, actual_correction, 0, 0, ncomp, 0);
    MultiFab::Subtract(correction_error, expected_correction, 0, 0, ncomp, 0);
    for (int comp = 0; comp < ncomp; ++comp) {
        const Real coarse_value = coarse_transfer.sum(comp);
        const Real fine_value = fine_transfer.sum(comp);
        const Real actual_value = actual_correction.sum(comp);
        const Real error = correction_error.norm0(comp);
        sbm_interface_coarse_transfer[static_cast<std::size_t>(comp)] += coarse_value;
        sbm_interface_fine_transfer[static_cast<std::size_t>(comp)] += fine_value;
        sbm_interface_reflux_correction[static_cast<std::size_t>(comp)] += actual_value;
        sbm_interface_coarse_scale[static_cast<std::size_t>(comp)] = amrex::max(
            sbm_interface_coarse_scale[static_cast<std::size_t>(comp)],
            coarse_transfer.norm0(comp));
        sbm_interface_fine_scale[static_cast<std::size_t>(comp)] = amrex::max(
            sbm_interface_fine_scale[static_cast<std::size_t>(comp)],
            fine_transfer.norm0(comp));
        sbm_interface_reflux_scale[static_cast<std::size_t>(comp)] = amrex::max(
            sbm_interface_reflux_scale[static_cast<std::size_t>(comp)],
            actual_correction.norm0(comp));
        sbm_interface_oracle_error[static_cast<std::size_t>(comp)] = amrex::max(
            sbm_interface_oracle_error[static_cast<std::size_t>(comp)], error);
    }
    sbm_interface_oracle_available = true;
    sbm_reflux_pre_state.reset();
    sbm_reflux_level = -1;
}

void ERF::validate_sbm_post_reflux(const int lev)
{
    ++sbm_post_reflux_validation_count;
    try {
        ::erf_sbm::validate_admissible_state(*sbm_auxiliary, *sbm_layout, lev);
    } catch (...) {
        ++sbm_post_reflux_material_rejection_count;
        throw;
    }
}

void ERF::synchronize_sbm_level_companions(const int lev,
                                           const amrex::BoxArray& ba,
                                           const amrex::DistributionMapping& dm,
                                           const bool preserve_overlap)
{
    if (lev < 0 || sbm_layout == nullptr) {
        amrex::Error("invalid SBM companion lifecycle request");
    }
    if (static_cast<int>(sbm_accepted_bulk_face_transfer.size()) <= lev) {
        sbm_accepted_bulk_face_transfer.resize(static_cast<std::size_t>(lev + 1));
    }
    if (static_cast<int>(sbm_initial_bulk_state.size()) <= lev) {
        sbm_initial_bulk_state.resize(static_cast<std::size_t>(lev + 1));
    }
    if (static_cast<int>(sbm_evaluation_density.size()) <= lev) {
        sbm_evaluation_density.resize(static_cast<std::size_t>(lev + 1));
    }

    auto& accepted = sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)];
    if (!accepted || !accepted->compatible_with(ba, dm, 2)) {
        // Accepted transfer is stage-local.  It is deliberately reset on a
        // layout change rather than copied through a stale face ownership map.
        accepted = std::make_unique<::erf_auxiliary::AuxiliaryFaceTransfer>();
        accepted->define(ba, dm, 2, 0);
    }

    auto& initial = sbm_initial_bulk_state[static_cast<std::size_t>(lev)];
    const bool compatible = initial && initial->boxArray() == ba &&
                            initial->DistributionMap() == dm && initial->nComp() == 2;
    if (!compatible) {
        auto replacement = std::make_unique<amrex::MultiFab>(ba, dm, 2, 0);
        replacement->setVal(Real(0.0));
        if (preserve_overlap && initial) {
            replacement->ParallelCopy(*initial, 0, 0, 2, amrex::IntVect(0),
                                      amrex::IntVect(0), amrex::Periodicity::NonPeriodic());
        }
        initial = std::move(replacement);
    }

    auto& evaluation_density = sbm_evaluation_density[static_cast<std::size_t>(lev)];
    const bool density_compatible = evaluation_density &&
        evaluation_density->boxArray() == ba && evaluation_density->DistributionMap() == dm &&
        evaluation_density->nComp() == 1 && evaluation_density->nGrow() >= 2;
    if (!density_compatible) {
        auto replacement = std::make_unique<amrex::MultiFab>(ba, dm, 1, 2);
        replacement->setVal(Real(0.0));
        if (preserve_overlap && evaluation_density) {
            replacement->ParallelCopy(*evaluation_density, 0, 0, 1, amrex::IntVect(0),
                                      amrex::IntVect(0), amrex::Periodicity::NonPeriodic());
        }
        evaluation_density = std::move(replacement);
    }
}

void ERF::initialize_sbm_auxiliary(const int lev)
{
    if (solverChoice.moisture_type != MoistureType::SBM) return;
    AMREX_ALWAYS_ASSERT(sbm_layout != nullptr);
    if (lev < 0) amrex::Error("SBM level index must be nonnegative");
    if (lev == 0) {
        sbm_step_count = 0;
        sbm_composite_snapshot_valid = false;
        sbm_composite_initial_totals.clear();
        sbm_minimum_accepted_limiter = Real(1.0);
        sbm_reflux_pre_state.reset();
        sbm_reflux_level = -1;
        sbm_interface_coarse_transfer.clear();
        sbm_interface_fine_transfer.clear();
        sbm_interface_reflux_correction.clear();
        sbm_interface_coarse_scale.clear();
        sbm_interface_fine_scale.clear();
        sbm_interface_reflux_scale.clear();
        sbm_interface_oracle_error.clear();
        sbm_interface_oracle_available = false;
        sbm_post_reflux_validation_count = 0;
        sbm_post_reflux_material_rejection_count = 0;
    }
    if (!sbm_auxiliary) {
        sbm_auxiliary = std::make_unique<::erf_auxiliary::AuxiliaryStateManager>(sbm_layout->auxiliary_layout());
    }
    synchronize_sbm_level_companions(lev, grids[lev], dmap[lev], false);
    if (!sbm_ownership) {
        sbm_ownership = std::make_unique<::erf_sbm::OwnershipRegistry>(true);
    }
    if (!sbm_auxiliary->has_level(lev)) {
        // Grouped WENO/FCT owns all transport scratch locally by complete
        // closure chunk.  Do not allocate a hidden full-layout manager FAB.
        // The legacy donor-cell path still uses the manager scratch contract,
        // so retain that workspace when the selected method requires it.
        const int scratch_components = solverChoice.sbm_transport_method == "GroupedFCT_WENOZ3" ?
            0 : sbm_layout->ncomp();
        sbm_auxiliary->define_level(lev, grids[lev], dmap[lev], 2, scratch_components);
    }
    auto& aux = sbm_auxiliary->output(lev);
    auto& core = vars_new[lev][Vars::cons];
    // On restart MakeNewLevel* is the allocation phase.  ReadCheckpointFile
    // owns restoration and schema validation; never overwrite checkpointed
    // auxiliary data with a manufactured or empty state here.
    if (!restart_chkfile.empty()) return;
    const auto& projection = *sbm_layout;
    const auto& population = projection.populations().front();
    const int nbins = population.grid.nbins();
    const int offset = population.mass_offset;

    // The variable-density qualification fixtures use the host's actual dry
    // density state as the carrier throughout regrid and transport.  Their
    // divergence-free manufactured carrier keeps that field stationary in
    // both host temporal contracts.  This is a manufactured P2 test field,
    // not a new thermodynamic or P3 process.
    const bool host_cfl_counterexample =
        solverChoice.sbm_manufactured_host_cfl_counterexample;
#ifdef ERF_SBM_QUALIFICATION_TEST_HOOKS
    const bool preserve_host_density = solverChoice.sbm_test_preserve_host_density;
#else
    const bool preserve_host_density = false;
#endif
    if (lev == 0 && (solverChoice.sbm_manufactured_variable_density ||
                     host_cfl_counterexample) &&
        !preserve_host_density) {
        auto& core_old = vars_old[lev][Vars::cons];
        const Real xlo = geom[lev].ProbLo(0);
        const Real xlen = geom[lev].ProbHi(0) - xlo;
        const Real dx = geom[lev].CellSize(0);
        for (MFIter mfi(core); mfi.isValid(); ++mfi) {
            const Box box = mfi.validbox();
            const auto state = core.array(mfi);
            const auto old_state = core_old.array(mfi);
            ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = xlo + (Real(i) + Real(0.5)) * dx;
                const Real rho = host_cfl_counterexample ?
                    ((i % 2 == 0) ? Real(0.37) : Real(1.20)) :
                    Real(1.0) + Real(0.15) *
                    std::sin(Real(6.2831853071795864769) * (x - xlo) / xlen);
                // Keep the manufactured variable-density field pressure
                // equilibrated.  RhoTheta is rho*theta, so setting it to
                // rho*300 would introduce a large pressure gradient and
                // make the host evolve density for a reason unrelated to
                // the SBM stage-density contract.
                const Real rhotheta = Real(300.0);
                state(i,j,k,Rho_comp) = rho;
                old_state(i,j,k,Rho_comp) = rho;
                state(i,j,k,RhoTheta_comp) = rhotheta;
                old_state(i,j,k,RhoTheta_comp) = rhotheta;
            });
        }
        core.FillBoundary(geom[lev].periodicity());
        core_old.FillBoundary(geom[lev].periodicity());
    }
    if (lev == 0 && host_cfl_counterexample && ParallelDescriptor::IOProcessor()) {
        const Real reconstructed_carrier = Real(0.125) *
            Real(0.5) * (Real(0.37) + Real(1.20));
        const Real actual_carrier = Real(0.125);
        Print() << std::setprecision(17)
                << "SBM manufactured carrier comparison reconstructed_u_rho="
                << reconstructed_carrier
                << " actual_avg_xmom=" << actual_carrier
                << " actual_over_reconstructed=" << actual_carrier / reconstructed_carrier
                << "\n";
    }

    // The manufactured regression supplies a nonzero ERF carrier field while
    // production inputs retain the ordinary initialized velocity/momentum.
    // These are the same face-centered fields later handed to the transport
    // kernel, so the qualification cannot pass through an independent donor
    // velocity reconstruction.
    if (lev == 0 && solverChoice.sbm_manufactured_velocity != Real(0.0)) {
        vars_new[lev][Vars::xvel].setVal(solverChoice.sbm_manufactured_velocity);
        vars_old[lev][Vars::xvel].setVal(solverChoice.sbm_manufactured_velocity);
        vars_new[lev][Vars::yvel].setVal(Real(0.0));
        vars_old[lev][Vars::yvel].setVal(Real(0.0));
        vars_new[lev][Vars::zvel].setVal(Real(0.0));
        vars_old[lev][Vars::zvel].setVal(Real(0.0));
        avg_xmom[lev].setVal(solverChoice.sbm_manufactured_velocity);
        avg_ymom[lev].setVal(Real(0.0));
        avg_zmom[lev].setVal(Real(0.0));
    }
    // There is intentionally no bulk-to-spectrum guess.  An empty spectrum is
    // allowed only when both compact condensate fields are zero.  Tests and
    // explicit prototype runs may request the deterministic manufactured state.
    if (!solverChoice.sbm_manufactured_initialization) {
        const Real qc_max = core.max(RhoQ2_comp);
        const Real qc_min = core.min(RhoQ2_comp);
        const Real qr_max = core.max(RhoQ3_comp);
        const Real qr_min = core.min(RhoQ3_comp);
        if (qc_max != Real(0.0) || qc_min != Real(0.0) ||
            qr_max != Real(0.0) || qr_min != Real(0.0)) {
            amrex::Error("SBM requires explicit spectral initialization for nonzero qc/qr; no bulk-to-bin guess is permitted");
        }
    }

    for (MFIter mfi(aux); mfi.isValid(); ++mfi) {
        const Box box = mfi.validbox();
        const auto aux_arr = aux.array(mfi);
        const auto core_arr = core.array(mfi);
        const bool manufactured = solverChoice.sbm_manufactured_initialization;
#ifdef ERF_SBM_QUALIFICATION_TEST_HOOKS
        const bool active_limiter = solverChoice.sbm_test_active_limiter;
#else
        const bool active_limiter = false;
#endif
        const bool variable_density = solverChoice.sbm_manufactured_variable_density;
        const Real xlo = geom[lev].ProbLo(0);
        const Real xlen = geom[lev].ProbHi(0) - xlo;
        const Real dx = geom[lev].CellSize(0);
        const int number_offset = population.number_offset;
        for (int b = 0; b < nbins; ++b) {
                const Real lower = number_offset >= 0 ?
                    population.grid.edges()[static_cast<std::size_t>(b)] : Real(0.0);
                const Real pivot = number_offset >= 0 ? population.grid.pivot(b) : Real(0.0);
                const Real upper = number_offset >= 0 ?
                    population.grid.edges()[static_cast<std::size_t>(b + 1)] : Real(0.0);
            ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real rho = core_arr(i,j,k,Rho_comp);
                const Real x = xlo + (Real(i) + Real(0.5)) * dx;
                // Place one low cell immediately upstream of the open
                // transport face.  The test-only carrier pattern closes its
                // left face, so the high-order outflow correction is genuinely
                // constrained instead of being hidden by a large inflow.
                const bool active_cell = active_limiter &&
                    (i % 16 == 7);
                const Real variation = (variable_density ||
                    host_cfl_counterexample) ? Real(1.0) :
                    active_limiter ?
                    (active_cell ? Real(0.001) : Real(1.999)) :
                    Real(1.0) + Real(0.25) *
                    std::sin(Real(6.2831853071795864769) * (x - xlo) / xlen);
                // Small deterministic positive values on both sides of the
                // projection split.  This is a transport manufactured field,
                // not a physical droplet or aerosol distribution.
                aux_arr(i,j,k,offset+b) = manufactured ?
                    rho * Real(1.0e-6) * Real(b+1) * variation : Real(0.0);
                if (number_offset >= 0) {
                    const Real number_pivot = active_cell ?
                        lower + Real(0.75) * (upper - lower) : pivot;
                    aux_arr(i,j,k,number_offset+b) =
                        number_pivot > Real(0.0) ? aux_arr(i,j,k,offset+b) / number_pivot : Real(0.0);
                }
            });
        }
    }
    aux.FillBoundary(geom[lev].periodicity());
    ::erf_sbm::SBMBulkProjection bulk_projection(*sbm_layout);
    for (MFIter mfi(aux); mfi.isValid(); ++mfi) {
        bulk_projection.apply_to_core(mfi.validbox(), aux.const_array(mfi), core.array(mfi));
    }
    core.FillBoundary(geom[lev].periodicity());
    amrex::MultiFab::Copy(*sbm_initial_bulk_state[static_cast<std::size_t>(lev)], core, RhoQ2_comp, 0, 1, 0);
    amrex::MultiFab::Copy(*sbm_initial_bulk_state[static_cast<std::size_t>(lev)], core, RhoQ3_comp, 1, 1, 0);

    Print() << "SBM P1 auxiliary state: components=" << sbm_layout->ncomp()
            << ", bins=" << nbins
            << ", cell-state bytes=" << sbm_auxiliary->state_resident_bytes()
            << ", face-transfer bytes=" << sbm_auxiliary->face_transfer_resident_bytes() +
               sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)]->resident_bytes()
            << ", total auxiliary bytes=" << sbm_auxiliary->resident_bytes() +
               sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)]->resident_bytes() + sbm_cell_bytes(*sbm_initial_bulk_state[static_cast<std::size_t>(lev)])
            << (solverChoice.sbm_manufactured_initialization ? " (manufactured initialization)" : " (empty initialization)")
            << std::endl;
}

void ERF::begin_sbm_step(const int lev, const amrex::MultiFab& core_old, const double old_time)
{
    if (solverChoice.moisture_type == MoistureType::SBM) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sbm_auxiliary != nullptr && sbm_auxiliary->has_level(lev),
                                         "SBM auxiliary state must be initialized before stepping");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<Long>(lev) < sbm_initial_bulk_state.size() &&
                                         sbm_initial_bulk_state[static_cast<std::size_t>(lev)] != nullptr,
                                         "SBM compact baseline must be initialized before stepping");
        if (lev == 0) {
            sbm_composite_initial_totals.assign(
                static_cast<std::size_t>(sbm_layout->ncomp()), Real(0.0));
            for (int level = 0; level <= finest_level; ++level) {
                const bool mask_coarse = level < finest_level && fine_mask[level+1] != nullptr;
                for (int comp = 0; comp < sbm_layout->ncomp(); ++comp) {
                    sbm_composite_initial_totals[static_cast<std::size_t>(comp)] += volWgtSumMF(
                        level, sbm_auxiliary->output(level), comp, *detJ_cc[level],
                        *mapfac[level][MapFacType::m_x], *mapfac[level][MapFacType::m_y],
                        mask_coarse, false);
                }
            }
            sbm_composite_snapshot_valid = true;
        }
        const bool first_fine_substep = lev == 0 ||
            std::abs(old_time - t_old[lev-1]) <=
            128.0 * std::numeric_limits<double>::epsilon() *
            (1.0 + std::max(std::abs(old_time), std::abs(t_old[lev-1])));
        sbm_auxiliary->begin_step(lev, old_time, first_fine_substep);
        // ERF swaps vars_old/vars_new before entering advance_dycore.  The
        // explicit state_old argument is therefore the actual full-step old
        // compact state, even on the second and subsequent time steps.
        amrex::MultiFab::Copy(*sbm_initial_bulk_state[static_cast<std::size_t>(lev)], core_old,
                              RhoQ2_comp, 0, 1, 0);
        amrex::MultiFab::Copy(*sbm_initial_bulk_state[static_cast<std::size_t>(lev)], core_old,
                              RhoQ3_comp, 1, 1, 0);
        auto& evaluation_density = *sbm_evaluation_density[static_cast<std::size_t>(lev)];
        // Preserve the host-prepared coarse/fine stencil, not only the valid
        // cells.  The evaluation spectrum is consumed by later WENO stages;
        // leaving its second ghost layer at the companion's initialization
        // value (zero) creates a false zero-density donor at fine interfaces.
        amrex::MultiFab::Copy(evaluation_density, core_old, Rho_comp, 0, 1,
                              evaluation_density.nGrowVect());
        evaluation_density.FillBoundary(geom[lev].periodicity());
        if (lev == 0) ++sbm_step_count;
    }
}

void ERF::advance_sbm_stage(const int lev,
                            Vector<MultiFab>& state_old,
                            Vector<MultiFab>& state_new,
                            Vector<MultiFab>& state_eval,
                            const double old_step_time,
                            const double old_stage_time,
                            const double new_stage_time,
                            const int stage,
                            const double full_step)
{
    if (solverChoice.moisture_type != MoistureType::SBM) return;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sbm_auxiliary != nullptr && sbm_auxiliary->has_level(lev) && sbm_layout != nullptr,
                                     "SBM auxiliary state is not ready");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        sbm_ownership != nullptr &&
        sbm_ownership->owns(RhoQ2_comp, ::erf_sbm::NativeWritePath::Advection) &&
        sbm_ownership->owns(RhoQ3_comp, ::erf_sbm::NativeWritePath::Advection),
        "SBM compact cloud/rain ownership contract is not active");
    const bool anelastic = solverChoice.anelastic[lev] == 1;
    const auto context = anelastic ?
        ::erf_auxiliary::make_anelastic_stage(stage, old_step_time, old_stage_time,
                                             new_stage_time, full_step,
                                             &state_old[IntVars::cons], &state_eval[IntVars::cons]) :
        ::erf_auxiliary::make_compressible_stage(stage, old_step_time, old_stage_time,
                                                new_stage_time, full_step,
                                                &state_old[IntVars::cons], &state_eval[IntVars::cons]);

    // A fine-level WENO stencil must see the coarse spectrum at the actual
    // stage time.  The manager owns the coarse old/output bracket produced by
    // the completed coarse step and performs the temporal FillPatch before
    // this level constructs any face flux.  Old and Evaluation are distinct
    // source views: prepare each one at the time consumed by this stage,
    // rather than using the manager's last evaluation timestamp for both.
    if (lev > 0) {
        const ::erf_sbm::SBMBulkProjection bulk_projection(*sbm_layout);
        auto project_prepared_view = [&](const amrex::MultiFab& auxiliary,
                                         amrex::MultiFab& core) {
            const IntVect auxiliary_ng = auxiliary.nGrowVect();
            const IntVect core_ng = core.nGrowVect();
            for (MFIter mfi(auxiliary); mfi.isValid(); ++mfi) {
                Box projection_box = mfi.validbox();
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                    projection_box.grow(dir, std::min(auxiliary_ng[dir], core_ng[dir]));
                }
                bulk_projection.apply_to_core(projection_box,
                                               auxiliary.const_array(mfi), core.array(mfi));
            }
            Gpu::streamSynchronize();
        };
        MultiFab coarse_rho_old(vars_old[lev-1][Vars::cons], make_alias, Rho_comp, 2);
        MultiFab coarse_rho_output(vars_new[lev-1][Vars::cons], make_alias, Rho_comp, 2);
        MultiFab fine_rho_old(state_old[IntVars::cons], make_alias, Rho_comp, 2);
        sbm_auxiliary->fill_stage_from_coarse(lev-1, lev,
                                              old_step_time,
                                              geom[lev-1], geom[lev], refRatio(lev-1),
                                              ::erf_auxiliary::AuxiliaryTimeView::Old,
                                              &coarse_rho_old, &coarse_rho_output,
                                              &fine_rho_old);
        // The compact host view is a cache of the provider state.  Publish
        // the projection over the same valid+ghost region immediately after
        // its spectral Old FillPatch transaction, before any host consumer can
        // read a stale compact ghost.
        project_prepared_view(sbm_auxiliary->old(lev), state_old[IntVars::cons]);
        // Compressible stage 1/2 and anelastic stage 1 read the evaluation
        // view.  Stage 0 reads only Old, so there is no synthetic evaluation
        // fill that could obscure a stale-view bug.
        if (stage > 0) {
            MultiFab fine_rho_evaluation(state_eval[IntVars::cons], make_alias, Rho_comp, 2);
            sbm_auxiliary->fill_stage_from_coarse(lev-1, lev,
                                                  old_stage_time,
                                                  geom[lev-1], geom[lev], refRatio(lev-1),
                                                  ::erf_auxiliary::AuxiliaryTimeView::Evaluation,
                                                  &coarse_rho_old, &coarse_rho_output,
                                                  &fine_rho_evaluation);
            project_prepared_view(sbm_auxiliary->evaluation(lev), state_eval[IntVars::cons]);
        }
    }

    // avg_*mom are ERF's actual dry-air carrier mass flux fields.  The
    // transport helper consumes them directly and never rebuilds rho*u from
    // the velocity MultiFabs.
#ifdef ERF_SBM_QUALIFICATION_TEST_HOOKS
    if (lev == 0 && solverChoice.sbm_test_active_limiter) {
        // The ERF time integrator owns these fields and may have regenerated
        // them since level initialization.  Reapply the coordinate-defined
        // qualification pattern at the final hand-off to SBM so the active
        // face is part of the actual production transport call.
        const int face_hi = geom[lev].Domain().bigEnd(0) + 1;
        const ERF_BC xhi_bc = phys_bc_type[Orientation(0, Orientation::high)];
        const bool xhi_outflow = xhi_bc == ERF_BC::outflow ||
            xhi_bc == ERF_BC::ho_outflow || xhi_bc == ERF_BC::open;
        for (MFIter mfi(avg_xmom[lev]); mfi.isValid(); ++mfi) {
            const Box box = mfi.validbox();
            const auto flux = avg_xmom[lev].array(mfi);
            const Real velocity = solverChoice.sbm_manufactured_velocity;
            ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                flux(i,j,k) = (i >= 8 && (i < face_hi || (xhi_outflow && i == face_hi))) ?
                    velocity : Real(0.0);
            });
        }
    }
#endif
    // This is a qualification-only carrier override for the independent
    // host-CFL counterexample and legacy stage-demand fixture.  The positive
    // variable-density RK3 qualification deliberately leaves this block
    // disabled so avg_*mom remain the fields constructed by ERF's dynamics.
#ifdef ERF_SBM_QUALIFICATION_TEST_HOOKS
    if (solverChoice.sbm_test_manufactured_carrier_override) {
        // Keep the variable-density qualification carrier synchronized with
        // the manufactured host state at the final production hand-off.  A
        // uniform mass flux is divergence-free on the periodic fixture, so
        // the host target density remains the same variable field while the
        // provider exercises its density-weighted transfer.
        // This qualification carrier is intentionally independent of host
        // velocity: keeping the manufactured variable-density field
        // stationary isolates the stage-density pairing while the actual
        // avg_*mom handoff remains nonzero and changes by stage.
        const Real base_carrier = solverChoice.sbm_manufactured_variable_density &&
            solverChoice.sbm_manufactured_velocity == Real(0.0) ? Real(0.125) :
            solverChoice.sbm_manufactured_velocity;
        const Real stage_carrier = solverChoice.sbm_manufactured_variable_density ?
            base_carrier * (Real(1.0) + Real(0.125) * Real(stage)) : base_carrier;
        avg_xmom[lev].setVal(stage_carrier);
        avg_ymom[lev].setVal(Real(0.0));
        avg_zmom[lev].setVal(Real(0.0));
    }
#endif
    Real stage_minimum_limiter = Real(1.0);
    const auto boundary_policy =
        ::erf_sbm::make_erf_transport_boundary_policy(geom[lev], phys_bc_type);
    // Grouped WENO and the coarse/fine donor preparation require the same
    // two-cell stencil depth as the provider state.  Capping these aliases at
    // one ghost cell leaves the second fine-level interface layer at the
    // prepared-view sentinel (zero), which turns the density-normalized donor
    // ratio into NaN at a coarse/fine boundary.
    MultiFab rho_anchor(state_old[IntVars::cons], make_alias, Rho_comp, 2);
    // The auxiliary predictor is the output accepted by the preceding SBM
    // stage.  Its matching dry density is captured from that same host stage
    // (rho_target below), whereas state_eval is S_sum after ERF's fast update.
    // Reusing state_eval here silently pairs two different temporal views.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<Long>(lev) < sbm_evaluation_density.size() &&
        sbm_evaluation_density[static_cast<std::size_t>(lev)] != nullptr,
        "SBM predictor density snapshot is not initialized");
    const MultiFab& predictor_density =
        *sbm_evaluation_density[static_cast<std::size_t>(lev)];
    MultiFab rho_target(state_new[IntVars::cons], make_alias, Rho_comp, 2);
    MultiFab state_eval_density(state_eval[IntVars::cons], make_alias, Rho_comp, 2);

    const MultiFab* density_for_transport = &predictor_density;
#ifdef ERF_SBM_QUALIFICATION_TEST_HOOKS
    // Negative qualification mutant: pair the accepted spectrum with ERF's
    // fast state-evaluation density instead of the predictor density from
    // the same temporal view.  The variable-density ratio contract below
    // must reject this semantic mismatch before any unrelated CFL path can
    // hide it.
    if (solverChoice.sbm_test_use_state_eval_density) {
        density_for_transport = &state_eval_density;
    }
#endif

    // Detect forbidden native qc/qr mutations before the provider projection
    // can overwrite their evidence.  At this point manager.output is the
    // accepted provider state that owns the compact fields currently present
    // in S_new.  The check is deliberately independent of the projection
    // performed by advance_stage itself.
    if (sbm_ownership->owns(RhoQ2_comp, ::erf_sbm::NativeWritePath::Advection)) {
#ifdef ERF_SBM_QUALIFICATION_TEST_HOOKS
        if (solverChoice.sbm_test_fault_injection == "qc_advection") {
            state_new[IntVars::cons].plus(Real(1.0e-7), RhoQ2_comp, 1, 0);
        } else if (solverChoice.sbm_test_fault_injection == "qr_diffusion") {
            state_new[IntVars::cons].plus(Real(1.0e-7), RhoQ3_comp, 1, 0);
        } else if (solverChoice.sbm_test_fault_injection == "bulk_clip") {
            state_new[IntVars::cons].setVal(Real(0.0), RhoQ2_comp, 1, 0);
        }
#endif
        const Real pre_projection_error = sbm_max_projection_error(
            *sbm_layout, sbm_auxiliary->output(lev), state_new[IntVars::cons]);
        const Real pre_projection_scale = amrex::max(
            sbm_auxiliary->output(lev).norm0(0),
            amrex::max(state_new[IntVars::cons].norm0(RhoQ2_comp),
                       state_new[IntVars::cons].norm0(RhoQ3_comp)));
        const Real pre_projection_tolerance = Real(4096.0) *
            std::numeric_limits<Real>::epsilon() * pre_projection_scale;
        if (pre_projection_error > pre_projection_tolerance) {
            std::ostringstream message;
            message << "SBM native qc/qr ownership invariant failed before projection:"
                    << " level=" << lev << " stage=" << stage
                    << " error=" << pre_projection_error
                    << " tolerance=" << pre_projection_tolerance;
            amrex::Error(message.str());
        }
    }
    ::erf_sbm::ActualStageDemand actual_stage_demand;
    ::erf_sbm::advance_stage(*sbm_auxiliary, *sbm_layout, context,
                            rho_anchor, *density_for_transport, rho_target, state_new[IntVars::cons],
                            avg_xmom[lev], avg_ymom[lev], avg_zmom[lev], geom[lev],
                            sbm_auxiliary->face_transfer_ledger(lev).stage(),
                            solverChoice.sbm_transport_method == "GroupedFCT_WENOZ3" ?
                                ::erf_sbm::TransportMethod::GroupedFCT_WENOZ3 :
                                ::erf_sbm::TransportMethod::DonorCell, lev,
                            solverChoice.sbm_diffusion_coeff, solverChoice.sbm_chunk_size,
                            solverChoice.sbm_composite_diagnostic_file.empty() ? nullptr : &stage_minimum_limiter,
                            boundary_policy, &actual_stage_demand);
    if (solverChoice.sbm_manufactured_variable_density) {
        const Real stage_ratio_error = sbm_max_manufactured_ratio_error(
            *sbm_layout, sbm_auxiliary->output(lev), state_new[IntVars::cons]);
        const Real stage_ratio_tolerance = Real(4096.0) *
            std::numeric_limits<Real>::epsilon() * Real(1.0e-6);
        if (ParallelDescriptor::IOProcessor()) {
            Print() << std::setprecision(17)
                    << "SBM variable-density stage ratio level=" << lev
                    << " stage=" << stage
                    << " error=" << stage_ratio_error
                    << " tolerance=" << stage_ratio_tolerance << '\n';
        }

        const Real predictor_target_difference = sbm_max_change(
            predictor_density, rho_target, 1);
        const Real predictor_eval_difference = sbm_max_change(
            predictor_density, state_eval_density, 1);
        const Real target_eval_difference = sbm_max_change(
            rho_target, state_eval_density, 1);
        const Real predictor_density_min = predictor_density.min(0);
        const Real predictor_density_max = predictor_density.max(0);
        const Real target_density_min = rho_target.min(0);
        const Real target_density_max = rho_target.max(0);
        const Real eval_density_min = state_eval_density.min(0);
        const Real eval_density_max = state_eval_density.max(0);
        const Real actual_carrier_max = amrex::max(
            avg_xmom[lev].norm0(0),
            amrex::max(avg_ymom[lev].norm0(0), avg_zmom[lev].norm0(0)));
        if (ParallelDescriptor::IOProcessor()) {
            Print() << std::setprecision(17)
                    << "SBM F01 variable-density views level=" << lev
                    << " stage=" << stage
                    << " predictor_target_max=" << predictor_target_difference
                    << " predictor_eval_max=" << predictor_eval_difference
                    << " target_eval_max=" << target_eval_difference
                    << " predictor_density_min=" << predictor_density_min
                    << " predictor_density_max=" << predictor_density_max
                    << " target_density_min=" << target_density_min
                    << " target_density_max=" << target_density_max
                    << " eval_density_min=" << eval_density_min
                    << " eval_density_max=" << eval_density_max
                    << " ratio_residual=" << stage_ratio_error
                    << " actual_carrier_max=" << actual_carrier_max
                    << " actual_carrier_demand=" << actual_stage_demand.maximum_rate
                    << "\n";
        }
        if (stage_ratio_error > stage_ratio_tolerance) {
            std::ostringstream message;
            message << "SBM variable-density stage ratio contract failed: level=" << lev
                    << " stage=" << stage << " error=" << stage_ratio_error
                    << " tolerance=" << stage_ratio_tolerance;
            amrex::Error(message.str());
        }
    }
    // The output spectrum just accepted by the manager is now paired with the
    // host dry density at this stage target.  Preserve only that scalar view
    // for the next callback; do not retain a full conserved-state duplicate.
    // This copy intentionally follows the diagnostics above: otherwise the
    // input predictor would be overwritten before the qualification compared
    // it with the current ERF target/evaluation views, making the positive
    // density-pairing check vacuous and masking the negative mutant.
    auto& next_predictor_density =
        *sbm_evaluation_density[static_cast<std::size_t>(lev)];
    amrex::MultiFab::Copy(next_predictor_density, state_new[IntVars::cons],
                          Rho_comp, 0, 1, next_predictor_density.nGrowVect());
    next_predictor_density.FillBoundary(geom[lev].periodicity());
    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
        const char* temporal_mode = anelastic ? "anelastic_heun" : "compressible_rk3";
        Print() << std::setprecision(17)
                << "SBM actual stage low-order demand level=" << actual_stage_demand.level
                << " stage=" << actual_stage_demand.stage_index
                << " temporal_mode=" << temporal_mode
                << " acoustic_substepping=disabled"
                << " tau=" << actual_stage_demand.stage_duration
                << " actual_rate=" << actual_stage_demand.maximum_rate
                << " actual_advective_rate=" << actual_stage_demand.advective_rate
                << " actual_diffusive_rate=" << actual_stage_demand.diffusive_rate
                << " tau_actual_rate=" << actual_stage_demand.maximum_tau_rate
                << " worst_cell=(" << actual_stage_demand.worst_i << ","
                << actual_stage_demand.worst_j << "," << actual_stage_demand.worst_k << ")\n";
    }
    if (!solverChoice.sbm_composite_diagnostic_file.empty()) {
        sbm_minimum_accepted_limiter = amrex::min(sbm_minimum_accepted_limiter,
                                                  stage_minimum_limiter);
    }

    const auto& ledger = sbm_auxiliary->face_transfer_ledger(lev);
    // YAFluxRegister consumes instantaneous per-area fluxes and applies the
    // supplied dt/dx factor.  Register the accepted spectral face flux at
    // exactly the same stage weights as the auxiliary ledger; this is the
    // physical I=A*integral(F dt) contract without a second area or time
    // multiplication.
    if (solverChoice.coupling_type == CouplingType::TwoWay &&
        context.accepted_ledger_weight() != Real(0.0)) {
        auto& stage_flux = ledger.stage();
        const auto dx = geom[lev].CellSizeArray();
        const Real register_dt = static_cast<Real>(full_step * context.accepted_ledger_weight());
        for (MFIter mfi(state_new[IntVars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const std::array<FArrayBox const*, AMREX_SPACEDIM> fluxes{
                AMREX_D_DECL(&stage_flux.x()[mfi], &stage_flux.y()[mfi], &stage_flux.z()[mfi])};
            if (lev < finest_level && sbm_flux_reg[lev+1] != nullptr) {
                sbm_flux_reg[lev+1]->CrseAdd(mfi, fluxes, dx.data(), register_dt, RunOn::Device);
            }
            if (lev > 0 && sbm_flux_reg[lev] != nullptr) {
                sbm_flux_reg[lev]->FineAdd(mfi, fluxes, dx.data(), register_dt, RunOn::Device);
            }
        }
        Gpu::streamSynchronize();
    }
    const ::erf_sbm::SBMBulkProjection bulk_projection(*sbm_layout);
    bulk_projection.apply_to_face_transfer(ledger.accepted(), *sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)]);

    if (context.completes_level_step && solverChoice.sbm_manufactured_initialization &&
        !solverChoice.sbm_diagnostic_file.empty()) {
        const auto& old = sbm_auxiliary->old(lev);
        const auto& output = sbm_auxiliary->output(lev);
        const int ncomp = sbm_layout->ncomp();
        const Real cell_volume = geom[lev].CellSize(0) * geom[lev].CellSize(1) * geom[lev].CellSize(2);
        Real initial_mass = sbm_total_mass(old, ncomp, cell_volume);
        Real final_mass = sbm_total_mass(output, ncomp, cell_volume);
        const Real initial_variation = old.max(0) - old.min(0);
        const Real transport_change = sbm_max_change(output, old, ncomp);
        const Real projection_error = sbm_max_projection_error(*sbm_layout, output,
                                                               state_new[IntVars::cons]);
        const Real face_projection_error = bulk_projection.max_face_projection_error(
            ledger.accepted(), *sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)]);
        const auto closure = ::erf_sbm::evaluate_accepted_transfer_closure(
            *sbm_layout, old, output, ledger.accepted(),
            *sbm_initial_bulk_state[static_cast<std::size_t>(lev)], 0, 1,
            state_new[IntVars::cons], RhoQ2_comp, RhoQ3_comp,
            *sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)], geom[lev]);
        const Real compact_mass = cell_volume *
            (state_new[IntVars::cons].sum(RhoQ2_comp) +
             state_new[IntVars::cons].sum(RhoQ3_comp));
        write_sbm_diagnostic(solverChoice.sbm_diagnostic_file,
                             solverChoice.anelastic[lev] == 1, ncomp,
                             sbm_step_count,
                             initial_mass, final_mass, initial_variation,
                             transport_change, projection_error,
                             face_projection_error, closure, compact_mass, cell_volume,
                             sbm_auxiliary->state_resident_bytes() + sbm_cell_bytes(*sbm_initial_bulk_state[static_cast<std::size_t>(lev)]),
                             sbm_auxiliary->face_transfer_resident_bytes() +
                                 sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)]->resident_bytes(),
                             sbm_auxiliary->resident_bytes() +
                                 sbm_accepted_bulk_face_transfer[static_cast<std::size_t>(lev)]->resident_bytes() +
                                 sbm_cell_bytes(*sbm_initial_bulk_state[static_cast<std::size_t>(lev)]));
    }
}
